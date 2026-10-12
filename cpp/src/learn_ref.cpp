// CPU reference of the batched body learner (batch_learner.h). Every formula and every accumulation order is
// organism.cpp's (learn_backprop's adam path, temporal_credit, temporal_backprop, adam_step), so at one stream and
// with -ffp-contract=off it reproduces Organism bit for bit; at S streams it is the definition the GPU learner
// is checked against.
#include "laminar/batch_learner.h"
#include "laminar/int4.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace laminar {
namespace {

struct Slot { float tg[Dims::F]; float reg_prev[4], G[4], hg[4], hv[4], hu[4], ha[Dims::F]; int8_t x1[Dims::IN], x1e[Dims::IN]; };

class RefLearner final : public Learner {
public:
    RefLearner(const LearnConfig& c, uint8_t* w) : c_(c), W_(w) {
        const size_t n = (size_t)c.units * Dims::UNIT_NIBBLES;
        sh_.assign(n, 0.f); m_.assign(n, 0.f); v_.assign(n, 0.f); g_.assign(n, 0.f);
        ring_.assign((size_t)c.streams * c.tc_window, Slot{});
        gws_.assign((size_t)c.streams * Dims::F, 0.f);
        hd_.assign(8 * Dims::F + 4, 0.f);
    }
    std::string name() const override { return "cpu-reference"; }
    void upload_all() override {}
    void upload_units(const uint32_t*, uint32_t) override {}
    void set_head(const float* hw_g, const float* hw_v, const float* hb) override {
        std::memcpy(hw_g_, hw_g, sizeof hw_g_); std::memcpy(hw_v_, hw_v, sizeof hw_v_); std::memcpy(hb_, hb, sizeof hb_);
    }
    void take_head_grads(float* hd) override { for (size_t i = 0; i < hd_.size(); ++i) { hd[i] += hd_[i]; hd_[i] = 0.f; } }
    void read_unit_state(uint32_t u, float* sh, float* m, float* v, float* g) override {
        const size_t b = (size_t)u * Dims::UNIT_NIBBLES, n = Dims::UNIT_NIBBLES * sizeof(float);
        if (sh) std::memcpy(sh, &sh_[b], n); if (m) std::memcpy(m, &m_[b], n); if (v) std::memcpy(v, &v_[b], n); if (g) std::memcpy(g, &g_[b], n);
    }
    void read_ring_tg(uint32_t s, uint32_t slot, float* tg) override { std::memcpy(tg, ring_[(size_t)s * c_.tc_window + slot].tg, sizeof(float) * Dims::F); }

    void step(const StepIn& in) override {
        for (uint32_t i = 0; i < in.nreset; ++i) { const size_t b = (size_t)in.reset[i] * Dims::UNIT_NIBBLES;
            std::fill(m_.begin() + b, m_.begin() + b + Dims::UNIT_NIBBLES, 0.f); std::fill(v_.begin() + b, v_.begin() + b + Dims::UNIT_NIBBLES, 0.f);
            std::fill(g_.begin() + b, g_.begin() + b + Dims::UNIT_NIBBLES, 0.f); }
        std::fill(gws_.begin(), gws_.end(), 0.f);
        // 1. direct credit. Receivers come before their senders (jobs are visited last to first), so a sender has every
        // receiver's message gradient (msg_) when its turn comes.
        msg_.assign((size_t)in.ndirect * Dims::B, 0.f); strm_.assign((size_t)c_.streams * Dims::B, 0.f);   // strm_: motif_stream, per stream
        for (uint32_t k = 0; k < in.ndirect; ++k) {
            const LTrace& tr = in.direct[k]; const LStream& ls = in.streams[tr.stream];
            float gout[Dims::F];
            const float ws1 = (tr.flags & LF_NOWS) ? 0.f : 1.f;   // ws_emit_max: not a workspace writer
            if (c_.split) for (int j = 0; j < Dims::F; ++j) gout[j] = (ws1 * ls.dl6[j] + ls.dl6m[tr.slot][j]) * tr.gain;   // ws_split: + its motif's own error
            else for (int j = 0; j < Dims::F; ++j) gout[j] = ls.dl6[j] * tr.gain;
            float* R = &strm_[(size_t)tr.stream * Dims::B];   // motif_stream: what the stream's later bodies need from this L6 (jobs are last to first)
            if (tr.flags & LF_L6) for (int j = 0; j < Dims::B; ++j) gout[j] += R[j];
            const bool st = (tr.flags & LF_STRM) != 0;
            backward(tr, gout, tr.exact ? ls.x1e : ls.x1, tr.exact ? nullptr : &gws_[(size_t)tr.stream * Dims::F], false,
                     &msg_[(size_t)k * Dims::B], tr.sender && !st ? &msg_[(size_t)(k + tr.sender) * Dims::B] : nullptr, st ? R : nullptr);
        }
        // 2. Adam steps
        for (uint32_t k = 0; k < in.ndue; ++k) { if (c_.muon) muon_step(in.due[k].unit, in.due[k].n_uses); else adam_step(in.due[k].unit, in.due[k].t, in.due[k].n_uses); }
        // 3. temporal credit, per stream (first the stream's ring injections, in order: Organism's tiles_learn precedes temporal_credit)
        uint32_t ij = 0;
        for (uint32_t s = 0; s < c_.streams; ++s) {
            const LStream& ls = in.streams[s];
            for (; ij < in.ninj && in.inj[ij].stream == s; ++ij) { Slot& e = ring_[(size_t)s * c_.tc_window + (ls.head + c_.tc_window - in.inj[ij].age) % c_.tc_window];
                for (int j = 0; j < Dims::F; ++j) e.tg[j] += in.inj[ij].g[j]; }
            if (!ls.active) continue;
            Slot* R = &ring_[(size_t)s * c_.tc_window];
            if (ls.h_valid) { Slot& e = R[ls.h_slot];
                for (int r = 0; r < 4; ++r) { e.reg_prev[r] = ls.h_reg_prev[r]; e.G[r] = ls.h_G[r]; e.hg[r] = ls.h_g[r]; e.hv[r] = ls.h_v[r]; e.hu[r] = ls.h_u[r]; }
                std::memcpy(e.ha, ls.h_a, sizeof e.ha); }
            float* gws = &gws_[(size_t)s * Dims::F];
            float greg[4];
            for (int r = 0; r < 4; ++r) { greg[r] = gws[28 + r] * c_.reg_scale / 100.f + ls.greg_ro[r]; gws[28 + r] = 0.f; }
            for (int i = 0; i < Dims::F; ++i) gws[i] *= ls.gscale;
            const uint32_t Wn = c_.tc_window; const float d = c_.decay;
            float sd[Dims::F]; for (int i = 0; i < Dims::F; ++i) sd[i] = 1.f;
            float sr[4] = {1.f, 1.f, 1.f, 1.f};
            for (uint32_t k = 1; k <= ls.fill; ++k) { Slot& e = R[(ls.head + Wn - k) % Wn];
                for (int i = 0; i < Dims::F; ++i) { e.tg[i] += sd[i] * gws[i]; sd[i] *= d; }
                for (int r = 0; r < 4; ++r) { const float dlt = sr[r] * greg[r]; if (dlt == 0.f) { sr[r] *= 1.f - e.G[r]; continue; }
                    const float dg = dlt * (e.hv[r] - e.reg_prev[r]) * e.hg[r] * (1.f - e.hg[r]), dv = dlt * e.hg[r] * 100.f * (1.f - std::tanh(e.hu[r]) * std::tanh(e.hu[r]));
                    for (int j = 0; j < Dims::F; ++j) { e.tg[j] += (hw_g_[r][j] * dg + hw_v_[r][j] * dv) / 128.f;
                        hd_[(size_t)r * Dims::F + j] += dg * e.ha[j]; hd_[(size_t)(4 + r) * Dims::F + j] += dv * e.ha[j]; }
                    hd_[8 * Dims::F + r] += dg; sr[r] *= 1.f - e.G[r]; } }
        }
        // 4. the oldest events' writers
        int32_t checked_stream = -1; bool any = false;
        for (uint32_t k = 0; k < in.ntemporal; ++k) {
            const LTrace& tr = in.temporal[k]; const Slot& e = ring_[(size_t)tr.stream * c_.tc_window + tr.slot];
            if ((int32_t)tr.stream != checked_stream) { checked_stream = tr.stream; any = false; for (int i = 0; i < Dims::F; ++i) any |= e.tg[i] != 0.f; }
            if (!any) continue;
            float go[Dims::F]; for (int j = 0; j < Dims::F; ++j) go[j] = e.tg[j] * tr.gain;
            backward(tr, go, tr.exact ? e.x1e : e.x1, nullptr, true);
        }
        // 5. store the current events
        for (uint32_t s = 0; s < c_.streams; ++s) {
            const LStream& ls = in.streams[s]; if (!ls.active) continue;
            Slot& e = ring_[(size_t)s * c_.tc_window + ls.head];
            std::memcpy(e.x1, ls.x1, Dims::IN); std::memcpy(e.x1e, ls.x1e, Dims::IN); std::fill(e.tg, e.tg + Dims::F, 0.f);
            for (int r = 0; r < 4; ++r) { e.reg_prev[r] = ls.reg_now[r]; e.G[r] = 0.f; }
        }
    }

private:
    LearnConfig c_; uint8_t* W_;
    std::vector<float> sh_, m_, v_, g_, gws_, hd_, msg_, strm_;
    std::vector<Slot> ring_;
    float hw_g_[4][Dims::F] = {}, hw_v_[4][Dims::F] = {}, hb_[4] = {};

    float live(int8_t y) const { return (y >= 127 || y <= -127) ? c_.bp_leak : 1.f; }
    void grad(uint32_t u, uint32_t nib0, const int8_t* x, int in, const float* g, int out, float scale) {   // Organism::adam_grad
        float* G = &g_[(size_t)u * Dims::UNIT_NIBBLES + nib0];
        for (int j = 0; j < out; ++j) { if (g[j] == 0.f) continue; const float gj = g[j] * scale; float* r = G + (size_t)j * in; for (int i = 0; i < in; ++i) r[i] += gj * (float)x[i]; }
    }
    // learn_backprop's per-trace body (bp_route messages, no return, pc_settle 0) and temporal_backprop's (no cuts); the
    // two differ only in the gws output, the gradient scale (which the trace carries) and the messages (direct only):
    // msg_in is what this body's receivers need from its L2, msg_out its sender's (null: none).
    // strm_out (motif_stream): this body's L3 read the mean of the earlier bodies' L6; its share goes to all of them.
    void backward(const LTrace& tr, const float* gout, const int8_t* x1, float* gws, bool temporal, const float* msg_in = nullptr, float* msg_out = nullptr, float* strm_out = nullptr) {
        const uint32_t u = tr.unit; const uint8_t* W = W_ + (size_t)u * Dims::UNIT_BYTES;
        float g6[Dims::F], g5[Dims::E] = {0}, g2[Dims::B] = {0}, g3[Dims::C] = {0}, g1[Dims::A] = {0};
        if (temporal) for (int j = 0; j < Dims::F; ++j) g6[j] = gout[j] * live(tr.l6[j]) / (float)(1 << tr.sh[LI_6]);
        else for (int j = 0; j < Dims::F; ++j) g6[j] = gout[j] != 0.f ? gout[j] * live(tr.l6[j]) / (float)(1 << tr.sh[LI_6]) * 1.f : 0.f;
        for (int j = 0; j < Dims::F; ++j) if (g6[j] != 0.f) for (int i = 0; i < Dims::E; ++i) g5[i] += (float)nib_get(W, Dims::NIB_L6 + j * Dims::IN6 + i) * g6[j];
        for (int i = 0; i < Dims::E; ++i) g5[i] *= live(tr.l5[i]) / (float)(1 << tr.sh[LI_5]) * 1.f;
        for (int j = 0; j < Dims::E; ++j) if (g5[j] != 0.f) for (int i = 0; i < Dims::IN5; ++i) {
            const float gx = (float)nib_get(W, Dims::NIB_L5 + j * Dims::IN5 + i) * g5[j];
            if (i < Dims::B) g2[i] += gx; else g3[i - Dims::B] += gx; }
        if (msg_in) for (int i = 0; i < Dims::B; ++i) g2[i] += msg_in[i];
        if (temporal) {   // temporal_backprop's order: L3 then L2
            for (int i = 0; i < Dims::C; ++i) g3[i] *= live(tr.l3[i]) / (float)(1 << tr.sh[LI_3]) * 1.f;
            for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + i) * g3[j];
            for (int i = 0; i < Dims::B; ++i) g2[i] *= live(tr.l2[i]) / (float)(1 << tr.sh[LI_2]) * 1.f;
        } else {          // learn_backprop's order (identical arithmetic)
            for (int i = 0; i < Dims::C; ++i) g3[i] *= live(tr.l3[i]) / (float)(1 << tr.sh[LI_3]) * 1.f;
            for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + i) * g3[j];
            if (msg_out) for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) for (int i = 0; i < Dims::B; ++i) msg_out[i] += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + Dims::A + i) * g3[j];
            if (strm_out) { const float sc = 0.5f / (float)tr.sender;
                for (int i = 0; i < Dims::B; ++i) { if (tr.x3[Dims::A + i] >= 127 || tr.x3[Dims::A + i] <= -127) continue; float a = 0.f;
                    for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) a += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + Dims::A + i) * g3[j];
                    strm_out[i] += a * sc; } }
            for (int i = 0; i < Dims::B; ++i) g2[i] *= live(tr.l2[i]) / (float)(1 << tr.sh[LI_2]) * 1.f;
        }
        for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L2 + j * Dims::IN2 + i) * g2[j];
        for (int i = 0; i < Dims::A; ++i) g1[i] *= live(tr.l1[i]) / (float)(1 << tr.sh[LI_1]) * 1.f;
        if (gws) for (int j = 0; j < Dims::A; ++j) if (g1[j] != 0.f) for (int i = 0; i < Dims::F; ++i) gws[i] += (float)nib_get(W, Dims::NIB_L1 + j * Dims::IN + 96 + i) * g1[j];
        grad(u, Dims::NIB_L6, tr.x6, Dims::IN6, g6, Dims::F, tr.scale);
        grad(u, Dims::NIB_L5, tr.x5, Dims::IN5, g5, Dims::E, tr.scale);
        grad(u, Dims::NIB_L2, tr.x2, Dims::IN2, g2, Dims::B, tr.scale);
        grad(u, Dims::NIB_L3, tr.x3, Dims::IN3, g3, Dims::C, tr.scale);
        grad(u, Dims::NIB_L1, x1, Dims::IN, g1, Dims::A, tr.scale);
    }
    void muon_step(uint32_t u, float n_uses) {   // Organism::muon_step (the same operations in the same order)
        const float inv_k = 1.f / std::max(1.f, n_uses), lr = c_.muon_lr, mu = c_.muon_mom;
        uint8_t* w = W_ + (size_t)u * Dims::UNIT_BYTES; const size_t b = (size_t)u * Dims::UNIT_NIBBLES;
        float* sh = &sh_[b]; float* m = &m_[b]; float* G = &g_[b];
        for (int n = 0; n < Dims::NIB_L4 + Dims::P4; ++n) { const int cw = nib_get(w, n); if ((int)std::lround(sh[n]) != cw) sh[n] = (float)cw; }
        static const int lay[6][3] = {{Dims::NIB_L1, Dims::IN, Dims::A}, {Dims::NIB_L2, Dims::IN2, Dims::B}, {Dims::NIB_L3, Dims::IN3, Dims::C},
                                      {Dims::NIB_L5, Dims::IN5, Dims::E}, {Dims::NIB_L6, Dims::IN6, Dims::F}, {Dims::NIB_L4, Dims::IN4, Dims::R}};
        std::vector<float>& U = mu_U_; std::vector<float>& X = mu_X_; std::vector<float>& A = mu_A_; std::vector<float>& B = mu_B_; std::vector<float>& T = mu_T_;
        for (const auto& ly : lay) {
            const int rows = ly[2], cols = ly[1]; const uint32_t off = (uint32_t)ly[0];
            U.assign((size_t)rows * cols, 0.f); bool any = false;
            for (int i = 0; i < rows * cols; ++i) { const float g = G[off + i] * inv_k; G[off + i] = 0.f; m[off + i] = mu * m[off + i] + g; U[i] = g + mu * m[off + i]; any |= U[i] != 0.f; }
            if (!any) continue;
            const bool tr = rows > cols; const int r = tr ? cols : rows, c = tr ? rows : cols;
            X.assign((size_t)r * c, 0.f); double n2 = 0.0;
            for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) { const float x = U[(size_t)i * cols + j]; if (tr) X[(size_t)j * c + i] = x; else X[(size_t)i * c + j] = x; n2 += (double)x * x; }
            const float in = 1.f / ((float)std::sqrt(n2) + 1e-7f); for (auto& x : X) x *= in;
            A.assign((size_t)r * r, 0.f); B.assign((size_t)r * r, 0.f); T.assign((size_t)r * c, 0.f);
            for (int it = 0; it < 5; ++it) {
                for (int i = 0; i < r; ++i) for (int k = 0; k < r; ++k) { float s2 = 0.f; for (int j = 0; j < c; ++j) s2 += X[(size_t)i * c + j] * X[(size_t)k * c + j]; A[(size_t)i * r + k] = s2; }
                for (int i = 0; i < r; ++i) for (int k = 0; k < r; ++k) { float s2 = 0.f; for (int j = 0; j < r; ++j) s2 += A[(size_t)i * r + j] * A[(size_t)j * r + k]; B[(size_t)i * r + k] = -4.7750f * A[(size_t)i * r + k] + 2.0315f * s2; }
                for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) { float s2 = 0.f; for (int k = 0; k < r; ++k) s2 += B[(size_t)i * r + k] * X[(size_t)k * c + j]; T[(size_t)i * c + j] = 3.4445f * X[(size_t)i * c + j] + s2; }
                X.swap(T);
            }
            const float scale = std::sqrt(std::max(1.f, (float)rows / (float)cols));
            for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) { const float o = tr ? X[(size_t)j * c + i] : X[(size_t)i * c + j]; const uint32_t n = off + (uint32_t)(i * cols + j);
                sh[n] = std::max(-8.49f, std::min(7.49f, sh[n] - lr * (scale * o + c_.adam_wd * sh[n])));
                const int cw = nib_get(w, n), nw = (int)std::lround(sh[n]); if (nw != cw) nib_set(w, n, nw); }
        }
    }
    std::vector<float> mu_U_, mu_X_, mu_A_, mu_B_, mu_T_;
    void adam_step(uint32_t u, uint32_t t, float n_uses) {   // Organism::adam_step
        const float c1 = 1.f - std::pow(0.9f, (float)t), c2 = 1.f - std::pow(0.999f, (float)t), lr = c_.adam_lr, inv_k = 1.f / std::max(1.f, n_uses);
        uint8_t* w = W_ + (size_t)u * Dims::UNIT_BYTES; const size_t b = (size_t)u * Dims::UNIT_NIBBLES;
        float* sh = &sh_[b]; float* m = &m_[b]; float* v = &v_[b]; float* G = &g_[b];
        for (int n = 0; n < Dims::NIB_L4 + Dims::P4; ++n) {
            const int cw = nib_get(w, n); if ((int)std::lround(sh[n]) != cw) sh[n] = (float)cw;
            const float gr = G[n] * inv_k; G[n] = 0.f;
            m[n] = 0.9f * m[n] + 0.1f * gr; v[n] = 0.999f * v[n] + 0.001f * gr * gr;
            if (v[n] == 0.f) continue;
            sh[n] = std::max(-8.49f, std::min(7.49f, sh[n] - lr * ((m[n] / c1) / (std::sqrt(v[n] / c2) + 1e-8f) + c_.adam_wd * sh[n])));
            const int nw = (int)std::lround(sh[n]); if (nw != cw) nib_set(w, n, nw);
        }
    }
};

} // namespace

std::unique_ptr<Learner> make_ref_learner(const LearnConfig& c, uint8_t* host_weights) { return std::make_unique<RefLearner>(c, host_weights); }

} // namespace laminar
