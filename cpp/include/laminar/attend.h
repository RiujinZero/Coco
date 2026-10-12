#pragma once
// Learned answer attention (seq_attn > 0): a small question-time block that reads the whole dialogue.
//
// Items are the dialogue's sentences, kept for the whole dialogue (not a 64-sentence window). An item's key is
// the mean of its words' learned embeddings, plus a learned projection of the organism's own workspace at the
// sentence's end, plus a learned speaker vector. The question is the last user turn, built the same way.
//
//   relevance   r_i  = q' A k_i  + wo o_i                (o_i: idf-weighted word overlap with the question)
//   pick        a_i  = r_i + lambda(q) rec_i,  alpha = softmax(a)   (lambda: how much this question wants the latest)
//   read        pooled = sum alpha_i k_i,  pooled_q = sum alpha_i (k_i * q)
//   aggregate   S = sum sigmoid(r2_i) * (tanh(V . k_i + Vq . (k_i * q) + bv) + wn num_i)   (a gated signed sum)
//   answer      a mixture: (1 - pi) x softmax over the answers observed so far, from [pooled, pooled_q, q, bumps(S), 1]
//               + pi x a pointer that COPIES a word from the attended sentences (alpha_i x a learned word
//               selector beta_iw = softmax_w((B'q) . E[w]))
//               + a PROPOSAL from the organism's other mechanisms (the within-episode chain's pick): the block
//               composes and vetoes; the three sources' weights are a softmax over learned logits from the
//               question and each source's confidence, so whom to trust for which question is learned
//   trust       sigmoid(Wg . [q, pmax, 1]): the probability the block is right; it speaks only above one half
//
// The QUESTION is read from the organism's own state: each of its words with the workspace at that word, pooled
// by learned attention. TRUST (the gate and the source mixture) learns only on held-out dialogues (one in five,
// chosen by the story's first sentence), where the answering parts never trained; elsewhere only they learn.
//
// Nothing names a task: the answer vocabulary is whatever replies were observed, numbers are digits in the text,
// and which reduction answers which question is in the learned weights. The block runs once per reply turn.
// Learning (from the observed reply's first token; never from the organism's own output):
//   backprop during a warm start (the developmental phase), a replay of the warm-start samples at the switch,
//   then activity-perturbed forward gradients: a random tangent on every site activation, one forward-mode pass
//   gives the directional derivative of the loss, and each site's weights move by that derivative x tangent x input.
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <array>
#include <cctype>
#include <cstdlib>

namespace laminar {

struct Dual { float v = 0.f, d = 0.f; Dual() {} Dual(float a, float b = 0.f) : v(a), d(b) {} };
inline Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
inline Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
inline Dual operator*(Dual a, Dual b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
inline Dual operator*(float s, Dual a) { return {s * a.v, s * a.d}; }
inline Dual& operator+=(Dual& a, Dual b) { a.v += b.v; a.d += b.d; return a; }
inline float val(float x) { return x; }
inline float val(const Dual& x) { return x.v; }
inline float fexp(float x) { return std::exp(x); }
inline Dual fexp(Dual x) { float e = std::exp(x.v); return {e, e * x.d}; }
inline float ftanh(float x) { return std::tanh(x); }
inline Dual ftanh(Dual x) { float t = std::tanh(x.v); return {t, (1.f - t * t) * x.d}; }
inline float fsig(float x) { return 1.f / (1.f + std::exp(-x)); }
inline Dual fsig(Dual x) { float s = 1.f / (1.f + std::exp(-x.v)); return {s, s * (1.f - s) * x.d}; }
inline float flog(float x) { return std::log(x); }
inline Dual flog(Dual x) { return {std::log(x.v), x.d / x.v}; }
inline float fsoftplus(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }
inline Dual fsoftplus(Dual x) { float s = 1.f / (1.f + std::exp(-x.v)); return {fsoftplus(x.v), s * x.d}; }
inline void addt(float&, float) {}
inline void addt(Dual& x, float u) { x.d += u; }

struct AnswerAttention {
    static constexpr int D = 16, FW = 32, HB = 15, H = 1 << HB, C = 2048, NPHI = 24, MAXW = 24, QW = 32;
    static constexpr int ZD = 3 * D + NPHI + 1;
    // ---- dense parameters, one vector (offsets below) ----
    enum : int { oP = 0, oA = oP + D * FW, oA2 = oA + D * D, oR = oA2 + D * D, oWrq = oR + 3 * D, oBr = oWrq + D, oWo = oBr + 1, oWo2 = oWo + 1,
                 oBg = oWo2 + 1, oVv = oBg + 1, oVq = oVv + D, oBv = oVq + D, oWn = oBv + 1, oWg = oWn + 1, oB = oWg + D + 2, oWpi = oB + D * D, oBpi = oWpi + D, oWpf = oBpi + 1, oWpp = oWpf + 2, oBpp = oWpp + D, oWpc = oBpp + 1, oWqa = oWpc + 1, oWc = oWqa + D, NTH = oWc + C * ZD };
    std::vector<float> th, gth, E;                          // dense params and their gradient; word embeddings [H][D]
    std::unordered_map<uint32_t, std::array<float, D>> gE;  // sparse embedding gradient
    // ---- configuration ----
    int mode = 0;            // 1 backprop throughout; 2 backprop warm start -> forward gradients; 3 warm start -> frozen; 4 forward gradients from scratch
    uint64_t warm = 20000; int epochs = 2; float lr = 0.1f, fg_lr = 0.003f, clip = 5.f, wy = 0.1f; size_t buffer_cap = 6000; int max_items = 1024, buf_items = 256;
    const int32_t* ws = nullptr;                             // the organism's persistent workspace (read at sentence ends)
    // ---- items of the current dialogue ----
    struct SItem { uint32_t w[QW]; uint8_t nw = 0, role = 2; int8_t s[FW]; float num = 0.f, o = 0.f; };
    std::vector<SItem> items; SItem cs; bool cs_open = false; int role = 2; size_t turn_start = 0;
    std::string tok; uint32_t ndlg = 1;
    std::vector<uint32_t> df_cnt, df_last;                  // [H] dialogues containing the word, last dialogue seen
    // ---- answer vocabulary (learned from observed replies; least-used replaced when full) ----
    std::unordered_map<uint32_t, int> cls_of; std::vector<uint32_t> cls_h; std::vector<std::string> cls_surf; std::vector<float> cls_cnt, cls_nl; int nc = 0;
    // ---- the current reply ----
    struct Sample { std::vector<SItem> items; SItem q; std::vector<int8_t> qs; uint32_t prop_h = 0; float prop_c = 0.f; bool held = false; };   // qs: [q.nw][FW] workspace at each question word
    Sample cur; bool pending = false, use = false; int off = 0, dec = -1; float dec_gate = 0.f; uint32_t dec_h = 0; std::string dec_s; bool dec_nl = false; float copy_nl = 0.5f;
    std::unordered_map<uint32_t, std::string> dsurf;         // this dialogue's words as written (what a copy emits)
    std::string tok_raw; std::vector<uint32_t> turn_w; std::vector<int8_t> turn_s;   // the current user turn's words and the workspace at each
    int holdout = 5; uint64_t n_held = 0; uint32_t prop_h = 0; float prop_c = 0.f; std::string prop_s;   // the other mechanisms' proposal for the coming reply (set by the host just before the reply begins)
    std::string gold_raw; bool gold_started = false, gold_done = false;
    // ---- warm-start buffer ----
    struct Stored { Sample s; uint32_t gold_h; float y; bool has_y; };
    std::vector<Stored> buf;
    // ---- stats ----
    uint64_t answers = 0, switched_at = 0; bool fg_phase = false, frozen = false;
    uint64_t w_n = 0, w_ok = 0, w_short = 0, w_short_ok = 0, w_open = 0, w_open_ok = 0, w_shortopen = 0;
    uint64_t n_prop = 0, n_prop_ok = 0; double sum_mp = 0.0, sum_pi = 0.0; uint64_t n_mix = 0;
    uint64_t t_open = 0, t_used = 0, n_trig = 0, n_noq = 0, n_noitems = 0, n_skip = 0; std::string history;
    uint64_t rng_s = 0x9E3779B97F4A7C15ULL;

    static uint32_t hash_tok(const std::string& w) { uint64_t h = 1469598103934665603ULL; for (char c : w) h = (h ^ (uint8_t)c) * 1099511628211ULL; h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32; return (uint32_t)h | 1u; }
    inline uint32_t eb(uint32_t h) const { return h & (H - 1); }
    float urand() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (float)((rng_s >> 11) * (1.0 / 9007199254740992.0)); }
    float gauss() { float u1 = std::max(urand(), 1e-7f), u2 = urand(); return std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2); }

    void init(int m, const int32_t* workspace, uint64_t seed) {
        mode = m; ws = workspace; rng_s ^= seed * 0x2545F4914F6CDD1DULL + 1;
        th.assign(NTH, 0.f); gth.assign(NTH, 0.f); E.resize((size_t)H * D);
        for (auto& e : E) e = 0.1f * gauss();
        for (int i = 0; i < D * FW; ++i) th[oP + i] = 0.1f * gauss();
        for (int j = 0; j < D; ++j) for (int l = 0; l < D; ++l) { th[oA + j * D + l] = (j == l ? 0.5f : 0.f) + 0.02f * gauss(); th[oA2 + j * D + l] = (j == l ? 0.5f : 0.f) + 0.02f * gauss(); th[oB + j * D + l] = (j == l ? 0.5f : 0.f) + 0.02f * gauss(); }
        df_cnt.assign(H, 0); df_last.assign(H, 0);
        cls_h.assign(C, 0); cls_surf.assign(C, ""); cls_cnt.assign(C, 0.f); cls_nl.assign(C, 0.f);
        fg_phase = mode == 4; frozen = false;
    }
    float idf(uint32_t h) const { uint32_t d = df_cnt[eb(h)]; return std::log((float)(ndlg + 1) / (float)(d + 1)); }

    // ---------------------------------------------------------------- reading the stream
    void end_sentence() {
        if (!cs_open) return; cs_open = false; if (!cs.nw) return;
        cs.role = (uint8_t)role;
        for (int j = 0; j < FW; ++j) cs.s[j] = ws ? (int8_t)std::lround(127.f * std::tanh((float)ws[j] / 256.f)) : 0;
        items.push_back(cs); cs = SItem{};
    }
    void reset_dialogue() { dsurf.clear(); items.clear(); cs = SItem{}; cs_open = false; role = 2; turn_start = 0; ++ndlg; pending = false; use = false; }
    void add_token(const std::string& t) {
        uint32_t h = hash_tok(t); if (dsurf.size() < 65536) dsurf.emplace(h, tok_raw); uint32_t b = eb(h); if (df_last[b] != ndlg) { df_last[b] = ndlg; ++df_cnt[b]; }
        if (!cs_open) { cs = SItem{}; cs_open = true; }
        if (cs.nw < QW) cs.w[cs.nw++] = h;
        if (role == 0 && turn_w.size() < 256) { turn_w.push_back(h); for (int j = 0; j < FW; ++j) turn_s.push_back(ws ? (int8_t)std::lround(127.f * std::tanh((float)ws[j] / 256.f)) : 0); }
        if (cs.num == 0.f && t.size() <= 4 && std::all_of(t.begin(), t.end(), [](char c) { return c >= '0' && c <= '9'; })) cs.num = (float)std::atoi(t.c_str());
    }
    void finish_token(uint8_t next) {
        if (tok == "user" && next == ':') { end_sentence(); role = 0; turn_start = items.size(); turn_w.clear(); turn_s.clear(); return; }
        if (tok == "assistant" && next == ':') { end_sentence(); trigger(); role = 1; return; }
        if (tok == "endofdialogue" || tok == "endoftext") { end_sentence(); reset_dialogue(); return; }
        add_token(tok);
    }
    // one captured byte (self = the organism's own emission)
    void observe(uint8_t v, bool self) {
        if (pending) {                                       // the reply's first token is the answer
            ++off;
            if (!gold_done) {
                const bool an = std::isalnum(v);
                if (!gold_started) { if (an) { gold_started = true; gold_raw += (char)v; } else if (v != ' ') { gold_done = true; gold_raw.clear(); if (!self) learn(false); pending = pending && use; } }
                else if (an) { if (gold_raw.size() < 24) gold_raw += (char)v; }
                else { gold_done = true; if (!self) learn(v == '\n'); pending = pending && use; }
            }
            if (use && off > (int)dec_s.size() + 1) { use = false; pending = !gold_done; }   // stop speaking; keep listening until the answer is complete
        }
        if (std::isalnum(v)) { if (tok.size() < 32) { tok += (char)std::tolower(v); tok_raw += (char)v; } }
        else {
            if (!tok.empty()) { finish_token(v); tok.clear(); } tok_raw.clear();
            if (v == '.' || v == '?' || v == '!' || v == '\n') end_sentence();
        }
    }
    // the reply begins: build the question and the item set, decide whether to speak
    void trigger() {
        ++n_trig; pending = true; use = false; off = -1; dec = -1; dec_h = 0; dec_s.clear(); gold_raw.clear(); gold_started = gold_done = false;
        cur.items.clear(); cur.q = SItem{}; cur.prop_h = prop_h; cur.prop_c = prop_c; if (prop_h && !prop_s.empty()) dsurf.emplace(prop_h, prop_s);
        // question: the tokens of the current user turn (its last QW), the workspace now
        { if (turn_w.empty()) { ++n_noq; pending = false; return; }
          size_t st = turn_w.size() > (size_t)QW ? turn_w.size() - QW : 0; cur.qs.clear();
          for (size_t i = st; i < turn_w.size(); ++i) { cur.q.w[cur.q.nw++] = turn_w[i]; cur.qs.insert(cur.qs.end(), turn_s.begin() + i * FW, turn_s.begin() + (i + 1) * FW); } cur.q.role = 0;
          for (int j = 0; j < FW; ++j) cur.q.s[j] = ws ? (int8_t)std::lround(127.f * std::tanh((float)ws[j] / 256.f)) : 0; }
        size_t end = std::min(turn_start, items.size()); if (!end) { ++n_noitems; pending = false; return; }
        size_t st = end > (size_t)max_items ? end - max_items : 0;
        float qidf = 1e-3f; for (int j = 0; j < cur.q.nw; ++j) { bool dup = false; for (int t = 0; t < j; ++t) dup |= cur.q.w[t] == cur.q.w[j]; if (!dup) qidf += idf(cur.q.w[j]); }
        cur.items.assign(items.begin() + st, items.begin() + end);
        { uint64_t h = 1469598103934665603ULL; const SItem& f0 = items[0]; for (int j = 0; j < f0.nw; ++j) h = (h ^ f0.w[j]) * 1099511628211ULL; h ^= h >> 31; cur.held = holdout > 0 && (h % (uint64_t)holdout) == 0; }
        for (auto& it : cur.items) { float o = 0.f;
            for (int j = 0; j < cur.q.nw; ++j) { bool dup = false; for (int t = 0; t < j; ++t) dup |= cur.q.w[t] == cur.q.w[j]; if (dup) continue; for (int k = 0; k < it.nw; ++k) if (it.w[k] == cur.q.w[j]) { o += idf(cur.q.w[j]); break; } }
            it.o = o / qidf; }
        Fw<float> f; forward(cur, -1, 0, 0.f, false, f, nullptr);
        dec_h = f.amax_h; dec_gate = f.gate_p; ++t_open; sum_mp += val(f.mp); sum_pi += val(f.pi); ++n_mix;
        { auto it = cls_of.find(dec_h); if (it != cls_of.end()) { dec = it->second; dec_s = cls_surf[dec]; dec_nl = cls_nl[dec] > 0.5f; } else { auto d = dsurf.find(dec_h); if (d != dsurf.end()) dec_s = d->second; dec_nl = copy_nl > 0.5f; } }
        use = dec_h && f.gate_p > 0.5f && !dec_s.empty(); if (use) ++t_used;
    }
    // generation: the chosen answer's bytes, then its usual ending
    bool boost(float* out, float& total, float gain, bool& forcing) {
        if (!use || !pending || dec_s.empty()) return false;
        const std::string& s = dec_s; int k = off; int byte = -1;
        if (k < 0) byte = ' '; else if (k < (int)s.size()) byte = (uint8_t)s[k]; else if (k == (int)s.size() && dec_nl) byte = '\n';
        if (byte < 0) { use = false; return false; }
        float w = gain * (8.f + total); out[byte] += w; total += w; forcing = true; return true;
    }

    // ---------------------------------------------------------------- the block
    template<class T> struct Fw {
        int N = 0; T q[D], qa[D], qa2[D]; std::vector<T> k, r, r2, a, al, g, pre, th_, val, lg; T lam, S; T z[ZD]; T gl, loss; float pmax = 0.f, gate_p = 0.f; int amax = -1; float lse = 0.f;
        T qb[D]; T pil, pi, ppl, mg, mp; std::vector<float> qv, qw; float pf[2] = {0.f, 0.f}; std::vector<T> sw, be; std::vector<int> wofs; uint32_t amax_h = 0; float P = 0.f, pg = 0.f, pc = 0.f, pp = 0.f; int gc = -1; uint32_t gh = 0;
    };
    struct Tan { std::vector<float> q, k, r, r2, pre, val, lg, s; float lam = 0.f, gl = 0.f, pil = 0.f, ppl = 0.f; };
    template<class T> void embed(const SItem& it, T* out, const float* u) const {
        const float inv = it.nw ? 1.f / it.nw : 0.f; float acc[D] = {};
        for (int t = 0; t < it.nw; ++t) { const float* e = &E[(size_t)eb(it.w[t]) * D]; for (int j = 0; j < D; ++j) acc[j] += e[j]; }
        for (int j = 0; j < D; ++j) { float x = acc[j] * inv + th[oR + it.role * D + j]; for (int f = 0; f < FW; ++f) x += th[oP + j * FW + f] * (it.s[f] * (1.f / 127.f)); out[j] = T(x); if (u) addt(out[j], u[j]); }
    }
    // the question: each word's embedding plus the workspace at that word, pooled by learned attention (float; the
    // q site's tangent is added after pooling). qv: [nw][D] word vectors, qw: pooling weights.
    template<class T> void embed_question(const Sample& s, T* out, const float* u, std::vector<float>& qv, std::vector<float>& qw) const {
        const SItem& q = s.q; const int nw = q.nw; qv.assign((size_t)std::max(nw, 1) * D, 0.f); qw.assign(std::max(nw, 1), 0.f);
        if (!nw || s.qs.size() < (size_t)nw * FW) { embed(q, out, u); qv.assign(D, 0.f); qw.assign(1, 0.f); return; }
        float mx = -1e30f;
        for (int t = 0; t < nw; ++t) { const float* e = &E[(size_t)eb(q.w[t]) * D]; const int8_t* st = &s.qs[(size_t)t * FW]; float a = 0.f;
            for (int j = 0; j < D; ++j) { float x = e[j] + th[oR + j]; for (int f2 = 0; f2 < FW; ++f2) x += th[oP + j * FW + f2] * (st[f2] * (1.f / 127.f)); qv[(size_t)t * D + j] = x; a += th[oWqa + j] * x; }
            qw[t] = a; mx = std::max(mx, a); }
        float z = 0.f; for (int t = 0; t < nw; ++t) { qw[t] = std::exp(qw[t] - mx); z += qw[t]; } for (int t = 0; t < nw; ++t) qw[t] /= z;
        for (int j = 0; j < D; ++j) { float x = 0.f; for (int t = 0; t < nw; ++t) x += qw[t] * qv[(size_t)t * D + j]; out[j] = T(x); if (u) addt(out[j], u[j]); }
    }
    void question_grad(const Sample& s, const std::vector<float>& qv, const std::vector<float>& qw, const float* dq) {
        const SItem& q = s.q; const int nw = q.nw;
        if (!nw || s.qs.size() < (size_t)nw * FW) { embed_grad(q, dq); return; }
        std::vector<float> dw(nw); float sd = 0.f; for (int t = 0; t < nw; ++t) { float x = 0.f; for (int j = 0; j < D; ++j) x += dq[j] * qv[(size_t)t * D + j]; dw[t] = x; sd += qw[t] * x; }
        for (int t = 0; t < nw; ++t) { const float da = qw[t] * (dw[t] - sd); float dv[D]; const int8_t* st = &s.qs[(size_t)t * FW];
            for (int j = 0; j < D; ++j) { dv[j] = qw[t] * dq[j] + da * th[oWqa + j]; gth[oWqa + j] += da * qv[(size_t)t * D + j]; }
            auto& ge = gE[eb(q.w[t])]; for (int j = 0; j < D; ++j) { ge[j] += dv[j]; gth[oR + j] += dv[j]; for (int f2 = 0; f2 < FW; ++f2) gth[oP + j * FW + f2] += dv[j] * (st[f2] * (1.f / 127.f)); } }
    }
    // gold: class index or -1; y: numeric target when has_y
    template<class T> void forward(const Sample& s, int gold, uint32_t gh, float y, bool has_y, Fw<T>& f, const Tan* u) const {
        f.gc = gold; f.gh = gh;
        const int N = (int)s.items.size(); f.N = N;
        embed_question(s, f.q, u ? u->q.data() : nullptr, f.qv, f.qw);
        f.k.assign((size_t)N * D, T(0.f)); f.r.assign(N, T(0.f)); f.r2 = f.r; f.a = f.r; f.al = f.r; f.g = f.r; f.pre = f.r; f.th_ = f.r; f.val = f.r;
        for (int i = 0; i < N; ++i) embed(s.items[i], &f.k[(size_t)i * D], u ? &u->k[(size_t)i * D] : nullptr);
        for (int l = 0; l < D; ++l) { T x(0.f), x2(0.f); for (int j = 0; j < D; ++j) { x += th[oA + j * D + l] * f.q[j]; x2 += th[oA2 + j * D + l] * f.q[j]; } f.qa[l] = x; f.qa2[l] = x2; }
        { T x(th[oBr]); for (int j = 0; j < D; ++j) x += th[oWrq + j] * f.q[j]; if (u) addt(x, u->lam); f.lam = x; }
        T vq[D]; for (int j = 0; j < D; ++j) vq[j] = T(th[oVv + j]) + th[oVq + j] * f.q[j];
        float amax_v = -1e30f;
        for (int i = 0; i < N; ++i) {
            const T* k = &f.k[(size_t)i * D]; const SItem& it = s.items[i];
            T r(th[oWo] * it.o), r2(th[oWo2] * it.o + th[oBg]), pre(th[oBv]);
            for (int j = 0; j < D; ++j) { r += f.qa[j] * k[j]; r2 += f.qa2[j] * k[j]; pre += vq[j] * k[j]; }
            if (u) { addt(r, u->r[i]); addt(r2, u->r2[i]); addt(pre, u->pre[i]); }
            f.r[i] = r; f.r2[i] = r2; f.pre[i] = pre;
            const float rec = -std::log(1.f + (float)(N - 1 - i));
            f.a[i] = r + rec * f.lam; amax_v = std::max(amax_v, val(f.a[i]));
            f.g[i] = fsig(r2); f.th_[i] = ftanh(pre); T v = f.th_[i] + T(th[oWn] * it.num); if (u) addt(v, u->val[i]); f.val[i] = v;
        }
        T Z(0.f); for (int i = 0; i < N; ++i) { f.al[i] = fexp(f.a[i] - T(amax_v)); Z += f.al[i]; }
        const float iz = 1.f / std::max(val(Z), 1e-30f); const T izt = T(iz) * (T(2.f) - Z * T(iz));   // 1/Z to first order for the dual part
        for (int i = 0; i < N; ++i) f.al[i] = f.al[i] * izt;
        T S(0.f); for (int i = 0; i < N; ++i) S += f.g[i] * f.val[i]; f.S = S;
        for (int j = 0; j < D; ++j) { T p(0.f), pq(0.f); for (int i = 0; i < N; ++i) { p += f.al[i] * f.k[(size_t)i * D + j]; pq += f.al[i] * (f.k[(size_t)i * D + j] * f.q[j]); } f.z[j] = p; f.z[D + j] = pq; f.z[2 * D + j] = f.q[j]; }
        for (int m = 0; m < NPHI; ++m) { T d = S - T((float)m); f.z[3 * D + m] = fexp(-1.f * (d * d)); }
        f.z[ZD - 1] = T(1.f);
        f.lg.assign(nc, T(0.f)); float mx = -1e30f; int am = -1;
        for (int c = 0; c < nc; ++c) { const float* w = &th[oWc + (size_t)c * ZD]; T x(0.f); for (int m = 0; m < ZD; ++m) x += w[m] * f.z[m]; if (u) addt(x, u->lg[c]); f.lg[c] = x; if (val(x) > mx) { mx = val(x); am = c; } }
        T lse(0.f); float se = 0.f; for (int c = 0; c < nc; ++c) se += std::exp(val(f.lg[c]) - mx);
        { T s2(0.f); for (int c = 0; c < nc; ++c) s2 += fexp(f.lg[c] - T(mx)); lse = flog(s2) + T(mx); }
        f.lse = val(lse); f.amax = am; (void)se;
        // the pointer: which word of each sentence this question wants, and how much it wants to copy at all
        for (int l = 0; l < D; ++l) { T x(0.f); for (int j = 0; j < D; ++j) x += th[oB + j * D + l] * f.q[j]; f.qb[l] = x; }
        f.wofs.assign(N + 1, 0); for (int i = 0; i < N; ++i) f.wofs[i + 1] = f.wofs[i] + s.items[i].nw;
        f.sw.assign(f.wofs[N], T(0.f)); f.be = f.sw;
        for (int i = 0; i < N; ++i) { const SItem& it = s.items[i]; if (!it.nw) continue; float m = -1e30f;
            for (int t = 0; t < it.nw; ++t) { const float* e = &E[(size_t)eb(it.w[t]) * D]; T x(0.f); for (int l = 0; l < D; ++l) x += e[l] * f.qb[l]; if (u) addt(x, u->s[f.wofs[i] + t]); f.sw[f.wofs[i] + t] = x; m = std::max(m, val(x)); }
            T z(0.f); for (int t = 0; t < it.nw; ++t) { f.be[f.wofs[i] + t] = fexp(f.sw[f.wofs[i] + t] - T(m)); z += f.be[f.wofs[i] + t]; }
            const float iz2 = 1.f / val(z); const T izz = T(iz2) * (T(2.f) - z * T(iz2)); for (int t = 0; t < it.nw; ++t) f.be[f.wofs[i] + t] = f.be[f.wofs[i] + t] * izz; }
        // copy or generate: the question, plus how sure each side is (the classifier's top probability, the pointer's
        // most concentrated word) as fixed inputs — a sharp copy and a diffuse one are different evidence
        { float gmax = 0.f; for (int c = 0; c < nc; ++c) gmax = std::max(gmax, std::exp(val(f.lg[c]) - f.lse));
          std::unordered_map<uint32_t, float> pw; float cmax = 0.f; for (int i = 0; i < N; ++i) { const float ai = val(f.al[i]); if (ai < 1e-5f) continue; for (int t = 0; t < s.items[i].nw; ++t) { float& v = pw[s.items[i].w[t]]; v += ai * val(f.be[f.wofs[i] + t]); cmax = std::max(cmax, v); } }
          f.pf[0] = gmax; f.pf[1] = cmax; }
        { T x(th[oBpi] + th[oWpf] * f.pf[0] + th[oWpf + 1] * f.pf[1]); for (int j = 0; j < D; ++j) x += th[oWpi + j] * f.q[j]; if (u) addt(x, u->pil); f.pil = x; }
        { T x(th[oBpp] + th[oWpc] * s.prop_c); for (int j = 0; j < D; ++j) x += th[oWpp + j] * f.q[j]; if (u) addt(x, u->ppl); f.ppl = x; }
        { const float mx3 = std::max(0.f, std::max(val(f.pil), s.prop_h ? val(f.ppl) : -1e30f)); T e0 = fexp(T(-mx3)), e1 = fexp(f.pil - T(mx3)), e2 = s.prop_h ? fexp(f.ppl - T(mx3)) : T(0.f);
          T z3 = e0 + e1 + e2; const float iz3 = 1.f / val(z3); const T izz3 = T(iz3) * (T(2.f) - z3 * T(iz3)); f.mg = e0 * izz3; f.pi = e1 * izz3; f.mp = e2 * izz3; }
        // decision (values only): every observed answer and every word of the dialogue competes
        { std::unordered_map<uint32_t, float> pcw; const float piv = val(f.pi), mgv = val(f.mg), mpv = val(f.mp);
          for (int i = 0; i < N; ++i) { const float ai = val(f.al[i]); if (ai < 1e-5f) continue; for (int t = 0; t < s.items[i].nw; ++t) pcw[s.items[i].w[t]] += ai * val(f.be[f.wofs[i] + t]); }
          float best = 0.f; uint32_t bh = 0;
          for (int c = 0; c < nc; ++c) { float pcv = 0.f; auto it = pcw.find(cls_h[c]); if (it != pcw.end()) { pcv = it->second; it->second = -1.f; } const float p = mgv * std::exp(val(f.lg[c]) - f.lse) + piv * pcv + (cls_h[c] == s.prop_h ? mpv : 0.f); if (p > best) { best = p; bh = cls_h[c]; } }
          for (auto& [h, v] : pcw) if (v >= 0.f) { const float p = piv * v + (h == s.prop_h ? mpv : 0.f); if (p > best) { best = p; bh = h; } }
          if (s.prop_h && !cls_of.count(s.prop_h) && !pcw.count(s.prop_h) && mpv > best) { best = mpv; bh = s.prop_h; }
          f.amax_h = bh; f.pmax = best; }
        T pg(0.f); if (gold >= 0) pg = fexp(f.lg[gold] - lse);
        T pc(0.f); if (gh) for (int i = 0; i < N; ++i) for (int t = 0; t < s.items[i].nw; ++t) if (s.items[i].w[t] == gh) pc += f.al[i] * f.be[f.wofs[i] + t];
        T P = f.mg * pg + f.pi * pc + ((gh && gh == s.prop_h) ? f.mp : T(0.f)); f.P = val(P); f.pg = val(pg); f.pc = val(pc); f.pp = (gh && gh == s.prop_h) ? 1.f : 0.f;
        T loss(0.f); if (f.P > 0.f) loss = -1.f * flog(P + T(1e-6f));
        if (has_y) { T d = S - T(y); float dv = val(d); loss += wy * (std::fabs(dv) < 1.f ? 0.5f * (d * d) : (dv > 0 ? d : -1.f * d) - T(0.5f)); }
        { T x(th[oWg + D + 1] + th[oWg + D] * f.pmax); for (int j = 0; j < D; ++j) x += th[oWg + j] * f.q[j]; if (u) addt(x, u->gl); f.gl = x; f.gate_p = fsig(val(x));
          const float t = (gh && f.amax_h == gh) ? 1.f : 0.f; loss += fsoftplus(x) - t * x; }
        f.loss = loss;
    }
    // site gradients -> parameter gradients (shared by backprop and forward gradients). Site values from f.
    struct Sites { std::vector<float> q, k, r, r2, pre, val, lg, s; float lam = 0.f, gl = 0.f, pil = 0.f, ppl = 0.f; };
    template<class T> void local_grads(const Sample& s, const Fw<T>& f, const Sites& g) {
        const int N = f.N;
        for (int c = 0; c < nc; ++c) { if (g.lg[c] == 0.f) continue; float* gw = &gth[oWc + (size_t)c * ZD]; for (int m = 0; m < ZD; ++m) gw[m] += g.lg[c] * val(f.z[m]); }
        for (int j = 0; j < D; ++j) gth[oWg + j] += g.gl * val(f.q[j]); gth[oWg + D] += g.gl * f.pmax; gth[oWg + D + 1] += g.gl;
        for (int j = 0; j < D; ++j) gth[oWrq + j] += g.lam * val(f.q[j]); gth[oBr] += g.lam;
        for (int i = 0; i < N; ++i) {
            const T* k = &f.k[(size_t)i * D]; const SItem& it = s.items[i];
            gth[oWn] += g.val[i] * it.num; gth[oBv] += g.pre[i]; gth[oBg] += g.r2[i]; gth[oWo2] += g.r2[i] * it.o; gth[oWo] += g.r[i] * it.o;
            for (int j = 0; j < D; ++j) { const float kj = val(k[j]), qj = val(f.q[j]); gth[oVv + j] += g.pre[i] * kj; gth[oVq + j] += g.pre[i] * kj * qj; }
            if (g.r[i] != 0.f || g.r2[i] != 0.f) for (int j = 0; j < D; ++j) { const float qj = val(f.q[j]); for (int l = 0; l < D; ++l) { const float kl = val(k[l]); gth[oA + j * D + l] += g.r[i] * qj * kl; gth[oA2 + j * D + l] += g.r2[i] * qj * kl; } }
            embed_grad(it, &g.k[(size_t)i * D]);
        }
        question_grad(s, f.qv, f.qw, g.q.data());
        for (int j = 0; j < D; ++j) gth[oWpi + j] += g.pil * val(f.q[j]); gth[oBpi] += g.pil; gth[oWpf] += g.pil * f.pf[0]; gth[oWpf + 1] += g.pil * f.pf[1];
        for (int j = 0; j < D; ++j) gth[oWpp + j] += g.ppl * val(f.q[j]); gth[oBpp] += g.ppl; gth[oWpc] += g.ppl * s.prop_c;
        float dqbE[D] = {};
        for (int i = 0; i < N; ++i) for (int t = 0; t < s.items[i].nw; ++t) { const float d = g.s[f.wofs[i] + t]; if (d == 0.f) continue; const uint32_t b = eb(s.items[i].w[t]); const float* e = &E[(size_t)b * D]; auto& ge = gE[b];
            for (int l = 0; l < D; ++l) { dqbE[l] += d * e[l]; ge[l] += d * val(f.qb[l]); } }
        for (int j = 0; j < D; ++j) { const float qj = val(f.q[j]); for (int l = 0; l < D; ++l) gth[oB + j * D + l] += qj * dqbE[l]; }
    }
    void embed_grad(const SItem& it, const float* d) {
        for (int j = 0; j < D; ++j) { gth[oR + it.role * D + j] += d[j]; for (int f = 0; f < FW; ++f) gth[oP + j * FW + f] += d[j] * (it.s[f] * (1.f / 127.f)); }
        if (!it.nw) return; const float inv = 1.f / it.nw;
        for (int t = 0; t < it.nw; ++t) { auto& e = gE[eb(it.w[t])]; for (int j = 0; j < D; ++j) e[j] += d[j] * inv; }
    }
    // exact site gradients by backprop through the float forward
    void backward(const Sample& s, const Fw<float>& f, float y, bool has_y, Sites& g) {
        const int gold = f.gc;
        const int N = f.N; g.q.assign(D, 0.f); g.k.assign((size_t)N * D, 0.f); g.r.assign(N, 0.f); g.r2.assign(N, 0.f); g.pre.assign(N, 0.f); g.val.assign(N, 0.f); g.lg.assign(nc, 0.f); g.s.assign(f.wofs[N], 0.f); g.pil = 0.f;
        float dz[ZD] = {}; const float dP = f.P > 0.f ? -1.f / (f.P + 1e-6f) : 0.f; const float pi = f.pi, mg = f.mg, mp = f.mp;
        if (gold >= 0 && dP != 0.f) for (int c = 0; c < nc; ++c) { const float p = std::exp(f.lg[c] - f.lse); g.lg[c] = dP * mg * f.pg * ((c == gold ? 1.f : 0.f) - p); const float* w = &th[oWc + (size_t)c * ZD]; for (int m = 0; m < ZD; ++m) dz[m] += g.lg[c] * w[m]; }
        g.pil = dP * pi * (f.pc - f.P); g.ppl = s.prop_h ? dP * mp * (f.pp - f.P) : 0.f; for (int j = 0; j < D; ++j) g.q[j] += g.pil * th[oWpi + j] + g.ppl * th[oWpp + j];
        std::vector<float> dcopy(N, 0.f); float dqb[D] = {};
        if (dP != 0.f && f.gh) for (int i = 0; i < N; ++i) { const SItem& it = s.items[i]; float ci = 0.f; for (int t = 0; t < it.nw; ++t) if (it.w[t] == f.gh) ci += f.be[f.wofs[i] + t]; if (ci == 0.f) continue;
            dcopy[i] = dP * pi * ci; for (int t = 0; t < it.nw; ++t) { const float ds = dP * pi * f.al[i] * f.be[f.wofs[i] + t] * ((it.w[t] == f.gh ? 1.f : 0.f) - ci); g.s[f.wofs[i] + t] = ds; const float* e = &E[(size_t)eb(it.w[t]) * D]; for (int l = 0; l < D; ++l) dqb[l] += ds * e[l]; } }
        for (int j = 0; j < D; ++j) for (int l = 0; l < D; ++l) g.q[j] += th[oB + j * D + l] * dqb[l];
        { const float t = (f.gh && f.amax_h == f.gh) ? 1.f : 0.f; g.gl = f.gate_p - t; for (int j = 0; j < D; ++j) g.q[j] += g.gl * th[oWg + j]; }
        float dS = 0.f; for (int m = 0; m < NPHI; ++m) dS += dz[3 * D + m] * f.z[3 * D + m] * (-2.f * (f.S - (float)m));
        if (has_y) { float d = f.S - y; dS += wy * std::max(-1.f, std::min(1.f, d)); }
        for (int j = 0; j < D; ++j) g.q[j] += dz[2 * D + j];
        std::vector<float> dal(N);
        float dqa[D] = {}, dqa2[D] = {}; float sd = 0.f;
        for (int i = 0; i < N; ++i) {
            const float* k = &f.k[(size_t)i * D]; float* dk = &g.k[(size_t)i * D];
            const float dval = dS * f.g[i]; g.val[i] = dval; const float dpre = dval * (1.f - f.th_[i] * f.th_[i]); g.pre[i] = dpre;
            const float dg = dS * f.val[i]; const float dr2 = dg * f.g[i] * (1.f - f.g[i]); g.r2[i] = dr2;
            float da = dcopy[i]; for (int j = 0; j < D; ++j) da += dz[j] * k[j] + dz[D + j] * k[j] * f.q[j]; dal[i] = da; sd += f.al[i] * da;
            for (int j = 0; j < D; ++j) { dk[j] += dpre * (th[oVv + j] + th[oVq + j] * f.q[j]) + dr2 * f.qa2[j] + f.al[i] * (dz[j] + dz[D + j] * f.q[j]);
                g.q[j] += dpre * th[oVq + j] * k[j] + f.al[i] * dz[D + j] * k[j]; dqa2[j] += dr2 * k[j]; }
        }
        float dlam = 0.f;
        for (int i = 0; i < N; ++i) { const float da = f.al[i] * (dal[i] - sd); g.r[i] = da; const float rec = -std::log(1.f + (float)(N - 1 - i)); dlam += da * rec;
            const float* k = &f.k[(size_t)i * D]; float* dk = &g.k[(size_t)i * D]; for (int j = 0; j < D; ++j) { dk[j] += da * f.qa[j]; dqa[j] += da * k[j]; } }
        g.lam = dlam; for (int j = 0; j < D; ++j) g.q[j] += dlam * th[oWrq + j];
        for (int j = 0; j < D; ++j) for (int l = 0; l < D; ++l) g.q[j] += th[oA + j * D + l] * dqa[l] + th[oA2 + j * D + l] * dqa2[l];
    }
    int split = 0;   // 0 all parameters; 1 answering parts only; 2 trust only (gate, source mixture)
    bool is_trust(int i) const { return (i >= oWg && i < oWg + D + 2) || (i >= oWpi && i < oWqa); }   // Wg; Wpi, Bpi, Wpf, Wpp, Bpp, Wpc
    void apply(float rate) {
        if (split) { for (int i = 0; i < NTH; ++i) if (is_trust(i) != (split == 2)) gth[i] = 0.f; if (split == 2) gE.clear(); }
        double n2 = 0.0; for (float x : gth) n2 += (double)x * x; for (auto& [b, e] : gE) for (float x : e) n2 += (double)x * x;
        const float sc = n2 > (double)clip * clip ? clip / (float)std::sqrt(n2) : 1.f;
        for (int i = 0; i < NTH; ++i) { th[i] -= rate * sc * gth[i]; gth[i] = 0.f; }
        for (auto& [b, e] : gE) { float* p = &E[(size_t)b * D]; for (int j = 0; j < D; ++j) p[j] -= rate * sc * e[j]; }
        gE.clear();
    }
    void step_backprop(const Sample& s, int gold, uint32_t gh, float y, bool has_y, float rate) {
        Fw<float> f; forward(s, gold, gh, y, has_y, f, nullptr); Sites g; backward(s, f, y, has_y, g); local_grads(s, f, g); apply(rate);
    }
    // forward gradients for the hidden sites; the output layers (answer logits, trust gate) take their exact LOCAL
    // error (the delta rule, as the organism's own readout does: no signal travels backwards through the block).
    // fg_k tangents per step are averaged (variance falls as 1/fg_k at fg_k forward-mode passes).
    int fg_k = 1; bool fg_local_out = true;
    void step_forward_gradient(const Sample& s, int gold, uint32_t gh, float y, bool has_y, float rate) {
        const int N = (int)s.items.size(); size_t nw = 0; for (auto& it : s.items) nw += it.nw;
        for (int rep = 0; rep < fg_k; ++rep) {
            Tan u; auto fill = [&](std::vector<float>& v, size_t n) { v.resize(n); for (auto& x : v) x = gauss(); };
            fill(u.q, D); fill(u.k, (size_t)N * D); fill(u.r, N); fill(u.r2, N); fill(u.pre, N); fill(u.val, N); fill(u.lg, nc); fill(u.s, nw); u.lam = gauss(); u.gl = gauss(); u.pil = gauss(); u.ppl = s.prop_h ? gauss() : 0.f;
            if (fg_local_out) { std::fill(u.lg.begin(), u.lg.end(), 0.f); u.gl = 0.f; }
            Fw<Dual> f; forward(s, gold, gh, y, has_y, f, &u);
            const float delta = f.loss.d / (float)fg_k;       // directional derivative of the loss along all site tangents at once
            Sites g; auto scale = [&](const std::vector<float>& a, std::vector<float>& b) { b.resize(a.size()); for (size_t i = 0; i < a.size(); ++i) b[i] = delta * a[i]; };
            scale(u.q, g.q); scale(u.k, g.k); scale(u.r, g.r); scale(u.r2, g.r2); scale(u.pre, g.pre); scale(u.val, g.val); scale(u.lg, g.lg); scale(u.s, g.s); g.lam = delta * u.lam; g.gl = delta * u.gl; g.pil = delta * u.pil; g.ppl = delta * u.ppl;
            if (fg_local_out && rep == 0) {                  // exact output errors from the forward values alone
                const float dP = f.P > 0.f ? -1.f / (f.P + 1e-6f) : 0.f; const float mg = val(f.mg);
                if (gold >= 0 && dP != 0.f) for (int c = 0; c < nc; ++c) { const float p = std::exp(val(f.lg[c]) - f.lse); g.lg[c] = dP * mg * f.pg * ((c == gold ? 1.f : 0.f) - p); }
                g.gl = f.gate_p - ((gh && f.amax_h == gh) ? 1.f : 0.f);
            }
            local_grads(s, f, g);
        }
        apply(rate);
    }

    // ---------------------------------------------------------------- learning from the observed reply
    int class_for(const std::string& raw, bool nl, bool insert) {
        std::string low; for (char c : raw) low += (char)std::tolower((uint8_t)c);
        uint32_t h = hash_tok(low); auto it = cls_of.find(h);
        if (it != cls_of.end()) { int c = it->second; cls_cnt[c] += 1.f; cls_nl[c] += 0.1f * ((nl ? 1.f : 0.f) - cls_nl[c]); if (insert && nl) cls_surf[c] = raw; return c; }   // the written form a short answer uses
        if (!insert) return -1;
        int c; if (nc < C) c = nc++; else { c = 0; for (int i = 1; i < C; ++i) if (cls_cnt[i] < cls_cnt[c]) c = i; cls_of.erase(cls_h[c]); }
        cls_of[h] = c; cls_h[c] = h; cls_surf[c] = raw; cls_cnt[c] = 1.f; cls_nl[c] = nl ? 1.f : 0.f;
        std::fill(&th[oWc + (size_t)c * ZD], &th[oWc + (size_t)(c + 1) * ZD], 0.f);
        return c;
    }
    void learn(bool nl) {
        if (gold_raw.empty() || cur.items.empty() || !cur.q.nw) { ++n_skip; return; }
        for (int c = 0; c < nc; ++c) cls_cnt[c] *= 0.99995f;   // slow: an answer vocabulary must survive long stretches of chat
        // the prediction made before learning (for the record)
        const uint32_t gh = hash_tok([&] { std::string l; for (char c : gold_raw) l += (char)std::tolower((uint8_t)c); return l; }());
        const int gold = class_for(gold_raw, nl, true); copy_nl += 0.05f * ((nl ? 1.f : 0.f) - copy_nl);
        const bool has_y = gold_raw.size() <= 3 && std::all_of(gold_raw.begin(), gold_raw.end(), [](char c) { return c >= '0' && c <= '9'; });
        const float y = has_y ? (float)std::atoi(gold_raw.c_str()) : 0.f;
        ++answers; ++w_n; const bool ok = dec_h && dec_h == gh; if (cur.prop_h) { ++n_prop; n_prop_ok += cur.prop_h == gh; } w_ok += ok; if (nl) { ++w_short; w_short_ok += ok; }
        if (dec_h && dec_gate > 0.5f) { ++w_open; w_open_ok += ok; if (nl) ++w_shortopen; }
        const bool warm_phase = (mode == 2 || mode == 3) && !fg_phase && !frozen;
        split = holdout > 0 ? (cur.held ? 2 : 1) : 0; n_held += cur.held;
        if (mode == 1 || warm_phase) step_backprop(cur, gold, gh, y, has_y, lr);
        else if (fg_phase && !frozen) step_forward_gradient(cur, gold, gh, y, has_y, fg_lr);
        if (warm_phase) {
            if (buf.size() < buffer_cap) { Stored st; st.s.q = cur.q; st.s.qs = cur.qs; st.s.held = cur.held; st.s.prop_h = cur.prop_h; st.s.prop_c = cur.prop_c; size_t n0 = cur.items.size() > (size_t)buf_items ? cur.items.size() - buf_items : 0; st.s.items.assign(cur.items.begin() + n0, cur.items.end()); st.gold_h = gh; st.y = y; st.has_y = has_y; buf.push_back(std::move(st)); }
            else { size_t j = (size_t)(urand() * answers); if (j < buffer_cap) { Stored& st = buf[j]; st.s.q = cur.q; st.s.qs = cur.qs; st.s.held = cur.held; st.s.prop_h = cur.prop_h; st.s.prop_c = cur.prop_c; size_t n0 = cur.items.size() > (size_t)buf_items ? cur.items.size() - buf_items : 0; st.s.items.assign(cur.items.begin() + n0, cur.items.end()); st.gold_h = gh; st.y = y; st.has_y = has_y; } }   // reservoir
            if (answers >= warm) switch_phase();
        }
        if (w_n >= 2000) window_report();
    }
    void switch_phase() {
        window_report(); history += "| switch at " + std::to_string(answers) + " answers: replay " + std::to_string(buf.size()) + " x " + std::to_string(epochs) + " ";
        for (int e = 0; e < epochs; ++e) {
            for (size_t i = buf.size(); i > 1; --i) { size_t j = (size_t)(urand() * i); std::swap(buf[i - 1], buf[j]); }
            for (auto& st : buf) { auto it = cls_of.find(st.gold_h); split = holdout > 0 ? (st.s.held ? 2 : 1) : 0; step_backprop(st.s, it == cls_of.end() ? -1 : it->second, st.gold_h, st.y, st.has_y, lr); }
        }
        buf.clear(); buf.shrink_to_fit(); switched_at = answers;
        if (mode == 2) fg_phase = true; else frozen = true;
        std::fprintf(stderr, "attn: switch at %llu answers -> %s\n", (unsigned long long)answers, mode == 2 ? "forward gradients" : "frozen");
    }
    void window_report() {
        if (!w_n) return;
        const char* ph = frozen ? "frozen" : fg_phase ? "fwdgrad" : "backprop";
        char b[256]; std::snprintf(b, sizeof b, "attn: %llu answers [%s] acc %.3f, short-answer acc %.3f (%llu), spoke %.3f of turns with precision %.3f, classes %d",
            (unsigned long long)answers, ph, (double)w_ok / w_n, w_short ? (double)w_short_ok / w_short : 0.0, (unsigned long long)w_short, (double)w_open / w_n, w_open ? (double)w_open_ok / w_open : 0.0, nc);
        std::fprintf(stderr, "%s\n", b);
        char h[96]; std::snprintf(h, sizeof h, "%s%llu:%.2f/%.2f ", ph[0] == 'b' ? "B" : ph[0] == 'f' && ph[1] == 'w' ? "F" : "Z", (unsigned long long)answers, w_short ? (double)w_short_ok / w_short : 0.0, w_open ? (double)w_open_ok / w_open : 0.0);
        history += h; if (history.size() > 900) history.erase(0, history.size() - 900);
        w_n = w_ok = w_short = w_short_ok = w_open = w_open_ok = w_shortopen = 0;
    }
    std::string report() const {
        char b[1200]; std::snprintf(b, sizeof b, "attention: %llu answers learned (%llu turns: %llu no question, %llu no items, %llu skipped), spoke on %llu of %llu reply turns, %d answer classes; held-out %llu, proposals %llu (right %llu), mean weights proposal %.2f copy %.2f; windows (phase answers:short-acc/precision) %s",
            (unsigned long long)answers, (unsigned long long)n_trig, (unsigned long long)n_noq, (unsigned long long)n_noitems, (unsigned long long)n_skip, (unsigned long long)t_used, (unsigned long long)t_open, nc, (unsigned long long)n_held, (unsigned long long)n_prop, (unsigned long long)n_prop_ok, n_mix ? sum_mp / n_mix : 0.0, n_mix ? sum_pi / n_mix : 0.0, history.c_str());
        return b;
    }
};

} // namespace laminar
