// Distributed-input cooperation test for Coco bodies (user protocol, 2026-09-24; docs/BODY_CIRCUIT_RESEARCH_LOG.md).
//
// No single unit sees everything needed to answer. Units are real Coco bodies: six layers, strict INT4 weights (born
// -2..2), the organism's right shifts {1,2,2,3,3,3}, int8 activations clamped at +-127, and the organism's quantised
// accumulator update (INT16 residual, quantum 4096, stochastic rounding). Wiring is explicit and fixed:
//   a sender's L2 reaches its receiver's L3 (message half) one tick later; every unit's L4 (fed by its L6) returns into
//   its own L2 (16) and L6 (8) on the next tick, so a completed result can be re-sent upward (recurrent processing).
// Only the ROOT's final answer is supervised (its readout at the last tick) unless local targets are switched on.
//
// usage: coop_test TASK RULE CONDITION [steps=40000] [seed=1]
//   TASK: xor2   (units A, B see bit a / bit b plus 3 random distractor bits each; root R sees nothing; 2 ticks)
//         par4   (4 input units -> 2 middle units -> root; 4 ticks)
//         copy2  (control: same wiring as xor2, answer = a; one sender suffices)
//   RULE: bp (exact backprop through units and ticks) | np (paired node perturbation, shared reward)
//   CONDITION: open (all routes delivered) | gated (usefulness-gated delivery, the organism's route gate: a message is
//              delivered only while its counterfactual gain EMA >= 0.05 bits, starting at 0.10) | local (open + every
//              non-root unit has its own readout and answer target) | oracle (senders' messages replaced by the true
//              intermediate value: the bit for xor2/copy2, the pair parity for par4's middle units)
// Prints accuracy on 2,000 fresh frozen examples every 4,000 steps, then (open/gated/local) a message intervention:
// each input unit's message is replaced by the message it sends for the opposite value of its bit; reports how often
// the root's answer flips (XOR/parity: a flip is the correct consequence of using the message).
#include "laminar/config.h"
#include "laminar/int4.h"
#include <algorithm>
#include <cstdlib>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace laminar;
namespace {
constexpr int IN = Dims::IN, A = Dims::A, B = Dims::B, C = Dims::C, E = Dims::E, F = Dims::F, R = Dims::R, R2 = Dims::R2, R6 = Dims::R6;
const int SH[6] = {1, 2, 2, 3, 3, 3};   // L1, L2, L3, L5, L6, L4
constexpr int QUANTUM = 64 << 6, HALF = QUANTUM / 2;
std::mt19937_64 rng;
float uni() { return (float)((rng() >> 40) * (1.0 / 16777216.0)); }

struct Body {
    std::vector<uint8_t> w; std::vector<int16_t> acc;   // UNIT_BYTES packed INT4, residual per synapse
    Body() : w(Dims::UNIT_BYTES), acc(Dims::UNIT_NIBBLES, 0) { for (int n = 0; n < Dims::UNIT_NIBBLES; ++n) nib_set(w.data(), n, (int)(rng() % 5) - 2); }
    int wt(int nib) const { return nib_get(w.data(), nib); }
    void layer(int nib0, int sh, const int8_t* x, int in, int8_t* y, int out) const {
        for (int j = 0; j < out; ++j) { int32_t s = 0; for (int i = 0; i < in; ++i) s += x[i] * wt(nib0 + j * in + i); y[j] = clamp8(s >> sh); }
    }
    void add(int nib, int32_t delta) {   // the organism's apply_acc16
        if (!delta) return; int32_t v = acc[nib] + delta;
        if (v >= HALF) { int cw = wt(nib); if (cw < 7) { nib_set(w.data(), nib, cw + 1); v -= QUANTUM; } else v = HALF - 1; }
        else if (v <= -HALF) { int cw = wt(nib); if (cw > -8) { nib_set(w.data(), nib, cw - 1); v += QUANTUM; } else v = -(HALF - 1); }
        acc[nib] = (int16_t)std::max(-32767, std::min(32767, v));
    }
};
// one tick of one unit: all layer inputs/outputs kept for learning
struct Tick { int8_t x2[A + R2], x3[A + B], x5[B + C], x6[E + R6], l1[A], l2[B], l3[C], l5[E], l6[F], l4[R]; };

struct Net {
    int nu = 0, root = 0, ticks = 2; std::vector<Body> body; std::vector<int> type; std::vector<std::vector<int>> senders;   // type[u]: which body (weights) unit u uses;   // senders[r]: units whose L2 reaches r
    std::vector<float> wout, bout;                               // root readout 2 x F (+ bias)
    std::vector<std::vector<float>> lw; std::vector<std::vector<float>> lb;   // local readouts (condition local)
    std::vector<float> util;                                     // gated: per (receiver, sender slot) usefulness EMA
};

int nclass = 2; int g_ninputs = 2; bool g_oracle_inputs_only = false;
struct Example { std::vector<int8_t> x; int y; std::vector<int> inter; std::vector<int> bits; };   // x: nu*IN; inter: oracle values per sender unit

// message slot of receiver r: senders' L2 halves side by side (2 senders x 12 dims) or the full 24 for one sender
void message(const Net& n, int r, const std::vector<Tick>& prev, const std::vector<int8_t>* oracle, const std::vector<uint8_t>& deliver, int8_t* m) {
    std::memset(m, 0, B);
    const auto& s = n.senders[r]; if (s.empty()) return;
    static const bool sum_mix = std::getenv("COOP_MIX") && std::string(std::getenv("COOP_MIX")) == "sum";
    if (sum_mix) { int32_t acc[B] = {0};
        for (size_t k = 0; k < s.size(); ++k) { if (!deliver.empty() && !deliver[(size_t)r * 4 + k]) continue; const bool orc = oracle && (!g_oracle_inputs_only || s[k] < g_ninputs);
            for (int i = 0; i < B; ++i) acc[i] += orc ? (*oracle)[s[k]] * (k ? -1 : 1) * ((i & 1) ? 1 : -1) : prev[s[k]].l2[i]; }
        for (int i = 0; i < B; ++i) m[i] = clamp8(acc[i]); return; }
    const int w = B / (int)s.size();
    for (size_t k = 0; k < s.size(); ++k) {
        if (!deliver.empty() && !deliver[(size_t)r * 4 + k]) continue;
        const bool orc = oracle && (!g_oracle_inputs_only || s[k] < g_ninputs);
        for (int i = 0; i < w; ++i) m[k * w + i] = orc ? (*oracle)[s[k]] : prev[s[k]].l2[i];
    }
}

// forward through all ticks; returns root logits; T[t][u] kept. perturbation: per (tick, unit, layer) sign noise
void forward(const Net& n, const Example& ex, std::vector<std::vector<Tick>>& T, float* logits, const std::vector<int8_t>* oracle,
             const std::vector<uint8_t>& deliver, int psign, uint64_t pseed, int pdelta, std::vector<std::vector<float>>* local_logits = nullptr,
             int swap_unit = -1, const int8_t* swap_l2 = nullptr) {
    T.assign(n.ticks, std::vector<Tick>(n.nu));
    auto perturb = [&](int8_t* v, int len, int t, int u, int l) { if (!psign) return; uint64_t h = pseed ^ ((uint64_t)t << 48) ^ ((uint64_t)u << 40) ^ ((uint64_t)l << 32);
        for (int j = 0; j < len; ++j) { h = h * 6364136223846793005ULL + 1442695040888963407ULL; int s = (h >> 63) ? 1 : -1; v[j] = clamp8(v[j] + psign * s * pdelta); } };
    for (int t = 0; t < n.ticks; ++t) for (int u = 0; u < n.nu; ++u) {
        Tick& k = T[t][u]; const Body& b = n.body[n.type[u]];
        b.layer(Dims::NIB_L1, SH[0], &ex.x[(size_t)u * IN], IN, k.l1, A); perturb(k.l1, A, t, u, 0);
        const Tick* pv = t ? &T[t - 1][u] : nullptr;
        std::memcpy(k.x2, k.l1, A); if (pv) std::memcpy(k.x2 + A, pv->l4, R2); else std::memset(k.x2 + A, 0, R2);
        b.layer(Dims::NIB_L2, SH[1], k.x2, A + R2, k.l2, B); perturb(k.l2, B, t, u, 1);
        if (swap_unit == u && swap_l2) std::memcpy(k.l2, swap_l2 + (size_t)t * B, B);   // intervention: this unit sends another message
        std::memcpy(k.x3, k.l1, A);
        if (t) message(n, u, T[t - 1], oracle, deliver, k.x3 + A); else std::memset(k.x3 + A, 0, B);
        b.layer(Dims::NIB_L3, SH[2], k.x3, A + B, k.l3, C); perturb(k.l3, C, t, u, 2);
        std::memcpy(k.x5, k.l2, B); std::memcpy(k.x5 + B, k.l3, C);
        b.layer(Dims::NIB_L5, SH[3], k.x5, B + C, k.l5, E); perturb(k.l5, E, t, u, 3);
        std::memcpy(k.x6, k.l5, E); if (pv) std::memcpy(k.x6 + E, pv->l4 + R2, R6); else std::memset(k.x6 + E, 0, R6);
        b.layer(Dims::NIB_L6, SH[4], k.x6, E + R6, k.l6, F); perturb(k.l6, F, t, u, 4);
        b.layer(Dims::NIB_L4, SH[5], k.l6, F, k.l4, R);
    }
    const Tick& o = T[n.ticks - 1][n.root];
    for (int c = 0; c < nclass; ++c) { float s = n.bout[c]; for (int j = 0; j < F; ++j) s += n.wout[c * F + j] * o.l6[j] / 128.f; logits[c] = s; }
    if (local_logits) { local_logits->assign(n.nu, std::vector<float>(nclass, 0.f));
        for (int u = 0; u < n.nu; ++u) if (u != n.root && !n.lw.empty()) { const Tick& q = T[n.ticks - 1][u];
            for (int c = 0; c < nclass; ++c) { float s = n.lb[u][c]; for (int j = 0; j < F; ++j) s += n.lw[u][c * F + j] * q.l6[j] / 128.f; (*local_logits)[u][c] = s; } } }
}
float xent(const float* lg, int y, float* p) { float mx = std::max(lg[0], lg[1]), z = 0; for (int c = 0; c < nclass; ++c) { p[c] = std::exp(lg[c] - mx); z += p[c]; } for (int c = 0; c < nclass; ++c) p[c] /= z; return -std::log(std::max(p[y], 1e-9f)); }

// quantised update of one layer from a float error vector g (pre-activation, per output) and int8 inputs x
void update(Body& b, int nib0, const int8_t* x, int in, const float* g, int out, float qs, int shift) {
    for (int j = 0; j < out; ++j) { float v = std::max(-127.f, std::min(127.f, g[j] * qs)); float fl = std::floor(v); int32_t q = (int32_t)fl + (uni() < v - fl ? 1 : 0); if (!q) continue;
        for (int i = 0; i < in; ++i) { int32_t d = -(int32_t)x[i] * q; int32_t f2 = d >> shift, frac = d - (f2 << shift); d = f2 + ((int32_t)(rng() & ((1u << shift) - 1)) < frac ? 1 : 0); b.add(nib0 + j * in + i, d); } }
}
}   // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: coop_test xor2|par4|copy2 bp|np open|gated|local|oracle [steps] [seed]\n"; return 1; }
    const std::string task = argv[1], rule = argv[2], cond = argv[3];
    const int steps = argc > 4 ? std::atoi(argv[4]) : 40000; rng.seed(argc > 5 ? std::atoll(argv[5]) : 1);
    Net n;
    if (task == "par4") { n.nu = 7; n.root = 6; n.ticks = 4; n.senders = {{}, {}, {}, {}, {0, 1}, {2, 3}, {4, 5}}; }
    else if (task == "relay") { n.nu = 3; n.root = 2; n.ticks = 4; n.senders = {{}, {0}, {1}}; }          // bit -> middle -> root
    else if (task == "relay2") { n.nu = 4; n.root = 3; n.ticks = 4; n.senders = {{}, {}, {0, 1}, {2}}; }   // XOR in the middle, relayed to the root
    else { n.nu = 3; n.root = 2; n.ticks = 2; n.senders = {{}, {}, {0, 1}}; }
    const int ninputs = task == "par4" ? 4 : task == "relay" ? 1 : 2; g_ninputs = ninputs;
    n.type.resize(n.nu); for (int u = 0; u < n.nu; ++u) n.type[u] = u;
    const char* share = std::getenv("COOP_SHARE");   // "types": all input units share one body, all combining units (receivers) another
    if (share && std::string(share) == "types") for (int u = 0; u < n.nu; ++u) n.type[u] = n.senders[u].empty() ? 0 : 1;
    if (const char* tk = std::getenv("COOP_TICKS")) n.ticks = std::atoi(tk);
    { int nb = 0; for (int u = 0; u < n.nu; ++u) nb = std::max(nb, n.type[u] + 1); for (int k = 0; k < nb; ++k) n.body.emplace_back(); }
    n.wout.assign(nclass * F, 0.f); n.bout.assign(nclass, 0.f); for (auto& v : n.wout) v = (uni() * 2 - 1) * 0.05f;
    if (cond == "local") { n.lw.assign(n.nu, std::vector<float>(nclass * F, 0.f)); n.lb.assign(n.nu, std::vector<float>(nclass, 0.f)); for (auto& r : n.lw) for (auto& v : r) v = (uni() * 2 - 1) * 0.05f; }
    n.util.assign((size_t)n.nu * 4, 0.10f);
    const bool gated = cond == "gated", oracle_mode = cond == "oracle" || cond == "oraclein", local = cond == "local";
    g_oracle_inputs_only = cond == "oraclein";   // only the INPUT units' messages are supplied; middle units must compute and relay

    const int curric = std::getenv("COOP_CURRIC") ? std::atoi(std::getenv("COOP_CURRIC")) : 0;   // par4: first N steps with bits 2,3 held at 0
    int cur_step = 0;
    auto sample = [&](std::mt19937_64& g) {
        Example ex; ex.x.assign((size_t)n.nu * IN, 0); ex.bits.resize(ninputs);
        for (int i = 0; i < ninputs; ++i) { ex.bits[i] = (int)(g() & 1); if (task == "par4" && i >= 2 && cur_step > 0 && cur_step <= curric) ex.bits[i] = 0;
            int8_t* x = &ex.x[(size_t)i * IN];
            for (int k = 0; k < 16; ++k) x[k] = ex.bits[i] ? 32 : -32;                                   // the unit's own bit
            for (int d = 1; d <= 3; ++d) { const int8_t v = (g() & 1) ? 32 : -32; for (int k = 0; k < 16; ++k) x[16 * d + k] = v; }   // distractors
        }
        for (int u = ninputs; u < n.nu; ++u) for (int k = 0; k < 16; ++k) ex.x[(size_t)u * IN + k] = 32;   // non-input units: a constant context code
        if (task == "copy2" || task == "relay") ex.y = ex.bits[0]; else { int p = 0; for (int b : ex.bits) p ^= b; ex.y = p; }
        ex.inter.assign(n.nu, 0);   // oracle: what each SENDER ideally sends (+-64 code)
        for (int i = 0; i < ninputs; ++i) ex.inter[i] = ex.bits[i] ? 64 : -64;
        if (task == "par4") { ex.inter[4] = (ex.bits[0] ^ ex.bits[1]) ? 64 : -64; ex.inter[5] = (ex.bits[2] ^ ex.bits[3]) ? 64 : -64; }
        if (task == "relay") ex.inter[1] = ex.inter[0];
        if (task == "relay2") ex.inter[2] = (ex.bits[0] ^ ex.bits[1]) ? 64 : -64;
        return ex;
    };
    std::mt19937_64 train_g(7), test_g(12345);
    std::vector<Example> test; for (int i = 0; i < 2000; ++i) test.push_back(sample(test_g));
    std::vector<uint8_t> deliver;   // gated: which (receiver, slot) links deliver this step
    auto oracle_vec = [&](const Example& ex) { std::vector<int8_t> o(n.nu); for (int u = 0; u < n.nu; ++u) o[u] = (int8_t)ex.inter[u]; return o; };
    auto accuracy = [&]() {
        int ok = 0; std::vector<std::vector<Tick>> T; float lg[2];
        std::vector<uint8_t> dl; if (gated) { dl.assign((size_t)n.nu * 4, 0); for (size_t i = 0; i < dl.size(); ++i) dl[i] = n.util[i] >= 0.05f; }
        for (auto& ex : test) { auto o = oracle_vec(ex); forward(n, ex, T, lg, oracle_mode ? &o : nullptr, dl, 0, 0, 0); ok += (lg[1] > lg[0]) == (ex.y == 1); }
        return 100.0 * ok / test.size();
    };
    const float lr_out = 0.005f, err_gain = 2.f; const int shift = 11 - 6;   // lr_shift=11 with 16-bit accumulators (the harness's best C++ setting)
    std::cout << task << " " << rule << " " << cond << ": step 0 accuracy " << accuracy() << "%" << std::flush;
    for (int step = 1; step <= steps; ++step) {
        cur_step = step; Example ex = sample(train_g); auto o = oracle_vec(ex); const auto* op = oracle_mode ? &o : nullptr;
        if (gated) { deliver.assign((size_t)n.nu * 4, 0); for (size_t i = 0; i < deliver.size(); ++i) deliver[i] = n.util[i] >= 0.05f; }
        std::vector<std::vector<Tick>> T; float lg[2], pr[2]; std::vector<std::vector<float>> llg;
        forward(n, ex, T, lg, op, deliver, 0, 0, 0, local ? &llg : nullptr);
        const float loss = xent(lg, ex.y, pr);
        if (gated) {   // counterfactual usefulness of each link: loss without it minus loss with it (bits), EMA 0.99 (organism route utility)
            for (int r = 0; r < n.nu; ++r) for (size_t k = 0; k < n.senders[r].size(); ++k) {
                const size_t id = (size_t)r * 4 + k; if (!deliver[id]) { n.util[id] = 0.99f * n.util[id]; continue; }   // an undelivered link earns nothing
                auto d2 = deliver; d2[id] = 0; std::vector<std::vector<Tick>> T2; float lg2[2], p2[2]; forward(n, ex, T2, lg2, op, d2, 0, 0, 0);
                n.util[id] = 0.99f * n.util[id] + 0.01f * (xent(lg2, ex.y, p2) - loss) / 0.6931f; }
        }
        // readout (float SGD) and the error at the root's L6
        float dl[2] = {pr[0] - (ex.y == 0), pr[1] - (ex.y == 1)};
        std::vector<std::vector<std::array<float, F>>> g6(n.ticks, std::vector<std::array<float, F>>(n.nu)); for (auto& a : g6) for (auto& b : a) b.fill(0.f);
        for (int j = 0; j < F; ++j) for (int c = 0; c < nclass; ++c) g6[n.ticks - 1][n.root][j] += dl[c] * n.wout[c * F + j];
        const Tick& ot = T[n.ticks - 1][n.root];
        for (int c = 0; c < nclass; ++c) { for (int j = 0; j < F; ++j) n.wout[c * F + j] -= lr_out * dl[c] * ot.l6[j] / 128.f; n.bout[c] -= lr_out * dl[c]; }
        if (local) for (int u = 0; u < n.nu; ++u) if (u != n.root) {   // every intermediate unit also answers on its own
            float lp[2]; xent(llg[u].data(), ex.y, lp); float ld[2] = {lp[0] - (ex.y == 0), lp[1] - (ex.y == 1)}; const Tick& q = T[n.ticks - 1][u];
            for (int j = 0; j < F; ++j) for (int c = 0; c < nclass; ++c) g6[n.ticks - 1][u][j] += ld[c] * n.lw[u][c * F + j];
            for (int c = 0; c < nclass; ++c) { for (int j = 0; j < F; ++j) n.lw[u][c * F + j] -= lr_out * ld[c] * q.l6[j] / 128.f; n.lb[u][c] -= lr_out * ld[c]; } }
        const float qs = err_gain * 127.f;
        // rules computed from the exact site gradients: bp (all of them), pcK (predictive-coding settling with K settle steps:
        // credit moves one layer per step, so a site more than K layer-steps from the answer receives none), fg / fgw
        // (forward gradients: one random tangent per hidden site, estimate (g.u)u, exact local delta rule at the root's L6;
        // fgw = backprop warm start for COOP_WARM steps first)
        const bool pc = rule.rfind("pc", 0) == 0; const int pcK = pc ? std::atoi(rule.c_str() + 2) : 1 << 30;
        const int warm = std::getenv("COOP_WARM") ? std::atoi(std::getenv("COOP_WARM")) : 20000;
        const bool fg = rule == "fg" || (rule == "fgw" && step > warm);
        const int fgk = std::getenv("COOP_FGK") ? std::atoi(std::getenv("COOP_FGK")) : 1;
        if (rule == "bp" || pc || rule == "fg" || rule == "fgw") {   // exact backprop through units and ticks (the forward's clamp masks and shifts)
            auto live = [](int8_t y) { return (y >= 127 || y <= -127) ? 0.f : 1.f; };
            constexpr int INF = 1 << 20;
            std::vector<std::vector<int>> dl4(n.ticks, std::vector<int>(n.nu, INF)), dl2x(n.ticks, std::vector<int>(n.nu, INF));
            struct SiteG { float g4[R], g6[F], g5[E], g2[B], g3[C], g1[A]; };
            std::vector<std::vector<SiteG>> G(n.ticks, std::vector<SiteG>(n.nu));
            std::vector<std::vector<std::array<float, B>>> gl2x(n.ticks, std::vector<std::array<float, B>>(n.nu));   // extra error on a unit's L2 from its receivers
            std::vector<std::vector<std::array<float, R>>> gl4(n.ticks, std::vector<std::array<float, R>>(n.nu));
            for (auto& a : gl2x) for (auto& b : a) b.fill(0.f); for (auto& a : gl4) for (auto& b : a) b.fill(0.f);
            for (int t = n.ticks - 1; t >= 0; --t) for (int u = n.nu - 1; u >= 0; --u) {
                const Tick& k = T[t][u]; Body& b = n.body[n.type[u]];
                float gl6o[F]; for (int j = 0; j < F; ++j) gl6o[j] = g6[t][u][j];
                // L4 (feeds next tick's L2 and L6 return)
                const bool out_site = u == n.root && t == n.ticks - 1;
                const int d4 = dl4[t][u], d6 = std::min(out_site ? 1 : INF, d4 + 1), d5 = d6 + 1, d3 = d5 + 1, d2 = std::min(d5 + 1, dl2x[t][u]), d1 = std::min(d2, d3) + 1;
                auto cut = [&](int d) { return d > pcK ? 0.f : 1.f; };   // pcK: beyond K settle steps no credit arrives
                float g4[R]; for (int q = 0; q < R; ++q) g4[q] = gl4[t][u][q] * live(k.l4[q]) / (1 << SH[5]) * cut(d4);
                for (int q = 0; q < R; ++q) if (g4[q] != 0.f) for (int j = 0; j < F; ++j) gl6o[j] += b.wt(Dims::NIB_L4 + q * F + j) * g4[q];
                float g6p[F]; for (int j = 0; j < F; ++j) g6p[j] = gl6o[j] * live(k.l6[j]) / (1 << SH[4]) * cut(d6);
                float g5[E] = {0}, gr6[R6] = {0};
                for (int j = 0; j < F; ++j) if (g6p[j] != 0.f) { for (int i = 0; i < E; ++i) g5[i] += b.wt(Dims::NIB_L6 + j * (E + R6) + i) * g6p[j]; for (int i = 0; i < R6; ++i) gr6[i] += b.wt(Dims::NIB_L6 + j * (E + R6) + E + i) * g6p[j]; }
                for (int i = 0; i < E; ++i) g5[i] *= live(k.l5[i]) / (1 << SH[3]) * cut(d5);
                float g2[B] = {0}, g3[C] = {0};
                for (int j = 0; j < E; ++j) if (g5[j] != 0.f) for (int i = 0; i < B + C; ++i) { float gx = b.wt(Dims::NIB_L5 + j * (B + C) + i) * g5[j]; if (i < B) g2[i] += gx; else g3[i - B] += gx; }
                for (int i = 0; i < B; ++i) g2[i] += gl2x[t][u][i];
                for (int i = 0; i < C; ++i) g3[i] *= live(k.l3[i]) / (1 << SH[2]) * cut(d3);
                float g1[A] = {0}, gmsg[B] = {0};
                for (int j = 0; j < C; ++j) if (g3[j] != 0.f) { for (int i = 0; i < A; ++i) g1[i] += b.wt(Dims::NIB_L3 + j * (A + B) + i) * g3[j]; for (int i = 0; i < B; ++i) gmsg[i] += b.wt(Dims::NIB_L3 + j * (A + B) + A + i) * g3[j]; }
                for (int i = 0; i < B; ++i) g2[i] *= live(k.l2[i]) / (1 << SH[1]) * cut(d2);
                float gr2[R2] = {0};
                for (int j = 0; j < B; ++j) if (g2[j] != 0.f) { for (int i = 0; i < A; ++i) g1[i] += b.wt(Dims::NIB_L2 + j * (A + R2) + i) * g2[j]; for (int i = 0; i < R2; ++i) gr2[i] += b.wt(Dims::NIB_L2 + j * (A + R2) + A + i) * g2[j]; }
                for (int i = 0; i < A; ++i) g1[i] *= live(k.l1[i]) / (1 << SH[0]) * cut(d1);
                // route the error: message half -> senders' L2 at t-1 (unless oracle / undelivered); returns -> own L4 at t-1
                if (t > 0 && (!oracle_mode || g_oracle_inputs_only)) { const auto& s = n.senders[u]; const int w = s.empty() ? 0 : B / (int)s.size();
                    for (size_t q = 0; q < s.size(); ++q) { if (gated && !deliver[(size_t)u * 4 + q]) continue; if (oracle_mode && s[q] < g_ninputs) continue;
                        dl2x[t - 1][s[q]] = std::min(dl2x[t - 1][s[q]], d3 + 1);
                        if (std::getenv("COOP_MIX") && std::string(std::getenv("COOP_MIX")) == "sum") { for (int i = 0; i < B; ++i) gl2x[t - 1][s[q]][i] += gmsg[i]; }
                        else for (int i = 0; i < w; ++i) gl2x[t - 1][s[q]][i] += gmsg[q * w + i]; } }
                if (t > 0) { for (int i = 0; i < R2; ++i) gl4[t - 1][u][i] += gr2[i]; for (int i = 0; i < R6; ++i) gl4[t - 1][u][R2 + i] += gr6[i]; dl4[t - 1][u] = std::min(dl4[t - 1][u], std::min(d2, d6) + 1); }
                SiteG& sg = G[t][u]; std::memcpy(sg.g4, g4, sizeof g4); std::memcpy(sg.g6, g6p, sizeof g6p); std::memcpy(sg.g5, g5, sizeof g5);
                std::memcpy(sg.g2, g2, sizeof g2); std::memcpy(sg.g3, g3, sizeof g3); std::memcpy(sg.g1, g1, sizeof g1);
            }
            if (fg) {   // forward gradients: replace every hidden site's exact error by (g.u)u for random tangents u (averaged over fgk)
                std::normal_distribution<float> N01(0.f, 1.f);
                std::vector<std::vector<SiteG>> U(n.ticks, std::vector<SiteG>(n.nu)), H(n.ticks, std::vector<SiteG>(n.nu));
                for (auto& a : H) for (auto& b : a) std::memset(&b, 0, sizeof b);
                for (int kk = 0; kk < fgk; ++kk) {
                    double dot = 0;
                    for (int t = 0; t < n.ticks; ++t) for (int u = 0; u < n.nu; ++u) { SiteG& uu = U[t][u]; const SiteG& g = G[t][u]; const bool outs = u == n.root && t == n.ticks - 1;
                        float* up = (float*)&uu; const float* gp = (const float*)&g; const int nf = sizeof(SiteG) / sizeof(float);
                        for (int i = 0; i < nf; ++i) { const bool is_out6 = outs && i >= R && i < R + F; up[i] = is_out6 ? 0.f : N01(rng); dot += (double)up[i] * gp[i]; } }
                    for (int t = 0; t < n.ticks; ++t) for (int u = 0; u < n.nu; ++u) { float* hp = (float*)&H[t][u]; const float* up = (const float*)&U[t][u]; const int nf = sizeof(SiteG) / sizeof(float);
                        for (int i = 0; i < nf; ++i) hp[i] += (float)(dot * up[i]) / fgk; }
                }
                for (int t = 0; t < n.ticks; ++t) for (int u = 0; u < n.nu; ++u) { SiteG& g = G[t][u]; const bool outs = u == n.root && t == n.ticks - 1;
                    float keep6[F]; std::memcpy(keep6, g.g6, sizeof keep6); g = H[t][u]; if (outs) std::memcpy(g.g6, keep6, sizeof keep6); }   // exact local output error kept
            }
            for (int t = 0; t < n.ticks; ++t) for (int u = 0; u < n.nu; ++u) {
                const Tick& k = T[t][u]; Body& b = n.body[n.type[u]]; const SiteG& g = G[t][u];
                update(b, Dims::NIB_L4, k.l6, F, g.g4, R, qs, shift);
                update(b, Dims::NIB_L6, k.x6, E + R6, g.g6, F, qs, shift);
                update(b, Dims::NIB_L5, k.x5, B + C, g.g5, E, qs, shift);
                update(b, Dims::NIB_L2, k.x2, A + R2, g.g2, B, qs, shift);
                update(b, Dims::NIB_L3, k.x3, A + B, g.g3, C, qs, shift);
                update(b, Dims::NIB_L1, &ex.x[(size_t)u * IN], IN, g.g1, A, qs, shift);
            }
        } else {   // paired node perturbation with a shared reward: dq = (loss+ - loss-) ; dw = -dq * s_j * x_i per traversed layer
            const uint64_t seed = rng(); const int pdelta = 8;
            std::vector<std::vector<Tick>> Tp, Tm; float lp[2], lm[2], pp[2], pm[2];
            forward(n, ex, Tp, lp, op, deliver, +1, seed, pdelta); forward(n, ex, Tm, lm, op, deliver, -1, seed, pdelta);
            const float dq = (xent(lp, ex.y, pp) - xent(lm, ex.y, pm)) / (2.f * pdelta) * 127.f;   // finite-difference directional derivative, per unit of noise
            for (int t = 0; t < n.ticks; ++t) for (int u = 0; u < n.nu; ++u) {
                const Tick& k = T[t][u]; Body& b = n.body[n.type[u]];
                auto layer_upd = [&](int l, int nib0, const int8_t* x, int in, int out) { uint64_t h = seed ^ ((uint64_t)t << 48) ^ ((uint64_t)u << 40) ^ ((uint64_t)l << 32);
                    std::vector<float> g(out); for (int j = 0; j < out; ++j) { h = h * 6364136223846793005ULL + 1442695040888963407ULL; g[j] = dq * ((h >> 63) ? 1.f : -1.f) / 127.f; }
                    update(b, nib0, x, in, g.data(), out, qs, shift); };
                layer_upd(0, Dims::NIB_L1, &ex.x[(size_t)u * IN], IN, A); layer_upd(1, Dims::NIB_L2, k.x2, A + R2, B); layer_upd(2, Dims::NIB_L3, k.x3, A + B, C);
                layer_upd(3, Dims::NIB_L5, k.x5, B + C, E);
                if (u == n.root && t == n.ticks - 1) { float g6p[F]; for (int j = 0; j < F; ++j) g6p[j] = g6[t][u][j] / (1 << SH[4]); update(b, Dims::NIB_L6, k.x6, E + R6, g6p, F, qs, shift); }   // the local supervised L6 delta
                else layer_upd(4, Dims::NIB_L6, k.x6, E + R6, F);
            }
        }
        if (step % 4000 == 0) { const int keep = cur_step; cur_step = 0; std::cout << "  " << step << ":" << accuracy() << "%" << std::flush; cur_step = keep; }
    }
    std::cout << "\n";
    if (!oracle_mode) {   // intervention: replace each input unit's message by the one it sends for the opposite value of its bit
        std::vector<uint8_t> dl; if (gated) { dl.assign((size_t)n.nu * 4, 0); for (size_t i = 0; i < dl.size(); ++i) dl[i] = n.util[i] >= 0.05f; }
        for (int u = 0; u < ninputs; ++u) {
            int flips = 0, n_ok = 0; std::vector<std::vector<Tick>> T, T2; float lg[2], lg2[2];
            for (auto ex : test) {
                forward(n, ex, T, lg, nullptr, dl, 0, 0, 0); const bool ans = lg[1] > lg[0]; n_ok += ans == (ex.y == 1);
                Example ex2 = ex; for (int k = 0; k < 16; ++k) ex2.x[(size_t)u * IN + k] = -ex2.x[(size_t)u * IN + k];   // the same unit, opposite bit
                forward(n, ex2, T2, lg2, nullptr, dl, 0, 0, 0);
                std::vector<int8_t> sw((size_t)n.ticks * B); for (int t = 0; t < n.ticks; ++t) std::memcpy(&sw[(size_t)t * B], T2[t][u].l2, B);
                forward(n, ex, T2, lg2, nullptr, dl, 0, 0, 0, nullptr, u, sw.data());   // original input, swapped message only
                flips += (lg2[1] > lg2[0]) != ans;
            }
            std::cout << "  intervention: unit " << u << "'s message swapped to its opposite-bit message -> root answer flips on " << 100.0 * flips / test.size() << "% (accuracy " << 100.0 * n_ok / test.size() << "%)\n";
        }
        if (task == "par4") {   // hierarchy check at the middle level: does unit 4 carry XOR(bit0, bit1) rather than the bits?
            std::vector<std::vector<Tick>> T, T2, T3; float lg[2], lg2[2]; double same = 0, diff = 0; int keep_ans = 0, flip_ans = 0;
            auto flipbit = [&](Example e, int u) { for (int k = 0; k < 16; ++k) e.x[(size_t)u * IN + k] = -e.x[(size_t)u * IN + k]; return e; };
            auto mdist = [&](const std::vector<std::vector<Tick>>& a, const std::vector<std::vector<Tick>>& b) { double d = 0; for (int t = 0; t < n.ticks; ++t) for (int i = 0; i < B; ++i) d += std::abs(a[t][4].l2[i] - b[t][4].l2[i]); return d / (n.ticks * B); };
            for (auto& ex : test) {
                forward(n, ex, T, lg, nullptr, dl, 0, 0, 0); const bool ans = lg[1] > lg[0];
                Example both = flipbit(flipbit(ex, 0), 1); forward(n, both, T2, lg2, nullptr, dl, 0, 0, 0); same += mdist(T, T2); keep_ans += (lg2[1] > lg2[0]) == ans;   // pair parity unchanged
                Example one = flipbit(ex, 0); forward(n, one, T3, lg2, nullptr, dl, 0, 0, 0); diff += mdist(T, T3);                                                          // pair parity flipped
                std::vector<int8_t> sw((size_t)n.ticks * B); for (int t = 0; t < n.ticks; ++t) std::memcpy(&sw[(size_t)t * B], T3[t][4].l2, B);
                forward(n, ex, T2, lg2, nullptr, dl, 0, 0, 0, nullptr, 4, sw.data()); flip_ans += (lg2[1] > lg2[0]) != ans;   // only unit 4's message swapped
            }
            std::cout << "  hierarchy: unit 4 message change when bits 0 AND 1 flip (pair parity same) " << same / test.size() << " vs bit 0 alone (pair parity flipped) " << diff / test.size()
                      << "; answer kept on double flip " << 100.0 * keep_ans / test.size() << "%; swapping ONLY unit 4's message to the opposite-parity one flips the answer on " << 100.0 * flip_ans / test.size() << "%\n";
        }
        if (gated) { std::cout << "  link usefulness:"; for (int r = 0; r < n.nu; ++r) for (size_t k = 0; k < n.senders[r].size(); ++k) std::cout << " " << n.senders[r][k] << "->" << r << "=" << n.util[(size_t)r * 4 + k]; std::cout << "\n"; }
    }
    return 0;
}
