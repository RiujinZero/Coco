// Controlled temporal harness: can Coco bodies carry CONTENT across time through gated registers, given EXACT credit?
// (docs/BODY_CIRCUIT_RESEARCH_LOG.md #23-24.) No sequence memory, routing, blending or locks: the confounds are removed.
//
// Language: "the NOUN FILLER^d VERB .\n" (the agree_probe language). At every byte the body of the current byte (one INT4 body
// per byte value, or ONE shared body with MODE=shared) reads: 16 features each of bytes t, t-1, t-2 (+-32) and the 4 registers
// (cells 124-127, reg * 32/64). Its L6 feeds a float readout (next byte) and a shared float WRITE HEAD:
//   reg <- (1 - g) reg + g v,  g = sigmoid(wg . l6/128 + bg),  v = 100 tanh(wv . l6/128)
// Training: next-byte cross-entropy at every byte; BPTT over the whole sentence (or truncated: BPTT=k steps through the
// registers); INT4 bodies learn through the organism's quantised accumulator rule, head and readout by float SGD.
// Scoring (frozen): minimal pairs, correct vs incorrect verb, by distance.
//
// usage: seq_probe SENTENCES [key=value...]   keys: regs=0|1  bptt=K (0 = full)  shared=0|1  head_lr  out_lr  bias  seed
#include "laminar/config.h"
#include "laminar/int4.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace laminar;
namespace {
constexpr int IN = Dims::IN, A = Dims::A, B = Dims::B, C = Dims::C, E = Dims::E, F = Dims::F, R2 = Dims::R2, R6 = Dims::R6, NR = 4;
int SH[5] = {1, 2, 2, 3, 3};
bool g_ste = false; float g_leak = 0.f; float g_rscale = 50.f; int g_fscale = 32;
constexpr int QUANTUM = 64 << 6, HALF = QUANTUM / 2;
std::mt19937_64 rng(1);
long g_flips = 0; int g_batch = 1, g_bbatch = 0;   /* bbatch: per-BODY minibatch, an Adam step after every bbatch uses of that body */   // accumulator weight flips (diag); opt=1 minibatch (sentences per Adam step)
float uni() { return (float)((rng() >> 40) * (1.0 / 16777216.0)); }
struct Body {
    std::vector<uint8_t> w; std::vector<int16_t> acc; std::vector<float> sh, am, av, gb; long t = 0, uses = 0; bool dirty = false;   // sh/am/av: opt=adam shadow + moments
    Body() : w(Dims::UNIT_BYTES), acc(Dims::UNIT_NIBBLES, 0) { for (int n = 0; n < Dims::UNIT_NIBBLES; ++n) nib_set(w.data(), n, (int)(rng() % 5) - 2); }
    int wt(int nib) const { return nib_get(w.data(), nib); }
    void layer(int nib0, int sh, const int8_t* x, int in, int8_t* y, int out) const { for (int j = 0; j < out; ++j) { int32_t s = 0; for (int i = 0; i < in; ++i) s += x[i] * wt(nib0 + j * in + i); y[j] = clamp8(s >> sh); } }
    void add(int nib, int32_t delta) { if (!delta) return; int32_t v = acc[nib] + delta;
        if (v >= HALF) { int cw = wt(nib); if (cw < 7) { nib_set(w.data(), nib, cw + 1); v -= QUANTUM; ++g_flips; } else v = HALF - 1; }
        else if (v <= -HALF) { int cw = wt(nib); if (cw > -8) { nib_set(w.data(), nib, cw - 1); v += QUANTUM; ++g_flips; } else v = -(HALF - 1); }
        acc[nib] = (int16_t)std::max(-32767, std::min(32767, v)); }
};
struct Step { uint8_t cur, next; int body; int body2 = -1; int8_t x[IN], x2[A + R2], x3[A + B], x5[B + C], x6[E + R6], l1[A], l2[B], l3[C], l5[E], l6[F];
    int8_t y2[A + R2], y3[A + B], y5[B + C], y6[E + R6], m1[A], m2[B], m3[C], m5[E], m6[F]; float ws[F];   /* dual: the shared per-byte body's activations; ws = sum of L6 */
    float reg_in[NR], g[NR], v[NR], zv[NR], p[256]; };
void feats(uint8_t c, int8_t* out) {
    for (int b = 0; b < 8; ++b) out[b] = (int8_t)(((c >> b) & 1) ? g_fscale : -g_fscale);
    const bool al = std::isalpha(c), up = std::isupper(c), sp = c == ' ', nl = c == '\n', pu = c == '.';
    const bool f[8] = {al, up, sp, nl, pu, al && std::strchr("aeiou", std::tolower(c)) != nullptr, c == 's', false};
    for (int k = 0; k < 8; ++k) out[8 + k] = (int8_t)(f[k] ? g_fscale : -g_fscale);
}
int g_opt = 0; float g_alr = 0.02f, g_leak_unused = 0.f;   // g_opt 1: exact float shadow per synapse + Adam (INT4 weight = round(shadow))
void adam_step(Body& b, float n_) {   // one Adam step of a body's shadow from its accumulated gradient
    b.dirty = false; ++b.t; const float c1 = 1.f - std::pow(0.9f, (float)b.t), c2 = 1.f - std::pow(0.999f, (float)b.t);
    for (int n = 0; n < Dims::UNIT_NIBBLES; ++n) { const float gr = b.gb[n] / n_; b.gb[n] = 0.f;
        b.am[n] = 0.9f * b.am[n] + 0.1f * gr; b.av[n] = 0.999f * b.av[n] + 0.001f * gr * gr;
        b.sh[n] -= g_alr * (b.am[n] / c1) / (std::sqrt(b.av[n] / c2) + 1e-8f); b.sh[n] = std::max(-8.49f, std::min(7.49f, b.sh[n]));
        nib_set(b.w.data(), n, (int)std::lround(b.sh[n])); }
}
void update(Body& b, int nib0, const int8_t* x, int in, const float* g, int out, float qs, int shift) {
    if (g_opt == 1) {   // exact gradient g_j * x_i on the shadow, Adam, then re-round the durable INT4 weight
        if (b.sh.empty()) { b.sh.resize(Dims::UNIT_NIBBLES); b.am.assign(Dims::UNIT_NIBBLES, 0.f); b.av.assign(Dims::UNIT_NIBBLES, 0.f); for (int n = 0; n < Dims::UNIT_NIBBLES; ++n) b.sh[n] = (float)b.wt(n); }
        if (g_batch > 1 || g_bbatch > 0) { if (b.gb.empty()) b.gb.assign(Dims::UNIT_NIBBLES, 0.f); b.dirty = true;
            for (int j = 0; j < out; ++j) if (g[j] != 0.f) for (int i = 0; i < in; ++i) b.gb[nib0 + j * in + i] += g[j] * (float)x[i];
            return; }
        const float c1 = 1.f - std::pow(0.9f, (float)std::max(1L, b.t)), c2 = 1.f - std::pow(0.999f, (float)std::max(1L, b.t));   // Adam bias correction
        for (int j = 0; j < out; ++j) { if (g[j] == 0.f) continue;
            for (int i = 0; i < in; ++i) { const int n = nib0 + j * in + i; const float gr = g[j] * (float)x[i];
                b.am[n] = 0.9f * b.am[n] + 0.1f * gr; b.av[n] = 0.999f * b.av[n] + 0.001f * gr * gr;
                b.sh[n] -= g_alr * (b.am[n] / c1) / (std::sqrt(b.av[n] / c2) + 1e-8f); b.sh[n] = std::max(-8.49f, std::min(7.49f, b.sh[n]));
                nib_set(b.w.data(), n, (int)std::lround(b.sh[n])); } }
        return;
    }
    for (int j = 0; j < out; ++j) { float v = std::max(-127.f, std::min(127.f, g[j] * qs)); float fl = std::floor(v); int32_t q = (int32_t)fl + (uni() < v - fl ? 1 : 0); if (!q) continue;
        for (int i = 0; i < in; ++i) { int32_t d = -(int32_t)x[i] * q; int32_t f2 = d >> shift, frac = d - (f2 << shift); d = f2 + ((int32_t)(rng() & ((1u << shift) - 1)) < frac ? 1 : 0); b.add(nib0 + j * in + i, d); } }
}
const char* NOUN[][2] = {{"dog", "dogs"}, {"cat", "cats"}, {"bird", "birds"}, {"child", "children"}, {"man", "men"}, {"girl", "girls"}, {"boy", "boys"}, {"horse", "horses"}, {"teacher", "teachers"}, {"farmer", "farmers"}, {"mouse", "mice"}, {"friend", "friends"}};
const char* ISARE[][2] = {{"is", "are"}};
bool g_isare = false;
const char* VERB[][2] = {{"runs", "run"}, {"sings", "sing"}, {"walks", "walk"}, {"sleeps", "sleep"}, {"jumps", "jump"}, {"waits", "wait"}, {"laughs", "laugh"}, {"swims", "swim"}, {"reads", "read"}, {"eats", "eat"}};
const char* FILL[] = {"very", "old", "small", "red", "happy", "quiet", "near", "the", "big", "green", "tired", "brave", "from", "in", "town", "garden", "house", "river", "with", "blue", "tall", "kind", "long", "cold", "bright"};
struct AV { std::vector<float>* p = nullptr; std::vector<float> g, m, v; long t = 0;   // minibatch Adam for a float parameter vector
    void init(std::vector<float>& x) { p = &x; g.assign(x.size(), 0.f); m = g; v = g; }
    void step(float lr) { ++t; const float c1 = 1.f - std::pow(0.9f, (float)t), c2 = 1.f - std::pow(0.999f, (float)t);
        for (size_t i = 0; i < p->size(); ++i) { const float gr = g[i]; g[i] = 0.f; m[i] = 0.9f * m[i] + 0.1f * gr; v[i] = 0.999f * v[i] + 0.001f * gr * gr; (*p)[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + 1e-8f); } } };
std::string sentence(std::mt19937_64& g, int d, int& num, int& verb) { const int n = g() % 12; num = g() % 2; verb = g() % 10; std::string s = "the "; s += NOUN[n][num]; for (int k = 0; k < d; ++k) { s += " "; s += FILL[g() % 25]; } return s; }
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> kv; for (int i = 2; i < argc; ++i) { std::string a = argv[i]; auto e = a.find('='); if (e != std::string::npos) kv[a.substr(0, e)] = a.substr(e + 1); }
    auto num_arg = [&](const char* k, double d) { return kv.count(k) ? std::stod(kv[k]) : d; };
    const long nsent = argc > 1 ? std::atol(argv[1]) : 50000; const bool regs = num_arg("regs", 1) != 0, shared = num_arg("shared", 0) != 0;
    const int bptt = (int)num_arg("bptt", 0); const float head_lr = (float)num_arg("head_lr", 0.02), out_lr = (float)num_arg("out_lr", 0.01), bias = (float)num_arg("bias", -2);
    const float err_gain = (float)num_arg("err_gain", 2); const int shift = (int)num_arg("lr_shift", 11) - 6; const bool curric = num_arg("curric", 0) != 0; const int win = (int)num_arg("win", 3); const bool direct = num_arg("direct", 0) != 0; g_isare = num_arg("isare", 0) != 0;   // isare: the verb is is/are; decision = first letter after the last space   // control: float readout straight from the input features   // bytes t..t-(win-1) as features (win <= 7)
    rng.seed((uint64_t)num_arg("seed", 1));
    long sat = 0, tot_o = 0;
    // bisection toward the organism (2026-09-25): ctx = the body is chosen by a hash of the last ctx bytes (the organism picks one body per byte
    // by a context key over a tree of ~20k); inorg = the organism's input (5 bytes + word/sentence position thermometers + the leaky persistent
    // workspace at 96..123, registers at 124..127); stream = registers / workspace / byte history carry across sentences (one continuous stream)
    const int ctxn = (int)num_arg("ctx", 1), nbod = (int)num_arg("nb", 4096); const bool inorg = num_arg("inorg", 0) != 0, stream = num_arg("stream", 0) != 0, nows = num_arg("nows", 0) != 0;   /* nows: inorg without the persistent-workspace cells */
    const bool dual = num_arg("dual", 0) != 0;   // dual: a shared per-byte body runs next to the context body every step (both into ws / registers / head)
    const int nprim = shared ? 1 : (ctxn > 1 ? nbod : 256);
    std::vector<Body> bodies(nprim + (dual ? 256 : 0));
    std::vector<float> Wx(256 * IN, 0.f), Wr(256 * NR, 0.f); const bool rdirect = num_arg("rdirect", 0) != 0; /* rdirect: the readout also reads the registers directly */ std::vector<float> Wo(256 * F), bo(256, 0.f), wg(NR * F), wv(NR * F), bg(NR, bias);
    for (auto& w : Wo) w = (uni() * 2 - 1) * 0.05f; for (auto& w : wg) w = (uni() * 2 - 1) * 0.1f; for (auto& w : wv) w = (uni() * 2 - 1) * 0.1f;
    const bool ropt = num_arg("ropt", 0) != 0; const float rlr = (float)num_arg("rlr", 0.002);   // ropt: readout + head by minibatch Adam (as the Python reference)
    AV aWo, abo, awg, awv, abg, aWx, aWr; aWo.init(Wo); abo.init(bo); awg.init(wg); awv.init(wv); abg.init(bg); aWx.init(Wx); aWr.init(Wr);
    // mtp=K: multi-token prediction - extra readout heads predict bytes t+2..t+K from the same ws (and registers, with rdirect);
    // training only (evaluation uses the next-byte head); head k's loss weighted mtp_w
    const int mtp = std::max(1, (int)num_arg("mtp", 1)); const float mtp_w = (float)num_arg("mtp_w", 1.0);
    std::vector<std::vector<float>> mWo(mtp), mbo(mtp), mWr(mtp); std::vector<AV> amWo(mtp), ambo(mtp), amWr(mtp);
    for (int h = 1; h < mtp; ++h) { mWo[h].resize(256 * F); for (auto& w : mWo[h]) w = (uni() * 2 - 1) * 0.05f; mbo[h].assign(256, 0.f); mWr[h].assign(256 * NR, 0.f);
        amWo[h].init(mWo[h]); ambo[h].init(mbo[h]); amWr[h].init(mWr[h]); }
    auto sgd = [&](AV& a, size_t i, float grad, float lr) { if (ropt) a.g[i] += grad; else (*a.p)[i] -= lr * grad; };
    auto live = [](int8_t y) { return (!g_ste && (y >= 127 || y <= -127)) ? g_leak : 1.f; };
    g_opt = (int)num_arg("opt", 0); g_alr = (float)num_arg("alr", 0.02); g_batch = (int)num_arg("batch", 1); g_bbatch = (int)num_arg("bbatch", 0);
    if (g_opt == 1) for (auto& b : bodies) { b.sh.resize(Dims::UNIT_NIBBLES); b.am.assign(Dims::UNIT_NIBBLES, 0.f); b.av.assign(Dims::UNIT_NIBBLES, 0.f); for (int n = 0; n < Dims::UNIT_NIBBLES; ++n) b.sh[n] = (float)b.wt(n); }
    g_ste = num_arg("ste", 0) != 0; g_leak = (float)num_arg("leak", 0); g_rscale = (float)num_arg("rscale", 50); g_fscale = (int)num_arg("fscale", 32); if (kv.count("sh1")) SH[0] = std::stoi(kv["sh1"]); if (kv.count("shh")) for (int i = 1; i < 5; ++i) SH[i] = std::stoi(kv["shh"]);
    // forward one sentence (string incl. the leading byte of context); returns steps
    float carry_reg[NR] = {0, 0, 0, 0}, carry_wsm[F] = {0}; std::string carry_hist(16, '\n'); float ws_scale = 0.f;
    auto orgfeats = [&](uint8_t c, int8_t* out) {   // the organism's byte_features (in_struct), at +-fscale
        const int8_t S2 = (int8_t)g_fscale; for (int b = 0; b < 8; ++b) out[b] = ((c >> b) & 1) ? S2 : (int8_t)-S2;
        const bool al = std::isalpha(c) != 0, up = std::isupper(c) != 0, dg = std::isdigit(c) != 0, sp = c == ' ' || c == '\t', pu = std::ispunct(c) != 0 && c != '\'' && c != '"';
        const int lo = std::tolower(c); const bool vw = al && (lo == 'a' || lo == 'e' || lo == 'i' || lo == 'o' || lo == 'u');
        const bool f[8] = {al, up, dg, sp, pu, vw, c == '\n' || c == '\r', c == '\'' || c == '"'}; for (int q = 0; q < 8; ++q) out[8 + q] = f[q] ? S2 : (int8_t)-S2; };
    auto run = [&](const std::string& s, std::vector<Step>& S, bool keep = false) {
        S.assign(s.size() - 1, Step{}); float reg[NR] = {0, 0, 0, 0}; float wsm[F] = {0};
        if (stream) { std::memcpy(reg, carry_reg, sizeof reg); std::memcpy(wsm, carry_wsm, sizeof wsm); }
        const std::string hs = (stream ? carry_hist : std::string(16, '\n')) + s; const size_t o = 16;
        auto at = [&](size_t t, int q) { return (uint8_t)hs[o + t - q]; };
        for (size_t t = 0; t + 1 < s.size(); ++t) { Step& k = S[t]; k.cur = (uint8_t)s[t]; k.next = (uint8_t)s[t + 1];
            if (shared) k.body = 0; else if (ctxn > 1) { uint64_t h = 1469598103934665603ULL; for (int q = 0; q < ctxn && q < 16; ++q) h = (h ^ at(t, q)) * 1099511628211ULL; h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32; k.body = (int)(h % (uint64_t)nbod); } else k.body = k.cur;
            const Body& b = bodies[k.body];
            std::memset(k.x, 0, IN);
            if (inorg) {
                for (int q = 0; q < 5; ++q) orgfeats(at(t, q), k.x + 16 * q);
                int pos = 0; while (pos < 15 && std::isalpha(at(t, pos))) ++pos;
                int j = pos; while (j < 15 && !std::isalpha(at(t, j))) ++j; int prev = 0; while (j + prev < 15 && std::isalpha(at(t, j + prev))) ++prev;
                int sent = 0; while (sent < 15) { const uint8_t c = at(t, sent); if (c == '.' || c == '!' || c == '?' || c == '\n') break; ++sent; }
                const int8_t S2 = (int8_t)g_fscale; int8_t* w = k.x + 80;
                for (int q = 0; q < 6; ++q) w[q] = pos > q ? S2 : (int8_t)-S2;
                for (int q = 0; q < 5; ++q) w[6 + q] = prev > q + 1 ? S2 : (int8_t)-S2;
                for (int q = 0; q < 5; ++q) w[11 + q] = sent > (1 << q) ? S2 : (int8_t)-S2;
                float m = 0.f; for (int q = 0; q < F; ++q) m += std::abs(wsm[q]); m /= F;
                if (keep) ws_scale = ws_scale == 0.f ? std::max(m, 1.f) : ws_scale + 0.001f * (m - ws_scale);
                const float gw = (float)g_fscale / std::max(ws_scale, 1.f); if (!nows) for (int q = 0; q < F; ++q) k.x[96 + q] = (int8_t)std::max(-127.f, std::min(127.f, wsm[q] * gw));
            } else for (int q = 0; q < win && q < 7; ++q) feats(t >= (size_t)q ? (uint8_t)s[t - q] : ' ', k.x + 16 * q);
            std::memcpy(k.reg_in, reg, sizeof reg); if (regs) for (int r = 0; r < NR; ++r) k.x[124 + r] = (int8_t)std::max(-127.f, std::min(127.f, reg[r] * g_rscale / 100.f));
            auto fwd = [&](const Body& bb, int8_t* x2, int8_t* x3, int8_t* x5, int8_t* x6, int8_t* l1, int8_t* l2, int8_t* l3, int8_t* l5, int8_t* l6) {
                bb.layer(Dims::NIB_L1, SH[0], k.x, IN, l1, A);
                std::memcpy(x2, l1, A); std::memset(x2 + A, 0, R2); bb.layer(Dims::NIB_L2, SH[1], x2, A + R2, l2, B);
                std::memcpy(x3, l1, A); std::memset(x3 + A, 0, B); bb.layer(Dims::NIB_L3, SH[2], x3, A + B, l3, C);
                std::memcpy(x5, l2, B); std::memcpy(x5 + B, l3, C); bb.layer(Dims::NIB_L5, SH[3], x5, B + C, l5, E);
                std::memcpy(x6, l5, E); std::memset(x6 + E, 0, R6); bb.layer(Dims::NIB_L6, SH[4], x6, E + R6, l6, F); };
            fwd(b, k.x2, k.x3, k.x5, k.x6, k.l1, k.l2, k.l3, k.l5, k.l6);
            for (int q = 0; q < F; ++q) k.ws[q] = (float)k.l6[q];
            if (dual) { k.body2 = nprim + k.cur; fwd(bodies[k.body2], k.y2, k.y3, k.y5, k.y6, k.m1, k.m2, k.m3, k.m5, k.m6); for (int q = 0; q < F; ++q) k.ws[q] += (float)k.m6[q]; }
            if (inorg) for (int q = 0; q < F; ++q) wsm[q] = wsm[q] * 0.5f + k.ws[q];   // ws_decay 128/256
            float lg[256], mx = -1e30f; for (int c = 0; c < 256; ++c) { float z = bo[c]; for (int j = 0; j < F; ++j) z += Wo[c * F + j] * k.ws[j] / 128.f; if (direct) for (int i = 0; i < IN; ++i) z += Wx[(size_t)c * IN + i] * k.x[i] / 32.f; if (rdirect) for (int r = 0; r < NR; ++r) z += Wr[c * NR + r] * reg[r] / 100.f; lg[c] = z; mx = std::max(mx, z); }
            float Z = 0; for (int c = 0; c < 256; ++c) { k.p[c] = std::exp(lg[c] - mx); Z += k.p[c]; } for (int c = 0; c < 256; ++c) k.p[c] /= Z;
            if (regs) for (int r = 0; r < NR; ++r) { float zg = bg[r], zv = 0; for (int j = 0; j < F; ++j) { zg += wg[r * F + j] * k.ws[j] / 128.f; zv += wv[r * F + j] * k.ws[j] / 128.f; }
                k.g[r] = 1.f / (1.f + std::exp(-zg)); k.zv[r] = zv; k.v[r] = 100.f * std::tanh(zv); reg[r] = (1 - k.g[r]) * reg[r] + k.g[r] * k.v[r]; }
        }
        if (stream && keep) { std::memcpy(carry_reg, reg, sizeof reg); std::memcpy(carry_wsm, wsm, sizeof wsm); const std::string h2 = carry_hist + s; carry_hist = h2.substr(h2.size() - 16); }
    };
    auto logprob = [&](const std::string& ctx, const std::string& comp) { std::vector<Step> S; std::string s = "\n" + ctx + comp; run(s, S); double lp = 0;
        for (size_t t = ctx.size(); t < S.size(); ++t) lp += std::log(std::max(S[t].p[S[t].next], 1e-9f)); return lp; };
    std::mt19937_64 dg(7);
    const float qs = err_gain * 127.f;
    for (long it = 0; it < nsent; ++it) {
        const int dmax = curric ? std::min(8, (int)(10.0 * it / (0.8 * nsent))) : 8;
        // NB: built in separate statements - the operands of a chained + are unsequenced and GCC evaluated ISARE[0][num] BEFORE
        // sentence() set num: every run before 2026-09-25 trained on a verb agreeing with the PREVIOUS sentence (no agreement at all)
        int num, verb; std::string s = "\n" + sentence(dg, dg() % (dmax + 1), num, verb); s += " "; s += g_isare ? ISARE[0][num] : VERB[verb][num]; s += " .\n";
        std::vector<Step> S; run(s, S, true); const int T = (int)S.size();
        { static double lsum = 0; static long ln = 0; for (auto& k : S) { lsum -= std::log(std::max(k.p[k.next], 1e-9f)); ++ln; } if ((it + 1) % (nsent / 5) == 0) { std::cout << "train loss " << lsum / ln / std::log(2.0) << " bits  "; lsum = 0; ln = 0; } }
        // backward through time
        float dreg[NR] = {0, 0, 0, 0};                   // d loss / d reg entering step t+1 (i.e. after step t's write)
        std::vector<std::vector<float>> dregs(T + 1, std::vector<float>(NR, 0.f));   // truncation bookkeeping by origin is approximated: bptt caps the chain length
        std::vector<int> age(NR, 0);
        for (int t = T - 1; t >= 0; --t) {
            Step& k = S[t]; Body& b = bodies[k.body];
            float gl6[F] = {0};
            for (int c = 0; c < 256; ++c) { const float e = k.p[c] - (c == k.next); if (std::abs(e) < 1e-6f) continue;
                for (int j = 0; j < F; ++j) { gl6[j] += e * Wo[c * F + j] / 128.f; sgd(aWo, c * F + j, e * k.ws[j] / 128.f, out_lr); } sgd(abo, c, e, out_lr);
                if (direct) for (int i = 0; i < IN; ++i) sgd(aWx, (size_t)c * IN + i, e * k.x[i] / 32.f, out_lr); }
            float dreg_in[NR] = {0, 0, 0, 0};
            if (rdirect) for (int c = 0; c < 256; ++c) { const float e = k.p[c] - (c == k.next); if (std::abs(e) < 1e-6f) continue;
                for (int r = 0; r < NR; ++r) { dreg_in[r] += e * Wr[c * NR + r] / 100.f; sgd(aWr, c * NR + r, e * k.reg_in[r] / 100.f, out_lr); } }
            for (int h = 1; h < mtp; ++h) {   // head h predicts the byte h steps after the next one: target S[t + h].next
                if (t + h >= T) break;
                const int tgt = S[t + h].next; float lg[256], mx = -1e30f;
                for (int c = 0; c < 256; ++c) { float z = mbo[h][c]; for (int j = 0; j < F; ++j) z += mWo[h][c * F + j] * k.ws[j] / 128.f;
                    if (rdirect) for (int r = 0; r < NR; ++r) z += mWr[h][c * NR + r] * k.reg_in[r] / 100.f; lg[c] = z; mx = std::max(mx, z); }
                float Z = 0; for (int c = 0; c < 256; ++c) { lg[c] = std::exp(lg[c] - mx); Z += lg[c]; }
                for (int c = 0; c < 256; ++c) { const float e = mtp_w * (lg[c] / Z - (c == tgt)); if (std::abs(e) < 1e-6f) continue;
                    for (int j = 0; j < F; ++j) { gl6[j] += e * mWo[h][c * F + j] / 128.f; sgd(amWo[h], c * F + j, e * k.ws[j] / 128.f, out_lr); } sgd(ambo[h], c, e, out_lr);
                    if (rdirect) for (int r = 0; r < NR; ++r) { dreg_in[r] += e * mWr[h][c * NR + r] / 100.f; sgd(amWr[h], c * NR + r, e * k.reg_in[r] / 100.f, out_lr); } }
            }
            if (regs) for (int r = 0; r < NR; ++r) {   // this step's write: reg_out = (1-g) reg_in + g v
                if (bptt && ++age[r] > bptt) { dreg[r] = 0.f; age[r] = 0; }
                const float d = dreg[r]; if (d == 0.f) continue;
                const float dg_ = d * (k.v[r] - k.reg_in[r]) * k.g[r] * (1 - k.g[r]), dv = d * k.g[r] * 100.f * (1 - std::tanh(k.zv[r]) * std::tanh(k.zv[r]));
                for (int j = 0; j < F; ++j) { gl6[j] += (wg[r * F + j] * dg_ + wv[r * F + j] * dv) / 128.f; sgd(awg, r * F + j, dg_ * k.ws[j] / 128.f, head_lr); sgd(awv, r * F + j, dv * k.ws[j] / 128.f, head_lr); }
                sgd(abg, r, dg_, head_lr); dreg_in[r] += d * (1 - k.g[r]);
            }
            // body backprop (gl6 = d loss / d ws, shared by every body summed into ws)
            auto bwd = [&](Body& bb, const int8_t* x2, const int8_t* x3, const int8_t* x5, const int8_t* x6, const int8_t* l1, const int8_t* l2, const int8_t* l3, const int8_t* l5, const int8_t* l6) {
                float g6[F], g5[E] = {0}, g2[B] = {0}, g3[C] = {0}, g1[A] = {0};
                for (int j = 0; j < F; ++j) g6[j] = gl6[j] * live(l6[j]) / (1 << SH[4]);
                for (int j = 0; j < F; ++j) if (g6[j] != 0) for (int i = 0; i < E; ++i) g5[i] += bb.wt(Dims::NIB_L6 + j * (E + R6) + i) * g6[j];
                for (int i = 0; i < E; ++i) g5[i] *= live(l5[i]) / (1 << SH[3]);
                for (int j = 0; j < E; ++j) if (g5[j] != 0) for (int i = 0; i < B + C; ++i) { float gx = bb.wt(Dims::NIB_L5 + j * (B + C) + i) * g5[j]; if (i < B) g2[i] += gx; else g3[i - B] += gx; }
                for (int i = 0; i < C; ++i) g3[i] *= live(l3[i]) / (1 << SH[2]);
                for (int j = 0; j < C; ++j) if (g3[j] != 0) for (int i = 0; i < A; ++i) g1[i] += bb.wt(Dims::NIB_L3 + j * (A + B) + i) * g3[j];
                for (int i = 0; i < B; ++i) g2[i] *= live(l2[i]) / (1 << SH[1]);
                for (int j = 0; j < B; ++j) if (g2[j] != 0) for (int i = 0; i < A; ++i) g1[i] += bb.wt(Dims::NIB_L2 + j * (A + R2) + i) * g2[j];
                for (int i = 0; i < A; ++i) g1[i] *= live(l1[i]) / (1 << SH[0]);
                if (regs) for (int r = 0; r < NR; ++r) { float gx = 0; for (int j = 0; j < A; ++j) gx += bb.wt(Dims::NIB_L1 + j * IN + 124 + r) * g1[j];
                    const bool inside = std::abs(k.reg_in[r] * g_rscale / 100.f) < 127.f; dreg_in[r] += inside ? gx * g_rscale / 100.f : 0.f; }   // the read at this step
                if (g_opt == 1 && g_batch == 1 && g_bbatch == 0) ++bb.t;
                update(bb, Dims::NIB_L6, x6, E + R6, g6, F, qs, shift); update(bb, Dims::NIB_L5, x5, B + C, g5, E, qs, shift);
                update(bb, Dims::NIB_L2, x2, A + R2, g2, B, qs, shift); update(bb, Dims::NIB_L3, x3, A + B, g3, C, qs, shift); update(bb, Dims::NIB_L1, k.x, IN, g1, A, qs, shift);
                if (g_opt == 1 && g_bbatch > 0 && ++bb.uses % g_bbatch == 0) adam_step(bb, (float)g_bbatch); };
            bwd(b, k.x2, k.x3, k.x5, k.x6, k.l1, k.l2, k.l3, k.l5, k.l6);
            if (dual && k.body2 >= 0) bwd(bodies[k.body2], k.y2, k.y3, k.y5, k.y6, k.m1, k.m2, k.m3, k.m5, k.m6);
            std::memcpy(dreg, dreg_in, sizeof dreg);   // flows to the previous step's write
        }
        if (ropt && (it + 1) % g_batch == 0) { aWo.step(rlr); abo.step(rlr); if (rdirect) aWr.step(rlr); awg.step(rlr); awv.step(rlr); abg.step(rlr); if (direct) aWx.step(rlr);
            for (int h = 1; h < mtp; ++h) { amWo[h].step(rlr); ambo[h].step(rlr); if (rdirect) amWr[h].step(rlr); } }
        if (g_opt == 1 && g_batch > 1 && g_bbatch == 0 && (it + 1) % g_batch == 0) for (auto& b : bodies) if (b.dirty) adam_step(b, (float)g_batch);
        if ((it + 1) % (nsent / 5) == 0 || it + 1 == nsent) {
            std::mt19937_64 tg(99); int ok[9] = {0}, tot[9] = {0};
            for (int d : {0, 1, 2, 4, 6, 8}) for (int i = 0; i < 200; ++i) { int nm, vb; std::string c = sentence(tg, d, nm, vb) + " ";
                ok[d] += logprob(c, std::string(VERB[vb][nm]) + " .") > logprob(c, std::string(VERB[vb][1 - nm]) + " ."); ++tot[d]; }
            if (std::getenv("DEBUG_ISARE")) { std::mt19937_64 eg(99); for (int i = 0; i < 8; ++i) { int nm, vb; std::string c = sentence(eg, 0, nm, vb) + " ?";
                std::vector<Step> S2; run("\n" + c, S2); const Step& k = S2.back(); std::cout << "[" << c << " num=" << nm << " cur=" << (int)k.cur << " a=" << k.p['a'] << " i=" << k.p['i'] << "] "; } std::cout << "\n"; }
            if (std::getenv("DEBUG_EVAL")) { std::mt19937_64 eg(99); for (int i = 0; i < 6; ++i) { int nm, vb; std::string c = sentence(eg, 0, nm, vb) + " "; std::string stem = std::string(VERB[vb][1]);
                std::vector<Step> S2; run("\n" + c + stem, S2); const Step& k = S2.back();
                std::cout << "[" << c << stem << "| num=" << nm << " p('s')=" << k.p['s'] << " p(' ')=" << k.p[' '] << " lp good " << logprob(c, std::string(VERB[vb][nm]) + " .") << " bad " << logprob(c, std::string(VERB[vb][1 - nm]) + " .") << "] "; } std::cout << "\n"; }
            std::cout << (it + 1) << " sentences:"; for (int d : {0, 1, 2, 4, 6, 8}) std::cout << "  d" << d << " " << 100.0 * ok[d] / tot[d];
            {   // the decision itself: after the verb stem, is the agreeing continuation ('s' singular / ' ' plural) the more probable?
                std::mt19937_64 tg2(99); int ok2[9] = {0}, tot2[9] = {0};
                for (int d : {0, 1, 2, 4, 6, 8}) for (int i = 0; i < 200; ++i) { int nm, vb; std::string c = sentence(tg2, d, nm, vb); c += g_isare ? std::string(" ?") : " " + std::string(VERB[vb][1]) + "?";
                    std::vector<Step> S2; run("\n" + c, S2); const Step& k = S2.back();   // isare: the step at the space; else at the stem's last letter
                    if (g_isare) ok2[d] += nm == 1 ? k.p['a'] > k.p['i'] : k.p['i'] > k.p['a'];
                    else ok2[d] += nm == 1 ? k.p[' '] > k.p['s'] : k.p['s'] > k.p[' ']; ++tot2[d]; }
                std::cout << "   | decision:"; for (int d : {0, 1, 2, 4, 6, 8}) std::cout << " d" << d << " " << 100.0 * ok2[d] / tot2[d];
            }
            if (std::getenv("DIAG")) {   // where is number lost? space body after the noun (d0): per layer saturation + sg/pl separation
                std::mt19937_64 tg3(99); double sat_[6] = {0}, n_[6] = {0}; std::vector<double> mu[2][6]; int cnt[2] = {0, 0};
                const int W_[6] = {IN, A, B, C, E, F};
                for (int L = 0; L < 6; ++L) for (int s2 = 0; s2 < 2; ++s2) mu[s2][L].assign(W_[L], 0.0);
                double da = 0;
                for (int i = 0; i < 400; ++i) { int nm, vb; std::string c = sentence(tg3, 0, nm, vb) + " ?"; std::vector<Step> S2; run("\n" + c, S2); const Step& k = S2.back();
                    const int8_t* ls[6] = {k.x, k.l1, k.l2, k.l3, k.l5, k.l6};
                    for (int L = 0; L < 6; ++L) for (int j = 0; j < W_[L]; ++j) { sat_[L] += std::abs(ls[L][j]) >= 127; n_[L] += 1; mu[nm][L][j] += ls[L][j]; }
                    ++cnt[nm]; da += (nm ? 1 : -1) * (std::log(k.p['a']) - std::log(k.p['i'])); }
                std::cout << "   | diag space-body d0:";
                const char* nmL[6] = {"x", "l1", "l2", "l3", "l5", "l6"};
                for (int L = 0; L < 6; ++L) { double sep = 0; for (int j = 0; j < W_[L]; ++j) sep += std::abs(mu[1][L][j] / cnt[1] - mu[0][L][j] / cnt[0]); std::cout << " " << nmL[L] << " sat " << (int)(100 * sat_[L] / n_[L]) << "% sep " << (int)(sep / W_[L]); }
                double wd = 0; for (int j = 0; j < F; ++j) wd += std::abs(Wo['a' * F + j] - Wo['i' * F + j]);
                std::cout << " |Wo_a-Wo_i| " << wd / F << " logit(a-i) pl-sg " << da / 400;
                // how many distinct weights / accumulator fill in the space body
                const Body& sb = bodies[shared ? 0 : ' ']; int hist[16] = {0}; for (int nn = 0; nn < Dims::UNIT_NIBBLES; ++nn) ++hist[sb.wt(nn) + 8];
                std::cout << " w-hist"; for (int h = 0; h < 16; ++h) std::cout << " " << hist[h];
                std::cout << " flips " << g_flips; g_flips = 0;
            }
            std::cout << std::endl;
        }
    }
    return 0;
}
