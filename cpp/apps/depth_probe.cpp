// Depth ladder (goal item G, docs/UNBOUNDED_CIRCUITS.md): does capability grow with the number of bodies layered per event?
//
// Task: x in 0..15 and a program of up to 8 ops (slots 1..k hold real ops - +1, x3, xor 10, rotate-left - the rest nop); the
// answer is the program applied to x. Everything sits in ONE event's input, so no memory is involved, only serial computation.
// A chain of H+1 INT4 bodies runs per example. Hop 0 is the entry body; hop h+1's body is RECRUITED from hop h+1's pool by hop
// h's role code (the sign bits of its L2): pool=1 is a fixed deep chain, pool>1 routes by what the recruiter computed. Each
// hop reads the program and x at L1; with msg=1 the recruiter's L2 at its L3 (motif_wire's message); with resid=1 the
// workspace so far (the sum of the earlier hops' L6) in cells 96..127. Its L6 adds to the workspace; a float readout reads the
// workspace. Exact backprop through the chain (messages and workspace), float shadows + minibatch Adam per body (the
// organism's adam=1), the readout by Adam.
//
// usage: depth_probe EXAMPLES [key=value ...]
//   hops=H (bodies - 1)  pool=P  msg=1  resid=1  kmax=8  curric=1  batch=32  alr=0.01  rlr=0.003  leak=0.1  seed=1
//   shared=1: every hop >= 1 recruits from ONE pool of P bodies (depth without new bodies: a body can serve any hop; the
//   hop index enters the hash only with hashhop=1)
#include "laminar/config.h"
#include "laminar/int4.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace laminar;
namespace {
constexpr int IN = Dims::IN, A = Dims::A, B = Dims::B, C = Dims::C, E = Dims::E, F = Dims::F, R2 = Dims::R2, R6 = Dims::R6;
constexpr int IN2 = A + R2, IN3 = A + B, IN5 = B + C, IN6 = E + R6, NOPS = 5, SLOTS = 8, NV = 16, RES0 = 96;
int SH[5] = {1, 2, 2, 3, 3};
float g_leak = 0.1f, g_alr = 0.01f;
std::mt19937_64 rng(1);
uint64_t mix64(uint64_t z) { z += 0x9E3779B97F4A7C15ULL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }

struct Body {
    std::vector<uint8_t> w; std::vector<float> sh, m, v, g; long t = 0; float uses = 0.f; bool dirty = false;
    int s[5]; float se[5] = {0, 0, 0, 0, 0}, me[5] = {0, 0, 0, 0, 0};   // homeo: this body's shift per layer, running saturated share / mean |y|
    Body() : w(Dims::UNIT_BYTES), sh(Dims::UNIT_NIBBLES), m(Dims::UNIT_NIBBLES, 0.f), v(Dims::UNIT_NIBBLES, 0.f), g(Dims::UNIT_NIBBLES, 0.f) {
        for (int n = 0; n < Dims::UNIT_NIBBLES; ++n) { const int q = (int)(rng() % 5) - 2; nib_set(w.data(), n, q); sh[n] = (float)q; } for (int L = 0; L < 5; ++L) s[L] = SH[L]; }
    void homeo(int L, const int8_t* y, int out, float bar_sat, float lo) {   // raise the shift when >bar of the outputs clamp, lower it when |y| is small
        int sat = 0, mg = 0; for (int j = 0; j < out; ++j) { const int a = y[j] < 0 ? -y[j] : y[j]; mg += a; sat += a >= 127; }
        se[L] += ((float)sat / out - se[L]) * (1.f / 32.f); me[L] += ((float)mg / out - me[L]) * (1.f / 32.f);
        if (se[L] > bar_sat && s[L] < 15) { ++s[L]; se[L] = 0.f; me[L] *= 0.5f; } else if (me[L] < lo * 0.5f && s[L] > 0) { --s[L]; me[L] *= 2.f; se[L] = 0.f; } }
    int wt(int nib) const { return nib_get(w.data(), nib); }
    void layer(int nib0, int s, const int8_t* x, int in, int8_t* y, int out) const {
        for (int j = 0; j < out; ++j) { int32_t z = 0; for (int i = 0; i < in; ++i) z += x[i] * wt(nib0 + j * in + i); y[j] = clamp8(z >> s); } }
    void grad(int nib0, const int8_t* x, int in, const float* gy, int out) {
        dirty = true; for (int j = 0; j < out; ++j) if (gy[j] != 0.f) for (int i = 0; i < in; ++i) g[nib0 + j * in + i] += gy[j] * (float)x[i]; }
    void step() {   // one Adam step over the uses since the last (gradient averaged per use)
        if (!dirty) return; dirty = false; ++t; const float c1 = 1.f - std::pow(0.9f, (float)t), c2 = 1.f - std::pow(0.999f, (float)t), n_ = std::max(1.f, uses); uses = 0.f;
        for (int n = 0; n < Dims::UNIT_NIBBLES; ++n) { const float gr = g[n] / n_; g[n] = 0.f; m[n] = 0.9f * m[n] + 0.1f * gr; v[n] = 0.999f * v[n] + 0.001f * gr * gr;
            if (v[n] > 0.f) { sh[n] -= g_alr * (m[n] / c1) / (std::sqrt(v[n] / c2) + 1e-8f); sh[n] = std::max(-8.49f, std::min(7.49f, sh[n])); nib_set(w.data(), n, (int)std::lround(sh[n])); } } }
};
struct Hop { int body; int sh[5]; float rs; int8_t x1[IN], x2[IN2], x3[IN3], x5[IN5], x6[IN6], l1[A], l2[B], l3[C], l5[E], l6[F]; };
struct AdamV { std::vector<float> p, g, m, v; long t = 0;
    void init(size_t n, float scale) { p.resize(n); for (auto& x : p) x = scale * ((float)(rng() >> 40) / 16777216.f * 2.f - 1.f); g.assign(n, 0.f); m = g; v = g; }
    void step(float lr, float n_) { ++t; const float c1 = 1.f - std::pow(0.9f, (float)t), c2 = 1.f - std::pow(0.999f, (float)t);
        for (size_t i = 0; i < p.size(); ++i) { const float gr = g[i] / n_; g[i] = 0.f; m[i] = 0.9f * m[i] + 0.1f * gr; v[i] = 0.999f * v[i] + 0.001f * gr * gr; p[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + 1e-8f); } } };

int apply_op(int op, int x) {
    switch (op) { case 1: return (x + 1) & 15; case 2: return (x * 3) & 15; case 3: return x ^ 10; case 4: return ((x << 1) | (x >> 3)) & 15; default: return x; } }
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> kv; for (int i = 2; i < argc; ++i) { std::string a = argv[i]; auto e = a.find('='); if (e != std::string::npos) kv[a.substr(0, e)] = a.substr(e + 1); }
    auto arg = [&](const char* k, double d) { return kv.count(k) ? std::stod(kv[k]) : d; };
    const long N = argc > 1 ? std::atol(argv[1]) : 200000;
    const int H = (int)arg("hops", 2), P = std::max(1, (int)arg("pool", 1)), kmax = std::min(SLOTS, (int)arg("kmax", 8)), batch = (int)arg("batch", 32);
    const bool msg = arg("msg", 1) != 0, resid = arg("resid", 1) != 0, curric = arg("curric", 1) != 0, shared = arg("shared", 0) != 0, hashhop = arg("hashhop", shared ? 0 : 1) != 0;
    const int msgsrc = (int)arg("msgsrc", 2);   // the layer a hop sends to the next one's L3: 2 (motif_wire: L2 reads only the input, so a chain of messages
                                                 // does not compose), 3 / 5 / 6 (they read the incoming message: serial composition)
    const int rmode = (int)arg("rmode", 0);   // what a hop reads as the workspace: 0 the running sum of the earlier hops' L6 (grows with depth
                                                // and clamps), 1 the previous hop's L6 only (a chain), 2 the mean of the earlier hops' L6
    const bool tape = arg("tape", 0) != 0;   // hop h sees only op h (at cells 16..20) and x only at hop 0: the value must flow through the chain
    const bool readlast = arg("readlast", 0) != 0;   // the readout reads only the last hop's L6 (the earlier hops feed the chain, not the answer)
    const float rlr = (float)arg("rlr", 0.003); g_alr = (float)arg("alr", 0.01); g_leak = (float)arg("leak", 0.1); rng.seed((uint64_t)arg("seed", 1));
    const bool homeo = arg("homeo", 0) != 0; const float homeo_sat = (float)arg("homeo_sat", 0.10), homeo_lo = (float)arg("homeo_lo", 24);
    if (kv.count("sh1")) SH[0] = std::stoi(kv["sh1"]); if (kv.count("shh")) for (int i = 1; i < 5; ++i) SH[i] = std::stoi(kv["shh"]);   // layer shifts (L1; the rest)
    std::vector<Body> bodies(1 + (size_t)(shared ? 1 : H) * P);
    auto pool_body = [&](int h, const int8_t* l2prev) {   // hop h >= 1: the body of hop h's pool that the recruiter's role code maps to
        uint64_t bits = 0; for (int j = 0; j < B; ++j) bits |= (uint64_t)(l2prev[j] > 0) << j;
        return 1 + (shared ? 0 : (h - 1) * P) + (P == 1 ? 0 : (int)(mix64(bits ^ (hashhop ? (uint64_t)h << 40 : 0)) % (uint64_t)P)); };
    AdamV Wo, bo; Wo.init((size_t)NV * F, 0.05f); bo.init(NV, 0.f);
    auto live = [](int8_t y) { return (y >= 127 || y <= -127) ? g_leak : 1.f; };
    const bool randdepth = arg("randdepth", 0) != 0;   // train on chains of random length (>= the program's): a depth-agnostic iterated body
    const bool readmean = arg("readmean", 0) != 0;   // the readout reads the MEAN of the hops' L6 (its input does not grow with depth)
    const int EH = shared ? std::max(H, std::min(SLOTS - 1, (int)arg("evalhops", H))) : H;   // shared bodies: also evaluate with more hops than trained (depth beyond training)
    std::vector<Hop> hops(EH + 1); float ws[F], wro[F], prob[NV], rsc = 1.f;   // ws: the chain's workspace (sum of L6); wro: what the readout reads
    // forward: the chain up to hop `upto` (inclusive); returns the prediction's probabilities in prob
    auto forward = [&](int x, const int* ops, int upto) {
        int8_t in[IN]; std::memset(in, 0, IN);
        for (int v = 0; v < NV; ++v) in[v] = v == x ? 32 : -32;
        for (int s = 0; s < SLOTS; ++s) for (int o = 0; o < NOPS; ++o) in[16 + s * NOPS + o] = ops[s] == o ? 32 : -32;
        std::fill(ws, ws + F, 0.f);
        for (int h = 0; h <= upto; ++h) { Hop& k = hops[h]; k.body = h == 0 ? 0 : pool_body(h, hops[h - 1].l2); const Body& b = bodies[k.body];
            std::memcpy(k.x1, in, IN);
            if (tape) { std::memset(k.x1, 0, 16 + SLOTS * NOPS); if (h == 0) std::memcpy(k.x1, in, 16);
                if (h < SLOTS) std::memcpy(k.x1 + 16, in + 16 + h * NOPS, NOPS); }
            k.rs = rmode == 1 ? 0.5f : rmode == 2 ? 0.5f / (float)std::max(1, h) : 0.25f;   // d (input cell) / d (L6 of an earlier hop that feeds it)
            if (resid && h > 0) for (int j = 0; j < F; ++j) k.x1[RES0 + j] = clamp8((int32_t)std::lround((rmode == 1 ? (float)hops[h - 1].l6[j] : ws[j]) * k.rs));
            for (int L = 0; L < 5; ++L) k.sh[L] = b.s[L];
            b.layer(Dims::NIB_L1, k.sh[0], k.x1, IN, k.l1, A);
            std::memcpy(k.x2, k.l1, A); std::memset(k.x2 + A, 0, R2); b.layer(Dims::NIB_L2, k.sh[1], k.x2, IN2, k.l2, B);
            std::memcpy(k.x3, k.l1, A); std::memset(k.x3 + A, 0, B);
            if (msg && h > 0 && msgsrc == 7) { for (int i = 0; i < B; ++i) k.x3[A + i] = clamp8((int32_t)std::lround(ws[i] * 0.5f / (float)h)); }   // the mean of the earlier hops' L6
            else if (msg && h > 0) { const Hop& q = hops[h - 1]; const int8_t* src = msgsrc == 3 ? q.l3 : msgsrc == 5 ? q.l5 : msgsrc == 6 ? q.l6 : q.l2;
                std::memcpy(k.x3 + A, src, std::min(B, msgsrc == 3 ? C : msgsrc == 5 ? E : msgsrc == 6 ? F : B)); }
            b.layer(Dims::NIB_L3, k.sh[2], k.x3, IN3, k.l3, C);
            std::memcpy(k.x5, k.l2, B); std::memcpy(k.x5 + B, k.l3, C); b.layer(Dims::NIB_L5, k.sh[3], k.x5, IN5, k.l5, E);
            std::memcpy(k.x6, k.l5, E); std::memset(k.x6 + E, 0, R6); b.layer(Dims::NIB_L6, k.sh[4], k.x6, IN6, k.l6, F);
            for (int j = 0; j < F; ++j) ws[j] += (float)k.l6[j]; }
        rsc = readmean ? 1.f / (float)(upto + 1) : 1.f; for (int j = 0; j < F; ++j) wro[j] = readlast ? (float)hops[upto].l6[j] : ws[j] * rsc;
        float mx = -1e30f; for (int c = 0; c < NV; ++c) { float z = bo.p[c]; for (int j = 0; j < F; ++j) z += Wo.p[(size_t)c * F + j] * wro[j] / 128.f; prob[c] = z; mx = std::max(mx, z); }
        float Z = 0.f; for (int c = 0; c < NV; ++c) { prob[c] = std::exp(prob[c] - mx); Z += prob[c]; } for (int c = 0; c < NV; ++c) prob[c] /= Z;
        int best = 0; for (int c = 1; c < NV; ++c) if (prob[c] > prob[best]) best = c; return best; };
    auto example = [&](std::mt19937_64& g, int k, int* ops) { const int x = (int)(g() % NV); for (int s = 0; s < SLOTS; ++s) ops[s] = s < k ? 1 + (int)(g() % (NOPS - 1)) : 0; return x; };
    auto answer = [&](int x, const int* ops) { for (int s = 0; s < SLOTS; ++s) x = apply_op(ops[s], x); return x; };
    auto evaluate = [&](int upto, double* acc) { std::mt19937_64 tg(99); int ops[SLOTS];
        for (int k = 1; k <= SLOTS; ++k) { int ok = 0; for (int i = 0; i < 500; ++i) { const int x = example(tg, k, ops); ok += forward(x, ops, upto) == answer(x, ops); } acc[k] = 100.0 * ok / 500; } };
    double lsum = 0; long ln = 0; int nb = 0;
    for (long it = 0; it < N; ++it) {
        const int kc = curric ? std::max(1, std::min(kmax, 1 + (int)((double)kmax * it / (0.6 * N)))) : kmax;
        int ops[SLOTS]; const int kk = 1 + (int)(rng() % kc); const int x = example(rng, kk, ops), y = answer(x, ops);
        const int U = randdepth ? (kk - 1) + (int)(rng() % (uint64_t)(H - (kk - 1) + 1)) : H;   // randdepth: any chain long enough for the program
        forward(x, ops, U); lsum -= std::log(std::max(prob[y], 1e-9f)); ++ln;
        // backward: readout, then the hops in reverse (a hop's L2 also gets its receiver's message gradient; with resid, every
        // later hop's gradient at its workspace cells reaches this hop's L6)
        float gws[F] = {0}, gres[F] = {0}, gmsg[B] = {0}, gprev[F] = {0};   // gres: to every earlier hop (rmode 0 / 2); gprev: to the previous hop only (rmode 1)
        for (int c = 0; c < NV; ++c) { const float e = prob[c] - (c == y ? 1.f : 0.f); bo.g[c] += e;
            for (int j = 0; j < F; ++j) { gws[j] += e * Wo.p[(size_t)c * F + j] / 128.f; Wo.g[(size_t)c * F + j] += e * wro[j] / 128.f; } }
        for (int h = U; h >= 0; --h) { Hop& k = hops[h]; Body& b = bodies[k.body]; b.uses += 1.f;
            float g6[F], g5[E] = {0}, g2[B] = {0}, g3[C] = {0}, g1[A] = {0}, gm[B] = {0}, gin[F]; std::memcpy(gin, gprev, sizeof gin); std::fill(gprev, gprev + F, 0.f);
            const int mdim = std::min(B, msgsrc == 3 ? C : msgsrc == 5 ? E : msgsrc == 6 ? F : B);   // the receiver's message gradient joins the sending layer
            for (int j = 0; j < F; ++j) g6[j] = ((!readlast || h == U ? gws[j] * rsc : 0.f) + gres[j] + gin[j] + (msgsrc == 6 && j < mdim ? gmsg[j] : 0.f)) * live(k.l6[j]) / (float)(1 << k.sh[4]);
            for (int j = 0; j < F; ++j) if (g6[j] != 0.f) for (int i = 0; i < E; ++i) g5[i] += (float)b.wt(Dims::NIB_L6 + j * IN6 + i) * g6[j];
            for (int i = 0; i < E; ++i) g5[i] = (g5[i] + (msgsrc == 5 && i < mdim ? gmsg[i] : 0.f)) * live(k.l5[i]) / (float)(1 << k.sh[3]);
            for (int j = 0; j < E; ++j) if (g5[j] != 0.f) for (int i = 0; i < IN5; ++i) { const float gx = (float)b.wt(Dims::NIB_L5 + j * IN5 + i) * g5[j]; if (i < B) g2[i] += gx; else g3[i - B] += gx; }
            for (int i = 0; i < C; ++i) g3[i] = (g3[i] + (msgsrc == 3 && i < mdim ? gmsg[i] : 0.f)) * live(k.l3[i]) / (float)(1 << k.sh[2]);
            for (int j = 0; j < C; ++j) if (g3[j] != 0.f) for (int i = 0; i < IN3; ++i) { const float gx = (float)b.wt(Dims::NIB_L3 + j * IN3 + i) * g3[j]; if (i < A) g1[i] += gx; else gm[i - A] += gx; }
            for (int i = 0; i < B; ++i) g2[i] = (g2[i] + (msgsrc == 2 ? gmsg[i] : 0.f)) * live(k.l2[i]) / (float)(1 << k.sh[1]);
            for (int j = 0; j < B; ++j) if (g2[j] != 0.f) for (int i = 0; i < A; ++i) g1[i] += (float)b.wt(Dims::NIB_L2 + j * IN2 + i) * g2[j];
            for (int i = 0; i < A; ++i) g1[i] *= live(k.l1[i]) / (float)(1 << k.sh[0]);
            if (resid && h > 0) for (int j = 0; j < F; ++j) { if (k.x1[RES0 + j] >= 127 || k.x1[RES0 + j] <= -127) continue; float gx = 0.f;   // this hop's read of the workspace
                for (int i = 0; i < A; ++i) gx += (float)b.wt(Dims::NIB_L1 + i * IN + RES0 + j) * g1[i]; (rmode == 1 ? gprev : gres)[j] += gx * k.rs; }
            b.grad(Dims::NIB_L6, k.x6, IN6, g6, F); b.grad(Dims::NIB_L5, k.x5, IN5, g5, E); b.grad(Dims::NIB_L2, k.x2, IN2, g2, B);
            b.grad(Dims::NIB_L3, k.x3, IN3, g3, C); b.grad(Dims::NIB_L1, k.x1, IN, g1, A);
            if (msg && msgsrc == 7) { if (h > 0) for (int i = 0; i < B; ++i) if (k.x3[A + i] < 127 && k.x3[A + i] > -127) gres[i] += gm[i] * 0.5f / (float)h; }   // to every earlier hop's L6
            else if (msg) std::memcpy(gmsg, gm, sizeof gmsg); }
        if (homeo) for (int h = 0; h <= U; ++h) { Hop& k = hops[h]; Body& b = bodies[k.body];
            b.homeo(0, k.l1, A, homeo_sat, homeo_lo); b.homeo(1, k.l2, B, homeo_sat, homeo_lo); b.homeo(2, k.l3, C, homeo_sat, homeo_lo); b.homeo(3, k.l5, E, homeo_sat, homeo_lo); b.homeo(4, k.l6, F, homeo_sat, homeo_lo); }
        if (++nb == batch) { nb = 0; for (auto& b : bodies) b.step(); Wo.step(rlr, (float)batch); bo.step(rlr, (float)batch); }
        if ((it + 1) % (N / 8) == 0 || it + 1 == N) {
            double acc[SLOTS + 1]; evaluate(H, acc);
            std::cout << (it + 1) << " examples (k <= " << kc << "): train loss " << lsum / ln / std::log(2.0) << " bits | accuracy by k:";
            for (int k = 1; k <= SLOTS; ++k) std::cout << " " << acc[k]; std::cout << std::endl; lsum = 0; ln = 0; }
    }
    if (P > 1) {   // routing: how many pool bodies each hop actually uses on the test set
        std::mt19937_64 tg(7); std::vector<std::vector<int>> used(H + 1, std::vector<int>(bodies.size(), 0)); int ops[SLOTS];
        for (int i = 0; i < 2000; ++i) { const int x = example(tg, 1 + (int)(tg() % kmax), ops); forward(x, ops, H); for (int h = 0; h <= H; ++h) ++used[h][hops[h].body]; }
        std::cout << "bodies used per hop:"; for (int h = 1; h <= H; ++h) { int n = 0; for (int c : used[h]) n += c > 0; std::cout << " h" << h << " " << n << "/" << P; } std::cout << "\n"; }
    {   // saturation: share of each layer's cells at +-127, per hop, on the test set
        std::mt19937_64 tg(5); int ops[SLOTS]; std::vector<std::array<double, 5>> sat(H + 1, std::array<double, 5>{0, 0, 0, 0, 0}); const int W_[5] = {A, B, C, E, F};
        for (int i = 0; i < 1000; ++i) { const int x = example(tg, 1 + (int)(tg() % kmax), ops); forward(x, ops, H);
            for (int h = 0; h <= H; ++h) { const int8_t* ls[5] = {hops[h].l1, hops[h].l2, hops[h].l3, hops[h].l5, hops[h].l6};
                for (int L = 0; L < 5; ++L) for (int j = 0; j < W_[L]; ++j) sat[h][L] += (ls[L][j] >= 127 || ls[L][j] <= -127) / (1000.0 * W_[L]); } }
        std::cout << "saturated % (l1 l2 l3 l5 l6) per hop:"; for (int h = 0; h <= H; ++h) { std::cout << "  h" << h; for (int L = 0; L < 5; ++L) std::cout << " " << (int)(100 * sat[h][L]); } std::cout << "\n"; }
    std::cout << "truncated chains (only hops 0..d run), accuracy by k:\n";
    for (int d = 0; d <= EH; ++d) { double acc[SLOTS + 1]; evaluate(d, acc); if (d == H + 1) std::cout << "  (beyond the trained depth:)" << std::endl; std::cout << "  d=" << d << ":"; for (int k = 1; k <= SLOTS; ++k) std::cout << " " << acc[k]; std::cout << "\n"; }
    return 0;
}
