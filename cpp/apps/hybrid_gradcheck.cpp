// Finite-difference check of HybridHead's backward pass on a tiny configuration (with a key/value cache in play).
#include "laminar/hybrid.h"
#include <cstdio>
#include <random>
using namespace laminar;
int main() {
    HybridHead h; h.d = 16; h.H = 2; h.dff = 32; h.ctx = 6; h.clip = 1e30f; h.lr = 0.f; h.init(7);
    std::mt19937 g(3); std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : h.P) if (x == 0.f) x = 0.05f * nd(g);          // non-zero output layer so every path carries gradient
    const int n = 5; std::vector<uint8_t> sym(n), tgt(n); std::vector<float> st((size_t)n * 32), base((size_t)n * 256);
    for (int i = 0; i < n; ++i) { sym[i] = g() % 256; tgt[i] = g() % 256; } for (auto& x : st) x = std::tanh(nd(g)); for (auto& x : base) x = 0.3f * nd(g);
    // warm the cache with one chunk (no learning), then check the gradient of the next chunk
    h.step(n, sym.data(), st.data(), base.data(), tgt.data(), false);
    const auto Kc = h.Kc, Vc = h.Vc; const int nc = h.ncache;
    auto loss = [&]() { h.Kc = Kc; h.Vc = Vc; h.ncache = nc; return h.step(n, sym.data(), st.data(), base.data(), tgt.data(), false) / n; };
    h.Kc = Kc; h.Vc = Vc; h.ncache = nc; h.step(n, sym.data(), st.data(), base.data(), tgt.data(), true);   // lr 0: fills G, P unchanged
    const auto G = h.G; double worst = 0;
    struct B { const char* name; size_t off, len; } blocks[] = {{"Eb", h.oEb, (size_t)256 * 16}, {"Ws", h.oWs, 16 * 32}, {"Wq", h.oWq, 256}, {"Wk", h.oWk, 256}, {"Wv", h.oWv, 256}, {"Wo", h.oWo, 256},
        {"W1", h.oW1, 512}, {"B1", h.oB1, 32}, {"W2", h.oW2, 512}, {"Wout", h.oWout, 4096}, {"Rb", h.oRb, (size_t)2 * (6 + 256)}};
    for (auto& b : blocks) { double bw = 0; int checked = 0;
        for (int k = 0; k < 40 && checked < 6; ++k) { size_t i = b.off + (g() % b.len); if (std::fabs(G[i]) < 1e-6 && k < 39) continue;
            const float old = h.P[i]; const float e = 1e-3f; h.P[i] = old + e; double lp = loss(); h.P[i] = old - e; double lm = loss(); h.P[i] = old;
            const double num = (lp - lm) / (2 * e); const double rel = std::fabs(num - G[i]) / std::max(1e-4, std::fabs(num) + std::fabs(G[i])); bw = std::max(bw, rel); ++checked; }
        std::printf("%-5s worst relative error %.2e (%d checked)\n", b.name, bw, checked); worst = std::max(worst, bw); }
    std::printf("gradcheck worst %.2e\n", worst);
}
