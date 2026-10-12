#pragma once
// "White matter" probe: cheap attention BETWEEN MOTIFS across recent bytes. Every body that emits becomes a token:
// its L6 emission projected to d plus a learned embedding of its MOTIF (the region it belongs to). The current byte's
// tokens (plus the byte's embedding) form a query that attends over the tokens of the last `P` bytes (keys/values
// cached, stop-gradient once they are in the past; learned bias by age). The read-out is a logit correction added
// to the organism's own log-probabilities. Backprop, exact and local to this block (the organism is untouched);
// Adam every `every` bytes; each byte is scored before it is learned from (online BPB).
// Modes (same cost, different information):  0 "motif": tokens carry motif identity (the white-matter test)
//   1 "sum": one token per byte, the summed workspace (motif identity erased)   2 "sym": the byte embedding only
//   3 "now": motif tokens of the current byte only (no attention across bytes)
#include <vector>
#include <deque>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>

namespace laminar {

struct WhiteMatter {
    int d = 16, P = 32, V = 256, FW = 32, nmotif = 2048, mode = 0, every = 32; float lr = 3e-3f;
    std::vector<float> Pm, G, M1, M2; size_t oA, oM, oE, oWq, oWk, oWv, oWo, oB, oBb, oRb, NP; uint64_t t = 0; int since = 0;
    struct Tok { std::vector<float> k, v; int byte = 0; };
    std::deque<Tok> win; int byte_i = 0;
    void init(uint64_t seed) {
        size_t o = 0; auto take = [&](size_t n) { size_t r = o; o += n; return r; };
        oA = take((size_t)d * FW); oM = take((size_t)nmotif * d); oE = take((size_t)V * d); oWq = take((size_t)d * d); oWk = take((size_t)d * d); oWv = take((size_t)d * d); oWo = take((size_t)d * d);
        oB = take((size_t)V * d); oBb = take(V); oRb = take(P + 1); NP = o;
        Pm.assign(NP, 0.f); G.assign(NP, 0.f); M1.assign(NP, 0.f); M2.assign(NP, 0.f);
        std::mt19937_64 g(seed); std::normal_distribution<float> n01(0.f, 1.f); auto fill = [&](size_t off, size_t n, float sd) { for (size_t i = 0; i < n; ++i) Pm[off + i] = sd * n01(g); };
        fill(oA, (size_t)d * FW, 0.2f); fill(oM, (size_t)nmotif * d, 0.1f); fill(oE, (size_t)V * d, 0.1f);
        for (size_t w : {oWq, oWk, oWv, oWo}) fill(w, (size_t)d * d, 1.f / std::sqrt((float)d));
    }
    size_t params() const { return NP; }
    static void mv(const float* W, const float* x, float* y, int m, int k) { for (int r = 0; r < m; ++r) { float s = 0.f; const float* w = W + (size_t)r * k; for (int j = 0; j < k; ++j) s += w[j] * x[j]; y[r] = s; } }
    // one byte. emitters: (motif, l6[FW]) pairs; ws: summed workspace; base: organism log-probs. Returns CE (nats), before learning.
    double step(uint8_t sym, const std::vector<std::pair<int, const int8_t*>>& em, const int32_t* ws, const float* base, uint8_t tgt) {
        const float isq = 1.f / std::sqrt((float)d);
        // current tokens
        struct Cur { std::vector<float> in, x; int motif; };
        std::vector<Cur> cur;
        auto mk = [&](const float* in, int motif) { Cur c; c.in.assign(in, in + FW); c.motif = motif; c.x.assign(d, 0.f); mv(&Pm[oA], in, c.x.data(), d, FW); if (motif >= 0) for (int j = 0; j < d; ++j) c.x[j] += Pm[oM + (size_t)motif * d + j]; cur.push_back(std::move(c)); };
        if (mode == 0 || mode == 3) for (auto& [m, l6] : em) { float in[32]; for (int f = 0; f < FW; ++f) in[f] = l6[f] / 128.f; mk(in, m % nmotif); }
        else if (mode == 1) { float in[32]; for (int f = 0; f < FW; ++f) in[f] = std::tanh(ws[f] / 128.f); mk(in, -1); }
        std::vector<float> c(d); for (int j = 0; j < d; ++j) c[j] = Pm[oE + (size_t)sym * d + j]; for (auto& x : cur) for (int j = 0; j < d; ++j) c[j] += x.x[j];
        // current tokens join the window (their keys/values carry gradient this byte)
        std::vector<std::vector<float>> ck(cur.size(), std::vector<float>(d)), cv(cur.size(), std::vector<float>(d));
        for (size_t i = 0; i < cur.size(); ++i) { mv(&Pm[oWk], cur[i].x.data(), ck[i].data(), d, d); mv(&Pm[oWv], cur[i].x.data(), cv[i].data(), d, d); }
        while (!win.empty() && byte_i - win.front().byte >= P) win.pop_front();
        const bool attend = mode != 3;
        std::vector<float> q(d), o(d, 0.f), u(d); mv(&Pm[oWq], c.data(), q.data(), d, d);
        const int nw = attend ? (int)win.size() : 0, ncur = (int)cur.size(), NT = nw + ncur;
        std::vector<float> al(NT); float mx = -1e30f;
        auto keyp = [&](int w) -> const float* { return w < nw ? win[w].k.data() : ck[w - nw].data(); };
        auto valp = [&](int w) -> const float* { return w < nw ? win[w].v.data() : cv[w - nw].data(); };
        auto age = [&](int w) { return w < nw ? byte_i - win[w].byte : 0; };
        for (int w = 0; w < NT; ++w) { const float* k = keyp(w); float s = 0.f; for (int j = 0; j < d; ++j) s += q[j] * k[j]; s = s * isq + Pm[oRb + age(w)]; al[w] = s; mx = std::max(mx, s); }
        float z = 0.f; for (int w = 0; w < NT; ++w) { al[w] = std::exp(al[w] - mx); z += al[w]; } for (int w = 0; w < NT; ++w) { al[w] /= z; const float* v = valp(w); for (int j = 0; j < d; ++j) o[j] += al[w] * v[j]; }
        if (!NT) std::fill(o.begin(), o.end(), 0.f);
        std::vector<float> wo(d); mv(&Pm[oWo], o.data(), wo.data(), d, d); for (int j = 0; j < d; ++j) u[j] = c[j] + wo[j];
        std::vector<float> lg(V); for (int v = 0; v < V; ++v) { float s = Pm[oBb + v]; const float* b = &Pm[oB + (size_t)v * d]; for (int j = 0; j < d; ++j) s += b[j] * u[j]; lg[v] = s + (base ? base[v] : 0.f); }
        float m2 = lg[0]; for (int v = 1; v < V; ++v) m2 = std::max(m2, lg[v]); double zz = 0; for (int v = 0; v < V; ++v) zz += std::exp((double)(lg[v] - m2));
        const double loss = std::log(zz) + m2 - lg[tgt];
        // ---- backward (exact, into G)
        std::vector<float> du(d, 0.f);
        for (int v = 0; v < V; ++v) { const float dl = (float)(std::exp((double)(lg[v] - m2)) / zz) - (v == tgt ? 1.f : 0.f); if (std::fabs(dl) < 1e-5f) continue; G[oBb + v] += dl; float* gb = &G[oB + (size_t)v * d]; const float* b = &Pm[oB + (size_t)v * d]; for (int j = 0; j < d; ++j) { gb[j] += dl * u[j]; du[j] += dl * b[j]; } }
        std::vector<float> dc = du, dq(d, 0.f), dO(d, 0.f);
        for (int r = 0; r < d; ++r) for (int j = 0; j < d; ++j) { G[oWo + (size_t)r * d + j] += du[r] * o[j]; dO[j] += Pm[oWo + (size_t)r * d + j] * du[r]; }
        std::vector<float> dal(NT); float sd = 0.f; for (int w = 0; w < NT; ++w) { const float* v = valp(w); float s = 0.f; for (int j = 0; j < d; ++j) s += dO[j] * v[j]; dal[w] = s; sd += al[w] * s; }
        std::vector<std::vector<float>> dx(ncur, std::vector<float>(d, 0.f));
        for (int w = 0; w < NT; ++w) { const float ds = al[w] * (dal[w] - sd); G[oRb + age(w)] += ds; const float* k = keyp(w); for (int j = 0; j < d; ++j) dq[j] += ds * isq * k[j];
            if (w >= nw) { const int i = w - nw; std::vector<float> dk(d), dv(d); for (int j = 0; j < d; ++j) { dk[j] = ds * isq * q[j]; dv[j] = al[w] * dO[j]; }
                for (int r = 0; r < d; ++r) for (int j = 0; j < d; ++j) { G[oWk + (size_t)r * d + j] += dk[r] * cur[i].x[j]; G[oWv + (size_t)r * d + j] += dv[r] * cur[i].x[j]; dx[i][j] += Pm[oWk + (size_t)r * d + j] * dk[r] + Pm[oWv + (size_t)r * d + j] * dv[r]; } } }
        for (int r = 0; r < d; ++r) for (int j = 0; j < d; ++j) { G[oWq + (size_t)r * d + j] += dq[r] * c[j]; dc[j] += Pm[oWq + (size_t)r * d + j] * dq[r]; }
        for (int j = 0; j < d; ++j) G[oE + (size_t)sym * d + j] += dc[j];
        for (int i = 0; i < ncur; ++i) { for (int j = 0; j < d; ++j) dx[i][j] += dc[j];
            for (int r = 0; r < d; ++r) for (int f = 0; f < FW; ++f) G[oA + (size_t)r * FW + f] += dx[i][r] * cur[i].in[f];
            if (cur[i].motif >= 0) for (int j = 0; j < d; ++j) G[oM + (size_t)cur[i].motif * d + j] += dx[i][j]; }
        // the current tokens become the past (stop-gradient from here on)
        if (attend) for (int i = 0; i < ncur; ++i) { Tok tk; tk.k = ck[i]; tk.v = cv[i]; tk.byte = byte_i; win.push_back(std::move(tk)); }
        ++byte_i;
        if (++since >= every) { since = 0; ++t; const float b1 = 0.9f, b2 = 0.99f, c1 = 1.f - std::pow(b1, (float)t), c2 = 1.f - std::pow(b2, (float)t), inv = 1.f / every;
            for (size_t i = 0; i < NP; ++i) { if (G[i] == 0.f && M1[i] == 0.f) continue; const float g = G[i] * inv; M1[i] = b1 * M1[i] + (1.f - b1) * g; M2[i] = b2 * M2[i] + (1.f - b2) * g * g; Pm[i] -= lr * (M1[i] / c1) / (std::sqrt(M2[i] / c2) + 1e-8f); G[i] = 0.f; } }
        return loss;
    }
};

} // namespace laminar
