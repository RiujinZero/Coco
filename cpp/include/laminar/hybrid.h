#pragma once
// Hybrid attention head: a transformer block (causal multi-head attention + MLP, pre-RMS-norm, learned relative
// position bias) placed after the motif bodies, the way hybrid models alternate recurrent/state-space blocks with
// attention blocks. Input per byte: the byte's embedding plus a projection of the bodies' workspace (the recurrent
// half's state); output: a logit CORRECTION added to the organism's own log-probabilities (base), so the block learns
// what the bodies and the sequence memory miss. Trained by backprop over chunks of `chunk` bytes with a key/value
// cache of the previous `ctx` positions (stop-gradient), Adam. Each chunk is scored BEFORE it is learned from, so the
// reported loss is online (prequential). About 0.93M parameters at d=256, 4 heads, MLP 1024.
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace laminar {

struct HybridHead {
    int d = 256, H = 4, dff = 1024, ctx = 128, V = 256, FW = 32; bool use_state = true, use_base = true;
    float lr = 1e-3f, b1 = 0.9f, b2 = 0.99f, eps = 1e-8f, clip = 1.f; uint64_t t = 0;
    // parameters (one vector) and offsets
    std::vector<float> P, G, M1, M2; size_t oEb, oWs, oWq, oWk, oWv, oWo, oW1, oB1, oW2, oB2, oWout, oBout, oRb, NP;
    // key/value cache from previous chunks (stop-gradient): [ncache][d]
    std::vector<float> Kc, Vc; int ncache = 0;
    void init(uint64_t seed) {
        size_t o = 0; auto take = [&](size_t n) { size_t r = o; o += n; return r; };
        oEb = take((size_t)V * d); oWs = take((size_t)d * FW); oWq = take((size_t)d * d); oWk = take((size_t)d * d); oWv = take((size_t)d * d); oWo = take((size_t)d * d);
        oW1 = take((size_t)dff * d); oB1 = take(dff); oW2 = take((size_t)d * dff); oB2 = take(d); oWout = take((size_t)V * d); oBout = take(V); oRb = take((size_t)H * (ctx + 256)); NP = o;
        P.assign(NP, 0.f); G.assign(NP, 0.f); M1.assign(NP, 0.f); M2.assign(NP, 0.f);
        std::mt19937_64 g(seed); std::normal_distribution<float> n01(0.f, 1.f);
        auto fill = [&](size_t off, size_t n, float sd) { for (size_t i = 0; i < n; ++i) P[off + i] = sd * n01(g); };
        fill(oEb, (size_t)V * d, 0.1f); fill(oWs, (size_t)d * FW, 0.1f);
        for (size_t w : {oWq, oWk, oWv}) fill(w, (size_t)d * d, 1.f / std::sqrt((float)d));
        fill(oWo, (size_t)d * d, 0.5f / std::sqrt((float)d)); fill(oW1, (size_t)dff * d, 1.f / std::sqrt((float)d)); fill(oW2, (size_t)d * dff, 0.5f / std::sqrt((float)dff));
        // the correction starts at zero: before learning, the hybrid predicts exactly what the organism does
        Kc.assign((size_t)ctx * d, 0.f); Vc.assign((size_t)ctx * d, 0.f); ncache = 0;
    }
    size_t params() const { return NP - (use_state ? 0 : (size_t)d * FW); }

    // out[n][m] = A[n][k] . W[m][k]^T (+ bias)
    static void mm(const float* A, const float* W, const float* bias, float* out, int n, int k, int m) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) { const float* a = A + (size_t)i * k; float* o = out + (size_t)i * m;
            for (int r = 0; r < m; ++r) { const float* w = W + (size_t)r * k; float s = bias ? bias[r] : 0.f; for (int j = 0; j < k; ++j) s += a[j] * w[j]; o[r] = s; } }
    }
    // dA[n][k] += dOut[n][m] . W[m][k];  dW[m][k] += dOut^T . A;  dbias[m] += sum dOut
    static void mm_back(const float* A, const float* W, const float* dOut, float* dA, float* dW, float* dbias, int n, int k, int m) {
        if (dA) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) { float* da = dA + (size_t)i * k; const float* g = dOut + (size_t)i * m; for (int r = 0; r < m; ++r) { const float gr = g[r]; if (gr == 0.f) continue; const float* w = W + (size_t)r * k; for (int j = 0; j < k; ++j) da[j] += gr * w[j]; } }
        }
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < m; ++r) { float* dw = dW + (size_t)r * k; float db = 0.f; for (int i = 0; i < n; ++i) { const float gr = dOut[(size_t)i * m + r]; if (gr == 0.f) continue; db += gr; const float* a = A + (size_t)i * k; for (int j = 0; j < k; ++j) dw[j] += gr * a[j]; } if (dbias) dbias[r] += db; }
    }
    static void rms(const float* x, float* y, float* r, int n, int dd) { for (int i = 0; i < n; ++i) { double s = 0; for (int j = 0; j < dd; ++j) s += (double)x[(size_t)i * dd + j] * x[(size_t)i * dd + j]; r[i] = (float)std::sqrt(s / dd + 1e-6); for (int j = 0; j < dd; ++j) y[(size_t)i * dd + j] = x[(size_t)i * dd + j] / r[i]; } }
    static void rms_back(const float* y, const float* r, const float* dy, float* dx, int n, int dd) { for (int i = 0; i < n; ++i) { double m = 0; for (int j = 0; j < dd; ++j) m += (double)dy[(size_t)i * dd + j] * y[(size_t)i * dd + j]; m /= dd; for (int j = 0; j < dd; ++j) dx[(size_t)i * dd + j] += (dy[(size_t)i * dd + j] - (float)m * y[(size_t)i * dd + j]) / r[i]; } }

    // one chunk: sym[n], state[n][FW] (already squashed to [-1,1]), base[n][V] log-probs (or null), tgt[n]. Returns the
    // summed cross-entropy (nats) of the combined prediction, computed before the update.
    double step(int n, const uint8_t* sym, const float* state, const float* base, const uint8_t* tgt, bool learn = true) {
        const int W = ncache + n, dh = d / H; const float isq = 1.f / std::sqrt((float)dh);
        std::vector<float> x((size_t)n * d), a((size_t)n * d), ra(n), q((size_t)n * d), Kall((size_t)W * d), Vall((size_t)W * d), att((size_t)n * H * W, 0.f), ao((size_t)n * d), o((size_t)n * d), y((size_t)n * d),
                           bb((size_t)n * d), rb(n), h1((size_t)n * dff), h1a((size_t)n * dff), m((size_t)n * d), z((size_t)n * d), c((size_t)n * d), rc(n), lg((size_t)n * V);
        for (int i = 0; i < n; ++i) { float* xi = &x[(size_t)i * d]; const float* e = &P[oEb + (size_t)sym[i] * d]; for (int j = 0; j < d; ++j) xi[j] = e[j];
            if (use_state) for (int j = 0; j < d; ++j) { const float* w = &P[oWs + (size_t)j * FW]; float s = 0.f; for (int f = 0; f < FW; ++f) s += w[f] * state[(size_t)i * FW + f]; xi[j] += s; } }
        rms(x.data(), a.data(), ra.data(), n, d);
        mm(a.data(), &P[oWq], nullptr, q.data(), n, d, d);
        std::memcpy(Kall.data(), Kc.data(), sizeof(float) * (size_t)ncache * d); std::memcpy(Vall.data(), Vc.data(), sizeof(float) * (size_t)ncache * d);
        mm(a.data(), &P[oWk], nullptr, &Kall[(size_t)ncache * d], n, d, d); mm(a.data(), &P[oWv], nullptr, &Vall[(size_t)ncache * d], n, d, d);
        const float* rbias = &P[oRb]; const int RB = ctx + 256;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) { const int gi = ncache + i; const int j0 = std::max(0, gi - ctx + 1);
            for (int h = 0; h < H; ++h) { float* pa = &att[((size_t)i * H + h) * W]; float mx = -1e30f; const float* qi = &q[(size_t)i * d + h * dh];
                for (int j = j0; j <= gi; ++j) { const float* kj = &Kall[(size_t)j * d + h * dh]; float s = 0.f; for (int t2 = 0; t2 < dh; ++t2) s += qi[t2] * kj[t2]; s = s * isq + rbias[h * RB + (gi - j)]; pa[j] = s; mx = std::max(mx, s); }
                float zz = 0.f; for (int j = j0; j <= gi; ++j) { pa[j] = std::exp(pa[j] - mx); zz += pa[j]; } for (int j = j0; j <= gi; ++j) pa[j] /= zz;
                float* aoi = &ao[(size_t)i * d + h * dh]; for (int t2 = 0; t2 < dh; ++t2) aoi[t2] = 0.f;
                for (int j = j0; j <= gi; ++j) { const float pj = pa[j]; const float* vj = &Vall[(size_t)j * d + h * dh]; for (int t2 = 0; t2 < dh; ++t2) aoi[t2] += pj * vj[t2]; } } }
        mm(ao.data(), &P[oWo], nullptr, o.data(), n, d, d);
        for (size_t i = 0; i < y.size(); ++i) y[i] = x[i] + o[i];
        rms(y.data(), bb.data(), rb.data(), n, d);
        mm(bb.data(), &P[oW1], &P[oB1], h1.data(), n, d, dff);
        for (size_t i = 0; i < h1.size(); ++i) h1a[i] = h1[i] > 0.f ? h1[i] : 0.f;
        mm(h1a.data(), &P[oW2], &P[oB2], m.data(), n, dff, d);
        for (size_t i = 0; i < z.size(); ++i) z[i] = y[i] + m[i];
        rms(z.data(), c.data(), rc.data(), n, d);
        mm(c.data(), &P[oWout], &P[oBout], lg.data(), n, d, V);
        // combined prediction and its loss (before learning)
        double loss = 0.0; std::vector<float> dl((size_t)n * V);
        for (int i = 0; i < n; ++i) { float* l = &lg[(size_t)i * V]; if (use_base && base) for (int v = 0; v < V; ++v) l[v] += base[(size_t)i * V + v];
            float mx = l[0]; for (int v = 1; v < V; ++v) mx = std::max(mx, l[v]); double zz = 0; for (int v = 0; v < V; ++v) zz += std::exp((double)(l[v] - mx));
            loss += std::log(zz) + mx - l[tgt[i]];
            for (int v = 0; v < V; ++v) dl[(size_t)i * V + v] = (float)(std::exp((double)(l[v] - mx)) / zz) / n; dl[(size_t)i * V + tgt[i]] -= 1.f / n; }
        // cache the newest ctx keys/values for the next chunk
        { const int keep = std::min(ctx, W); std::memcpy(Kc.data(), &Kall[(size_t)(W - keep) * d], sizeof(float) * (size_t)keep * d); std::memcpy(Vc.data(), &Vall[(size_t)(W - keep) * d], sizeof(float) * (size_t)keep * d); }
        const int prev_cache = ncache; ncache = std::min(ctx, W);
        if (!learn) return loss;
        // ---- backward
        std::fill(G.begin(), G.end(), 0.f);
        std::vector<float> dc((size_t)n * d, 0.f), dz((size_t)n * d, 0.f), dh1((size_t)n * dff, 0.f), dbb((size_t)n * d, 0.f), dy((size_t)n * d, 0.f), dao((size_t)n * d, 0.f), dq((size_t)n * d, 0.f),
                           dK((size_t)W * d, 0.f), dV((size_t)W * d, 0.f), da((size_t)n * d, 0.f), dx((size_t)n * d, 0.f);
        mm_back(c.data(), &P[oWout], dl.data(), dc.data(), &G[oWout], &G[oBout], n, d, V);
        rms_back(c.data(), rc.data(), dc.data(), dz.data(), n, d);
        std::vector<float> dm = dz; mm_back(h1a.data(), &P[oW2], dm.data(), dh1.data(), &G[oW2], &G[oB2], n, dff, d);
        for (size_t i = 0; i < dh1.size(); ++i) if (h1[i] <= 0.f) dh1[i] = 0.f;
        mm_back(bb.data(), &P[oW1], dh1.data(), dbb.data(), &G[oW1], &G[oB1], n, d, dff);
        dy = dz; rms_back(bb.data(), rb.data(), dbb.data(), dy.data(), n, d);
        mm_back(ao.data(), &P[oWo], dy.data(), dao.data(), &G[oWo], nullptr, n, d, d);
        float* grb = &G[oRb];
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < H; ++h) {                          // one thread per head: dK/dV of a head are written only here
            for (int i = 0; i < n; ++i) { const int gi = prev_cache + i; const int j0 = std::max(0, gi - ctx + 1); const float* pa = &att[((size_t)i * H + h) * W]; const float* go = &dao[(size_t)i * d + h * dh];
                const float* qi = &q[(size_t)i * d + h * dh]; float* dqi = &dq[(size_t)i * d + h * dh]; float sdp = 0.f;
                std::vector<float> dp(gi + 1, 0.f);
                for (int j = j0; j <= gi; ++j) { const float* vj = &Vall[(size_t)j * d + h * dh]; float s = 0.f; for (int t2 = 0; t2 < dh; ++t2) s += go[t2] * vj[t2]; dp[j] = s; sdp += pa[j] * s;
                    if (j >= prev_cache) { float* dvj = &dV[(size_t)j * d + h * dh]; for (int t2 = 0; t2 < dh; ++t2) dvj[t2] += pa[j] * go[t2]; } }
                for (int j = j0; j <= gi; ++j) { const float ds = pa[j] * (dp[j] - sdp); grb[h * RB + (gi - j)] += ds; const float* kj = &Kall[(size_t)j * d + h * dh];
                    for (int t2 = 0; t2 < dh; ++t2) dqi[t2] += ds * isq * kj[t2];
                    if (j >= prev_cache) { float* dkj = &dK[(size_t)j * d + h * dh]; for (int t2 = 0; t2 < dh; ++t2) dkj[t2] += ds * isq * qi[t2]; } } } }
        mm_back(a.data(), &P[oWq], dq.data(), da.data(), &G[oWq], nullptr, n, d, d);
        mm_back(a.data(), &P[oWk], &dK[(size_t)prev_cache * d], da.data(), &G[oWk], nullptr, n, d, d);
        mm_back(a.data(), &P[oWv], &dV[(size_t)prev_cache * d], da.data(), &G[oWv], nullptr, n, d, d);
        dx = dy; rms_back(a.data(), ra.data(), da.data(), dx.data(), n, d);
        for (int i = 0; i < n; ++i) { const float* g = &dx[(size_t)i * d]; float* ge = &G[oEb + (size_t)sym[i] * d]; for (int j = 0; j < d; ++j) ge[j] += g[j];
            if (use_state) for (int j = 0; j < d; ++j) { float* gw = &G[oWs + (size_t)j * FW]; for (int f = 0; f < FW; ++f) gw[f] += g[j] * state[(size_t)i * FW + f]; } }
        // ---- Adam with global-norm clipping
        double n2 = 0.0; for (float g : G) n2 += (double)g * g; const float sc = n2 > (double)clip * clip ? clip / (float)std::sqrt(n2) : 1.f;
        ++t; const float c1 = 1.f - std::pow(b1, (float)t), c2 = 1.f - std::pow(b2, (float)t);
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)NP; ++i) { const float g = G[i] * sc; M1[i] = b1 * M1[i] + (1.f - b1) * g; M2[i] = b2 * M2[i] + (1.f - b2) * g * g; P[i] -= lr * (M1[i] / c1) / (std::sqrt(M2[i] / c2) + eps); }
        return loss;
    }
};

} // namespace laminar
