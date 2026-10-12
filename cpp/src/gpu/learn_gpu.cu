// CUDA port of the batched body learner (batch_learner.h). The float optimiser state of every body (Adam shadow,
// moments, gradient buffer: 16 B per synapse) and the temporal-credit rings live on the GPU; the host sends each step's
// learning records and receives the INT4 weights of the bodies that took an Adam step. Arithmetic follows RefLearner
// (learn_ref.cpp) expression for expression; results differ from it only by float summation order (atomics) and
// the device's transcendental functions.
#include "laminar/batch_learner.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace laminar {
namespace {

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " #x); } while (0)

constexpr int NIB = Dims::UNIT_NIBBLES, UB = Dims::UNIT_BYTES, F = Dims::F;

struct DSlot { float tg[F]; float reg_prev[4], G[4], hg[4], hv[4], hu[4]; float ha[F]; int8_t x1[Dims::IN], x1e[Dims::IN]; };

__device__ __forceinline__ float nibf(const uint8_t* w, int n) { const int b = w[n >> 1]; const int v = (n & 1) ? (b >> 4) : (b & 15); return (float)((v ^ 8) - 8); }
__device__ __forceinline__ int nibi(const uint8_t* w, int n) { const int b = w[n >> 1]; const int v = (n & 1) ? (b >> 4) : (b & 15); return (v ^ 8) - 8; }
__device__ __forceinline__ float live(int8_t y, float leak) { return (y >= 127 || y <= -127) ? leak : 1.f; }

// ---- structural sync -----------------------------------------------------------------------------------------------------
__global__ void k_scatter(uint8_t* W, const uint32_t* ids, const uint8_t* src, int n) {
    const int k = blockIdx.x; if (k >= n) return; uint8_t* d = W + (size_t)ids[k] * UB; const uint8_t* s = src + (size_t)k * UB;
    for (int i = threadIdx.x; i < UB; i += blockDim.x) d[i] = s[i];
}
__global__ void k_reset(float* m, float* v, float* g, const uint32_t* ids, int n) {
    const int k = blockIdx.x; if (k >= n) return; const size_t b = (size_t)ids[k] * NIB;
    for (int i = threadIdx.x; i < NIB; i += blockDim.x) { m[b + i] = 0.f; v[b + i] = 0.f; g[b + i] = 0.f; }
}
__global__ void k_gather(const uint8_t* W, const AdamDue* due, uint8_t* out, int n) {
    const int k = blockIdx.x; if (k >= n) return; const uint8_t* s = W + (size_t)due[k].unit * UB; uint8_t* d = out + (size_t)k * UB;
    for (int i = threadIdx.x; i < UB; i += blockDim.x) d[i] = s[i];
}

// ---- bp_route: each receiver's message gradient to its sender's L2 (learn_backprop's push), before any sender's pass --------
// A receiver's L3 read its sender's L2, so the sender needs W3[:, A..] g3; g3 depends only on the receiver's own output
// error, so all receivers can run first, in parallel. blockDim = 128; one block per direct job.
__global__ void __launch_bounds__(128) k_msg(const LTrace* jobs, int n, const LStream* streams, const uint8_t* W, float* bpmsg, float leak, int split) {
    const int k = blockIdx.x; if (k >= n) return;
    const LTrace& tr = jobs[k]; if (!tr.sender) return;
    const int tid = threadIdx.x;
    __shared__ float g6[F], g5[Dims::E], g3[Dims::C];
    const uint8_t* w = W + (size_t)tr.unit * UB; const LStream& ls = streams[tr.stream];
    if (tid < F) { const float go = split ? (((tr.flags & LF_NOWS) ? 0.f : ls.dl6[tid]) + ls.dl6m[tr.slot][tid]) * tr.gain : ls.dl6[tid] * tr.gain; g6[tid] = go != 0.f ? go * live(tr.l6[tid], leak) / (float)(1 << tr.sh[LI_6]) : 0.f; }
    __syncthreads();
    if (tid < Dims::E) { float a = 0.f; for (int j = 0; j < F; ++j) if (g6[j] != 0.f) a += nibf(w, Dims::NIB_L6 + j * Dims::IN6 + tid) * g6[j];
        g5[tid] = a * (live(tr.l5[tid], leak) / (float)(1 << tr.sh[LI_5])); }
    __syncthreads();
    if (tid < Dims::C) { float a = 0.f; for (int j = 0; j < Dims::E; ++j) if (g5[j] != 0.f) a += nibf(w, Dims::NIB_L5 + j * Dims::IN5 + Dims::B + tid) * g5[j];
        g3[tid] = a * (live(tr.l3[tid], leak) / (float)(1 << tr.sh[LI_3])); }
    __syncthreads();
    if (tid < Dims::B) { float a = 0.f; for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) a += nibf(w, Dims::NIB_L3 + j * Dims::IN3 + Dims::A + tid) * g3[j];
        if (a != 0.f) atomicAdd(&bpmsg[(size_t)(k + tr.sender) * Dims::B + tid], a); }
}

// ---- motif_stream: the chain's stream gradient, one round per position, last to first (a receiver's share depends on its own
// receivers'). Each stream has one job per position. A round: the job at position pos reads its stream's accumulator R (what the
// later bodies need from its L6, kept in rin for k_backward), and a receiver adds W3[:, A..] g3 x 0.5 / pos (the mean's share,
// zero where its message cell clamped) to R. blockDim = 128; one block per direct job.
__global__ void __launch_bounds__(128) k_stream(const LTrace* jobs, int n, const LStream* streams, const uint8_t* W, float* R, float* rin, int pos, float leak, int split) {
    const int k = blockIdx.x; if (k >= n) return;
    const LTrace& tr = jobs[k]; const bool st = (tr.flags & LF_STRM) != 0; if ((st ? (int)tr.sender : 0) != pos) return;
    const int tid = threadIdx.x; float* Rs = R + (size_t)tr.stream * Dims::B;
    __shared__ float r[Dims::B], g6[F], g5[Dims::E], g3[Dims::C];
    if (tid < Dims::B) { r[tid] = (tr.flags & LF_L6) ? Rs[tid] : 0.f; rin[(size_t)k * Dims::B + tid] = r[tid]; }
    if (!st) return;
    __syncthreads();
    const uint8_t* w = W + (size_t)tr.unit * UB; const LStream& ls = streams[tr.stream];
    if (tid < F) { float go = split ? (((tr.flags & LF_NOWS) ? 0.f : ls.dl6[tid]) + ls.dl6m[tr.slot][tid]) * tr.gain : ls.dl6[tid] * tr.gain; if (tid < Dims::B) go += r[tid]; g6[tid] = go != 0.f ? go * live(tr.l6[tid], leak) / (float)(1 << tr.sh[LI_6]) : 0.f; }
    __syncthreads();
    if (tid < Dims::E) { float a = 0.f; for (int j = 0; j < F; ++j) if (g6[j] != 0.f) a += nibf(w, Dims::NIB_L6 + j * Dims::IN6 + tid) * g6[j];
        g5[tid] = a * (live(tr.l5[tid], leak) / (float)(1 << tr.sh[LI_5])); }
    __syncthreads();
    if (tid < Dims::C) { float a = 0.f; for (int j = 0; j < Dims::E; ++j) if (g5[j] != 0.f) a += nibf(w, Dims::NIB_L5 + j * Dims::IN5 + Dims::B + tid) * g5[j];
        g3[tid] = a * (live(tr.l3[tid], leak) / (float)(1 << tr.sh[LI_3])); }
    __syncthreads();
    if (tid < Dims::B && tr.x3[Dims::A + tid] < 127 && tr.x3[Dims::A + tid] > -127) { float a = 0.f;
        for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) a += nibf(w, Dims::NIB_L3 + j * Dims::IN3 + Dims::A + tid) * g3[j];
        Rs[tid] += a * (0.5f / (float)pos); }
}

// ---- one body's backward pass and gradient (learn_backprop / temporal_backprop) -------------------------------------------
// blockDim = 128. Direct jobs read their stream's readout error (and bpmsg: their receivers' message gradient at their L2);
// temporal jobs their ring slot's accumulated future error.
__global__ void __launch_bounds__(128) k_backward(const LTrace* jobs, int n, const LStream* streams, const DSlot* ring, int Wn,
                                                  const uint8_t* W, float* G, float* gws, int temporal, float leak, const float* bpmsg, const float* rin, int split) {
    const int k = blockIdx.x; if (k >= n) return;
    const LTrace& tr = jobs[k]; const int tid = threadIdx.x;
    __shared__ float g6[F], g5[Dims::E], g2[Dims::B], g3[Dims::C], g1[Dims::A];
    const uint32_t u = tr.unit; const uint8_t* w = W + (size_t)u * UB;
    const LStream& ls = streams[tr.stream];
    const DSlot* slot = temporal ? &ring[(size_t)tr.stream * Wn + tr.slot] : nullptr;
    float go = 0.f;
    if (tid < F) { go = temporal ? slot->tg[tid] * tr.gain : split ? (((tr.flags & LF_NOWS) ? 0.f : ls.dl6[tid]) + ls.dl6m[tr.slot][tid]) * tr.gain : ls.dl6[tid] * tr.gain; if (rin && tid < Dims::B) go += rin[(size_t)k * Dims::B + tid]; }   // + motif_stream's
    if (temporal && !__syncthreads_or(tid < F && slot->tg[tid] != 0.f)) return;   // temporal_backprop: nothing arrived for this event
    if (tid < F) g6[tid] = go != 0.f ? go * live(tr.l6[tid], leak) / (float)(1 << tr.sh[LI_6]) : 0.f;
    __syncthreads();
    if (tid < Dims::E) { float a = 0.f; for (int j = 0; j < F; ++j) if (g6[j] != 0.f) a += nibf(w, Dims::NIB_L6 + j * Dims::IN6 + tid) * g6[j];
        g5[tid] = a * (live(tr.l5[tid], leak) / (float)(1 << tr.sh[LI_5])); }
    __syncthreads();
    if (tid < Dims::IN5) { float a = 0.f; for (int j = 0; j < Dims::E; ++j) if (g5[j] != 0.f) a += nibf(w, Dims::NIB_L5 + j * Dims::IN5 + tid) * g5[j];
        if (tid < Dims::B) g2[tid] = (bpmsg ? a + bpmsg[(size_t)k * Dims::B + tid] : a) * (live(tr.l2[tid], leak) / (float)(1 << tr.sh[LI_2]));
        else g3[tid - Dims::B] = a * (live(tr.l3[tid - Dims::B], leak) / (float)(1 << tr.sh[LI_3])); }
    __syncthreads();
    if (tid < Dims::A) { float a = 0.f;
        for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) a += nibf(w, Dims::NIB_L3 + j * Dims::IN3 + tid) * g3[j];
        for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) a += nibf(w, Dims::NIB_L2 + j * Dims::IN2 + tid) * g2[j];
        g1[tid] = a * (live(tr.l1[tid], leak) / (float)(1 << tr.sh[LI_1])); }
    __syncthreads();
    if (!temporal && !tr.exact && tid < F) { float a = 0.f; for (int j = 0; j < Dims::A; ++j) if (g1[j] != 0.f) a += nibf(w, Dims::NIB_L1 + j * Dims::IN + 96 + tid) * g1[j];
        if (a != 0.f) atomicAdd(&gws[(size_t)tr.stream * F + tid], a); }
    const int8_t* x1 = temporal ? (tr.exact ? slot->x1e : slot->x1) : (tr.exact ? ls.x1e : ls.x1);
    float* Gu = G + (size_t)u * NIB; const float sc = tr.scale;
    auto acc = [&](int nib0, const int8_t* x, int in, const float* g, int out) {
        for (int idx = tid; idx < in * out; idx += 128) { const int j = idx / in, i = idx - j * in; const float gj = g[j];
            if (gj != 0.f && x[i] != 0) atomicAdd(&Gu[nib0 + idx], (gj * sc) * (float)x[i]); }
    };
    acc(Dims::NIB_L6, tr.x6, Dims::IN6, g6, F);
    acc(Dims::NIB_L5, tr.x5, Dims::IN5, g5, Dims::E);
    acc(Dims::NIB_L2, tr.x2, Dims::IN2, g2, Dims::B);
    acc(Dims::NIB_L3, tr.x3, Dims::IN3, g3, Dims::C);
    acc(Dims::NIB_L1, x1, Dims::IN, g1, Dims::A);
}

// ---- Adam step (Organism::adam_step), one block per body, one thread per weight byte (two nibbles) -------------------------
__global__ void k_adam(const AdamDue* due, int n, uint8_t* W, float* SH, float* M, float* V, float* G, float lr, float wd) {
    const int k = blockIdx.x; if (k >= n) return;
    const uint32_t u = due[k].unit; const float t = (float)due[k].t;
    const float c1 = 1.f - powf(0.9f, t), c2 = 1.f - powf(0.999f, t), inv_k = 1.f / fmaxf(1.f, due[k].n_uses);
    uint8_t* w = W + (size_t)u * UB; const size_t b0 = (size_t)u * NIB;
    for (int byte = threadIdx.x; byte < UB; byte += blockDim.x) {
        int nb = w[byte];
        for (int h = 0; h < 2; ++h) {
            const size_t q = b0 + 2 * byte + h;
            const int cw = (((h ? (nb >> 4) : nb) & 15) ^ 8) - 8;
            float sh = SH[q]; if ((int)lroundf(sh) != cw) sh = (float)cw;
            const float gr = G[q] * inv_k; G[q] = 0.f;
            const float m = 0.9f * M[q] + 0.1f * gr, v = 0.999f * V[q] + 0.001f * gr * gr; M[q] = m; V[q] = v;
            if (v != 0.f) {
                sh = fmaxf(-8.49f, fminf(7.49f, sh - lr * ((m / c1) / (sqrtf(v / c2) + 1e-8f) + wd * sh)));
                const int nw = (int)lroundf(sh);
                if (nw != cw) nb = h ? ((nb & 0x0F) | ((nw & 15) << 4)) : ((nb & 0xF0) | (nw & 15));
            }
            SH[q] = sh;
        }
        w[byte] = (uint8_t)nb;
    }
}

// ---- Muon step (Organism::muon_step), one block per body: per layer matrix, Nesterov momentum orthogonalised by 5 Newton-Schulz
// iterations in shared memory (on the short side, rows padded to c + 1 floats: no bank conflicts in X X^T); the weights are the
// rounded shadows, written once after every layer ---------------------------------------------------------------------------------
__global__ void k_muon(const AdamDue* due, int n, uint8_t* W, float* SH, float* M, float* G, float lr, float mu, float wd) {
    const int k = blockIdx.x; if (k >= n) return;
    __shared__ float buf0[16 * 129], buf1[16 * 129], A[576], Bm[576], red[256];   // r x (c + 1) <= 16 x 129 (L1), 24 x 33 (L2 / L6 / L4)
    const uint32_t u = due[k].unit; const float inv_k = 1.f / fmaxf(1.f, due[k].n_uses);
    uint8_t* w = W + (size_t)u * UB; const size_t b0 = (size_t)u * NIB;
    for (int byte = threadIdx.x; byte < UB; byte += blockDim.x) {   // resync: shadows whose weight was written elsewhere
        const int nb = w[byte];
        for (int h = 0; h < 2; ++h) { const size_t q = b0 + 2 * byte + h; const int cw = (((h ? (nb >> 4) : nb) & 15) ^ 8) - 8; if ((int)lroundf(SH[q]) != cw) SH[q] = (float)cw; }
    }
    __syncthreads();
    const int offs[6] = {Dims::NIB_L1, Dims::NIB_L2, Dims::NIB_L3, Dims::NIB_L5, Dims::NIB_L6, Dims::NIB_L4};
    const int ins[6] = {Dims::IN, Dims::IN2, Dims::IN3, Dims::IN5, Dims::IN6, Dims::IN4}, outs[6] = {Dims::A, Dims::B, Dims::C, Dims::E, Dims::F, Dims::R};
    for (int li = 0; li < 6; ++li) {
        const int rows = outs[li], cols = ins[li], off = offs[li], nel = rows * cols; const bool tr = rows > cols; const int r = tr ? cols : rows, c = tr ? rows : cols, cs = c + 1;
        float* X = buf0; float* T = buf1; float part = 0.f; int any = 0;
        for (int i = threadIdx.x; i < nel; i += blockDim.x) {
            const size_t q = b0 + off + i; const float g = G[q] * inv_k; G[q] = 0.f; const float m = mu * M[q] + g; M[q] = m; const float x = g + mu * m;
            const int ii = i / cols, jj = i - ii * cols; X[tr ? jj * cs + ii : ii * cs + jj] = x; part += x * x; any |= x != 0.f;
        }
        red[threadIdx.x] = part;
        if (!__syncthreads_or(any)) continue;
        for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
        const float inv = 1.f / (sqrtf(red[0]) + 1e-7f);
        for (int e = threadIdx.x; e < nel; e += blockDim.x) { const int i = e / c, j = e - i * c; X[i * cs + j] *= inv; }
        __syncthreads();
        for (int it = 0; it < 5; ++it) {
            for (int e = threadIdx.x; e < r * r; e += blockDim.x) { const int i = e / r, kk = e - i * r; float s2 = 0.f; for (int j = 0; j < c; ++j) s2 += X[i * cs + j] * X[kk * cs + j]; A[e] = s2; }
            __syncthreads();
            for (int e = threadIdx.x; e < r * r; e += blockDim.x) { const int i = e / r, kk = e - i * r; float s2 = 0.f; for (int j = 0; j < r; ++j) s2 += A[i * r + j] * A[j * r + kk]; Bm[e] = -4.7750f * A[e] + 2.0315f * s2; }
            __syncthreads();
            for (int e = threadIdx.x; e < nel; e += blockDim.x) { const int i = e / c, j = e - i * c; float s2 = 0.f; for (int kk = 0; kk < r; ++kk) s2 += Bm[i * r + kk] * X[kk * cs + j]; T[i * cs + j] = 3.4445f * X[i * cs + j] + s2; }
            __syncthreads();
            float* t = X; X = T; T = t;
        }
        const float scale = sqrtf(fmaxf(1.f, (float)rows / (float)cols));
        for (int i = threadIdx.x; i < nel; i += blockDim.x) {
            const int ii = i / cols, jj = i - ii * cols; const float o = X[tr ? jj * cs + ii : ii * cs + jj]; const size_t q = b0 + off + i;
            const float sh = SH[q]; SH[q] = fmaxf(-8.49f, fminf(7.49f, sh - lr * (scale * o + wd * sh)));
        }
        __syncthreads();
    }
    for (int byte = threadIdx.x; byte < UB; byte += blockDim.x) {   // the weights: the rounded shadows
        int nb = w[byte];
        for (int h = 0; h < 2; ++h) { const int nw = (int)lroundf(SH[b0 + 2 * byte + h]); nb = h ? ((nb & 0x0F) | ((nw & 15) << 4)) : ((nb & 0xF0) | (nw & 15)); }
        w[byte] = (uint8_t)nb;
    }
}

// ---- ring injections (tiles' credit to earlier events' writers), in order, before the ring's own credit; one block per stream ----
__global__ void k_inject(const LStream* streams, int S, const LInject* inj, const uint32_t* off, DSlot* ring, int Wn) {
    const int s = blockIdx.x; if (s >= S) return; const int j = threadIdx.x;
    for (uint32_t i = off[s]; i < off[s + 1]; ++i) { DSlot& e = ring[(size_t)s * Wn + (streams[s].head + Wn - inj[i].age) % Wn]; e.tg[j] += inj[i].g[j]; }
}

// ---- temporal credit (Organism::temporal_credit), one block of 32 threads per stream --------------------------------------
__global__ void k_ring(const LStream* streams, int S, DSlot* ring, int Wn, const float* gws_all, const float* head, float* hd, float decay, float reg_scale) {
    const int s = blockIdx.x; if (s >= S) return;
    const LStream& ls = streams[s]; if (!ls.active) return;
    const int j = threadIdx.x;
    DSlot* R = ring + (size_t)s * Wn;
    if (ls.h_valid) { DSlot& e = R[ls.h_slot]; e.ha[j] = ls.h_a[j];
        if (j < 4) { e.reg_prev[j] = ls.h_reg_prev[j]; e.G[j] = ls.h_G[j]; e.hg[j] = ls.h_g[j]; e.hv[j] = ls.h_v[j]; e.hu[j] = ls.h_u[j]; } }
    __syncthreads();
    const float* gws = gws_all + (size_t)s * F;
    float greg[4]; for (int r = 0; r < 4; ++r) greg[r] = gws[28 + r] * reg_scale / 100.f + ls.greg_ro[r];
    const float g = (j >= 28 ? 0.f : gws[j]) * ls.gscale;
    const float* hw_g = head; const float* hw_v = head + 4 * F;
    float sd = 1.f, sr[4] = {1.f, 1.f, 1.f, 1.f}, hdg[4] = {0, 0, 0, 0}, hdv[4] = {0, 0, 0, 0}, hbp[4] = {0, 0, 0, 0};
    for (int k = 1; k <= ls.fill; ++k) {
        DSlot& e = R[(ls.head + Wn - k) % Wn];
        float tg = e.tg[j] + sd * g; sd *= decay;
        for (int r = 0; r < 4; ++r) { const float dlt = sr[r] * greg[r]; if (dlt == 0.f) { sr[r] *= 1.f - e.G[r]; continue; }
            const float th = tanhf(e.hu[r]);
            const float dg = dlt * (e.hv[r] - e.reg_prev[r]) * e.hg[r] * (1.f - e.hg[r]), dv = dlt * e.hg[r] * 100.f * (1.f - th * th);
            tg += (hw_g[r * F + j] * dg + hw_v[r * F + j] * dv) / 128.f;
            hdg[r] += dg * e.ha[j]; hdv[r] += dv * e.ha[j]; hbp[r] += dg; sr[r] *= 1.f - e.G[r]; }
        e.tg[j] = tg;
    }
    for (int r = 0; r < 4; ++r) { if (hdg[r] != 0.f) atomicAdd(&hd[r * F + j], hdg[r]); if (hdv[r] != 0.f) atomicAdd(&hd[(4 + r) * F + j], hdv[r]); }
    if (j == 0) for (int r = 0; r < 4; ++r) if (hbp[r] != 0.f) atomicAdd(&hd[8 * F + r], hbp[r]);
}
// temporal_credit's store of the current event into its slot
__global__ void k_store(const LStream* streams, int S, DSlot* ring, int Wn) {
    const int s = blockIdx.x; if (s >= S) return;
    const LStream& ls = streams[s]; if (!ls.active) return;
    DSlot& e = ring[(size_t)s * Wn + ls.head]; const int t = threadIdx.x;
    e.x1[t] = ls.x1[t]; e.x1e[t] = ls.x1e[t];
    if (t < F) e.tg[t] = 0.f;
    if (t < 4) { e.reg_prev[t] = ls.reg_now[t]; e.G[t] = 0.f; }
}

// ---- host side ----------------------------------------------------------------------------------------------------------
template <class T> struct DBuf {
    T* p = nullptr; size_t cap = 0;
    void ensure(size_t n) { if (n <= cap) return; if (p) cudaFree(p); cap = std::max<size_t>(n, cap * 2); CU(cudaMalloc(&p, cap * sizeof(T))); }
    ~DBuf() { if (p) cudaFree(p); }
};
template <class T> struct HBuf {   // pinned
    T* p = nullptr; size_t cap = 0;
    void ensure(size_t n) { if (n <= cap) return; if (p) cudaFreeHost(p); cap = std::max<size_t>(n, cap * 2); CU(cudaMallocHost(&p, cap * sizeof(T))); }
    ~HBuf() { if (p) cudaFreeHost(p); }
};

class GpuLearner final : public Learner {
public:
    GpuLearner(const LearnConfig& c, uint8_t* host_w) : c_(c), hw_(host_w) {
        int dev = 0; CU(cudaGetDeviceCount(&dev)); if (!dev) throw std::runtime_error("no CUDA device");
        cudaDeviceProp pr; CU(cudaGetDeviceProperties(&pr, 0)); dev_name_ = pr.name;
        CU(cudaStreamCreate(&cs_)); CU(cudaEventCreate(&ev0_)); CU(cudaEventCreate(&ev1_));
        const size_t n = (size_t)c.units * NIB;
        W_.ensure((size_t)c.units * UB); SH_.ensure(n); M_.ensure(n); V_.ensure(n); G_.ensure(n);
        CU(cudaMemset(SH_.p, 0, n * sizeof(float))); CU(cudaMemset(M_.p, 0, n * sizeof(float))); CU(cudaMemset(V_.p, 0, n * sizeof(float))); CU(cudaMemset(G_.p, 0, n * sizeof(float)));
        ring_.ensure((size_t)c.streams * c.tc_window); CU(cudaMemset(ring_.p, 0, (size_t)c.streams * c.tc_window * sizeof(DSlot)));
        gws_.ensure((size_t)c.streams * F); head_.ensure(8 * F + 4); hd_.ensure(8 * F + 4); CU(cudaMemset(hd_.p, 0, (8 * F + 4) * sizeof(float)));
        streams_.ensure(c.streams); head_host_.ensure(8 * F + 4); hd_host_.ensure(8 * F + 4); std::memset(hd_host_.p, 0, (8 * F + 4) * sizeof(float)); hd_acc_.assign(8 * F + 4, 0.f);
        launcher_ = std::thread([this] { launcher(); });
    }
    ~GpuLearner() override {
        try { wait(); } catch (...) {}
        { std::lock_guard<std::mutex> l(mu_); stop_ = true; } cv_.notify_all(); launcher_.join();
        if (cs_) cudaStreamDestroy(cs_);
    }
    std::string name() const override { return "cuda:" + dev_name_; }
    double seconds_in_step() const override { return sec_; }
    std::string report() const override { char b[160]; std::snprintf(b, sizeof b, "gpu: %llu steps, device busy %.1f us/step, host wait %.1f us/step", (unsigned long long)nstep_, nstep_ ? 1e3 * gpu_ms_ / nstep_ : 0.0, nstep_ ? 1e6 * wait_s_ / nstep_ : 0.0); return b; }

    void upload_all() override { wait(); CU(cudaMemcpy(W_.p, hw_, (size_t)c_.units * UB, cudaMemcpyHostToDevice)); }
    void upload_units(const uint32_t* ids, uint32_t n) override { wait();
        for (uint32_t i = 0; i < n; ++i) CU(cudaMemcpy(W_.p + (size_t)ids[i] * UB, hw_ + (size_t)ids[i] * UB, UB, cudaMemcpyHostToDevice));
    }
    // The head weights travel with the next step's launch; its gradients come back with every step's (no extra round trips).
    void set_head(const float* hw_g, const float* hw_v, const float* hb) override { wait();
        float* h = head_host_.p; std::memcpy(h, hw_g, 4 * F * sizeof(float)); std::memcpy(h + 4 * F, hw_v, 4 * F * sizeof(float)); std::memcpy(h + 8 * F, hb, 4 * sizeof(float));
        head_dirty_ = true;
    }
    void take_head_grads(float* hd) override { wait();
        for (int i = 0; i < 8 * F + 4; ++i) { hd[i] += hd_acc_[i]; hd_acc_[i] = 0.f; }
    }
    void read_unit_state(uint32_t u, float* sh, float* m, float* v, float* g) override { wait();
        const size_t b = (size_t)u * NIB, n = NIB * sizeof(float);
        if (sh) CU(cudaMemcpy(sh, SH_.p + b, n, cudaMemcpyDeviceToHost)); if (m) CU(cudaMemcpy(m, M_.p + b, n, cudaMemcpyDeviceToHost));
        if (v) CU(cudaMemcpy(v, V_.p + b, n, cudaMemcpyDeviceToHost)); if (g) CU(cudaMemcpy(g, G_.p + b, n, cudaMemcpyDeviceToHost));
    }
    void read_ring_tg(uint32_t s, uint32_t slot, float* tg) override { wait();
        CU(cudaMemcpy(tg, ring_.p + (size_t)s * c_.tc_window + slot, F * sizeof(float), cudaMemcpyDeviceToHost));   // tg is the slot's first member
    }

    void step(const StepIn& in) override { submit(in); wait(); }
    void wait() override {
        if (!pending_) return;
        const auto t0 = std::chrono::steady_clock::now();
        { std::unique_lock<std::mutex> l(mu_); cv_.wait(l, [&] { return !launching_; }); }   // the launcher thread has queued everything
        CU(cudaStreamSynchronize(cs_));
        CU(cudaGetLastError());
        { float ms = 0; CU(cudaEventElapsedTime(&ms, ev0_, ev1_)); gpu_ms_ += ms; ++nstep_; wait_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
        for (size_t i = 0; i < due_ids_.size(); ++i) std::memcpy(hw_ + (size_t)due_ids_[i] * UB, hgather_.p + i * UB, UB);
        for (int i = 0; i < 8 * F + 4; ++i) hd_acc_[i] += hd_host_.p[i];
        pending_ = false;
        sec_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    // Fills the pinned staging buffer here; the CUDA calls (the costly part under WSL) run on the launcher thread.
    void submit(const StepIn& in) override {
        wait();
        const auto t0 = std::chrono::steady_clock::now();
        const int S = (int)c_.streams;
        // one staging buffer up, laid out as: streams | direct | temporal | due | dirty ids | reset ids | dirty weights
        auto al = [](size_t x) { return (x + 255) & ~(size_t)255; };
        Launch& L = next_;
        L.o_str = 0; L.o_dir = L.o_str + al(S * sizeof(LStream)); L.o_tmp = L.o_dir + al(in.ndirect * sizeof(LTrace)); L.o_due = L.o_tmp + al(in.ntemporal * sizeof(LTrace));
        L.o_dirty = L.o_due + al(in.ndue * sizeof(AdamDue)); L.o_reset = L.o_dirty + al(in.ndirty * sizeof(uint32_t)); L.o_wts = L.o_reset + al(in.nreset * sizeof(uint32_t));
        L.o_inj = L.o_wts + al((size_t)in.ndirty * UB); L.o_injoff = L.o_inj + al((size_t)in.ninj * sizeof(LInject));
        L.total = L.o_injoff + al((size_t)(S + 1) * sizeof(uint32_t)); L.ninj = in.ninj;
        L.head = head_dirty_; head_dirty_ = false;
        L.ndirect = in.ndirect; L.ntemporal = in.ntemporal; L.ndue = in.ndue; L.ndirty = in.ndirty; L.nreset = in.nreset;
        L.msg = false; L.maxpos = -1;
        for (uint32_t i = 0; i < in.ndirect; ++i) { const LTrace& t = in.direct[i];
            if (t.flags & LF_STRM) L.maxpos = std::max(L.maxpos, (int)t.sender); else if (t.flags & LF_L6) L.maxpos = std::max(L.maxpos, 0); else if (t.sender) L.msg = true; }
        up_.ensure(L.total); dup_.ensure(L.total);
        if (in.ndue) { gather_.ensure((size_t)in.ndue * UB); hgather_.ensure((size_t)in.ndue * UB); }
        uint8_t* h = up_.p;
        std::memcpy(h + L.o_str, in.streams, S * sizeof(LStream));
        if (in.ndirect) std::memcpy(h + L.o_dir, in.direct, in.ndirect * sizeof(LTrace));
        if (in.ntemporal) std::memcpy(h + L.o_tmp, in.temporal, in.ntemporal * sizeof(LTrace));
        if (in.ndue) std::memcpy(h + L.o_due, in.due, in.ndue * sizeof(AdamDue));
        if (in.ndirty) std::memcpy(h + L.o_dirty, in.dirty, in.ndirty * sizeof(uint32_t));
        if (in.nreset) std::memcpy(h + L.o_reset, in.reset, in.nreset * sizeof(uint32_t));
        for (uint32_t i = 0; i < in.ndirty; ++i) std::memcpy(h + L.o_wts + (size_t)i * UB, hw_ + (size_t)in.dirty[i] * UB, UB);
        if (in.ninj) { std::memcpy(h + L.o_inj, in.inj, (size_t)in.ninj * sizeof(LInject));   // + each stream's range (injections are grouped by stream)
            uint32_t* off = (uint32_t*)(h + L.o_injoff); uint32_t i = 0; for (int s = 0; s <= S; ++s) { while (i < in.ninj && (int)in.inj[i].stream < s) ++i; off[s] = i; } }
        due_ids_.resize(in.ndue); for (uint32_t i = 0; i < in.ndue; ++i) due_ids_[i] = in.due[i].unit;
        pending_ = true;
        { std::lock_guard<std::mutex> l(mu_); launching_ = true; }
        cv_.notify_all();
        sec_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

private:
    struct Launch { size_t o_str, o_dir, o_tmp, o_due, o_dirty, o_reset, o_wts, o_inj, o_injoff, total; uint32_t ndirect, ntemporal, ndue, ndirty, nreset, ninj; bool head, msg; int maxpos; };
    Launch next_{};
    std::thread launcher_; std::mutex mu_; std::condition_variable cv_; bool launching_ = false, stop_ = false;
    void launcher() {
        for (;;) {
            { std::unique_lock<std::mutex> l(mu_); cv_.wait(l, [&] { return launching_ || stop_; }); if (stop_) return; }
            launch(next_);
            { std::lock_guard<std::mutex> l(mu_); launching_ = false; }
            cv_.notify_all();
        }
    }
    void launch(const Launch& L) {
        const int S = (int)c_.streams, Wn = (int)c_.tc_window;
        uint8_t* d = dup_.p;
        CU(cudaEventRecord(ev0_, cs_));
        CU(cudaMemcpyAsync(d, up_.p, L.total, cudaMemcpyHostToDevice, cs_));
        if (L.head) CU(cudaMemcpyAsync(head_.p, head_host_.p, (8 * F + 4) * sizeof(float), cudaMemcpyHostToDevice, cs_));
        const LStream* dS = (const LStream*)(d + L.o_str);
        if (L.ndirty) k_scatter<<<L.ndirty, 256, 0, cs_>>>(W_.p, (const uint32_t*)(d + L.o_dirty), d + L.o_wts, (int)L.ndirty);
        if (L.nreset) k_reset<<<L.nreset, 256, 0, cs_>>>(M_.p, V_.p, G_.p, (const uint32_t*)(d + L.o_reset), (int)L.nreset);
        CU(cudaMemsetAsync(gws_.p, 0, (size_t)S * F * sizeof(float), cs_));
        if (L.msg) {   // bp_route: receivers' message gradients first
            bpmsg_.ensure((size_t)L.ndirect * Dims::B); CU(cudaMemsetAsync(bpmsg_.p, 0, (size_t)L.ndirect * Dims::B * sizeof(float), cs_));
            k_msg<<<L.ndirect, 128, 0, cs_>>>((const LTrace*)(d + L.o_dir), (int)L.ndirect, dS, W_.p, bpmsg_.p, c_.bp_leak, (int)c_.split);
        }
        const bool strm = L.ndirect && L.maxpos >= 0;
        if (strm) {   // motif_stream: the rounds, last position first
            strmR_.ensure((size_t)S * Dims::B); strmIn_.ensure((size_t)L.ndirect * Dims::B);
            CU(cudaMemsetAsync(strmR_.p, 0, (size_t)S * Dims::B * sizeof(float), cs_)); CU(cudaMemsetAsync(strmIn_.p, 0, (size_t)L.ndirect * Dims::B * sizeof(float), cs_));
            for (int pos = L.maxpos; pos >= 0; --pos) k_stream<<<L.ndirect, 128, 0, cs_>>>((const LTrace*)(d + L.o_dir), (int)L.ndirect, dS, W_.p, strmR_.p, strmIn_.p, pos, c_.bp_leak, (int)c_.split);
        }
        if (L.ndirect) k_backward<<<L.ndirect, 128, 0, cs_>>>((const LTrace*)(d + L.o_dir), (int)L.ndirect, dS, ring_.p, Wn, W_.p, G_.p, gws_.p, 0, c_.bp_leak, L.msg ? bpmsg_.p : nullptr, strm ? strmIn_.p : nullptr, (int)c_.split);
        if (L.ndue && c_.muon) k_muon<<<L.ndue, 256, 0, cs_>>>((const AdamDue*)(d + L.o_due), (int)L.ndue, W_.p, SH_.p, M_.p, G_.p, c_.muon_lr, c_.muon_mom, c_.adam_wd);
        else if (L.ndue) k_adam<<<L.ndue, 256, 0, cs_>>>((const AdamDue*)(d + L.o_due), (int)L.ndue, W_.p, SH_.p, M_.p, V_.p, G_.p, c_.adam_lr, c_.adam_wd);
        if (L.ninj) k_inject<<<S, 32, 0, cs_>>>(dS, S, (const LInject*)(d + L.o_inj), (const uint32_t*)(d + L.o_injoff), ring_.p, Wn);
        k_ring<<<S, 32, 0, cs_>>>(dS, S, ring_.p, Wn, gws_.p, head_.p, hd_.p, c_.decay, c_.reg_scale);
        if (L.ntemporal) k_backward<<<L.ntemporal, 128, 0, cs_>>>((const LTrace*)(d + L.o_tmp), (int)L.ntemporal, dS, ring_.p, Wn, W_.p, G_.p, gws_.p, 1, c_.bp_leak, nullptr, nullptr, 0);
        k_store<<<S, 128, 0, cs_>>>(dS, S, ring_.p, Wn);
        if (L.ndue) {
            k_gather<<<L.ndue, 256, 0, cs_>>>(W_.p, (const AdamDue*)(d + L.o_due), gather_.p, (int)L.ndue);
            CU(cudaMemcpyAsync(hgather_.p, gather_.p, (size_t)L.ndue * UB, cudaMemcpyDeviceToHost, cs_));
        }
        CU(cudaMemcpyAsync(hd_host_.p, hd_.p, (8 * F + 4) * sizeof(float), cudaMemcpyDeviceToHost, cs_));
        CU(cudaMemsetAsync(hd_.p, 0, (8 * F + 4) * sizeof(float), cs_));
        CU(cudaEventRecord(ev1_, cs_));
    }
    LearnConfig c_; uint8_t* hw_; std::string dev_name_; cudaStream_t cs_ = nullptr; double sec_ = 0; bool pending_ = false; cudaEvent_t ev0_{}, ev1_{}; double gpu_ms_ = 0, wait_s_ = 0; uint64_t nstep_ = 0; std::vector<uint32_t> due_ids_;
    DBuf<uint8_t> W_; DBuf<float> SH_, M_, V_, G_; DBuf<DSlot> ring_; DBuf<float> gws_, head_, hd_, bpmsg_, strmR_, strmIn_; DBuf<LStream> streams_;
    DBuf<uint8_t> dup_, gather_; HBuf<uint8_t> up_, hgather_; HBuf<float> head_host_, hd_host_; std::vector<float> hd_acc_; bool head_dirty_ = false;
};

} // namespace

std::unique_ptr<Learner> make_gpu_learner(const LearnConfig& c, uint8_t* host_weights) { return std::make_unique<GpuLearner>(c, host_weights); }

} // namespace laminar
