#pragma once
// Packed INT4 durable weight store and layer kernels (runbook §2, §5).
//
//  * two signed INT4 weights per byte: nibble 2k = low nibble of byte k, nibble 2k+1 = high nibble
//  * unit-major, layer-contiguous, 2816 bytes per unit, 64-byte aligned
//  * kernels: scalar nibble-unpack reference, AVX2 unpack-and-madd, and "expand" (unpack the
//    active unit's layer to INT8 scratch then INT8 dot) — all must produce identical INT32 sums.
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "config.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace laminar {

inline int8_t nib_get(const uint8_t* base, uint32_t nib) {
    uint8_t b = base[nib >> 1];
    int v = (nib & 1) ? (b >> 4) : (b & 15);
    return (int8_t)((v ^ 8) - 8);          // sign-extend 4 bits
}
inline void nib_set(uint8_t* base, uint32_t nib, int v) {
    uint8_t& b = base[nib >> 1];
    v &= 15;
    if (nib & 1) b = (uint8_t)((b & 0x0F) | (v << 4)); else b = (uint8_t)((b & 0xF0) | v);
}

inline int8_t clamp8(int32_t v) { return (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v); }

// 64-byte aligned byte buffer (no exceptions in hot path, plain malloc).
struct AlignedBuf {
    uint8_t* p = nullptr; size_t n = 0;
    void alloc(size_t bytes) {
        release();
        n = bytes;
#if defined(_MSC_VER)
        p = (uint8_t*)_aligned_malloc(bytes ? bytes : 64, 64);
#else
        size_t r = (bytes + 63) & ~(size_t)63; p = (uint8_t*)std::aligned_alloc(64, r ? r : 64);
#endif
        std::memset(p, 0, bytes);
    }
    void release() {
#if defined(_MSC_VER)
        if (p) _aligned_free(p);
#else
        if (p) std::free(p);
#endif
        p = nullptr; n = 0;
    }
    ~AlignedBuf() { release(); }
};

// ---- scalar reference --------------------------------------------------------------------
// out[j] = clamp( (sum_i x[i] * W[j][i]) >> shift )   ; W row j at nibble off + j*in
inline void layer_scalar(const uint8_t* unit, const LayerDesc& L, const int8_t* x, int8_t* out) {
    for (int j = 0; j < L.out; ++j) {
        const uint8_t* row = unit + ((L.nib_off + (uint32_t)j * L.in) >> 1);   // in is even -> byte aligned
        int32_t acc = 0;
        for (int i = 0; i < L.in; i += 2) {
            uint8_t b = row[i >> 1];
            int w0 = ((b & 15) ^ 8) - 8, w1 = ((b >> 4) ^ 8) - 8;
            acc += x[i] * w0 + x[i + 1] * w1;
        }
        out[j] = clamp8(acc >> L.shift);
    }
}

// ---- AVX2: unpack 16 nibbles (8 bytes) to 16 x int16 and madd against pre-deinterleaved x -----
#if defined(__AVX2__)
// x_even[i] = x[2i], x_odd[i] = x[2i+1] (each in/2 long), so low nibbles pair with x_even.
struct XPacked { alignas(32) int16_t even[Dims::IN / 2]; alignas(32) int16_t odd[Dims::IN / 2]; };
inline void pack_x(const int8_t* x, int in, XPacked& xp) {
    for (int i = 0; i < in / 2; ++i) { xp.even[i] = x[2 * i]; xp.odd[i] = x[2 * i + 1]; }
}
inline int32_t hsum256(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
inline void layer_avx2(const uint8_t* unit, const LayerDesc& L, const XPacked& xp, int8_t* out) {
    const __m128i m15 = _mm_set1_epi8(0x0F), m8 = _mm_set1_epi8(8);
    const int half = L.in / 2;               // bytes per row
    const int chunks = half / 8;             // 8 bytes = 16 nibbles per chunk
    for (int j = 0; j < L.out; ++j) {
        const uint8_t* row = unit + ((L.nib_off + (uint32_t)j * L.in) >> 1);
        __m256i acc = _mm256_setzero_si256();
        int c = 0;
        for (; c < chunks; ++c) {
            __m128i b = _mm_loadl_epi64((const __m128i*)(row + c * 8));
            __m128i lo = _mm_sub_epi8(_mm_xor_si128(_mm_and_si128(b, m15), m8), m8);
            __m128i hi = _mm_sub_epi8(_mm_xor_si128(_mm_and_si128(_mm_srli_epi16(b, 4), m15), m8), m8);
            __m256i lo16 = _mm256_cvtepi8_epi16(lo), hi16 = _mm256_cvtepi8_epi16(hi);   // only first 8 lanes valid
            __m256i xe = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(xp.even + c * 8)));
            __m256i xo = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(xp.odd + c * 8)));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo16, xe));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(hi16, xo));
        }
        int32_t s = hsum256(acc);
        for (int i = c * 8; i < half; ++i) {          // 4-byte tails (L3/L5/L6 rows)
            uint8_t b = row[i];
            s += xp.even[i] * (((b & 15) ^ 8) - 8) + xp.odd[i] * (((b >> 4) ^ 8) - 8);
        }
        out[j] = clamp8(s >> L.shift);
    }
}
#endif

// ---- expand variant: unpack the whole layer of the ACTIVE unit into INT8 scratch, then dot ----
inline void layer_expand(const uint8_t* unit, const LayerDesc& L, const int8_t* x, int8_t* out, int8_t* scratch /* >= out*in */) {
    const uint8_t* base = unit + (L.nib_off >> 1);
    const int n = L.in * L.out;
    for (int k = 0; k < n; k += 2) { uint8_t b = base[k >> 1]; scratch[k] = (int8_t)(((b & 15) ^ 8) - 8); scratch[k + 1] = (int8_t)(((b >> 4) ^ 8) - 8); }
    for (int j = 0; j < L.out; ++j) {
        const int8_t* row = scratch + j * L.in;
        int32_t acc = 0;
#if defined(__AVX2__)
        __m256i a = _mm256_setzero_si256(); int i = 0;
        for (; i + 16 <= L.in; i += 16) {
            __m256i w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(row + i)));
            __m256i xv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(x + i)));
            a = _mm256_add_epi32(a, _mm256_madd_epi16(w, xv));
        }
        acc = hsum256(a);
        for (; i < L.in; ++i) acc += x[i] * row[i];
#else
        for (int i = 0; i < L.in; ++i) acc += x[i] * row[i];
#endif
        out[j] = clamp8(acc >> L.shift);
    }
}

// Dispatcher used by the organism. Chooses at runtime but the body of each variant is specialised
// on the layer descriptor's (in, out) which are compile-time constants at call sites.
struct KernelCtx {
    Kernel kind = Kernel::Scalar;
    std::vector<int8_t> scratch = std::vector<int8_t>(Dims::P1);
#if defined(__AVX2__)
    XPacked xp;
#endif
    void run(const uint8_t* unit, const LayerDesc& L, const int8_t* x, int8_t* out) {
        switch (kind) {
#if defined(__AVX2__)
        case Kernel::Avx2: pack_x(x, L.in, xp); layer_avx2(unit, L, xp, out); return;
#else
        case Kernel::Avx2: layer_scalar(unit, L, x, out); return;
#endif
        case Kernel::Expand: layer_expand(unit, L, x, out, scratch.data()); return;
        default: layer_scalar(unit, L, x, out); return;
        }
    }
};

} // namespace laminar
