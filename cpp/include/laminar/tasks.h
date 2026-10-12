#pragma once
// Task streams (design doc §10, §15; runbook §11, §14).
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include "rng.h"

namespace laminar {

// Synthetic vocabulary: V symbols, C contexts. Each context c owns a random map next_c[sym].
// The stream stays in a context for a run of `run_len` symbols, then jumps to another.
// With probability `noise` the next symbol is replaced by a uniform random symbol.
// The context is only inferable from history (the key's context bits), which is exactly what
// the lock/tumbler machinery has to discover.
struct SyntheticVocab {
    uint32_t V = 32, C = 8; double noise = 0.08; uint32_t run_len = 64;
    std::vector<uint8_t> maps; // C*V
    Rng rng; uint32_t ctx = 0, cur = 0, left = 0;
    SyntheticVocab(uint32_t v, uint32_t c, double n, uint64_t seed, uint32_t run = 64) : V(v), C(c), noise(n), run_len(run), rng(seed) {
        maps.resize((size_t)C * V);
        for (uint32_t k = 0; k < C; ++k) {
            // random permutation per context (derangement not required)
            std::vector<uint8_t> p(V); for (uint32_t i = 0; i < V; ++i) p[i] = (uint8_t)i;
            for (uint32_t i = V - 1; i > 0; --i) { uint32_t j = rng.below(i + 1); std::swap(p[i], p[j]); }
            for (uint32_t i = 0; i < V; ++i) maps[(size_t)k * V + i] = p[i];
        }
        cur = rng.below(V);
    }
    std::vector<uint8_t> generate(size_t n) {
        std::vector<uint8_t> out(n);
        for (size_t i = 0; i < n; ++i) {
            if (left == 0) { ctx = rng.below(C); left = run_len; }
            out[i] = (uint8_t)cur;
            uint8_t nxt = maps[(size_t)ctx * V + cur];
            if (rng.uniform() < noise) nxt = (uint8_t)rng.below(V);
            cur = nxt; --left;
        }
        return out;
    }
};

inline std::vector<uint8_t> read_file_bytes(const std::string& path, size_t max_bytes = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (max_bytes && d.size() > max_bytes) d.resize(max_bytes);
    return d;
}

} // namespace laminar
