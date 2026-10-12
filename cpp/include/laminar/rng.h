#pragma once
#include <cstdint>

// Deterministic PRNG shared bit-for-bit with the Python reference (python/laminar_ref/rng.py).
namespace laminar {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 1) : s(seed) {}
    // splitmix64
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    uint32_t below(uint32_t n) { return (uint32_t)(next() % n); }
    double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
};

// murmur3 finalizer: stateless 64-bit mixer used for keys, route keys and perturbation signs.
inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

} // namespace laminar
