// Kernel equivalence (runbook §13.5: re-run after every kernel change) and address-tree invariants.
#include "laminar/int4.h"
#include "laminar/address.h"
#include "laminar/rng.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <set>

using namespace laminar;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    Rng r(42);
    // nibble pack roundtrip
    std::vector<uint8_t> unit(Dims::UNIT_BYTES, 0);
    for (int n = 0; n < Dims::UNIT_NIBBLES; ++n) nib_set(unit.data(), n, (int)r.below(16) - 8);
    for (int n = 0; n < Dims::UNIT_NIBBLES; ++n) { int v = (int)((mix64(n) % 16)) - 8; nib_set(unit.data(), n, v); CHECK(nib_get(unit.data(), n) == v, "nib %d", n); }

    // kernels agree across all layers on random data
    uint8_t shifts[NUM_LAYERS] = {1, 2, 2, 3, 3, 3};
    for (int trial = 0; trial < 200; ++trial) {
        for (auto& b : unit) b = (uint8_t)r.next();
        for (int li = 0; li < NUM_LAYERS; ++li) {
            LayerDesc L = layer_desc(li, shifts);
            int8_t x[Dims::IN]; for (int i = 0; i < L.in; ++i) x[i] = (int8_t)((int)r.below(255) - 127);
            int8_t o1[64], o2[64], o3[64]; std::vector<int8_t> scratch(Dims::P1);
            layer_scalar(unit.data(), L, x, o1);
            layer_expand(unit.data(), L, x, o2, scratch.data());
            CHECK(std::memcmp(o1, o2, L.out) == 0, "expand != scalar layer %d", li);
#if defined(__AVX2__)
            XPacked xp; pack_x(x, L.in, xp); layer_avx2(unit.data(), L, xp, o3);
            CHECK(std::memcmp(o1, o3, L.out) == 0, "avx2 != scalar layer %d trial %d", li, trial);
#endif
        }
    }

    // address tree: walk+match must equal brute-force scan over all live locks, before and after maintenance
    AddressTree t; t.init(4);
    std::vector<uint32_t> ids;
    for (int i = 0; i < 3000; ++i) {
        uint64_t key = r.next(); uint64_t mask = 0xFF | (r.next() & 0xFFFFFF00ULL & (r.next() | r.next()));
        ids.push_back(t.new_lock(key, mask, (uint32_t)i, 0));
    }
    auto verify = [&](const char* tag) {
        for (int q = 0; q < 2000; ++q) {
            uint64_t key = (q & 1) ? t.locks[ids[r.below((uint32_t)ids.size())]].value ^ (r.next() & 0xFFFFFF00ULL) : r.next();
            std::set<uint32_t> brute; for (auto id : ids) { auto& L = t.locks[id]; if (L.alive && ((key ^ L.value) & L.mask) == 0) brute.insert(id); }
            uint32_t depth; int32_t leaf = t.walk(key, depth); uint32_t out[4096]; uint32_t n = t.match(leaf, key, out, 4096);
            std::set<uint32_t> got(out, out + n);
            CHECK(got == brute, "%s: tree match (%zu) != brute (%zu)", tag, got.size(), brute.size());
        }
    };
    verify("before-maintain");
    uint32_t splits = 0; for (int i = 0; i < 50; ++i) { uint32_t s = t.maintain(); splits += s; if (!s) break; }
    CHECK(splits > 0, "no leaf splits happened");
    verify("after-maintain");
    // late births with sensory-only masks must still be found for every context
    for (int i = 0; i < 200; ++i) ids.push_back(t.new_lock(r.next(), 0xFF, 100000 + i, 1));
    verify("after-late-births");
    for (int i = 0; i < 100; ++i) t.kill_lock(ids[r.below(3000)]);
    verify("after-deaths");
    double dm; uint32_t p95; double occ; t.depth_stats(dm, p95, occ);
    std::printf("tree: nodes %zu depth mean %.1f p95 %u occ %.2f splits %u\n", t.num_nodes(), dm, p95, occ, splits);
    std::printf(fails ? "FAILED (%d)\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
