#pragma once
#include <bit>
// Learned lock/key addressing (design doc §5-6, runbook §3, §9).
//
//  Lock  : (value, mask); accept iff ((key ^ value) & mask) == 0
//  Tree  : adaptive binary decision tree in flat SoA arrays bit[], left[], right[],
//          leaf_offset[], leaf_count[]. Internal nodes have left >= 0; leaves have left == -1.
//          A leaf only ever splits on a bit that is masked in EVERY lock it holds, so each lock
//          goes to exactly one side. A newly born lock (sensory-only mask) is inserted into every
//          leaf consistent with its mask, so ambiguity is preserved rather than invented.
//  Evidence: per conflicted lock, conditional future histograms H0(y)=P(y|lock,b=0),
//          H1(y)=P(y|lock,b=1) for each candidate tumbler bit b. Split when the G statistic
//          (2 * sum n ln(n/e), a divergence relative to within-group sampling variation) exceeds
//          a threshold with minimum evidence on both branches. Prediction error alone never splits.
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include "config.h"

namespace laminar {

struct Lock {
    uint64_t value = 0, mask = 0;
    uint32_t unit = 0;          // owning route body
    uint32_t hits = 0;
    uint32_t last_hit_block = 0, born_block = 0;
    uint32_t err_ema = 0;       // wrong-prediction rate, fixed point /65535
    int32_t evidence = -1;      // slot in EvidencePool or -1
    float utility = 0.f;        // EMA of bits gained over the organism's prior when this lock admitted (ALM route utility)
    uint8_t alive = 0;
};

struct Evidence {
    // counts[(y*2 + v)*64 + bit]: for one record, slab v=1 gets the key's tumbler bits added and slab v=0
    // gets their complement, as two 64-lane uint16 vector adds. n[v*64 + bit] likewise. Lanes 56..63 unused.
    static constexpr int LANES = 64;
    std::vector<uint16_t> counts;
    std::vector<uint32_t> n;
    // child nursery (ALM's route lifecycle): conditional next-symbol counts per value of the lowest still
    // unmasked lag receptor (previous symbol at lag 1, 2 or 3). A child lock on that (lag, symbol) is admitted
    // once it has enough confirmations and shows information gain in bits over the parent.
    std::vector<uint16_t> nursery;   // [256 symbols][buckets]
    std::vector<uint32_t> nursery_n; // [256]
    std::vector<uint32_t> parent; uint32_t ptot = 0;   // running column sums of the nursery (the parent distribution), kept in record()
    uint64_t dirty[4] = {0, 0, 0, 0}; uint32_t samples_eval = 0;   // symbol rows with new samples since the last evaluation
    uint32_t samples_split = 0; int last_split = -1; float last_g = 0.f;   // best_split memo: recomputed only on new evidence
    int lag = 1;
    uint32_t lock = UINT32_MAX;
    uint32_t samples = 0;
    template <class Ar> void state_io(Ar& ar) { ar(counts); ar(n); ar(nursery); ar(nursery_n); ar(parent); ar(ptot); ar(dirty); ar(samples_eval); ar(samples_split); ar(last_split); ar(last_g); ar(lag); ar(lock); ar(samples); }
    void reset(uint32_t buckets) {
        counts.assign((size_t)buckets * 2 * LANES, 0);
        n.assign((size_t)2 * LANES, 0);
        nursery.assign((size_t)256 * buckets, 0); nursery_n.assign(256, 0);
        parent.assign(buckets, 0); ptot = 0; dirty[0] = dirty[1] = dirty[2] = dirty[3] = 0; samples_eval = 0; samples_split = 0; last_split = -1; last_g = 0.f;
        samples = 0; lock = UINT32_MAX; lag = 1;
    }
};

struct EvidencePool {
    std::vector<Evidence> slots; std::vector<uint32_t> free_slots; uint32_t buckets = 32;
    template <class Ar> void state_io(Ar& ar) { ar(slots); ar(free_slots); ar(buckets); }
    void init(uint32_t nslots, uint32_t b) {
        buckets = b; slots.resize(nslots); free_slots.clear();
        for (uint32_t i = 0; i < nslots; ++i) { slots[i].reset(b); free_slots.push_back(nslots - 1 - i); }
    }
    int32_t acquire(uint32_t lock, uint64_t mask = 0) {
        if (free_slots.empty()) return -1;
        uint32_t s = free_slots.back(); free_slots.pop_back();
        slots[s].reset(buckets); slots[s].lock = lock;
        // the lowest lag whose 8 key bits are not yet all masked is this lock's next receptor
        slots[s].lag = 0; for (int l = 1; l <= 3; ++l) { uint64_t lb = 0xFFULL << (8 * l); if ((mask & lb) != lb) { slots[s].lag = l; break; } }
        return (int32_t)s;
    }
    // Admissible children of a conflicted lock: (symbol, gain_bits) for every nursery entry with enough
    // confirmations whose Dirichlet-smoothed conditional distribution gains >= min_bits over the parent's.
    // Incremental: the parent distribution is a running sum kept by record(), and only the symbol rows that received
    // samples since the last evaluation are scored (a slot with no new samples costs nothing). This was 29% of all
    // time as a full 256 x buckets scan per slot per block.
    void nursery_candidates(int32_t s, uint32_t min_n, float min_bits, std::vector<std::pair<uint32_t, float>>& out) {
        Evidence& e = slots[s]; out.clear(); if (!e.lag || e.samples == e.samples_eval || !e.ptot) return;
        e.samples_eval = e.samples;
        const float alpha = 1.f, ptot = (float)e.ptot;
        for (int w = 0; w < 4; ++w) { uint64_t m = e.dirty[w]; e.dirty[w] = 0;
            while (m) { int sy = 64 * w + std::countr_zero(m); m &= m - 1;
                uint32_t n = e.nursery_n[sy]; if (n < min_n) { e.dirty[w] |= 1ULL << (sy & 63); continue; }   // not yet confirmed: stays dirty
                const uint16_t* row = &e.nursery[(size_t)sy * buckets]; float gain = 0.f;
                for (uint32_t y = 0; y < buckets; ++y) { float c = row[y]; if (c <= 0.f) continue; float py = (float)e.parent[y];
                    float pc = (c + alpha * py / ptot) / (n + alpha);          // child, smoothed toward parent
                    float pp = (py + alpha / buckets) / (ptot + alpha);        // parent
                    gain += (c / n) * std::log2(pc / pp); }
                if (gain >= min_bits) out.emplace_back((uint32_t)sy, gain); }
        }
    }
    void release(int32_t s) { if (s >= 0) { slots[s].lock = UINT32_MAX; free_slots.push_back((uint32_t)s); } }
    void record(int32_t s, uint64_t key, uint32_t y) {
        Evidence& e = slots[s];
        uint32_t yb = y % buckets;
        uint16_t* c0 = &e.counts[(size_t)yb * 2 * Evidence::LANES]; uint16_t* c1 = c0 + Evidence::LANES;
        uint32_t* n0 = e.n.data(); uint32_t* n1 = n0 + Evidence::LANES;
        const uint64_t bits = (key >> Dims::SENSORY_BITS) & ((1ULL << Dims::TUMBLER_BITS) - 1);
#if defined(__AVX2__)
        // expand 64 key bits to 64 x uint16 {0,1}: 4 chunks of 16 bits
        const __m256i lanebit = _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, (short)32768);
        const __m256i one16 = _mm256_set1_epi16(1);
        for (int ch = 0; ch < 4; ++ch) {
            __m256i bv = _mm256_set1_epi16((short)((bits >> (16 * ch)) & 0xFFFF));
            __m256i b1 = _mm256_and_si256(_mm256_cmpeq_epi16(_mm256_and_si256(bv, lanebit), lanebit), one16);   // bit set -> 1
            __m256i b0 = _mm256_xor_si256(b1, one16);                                                             // complement
            _mm256_storeu_si256((__m256i*)(c1 + 16 * ch), _mm256_adds_epu16(_mm256_loadu_si256((const __m256i*)(c1 + 16 * ch)), b1));
            _mm256_storeu_si256((__m256i*)(c0 + 16 * ch), _mm256_adds_epu16(_mm256_loadu_si256((const __m256i*)(c0 + 16 * ch)), b0));
            __m256i w1lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b1)), w1hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b1, 1));
            __m256i w0lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b0)), w0hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b0, 1));
            _mm256_storeu_si256((__m256i*)(n1 + 16 * ch), _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(n1 + 16 * ch)), w1lo));
            _mm256_storeu_si256((__m256i*)(n1 + 16 * ch + 8), _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(n1 + 16 * ch + 8)), w1hi));
            _mm256_storeu_si256((__m256i*)(n0 + 16 * ch), _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(n0 + 16 * ch)), w0lo));
            _mm256_storeu_si256((__m256i*)(n0 + 16 * ch + 8), _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(n0 + 16 * ch + 8)), w0hi));
        }
#else
        for (int b = 0; b < Dims::TUMBLER_BITS; ++b) {
            uint32_t v = (uint32_t)((bits >> b) & 1);
            uint16_t& c = (v ? c1 : c0)[b]; c += (c < 65535);
            ++(v ? n1 : n0)[b];
        }
#endif
        if (e.lag) { uint32_t sym = (uint32_t)((key >> (8 * e.lag)) & 0xFF); uint16_t& c = e.nursery[(size_t)sym * buckets + yb]; if (c < 65535) { ++c; ++e.parent[yb]; ++e.ptot; } ++e.nursery_n[sym]; e.dirty[sym >> 6] |= 1ULL << (sym & 63); }
        ++e.samples;
    }
    // Returns best tumbler bit (0..55) and its G statistic; bit=-1 if no candidate reaches min evidence.
    int best_split(int32_t s, uint32_t min_ev, float& g_out) {
        Evidence& e = slots[s];
        if (e.samples == e.samples_split) { g_out = e.last_g; return e.last_split; }   // no new evidence since the last evaluation: same answer
        e.samples_split = e.samples;
        int best = -1; float bestg = 0.f;
        for (int b = 0; b < Dims::TUMBLER_BITS; ++b) {
            uint32_t n0 = e.n[b], n1 = e.n[Evidence::LANES + b];
            if (n0 < min_ev || n1 < min_ev) continue;
            float g = 0.f; float N = (float)(n0 + n1);
            for (uint32_t y = 0; y < buckets; ++y) {
                const uint16_t* slab = &e.counts[(size_t)y * 2 * Evidence::LANES];
                float c0 = slab[b], c1 = slab[Evidence::LANES + b];
                float tot = c0 + c1; if (tot == 0.f) continue;
                float e0 = tot * n0 / N, e1 = tot * n1 / N;
                if (c0 > 0.f) g += c0 * std::log(c0 / e0);
                if (c1 > 0.f) g += c1 * std::log(c1 / e1);
            }
            g *= 2.f;
            if (g > bestg) { bestg = g; best = b; }
        }
        e.last_split = best; e.last_g = bestg; g_out = bestg; return best;
    }
};

struct AddressTree {
    // node SoA
    std::vector<uint8_t> bit; std::vector<int32_t> left, right;
    std::vector<uint32_t> leaf_offset, leaf_count, leaf_cap;
    std::vector<uint32_t> pool;                  // lock ids, slab-allocated per leaf
    std::vector<Lock> locks; std::vector<uint32_t> free_locks;
    std::vector<std::vector<uint32_t>> lock_leaves; // cold: leaves each lock lives in
    uint32_t leaf_max = 4;
    template <class Ar> void state_io(Ar& ar) { ar(bit); ar(left); ar(right); ar(leaf_offset); ar(leaf_count); ar(leaf_cap); ar(pool); ar(locks); ar(free_locks); ar(lock_leaves); ar(leaf_max); }

    void init(uint32_t leaf_max_) {
        leaf_max = leaf_max_;
        bit.assign(1, 0); left.assign(1, -1); right.assign(1, -1);
        leaf_offset.assign(1, 0); leaf_count.assign(1, 0); leaf_cap.assign(1, 8);
        pool.assign(8, 0); locks.clear(); free_locks.clear(); lock_leaves.clear();
    }
    size_t num_nodes() const { return bit.size(); }
    size_t live_locks() const { size_t n = 0; for (auto& l : locks) n += l.alive; return n; }

    // ---- hot path -------------------------------------------------------------------------
    inline int32_t walk(uint64_t key, uint32_t& depth) const {
        int32_t n = 0; depth = 0;
        while (left[n] >= 0) { n = ((key >> bit[n]) & 1) ? right[n] : left[n]; ++depth; }
        return n;
    }
    inline uint32_t match(int32_t leaf, uint64_t key, uint32_t* out, uint32_t cap) const {
        const uint32_t* ids = pool.data() + leaf_offset[leaf];
        uint32_t c = leaf_count[leaf], m = 0;
        for (uint32_t i = 0; i < c && m < cap; ++i) {
            const Lock& L = locks[ids[i]];
            if (((key ^ L.value) & L.mask) == 0) out[m++] = ids[i];
        }
        return m;
    }

    // ---- structural (between event blocks, or rare births) --------------------------------
    uint32_t new_lock(uint64_t value, uint64_t mask, uint32_t unit, uint32_t block) {
        uint32_t id;
        if (!free_locks.empty()) { id = free_locks.back(); free_locks.pop_back(); }
        else { id = (uint32_t)locks.size(); locks.emplace_back(); lock_leaves.emplace_back(); }
        Lock& L = locks[id]; L = Lock(); L.value = value; L.mask = mask; L.unit = unit; L.born_block = block; L.last_hit_block = block; L.alive = 1;
        lock_leaves[id].clear();
        insert(0, id);
        return id;
    }
    void kill_lock(uint32_t id) {
        for (uint32_t leaf : lock_leaves[id]) leaf_remove(leaf, id);
        lock_leaves[id].clear(); locks[id].alive = 0; free_locks.push_back(id);
    }
    // Move locks of leaves that grew past leaf_max under a new decision bit (masked in all of them).
    uint32_t maintain() {
        uint32_t splits = 0;
        for (int32_t n = 0; n < (int32_t)bit.size(); ++n) {
            if (left[n] >= 0 || leaf_count[n] <= leaf_max) continue;
            uint32_t off = leaf_offset[n], c = leaf_count[n];
            uint64_t all = ~0ULL; for (uint32_t i = 0; i < c; ++i) all &= locks[pool[off + i]].mask;
            int bestb = -1; uint32_t bestimb = UINT32_MAX;
            for (int b = 0; b < 64; ++b) {
                if (!((all >> b) & 1)) continue;
                uint32_t n1 = 0; for (uint32_t i = 0; i < c; ++i) n1 += (uint32_t)((locks[pool[off + i]].value >> b) & 1);
                if (n1 == 0 || n1 == c) continue;
                uint32_t imb = (n1 > c - n1) ? n1 - (c - n1) : (c - n1) - n1;
                if (imb < bestimb) { bestimb = imb; bestb = b; }
            }
            if (bestb < 0) continue;                       // genuine ambiguity: leaf stays fat
            std::vector<uint32_t> ids(pool.begin() + off, pool.begin() + off + c);
            int32_t l = add_leaf(), r = add_leaf();
            bit[n] = (uint8_t)bestb; left[n] = l; right[n] = r; leaf_count[n] = 0;
            for (uint32_t id : ids) {
                auto& lv = lock_leaves[id]; lv.erase(std::remove(lv.begin(), lv.end(), (uint32_t)n), lv.end());
                int32_t side = ((locks[id].value >> bestb) & 1) ? r : l;
                leaf_push(side, id); lock_leaves[id].push_back((uint32_t)side);
            }
            ++splits;
        }
        return splits;
    }
    void depth_stats(double& mean, uint32_t& p95, double& mean_occ) const {
        std::vector<uint32_t> ds; double occ = 0; size_t leaves = 0;
        rec_depth(0, 0, ds);
        for (size_t n = 0; n < bit.size(); ++n) if (left[n] < 0) { occ += leaf_count[n]; ++leaves; }
        if (ds.empty()) { mean = 0; p95 = 0; mean_occ = 0; return; }
        std::sort(ds.begin(), ds.end()); double s = 0; for (auto d : ds) s += d;
        mean = s / ds.size(); p95 = ds[(size_t)(0.95 * (ds.size() - 1))]; mean_occ = leaves ? occ / leaves : 0;
    }

private:
    void rec_depth(int32_t n, uint32_t d, std::vector<uint32_t>& ds) const {
        if (left[n] < 0) { ds.push_back(d); return; }
        rec_depth(left[n], d + 1, ds); rec_depth(right[n], d + 1, ds);
    }
    int32_t add_leaf() {
        bit.push_back(0); left.push_back(-1); right.push_back(-1);
        leaf_offset.push_back((uint32_t)pool.size()); leaf_count.push_back(0); leaf_cap.push_back(8);
        pool.resize(pool.size() + 8, 0);
        return (int32_t)bit.size() - 1;
    }
    void leaf_push(int32_t leaf, uint32_t id) {
        if (leaf_count[leaf] == leaf_cap[leaf]) {              // slab grow: move to end of pool
            uint32_t nc = leaf_cap[leaf] * 2, no = (uint32_t)pool.size();
            pool.resize(pool.size() + nc, 0);
            std::copy(pool.begin() + leaf_offset[leaf], pool.begin() + leaf_offset[leaf] + leaf_count[leaf], pool.begin() + no);
            leaf_offset[leaf] = no; leaf_cap[leaf] = nc;
        }
        pool[leaf_offset[leaf] + leaf_count[leaf]++] = id;
    }
    void leaf_remove(int32_t leaf, uint32_t id) {
        uint32_t off = leaf_offset[leaf], c = leaf_count[leaf];
        for (uint32_t i = 0; i < c; ++i) if (pool[off + i] == id) { pool[off + i] = pool[off + c - 1]; --leaf_count[leaf]; return; }
    }
    void insert(int32_t n, uint32_t id) {
        while (left[n] >= 0) {
            const Lock& L = locks[id];
            if ((L.mask >> bit[n]) & 1) n = ((L.value >> bit[n]) & 1) ? right[n] : left[n];
            else { insert(left[n], id); n = right[n]; }     // wildcard: both subtrees
        }
        leaf_push(n, id); lock_leaves[id].push_back((uint32_t)n);
    }
};

} // namespace laminar
