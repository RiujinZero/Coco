#pragma once
// Per-block telemetry (runbook §12 / design doc §13). Every counter listed there is here.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace laminar {

inline uint64_t tsc() { return __rdtsc(); }

enum Stage { ST_INPUT = 0, ST_ADDRESS, ST_L1, ST_L23, ST_NEIGHBOR, ST_L5, ST_L6, ST_L4, ST_READOUT, ST_LEARN, ST_EVIDENCE, ST_STRUCT, NUM_STAGES };
static const char* STAGE_NAMES[NUM_STAGES] = {"input", "address", "l1", "l2l3", "neighbor", "l5", "l6", "l4", "readout", "learn", "evidence", "struct"};

struct BlockStats {
    uint64_t events = 0, bytes = 0;
    double loss_sum = 0; uint64_t correct = 0;
    uint64_t addr_decisions = 0, candidate_locks = 0, no_candidate = 0;
    uint64_t l1_admissions = 0, l1_survive = 0, l23_survive = 0, l5_survive = 0, l6_survive = 0, l4_runs = 0;
    uint64_t neighbor_tests = 0, neighbor_accepts = 0, dedup_skips = 0;
    uint64_t macs = 0, weight_bytes = 0;
    uint64_t acc_updates = 0, level_changes = 0;
    uint64_t lock_births = 0, lock_splits = 0, lock_deaths = 0, leaf_splits = 0;
    uint64_t novel_events = 0, trace_overflow = 0;
    uint64_t episodic_births = 0, episodic_evictions = 0, exact_admissions = 0, imprint_ok = 0;
    uint64_t nursery_admissions = 0, lock_recycled = 0;
    std::vector<uint32_t> queue_sizes;   // per-event active count
    std::vector<uint32_t> depths;        // per-event address depth
    uint64_t stage_cycles[NUM_STAGES] = {0};
    uint64_t recirc_passes = 0;
    double seconds = 0;
    // structural snapshot
    uint64_t live_locks = 0, live_units = 0, tree_nodes = 0; double depth_mean = 0, leaf_occ = 0; uint32_t depth_p95 = 0;

    void reset() { *this = BlockStats(); }
    static uint32_t pct(std::vector<uint32_t>& v, double p) {
        if (v.empty()) return 0; std::nth_element(v.begin(), v.begin() + (size_t)(p * (v.size() - 1)), v.end());
        return v[(size_t)(p * (v.size() - 1))];
    }
    std::string report(uint64_t block_idx, double vocab_ln2 = 0.6931471805599453) {
        char buf[2048]; double e = events ? (double)events : 1.0;
        std::vector<uint32_t> q = queue_sizes;
        uint32_t qmax = q.empty() ? 0 : *std::max_element(q.begin(), q.end());
        uint32_t q50 = pct(q, 0.5), q95 = pct(q, 0.95), q99 = pct(q, 0.99);
        uint64_t cyc = 0; for (int s = 0; s < NUM_STAGES; ++s) cyc += stage_cycles[s];
        std::string st;
        for (int s = 0; s < NUM_STAGES; ++s) { char t[64]; std::snprintf(t, sizeof t, " %s=%.0f%%", STAGE_NAMES[s], cyc ? 100.0 * stage_cycles[s] / cyc : 0.0); st += t; }
        std::snprintf(buf, sizeof buf,
            "blk %llu | %.1f KB/s (%.0f ev/s) | loss %.4f nat (%.3f bits) acc %.2f%% | addr %.1f dec %.2f cand nocand %.3f depth %.1f/p95 %u occ %.2f"
            " | L1 %.2f surv %.2f L23 %.2f L5 %.2f L6 %.2f L4 %.3f | nbr test %.2f acc %.3f dedup %.3f | MAC %.0f wbytes %.0f | accupd %.1f lvl %.3f"
            " | births %llu splits %llu leafsplit %llu deaths %llu | novel %.3f | q p50 %u p95 %u p99 %u max %u | locks %llu units %llu nodes %llu | ovf %llu |%s",
            (unsigned long long)block_idx, seconds > 0 ? bytes / seconds / 1024.0 : 0.0, seconds > 0 ? events / seconds : 0.0,
            loss_sum / e, loss_sum / e / vocab_ln2, 100.0 * correct / e,
            addr_decisions / e, candidate_locks / e, no_candidate / e, depth_mean, depth_p95, leaf_occ,
            l1_admissions / e, l1_survive / e, l23_survive / e, l5_survive / e, l6_survive / e, l4_runs / e,
            neighbor_tests / e, neighbor_accepts / e, dedup_skips / e, macs / e, weight_bytes / e, acc_updates / e, level_changes / e,
            (unsigned long long)lock_births, (unsigned long long)lock_splits, (unsigned long long)leaf_splits, (unsigned long long)lock_deaths,
            novel_events / e, q50, q95, q99, qmax,
            (unsigned long long)live_locks, (unsigned long long)live_units, (unsigned long long)tree_nodes, (unsigned long long)trace_overflow, st.c_str());
        return buf;
    }
    std::string json(uint64_t block_idx) {
        char buf[4096]; double e = events ? (double)events : 1.0;
        std::vector<uint32_t> q = queue_sizes; uint32_t qmax = q.empty() ? 0 : *std::max_element(q.begin(), q.end());
        uint32_t q50 = pct(q, 0.5), q95 = pct(q, 0.95), q99 = pct(q, 0.99);
        std::string st = "{";
        for (int s = 0; s < NUM_STAGES; ++s) { char t[64]; std::snprintf(t, sizeof t, "%s\"%s\":%llu", s ? "," : "", STAGE_NAMES[s], (unsigned long long)stage_cycles[s]); st += t; }
        st += "}";
        std::snprintf(buf, sizeof buf,
            "{\"block\":%llu,\"events\":%llu,\"seconds\":%.6f,\"kbps\":%.3f,\"loss_nat\":%.6f,\"acc\":%.5f,\"addr_dec\":%.4f,\"cand\":%.4f,\"nocand\":%.5f,"
            "\"depth_mean\":%.3f,\"depth_p95\":%u,\"leaf_occ\":%.3f,\"l1_adm\":%.4f,\"l1_surv\":%.4f,\"l23_surv\":%.4f,\"l5_surv\":%.4f,\"l6_surv\":%.4f,\"l4_runs\":%.4f,"
            "\"nbr_tests\":%.4f,\"nbr_acc\":%.4f,\"macs\":%.2f,\"wbytes\":%.2f,\"acc_upd\":%.3f,\"lvl_chg\":%.4f,\"births\":%llu,\"splits\":%llu,\"leaf_splits\":%llu,\"deaths\":%llu,"
            "\"novel\":%.4f,\"q50\":%u,\"q95\":%u,\"q99\":%u,\"qmax\":%u,\"locks\":%llu,\"units\":%llu,\"nodes\":%llu,\"stages\":%s}",
            (unsigned long long)block_idx, (unsigned long long)events, seconds, seconds > 0 ? bytes / seconds / 1024.0 : 0.0, loss_sum / e, (double)correct / e,
            addr_decisions / e, candidate_locks / e, no_candidate / e, depth_mean, depth_p95, leaf_occ,
            l1_admissions / e, l1_survive / e, l23_survive / e, l5_survive / e, l6_survive / e, l4_runs / e,
            neighbor_tests / e, neighbor_accepts / e, macs / e, weight_bytes / e, acc_updates / e, level_changes / e,
            (unsigned long long)lock_births, (unsigned long long)lock_splits, (unsigned long long)leaf_splits, (unsigned long long)lock_deaths,
            novel_events / e, q50, q95, q99, qmax, (unsigned long long)live_locks, (unsigned long long)live_units, (unsigned long long)tree_nodes, st.c_str());
        return buf;
    }
};

struct StageTimer {
    bool on; uint64_t t0; int stage; BlockStats* s;
    StageTimer(BlockStats* st, int stage_, bool enabled) : on(enabled), t0(enabled ? tsc() : 0), stage(stage_), s(st) {}
    ~StageTimer() { if (on) s->stage_cycles[stage] += tsc() - t0; }
};

} // namespace laminar
