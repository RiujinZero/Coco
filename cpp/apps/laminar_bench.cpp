// Memory/cache stress test (runbook §11): the real 200k x 5632 packed INT4 store (~563 MB) with
// randomly scattered route bodies, under four activity regimes. Reports throughput vs metabolism.
//   mode=familiar     sparse familiar stream, learned-like death thresholds, nothing novel
//   mode=moderate     half the locks look novel (lower resistance -> more survival)
//   mode=storm        random 256-symbol stream, every event births/uses a new lock
//   mode=pathological death thresholds 0, L4 always, max hops: full 5632-synapse bodies
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include <cstdio>
#include <string>

using namespace laminar;

int main(int argc, char** argv) {
    std::map<std::string, std::string> rest;
    Params p = Params::from_args(argc, argv, &rest);
    auto geti = [&](const char* k, uint64_t d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtoull(it->second.c_str(), nullptr, 10); };
    std::string mode = rest.count("mode") ? rest["mode"] : "familiar";
    uint64_t events = geti("events", 200000), fake = geti("fake_locks", 150000);
    p.train = rest.count("train") ? p.train : false;
    p.stage_timers = true;
    uint32_t vocab = 32, novel_pct = 0;
    // Untrained random bodies never die at the default thresholds, so the familiar/moderate regimes emulate a
    // learned-sparse organism with raised death thresholds; survivals are reported so nothing is hidden.
    if (mode == "familiar") { p.surprise_thresh = 1u << 30; p.death_l1 = p.death_l23 = p.death_l5 = p.death_l6 = 26; }
    else if (mode == "moderate") { p.surprise_thresh = 1u << 30; p.death_l1 = p.death_l23 = p.death_l5 = p.death_l6 = 20; novel_pct = 50; }
    else if (mode == "storm") { vocab = 256; p.structural = true; }
    else if (mode == "pathological") { p.death_l1 = p.death_l23 = p.death_l5 = p.death_l6 = 0; p.recirc_thresh = 0; p.surprise_thresh = 1u << 30; p.assoc_extra_bits = 0; }
    p.vocab = vocab;
    Organism org(p);
    std::printf("bench mode=%s units=%u (store %.0f MB) fake_locks=%llu events=%llu kernel=%d\n", mode.c_str(), p.units,
                (double)p.units * Dims::UNIT_BYTES / 1048576.0, (unsigned long long)fake, (unsigned long long)events, (int)p.kernel);
    org.seed_fake_population((uint32_t)fake, novel_pct);
    std::vector<uint8_t> stream;
    if (mode == "storm") { Rng r(p.seed + 7); stream.resize(events + 1); for (auto& b : stream) b = (uint8_t)r.below(256); }
    else { SyntheticVocab task(vocab, 8, 0.08, p.seed + 1000, 64); stream = task.generate(events + 1); }
    BlockStats agg; size_t pos = 0;
    while (pos + 1 < stream.size()) {
        size_t n = std::min<size_t>(p.block, stream.size() - 1 - pos);
        org.run_block(stream.data() + pos, n + 1); org.end_block();
        BlockStats& s = org.stats();
        std::printf("%s\n", s.report(org.block_index() - 1).c_str());
        agg.events += s.events; agg.macs += s.macs; agg.weight_bytes += s.weight_bytes; agg.seconds += s.seconds; agg.bytes += s.bytes; agg.l1_admissions += s.l1_admissions;
        for (int k = 0; k < NUM_STAGES; ++k) agg.stage_cycles[k] += s.stage_cycles[k];
        s.reset(); pos += n;
    }
    std::printf("BENCH %s: %.1f KB/s, %.0f events/s, %.0f MACs/event, %.0f weight bytes/event, %.2f admissions/event, %.1f GB/s weight traffic\n",
                mode.c_str(), agg.bytes / agg.seconds / 1024.0, agg.events / agg.seconds, (double)agg.macs / agg.events, (double)agg.weight_bytes / agg.events,
                (double)agg.l1_admissions / agg.events, agg.weight_bytes / agg.seconds / 1e9);
    return 0;
}
