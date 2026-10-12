// Milestone D (runbook §14, design doc §15): raw-byte online next-byte prediction.
//   mode=adaptive  adaptive locks + learning (default)
//   mode=frozen    locks frozen after `warm` bytes (weights keep learning)
//   mode=nomem     routing/memory disabled: no neighbour routing, no L4 recirculation, sensory-only locks
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include <cstdio>
#include <string>

using namespace laminar;

int main(int argc, char** argv) {
    std::map<std::string, std::string> rest;
    Params p = Params::from_args(argc, argv, &rest);
    p.vocab = 256;
    if (!rest.count("file")) { std::fprintf(stderr, "usage: laminar_bytes file=PATH [max_bytes=N] [mode=adaptive|frozen|nomem] [warm=N] [epochs=N] [log=PATH] ...params\n"); return 1; }
    auto geti = [&](const char* k, uint64_t d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtoull(it->second.c_str(), nullptr, 10); };
    std::string mode = rest.count("mode") ? rest["mode"] : "adaptive";
    uint64_t warm = geti("warm", 16384), epochs = geti("epochs", 1);
    std::vector<uint8_t> data = read_file_bytes(rest["file"], geti("max_bytes", 0));
    if (data.size() < 3) { std::fprintf(stderr, "could not read %s\n", rest["file"].c_str()); return 1; }
    if (p.future_buckets == 32) p.future_buckets = 64;   // finer futures for 256-symbol streams
    if (mode == "nomem") { p.max_hops = 0; p.recirc_thresh = 1 << 20; p.structural = true; p.split_min_evidence = UINT32_MAX; }
    Organism org(p);
    if (rest.count("log")) org.open_log(rest["log"]);
    std::printf("file=%s bytes=%zu mode=%s units=%u epochs=%llu\n", rest["file"].c_str(), data.size(), mode.c_str(), p.units, (unsigned long long)epochs);

    const bool quiet = rest.count("quiet") > 0;
    const uint64_t progress = geti("progress", 0);       // print a windowed BPB line every N bytes
    uint64_t seen = 0; BlockStats agg; BlockStats win; uint64_t win_next = progress;
    for (uint64_t e = 0; e < epochs; ++e) {
        size_t pos = 0; BlockStats ep;
        while (pos + 1 < data.size()) {
            size_t n = std::min<size_t>(p.block, data.size() - 1 - pos);
            if (mode == "frozen" && seen >= warm && p.structural) { org.set_mode(true, false); p.structural = false; std::printf("-- locks frozen at %llu bytes\n", (unsigned long long)seen); }
            org.run_block(data.data() + pos, n + 1); org.end_block();
            BlockStats& s = org.stats(); seen += s.events;
            if (!quiet) { std::printf("[%s e%llu] %s\n", mode.c_str(), (unsigned long long)e, s.report(org.block_index() - 1).c_str()); std::fflush(stdout); }
            if (progress) {
                win.events += s.events; win.loss_sum += s.loss_sum; win.correct += s.correct; win.macs += s.macs; win.seconds += s.seconds; win.bytes += s.bytes; win.episodic_births += s.episodic_births; win.lock_splits += s.lock_splits + s.nursery_admissions;
                if (seen >= win_next) { win_next += progress;
                    std::printf("  %6.2f MB | window BPB %.4f acc %.1f%% | %.0f KB/s | MACs/byte %.0f | splits %llu epi births %llu | locks %zu units %zu\n", seen / 1048576.0,
                                win.loss_sum / win.events / 0.6931471805599453, 100.0 * win.correct / win.events, win.bytes / win.seconds / 1024.0, (double)win.macs / win.events,
                                (unsigned long long)win.lock_splits, (unsigned long long)win.episodic_births, org.tree.live_locks(), org.live_units());
                    if (p.agency) std::printf("  %s\n", org.agency_report().c_str());
                    std::fflush(stdout); win.reset(); }
            }
            for (BlockStats* a : {&agg, &ep}) {
                a->events += s.events; a->loss_sum += s.loss_sum; a->correct += s.correct; a->macs += s.macs; a->seconds += s.seconds; a->bytes += s.bytes;
                a->l1_admissions += s.l1_admissions; a->l6_survive += s.l6_survive; a->neighbor_accepts += s.neighbor_accepts; a->novel_events += s.novel_events;
                a->lock_births += s.lock_births; a->lock_splits += s.lock_splits; a->level_changes += s.level_changes;
                a->nursery_admissions += s.nursery_admissions; a->lock_recycled += s.lock_recycled;
                a->queue_sizes.insert(a->queue_sizes.end(), s.queue_sizes.begin(), s.queue_sizes.end());
            }
            s.reset(); pos += n;
        }
        double ev = (double)ep.events;
        std::printf("EPOCH %llu [%s]: BPB %.4f acc %.2f%% | %.1f KB/s | MACs/byte %.0f adm %.2f emit %.2f nbr %.3f qmax %u | novel %.3f lvl %.1f | births %llu splits %llu admits %llu recycled %llu | locks %zu units %zu\n",
                    (unsigned long long)e, mode.c_str(), ep.loss_sum / ev / 0.6931471805599453, 100.0 * ep.correct / ev, ep.bytes / ep.seconds / 1024.0,
                    ep.macs / ev, ep.l1_admissions / ev, ep.l6_survive / ev, ep.neighbor_accepts / ev, ep.queue_sizes.empty() ? 0u : *std::max_element(ep.queue_sizes.begin(), ep.queue_sizes.end()),
                    ep.novel_events / ev, ep.level_changes / ev, (unsigned long long)ep.lock_births, (unsigned long long)ep.lock_splits, (unsigned long long)ep.nursery_admissions, (unsigned long long)ep.lock_recycled, org.tree.live_locks(), org.live_units());
        if (p.agency) std::printf("  %s\n", org.agency_report().c_str());
        std::printf("  %s\n", org.kclass_report().c_str());
        { std::string c = org.composer_report(); if (!c.empty()) std::printf("  %s\n", c.c_str()); }
        std::fflush(stdout);
    }
    std::printf("TOTAL: %llu bytes, BPB %.4f, acc %.2f%%, %.1f KB/s, %.0f MACs/byte, locks %zu units %zu\n", (unsigned long long)agg.events,
                agg.loss_sum / agg.events / 0.6931471805599453, 100.0 * agg.correct / agg.events, agg.bytes / agg.seconds / 1024.0, (double)agg.macs / agg.events,
                org.tree.live_locks(), org.live_units());
    return 0;
}
