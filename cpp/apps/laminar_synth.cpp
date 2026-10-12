// Milestones A/B/C (runbook §14): synthetic vocabulary, online learning, then frozen evaluation.
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include <cstdio>
#include <string>

using namespace laminar;

int main(int argc, char** argv) {
    std::map<std::string, std::string> rest;
    Params p = Params::from_args(argc, argv, &rest);
    auto geti = [&](const char* k, uint64_t d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtoull(it->second.c_str(), nullptr, 10); };
    auto getf = [&](const char* k, double d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtod(it->second.c_str(), nullptr); };
    uint64_t n_train = geti("train_events", 120000), n_eval = geti("eval_events", 20000);
    uint32_t contexts = (uint32_t)geti("contexts", 8), run = (uint32_t)geti("run", 64);
    double noise = getf("noise", 0.08);
    uint64_t report_every = geti("report", p.block);

    SyntheticVocab task(p.vocab, contexts, noise, p.seed + 1000, run);
    std::vector<uint8_t> stream = task.generate(n_train + n_eval + 1);

    Organism org(p);
    if (rest.count("log")) org.open_log(rest["log"]);
    if (rest.count("trace")) org.open_trace(rest["trace"]);

    auto drive = [&](size_t begin, size_t end, const char* phase) {
        size_t pos = begin; uint64_t since = 0; BlockStats agg;
        while (pos < end) {
            size_t n = std::min<size_t>(p.block, end - pos);
            org.run_block(stream.data() + pos, n + 1);
            org.end_block();
            BlockStats& s = org.stats();
            since += s.events;
            if (since >= report_every) { std::printf("[%s] %s\n", phase, s.report(org.block_index() - 1).c_str()); std::fflush(stdout); since = 0; }
            agg.events += s.events; agg.correct += s.correct; agg.loss_sum += s.loss_sum; agg.macs += s.macs; agg.seconds += s.seconds; agg.bytes += s.bytes;
            s.reset();
            pos += n;
        }
        return agg;
    };

    std::printf("units=%u vocab=%u contexts=%u noise=%.3f train=%llu eval=%llu kernel=%d hidden=%d\n", p.units, p.vocab, contexts, noise,
                (unsigned long long)n_train, (unsigned long long)n_eval, (int)p.kernel, (int)p.hidden);
    BlockStats tr = drive(0, n_train, "train");
    std::printf("TRAIN: %llu events, acc %.2f%%, loss %.4f nat, %.1f KB/s, %.0f MACs/event\n", (unsigned long long)tr.events, 100.0 * tr.correct / tr.events,
                tr.loss_sum / tr.events, tr.bytes / tr.seconds / 1024.0, (double)tr.macs / tr.events);

    org.set_mode(false, false);
    BlockStats ev = drive(n_train, n_train + n_eval, "frozen");
    std::printf("FROZEN: %llu events, acc %.2f%% (chance %.3f%%, noise ceiling ~%.2f%%), loss %.4f nat, %.1f KB/s, %.0f MACs/event, live locks %zu, live units %zu\n",
                (unsigned long long)ev.events, 100.0 * ev.correct / ev.events, 100.0 / p.vocab, 100.0 * ((1 - noise) + noise / p.vocab),
                ev.loss_sum / ev.events, ev.bytes / ev.seconds / 1024.0, (double)ev.macs / ev.events, org.tree.live_locks(), org.live_units());
    return 0;
}
