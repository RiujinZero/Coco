// CPU profile of the BabyLM training hot path, as input to the GPU port (docs/GPU_PORT_LOG.md).
// Reads BYTES of the Strict-Small corpus (files interleaved in 64 KiB chunks like model_babylm) with the model_babylm
// defaults plus the given overrides, and reports throughput and the stage-timer breakdown per REPORT bytes.
//
// usage: gpu_profile BYTES REPORT [key=value ...]
#include "laminar/organism.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace laminar;

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: gpu_profile BYTES REPORT [key=value ...]\n"; return 1; }
    const size_t total = std::stoull(argv[1]), every = std::stoull(argv[2]);
    const std::string dir = std::string(std::getenv("HOME")) + "/babylm/BabyLM-2026-Strict-Small/";
    static const char* files[] = {"bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt", "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt"};
    std::vector<std::string> text; for (auto f : files) { std::ifstream in(dir + f, std::ios::binary); text.emplace_back(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()); }
    std::string stream; size_t off[6] = {0};
    for (int k = 0; stream.size() < total; k = (k + 1) % 6) { const size_t n = std::min<size_t>(65536, text[k].size() - off[k]); stream.append(text[k], off[k], n); off[k] += n; }
    stream.resize(total);

    std::vector<std::string> params = {"vocab=256", "units=20000", "hops=4", "hops_free=2", "hidden=modulated", "acc_bits=16", "agency=1", "seq_window=1048576",
        "seq_bits=16", "seq_cbits=16", "seed=1", "agency_seed=7", "episodic=1", "hist=7", "agency_ops=hold,attend_episodic,attend_generic,to_memory,to_recent"};
    for (int i = 3; i < argc; ++i) params.push_back(argv[i]);
    std::vector<char*> av; std::string prog = "gpu_profile"; av.push_back(prog.data()); for (auto& p : params) av.push_back(p.data());
    Organism m(Params::from_args((int)av.size(), av.data()));
    m.set_mode(true, true);

    uint64_t cyc[NUM_STAGES] = {0};
    const auto t0 = std::chrono::steady_clock::now(); auto tb = t0;
    for (size_t o = 0; o + 1 < stream.size(); o += every) {
        const size_t n = std::min(every + 1, stream.size() - o);
        m.stats() = BlockStats{};
        m.run_block((const uint8_t*)stream.data() + o, n);   // (the first byte of each block is the last outcome of the previous one)
        m.end_block();
        BlockStats& s = m.stats(); uint64_t c = 0; for (int k = 0; k < NUM_STAGES; ++k) { c += s.stage_cycles[k]; cyc[k] += s.stage_cycles[k]; }
        const auto tn = std::chrono::steady_clock::now(); const double sec = std::chrono::duration<double>(tn - tb).count(); tb = tn;
        std::printf("bytes %8zu  %7.0f B/s  loss %.3f bits  cand %.2f L1 %.2f L6 %.2f  |", o + n - 1, (n - 1) / sec, s.loss_sum / std::max<uint64_t>(1, s.events) / 0.6931471805599453,
                    (double)s.candidate_locks / s.events, (double)s.l1_admissions / s.events, (double)s.l6_survive / s.events);
        for (int k = 0; k < NUM_STAGES; ++k) if (c && s.stage_cycles[k] * 100 >= c) std::printf(" %s %.0f%%", STAGE_NAMES[k], 100.0 * s.stage_cycles[k] / c);
        std::printf("\n"); std::fflush(stdout);
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    uint64_t c = 0; for (int k = 0; k < NUM_STAGES; ++k) c += cyc[k];
    std::printf("TOTAL %zu bytes %.1f s = %.0f B/s |", stream.size(), sec, stream.size() / sec);
    for (int k = 0; k < NUM_STAGES; ++k) std::printf(" %s %.1f%%", STAGE_NAMES[k], c ? 100.0 * cyc[k] / c : 0.0);
    std::printf("\n");
    return 0;
}
