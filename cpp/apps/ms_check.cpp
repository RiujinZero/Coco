// Checks and speed test of the multi-stream trainer and its GPU learner (docs/GPU_PORT_LOG.md).
//
//   ms_check exact BYTES [key=value ...]                 one stream, CPU reference learner, against Organism: every event's
//                                                       loss and prediction bit-identical, weights identical after every block
//   ms_check dual BYTES STREAMS THREADS [key=value ...]  S streams, the GPU learner checked step by step against the reference
//   ms_check speed BYTES STREAMS THREADS ref|gpu [key=value ...]   throughput and phase times
//
// Data: the BabyLM Strict-Small training files interleaved in 64 KiB chunks (as model_babylm does, unshuffled); with S
// streams, stream s reads chunks s, s + S, s + 2S, ... End of block (structural maintenance) every 65536 events.
// Model parameters: model_babylm's defaults, then the BabyLM Adam + registers configuration, then the overrides.
#include "laminar/multistream.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace laminar;
using Clock = std::chrono::steady_clock;

static std::vector<std::string> corpus_chunks(size_t total) {
    const std::string dir = std::string(std::getenv("HOME")) + "/babylm/BabyLM-2026-Strict-Small/";
    static const char* files[] = {"bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt", "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt"};
    std::vector<std::string> text; for (auto f : files) { std::ifstream in(dir + f, std::ios::binary); if (!in) throw std::runtime_error(std::string("missing ") + dir + f); text.emplace_back(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()); }
    std::vector<std::string> chunks; size_t got = 0, off[6] = {0};
    for (int k = 0; got < total; k = (k + 1) % 6) {
        if (off[k] >= text[k].size()) continue;
        const size_t n = std::min<size_t>({65536, text[k].size() - off[k], total - got});
        chunks.push_back(text[k].substr(off[k], n)); off[k] += n; got += n;
    }
    return chunks;
}
static Params make_params(int argc, char** argv, int first) {
    std::vector<std::string> ps = {"vocab=256", "units=20000", "hops=4", "hops_free=2", "hidden=modulated", "acc_bits=16", "agency=1", "seq_window=1048576",
        "seq_bits=16", "seq_cbits=16", "seed=1", "agency_seed=7", "episodic=1", "hist=7", "agency_ops=hold,attend_episodic,attend_generic,to_memory,to_recent",
        "body_err=2", "err_gain=2", "err_sparse=0", "in_struct=1", "hidden=backprop", "recirc=128", "seq_gate=1", "seq_linear=1", "stoch_round=1", "hops=0", "bp_leak=0.1",
        "adam=1", "ro_adam=1", "tc_window=64", "ws_regs=2", "reg_bias=-3", "reg_readout=1"};
    for (int i = first; i < argc; ++i) ps.push_back(argv[i]);
    std::vector<char*> av; std::string prog = "ms_check"; av.push_back(prog.data()); for (auto& p : ps) av.push_back(p.data());
    return Params::from_args((int)av.size(), av.data());
}
static void stream_data(const std::vector<std::string>& chunks, uint32_t S, std::vector<std::string>& per) {
    per.assign(S, {}); for (size_t c = 0; c < chunks.size(); ++c) per[c % S] += chunks[c];
}
static constexpr uint64_t BLOCK = 65536;

static int run_exact(size_t bytes, const Params& p) {
    const auto chunks = corpus_chunks(bytes);
    std::string all; for (auto& c : chunks) all += c;
    Organism ref(p); ref.set_mode(true, true);
    MultiStream ms(p, 1, LearnerKind::Reference, 1); ms.set_train(true, true);
    std::vector<float> lr, lm; std::vector<uint32_t> pr, pm;
    ref.on_event = [&](uint8_t, uint8_t target, const int32_t*, const float* probs) { lr.push_back(-std::log(std::max(probs[target], 1e-9f)));
        uint32_t best = 0; for (uint32_t v = 1; v < 256; ++v) if (probs[v] > probs[best]) best = v; pr.push_back(best); };
    ms.on_event = [&](uint32_t, uint8_t, uint8_t, float loss, uint32_t pred) { lm.push_back(loss); pm.push_back(pred); };
    uint8_t last = 0; double sref = 0, sms = 0; uint64_t ev = 0;
    for (size_t o = 0; o < all.size(); o += BLOCK) {
        const size_t n = std::min<size_t>(BLOCK, all.size() - o);
        std::vector<uint8_t> buf(n + 1); buf[0] = last; std::memcpy(buf.data() + 1, all.data() + o, n); last = buf[n];
        auto t0 = Clock::now(); ref.run_block(buf.data(), buf.size()); ref.end_block(); ref.stats() = BlockStats{}; sref += std::chrono::duration<double>(Clock::now() - t0).count();
        t0 = Clock::now(); ms.set_stream_data(0, (const uint8_t*)all.data() + o, n); while (ms.any_active()) ms.step(); ms.end_block(); sms += std::chrono::duration<double>(Clock::now() - t0).count();
        if (lr.size() != lm.size()) { std::printf("FAIL: event counts differ (%zu vs %zu)\n", lr.size(), lm.size()); return 1; }
        for (size_t i = 0; i < lr.size(); ++i) {
            uint32_t a, b; std::memcpy(&a, &lr[i], 4); std::memcpy(&b, &lm[i], 4);
            if (a != b || pr[i] != pm[i]) { std::printf("FAIL: first divergence at event %llu: loss %.9g vs %.9g, prediction %u vs %u\n", (unsigned long long)(ev + i), lr[i], lm[i], pr[i], pm[i]); return 1; }
        }
        double loss = 0; for (float x : lr) loss += x;
        ev += lr.size(); lr.clear(); lm.clear(); pr.clear(); pm.clear();
        const bool wsame = std::memcmp(ref.weights(), ms.organism().weights(), (size_t)p.units * Dims::UNIT_BYTES) == 0;
        std::printf("block %3zu: %llu events identical, loss %.4f bits/byte, live units %zu vs %zu, weights %s  (%.1f s / %.1f s)\n", o / BLOCK, (unsigned long long)ev,
                    loss / n / 0.6931471805599453, ref.live_units(), ms.organism().live_units(), wsame ? "identical" : "DIFFER", sref, sms);
        std::fflush(stdout);
        if (!wsame || ref.live_units() != ms.organism().live_units()) { std::printf("FAIL: state differs after block %zu\n", o / BLOCK); return 1; }
    }
    std::printf("PASS: %llu events bit-identical to Organism (loss, prediction, weights, population)\n", (unsigned long long)ev);
    return 0;
}

static void report_line(MultiStream& ms, double sec, uint64_t ev0) {
    auto& s = ms.stats(); const double e = (double)std::max<uint64_t>(1, s.events);
    std::printf("events %llu  %.0f B/s  loss %.4f bits  acc %.2f%% | per step: P1 %.0f S1 %.0f P2 %.0f S2 %.0f P3 %.0f S3 %.0f (reduce %.0f ro %.0f build %.0f learner %.0f ring %.0f) S4 %.0f P4 %.0f us, end_block %.0f upload %.0f us/step, wall %.0f us/step | jobs/step %.1f direct %.1f temporal, adam steps %.1f/step, uses per step %.2f\n",
                (unsigned long long)ms.events(), (ms.events() - ev0) / sec, s.loss_sum / e / 0.6931471805599453, 100.0 * s.correct / e,
                1e6 * s.sec_p1 / s.steps, 1e6 * s.sec_s1 / s.steps, 1e6 * s.sec_p2 / s.steps, 1e6 * s.sec_s2 / s.steps, 1e6 * s.sec_p3 / s.steps, 1e6 * s.sec_s3 / s.steps,
                1e6 * s.s3_reduce / s.steps, 1e6 * s.s3_ro / s.steps, 1e6 * s.s3_build / s.steps, 1e6 * s.sec_learn / s.steps, 1e6 * s.s3_ring / s.steps, 1e6 * s.sec_s4 / s.steps, 1e6 * s.sec_p4 / s.steps, 1e6 * s.sec_block / s.steps, 1e6 * s.sec_upload / s.steps, 1e6 * sec / s.steps, (double)s.direct_jobs / s.steps, (double)s.temporal_jobs / s.steps,
                (double)s.adam_steps / s.steps, s.adam_steps ? (double)s.adam_uses / s.adam_steps : 0.0);
    std::fflush(stdout);
}

static int run_streams(size_t bytes, uint32_t S, unsigned threads, LearnerKind kind, const Params& p, bool dual) {
    const auto chunks = corpus_chunks(bytes);
    if (chunks.size() < S) throw std::runtime_error("fewer 64 KiB chunks than streams: give at least STREAMS x 64 KiB");
    std::vector<std::string> per; stream_data(chunks, S, per);
    MultiStream ms(p, S, kind, threads); ms.set_train(true, true); ms.set_overlap(std::getenv("MS_OVERLAP") && std::string(std::getenv("MS_OVERLAP")) == "1");
    std::printf("%u streams, %u threads, learner %s, %zu bytes\n", S, threads, ms.learner_name().c_str(), bytes); std::fflush(stdout);
    for (uint32_t s = 0; s < S; ++s) ms.set_stream_data(s, (const uint8_t*)per[s].data(), per[s].size());
    const auto t0 = Clock::now(); auto tb = t0; uint64_t next_block = BLOCK, ev_b = 0;
    while (ms.any_active()) {
        ms.step();
        if (ms.events() >= next_block) {
            ms.end_block(); next_block += BLOCK;
            if ((next_block / BLOCK) % 8 == 0) { const double sec = std::chrono::duration<double>(Clock::now() - tb).count(); report_line(ms, sec, ev_b); if (dual) std::printf("  %s\n", ms.learner().report().c_str());
                ms.stats() = MultiStream::Stats{}; tb = Clock::now(); ev_b = ms.events(); }
        }
    }
    ms.sync();
    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    if (ms.stats().steps) report_line(ms, std::chrono::duration<double>(Clock::now() - tb).count(), ev_b);
    std::printf("TOTAL %llu events in %.1f s = %.0f bytes/s (%u streams, learner %s)\n", (unsigned long long)ms.events(), sec, ms.events() / sec, S, ms.learner_name().c_str());
    std::printf("%s\n", ms.learner().report().c_str());
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) { std::cerr << "usage: ms_check exact BYTES | dual BYTES STREAMS THREADS | speed BYTES STREAMS THREADS ref|gpu [key=value ...]\n"; return 2; }
        const std::string mode = argv[1]; const size_t bytes = std::stoull(argv[2]);
        if (mode == "exact") return run_exact(bytes, make_params(argc, argv, 3));
        if (mode == "dual") return run_streams(bytes, std::stoul(argv[3]), std::stoul(argv[4]), LearnerKind::Dual, make_params(argc, argv, 5), true);
        if (mode == "speed") return run_streams(bytes, std::stoul(argv[3]), std::stoul(argv[4]), std::string(argv[5]) == "gpu" ? LearnerKind::Gpu : LearnerKind::Reference, make_params(argc, argv, 6), false);
        std::cerr << "unknown mode\n"; return 2;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
