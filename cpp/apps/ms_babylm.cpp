// BabyLM Strict-Small training with the multi-stream trainer (docs/GPU_PORT_LOG.md). Same corpus preparation as
// model_babylm (64 KiB chunks at line boundaries, last 0.5% of every file held out, chunk order shuffled per epoch with
// seed 1000 + epoch); stream s reads the epoch's chunks s, s + S, s + 2S, ... Structural maintenance every 65536 events.
// At each epoch end stream 0's state is exported into the organism and saved as DIR/epoch_K.state and
// DIR/checkpoint.state (readable by model_babylm_score, whose config.json this writes), and held-out bits per byte is
// measured in a forked child. No resume.
//
// usage: ms_babylm DIR STREAMS THREADS ref|gpu [--epochs E] [--overlap] [key=value model overrides]
#include "laminar/multistream.h"
#include "laminar/state_io.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>

using namespace laminar;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static const char* files[] = {"bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt", "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt"};
static constexpr size_t CHUNK = 64 * 1024;

int main(int argc, char** argv) {
    try {
        if (argc < 5) throw std::runtime_error("usage: ms_babylm DIR STREAMS THREADS ref|gpu [--epochs E] [key=value]");
        const fs::path dir = argv[1]; const uint32_t S = std::stoul(argv[2]); const unsigned threads = std::stoul(argv[3]);
        const LearnerKind kind = std::string(argv[4]) == "gpu" ? LearnerKind::Gpu : LearnerKind::Reference;
        uint32_t epochs = 1; bool overlap = false; std::vector<std::string> extra; std::string init;
        for (int i = 5; i < argc; ++i) { const std::string a = argv[i]; if (a == "--epochs" && i + 1 < argc) epochs = std::stoul(argv[++i]); else if (a == "--overlap") overlap = true;
            else if (a == "--init" && i + 1 < argc) init = argv[++i]; else extra.push_back(a); }   // --init STATE: start from a saved model (same configuration)
        std::vector<std::string> params = {"vocab=256", "units=20000", "hops=4", "hops_free=2", "hidden=modulated", "acc_bits=16", "agency=1", "seq_window=1048576",
            "seq_bits=16", "seq_cbits=16", "seed=1", "agency_seed=7", "episodic=1", "hist=7", "agency_ops=hold,attend_episodic,attend_generic,to_memory,to_recent",
            "body_err=2", "err_gain=2", "err_sparse=0", "in_struct=1", "hidden=backprop", "recirc=128", "seq_gate=1", "seq_linear=1", "stoch_round=1", "hops=0", "bp_leak=0.1",
            "adam=1", "ro_adam=1", "tc_window=64", "ws_regs=2", "reg_bias=-3", "reg_readout=1"};
        params.insert(params.end(), extra.begin(), extra.end());

        const std::string corpus_dir = std::string(std::getenv("HOME")) + "/babylm/BabyLM-2026-Strict-Small";
        std::vector<std::string> chunks; std::vector<std::vector<float>> chunk_w; std::string heldout;   // chunk_w: per-byte weights (MS_EXTRA_W; empty = 1)
        for (const char* f : files) {   // model_babylm's preparation
            std::ifstream in(corpus_dir + "/" + f, std::ios::binary); if (!in) throw std::runtime_error(std::string("missing ") + f);
            std::string text((std::istreambuf_iterator<char>(in)), {});
            size_t cut = text.size() - text.size() / 200; cut = text.rfind('\n', cut); if (cut == std::string::npos) cut = text.size();
            heldout += text.substr(cut + 1 <= text.size() ? cut + 1 : text.size()); text.resize(cut + 1 <= text.size() ? cut + 1 : text.size());
            for (size_t o = 0; o < text.size();) {
                size_t e = std::min(text.size(), o + CHUNK); if (e < text.size()) { size_t nl = text.rfind('\n', e); if (nl != std::string::npos && nl > o) e = nl + 1; }
                chunks.push_back(text.substr(o, e - o)); o = e;
            }
        }
        if (heldout.size() > (2u << 20)) heldout.resize(2u << 20);
        if (const char* hb = std::getenv("MS_HELDOUT_BYTES")) heldout.resize(std::min(heldout.size(), (size_t)std::stoull(hb)));   // a shorter held-out (slow models)
        if (const char* sub = std::getenv("MS_SUBSET")) { const size_t k = std::max(1, std::atoi(sub));   // a shorter epoch: every k-th BabyLM chunk
            std::vector<std::string> kept; for (size_t i = 0; i < chunks.size(); i += k) kept.push_back(chunks[i]); chunks.swap(kept); }
        if (const char* ex = std::getenv("MS_EXTRA")) {   // extra training text (e.g. selected generations: the prompt-following objective),
            std::ifstream in(ex, std::ios::binary); if (!in) throw std::runtime_error(std::string("missing MS_EXTRA ") + ex);   // repeated MS_EXTRA_REP times
            std::string text((std::istreambuf_iterator<char>(in)), {}); const int rep = std::getenv("MS_EXTRA_REP") ? std::max(1, std::atoi(std::getenv("MS_EXTRA_REP"))) : 1;
            std::vector<float> tw;   // MS_EXTRA_W: per record "PLEN TLEN W" (bytes) -> the prompt's bytes weight 1, the rest W
            if (const char* wf = std::getenv("MS_EXTRA_W"); wf && *wf) { std::ifstream wi(wf); if (!wi) throw std::runtime_error(std::string("missing MS_EXTRA_W ") + wf);
                tw.assign(text.size(), 1.f); size_t o = 0, plen, tlen; float w;
                while (o < text.size() && (wi >> plen >> tlen >> w)) { const size_t e = std::min(text.size(), o + tlen); for (size_t i = std::min(e, o + plen); i < e; ++i) tw[i] = w; o = e; }
                if (o != text.size()) throw std::runtime_error("MS_EXTRA_W does not cover MS_EXTRA exactly"); }
            chunk_w.resize(chunks.size());
            for (int r = 0; r < rep; ++r) for (size_t o = 0; o < text.size();) { size_t e = std::min(text.size(), o + CHUNK);
                if (e < text.size()) { size_t nl = text.rfind('\n', e); if (nl != std::string::npos && nl > o) e = nl + 1; } chunks.push_back(text.substr(o, e - o));
                chunk_w.push_back(tw.empty() ? std::vector<float>{} : std::vector<float>(tw.begin() + (long)o, tw.begin() + (long)e)); o = e; } }
        chunk_w.resize(chunks.size());

        fs::create_directories(dir);
        if (fs::exists(dir / "config.json")) throw std::runtime_error("run directory exists: refusing to overwrite");
        { std::ostringstream decl; decl << "{\"schema\":\"ms_babylm_v1\",\"profile\":\"strict-small\",\"streams\":" << S << ",\"learner\":\"" << argv[4] << "\",\"overlap\":" << (overlap ? "true" : "false") << ",\"model_params\":[";
          for (size_t i = 0; i < params.size(); ++i) decl << (i ? "," : "") << '"' << params[i] << '"';
          decl << "],\"epochs\":" << epochs << ",\"chunks\":" << chunks.size() << "}\n"; std::ofstream(dir / "config.json") << decl.str(); }

        std::vector<char*> av; std::string prog = "ms_babylm"; av.push_back(prog.data()); for (auto& p : params) av.push_back(p.data());
        MultiStream ms(Params::from_args((int)av.size(), av.data()), S, kind, threads); ms.set_train(true, true); ms.set_overlap(overlap);
        if (!init.empty()) { { state::Reader r(init); ms.organism().load_state(r); } ms.set_train(true, true); ms.reload_from_container();
            std::ofstream(dir / "init.txt") << init << "\n"; std::cout << "initialised from " << init << "\n"; }
        std::ofstream log(dir / "progress.jsonl", std::ios::app);
        std::cout << S << " streams, " << threads << " threads, learner " << ms.learner_name() << ", " << chunks.size() << " chunks\n" << std::flush;

        auto heldout_bpb = [&]() -> double {   // forked child: the parent's model is untouched
            int fds[2]; if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
            std::cout.flush(); pid_t pid = fork(); if (pid < 0) throw std::runtime_error("fork failed");
            if (pid == 0) {
                close(fds[0]); Organism& m = ms.organism(); m.set_mode(false, false); m.stats() = BlockStats{};
                std::vector<uint8_t> buf(heldout.size() + 1); buf[0] = m.last_symbol(); std::memcpy(buf.data() + 1, heldout.data(), heldout.size()); m.run_block(buf.data(), buf.size());
                double v = m.stats().loss_sum / std::max<uint64_t>(1, m.stats().events) / 0.6931471805599453;
                ssize_t w = write(fds[1], &v, sizeof v); (void)w; _exit(0);
            }
            close(fds[1]); double v = 0; ssize_t r = read(fds[0], &v, sizeof v); close(fds[0]); int st = 0; waitpid(pid, &st, 0);
            if (r != (ssize_t)sizeof v) throw std::runtime_error("held-out scoring failed");
            return v;
        };
        const auto start = Clock::now();
        for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
            std::vector<uint32_t> order(chunks.size()); for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
            std::mt19937_64 sh(1000 + epoch); std::shuffle(order.begin(), order.end(), sh);
            std::vector<std::string> per(S); for (size_t c = 0; c < order.size(); ++c) per[c % S] += chunks[order[c]];
            const bool weighted = std::any_of(chunk_w.begin(), chunk_w.end(), [](const std::vector<float>& w) { return !w.empty(); });
            std::vector<std::vector<float>> per_w(weighted ? S : 0);
            if (weighted) for (size_t c = 0; c < order.size(); ++c) { const std::vector<float>& w = chunk_w[order[c]]; std::vector<float>& d = per_w[c % S];
                if (w.empty()) d.insert(d.end(), chunks[order[c]].size(), 1.f); else d.insert(d.end(), w.begin(), w.end()); }
            for (uint32_t s = 0; s < S; ++s) { ms.set_stream_data(s, (const uint8_t*)per[s].data(), per[s].size()); if (weighted) ms.set_stream_weights(s, per_w[s].data()); }
            uint64_t next_block = ms.events() + 65536, next_report = ms.events() + (4u << 20); auto tr = Clock::now(); uint64_t ev_r = ms.events();
            while (ms.any_active()) {
                ms.step();
                if (ms.events() >= next_block) { ms.end_block(); next_block += 65536; }
                if (ms.events() >= next_report) {
                    auto& st = ms.stats(); const double sec = std::chrono::duration<double>(Clock::now() - tr).count();
                    std::cout << "epoch " << epoch << " events " << ms.events() << " loss " << st.loss_sum / std::max<uint64_t>(1, st.events) / 0.6931471805599453
                              << " bits/byte, " << (uint64_t)((ms.events() - ev_r) / sec) << " B/s, live " << ms.organism().live_units() << '\n' << std::flush;
                    log << "{\"event\":\"progress\",\"epoch\":" << epoch << ",\"events\":" << ms.events() << ",\"train_bpb\":" << st.loss_sum / std::max<uint64_t>(1, st.events) / 0.6931471805599453
                        << ",\"bytes_per_s\":" << (ms.events() - ev_r) / sec << "}\n" << std::flush;
                    ms.stats() = MultiStream::Stats{}; tr = Clock::now(); ev_r = ms.events(); next_report += 4u << 20;
                }
            }
            ms.export_stream(0);
            for (const fs::path to : {dir / ("epoch_" + std::to_string(epoch + 1) + ".state"), dir / "checkpoint.state"}) {
                const fs::path tmp = to.string() + ".tmp"; { state::Writer w(tmp.string()); ms.organism().save_state(w); w.finish(); } fs::rename(tmp, to); }
            const double bpb = heldout_bpb(), sec = std::chrono::duration<double>(Clock::now() - start).count();
            std::cout << "epoch " << epoch + 1 << " done after " << sec << " s, heldout " << bpb << " bits/byte, live " << ms.organism().live_units() << '\n' << std::flush;
            log << "{\"event\":\"epoch\",\"epoch\":" << epoch + 1 << ",\"events\":" << ms.events() << ",\"heldout_bpb\":" << bpb << ",\"seconds\":" << sec << "}\n" << std::flush;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
