// BabyLM training: the model reads the BabyLM 2026 corpus (Strict 100M words or Strict-Small 10M words) with its
// ordinary byte-prediction learning, under the challenge's rule of at most 10 epochs.
//
// Data: the six corpus files are cut into ~64 KiB chunks at line boundaries and interleaved in a fixed shuffled order
// per epoch (seed = epoch), so no single domain fills a long stretch of the stream. The last 0.5% of every file (cut at
// a line) is held out for bits-per-byte tracking and never trained on.
// Checkpoints every 16 MiB of stream (current + previous) and a copy at the end of every epoch (epoch_K.state) for the
// challenge's intermediate-checkpoint evaluations. Held-out scoring runs in a forked child, so measuring never changes
// the model. An existing DIR resumes exactly.
//
// usage: model_babylm strict|strict-small|smoke DIR [--epochs E] [--units N] [--score ITEMS_DIR] [--workers W]
//                     [--no-checkpoint] [key=value model overrides]
//   --score: at every epoch end, score ITEMS_DIR/{blimp,entity_tracking,...}.jsonl in-process (scored_<task>_e<K>.jsonl).
//            A forked scoring child primes itself with a newline and forks one grandchild per candidate, exactly as
//            model_babylm_score does from a checkpoint; the training process is never touched.
//   --no-checkpoint: for configurations a checkpoint cannot cover yet (composer levels); no resume.
//   --gate-tol T: the body market's counterfactual gate accepts a cull batch when retention loss rises by at most T (default 0).
#include "laminar/organism.h"
#include "laminar/state_io.h"
#include "answer_tasks.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>

using namespace laminar;
// Feeds bytes as a continuation of the model's own stream (the last symbol, then the bytes), through the public run_block.
static void feed_bytes(laminar::Organism& m, const unsigned char* p, size_t n) {
    std::vector<uint8_t> buf(n + 1); buf[0] = m.last_symbol(); std::memcpy(buf.data() + 1, p, n); m.run_block(buf.data(), buf.size());
}
static void feed_str(laminar::Organism& m, const std::string& s) { feed_bytes(m, (const unsigned char*)s.data(), s.size()); }
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static const char* files[] = {"bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt", "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt"};
static constexpr size_t CHUNK = 64 * 1024, CKPT_BYTES = 16u << 20;

struct Run {
    uint64_t epoch = 0, chunk = 0, bytes_total = 0, words_total = 0;
    double best_bpb = 1e9;   // retention controller: best smoothed held-out bits per byte so far
    double ema_bpb = -1;       // smoothed held-out signal (EMA 0.5) so checkpoint noise does not drive the controller
    uint64_t last_bodies = 0;  // market population at the previous check (parsimony)
    uint64_t gate_calls = 0;   // seeds the gate's retention sample (deterministic across resume)
    template <class Ar> void state_io(Ar& ar) { ar(epoch); ar(chunk); ar(bytes_total); ar(words_total); ar(best_bpb); ar(ema_bpb); ar(last_bodies); ar(gate_calls); }
};

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: model_babylm strict|strict-small|smoke DIR [--epochs E] [--units N] [key=value]");
        const std::string profile = argv[1]; const fs::path dir = argv[2];
        std::string corpus_dir = std::string(std::getenv("HOME")) + "/babylm/" + (profile == "strict" ? "BabyLM-2026-Strict" : "BabyLM-2026-Strict-Small");
        uint32_t units = 20000, epochs = profile == "smoke" ? 1 : 10; size_t smoke_bytes = profile == "smoke" ? (4u << 20) : 0;
        std::string score_dir; int workers = 24; bool checkpoints = true; double gate_tol = 0.0; bool curriculum = false;
        if (profile != "strict" && profile != "strict-small" && profile != "smoke") throw std::runtime_error("unknown profile");
        std::vector<std::string> extra;
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--epochs" && i + 1 < argc) epochs = std::stoul(argv[++i]);
            else if (a == "--units" && i + 1 < argc) units = std::stoul(argv[++i]);
            else if (a == "--score" && i + 1 < argc) score_dir = argv[++i];
            else if (a == "--workers" && i + 1 < argc) workers = std::stoi(argv[++i]);
            else if (a == "--no-checkpoint") checkpoints = false;
            else if (a == "--gate-tol" && i + 1 < argc) gate_tol = std::stod(argv[++i]);
            else if (a == "--curriculum") curriculum = true;
            else if (a.find('=') != std::string::npos) extra.push_back(a);
            else throw std::runtime_error("unknown option " + a);
        }
        if (epochs > 10) throw std::runtime_error("BabyLM allows at most 10 epochs");
        std::vector<std::string> params = {"vocab=256", "units=" + std::to_string(units), "hops=4", "hops_free=2", "hidden=modulated", "acc_bits=16",
            "agency=1", "seq_window=1048576", "seq_bits=16", "seq_cbits=16", "seed=1", "agency_seed=7", "episodic=1", "hist=7",
            "agency_ops=hold,attend_episodic,attend_generic,to_memory,to_recent"};
        params.insert(params.end(), extra.begin(), extra.end());

        // corpus: chunks at line boundaries; the last 0.5% of each file held out
        std::vector<uint8_t> chunk_file; std::vector<std::string> chunks; std::string heldout; uint64_t corpus_words = 0;
        for (size_t fi = 0; fi < sizeof files / sizeof files[0]; ++fi) { const char* f = files[fi];
            std::ifstream in(corpus_dir + "/" + f, std::ios::binary); if (!in) throw std::runtime_error(std::string("missing ") + f);
            std::string text((std::istreambuf_iterator<char>(in)), {});
            if (smoke_bytes) text.resize(std::min(text.size(), smoke_bytes / 6));
            size_t cut = text.size() - text.size() / 200; cut = text.rfind('\n', cut); if (cut == std::string::npos) cut = text.size();
            heldout += text.substr(cut + 1 <= text.size() ? cut + 1 : text.size()); text.resize(cut + 1 <= text.size() ? cut + 1 : text.size());
            for (size_t o = 0; o < text.size();) {
                size_t e = std::min(text.size(), o + CHUNK); if (e < text.size()) { size_t nl = text.rfind('\n', e); if (nl != std::string::npos && nl > o) e = nl + 1; }
                chunks.push_back(text.substr(o, e - o)); chunk_file.push_back((uint8_t)fi); o = e;
            }
            bool in_word = false; for (unsigned char c : text) { const bool w = c > ' '; corpus_words += w && !in_word; in_word = w; }
        }
        if (heldout.size() > (2u << 20)) heldout.resize(2u << 20);   // bpb tracking on at most 2 MiB

        std::ostringstream decl;
        decl << "{\"schema\":\"model_babylm_v1\",\"profile\":\"" << profile << "\",\"corpus\":\"" << corpus_dir << "\",\"model_params\":[";
        for (size_t i = 0; i < params.size(); ++i) decl << (i ? "," : "") << '"' << params[i] << '"';
        decl << "],\"epochs\":" << epochs << ",\"chunks\":" << chunks.size() << ",\"chunk_bytes\":" << CHUNK << ",\"train_words\":" << corpus_words
             << ",\"heldout_bytes\":" << heldout.size() << ",\"order\":\"chunks shuffled per epoch (seed = epoch)\",\"heldout\":\"last 0.5% of every file\"}\n";
        fs::create_directories(dir);
        const bool resume = checkpoints && fs::exists(dir / "checkpoint.state");
        if (!checkpoints && fs::exists(dir / "config.json")) throw std::runtime_error("--no-checkpoint needs a fresh directory");
        if (!resume) { if (fs::exists(dir / "config.json")) throw std::runtime_error("declared run has no checkpoint: refusing to overwrite"); std::ofstream(dir / "config.json") << decl.str(); }
        else { std::ifstream f(dir / "config.json"); if (std::string((std::istreambuf_iterator<char>(f)), {}) != decl.str()) throw std::runtime_error("declared configuration changed"); }

        std::vector<char*> av; std::string prog = "babylm"; av.push_back(prog.data()); for (auto& p : params) av.push_back(p.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        Run rs;
        auto save = [&](const fs::path& to) {
            if (!checkpoints) return;
            const fs::path tmp = to.string() + ".tmp";
            { state::Writer w(tmp.string()); model.save_state(w); w.section("run"); rs.state_io(w); w.finish(); }
            if (to == dir / "checkpoint.state" && fs::exists(to)) fs::rename(to, dir / "checkpoint.prev.state");
            fs::rename(tmp, to);
        };
        auto load = [&](const fs::path& from) { state::Reader r(from.string()); model.load_state(r); r.section("run"); rs.state_io(r); r.finish(); };
        if (resume) { try { load(dir / "checkpoint.state"); } catch (const std::exception& e) { std::cout << "checkpoint unreadable (" << e.what() << "), using previous\n"; load(dir / "checkpoint.prev.state"); } }
        std::ofstream log(dir / "progress.jsonl", std::ios::app); log << std::setprecision(6);
        // held-out bits per byte, measured in a forked child (the parent's model is untouched)
        auto heldout_bpb = [&]() -> double {
            int fds[2]; if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
            std::cout.flush(); pid_t pid = fork(); if (pid < 0) throw std::runtime_error("fork failed");
            if (pid == 0) {
                close(fds[0]); model.set_mode(false, false); model.stats() = BlockStats{};
                feed_str(model, heldout);
                double v = model.stats().loss_sum / std::max<uint64_t>(1, model.stats().events) / 0.6931471805599453;
                ssize_t w = write(fds[1], &v, sizeof v); (void)w; _exit(0);
            }
            close(fds[1]); double v = 0; ssize_t r = read(fds[0], &v, sizeof v); close(fds[0]); int st = 0; waitpid(pid, &st, 0);
            if (r != (ssize_t)sizeof v) throw std::runtime_error("held-out scoring failed");
            return v;
        };
        // Zero-shot scoring of every ITEMS_DIR/*.jsonl from the live model, without touching it (see usage).
        auto score_items = [&](const std::string& items_dir, const fs::path& out_dir, uint64_t epoch) {
            std::cout.flush(); std::cerr.flush();
            pid_t master = fork(); if (master < 0) throw std::runtime_error("fork failed");
            if (master == 0) {
                try {
                    model.set_mode(false, false); feed_str(model, "\n");
                    for (const auto& entry : fs::directory_iterator(items_dir)) {
                        if (entry.path().extension() != ".jsonl") continue;
                        struct C { std::string line, context, completion; }; std::vector<C> cands;
                        { std::ifstream in(entry.path()); std::string line; while (std::getline(in, line)) if (!line.empty()) cands.push_back({line, laminar::answers::field(line, "context"), laminar::answers::field(line, "completion")}); }
                        std::ofstream out(out_dir / ("scored_" + entry.path().stem().string() + "_e" + std::to_string(epoch) + ".jsonl"));
                        struct Job { pid_t pid; int fd; size_t idx; }; std::vector<Job> running; size_t next = 0;
                        auto reap = [&](Job j) {
                            double v[2] = {0, 0}; ssize_t got = 0; char* q = (char*)v;
                            while (got < (ssize_t)sizeof v) { ssize_t r = read(j.fd, q + got, sizeof v - got); if (r <= 0) break; got += r; }
                            close(j.fd); int st = 0; waitpid(j.pid, &st, 0);
                            if (got != (ssize_t)sizeof v) throw std::runtime_error("a scoring child failed");
                            std::string line = cands[j.idx].line; line.pop_back();
                            out << line << ",\"logprob\":" << -v[0] << ",\"bytes\":" << (long long)v[1] << "}\n";
                        };
                        while (next < cands.size() || !running.empty()) {
                            while (next < cands.size() && (int)running.size() < workers) {
                                int fds[2]; if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
                                pid_t pid = fork(); if (pid < 0) throw std::runtime_error("fork failed");
                                if (pid == 0) {
                                    close(fds[0]); feed_str(model, cands[next].context); model.stats() = BlockStats{}; feed_str(model, cands[next].completion);
                                    double v[2] = {model.stats().loss_sum, (double)cands[next].completion.size()};
                                    ssize_t w = write(fds[1], v, sizeof v); (void)w; _exit(0);
                                }
                                close(fds[1]); running.push_back({pid, fds[0], next++});
                            }
                            reap(running.front()); running.erase(running.begin());
                        }
                    }
                    _exit(0);
                } catch (const std::exception& e) { std::cerr << "scoring: " << e.what() << '\n'; _exit(1); }
            }
            int st = 0; waitpid(master, &st, 0);
            if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) throw std::runtime_error("in-process scoring failed");
            std::cout << "scored epoch " << epoch << " items from " << items_dir << '\n' << std::flush;
        };
        auto report = [&](const char* what) {
            const double bpb = heldout_bpb();
            const bool market = model.params().market;
            const auto mc = model.market_census();
            const double smooth = rs.ema_bpb < 0 ? bpb : 0.5 * rs.ema_bpb + 0.5 * bpb; rs.ema_bpb = smooth;
            if (market && std::string(what) != "initial")   // regressed / improved / flat-while-growing (parsimony), on the smoothed signal
                model.market_adjust(smooth > rs.best_bpb + 0.01, smooth < rs.best_bpb - 0.01, mc.bodies > rs.last_bodies + rs.last_bodies / 20);
            rs.best_bpb = std::min(rs.best_bpb, smooth); rs.last_bodies = mc.bodies;
            log << "{\"event\":\"" << what << "\",\"epoch\":" << rs.epoch << ",\"chunk\":" << rs.chunk << ",\"bytes\":" << rs.bytes_total << ",\"words\":" << rs.words_total
                << ",\"heldout_bpb\":" << bpb << ",\"live_units\":" << model.live_units() << ",\"exact_locks\":" << model.exact_locks()
                << ",\"episodic\":[" << model.episodic_totals().births << ',' << model.episodic_totals().deferred << ',' << model.episodic_totals().evicted << ',' << model.episodic_totals().evicted_unused
                << ',' << model.episodic_totals().promoted << ',' << model.episodic_totals().promoted_splits << ',' << model.episodic_totals().retired << ']'
                << ",\"market\":{\"bar\":" << model.market_bar() << ",\"bodies\":" << mc.bodies << ",\"snapshots\":" << mc.snapshots << ",\"probation\":" << mc.probation
                << ",\"reuse\":[" << mc.reuse_p10 << ',' << mc.reuse_p50 << ',' << mc.reuse_p90 << "],\"fitness\":[" << mc.fitness_p10 << ',' << mc.fitness_p50 << ',' << mc.fitness_p90
                << "],\"pass\":" << model.market_totals().probation_pass << ",\"fail\":" << model.market_totals().probation_fail << ",\"culled_general\":" << model.market_totals().culled_general
                << ",\"culled_snapshot\":" << model.market_totals().culled_snapshot << ",\"cooldown_skips\":" << model.market_totals().cooldown_skips
                << ",\"raises\":" << model.market_totals().bar_raises << ",\"relaxes\":" << model.market_totals().bar_relaxes << ",\"parsimony\":" << model.market_totals().parsimony_raises
                << ",\"gate_accepts\":" << model.market_totals().gate_accepts << ",\"gate_rejects\":" << model.market_totals().gate_rejects
                << ",\"contest_free\":" << model.market_totals().contest_free << ",\"contest_wins\":" << model.market_totals().contest_wins
                << ",\"contest_losses\":" << model.market_totals().contest_losses << ",\"throttled\":" << model.market_totals().throttled << "}" << "}\n";
            log.flush();
            std::cout << what << " epoch " << rs.epoch << " chunk " << rs.chunk << '/' << chunks.size() << " heldout " << bpb << " bits/byte, live " << model.live_units() << '\n' << std::flush;
        };
        // Counterfactual anti-regression gate for the body market: the batch is released in one forked child and kept in
        // another; both read the same retention sample (16 x 2 KiB windows from training chunks already read this epoch
        // or earlier -- never the held-out text) with learning off. Accept when the loss does not rise by more than
        // --gate-tol bits/byte (default 0: a per-batch tolerance accumulates -- 76 batches x 0.002 drifted 0.15 in one epoch).
        std::vector<uint32_t> epoch_order; uint64_t epoch_order_for = UINT64_MAX;
        if (model.params().market) model.market_gate = [&](const std::vector<uint32_t>& batch) -> bool {
            if (epoch_order_for != rs.epoch) { epoch_order.resize(chunks.size()); for (uint32_t i = 0; i < chunks.size(); ++i) epoch_order[i] = i; std::mt19937_64 sh(1000 + rs.epoch); std::shuffle(epoch_order.begin(), epoch_order.end(), sh);
                if (curriculum && rs.epoch == 0) { static const int rank[6] = {2, 0, 5, 3, 4, 1}; std::stable_sort(epoch_order.begin(), epoch_order.end(), [&](uint32_t a, uint32_t b) { return rank[chunk_file[a]] < rank[chunk_file[b]]; }); }
                epoch_order_for = rs.epoch; }
            const size_t seen = rs.epoch > 0 ? chunks.size() : std::max<size_t>(1, rs.chunk);
            std::mt19937_64 g(777 + rs.gate_calls++); std::string sample;
            for (int w = 0; w < 16; ++w) {
                const std::string& c = chunks[rs.epoch > 0 ? std::uniform_int_distribution<size_t>(0, chunks.size() - 1)(g) : epoch_order[std::uniform_int_distribution<size_t>(0, seen - 1)(g)]];
                const size_t len = std::min<size_t>(2048, c.size()); const size_t off = c.size() > len ? std::uniform_int_distribution<size_t>(0, c.size() - len)(g) : 0;
                sample += c.substr(off, len); sample += '\n';
            }
            int fa[2], fb[2]; if (pipe(fa) != 0 || pipe(fb) != 0) throw std::runtime_error("pipe failed");
            std::cout.flush();
            auto child = [&](int* fd, bool cull) -> pid_t {
                pid_t pid = fork(); if (pid < 0) throw std::runtime_error("fork failed");
                if (pid == 0) {
                    close(fd[0]); if (cull) model.market_cull(batch);
                    model.set_mode(false, false); model.stats() = BlockStats{}; feed_str(model, sample);
                    double v = model.stats().loss_sum / std::max<uint64_t>(1, model.stats().events) / 0.6931471805599453;
                    ssize_t w = write(fd[1], &v, sizeof v); (void)w; _exit(0);
                }
                close(fd[1]); return pid;
            };
            const pid_t pa = child(fa, false), pb = child(fb, true);
            double keep = 0, drop = 0; ssize_t ra = read(fa[0], &keep, sizeof keep), rb = read(fb[0], &drop, sizeof drop);
            close(fa[0]); close(fb[0]); int st = 0; waitpid(pa, &st, 0); waitpid(pb, &st, 0);
            if (ra != (ssize_t)sizeof keep || rb != (ssize_t)sizeof drop) throw std::runtime_error("gate measurement failed");
            return drop <= keep + gate_tol;   // both children read the identical sample: the difference is the batch's own effect
        };
        if (!resume) { model.set_mode(true, true); report("initial"); save(dir / "checkpoint.state"); }
        const auto start = Clock::now(); uint64_t since_ckpt = 0;
        while (rs.epoch < epochs) {
            std::vector<uint32_t> order(chunks.size()); for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
            std::mt19937_64 sh(1000 + rs.epoch); std::shuffle(order.begin(), order.end(), sh);
            if (curriculum && rs.epoch == 0) {   // first epoch simple before complex: child-directed and conversational speech, subtitles, simple wiki, then books
                static const int rank[6] = {2, 0, 5, 3, 4, 1};   // index = files[] order: bnc_spoken, childes, gutenberg, open_subtitles, simple_wiki, switchboard
                std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return rank[chunk_file[a]] < rank[chunk_file[b]]; });
            }
            model.set_mode(true, true);
            for (; rs.chunk < order.size(); ++rs.chunk) {
                const std::string& c = chunks[order[rs.chunk]];
                feed_str(model, c);
                model.end_block(); model.stats() = BlockStats{};
                rs.bytes_total += c.size(); since_ckpt += c.size();
                bool in_word = false; for (unsigned char b : c) { const bool w = b > ' '; rs.words_total += w && !in_word; in_word = w; }
                if (since_ckpt >= CKPT_BYTES) { ++rs.chunk; if (model.params().market) report("checkpoint"); save(dir / "checkpoint.state"); --rs.chunk; since_ckpt = 0; }
            }
            ++rs.epoch; rs.chunk = 0;
            report("epoch");
            if (model.bp_outputs) std::cout << "backprop: " << model.bp_saturated << " of " << model.bp_outputs << " hidden outputs saturated\n";
            save(dir / "checkpoint.state"); save(dir / ("epoch_" + std::to_string(rs.epoch) + ".state"));
            if (!score_dir.empty()) score_items(score_dir, dir, rs.epoch);
            std::cout << "epoch " << rs.epoch << " done after " << std::chrono::duration<double>(Clock::now() - start).count() << " s this process\n" << std::flush;
        }
        std::ofstream(dir / "final.json") << "{\"complete\":true,\"epochs\":" << epochs << ",\"words\":" << rs.words_total << ",\"bytes\":" << rs.bytes_total << "}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
