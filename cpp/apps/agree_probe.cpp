// Long-range agreement probe: can bodies carry CONTENT across time (item 2, docs/BODY_CIRCUIT_RESEARCH_LOG.md)?
//
// A synthetic language: "the NOUN FILLER^d VERB ." where NOUN is singular or plural (dog/dogs), the VERB must agree
// (runs/run), and d = 0..8 random filler words sit between them. Fillers are random, so neither the sequence memory
// (which needs exact repeated context) nor a single body (which sees ~5 bytes) can see noun and verb together; the number
// has to be carried across bytes by bodies writing and reading the workspace, or by routes/returns.
// Train the organism online on TRAIN_BYTES of this language (learning on), then score, frozen, minimal pairs (correct vs
// incorrect verb after the same context) by distance, under ablations: actual; constant bodies (own mean L6); no workspace
// memory read; silent bodies (sequence memory alone).
//
// usage: agree_probe TRAIN_BYTES [key=value organism params...]
//   MS=S (build with the multi-stream trainer, AGREE_MS): train with S parallel streams on the GPU learner (each stream reads a
//   contiguous slice of the same text), export stream 0 into the organism, then score exactly as below (CPU arms).
//   MIX=f MIX_FILE=path: dilution. A fraction f of the training BYTES are lines of real text (read in order from MIX_FILE,
//   wrapping), interleaved line by line with the synthetic sentences; the test is unchanged. Bisects between the synthetic
//   language (the circuit forms) and real text (it does not).
//   ARMS=n: score only the first n arms (1 = actual).
#include "laminar/organism.h"
#include "laminar/state_io.h"
#ifdef AGREE_MS
#include "laminar/multistream.h"
#endif
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace laminar;
static const char* NOUN[][2] = {{"dog", "dogs"}, {"cat", "cats"}, {"bird", "birds"}, {"child", "children"}, {"man", "men"}, {"girl", "girls"},
    {"boy", "boys"}, {"horse", "horses"}, {"teacher", "teachers"}, {"farmer", "farmers"}, {"mouse", "mice"}, {"friend", "friends"}};
static const char* VERB[][2] = {{"runs", "run"}, {"sings", "sing"}, {"walks", "walk"}, {"sleeps", "sleep"}, {"jumps", "jump"}, {"waits", "wait"},
    {"laughs", "laugh"}, {"swims", "swim"}, {"reads", "read"}, {"eats", "eat"}};
static const char* FILL[] = {"very", "old", "small", "red", "happy", "quiet", "near", "the", "big", "green", "tired", "brave", "from", "in",
    "town", "garden", "house", "river", "with", "blue", "tall", "kind", "long", "cold", "bright"};
static void feed_text(Organism& m, const std::string& s) { std::vector<uint8_t> b(s.size() + 1); b[0] = m.last_symbol(); std::memcpy(b.data() + 1, s.data(), s.size()); m.run_block(b.data(), b.size()); }

struct Item { std::string ctx, good, bad; int d; };
static std::string sentence(std::mt19937_64& g, int d, int& num, int& verb) {
    const int n = g() % 12; num = g() % 2; verb = g() % 10; std::string s = "the "; s += NOUN[n][num];
    for (int k = 0; k < d; ++k) { s += " "; s += FILL[g() % 25]; }
    return s;   // the verb follows
}
int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: agree_probe TRAIN_BYTES [params]");
        if (std::getenv("ISARE")) for (auto& v : VERB) { v[0] = "is"; v[1] = "are"; }   // the easier is/are task (a float GRU needs ~160k sentences)
        const size_t ntr = std::stoul(argv[1]);
        std::vector<std::string> args{"agree"}; for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        // ORACLE=1: an oracle WRITER. When a noun ends (next byte is a space), its number is written into the slowest workspace
        // dimension (31), during training and testing; readers must still learn to use it.
        bool oracle = std::getenv("ORACLE") && std::string(std::getenv("ORACLE")) == "1";
        // TAPER=f: guidance then withdrawal. The oracle writes on a fraction 1 - progress/f of nouns during training (0 after f), and
        // NEVER during calibration or testing: what is scored is only what the bodies learned to write and read.
        const double taper = std::getenv("TAPER") ? std::atof(std::getenv("TAPER")) : 0.0; double progress = 0.0; std::mt19937_64 og(5);
        std::string word; auto oracle_hook = [&](uint8_t sym, uint8_t target) {
            if (std::isalpha(sym)) word.push_back((char)sym); else word.clear();
            if (!oracle || target != ' ' || word.empty()) return;
            if (taper > 0.0 && (double)(og() >> 11) * (1.0 / 9007199254740992.0) >= std::max(0.0, 1.0 - progress / taper)) return;
            for (auto& nn : NOUN) for (int k = 0; k < 2; ++k) if (word == nn[k]) { if (std::getenv("WRITER_TARGET")) { model.diag_reg_target = k ? 100.f : -100.f; model.diag_reg_has_target = true; } else if (std::getenv("ORACLE_REG")) model.ws_register(0) = k ? 100.f : -100.f; else model.ws_memory()[31] = k ? 4000 : -4000; return; }
        };
        if (oracle) model.on_event = [&](uint8_t s, uint8_t t, const int32_t*, const float*) { oracle_hook(s, t); };
        std::mt19937_64 g(1);
        const bool curric = std::getenv("CURRIC_D") != nullptr;   // distance curriculum: max distance grows 0 -> 8 over the first 80% of training
        const double mix = std::getenv("MIX") ? std::atof(std::getenv("MIX")) : 0.0; std::string mixsrc; size_t mixpos = 0, mixbytes = 0;
        if (mix > 0.0) { const char* mf = std::getenv("MIX_FILE"); if (!mf) throw std::runtime_error("MIX needs MIX_FILE");
            std::ifstream in(mf, std::ios::binary); mixsrc.assign((std::istreambuf_iterator<char>(in)), {}); if (mixsrc.empty()) throw std::runtime_error("MIX_FILE empty"); }
        std::string text; while (text.size() < ntr) { const int dmax = curric ? std::min(8, (int)(10.0 * text.size() / (0.8 * ntr))) : 8;
            if (mix > 0.0 && (double)mixbytes < mix * (double)(text.size() + 1)) {   // next real-text line
                size_t e = mixsrc.find('\n', mixpos); if (e == std::string::npos) e = mixsrc.size() - 1;
                const std::string line = mixsrc.substr(mixpos, e + 1 - mixpos); mixpos = e + 1 >= mixsrc.size() ? 0 : e + 1;
                text += line; mixbytes += line.size(); continue; }
            int num, verb; std::string s = sentence(g, g() % (dmax + 1), num, verb); s += " "; s += VERB[verb][num == 0 ? 0 : 1]; s += " .\n"; text += s; }
        if (mix > 0.0) std::cout << "training text: " << text.size() << " bytes, " << 100.0 * mixbytes / text.size() << "% real text\n";
        model.set_mode(true, true);
        const char* ms_env = std::getenv("MS");
        if (ms_env && std::atoi(ms_env) > 0) {
#ifdef AGREE_MS
            const uint32_t S = (uint32_t)std::atoi(ms_env); const unsigned threads = std::getenv("MS_THREADS") ? (unsigned)std::atoi(std::getenv("MS_THREADS")) : 12u;
            MultiStream ms(Params::from_args((int)av.size(), av.data()), S, std::getenv("MS_REF") ? LearnerKind::Reference : LearnerKind::Gpu, threads);
            ms.set_train(true, true);
            std::vector<std::string> per(S); const size_t slice = text.size() / S + 1; size_t off = 0;   // contiguous slices cut at sentence ends
            for (uint32_t k = 0; k < S && off < text.size(); ++k) { size_t e = std::min(text.size(), off + slice); if (e < text.size()) { const size_t nl = text.find('\n', e); e = nl == std::string::npos ? text.size() : nl + 1; } per[k] = text.substr(off, e - off); off = e; }
            for (uint32_t k = 0; k < S; ++k) ms.set_stream_data(k, (const uint8_t*)per[k].data(), per[k].size());
            uint64_t next_block = 65536; while (ms.any_active()) { ms.step(); if (ms.events() >= next_block) { ms.end_block(); next_block += 65536; } }
            ms.export_stream(0);
            const std::string tmp = "/tmp/agree_ms_" + std::to_string(getpid()) + ".state";
            { state::Writer w(tmp); ms.organism().save_state(w); w.finish(); } { state::Reader r(tmp); model.load_state(r); } std::remove(tmp.c_str());
            std::cout << "trained with " << S << " streams (" << ms.learner_name() << "), " << ms.events() << " events\n" << std::flush;
#else
            throw std::runtime_error("MS=S needs the multi-stream build (-DAGREE_MS)");
#endif
        } else
        for (size_t off = 0; off < text.size(); off += 262144) { const size_t n = std::min<size_t>(262144, text.size() - off); progress = (double)off / text.size();
            std::vector<uint8_t> b(text.begin() + off, text.begin() + off + n); b.insert(b.begin(), model.last_symbol()); model.run_block(b.data(), b.size()); model.end_block(); }
        if (const char* sv = std::getenv("SAVE_STATE")) { state::Writer w(sv); model.save_state(w); w.finish(); std::cout << "saved " << sv << "\n"; }   // for pre-pretraining
        if (!model.motif_bases().empty() && model.params().fm_wire) {   // routed (wired) motifs: how many bodies does the route gate silence?
            const auto& fb = model.motif_bases(); const auto& ru = model.route_utilities(); const auto& P = model.params();
            std::cout << "route utility per motif (mean, % below the gate " << P.route_gate_bits << "):";
            for (size_t m = 1; m < fb.size(); ++m) { const uint32_t lo = fb[m], hi = lo + (P.fm_h[m] <= 1 ? 256u : P.fm_n[m]); double s = 0; uint32_t below = 0;
                for (uint32_t u = lo; u < hi; ++u) { s += ru[u]; below += ru[u] < P.route_gate_bits; }
                std::cout << "  h" << P.fm_h[m] << " " << s / (hi - lo) << " " << 100.0 * below / (hi - lo) << "%"; }
            std::cout << "\n"; }
        if (model.params().mtp) { std::cout << "mtp heads, mean loss over training (nats):"; for (uint32_t h = 0; h < model.params().mtp && h < 8; ++h) std::cout << " +" << h + 1 << " " << model.mtp_loss[h] / std::max<uint64_t>(1, model.mtp_n[h]); std::cout << "\n"; }
        model.set_mode(false, false); if (taper > 0.0) oracle = false;   // withdrawn: calibration and test see no oracle
        feed_text(model, "\n");
        std::mt19937_64 tg(99); std::vector<Item> items;
        for (int d : {0, 1, 2, 4, 6, 8}) for (int i = 0; i < 200; ++i) { int num, verb; std::string c = sentence(tg, d, num, verb) + " ";
            items.push_back({c, std::string(VERB[verb][num]) + " .", std::string(VERB[verb][1 - num]) + " .", d}); }
        // calibration for constant bodies: mean L6 of each body over fresh sentences
        std::vector<double> sum6((size_t)model.params().units * Dims::F, 0.0); std::vector<uint32_t> n6(model.params().units, 0);
        model.on_event = [&](uint8_t s0, uint8_t t0, const int32_t*, const float*) { oracle_hook(s0, t0); model.for_each_trace([&](const Trace& t) { if (!t.emitted) return; for (int j = 0; j < Dims::F; ++j) sum6[(size_t)t.unit * Dims::F + j] += t.l6[j]; ++n6[t.unit]; }); };
        { std::mt19937_64 cg(7); std::string cal; while (cal.size() < 65536) { int num, verb; cal += sentence(cg, cg() % 9, num, verb); cal += " "; cal += VERB[verb][num]; cal += " .\n"; } feed_text(model, cal); }
        model.on_event = nullptr; if (oracle) model.on_event = [&](uint8_t s, uint8_t t, const int32_t*, const float*) { oracle_hook(s, t); };
        if (model.params().ws_regs) {   // does any register hold the noun's number where the verb is read? (no oracle unless ORACLE without TAPER)
            for (int d : {0, 2, 6}) {
                std::mt19937_64 dg(1234 + d); double m[2][4] = {{0}}, m2[2][4] = {{0}}; int n[2] = {0, 0};
                for (int i = 0; i < 400; ++i) { int num, verb; const std::string c = sentence(dg, d, num, verb) + " "; feed_text(model, c);
                    for (int r = 0; r < 4; ++r) { const double v = model.ws_register(r); m[num][r] += v; m2[num][r] += v * v; } ++n[num];
                    feed_text(model, std::string(VERB[verb][num]) + " .\n"); }
                std::cout << "register content at the verb, distance " << d << ":";
                for (int r = 0; r < 4; ++r) { const double a = m[0][r] / n[0], b = m[1][r] / n[1], va = m2[0][r] / n[0] - a * a, vb = m2[1][r] / n[1] - b * b;
                    std::cout << "  r" << r << " sg " << a << " pl " << b << " (separation " << (b - a) / std::sqrt(0.5 * (va + vb) + 1e-9) << ")"; }
                std::cout << "\n";
            }
        }
        // arms 0-3 as before; with fixed motifs also: each motif silenced (4 + m) and the motif messages cut (last)
        std::vector<std::string> arm = {"actual", "constant bodies", "no workspace read", "silent bodies"};
        const auto& fmb = model.motif_bases(); const auto& fmh = model.params().fm_h; const auto& fmn = model.params().fm_n;
        for (size_t m = 0; m < fmb.size(); ++m) arm.push_back("silent motif h" + std::to_string(fmh[m]));
        const bool wired = !fmb.empty() && model.params().fm_wire; if (wired) arm.push_back("no motif messages");
        const bool wired_arm = wired; if (std::getenv("ARMS")) arm.resize(std::max(1, std::min((int)arm.size(), std::atoi(std::getenv("ARMS")))));
        if (std::getenv("MEM_ARMS")) { if (model.has_tiles()) arm.push_back("no tile read"); if (model.params().ws_regs) arm.push_back("no register read"); }   // what carries number?
        std::cout << "distance:           0      1      2      4      6      8   (% correct verb preferred)\n";
        for (int a = 0; a < (int)arm.size(); ++a) {
            int fds[2]; if (pipe(fds) != 0) return 1; std::cout.flush(); pid_t pid = fork();
            if (pid == 0) {   // one arm in a child; each item in a grandchild from the same state
                close(fds[0]);
                if (a == 1) model.l6_hook = [&](uint32_t u, int8_t* l6) { if (!n6[u]) return; for (int j = 0; j < Dims::F; ++j) l6[j] = (int8_t)std::lround(sum6[(size_t)u * Dims::F + j] / n6[u]); };
                if (a == 2) model.diag_zero_wsin = true;
                if (a == 3) model.l6_hook = [](uint32_t, int8_t* l6) { std::memset(l6, 0, Dims::F); };
                if (a >= 4 && (size_t)(a - 4) < fmb.size()) { const uint32_t lo = fmb[a - 4], hi = lo + (fmh[a - 4] <= 1 ? 256u : fmn[a - 4]);
                    model.l6_hook = [lo, hi](uint32_t u, int8_t* l6) { if (u >= lo && u < hi) std::memset(l6, 0, Dims::F); }; }
                if (wired_arm && arm[a] == "no motif messages") model.diag_zero_msgs_pub(true);
                if (arm[a] == "no tile read") model.diag_zero_tiles = true;
                if (arm[a] == "no register read") model.set_diag_zero_regs(true);
                int ok[9] = {0}, tot[9] = {0};
                for (auto& it : items) {
                    double lp[2];
                    for (int k = 0; k < 2; ++k) { int p2[2]; if (pipe(p2) != 0) _exit(1); pid_t c = fork();
                        if (c == 0) { close(p2[0]); feed_text(model, it.ctx); model.stats() = BlockStats{}; feed_text(model, k ? it.bad : it.good); double v = model.stats().loss_sum; ssize_t w = write(p2[1], &v, sizeof v); (void)w; _exit(0); }
                        close(p2[1]); double v = 0; ssize_t r = read(p2[0], &v, sizeof v); (void)r; close(p2[0]); waitpid(c, nullptr, 0); lp[k] = -v; }
                    ok[it.d] += lp[0] > lp[1]; ++tot[it.d];
                }
                char line[256]; int o = std::snprintf(line, sizeof line, "%-18s", arm[a].c_str());
                for (int d : {0, 1, 2, 4, 6, 8}) o += std::snprintf(line + o, sizeof line - o, " %5.1f ", 100.0 * ok[d] / std::max(1, tot[d]));
                std::snprintf(line + o, sizeof line - o, "\n"); ssize_t w = write(fds[1], line, std::strlen(line)); (void)w; _exit(0);
            }
            close(fds[1]); char buf[256] = {0}; ssize_t r = read(fds[0], buf, sizeof buf - 1); (void)r; close(fds[0]); waitpid(pid, nullptr, 0); std::cout << buf;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
