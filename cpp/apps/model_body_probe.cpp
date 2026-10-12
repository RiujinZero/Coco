// What do the bodies compute? A read-only probe of a saved BabyLM model.
//
// usage: model_body_probe DIR STATE_FILE TEXT_FILE [calibrate_bytes=262144] [measure_bytes=262144]
//   DIR: run directory with config.json ("model_params"). TEXT_FILE: held-out text (never trained on).
// Learning is off. Pass 1 (calibrate) records each body's MEAN L6 emission and its first-seen activation sign
// pattern per layer. Pass 2 (measure) reports, over the bytes that follow:
//   * depth: how many executed bodies reach L1 / L2-3 / L5 / L6 and emit;
//   * per layer: mean |activation| and the fraction pinned at the +-127 clamp;
//   * variation: per layer, the fraction of output signs that differ from the body's first-seen pattern
//     (0 = the body computes the same pattern whatever the context);
//   * the ablation: body-only loss (readout of the workspace, no sequence memory) with every emitting body's L6
//     replaced by its own calibrated mean (a body = a constant vector chosen by routing), against the actual loss
//     and against an empty workspace. If "constant bodies" ~ "actual", the context-dependent computation inside the
//     bodies carries nothing; all the value is in which bodies the locks select.
#include "laminar/organism.h"
#include "laminar/state_io.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include <random>
#include <sys/wait.h>
#include <unistd.h>

using namespace laminar;

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: model_body_probe DIR STATE_FILE TEXT_FILE [calibrate_bytes] [measure_bytes]");
        const std::string dir = argv[1], state_file = argv[2];
        const size_t ncal = argc > 4 ? std::stoul(argv[4]) : 262144, nmeas = argc > 5 ? std::stoul(argv[5]) : 262144;
        std::ifstream cf(dir + "/config.json"); std::string config((std::istreambuf_iterator<char>(cf)), {});
        std::smatch m;
        if (!std::regex_search(config, m, std::regex("\"model_params\":\\[([^\\]]*)\\]"))) throw std::runtime_error("no model_params in config");
        std::vector<std::string> args{"probe"};
        { std::string list = m[1]; std::regex item("\"([^\"]*)\""); for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) args.push_back((*it)[1]); }
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        { state::Reader r(state_file); model.load_state(r); }
        model.set_mode(false, false);
        const uint32_t U = model.params().units; const int32_t xg = model.params().exact_gain;
        {   // how far have the durable weights moved from the birth distribution (uniform {-2..2})?
            const int nib0[6] = {Dims::NIB_L1, Dims::NIB_L2, Dims::NIB_L3, Dims::NIB_L5, Dims::NIB_L6, Dims::UNIT_NIBBLES}; const char* nm[5] = {"L1", "L2", "L3", "L5", "L6"};
            for (int l = 0; l < 5; ++l) { uint64_t h[16] = {0}, tot = 0;
                for (uint32_t u = 0; u < U; ++u) { const uint8_t* w = model.weights() + (size_t)u * Dims::UNIT_BYTES; for (int n = nib0[l]; n < nib0[l + 1]; ++n) { ++h[nib_get(w, n) + 8]; ++tot; } }
                std::printf("weights %s: beyond +-2 %.2f%%  hist", nm[l], 100.0 * (tot - h[6] - h[7] - h[8] - h[9] - h[10]) / std::max<uint64_t>(tot, 1));
                for (int k = 0; k < 16; ++k) std::printf(" %.3f", (double)h[k] / std::max<uint64_t>(tot, 1)); std::printf("\n"); }
        }
        if (const char* other = std::getenv("WEIGHTS_CMP")) {   // fraction of weights that differ from another state of the same run, per layer
            Organism m2(Params::from_args((int)av.size(), av.data())); { state::Reader r(other); m2.load_state(r); }
            const int nib0[6] = {Dims::NIB_L1, Dims::NIB_L2, Dims::NIB_L3, Dims::NIB_L5, Dims::NIB_L6, Dims::NIB_L4}; const char* nm[5] = {"L1", "L2", "L3", "L5", "L6"};
            for (int l = 0; l < 5; ++l) { uint64_t diff = 0, tot = 0, units_same = 0;
                for (uint32_t u = 0; u < U; ++u) { const uint8_t* a = model.weights() + (size_t)u * Dims::UNIT_BYTES; const uint8_t* b = m2.weights() + (size_t)u * Dims::UNIT_BYTES; uint64_t d = 0;
                    for (int n = nib0[l]; n < nib0[l + 1]; ++n) d += nib_get(a, n) != nib_get(b, n);
                    diff += d; tot += nib0[l + 1] - nib0[l]; units_same += d == 0; }
                std::printf("changed vs %s: %s %.2f%% of weights, %.1f%% of units unchanged\n", other, nm[l], 100.0 * diff / tot, 100.0 * units_same / U); }
        }
        if (std::getenv("REGS_PROBE")) {   // did the long-range circuit form on real text? the agree_probe language, fed to the frozen model
            static const char* NOUN[][2] = {{"dog", "dogs"}, {"cat", "cats"}, {"bird", "birds"}, {"child", "children"}, {"man", "men"}, {"girl", "girls"}, {"boy", "boys"}, {"horse", "horses"}, {"teacher", "teachers"}, {"farmer", "farmers"}, {"mouse", "mice"}, {"friend", "friends"}};
            static const char* FILL[] = {"very", "old", "small", "red", "happy", "quiet", "near", "the", "big", "green", "tired", "brave", "from", "in", "town", "garden", "house", "river", "with", "blue", "tall", "kind", "long", "cold", "bright"};
            auto feed_s = [&](const std::string& s) { std::vector<uint8_t> b(s.begin(), s.end()); b.insert(b.begin(), model.last_symbol()); model.run_block(b.data(), b.size()); };
            bool want = false; float pa = 0.f, pi = 0.f;
            model.on_event = [&](uint8_t, uint8_t, const int32_t*, const float* pr) { if (want) { pa = pr['a']; pi = pr['i']; } };
            std::mt19937_64 g(4242); feed_s("\n");
            std::printf("register probe (is/are after 'the NOUN FILLER^d '): %% correct (p(a) vs p(i)) and per-register sg/pl separation\n");
            for (int d : {0, 1, 2, 4, 6, 8}) {
                int ok = 0, n = 0; double m[2][4] = {{0}}, m2[2][4] = {{0}}; int c[2] = {0, 0};
                for (int i = 0; i < 300; ++i) {
                    const int nn = (int)(g() % 12), num = (int)(g() % 2); std::string s = "the "; s += NOUN[nn][num];
                    for (int k = 0; k < d; ++k) { s += " "; s += FILL[g() % 25]; } s += " ";
                    feed_s(s);
                    for (int r = 0; r < 4; ++r) { const double v = model.ws_register(r); m[num][r] += v; m2[num][r] += v * v; } ++c[num];
                    want = true; feed_s(num ? "a" : "i"); want = false;
                    ok += num ? pa > pi : pi > pa; ++n;
                    feed_s(num ? "re .\n" : "s .\n");
                }
                std::printf("d%d  %5.1f%%  regs:", d, 100.0 * ok / n);
                for (int r = 0; r < 4; ++r) { const double a = m[0][r] / c[0], b = m[1][r] / c[1], va = m2[0][r] / c[0] - a * a, vb = m2[1][r] / c[1] - b * b;
                    std::printf("  r%d %.0f/%.0f sep %.2f", r, a, b, (b - a) / std::sqrt(0.5 * (va + vb) + 1e-9)); }
                std::printf("\n");
            }
            model.on_event = nullptr; return 0;
        }
        if (std::getenv("REGS_ABLATE")) {   // do the registers carry anything on real text? held-out bits/byte with the registers read as 0
            std::string text; { std::ifstream in(argv[3], std::ios::binary); text.assign((std::istreambuf_iterator<char>(in)), {}); }
            const size_t n = std::min(text.size(), nmeas);
            for (int arm = 0; arm < (model.has_tiles() ? 3 : 2); ++arm) {
                int fds[2]; if (pipe(fds) != 0) return 1; std::cout.flush(); pid_t pid = fork();
                if (pid == 0) { close(fds[0]); model.diag_zero_regs = arm == 1; model.diag_zero_tiles = arm == 2; model.stats() = BlockStats{};
                    std::vector<uint8_t> b(n + 1); b[0] = model.last_symbol(); std::memcpy(b.data() + 1, text.data(), n); model.run_block(b.data(), b.size());
                    double v = model.stats().loss_sum / std::max<uint64_t>(1, model.stats().events) / 0.6931471805599453; ssize_t w = write(fds[1], &v, sizeof v); (void)w; _exit(0); }
                close(fds[1]); double v = 0; ssize_t r = read(fds[0], &v, sizeof v); (void)r; close(fds[0]); waitpid(pid, nullptr, 0);
                std::printf("held-out bits/byte, %s: %.5f\n", arm == 2 ? "tiles read as 0" : arm ? "registers read as 0" : "actual", v); }
            return 0;
        }
        if (std::getenv("WEIGHTS_ONLY")) return 0;
        std::string text; { std::ifstream in(argv[3], std::ios::binary); text.assign((std::istreambuf_iterator<char>(in)), {}); }
        if (text.size() < ncal + nmeas + 2) throw std::runtime_error("text too short");

        std::vector<double> sum6((size_t)U * Dims::F, 0.0); std::vector<uint32_t> n6(U, 0);
        constexpr int NL = 5; const int width[NL] = {Dims::A, Dims::B, Dims::C, Dims::E, Dims::F}; const char* name[NL] = {"L1", "L2", "L3", "L5", "L6"};
        std::vector<std::vector<int8_t>> first(NL); std::vector<std::vector<uint8_t>> seen(NL);
        for (int l = 0; l < NL; ++l) { first[l].assign((size_t)U * width[l], 0); seen[l].assign(U, 0); }
        std::vector<std::vector<double>> sumH(4); std::vector<std::vector<uint32_t>> nH(4);   // per-body mean of L1, L2, L3, L5
        for (int l = 0; l < 4; ++l) { sumH[l].assign((size_t)U * width[l], 0.0); nH[l].assign(U, 0); }
        auto layer = [](const Trace& t, int l) -> const int8_t* { return l == 0 ? t.l1 : l == 1 ? t.l2 : l == 2 ? t.l3 : l == 3 ? t.l5 : t.l6; };
        auto reached = [](const Trace& t, int l) { return l == 0 ? (t.reached & R_L1) : l <= 2 ? (t.reached & R_L23) : l == 3 ? (t.reached & R_L5) : (t.reached & R_L6); };

        bool measuring = false;
        uint64_t bytes = 0, traces = 0, depth[5] = {0}, emitters = 0, uncal = 0;
        uint64_t outs[NL] = {0}, sat[NL] = {0}, flips[NL] = {0}, cmp[NL] = {0}; double absum[NL] = {0};
        double loss_act = 0, loss_const = 0, loss_empty = 0, dev6 = 0, mag6 = 0;
        model.on_event = [&](uint8_t, uint8_t target, const int32_t* ws, const float*) {
            if (!measuring) {
                model.for_each_trace([&](const Trace& t) {
                    const uint32_t u = t.unit;
                    for (int l = 0; l < NL; ++l) if (reached(t, l) && !seen[l][u]) { std::memcpy(&first[l][(size_t)u * width[l]], layer(t, l), width[l]); seen[l][u] = 1; }
                    if (t.emitted) { const int g = model.exact_unit(u) ? xg : 1; for (int j = 0; j < Dims::F; ++j) sum6[(size_t)u * Dims::F + j] += g * t.l6[j]; ++n6[u]; }
                    for (int l = 0; l < 4; ++l) if (reached(t, l)) { const int8_t* y = layer(t, l); for (int j = 0; j < width[l]; ++j) sumH[l][(size_t)u * width[l] + j] += y[j]; ++nH[l][u]; }
                });
                return;
            }
            ++bytes; int32_t wc[Dims::F], w0[Dims::F] = {0}; std::memcpy(wc, ws, sizeof wc);
            model.for_each_trace([&](const Trace& t) {
                const uint32_t u = t.unit; ++traces;
                depth[0] += (t.reached & R_L1) != 0; depth[1] += (t.reached & R_L23) != 0; depth[2] += (t.reached & R_L5) != 0; depth[3] += (t.reached & R_L6) != 0; depth[4] += t.emitted != 0;
                for (int l = 0; l < NL; ++l) {
                    if (!reached(t, l)) continue; const int8_t* y = layer(t, l);
                    for (int j = 0; j < width[l]; ++j) { ++outs[l]; absum[l] += std::abs((int)y[j]); sat[l] += y[j] >= 127 || y[j] <= -127; }
                    if (seen[l][u]) { const int8_t* f = &first[l][(size_t)u * width[l]]; for (int j = 0; j < width[l]; ++j) { flips[l] += (y[j] > 0) != (f[j] > 0); ++cmp[l]; } }
                }
                if (!t.emitted) return; ++emitters;
                if (!n6[u]) { ++uncal; return; }
                const int g = model.exact_unit(u) ? xg : 1;
                for (int j = 0; j < Dims::F; ++j) { const double mu = sum6[(size_t)u * Dims::F + j] / n6[u];
                    wc[j] += (int32_t)std::lround(mu) - g * t.l6[j]; dev6 += std::abs(g * t.l6[j] - mu); mag6 += std::abs(g * t.l6[j]); }
            });
            loss_act += model.body_loss(ws, target); loss_const += model.body_loss(wc, target); loss_empty += model.body_loss(w0, target);
        };
        std::vector<uint8_t> buf; auto feed = [&](size_t off, size_t n) { buf.assign(text.begin() + off, text.begin() + off + n + 1); model.run_block(buf.data(), buf.size()); };
        feed(0, ncal);
        measuring = true;
        // the ablation through the WHOLE model (sequence memory included), each arm from the same post-calibration state
        // traffic split: bodies by lifetime hits (lock admissions), cut at the emission-weighted median
        std::vector<std::pair<uint32_t, uint32_t>> hv; for (uint32_t u = 0; u < U; ++u) if (n6[u]) hv.emplace_back(model.unit_hits(u), n6[u]);
        std::sort(hv.begin(), hv.end()); uint64_t tot = 0, acc = 0; for (auto& h : hv) tot += h.second; uint32_t hcut = 0; for (auto& h : hv) { acc += h.second; if (acc * 2 >= tot) { hcut = h.first; break; } }
        { std::vector<uint32_t> hs; for (uint32_t u = 0; u < U; ++u) if (model.unit_live(u)) hs.push_back(model.unit_hits(u)); std::sort(hs.begin(), hs.end());
          std::cout << "lifetime hits per live body: p10 " << hs[hs.size() / 10] << "  p50 " << hs[hs.size() / 2] << "  p90 " << hs[hs.size() * 9 / 10] << "  p99 " << hs[hs.size() * 99 / 100]
                    << "  (emission-weighted median " << hcut << ")\n"; }
        const char* arm_name[9] = {"actual", "constant bodies (own mean L6)", "silent bodies (L6 = 0)", "constant BROAD bodies only (hits > median)", "constant NARROW bodies only",
                                   "constant L1 (own mean)", "constant L2 (own mean; also fixes its routing key)", "constant L3 (own mean)", "constant L5 (own mean)"};
        for (int arm = 0; arm < 9; ++arm) {
            std::cout.flush(); pid_t pid = fork();
            if (pid == 0) {
                double bl = 0; uint64_t bn = 0;
                model.on_event = [&](uint8_t, uint8_t target, const int32_t* ws, const float*) { bl += model.body_loss(ws, target); ++bn; };
                if (arm >= 5) { const int l = arm - 5, li = l == 0 ? LI_1 : l == 1 ? LI_2 : l == 2 ? LI_3 : LI_5;
                    model.layer_hook = [&, l, li](uint32_t u, int lay, int8_t* y) { if (lay != li || !nH[l][u]) return;
                        for (int j = 0; j < width[l]; ++j) { long v = std::lround(sumH[l][(size_t)u * width[l] + j] / nH[l][u]); y[j] = (int8_t)std::max(-127L, std::min(127L, v)); } }; }
                if (arm == 1 || arm == 3 || arm == 4) model.l6_hook = [&, arm](uint32_t u, int8_t* l6) { if (!n6[u]) return;
                    if (arm == 3 && model.unit_hits(u) <= hcut) return; if (arm == 4 && model.unit_hits(u) > hcut) return; const int g = model.exact_unit(u) ? xg : 1;
                    for (int j = 0; j < Dims::F; ++j) { long v = std::lround(sum6[(size_t)u * Dims::F + j] / n6[u] / g); l6[j] = (int8_t)std::max(-127L, std::min(127L, v)); } };
                if (arm == 2) model.l6_hook = [](uint32_t, int8_t* l6) { std::memset(l6, 0, Dims::F); };
                model.stats() = BlockStats{}; feed(ncal, nmeas);
                std::cout << "full-model bits/byte, " << arm_name[arm] << ": " << model.stats().loss_sum / model.stats().bytes / std::log(2.0) << "   body-only " << bl / std::max<uint64_t>(1, bn) / std::log(2.0)
                          << "   neighbour accepts/byte " << (double)model.stats().neighbor_accepts / std::max<uint64_t>(1, model.stats().bytes) << "\n" << std::flush;
                _exit(0);
            }
            int st = 0; waitpid(pid, &st, 0);
        }
        feed(ncal, nmeas);
        {   // how far have the hidden weights moved from birth? (sensory-born bodies only: split/nursery/snapshot bodies are copies)
            const int lis[6] = {LI_1, LI_2, LI_3, LI_5, LI_6, LI_4}; const char* ln[6] = {"L1", "L2", "L3", "L5", "L6", "L4"}; uint64_t ch[6] = {0}, tot[6] = {0}, n = 0;
            for (uint32_t u = 0; u < U; ++u) { if (!model.unit_live(u) || model.unit_copied(u)) continue; ++n;
                for (int k = 0; k < 6; ++k) { const LayerDesc& d = layer_desc(lis[k], model.params().shifts); ch[k] += model.weights_changed(u, lis[k]); tot[k] += (uint64_t)d.in * d.out; } }
            std::cout << "weights changed since birth (" << n << " sensory-born bodies):"; for (int k = 0; k < 6; ++k) std::cout << "  " << ln[k] << " " << (double)ch[k] / std::max<uint64_t>(1, tot[k]); std::cout << "\n";
        }
        const double L2 = std::log(2.0);
        std::cout << "measured " << bytes << " bytes, " << (double)traces / bytes << " bodies executed per byte, " << (double)emitters / bytes << " emit"
                  << " (" << uncal << " emissions from bodies unseen in calibration kept as-is)\n";
        std::cout << "reach  L1 " << (double)depth[0] / traces << "  L2-3 " << (double)depth[1] / traces << "  L5 " << (double)depth[2] / traces
                  << "  L6 " << (double)depth[3] / traces << "  emit " << (double)depth[4] / traces << "\n";
        for (int l = 0; l < NL; ++l) std::cout << name[l] << "  mean|y| " << absum[l] / std::max<uint64_t>(1, outs[l]) << "  saturated " << (double)sat[l] / std::max<uint64_t>(1, outs[l])
                                                  << "  sign flips vs first-seen " << (double)flips[l] / std::max<uint64_t>(1, cmp[l]) << "\n";
        std::cout << "L6 emission: mean |deviation from the body's own mean| / mean |emission| = " << dev6 / std::max(1.0, mag6) << "\n";
        std::cout << "body-only bits/byte: actual " << loss_act / bytes / L2 << "  constant bodies " << loss_const / bytes / L2 << "  empty workspace " << loss_empty / bytes / L2 << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
