// Item G diagnostic: the learned fan-in / fan-out of a motif_mix or motif_attn model. Every body at motif position m (m >= 1)
// reads the earlier positions' L6 through softmax weights.
//   static (motif_mix): per position, the mean gate on each sender position, the effective fan-in exp(entropy) (1 = reads one
//   sender, m = reads all equally), the share of sharp bodies (max gate >= 0.8), and per sender position its fan-out: the
//   share of later bodies reading it with a gate >= 0.2.
//   dynamic (TEXT given; any mixing model, motif_attn's weights depend on content): the frozen model reads the text and every
//   event's weights are measured - mean weight per sender, effective fan-in per event, sharp events, how much a weight varies
//   from event to event (std: 0 = a fixed wiring, content routing moves it), and per sender the later positions reading it
//   with a weight >= 0.2 per event (fan-out).
//
// usage: mix_probe DIR STATE_FILE [TEXT_FILE [BYTES]]
#include "laminar/organism.h"
#include "laminar/state_io.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

using namespace laminar;

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: mix_probe DIR STATE_FILE [TEXT_FILE [BYTES]]");
        const std::string dir = argv[1], state_file = argv[2];
        std::ifstream cf(dir + "/config.json"); std::string config((std::istreambuf_iterator<char>(cf)), {}); std::smatch mm;
        if (!std::regex_search(config, mm, std::regex("\"model_params\":\\[([^\\]]*)\\]"))) throw std::runtime_error("no model_params in " + dir + "/config.json");
        std::vector<std::string> args{"mix_probe"};
        { std::string list = mm[1]; std::regex item("\"([^\"]*)\""); for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) args.push_back((*it)[1]); }
        args.push_back("train=0");
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        { state::Reader r(state_file); model.load_state(r); }
        const std::vector<float>& th = model.motif_mix_logits(); const std::vector<uint32_t>& base = model.motif_bases();
        if (base.size() < 2) throw std::runtime_error("not a motif model");
        const size_t M = std::min<size_t>(base.size(), 8);
        if (!th.empty()) {
            std::vector<std::vector<double>> fanout(M, std::vector<double>(M, 0.0));   // [sender][receiver position]: share of receivers reading it >= 0.2
            std::printf("STATIC (per-body gate logits)\npos  bodies  mean gate on sender position 0..m-1                    fan-in exp(H)  sharp(max>=0.8)\n");
            for (size_t m = 1; m < M; ++m) {
                const uint32_t b0 = base[m], b1 = m + 1 < base.size() ? base[m + 1] : b0 + (base[m] - base[m - 1]);
                std::vector<double> mean(m, 0.0); double fan = 0.0, sharp = 0.0; uint32_t n = 0;
                for (uint32_t u = b0; u < b1 && (size_t)u * 8 + 7 < th.size(); ++u, ++n) {
                    const float* t = &th[(size_t)u * 8]; double mx = -1e30, Z = 0.0, pi[8], H = 0.0, pmax = 0.0;
                    for (size_t q = 0; q < m; ++q) mx = std::max(mx, (double)t[q]);
                    for (size_t q = 0; q < m; ++q) { pi[q] = std::exp(t[q] - mx); Z += pi[q]; }
                    for (size_t q = 0; q < m; ++q) { pi[q] /= Z; mean[q] += pi[q]; if (pi[q] > 0) H -= pi[q] * std::log(pi[q]); pmax = std::max(pmax, pi[q]); fanout[q][m] += pi[q] >= 0.2; }
                    fan += std::exp(H); sharp += pmax >= 0.8;
                }
                std::printf("%3zu  %6u  ", m, n); for (size_t q = 0; q < m; ++q) std::printf("%.3f ", mean[q] / std::max(1u, n));
                for (size_t q = m; q < 8; ++q) std::printf("      "); std::printf("  %5.2f of %zu      %.3f\n", fan / std::max(1u, n), m, sharp / std::max(1u, n));
                for (size_t q = 0; q < m; ++q) fanout[q][m] /= std::max(1u, n);
            }
            std::printf("fan-out: per sender position, the later positions' share of bodies reading it with a gate >= 0.2 (and their sum)\n");
            for (size_t q = 0; q + 1 < M; ++q) { double s = 0.0; std::printf("%3zu  ", q); for (size_t m = q + 1; m < M; ++m) { std::printf("->%zu %.2f  ", m, fanout[q][m]); s += fanout[q][m]; } std::printf("  sum %.2f\n", s); }
        }
        if (argc < 4) return 0;
        std::ifstream tf(argv[3], std::ios::binary); std::string text((std::istreambuf_iterator<char>(tf)), {});
        const size_t nb = argc > 4 ? std::stoul(argv[4]) : 100000; if (text.size() > nb) text.resize(nb);
        const std::vector<float>& pi = model.motif_mix_weights();
        double S[8][8] = {{0}}, S2[8][8] = {{0}}, fan[8] = {0}, sharp[8] = {0}, fo[8] = {0}; uint64_t n[8] = {0}, events = 0;
        model.set_mode(false, false);
        model.on_event = [&](uint8_t, uint8_t, const int32_t*, const float*) {
            ++events; if (pi.size() < M * 8) return;
            for (size_t m = 1; m < M; ++m) { const float* w = &pi[m * 8]; double Z = 0.0; for (size_t q = 0; q < m; ++q) Z += w[q]; if (Z < 0.5) continue;   // not reached / not mixed
                double H = 0.0, pmax = 0.0; ++n[m];
                for (size_t q = 0; q < m; ++q) { S[m][q] += w[q]; S2[m][q] += (double)w[q] * w[q]; if (w[q] > 0) H -= w[q] * std::log(w[q]); pmax = std::max(pmax, (double)w[q]); fo[q] += w[q] >= 0.2f; }
                fan[m] += std::exp(H); sharp[m] += pmax >= 0.8; }
        };
        std::vector<uint8_t> buf(text.size() + 1); buf[0] = '\n'; std::memcpy(buf.data() + 1, text.data(), text.size()); model.run_block(buf.data(), buf.size());
        std::printf("\nDYNAMIC over %llu events of %s\npos  events  mean weight on sender 0..m-1 (std across events)                         fan-in exp(H)  sharp(max>=0.8)\n", (unsigned long long)events, argv[3]);
        for (size_t m = 1; m < M; ++m) { const double k = std::max<double>(1, n[m]); std::printf("%3zu  %6llu  ", m, (unsigned long long)n[m]);
            for (size_t q = 0; q < m; ++q) { const double mu = S[m][q] / k; std::printf("%.3f(%.3f) ", mu, std::sqrt(std::max(0.0, S2[m][q] / k - mu * mu))); }
            for (size_t q = m; q < 8; ++q) std::printf("             "); std::printf("  %5.2f of %zu      %.3f\n", fan[m] / k, m, sharp[m] / k); }
        std::printf("fan-out per event: later positions reading each sender with a weight >= 0.2\n");
        for (size_t q = 0; q + 1 < M; ++q) std::printf("%3zu  %.2f of %zu\n", q, fo[q] / std::max<double>(1, events), M - 1 - q);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
