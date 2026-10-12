// Can the organism's output path learn at all? A direct training-curve harness.
//
// usage: model_unit_check TEXT_FILE TRAIN_BYTES HELDOUT_BYTES [key=value model params...]
//   Trains online on the first TRAIN_BYTES of TEXT_FILE (blocks of 256 KiB, printing the TRAINING bits/byte of each),
//   then scores the next HELDOUT_BYTES (a) frozen (train=0, structural=0, as model_babylm's held-out child does) and
//   (b) still learning (train=1, structural=0). A large gap between the late training loss and (a) means the
//   evaluation path, not learning, is broken; both high means the learners cannot fit.
//   TEXT_FILE "bigram:SEED" generates a synthetic order-1 Markov source over 64 symbols instead (known entropy).
#include "laminar/organism.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

using namespace laminar;

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: model_unit_check TEXT_FILE TRAIN_BYTES HELDOUT_BYTES [params]");
        const std::string src = argv[1]; const size_t ntr = std::stoul(argv[2]), nho = std::stoul(argv[3]);
        std::string text; double src_entropy = -1;
        if (src.rfind("bigram:", 0) == 0) {   // a random sparse Markov chain: each symbol has 4 likely successors
            std::mt19937_64 g(std::stoull(src.substr(7))); const int S = 64; std::vector<std::vector<double>> P(S, std::vector<double>(S, 0.002));
            for (int a = 0; a < S; ++a) { double w[4] = {0.5, 0.25, 0.15, 0.1}; for (int k = 0; k < 4; ++k) P[a][g() % S] += w[k]; double z = 0; for (double v : P[a]) z += v; for (double& v : P[a]) v /= z; }
            std::vector<double> pi(S, 1.0 / S); double H = 0; for (int it = 0; it < 200; ++it) { std::vector<double> q(S, 0); for (int a = 0; a < S; ++a) for (int b = 0; b < S; ++b) q[b] += pi[a] * P[a][b]; pi = q; }
            for (int a = 0; a < S; ++a) for (int b = 0; b < S; ++b) if (P[a][b] > 0) H -= pi[a] * P[a][b] * std::log2(P[a][b]);
            src_entropy = H; int s = 0; std::uniform_real_distribution<double> U(0, 1);
            for (size_t i = 0; i < ntr + nho + 2; ++i) { double r = U(g), c = 0; int b = 0; for (; b < S - 1; ++b) { c += P[s][b]; if (r < c) break; } s = b; text.push_back((char)(' ' + s)); }
        } else { std::ifstream in(src, std::ios::binary); text.assign((std::istreambuf_iterator<char>(in)), {}); }
        if (text.size() < ntr + nho + 2) throw std::runtime_error("text too short");
        double replay = 0.0;   // "replay=R": after every 4 KiB of new text, re-feed R x 4 KiB of earlier text in 256-byte windows at random offsets
        std::vector<std::string> args{"unit"}; for (int i = 4; i < argc; ++i) { std::string a = argv[i]; if (a.rfind("replay=", 0) == 0) replay = std::stod(a.substr(7)); else args.push_back(a); }
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        if (src_entropy >= 0) std::cout << "synthetic source entropy " << src_entropy << " bits/symbol\n";
        const double L2 = std::log(2.0); const size_t BLK = 256 * 1024;
        model.set_mode(true, true);
        for (size_t off = 0; off < ntr; off += BLK) {
            const size_t n = std::min(BLK, ntr - off); model.stats() = BlockStats{};
            if (replay <= 0.0) model.run_block((const uint8_t*)text.data() + off, n + 1);
            else {   // new text in 4 KiB pieces, each followed by replayed windows (the reported loss covers both)
                std::mt19937_64 rg(off + 17);
                for (size_t o = 0; o < n; o += 4096) {
                    const size_t m = std::min<size_t>(4096, n - o); model.run_block((const uint8_t*)text.data() + off + o, m + 1);
                    const size_t seen = off + o + m; if (seen < 1024) continue;
                    for (double r = replay * 4096; r >= 256; r -= 256) { const size_t at = rg() % (seen - 257); model.run_block((const uint8_t*)text.data() + at, 257); }
                }
            }
            model.end_block();
            std::cout << "train " << (off + n) / 1024 << " KiB  bits/byte " << model.stats().loss_sum / std::max<uint64_t>(1, model.stats().bytes) / L2 << "\n" << std::flush;
        }
        auto score = [&](bool learn) {
            model.set_mode(learn, false); model.stats() = BlockStats{};
            std::vector<uint8_t> buf(text.begin() + ntr, text.begin() + ntr + nho + 1);
            model.run_block(buf.data(), buf.size());
            return model.stats().loss_sum / std::max<uint64_t>(1, model.stats().bytes) / L2;
        };
        std::cout << "held-out frozen " << score(false) << "\n";
        std::cout << "held-out learning " << score(true) << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
