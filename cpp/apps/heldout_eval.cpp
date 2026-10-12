// Held-out bits per byte of a saved model, with parameter overrides applied after the run's own (e.g. iter_eval=8: the G
// extrapolation test - an iterated processor run for more hops than it trained with). Frozen: no learning, no structure.
//
// usage: heldout_eval DIR STATE_FILE TEXT_FILE [BYTES] [key=value ...]
#include "laminar/organism.h"
#include "laminar/state_io.h"
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
        if (argc < 4) throw std::runtime_error("usage: heldout_eval DIR STATE_FILE TEXT_FILE [BYTES] [key=value ...]");
        const std::string dir = argv[1], state_file = argv[2];
        size_t nb = 1 << 20; int a0 = 4;
        if (argc > 4 && std::strchr(argv[4], '=') == nullptr) { nb = std::stoul(argv[4]); a0 = 5; }
        std::ifstream cf(dir + "/config.json"); std::string config((std::istreambuf_iterator<char>(cf)), {}); std::smatch mm;
        if (!std::regex_search(config, mm, std::regex("\"model_params\":\\[([^\\]]*)\\]"))) throw std::runtime_error("no model_params in " + dir + "/config.json");
        std::vector<std::string> args{"heldout_eval"};
        { std::string list = mm[1]; std::regex item("\"([^\"]*)\""); for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) args.push_back((*it)[1]); }
        args.push_back("train=0");
        std::string over; for (int i = a0; i < argc; ++i) { args.push_back(argv[i]); over += std::string(" ") + argv[i]; }
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        { state::Reader r(state_file); model.load_state(r); }
        model.set_mode(false, false); model.stats() = BlockStats{};
        std::ifstream tf(argv[3], std::ios::binary); std::string text((std::istreambuf_iterator<char>(tf)), {});
        if (text.size() > nb) text.resize(nb);
        std::vector<uint8_t> buf(text.size() + 1); buf[0] = '\n'; std::memcpy(buf.data() + 1, text.data(), text.size());
        model.run_block(buf.data(), buf.size());
        const double bpb = model.stats().loss_sum / std::max<uint64_t>(1, model.stats().events) / 0.6931471805599453;
        std::printf("heldout %.5f bits/byte over %zu bytes of %s (overrides:%s)\n", bpb, text.size(), argv[3], over.empty() ? " none" : over.c_str());
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
