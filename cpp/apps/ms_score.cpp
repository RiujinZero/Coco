// Batched zero-shot scoring of BabyLM items with the multi-stream trainer (docs/GPU_PORT_LOG.md): the same protocol and
// output as model_babylm_score, without a fork per candidate. The checkpoint is loaded and frozen, reads a newline
// through Organism, and then STREAMS candidates at a time are read in lockstep, each by a stream that starts from a copy
// of that frozen state (MultiStream::score). Per-candidate log-probabilities equal model_babylm_score's.
//
// usage: ms_score DIR ITEMS.jsonl OUT.jsonl [streams=64] [threads=12] [limit=0] [STATE_FILE=DIR/checkpoint.state]
#include "laminar/multistream.h"
#include "laminar/state_io.h"
#include "answer_tasks.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>

using namespace laminar;
namespace an = laminar::answers;

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: ms_score DIR ITEMS.jsonl OUT.jsonl [streams] [threads] [limit] [STATE_FILE]");
        const std::string dir = argv[1]; const uint32_t S = argc > 4 ? std::stoul(argv[4]) : 64; const unsigned threads = argc > 5 ? std::stoul(argv[5]) : 12;
        const size_t limit = argc > 6 ? std::stoul(argv[6]) : 0; const std::string state_file = argc > 7 ? std::string(argv[7]) : dir + "/checkpoint.state";
        std::ifstream cf(dir + "/config.json"); std::string config((std::istreambuf_iterator<char>(cf)), {});
        std::smatch m;
        if (!std::regex_search(config, m, std::regex("\"model_params\":\\[([^\\]]*)\\]"))) throw std::runtime_error("no model_params in config");
        std::vector<std::string> args{"score"};
        { std::string list = m[1]; std::regex item("\"([^\"]*)\""); for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) args.push_back((*it)[1]); }
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        MultiStream ms(Params::from_args((int)av.size(), av.data()), S, LearnerKind::None, threads);
        Organism& model = ms.organism();
        { state::Reader r(state_file); model.load_state(r); }   // the organism is first; the rest is run state
        model.set_mode(false, false); ms.set_train(false, false);
        { const uint8_t buf[2] = {model.last_symbol(), '\n'}; model.run_block(buf, 2); }   // as model_babylm_score: one newline

        std::vector<std::string> lines; std::vector<MultiStream::ScoreItem> items;
        { std::ifstream in(argv[2]); std::string line;
          while (std::getline(in, line)) { if (line.empty()) continue; items.push_back({an::field(line, "context"), an::field(line, "completion")}); lines.push_back(std::move(line)); if (limit && items.size() >= limit) break; } }
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<double> loss = ms.score(items);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::ofstream out(argv[3]);
        for (size_t i = 0; i < items.size(); ++i) { std::string line = lines[i]; line.pop_back();
            out << line << ",\"logprob\":" << -loss[i] << ",\"bytes\":" << (long long)items[i].completion.size() << "}\n"; }
        std::cerr << "scored " << items.size() << " candidates in " << sec << " s (" << S << " streams, " << threads << " threads)\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
