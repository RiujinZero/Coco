// Free generation from a saved model (read-only: the checkpoint is never written).
// usage: model_generate DIR "PROMPT" BYTES [greedy|T]
//   DIR is a model_task_learner or model_answer_learner scenario directory (config.json + checkpoint.state).
//   The frozen model reads PROMPT, then writes BYTES bytes, greedy or sampled at temperature T.
#include "laminar/model_task.h"
#include <fstream>
#include <iostream>
#include <regex>

using namespace laminar;

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: model_generate DIR PROMPT BYTES [greedy|T]");
        const std::string dir = argv[1], prompt = argv[2]; const size_t n = std::stoul(argv[3]);
        const std::string how = argc > 4 ? argv[4] : "greedy";
        std::ifstream cf(dir + "/config.json"); std::string config((std::istreambuf_iterator<char>(cf)), {});
        std::smatch m;
        if (!std::regex_search(config, m, std::regex("\"model_params\":\\[([^\\]]*)\\]"))) throw std::runtime_error("no model_params in config");
        std::vector<std::string> args{"generate"};
        { std::string list = m[1]; std::regex item("\"([^\"]*)\""); for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) args.push_back((*it)[1]); }
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        TaskModelConfig c;
        auto grab = [&](const char* key) -> uint32_t { std::smatch k; return std::regex_search(config, k, std::regex(std::string("\"") + key + "\":([0-9]+)")) ? (uint32_t)std::stoul(k[1]) : 0u; };
        c.body_learning = config.find("\"body_learning\":true") != std::string::npos || config.find("model_answer_learner") != std::string::npos;
        c.reencode = grab("reencode"); c.replay_missions = grab("replay_missions"); c.success_replay = config.find("model_answer_learner") != std::string::npos ? 0 : grab("success_replay");
        c.reencode_bodies = config.find("\"reencode_bodies\":true") != std::string::npos;
        c.success_budget = grab("success_budget"); c.success_solo = config.find("\"success_solo\":true") != std::string::npos;
        c.success_prune = config.find("\"success_prune\":true") != std::string::npos; c.success_stratify = config.find("\"success_stratify\":true") != std::string::npos;
        if (config.find("\"policy_lr\":0,") != std::string::npos) { c.policy_lr = 0; c.outcome_mod = 0; c.body_credit = 0; }
        Organism model(Params::from_args((int)av.size(), av.data())); TaskModel tm(model, c, 3);
        { state::Reader r(dir + "/checkpoint.state"); model.load_state(r); tm.state_io(r); }   // run state not needed
        model.set_mode(false, false);
        tm.read_text((const uint8_t*)prompt.data(), prompt.size());
        std::string out;
        while (out.size() < n) out += how == "greedy" ? tm.generate(n - out.size(), false) : tm.generate(n - out.size(), true, std::stof(how));
        std::cout << prompt << out << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
