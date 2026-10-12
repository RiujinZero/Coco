// Inspect a model task learner checkpoint (read-only: the checkpoint file is never written).
//   1. Readout weight magnitude per input block (decision workspace, persistent workspace, pooled step
//      workspace, bias) and per operation, plus the feature normalisers.
//   2. Feature diversity: the frozen model reads a set of different public observations (each followed by
//      the same decision frame) and we report the body features' spread and the resulting policies. If body
//      features for different observations are nearly identical, the representation has collapsed; if they
//      differ but the policy is the same saturated answer, the readout has.
//
// usage: model_task_inspect SCENARIO_DIR
#include "laminar/model_task.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

using namespace laminar;

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: model_task_inspect SCENARIO_DIR");
        const std::string dir = argv[1];
        std::ifstream cf(dir + "/config.json"); std::string config((std::istreambuf_iterator<char>(cf)), {});
        std::smatch m;
        if (!std::regex_search(config, m, std::regex("\"model_params\":\\[([^\\]]*)\\]"))) throw std::runtime_error("no model_params in config");
        std::vector<std::string> args{"inspect"};
        { std::string list = m[1]; std::regex item("\"([^\"]*)\""); for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) args.push_back((*it)[1]); }
        std::vector<char*> argv2; for (auto& a : args) argv2.push_back(a.data());
        Params p = Params::from_args((int)argv2.size(), argv2.data());
        TaskModelConfig c; c.body_learning = config.find("\"body_learning\":true") != std::string::npos;
        if (std::regex_search(config, m, std::regex("\"reencode\":([0-9]+)"))) c.reencode = (uint32_t)std::stoul(m[1]);
        c.reencode_bodies = config.find("\"reencode_bodies\":true") != std::string::npos;
        if (config.find("\"policy_lr\":0,") != std::string::npos) { c.policy_lr = 0; c.outcome_mod = 0; c.body_credit = 0; }
        Organism model(p); TaskModel tm(model, c, 1);
        { state::Reader r(dir + "/checkpoint.state"); model.load_state(r); tm.state_io(r); }
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "checkpoint " << dir << ": live units " << model.live_units() << ", decisions so far " << tm.metrics.decisions << ", baseline " << tm.baseline() << "\n";
        const auto& W = tm.readout(); const int F = Dims::F, D = TaskModel::D;
        const char* blocks[] = {"decision ws", "persistent ws", "pooled ws"};
        std::cout << "readout |w| per op (L2 over each block) and bias:\n";
        for (int a = 0; a < 7; ++a) {
            std::cout << "  op" << a << ":";
            for (int b = 0; b < 3; ++b) { double s = 0; for (int k = 0; k < F; ++k) s += W[a * D + b * F + k] * W[a * D + b * F + k]; std::cout << ' ' << blocks[b] << '=' << std::sqrt(s); }
            std::cout << " bias=" << W[a * D + 3 * F] << '\n';
        }
        if (tm.situations()) std::cout << "re-encoded stored situations: " << tm.situations() << " stored, readout agrees with the teacher on "
                                       << 100.0 * tm.reencode_agreement(2000) << "% of 2000 (no updates)\n";
        if (tm.situations()) {
            auto show = [&](const char* what) {
                auto [ch, lb] = tm.reencode_breakdown("OArBrDrErQrRrdrejfr", {1, 5, 6}, 400);
                std::cout << "  replayed situations with observation ArBrDrErQrRrdrejfr, menu {1,5,6} (" << what << "): readout op1/op5/op6 = "
                          << ch[1] << '/' << ch[5] << '/' << ch[6] << "   teacher = " << lb[1] << '/' << lb[5] << '/' << lb[6] << '\n';
            };
            show("agency focus as left by the last mission");
            const bool on = tm.agency_focus_on(); tm.agency_focus_on() = false; show("agency focus off"); tm.agency_focus_on() = on;
            float saved[8]; std::memcpy(saved, model.task_obs, sizeof saved); std::fill(model.task_obs, model.task_obs + 8, 0.f);
            show("task state slots zeroed"); std::memcpy(model.task_obs, saved, sizeof saved);
        }
        const auto& sc = tm.feature_scale();
        for (int b = 0; b < 3; ++b) { double s = 0; for (int k = 0; k < F; ++k) s += sc[b * F + k]; std::cout << "  mean normaliser " << blocks[b] << ": " << s / F << '\n'; }
        // feature diversity over distinct public observations seen in the probes
        const std::vector<std::pair<std::string, std::vector<int>>> probes = {
            {"ArBrDrQrRrdrejfr", {1, 5, 6}}, {"ArBrDrErQrRrdrejfr", {1, 5, 6}}, {"ArBrDrFrGrQrRrdrejfr", {1, 5, 6}},
            {"ArBrDrErNrQrRrdrejfr", {1, 5, 6}}, {"ArBrDrErPrTkfr", {0, 1, 2, 3, 4, 5}}, {"ArBrDrFrHrTkfr", {0, 1, 2, 3, 4, 5}}};
        model.set_mode(false, false);
        std::vector<std::array<float, TaskModel::D>> feats;
        std::cout << "frozen decisions:\n";
        for (const auto& [obs, menu] : probes) {
            tm.observe(TaskChannel::observation, obs, true);
            auto ch = tm.decide(menu, false); feats.push_back(tm.last_features());
            tm.act(ch.op, true, false);
            std::cout << "  " << std::setw(22) << obs << " -> op" << ch.op << "  p=[";
            for (size_t i = 0; i < menu.size(); ++i) std::cout << (i ? " " : "") << ch.policy[menu[i]];
            std::cout << "]\n";
        }
        std::cout << "pairwise feature distance per block (mean |x_i - x_j| over features):\n";
        for (int b = 0; b < 3; ++b) {
            double tot = 0; int n = 0;
            for (size_t i = 0; i < feats.size(); ++i) for (size_t j = i + 1; j < feats.size(); ++j) { double d = 0; for (int k = 0; k < F; ++k) d += std::fabs(feats[i][b * F + k] - feats[j][b * F + k]); tot += d / F; ++n; }
            double mag = 0; for (auto& f : feats) for (int k = 0; k < F; ++k) mag += std::fabs(f[b * F + k]);
            std::cout << "  " << blocks[b] << ": distance " << tot / n << "  mean |x| " << mag / (feats.size() * F) << '\n';
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
