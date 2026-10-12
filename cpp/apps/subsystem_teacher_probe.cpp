// Can the trained subsystem policy serve as the model's teacher over FROZEN tools?
//
// The subsystem baseline exported its controller policy weights every 20 missions. The model task learner
// builds the tools once (same declared setup) and freezes them, while the baseline's knowledge kept learning
// during its run (model_version 114 -> 5,208). This probe loads one exported weight vector into a controller
// built on the frozen tools and runs the same 46 frozen developmental probes, with the policy deciding every
// operation (epsilon 0, no hints, no learning). A high score means the saved policy alone is a usable teacher;
// a low one means the policy depends on knowledge the frozen tools lack, and a full subsystem checkpoint
// (controller + knowledge) is needed instead.
//
// usage: subsystem_teacher_probe WEIGHTS.jsonl LABEL OUT_PREFIX [probes=46]
#define LAMINAR_GOAL_MIXED_LIBRARY
#include "goal_continuing_bootstrap.cpp"
#include "subsystem_policy.h"

int main(int argc, char** argv) {
    try {
        gl::require(argc >= 4, "usage: subsystem_teacher_probe WEIGHTS.jsonl LABEL OUT_PREFIX [probes]");
        const std::string label = argv[2], prefix = argv[3];
        const size_t probes = argc > 4 ? std::stoul(argv[4]) : 46;
        const auto loaded = laminar::subsystem_policy::load(argv[1], label); const int hidden = loaded.hidden; const auto& weights = loaded.weights;
        MixedOutputs out(prefix);
        GoalControlledRelationalSession tools(1, hidden, true);
        tools.knowledge = RelationalSession(1, 32768, 1024); ms::configure_initial(tools);
        auto fixture = gt::make_fixture(); fixture.lessons.apply(tools.knowledge); export_bootstrap(fixture.lessons, out.bootstrap);
        rt::Component reader; reader.teach(tools, out.bootstrap); lt::extend(tools, out.bootstrap, "mixed_long");
        ms::teach_initial(tools, out.bootstrap, "mixed_initial"); ms::extend(tools, out.bootstrap, "mixed_extension");
        tools.knowledge.rebuild();
        gl::require(weights.size() == tools.policy.weights.size(), "weight vector does not match the controller shape");
        tools.policy.weights = weights;
        const auto version = tools.knowledge.version();
        std::cout << "frozen tools knowledge version " << version << ", policy " << label << " (" << weights.size() << " weights)\n";
        tools.set_learning(false);
        std::array<size_t, 8> fam_success{}, fam_cases{}; size_t successes = 0;
        for (size_t index = 0; index < probes; ++index) {
            const auto family = index < 32 ? index % 8 : 0;
            auto options = mixed_options(index, family, "teacher_probe", true);
            auto frozen = tools; auto reader_copy = reader.clone();
            std::unique_ptr<gl::Environment> environment;
            if (index < 32) environment = mixed_environment(family, index, true, false, reader_copy, out.bootstrap, options.id);
            else environment = std::make_unique<st::Environment>(st::heldout_cases().at(index - 32));
            auto result = gl::run_mission(frozen, *environment, options, out.trace, out.missions);
            successes += result.success; fam_success[family] += result.success; ++fam_cases[family];
        }
        out.evaluations << "{\"policy\":" << gl::quote(label) << ",\"tools_knowledge_version\":" << version << ",\"successes\":" << successes
                        << ",\"cases\":" << probes << ",\"family_successes\":"; gl::numbers(out.evaluations, fam_success);
        out.evaluations << ",\"family_cases\":"; gl::numbers(out.evaluations, fam_cases); out.evaluations << "}\n";
        std::cout << "policy " << label << " over frozen tools: " << successes << '/' << probes << " family ";
        for (size_t f = 0; f < 8; ++f) std::cout << family_names[f] << '=' << fam_success[f] << '/' << fam_cases[f] << ' ';
        std::cout << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
