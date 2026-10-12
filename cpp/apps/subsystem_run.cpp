// The declared 2,000-episode continuing subsystem run (goal_extended_bootstrap.cpp), unchanged in every
// scientific respect, plus complete checkpoints: the goal controller with its policy, credits and relational
// knowledge, and the reader, every 20 missions. A crash or WSL failure resumes from the last checkpoint and
// continues byte-identically (logs are truncated back to the checkpoint and reopened). At the end the complete
// subsystem state is written to PREFIX_final.state, so the converged subsystem can be restored exactly, for
// example as the model task learner's teacher.
//
// usage: subsystem_run smoke|full PREFIX [--stop-after N]
// A fresh PREFIX declares and starts the run; an existing PREFIX_checkpoint.state resumes it.
#include "goal_extended_declaration.h"
#include "laminar/state_io.h"

namespace fs = std::filesystem;
static const char* log_suffixes[] = {"_trace.jsonl", "_missions.jsonl", "_weights.jsonl", "_evaluations.jsonl", "_bootstrap.jsonl", "_phases.jsonl", "_computations.jsonl"};

struct RunState {
    uint64_t next = 0, fits = 0, warm_success = 0, autonomous_success = 0;
    std::array<uint64_t, 8> occurrences{}, successes{};
    std::vector<uint64_t> log_sizes;
    template <class Ar> void state_io(Ar& ar) { ar(next); ar(fits); ar(warm_success); ar(autonomous_success); ar(occurrences); ar(successes); ar(log_sizes); }
};
static void save_state(const std::string& path, GoalControlledRelationalSession& learner, rt::Component& reader, RunState& rs) {
    const std::string tmp = path + ".tmp";
    { laminar::state::Writer w(tmp); w.section("subsystem"); laminar::state::io(w, learner); w.section("reader"); laminar::state::io(w, reader); w.section("run"); laminar::state::io(w, rs); w.finish(); }
    if (fs::exists(path)) fs::rename(path, path + ".prev");
    fs::rename(tmp, path);
}
static void load_state(const std::string& path, GoalControlledRelationalSession& learner, rt::Component& reader, RunState& rs) {
    laminar::state::Reader r(path); r.section("subsystem"); laminar::state::io(r, learner); r.section("reader"); laminar::state::io(r, reader); r.section("run"); laminar::state::io(r, rs); r.finish();
}
static uint64_t fingerprint(GoalControlledRelationalSession& learner, rt::Component& reader) {
    laminar::state::Writer w; laminar::state::io(w, learner); laminar::state::io(w, reader); return w.fingerprint();
}

int main(int argc, char** argv) {
    try {
        gl::require(argc == 3 || argc == 5, "usage: subsystem_run smoke|full PREFIX [--stop-after N]");
        const std::string profile = argv[1], prefix = argv[2];
        gl::require(profile == "smoke" || profile == "full", "unknown profile"); const bool smoke = profile == "smoke";
        const size_t stop_after = argc == 5 && std::string(argv[3]) == "--stop-after" ? std::stoull(argv[4]) : SIZE_MAX;
        const auto declaration = extended_configuration(smoke);
        const std::string ckpt = prefix + "_checkpoint.state";
        bool resume = fs::exists(ckpt);
        if (!resume) {
            gl::require(!fs::exists(prefix + "_config.json"), "declared run has no checkpoint: refusing to overwrite its evidence");
            std::ofstream config(prefix + "_config.json"); config << declaration;
        } else {
            std::ifstream config(prefix + "_config.json");
            gl::require(std::string((std::istreambuf_iterator<char>(config)), {}) == declaration, "configuration changed");
        }
        GoalControlledRelationalSession learner(1, 1316, true);
        rt::Component reader;
        RunState rs;
        const auto start = Clock::now();
        if (resume) {
            try { load_state(ckpt, learner, reader, rs); }
            catch (const std::exception& e) {
                std::cout << "current checkpoint unreadable (" << e.what() << "); using the previous one\n" << std::flush;
                learner = GoalControlledRelationalSession(1, 1316, true); reader = rt::Component(); rs = RunState{};
                load_state(ckpt + ".prev", learner, reader, rs);
            }
            for (size_t i = 0; i < 7; ++i) fs::resize_file(prefix + log_suffixes[i], rs.log_sizes.at(i));
            std::cout << "resumed at mission " << rs.next << '\n' << std::flush;
        }
        MixedOutputs out(prefix, resume ? std::ios::app : std::ios::trunc);
        const auto probe_config = mixed_config(smoke);
        auto checkpoint = [&]() {
            for (auto* f : {&out.trace, &out.missions, &out.parameters, &out.evaluations, &out.bootstrap, &out.phases, &out.computations}) f->flush();
            rs.log_sizes.clear(); for (auto* s : log_suffixes) rs.log_sizes.push_back((uint64_t)fs::file_size(prefix + s));
            save_state(ckpt, learner, reader, rs);
        };
        if (!resume) {   // identical setup to goal_extended_bootstrap
            learner.knowledge = RelationalSession(1, 32768, 1024); ms::configure_initial(learner);
            auto fixture = gt::make_fixture(); fixture.lessons.apply(learner.knowledge); export_bootstrap(fixture.lessons, out.bootstrap);
            reader.teach(learner, out.bootstrap); lt::extend(learner, out.bootstrap, "mixed_long");
            ms::teach_initial(learner, out.bootstrap, "mixed_initial"); if (smoke) ms::extend(learner, out.bootstrap, "mixed_extension");
            gl::weights(out.parameters, learner, "initial", "continuing");
            mixed_evaluate(learner, reader, probe_config, "before", out);
            checkpoint();
        }
        const size_t total = smoke ? 48 : 2000, warm = smoke ? 40 : 1600;
        size_t done_here = 0;
        for (size_t index = rs.next; index < total; ++index) {
            if (!smoke && index == 100) {
                const auto weights = learner.policy.weights; const auto updates = learner.updates();
                learner.knowledge.rebuild(); ledger(out.phases, "before_skill_extension", learner);
                ms::extend(learner, out.bootstrap, "mixed_extension");
                gl::require(weights == learner.policy.weights && updates == learner.updates() && learner.credits().size() == 100, "extension replaced learner state");
                ledger(out.phases, "after_skill_extension", learner);
            }
            const auto family = extended_order[index % extended_order.size()];
            size_t environment_index = 8 * rs.occurrences[family] + family; ++rs.occurrences[family];
            if (family >= 3 && (smoke || index >= 100)) environment_index += 8000;
            const bool guided = index < warm; const std::string phase = guided ? "warm" : "autonomous";
            auto options = mixed_options(index, family, phase);
            options.hint_probability = assistance(index, smoke); options.epsilon = guided ? 0.f : .1f;
            options.task_order_seed = 1000 + unsigned(environment_index);
            auto environment = mixed_environment(family, environment_index, false, smoke, reader, out.bootstrap, options.id);
            const auto began = Clock::now(); auto result = gl::run_mission(learner, *environment, options, out.trace, out.missions);
            rs.successes[family] += result.success; if (guided) rs.warm_success += result.success; else { rs.autonomous_success += result.success; gl::require(result.hints == 0, "autonomous guidance"); }
            export_computations(learner, options.id, out.computations);
            ledger(out.phases, options.id, learner, std::chrono::duration<double>(Clock::now() - began).count());
            if (guided && (index + 1) % 20 == 0) {
                const auto steps = size_t(std::ceil(10000 * options.hint_probability));
                const auto prior = learner.updates(), version = learner.knowledge.version(); const auto values = learner.empirical_values();
                const auto fitted = learner.fit_balanced_demonstrations(steps, gl::curriculum_demonstration_source); rs.fits += fitted;
                gl::require(fitted == steps && learner.updates() == prior + steps && learner.knowledge.version() == version && learner.empirical_values() == values,
                            "balanced fitter changed semantics or unexpected update count");
                gl::require(rs.fits <= 800000, "declared fit budget exceeded");
                out.phases << "{\"event\":\"balanced_fit\",\"after_episode\":" << index + 1 << ",\"guidance\":" << options.hint_probability
                    << ",\"fits\":" << fitted << ",\"cumulative_fits\":" << rs.fits << ",\"representatives\":" << demonstration_states(learner) << "}\n"; out.phases.flush();
            }
            if ((index + 1) % 20 == 0 || index + 1 == total) gl::weights(out.parameters, learner, options.id, "continuing");
            std::cout << options.id << " family=" << family_names[family] << " success=" << result.success << " decisions=" << result.decisions << " updates=" << learner.updates() << '\n' << std::flush;
            if ((!smoke && (index + 1) % 400 == 0) || (smoke && index + 1 == total)) mixed_evaluate(learner, reader, probe_config, "after" + std::to_string(index + 1), out);
            rs.next = index + 1; ++done_here;
            if ((index + 1) % 20 == 0 || index + 1 == total) {
                checkpoint();
                if (done_here >= stop_after && index + 1 < total) { std::cout << "stopping after checkpoint at mission " << rs.next << '\n'; return 0; }
            }
        }
        { laminar::state::Writer w(prefix + "_final.state"); w.section("subsystem"); laminar::state::io(w, learner); w.section("reader"); laminar::state::io(w, reader); w.section("run"); laminar::state::io(w, rs); w.finish(); }
        std::ofstream summary(prefix + "_summary.json"); summary << std::setprecision(17);
        summary << "{\"complete\":true,\"episodes\":" << total << ",\"warm_successes\":" << rs.warm_success << ",\"autonomous_successes\":" << rs.autonomous_success
                << ",\"family_counts\":"; gl::numbers(summary, rs.occurrences); summary << ",\"family_successes\":"; gl::numbers(summary, rs.successes);
        summary << ",\"direct_fits\":" << rs.fits << ",\"updates\":" << learner.updates() << ",\"credits\":" << learner.credits().size()
                << ",\"final_state\":\"" << prefix << "_final.state\",\"state_fingerprint\":\"" << std::hex << fingerprint(learner, reader) << std::dec
                << "\",\"seconds_this_process\":" << std::chrono::duration<double>(Clock::now() - start).count() << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
