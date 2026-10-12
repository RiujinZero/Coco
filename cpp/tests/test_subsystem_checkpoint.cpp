// Full subsystem checkpoint: the goal controller (policy, target policy, credits, empirical values, mission
// history, agenda) and its relational knowledge (concepts, language and transition receipts, frames, primitive
// store, composed learner, procedures, RNG) round-trip exactly.
//   1. save -> load into a fresh session gives the same fingerprint;
//   2. the original and the restored session, driven through the same further missions with learning on,
//      produce identical decision traces, identical results and identical final fingerprints.
#include "../apps/goal_stopping_environment.h"
#include "laminar/state_io.h"
#include <filesystem>
#include <iostream>
#include <sstream>

using namespace laminar::relational;
namespace gl = laminar::goal_learning;
namespace st = laminar::goal_learning::stopping;
using laminar::goal_test::require;

static uint64_t fingerprint(GoalControlledRelationalSession& s) { laminar::state::Writer w; laminar::state::io(w, s); return w.fingerprint(); }

// Real missions through the ordinary harness: the stopping family, the controller's own policy with
// exploration and tapered protocol hints, semantic learning and terminal credit. The full decision log
// (observations, menus, operations, callbacks, credit) is the trace compared between the two sessions.
static std::string mission(GoalControlledRelationalSession& s, const std::string& id, size_t index) {
    auto cases = st::training_cases(); auto spec = cases[index % cases.size()]; spec.prefix += id; spec.order_seed = 1000 + (unsigned)index;
    st::Environment env(spec);
    gl::MissionOptions o; o.id = id; o.arm = "checkpoint_test"; o.index = index; o.terminal_updates = 64; o.epsilon = 0.2f;
    o.hint_probability = index % 2 ? 0.5 : 0.0; o.demonstration = st::demonstration; o.demonstration_source = st::demonstration_source;
    o.assistance_seed = 7000 + (uint32_t)index; o.task_order_seed = 1000 + (uint32_t)index; o.limits = st::limits(); o.prices = st::prices();
    std::ostringstream log, results; log.precision(17); results.precision(17);
    auto r = gl::run_mission(s, env, o, log, results);
    return log.str() + results.str();
}

int run() {
    auto fixture = laminar::goal_test::make_fixture();
    GoalControlledRelationalSession a(11, 16, true); a.knowledge = fixture.make_knowledge(11);
    for (size_t i = 0; i < 6; ++i) mission(a, "warm" + std::to_string(i), i);
    const std::string file = (std::filesystem::temp_directory_path() / "coco_subsystem_test.state").string();
    { laminar::state::Writer w(file); laminar::state::io(w, a); w.finish(); }
    GoalControlledRelationalSession b(99, 16, true);   // different seed: everything must come from the checkpoint
    { laminar::state::Reader r(file); laminar::state::io(r, b); r.finish(); }
    require(fingerprint(a) == fingerprint(b), "restored subsystem has the saved fingerprint");
    for (unsigned i = 0; i < 4; ++i) {
        const auto ta = mission(a, "cont" + std::to_string(i), i + 6), tb = mission(b, "cont" + std::to_string(i), i + 6);
        require(ta == tb, "continued missions diverge after restore");
    }
    require(fingerprint(a) == fingerprint(b), "continued subsystems stay identical");
    require(a.policy.weights == b.policy.weights && a.updates() == b.updates() && a.knowledge.version() == b.knowledge.version(), "policy, updates and knowledge identical");
    std::filesystem::remove(file);
    std::cout << "Subsystem checkpoint: exact save/load fingerprint and identical continued learning (6 warm + 4 continued stopping missions) passed\n";
    return 0;
}
int main() { try { return run(); } catch (const std::exception& e) { std::cerr << "FAILED: " << e.what() << '\n'; return 1; } }
