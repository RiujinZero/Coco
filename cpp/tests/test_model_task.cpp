// Contracts for the model task learner's integration points:
//  1. An external (model) decision executes through the goal controller without the
//     controller's own policy being evaluated, and carries its provenance.
//  2. An external decision cannot be mixed with a teacher hint, epsilon or memory policy,
//     and must name a legal operation.
//  3. The model checkpoint round-trips exactly: continuing from a loaded state produces
//     the same predictions and the same final fingerprint as continuing uninterrupted,
//     including the agency kernel's task channel and the task readout.
#include "laminar/goal_control.h"
#include "laminar/model_task.h"
#include "goal_control_fixture.h"
#include <filesystem>
#include <iostream>

using namespace laminar::relational;
using laminar::goal_test::require;

static PublicGoal request(const laminar::goal_test::PublicRequestFixture& source) {
    PublicGoal goal; goal.id = source.id; goal.world_id = source.world_id; goal.text = source.text;
    goal.names = source.names; goal.visible = source.visible; goal.actions = source.actions;
    goal.effect = source.effect; goal.deadline = source.deadline; return goal;
}
static GoalControlLimits limits() { GoalControlLimits r; r.derived = 0; r.decisions = 96; r.motors = 32; r.preview.work = 100000; return r; }
static GoalControlPrices prices() { GoalControlPrices r; r.preview = {.00001, .00001, .00001, .00001, .00001}; return r; }
static void external_decision() {
    auto fixture = laminar::goal_test::make_fixture();
    GoalControlledRelationalSession learner(7); learner.knowledge = fixture.make_knowledge(7);
    require(learner.begin_mission("ext", limits(), prices()), "mission begins");
    require(learner.offer_goal(request(fixture.request("physical"))), "public goal offered");
    const auto weights = learner.policy.weights;
    // Bias the controller's own policy away from next_goal: if it were evaluated it would not agree.
    for (auto& w : learner.policy.weights) w = 0;
    learner.policy.weights[learner.policy.output_bias() + int(ControlOp::activate_goal)] = 5;
    const auto biased = learner.policy.weights;
    auto step = learner.choose(0, false, std::nullopt, ControlHint{ControlOp::next_goal, "model_readout"});
    require(step.operation == ControlOp::next_goal && step.decider == "model_readout" && step.teacher.empty(), "external decision executes with provenance");
    require(learner.policy.weights == biased, "controller policy untouched");
    bool rejected = false;
    try { learner.choose(0, false, ControlHint{ControlOp::activate_goal, "teacher"}, ControlHint{ControlOp::activate_goal, "model_readout"}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "external decision cannot be mixed with a teacher hint");
    rejected = false;
    try { learner.choose(0.5f, false, std::nullopt, ControlHint{ControlOp::activate_goal, "model_readout"}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "external decision cannot be mixed with exploration");
    rejected = false;
    try { learner.choose(0, false, std::nullopt, ControlHint{ControlOp::act, "model_readout"}); }   // act is not legal in the agenda
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "external decision must be legal");
    learner.policy.weights = weights;
    // a frozen controller still executes the model's decisions (tools stay frozen while the model learns)
    GoalControlledRelationalSession frozen(7); frozen.knowledge = fixture.make_knowledge(7); frozen.set_learning(false);
    require(frozen.begin_mission("ext2", limits(), prices()) && frozen.offer_goal(request(fixture.request("physical"))), "frozen mission");
    auto frozen_step = frozen.choose(0, false, std::nullopt, ControlHint{ControlOp::next_goal, "model_readout"});
    require(frozen_step.decider == "model_readout", "frozen tools execute model decisions");
}

static laminar::Params small_params() {
    const char* args[] = {"t", "vocab=256", "units=4096", "hops=4", "hops_free=2", "hidden=modulated", "acc_bits=16", "agency=1", "agency_task=1",
                          "agency_trace_life=32", "seq_window=65536", "seq_bits=12", "seq_cbits=12", "episodic=1", "hist=7",
                          "agency_ops=hold,attend_episodic,attend_generic,to_memory,to_recent"};
    return laminar::Params::from_args((int)(sizeof(args) / sizeof(args[0])), const_cast<char**>(args));
}
static void drive(laminar::TaskModel& tm, int steps, int salt) {
    for (int i = 0; i < steps; ++i) {
        tm.observe(laminar::TaskChannel::observation, "ArBr" + std::string(1, (char)('a' + (i + salt) % 7)), true);
        std::array<float, 8> state{}; state[0] = (float)(i % 2); tm.set_task_state(state);
        auto c = tm.decide({1, 5, 6}, i % 3 == 0);
        const int label = (i + salt) % 2 ? 5 : 6;
        tm.act(i % 3 == 0 ? c.op : label, i % 3 == 0, i % 3 != 0);
        tm.tool_outcome(i % 4 == 0 ? 1.f : 0.5f);
        tm.observe(laminar::TaskChannel::tool_result, "interpreted reply true", true);
    }
    tm.mission_outcome(salt % 2 ? 1.f : -1.f); tm.end_mission();
}
static void checkpoint_roundtrip() {
    const std::string file = (std::filesystem::temp_directory_path() / "coco_model_task_test.state").string();
    laminar::Organism a(small_params()); laminar::TaskModel ta(a, {}, 5);
    drive(ta, 40, 1);
    { laminar::state::Writer w(file); a.save_state(w); ta.state_io(w); w.finish(); }
    drive(ta, 30, 2);
    const uint64_t uninterrupted = [&] { laminar::state::Writer w; a.save_state(w); ta.state_io(w); return w.fingerprint(); }();
    laminar::Organism b(small_params()); laminar::TaskModel tb(b, {}, 5);
    { laminar::state::Reader r(file); b.load_state(r); tb.state_io(r); r.finish(); }
    drive(tb, 30, 2);
    const uint64_t resumed = [&] { laminar::state::Writer w; b.save_state(w); tb.state_io(w); return w.fingerprint(); }();
    require(uninterrupted == resumed, "resumed model reaches the uninterrupted state");
    require(tb.metrics.l6_task_updates > 0 && tb.metrics.modulated_task_updates > 0, "task credit reached the bodies");
    require(b.kernel().routes_updated > 0, "agency kernel received consequences");
    // a corrupted file is refused
    { std::FILE* f = std::fopen(file.c_str(), "r+b"); std::fseek(f, 4096, SEEK_SET); int c = std::fgetc(f); std::fseek(f, 4096, SEEK_SET); std::fputc(c ^ 0x5A, f); std::fclose(f); }
    bool refused = false;
    try { laminar::Organism c(small_params()); laminar::TaskModel tc(c, {}, 5); laminar::state::Reader r(file); c.load_state(r); tc.state_io(r); r.finish(); }
    catch (const std::exception&) { refused = true; }
    require(refused, "corrupt checkpoint refused");
    std::filesystem::remove(file);
    // body output ablation removes all workspace input to the readout
    laminar::Organism d(small_params()); laminar::TaskModel td(d, {}, 5); drive(td, 20, 3);
    td.ablate_body_output = true; d.set_mode(false, false);
    td.observe(laminar::TaskChannel::observation, "ArBrc", true);
    auto c1 = td.decide({1, 5, 6}, false);
    td.observe(laminar::TaskChannel::observation, "QrRrdrejfr", true);
    auto c2 = td.decide({1, 5, 6}, false);
    require(c1.policy == c2.policy, "ablated readout ignores body activity");
}

// Grouped attempts: a success is kept, and a failure that diverged from it in the same state becomes a correction.
static void grouped_attempts() {
    laminar::TaskModelConfig cfg; cfg.success_replay = 1; cfg.success_prune = true;
    laminar::Organism m(small_params()); laminar::TaskModel tm(m, cfg, 9); drive(tm, 5, 1);
    auto attempt = [&](int second, bool ok) {
        m.set_mode(false, false); m.replay = true; tm.begin_attempt();
        tm.observe(laminar::TaskChannel::observation, "Ar", true); tm.decide({1, 5}, true); tm.act(1, true, false);
        tm.observe(laminar::TaskChannel::observation, "Br", true); tm.decide({1, 5}, true); tm.act(second, true, false);
        tm.end_attempt(ok); m.replay = false; m.set_mode(true, true);
    };
    const size_t before = tm.success_tapes();
    attempt(5, true); attempt(1, false); attempt(1, false);
    auto g = tm.finish_attempts(3);
    require(g.successes == 1 && g.failures == 2 && g.stored == 1 && g.corrections == 2, "success stored, both diverging failures corrected");
    require(tm.success_tapes() == before + 3, "success and corrections enter the success store");
    attempt(1, false); attempt(5, false);
    auto none = tm.finish_attempts(3);
    require(none.stored == 0 && none.corrections == 0 && tm.success_tapes() == before + 3, "no success: nothing to contrast, nothing stored");
    attempt(5, true); attempt(5, true);
    auto all = tm.finish_attempts(3);
    require(all.stored == 1 && all.corrections == 0, "all successes: one tape");
}

int main() {
    external_decision();
    checkpoint_roundtrip();
    grouped_attempts();
    std::cout << "Model task learner: external decision provenance, exclusive selection, frozen tools, exact checkpoint resume, corruption refusal, body-output ablation and grouped-attempt credit passed\n";
}
