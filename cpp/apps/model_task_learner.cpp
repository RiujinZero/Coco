// The model as the task learner (docs/MODEL_TASK_LEARNER.md).
//
// One persistent model (bodies, routing, persistent workspace, sequence memory,
// agency kernel) performs the mixed goal curriculum. The existing subsystem
// supplies teaching, memory and tools: the seven-operation goal controller, the
// relational knowledge/computation library and the reader are built once with
// the same declared setup as the continuing bootstrap, then FROZEN and copied
// fresh for every mission. They execute the operation the model chooses and
// return public results; their own 32-unit policy is never evaluated. The
// declared teacher (continuing_curriculum_v1) supplies demonstrations during the
// guided and tapering phases and is absent in autonomous training and probes.
//
// usage: model_task_learner PROFILE ARM DIR [--stop-after N] [--teacher rule|policy:FILE:LABEL]
//                            [--pretrain FILE --pretrain-bytes N] [key=value ...]
//   PROFILE smoke|resume|full    ARM learned|frozen_bodies|fixed_agency|readout_only
// --teacher policy:... replaces the hand-written demonstration rule with a trained subsystem policy (public
// 32 floats + menu only). --pretrain runs the model's ordinary byte-prediction learning over a corpus before
// any task; the frozen-bodies arm then freezes the pretrained bodies.
// A fresh DIR declares the run; an existing DIR resumes from its checkpoint.
#define LAMINAR_GOAL_MIXED_LIBRARY
#include "goal_continuing_bootstrap.cpp"
#include "laminar/model_task.h"
#include "subsystem_policy.h"
#include <csignal>

using laminar::Organism;
using laminar::Params;
using laminar::TaskModel;
using laminar::TaskModelConfig;
using laminar::TaskChannel;
namespace fs = std::filesystem;

static constexpr std::array<size_t, 20> family_cycle = {1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0, 3, 4, 5, 6, 7, 0};
static std::string teacher_tag = "teacher_continuing_curriculum_v1";
static std::function<int(const gl::Observation&, const std::vector<int>&)> teacher_fn = gl::curriculum_demonstration;
static constexpr const char* model_tag = "model_readout";
static std::vector<uint8_t> interleave_text, eval_text;   // loaded once at startup (see --interleave, --text-eval)
static uint64_t fnv(const std::vector<uint8_t>& v) { uint64_t h = 1469598103934665603ULL; for (uint8_t c : v) { h ^= c; h *= 1099511628211ULL; } return h; }
static std::vector<uint8_t> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary); gl::require(bool(f), "cannot open text file");
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

struct Profile {
    std::string name, arm;
    std::vector<std::string> extra;   // declared model-parameter overrides (key=value), applied after the profile's own
    std::string teacher = "rule";     // rule | policy:FILE:LABEL
    uint64_t teacher_digest = 0;
    std::string pretrain_file; uint64_t pretrain_bytes = 0, pretrain_digest = 0;
    bool critic = false; uint32_t rehearsal = 0;   // outcome-credit variants (--critic, --rehearsal N)
    bool no_outcome = false;                        // --no-outcome: mission outcomes train nothing (readout or bodies)
    uint32_t reencode = 0; bool reencode_bodies = false;   // --reencode N [--reencode-bodies]: replay stored situations through current bodies
    uint32_t replay_missions = 0;                           // --replay-missions K: replay whole demonstrated missions (tapes)
    uint32_t success_replay = 0, success_budget = 0;        // --success-replay K [--success-budget B]: replay verified own successes (self-imitation)
    bool success_solo = false, success_prune = false, success_stratify = false;   // --success-solo --success-prune --success-stratify
    // Curriculum mixing and forgetting measurement:
    //   --interleave FILE --interleave-bytes N : before every training mission the model reads the next N bytes of
    //     FILE (wrapping) with ordinary learning on, then structural maintenance runs (as at a mission end).
    //   --text-eval FILE : at every evaluation checkpoint the frozen model reads FILE (held-out text) and its
    //     bits per byte are logged next to the probes.
    uint32_t attempts = 0;   // --attempts K: every teacherless training mission is first tried K times frozen (grouped outcome credit)
    uint32_t text_replay_kb = 0;   // --text-replay-kb N: one extra teacher tape replayed per N KiB of interleaved text per mission
    std::string interleave_file, text_eval_file; uint64_t interleave_bytes = 0, interleave_digest = 0, text_eval_digest = 0;
    std::string withhold;                                   // --withhold FAMILY: the teacher never acts in this family's missions
    uint32_t units = 16384;
    size_t total = 48, full = 20, taper_end = 40, checkpoint_every = 8, eval_every = 0, probes = 8;
    std::vector<std::string> params() const {
        std::vector<std::string> v = {"vocab=256", "units=" + std::to_string(units), "hops=4", "hops_free=2", "hidden=modulated", "acc_bits=16",
            "agency=" + std::string(arm == "fixed_agency" ? "2" : "1"), "agency_task=1", "agency_trace_life=32",
            "seq_window=1048576", "seq_bits=16", "seq_cbits=16", "seed=1", "agency_seed=7",
            "episodic=1", "hist=7", "agency_ops=hold,attend_episodic,attend_generic,to_memory,to_recent"};
        v.insert(v.end(), extra.begin(), extra.end());
        return v;
    }
    std::string declaration() const {
        std::ostringstream o;
        o << "{\"schema\":\"model_task_learner_v1\",\"profile\":" << gl::quote(name) << ",\"arm\":" << gl::quote(arm)
          << ",\"model_params\":"; gl::strings(o, params());
        TaskModelConfig c; c.body_learning = arm == "learned" || arm == "fixed_agency"; c.critic = critic; c.rehearsal = rehearsal; c.reencode = reencode; c.reencode_bodies = reencode_bodies; c.replay_missions = replay_missions; c.success_replay = success_replay; c.success_budget = success_budget; c.success_solo = success_solo; c.success_prune = success_prune; c.success_stratify = success_stratify; if (no_outcome) { c.policy_lr = 0; c.outcome_mod = 0; c.body_credit = 0; }   // == task_config(*this)
        o << ",\"task_model\":{\"actions\":" << c.actions << ",\"readout\":\"" << c.actions << "x" << TaskModel::D
          << " linear (decision workspace, persistent workspace, pooled observation+decision workspace, bias); NLMS steps\""
          << ",\"readout_lr\":" << c.readout_lr << ",\"policy_lr\":" << c.policy_lr << ",\"l6_gain\":" << c.l6_gain << ",\"demo_mod\":" << c.demo_mod
          << ",\"outcome_mod\":" << c.outcome_mod << ",\"body_credit\":" << c.body_credit << ",\"route_credit\":" << c.route_credit << ",\"gamma\":" << c.gamma
          << ",\"baseline_rate\":" << c.baseline_rate << ",\"explore\":" << c.explore << ",\"max_pooled_events\":" << TaskModel::max_pooled_events
          << ",\"critic\":" << (c.critic ? "true" : "false") << ",\"critic_lr\":" << c.critic_lr << ",\"rehearsal\":" << c.rehearsal << ",\"rehearsal_capacity\":" << c.rehearsal_capacity << ",\"reencode\":" << c.reencode << ",\"reencode_context\":" << c.reencode_context << ",\"reencode_bodies\":" << (c.reencode_bodies ? "true" : "false") << ",\"replay_missions\":" << c.replay_missions << ",\"tape_capacity\":" << c.tape_capacity << ",\"success_replay\":" << c.success_replay << ",\"success_capacity\":" << c.success_capacity << ",\"success_budget\":" << c.success_budget << ",\"success_solo\":" << (c.success_solo ? "true" : "false") << ",\"success_prune\":" << (c.success_prune ? "true" : "false") << ",\"success_stratify\":" << (c.success_stratify ? "true" : "false") << ",\"mission_tag\":\"task family\""
          << ",\"body_learning\":" << (c.body_learning ? "true" : "false") << ",\"freeze_weights\":" << (arm == "frozen_bodies" ? "true,\"frozen_note\":\"structure and birth weights still allocate; no synapse changes\"" : "false") << "}"
          << ",\"total_missions\":" << total << ",\"full_guidance\":" << full << ",\"taper_end\":" << taper_end
          << ",\"family_cycle\":[1,2,1,2,1,2,0,1,2,1,2,1,2,0,3,4,5,6,7,0]"
          << ",\"checkpoint_every\":" << checkpoint_every << ",\"eval_every\":" << eval_every << ",\"probes\":" << probes
          << ",\"tools\":\"GoalControlledRelationalSession(seed1,hidden32 never evaluated) + raw fixture + reader + long extension + initial and extension skills, rebuilt then frozen; copied per mission; reader cloned per mission\""
          << ",\"teacher\":" << gl::quote(teacher == "rule" ? std::string("continuing_curriculum_v1 hand-written rule") : "trained subsystem policy " + teacher)
          << ",\"teacher_digest\":\"" << std::hex << teacher_digest << std::dec << "\""
          << ",\"teacher_protocol\":\"public 32 floats and legal menu only; consulted with the guidance probability; never in autonomous training or probes\""
          << ",\"interleave\":{\"file\":" << gl::quote(interleave_file) << ",\"bytes_per_mission\":" << interleave_bytes << ",\"digest\":\"" << std::hex << interleave_digest << std::dec << "\"}"
          << ",\"text_eval\":{\"file\":" << gl::quote(text_eval_file) << ",\"digest\":\"" << std::hex << text_eval_digest << std::dec << "\"}"
          << ",\"text_replay_kb\":" << text_replay_kb << ",\"grouped_attempts\":" << attempts << ",\"grouped_protocol\":\"teacherless training missions (guidance 0) are first attempted K times with the model frozen and sampling; successes become success tapes, failures a correction at the first divergence from a success; then the ordinary learning mission runs\""
          << ",\"withheld_family\":" << gl::quote(withhold) << ",\"withheld_protocol\":\"the teacher is never consulted in missions of the withheld family (no demonstrations, no labels); the model acts and samples; only the environment's verified outcome is available\""
          << ",\"pretrain\":{\"file\":" << gl::quote(pretrain_file) << ",\"bytes\":" << pretrain_bytes << ",\"digest\":\"" << std::hex << pretrain_digest << std::dec
          << "\",\"protocol\":\"ordinary byte prediction (all body learning rules, structure on) before any task; frozen_bodies freezes afterwards\"}"
          << ",\"autonomous\":\"model samples from (1-0.1)*pi+0.1*uniform; probes use the readout argmax\""
          << ",\"probes_protocol\":\"each probe mission starts from the checkpointed model, frozen (no learning, no structure), no teacher; repeated with body output ablated\""
          << ",\"limits\":\"mixed_limits(family), zero prices, 256 decisions, 32 motors\"}\n";
        return o.str();
    }
};
static Profile make_profile(const std::string& name, const std::string& arm) {
    Profile p; p.name = name; p.arm = arm;
    if (arm != "learned" && arm != "frozen_bodies" && arm != "fixed_agency" && arm != "readout_only") throw std::runtime_error("unknown arm " + arm);
    if (name == "smoke") { p.units = 16384; p.total = 48; p.full = 20; p.taper_end = 40; p.checkpoint_every = 8; p.eval_every = 0; p.probes = 8; }
    else if (name == "resume") { p.units = 16384; p.total = 12; p.full = 4; p.taper_end = 8; p.checkpoint_every = 3; p.eval_every = 6; p.probes = 2; }
    else if (name == "full") { p.units = 200000; p.total = 2000; p.full = 800; p.taper_end = 1600; p.checkpoint_every = 20; p.eval_every = 400; p.probes = 46; }
    else throw std::runtime_error("unknown profile " + name);
    return p;
}
static double guidance(const Profile& p, size_t index) {
    if (index < p.full) return 1.0;
    if (index >= p.taper_end) return 0.0;
    return double(p.taper_end - 1 - index) / double(p.taper_end - p.full);
}
static Params model_params(const Profile& p) {
    std::vector<std::string> args = p.params(); std::vector<char*> argv; std::string prog = "model";
    argv.push_back(prog.data()); for (auto& a : args) argv.push_back(a.data());
    return Params::from_args((int)argv.size(), argv.data());
}
static TaskModelConfig task_config(const Profile& p) {
    TaskModelConfig c; c.body_learning = p.arm == "learned" || p.arm == "fixed_agency"; c.critic = p.critic; c.rehearsal = p.rehearsal; c.reencode = p.reencode; c.reencode_bodies = p.reencode_bodies; c.replay_missions = p.replay_missions; c.success_replay = p.success_replay; c.success_budget = p.success_budget; c.success_solo = p.success_solo; c.success_prune = p.success_prune; c.success_stratify = p.success_stratify; if (p.no_outcome) { c.policy_lr = 0; c.outcome_mod = 0; c.body_credit = 0; } return c;
}

// ---- rendering public task content as bytes ----------------------------------------------------------------------
static std::string render_scene(const StampedScene& s, size_t cap = 200) {
    std::string o = "rev " + std::to_string(s.revision) + (s.scene.complete ? " complete:" : " partial:");
    for (const auto& a : s.scene.facts) { o += ' '; o += a.predicate; for (const auto& x : a.args) { o += ' '; o += x; } o += ';'; if (o.size() >= cap) break; }
    if (o.size() > cap) o.resize(cap);
    return o;
}
// The controller's 32 public floats: each nonzero feature as an index letter and a signed level.
static std::string render_observation(const gl::Observation& o) {
    std::string s;
    for (int i = 0; i < 32; ++i) {
        if (o[i] == 0.f) continue;
        const int q = std::clamp((int)std::lround(o[i] * 8.f), -8, 8);
        s += (char)(i < 26 ? 'A' + i : 'a' + (i - 26)); s += (char)('j' + q);
    }
    return s;
}
static const char* answer_name(Answer a) { return a == Answer::True ? "true" : a == Answer::False ? "false" : "unknown"; }

struct Logs {
    std::ofstream missions, decisions, evaluations, timing, bootstrap;
    explicit Logs(const fs::path& dir, bool append) {
        const auto mode = append ? std::ios::app : std::ios::trunc;
        missions.open(dir / "missions.jsonl", mode); decisions.open(dir / "decisions.jsonl", mode); evaluations.open(dir / "evaluations.jsonl", mode);
        timing.open(dir / "timing.jsonl", mode); bootstrap.open(dir / "tools_bootstrap.jsonl", std::ios::trunc);
        for (auto* f : {&missions, &decisions, &evaluations, &timing, &bootstrap}) { gl::require(bool(*f), "cannot open log"); *f << std::boolalpha << std::setprecision(9); }
    }
};

struct MissionSpec {
    std::string id, phase; size_t index = 0, family = 0;
    double guidance = 0; bool learning = true, sample = false;
    uint32_t assistance_seed = 0;
};
struct MissionOutcome {
    bool success = false; size_t decisions = 0, model_executed = 0, demonstrations = 0, agree = 0, rejected = 0, posted = 0;
    bool interrupted = false, resumed = false;
};

// One mission: the frozen tools execute; the model observes, decides and learns.
static MissionOutcome run_mission(TaskModel& tm, Organism& model, const GoalControlledRelationalSession& tools, gl::Environment& env,
                                  const MissionSpec& spec, std::ostream& dlog) {
    GoalControlledRelationalSession session = tools;
    const auto limits = mixed_limits(spec.family); const auto prices = st::prices();
    gl::require(session.begin_mission(spec.id, limits, prices), "mission begin rejected");
    const auto& schedule = env.schedule();
    gl::require(!schedule.empty() && schedule.front().clock == 0, "missing initial public request");
    MissionOutcome out; std::mt19937 assistance(spec.assistance_seed);
    std::set<std::string> interrupted_goals;
    float last_value = 0.5f, last_confidence = 0.f; bool last_rejected = false;
    auto post = [&]() {
        while (out.posted < schedule.size() && schedule[out.posted].clock == session.clock()) {
            auto goal = schedule[out.posted].request; goal.visible = env.read(goal.world_id);
            gl::require(session.offer_goal(goal), "scheduled public input rejected"); ++out.posted;
            std::string text = goal.text + " | deadline " + std::to_string(goal.deadline) + " | " + render_scene(goal.visible, 160);
            tm.observe(TaskChannel::arrival, text, true);
        }
    };
    post();
    std::string inspected_seen;
    while (session.clock() < limits.decisions && !gl::closed(session)) {
        post();
        const auto observation = session.observe(); const auto menu = gl::public_menu(session, limits);
        tm.observe(TaskChannel::observation, render_observation(observation), true);
        tm.set_task_state({observation[29], observation[12], observation[30], observation[16], observation[17], last_rejected ? 1.f : 0.f, last_value, last_confidence});
        const bool consult = spec.guidance > 0 && std::generate_canonical<double, 53>(assistance) < spec.guidance;
        const int label = consult ? teacher_fn(observation, menu) : -1;
        const auto choice = tm.decide(menu, spec.sample && !consult);
        const int executed = consult ? label : choice.op;
        const auto active_before = session.active_goal();
        const auto issued = session.choose(0, false, std::nullopt, ControlHint{ControlOp(executed), consult ? teacher_tag : model_tag});
        gl::require(issued.decider == (consult ? teacher_tag : model_tag) && issued.teacher.empty(), "decision provenance drift");
        tm.act(executed, !consult, consult);
        out.decisions++; out.model_executed += !consult; out.demonstrations += consult; out.agree += consult && choice.op == label;
        // tool callbacks: identical protocol to the goal harness
        std::string evidence; bool correct = false, satisfaction = false, finished = false;
        if (issued.operation == ControlOp::activate_goal) {
            gl::require(session.activate(issued.token), "owned activation rejected");
            if (!active_before.empty() && active_before != session.active_goal()) { interrupted_goals.insert(active_before); out.interrupted = true; }
            out.resumed |= interrupted_goals.count(session.active_goal()) != 0;
        } else if (issued.operation == ControlOp::act && !issued.actions.empty()) {
            const auto& world = session.goals().at(active_before).request.world_id;
            std::vector<Scene> observed; std::vector<bool> exec; StampedScene last;
            for (const auto& action : issued.actions) { last = env.execute(world, action); observed.push_back(last.scene); exec.push_back(true); }
            gl::require(session.actions(issued.token, observed, exec, last.revision), "owned physics callback rejected");
            evidence = "executed " + std::to_string(issued.actions.size()) + " | " + render_scene(last);
        } else if (issued.operation == ControlOp::ask || issued.operation == ControlOp::retrieve) {
            const auto view = env.read(session.goals().at(active_before).request.world_id);
            gl::require(session.read(issued.token, view, issued.operation == ControlOp::ask ? "verified_environment" : "trusted_memory"), "owned read callback rejected");
            evidence = render_scene(view);
        } else if (issued.operation == ControlOp::finish) {
            satisfaction = env.satisfaction(session.goals().at(active_before).request);
            correct = issued.reply == (satisfaction ? Answer::True : Answer::False);
            if (session.goals().at(active_before).request.effect) correct &= satisfaction;
            gl::require(session.close_goal(issued.token, correct, satisfaction), "owned goal label rejected"); finished = true;
        }
        const auto& row = session.mission_trace().back();
        last_rejected = row.rejected; last_confidence = choice.confidence; out.rejected += row.rejected;
        last_value = row.rejected ? 0.f : finished ? (correct ? 1.f : 0.f) : 0.5f;
        tm.tool_outcome(last_value);
        tm.observe(TaskChannel::tool_result, row.status + " reply " + answer_name(row.reply) + (row.rejected ? " rejected" : ""), true);
        if (!evidence.empty()) tm.observe(TaskChannel::evidence, evidence, true);
        if (finished) tm.observe(TaskChannel::outcome, correct ? "goal closed correct" : "goal closed incorrect", true);
        // a newly inspected goal: a public request, or a controller-generated proposal (never learned as evidence)
        if (session.in_agenda() && !session.inspected_goal().empty() && session.inspected_goal() != inspected_seen) {
            inspected_seen = session.inspected_goal();
            const auto& g = session.goals().at(inspected_seen).request;
            tm.observe(g.generated ? TaskChannel::proposal : TaskChannel::request, g.text, !g.generated);
        }
        dlog << "{\"mission\":" << gl::quote(spec.id) << ",\"clock\":" << row.clock << ",\"menu\":"; gl::numbers(dlog, menu);
        dlog << ",\"observation\":" << gl::quote(render_observation(observation)) << ",\"model_choice\":" << choice.op << ",\"policy\":[";
        for (size_t i = 0; i < menu.size(); ++i) { if (i) dlog << ','; dlog << choice.policy[menu[i]]; }
        dlog << "],\"teacher\":" << label << ",\"executed\":" << executed << ",\"decider\":" << gl::quote(issued.decider)
             << ",\"status\":" << gl::quote(row.status) << ",\"rejected\":" << row.rejected << "}\n";
    }
    bool known_correct = gl::closed(session);
    for (const auto& s : session.submissions()) if (s.parent.empty()) known_correct &= s.correct && (!s.deadline || s.clock <= s.deadline);
    gl::require(!known_correct || out.posted == schedule.size(), "successful early close omitted a required future request");
    out.success = known_correct && out.posted == schedule.size();
    gl::require(session.finish_mission(0, false), "mission close rejected");
    tm.tool_outcome(out.success ? 1.f : 0.f);
    tm.observe(TaskChannel::outcome, out.success ? "mission success" : "mission failure", true);
    tm.mission_outcome(out.success ? 1.f : -1.f);
    tm.end_mission();
    (void)model;
    return out;
}

// ---- persistent run state ----------------------------------------------------------------------------------------
struct RunState {
    uint64_t next = 0;                       // next training mission index
    std::array<uint64_t, 8> occurrences{}, successes{};
    uint64_t guided_success = 0, autonomous_success = 0;
    int64_t last_eval = -1;                  // mission count at which the last evaluation completed (-1: none)
    uint64_t text_pos = 0;                   // next interleaved text byte
    std::vector<uint64_t> log_sizes;         // missions, decisions, evaluations at the checkpoint
    template <class Ar> void state_io(Ar& ar) { ar(next); ar(occurrences); ar(successes); ar(guided_success); ar(autonomous_success); ar(last_eval); ar(log_sizes); ar(text_pos); }
};
static void save_checkpoint(const fs::path& dir, Organism& model, TaskModel& tm, RunState& rs, Logs& logs) {
    for (auto* f : {&logs.missions, &logs.decisions, &logs.evaluations}) f->flush();
    rs.log_sizes = {(uint64_t)fs::file_size(dir / "missions.jsonl"), (uint64_t)fs::file_size(dir / "decisions.jsonl"), (uint64_t)fs::file_size(dir / "evaluations.jsonl")};
    const fs::path tmp = dir / "checkpoint.state.tmp", cur = dir / "checkpoint.state", prev = dir / "checkpoint.prev.state";
    { laminar::state::Writer w(tmp.string()); model.save_state(w); tm.state_io(w); w.section("run"); rs.state_io(w); w.finish(); }
    { laminar::state::Reader r(tmp.string()); }   // magic readable before it replaces the current checkpoint
    if (fs::exists(cur)) fs::rename(cur, prev);
    fs::rename(tmp, cur);
}
static void load_checkpoint(const fs::path& file, Organism& model, TaskModel& tm, RunState& rs) {
    laminar::state::Reader r(file.string()); model.load_state(r); tm.state_io(r); r.section("run"); rs.state_io(r); r.finish();
}
static uint64_t fingerprint(Organism& model, TaskModel& tm) {
    laminar::state::Writer w; model.save_state(w); tm.state_io(w); return w.fingerprint();
}

struct Telemetry {
    static void mission(std::ostream& o, const MissionSpec& spec, const MissionOutcome& r, Organism& model, const TaskModel& tm, const RunState& rs) {
        const auto& k = model.kernel();
        o << "{\"id\":" << gl::quote(spec.id) << ",\"phase\":" << gl::quote(spec.phase) << ",\"index\":" << spec.index << ",\"family\":" << gl::quote(family_names[spec.family])
          << ",\"guidance\":" << spec.guidance << ",\"success\":" << r.success << ",\"decisions\":" << r.decisions << ",\"model_executed\":" << r.model_executed
          << ",\"demonstrations\":" << r.demonstrations << ",\"agreement\":" << r.agree << ",\"rejected\":" << r.rejected
          << ",\"interrupted\":" << r.interrupted << ",\"resumed\":" << r.resumed
          << ",\"live_units\":" << model.live_units() << ",\"exact_locks\":" << model.exact_locks()
          << ",\"task_l6_updates\":" << tm.metrics.l6_task_updates << ",\"task_modulated_updates\":" << tm.metrics.modulated_task_updates
          << ",\"credited_bodies\":" << tm.metrics.credited_bodies << ",\"credited_routes\":" << tm.metrics.credited_routes
          << ",\"bytes\":" << tm.metrics.bytes << ",\"baseline\":" << tm.baseline() << ",\"episodic\":[" << model.episodic_totals().births << ',' << model.episodic_totals().deferred << ',' << model.episodic_totals().evicted << ',' << model.episodic_totals().evicted_unused << ',' << model.episodic_totals().promoted << ',' << model.episodic_totals().promoted_splits << ',' << model.episodic_totals().retired << ']' << ",\"success_tapes\":" << tm.success_tapes() << ",\"successes_stored\":" << tm.metrics.successes_stored << ",\"labels_pruned\":" << tm.metrics.labels_pruned << ",\"corrections_stored\":" << tm.metrics.corrections_stored << ",\"attempts\":" << tm.metrics.attempts
          << ",\"agency_commits\":" << k.commitment.commits << ",\"agency_routes_updated\":" << k.routes_updated << ",\"agency_emitted\":"; gl::numbers(o, k.emitted_count);
        o << ",\"cumulative_successes\":" << (rs.guided_success + rs.autonomous_success) << "}\n";
        o.flush();
    }
};

// Frozen probes: every probe mission starts from the checkpoint just written; nothing a probe does persists.
static int64_t abort_during_eval = -1;   // resume-test hook: the process dies (exit 3) after one probe of this evaluation
static std::array<size_t, 2> evaluate(const Profile& prof, const fs::path& dir, const std::string& label, Organism& model, TaskModel& tm,
                                      const GoalControlledRelationalSession& tools, const rt::Component& reader, Logs& logs, uint64_t at) {
    std::array<size_t, 2> total{};
    for (int ablate = 0; ablate < 2; ++ablate) {
        std::array<size_t, 8> fam_success{}, fam_cases{}; size_t successes = 0, decisions = 0;
        for (size_t index = 0; index < prof.probes; ++index) {
            RunState ignored; load_checkpoint(dir / "checkpoint.state", model, tm, ignored);
            model.set_mode(false, false); tm.ablate_body_output = ablate == 1;
            const size_t family = index < 32 ? index % 8 : 0;
            auto reader_copy = reader.clone();
            MissionSpec spec; spec.id = "eval_" + label + (ablate ? "_ablated_" : "_") + std::to_string(index); spec.phase = ablate ? "probe_ablated" : "probe";
            spec.index = index; spec.family = family; spec.guidance = 0; spec.learning = false; spec.sample = false;
            std::unique_ptr<gl::Environment> env;
            if (index < 32) env = mixed_environment(family, index, true, prof.name != "full", reader_copy, logs.bootstrap, spec.id);
            else env = std::make_unique<st::Environment>(st::heldout_cases().at(index - 32));
            auto r = run_mission(tm, model, tools, *env, spec, logs.decisions);
            successes += r.success; fam_success[family] += r.success; ++fam_cases[family]; decisions += r.decisions;
            if (abort_during_eval >= 0 && (uint64_t)abort_during_eval == at) { logs.decisions.flush(); std::cout << "aborting inside evaluation at " << at << '\n' << std::flush; std::_Exit(3); }
        }
        logs.evaluations << "{\"checkpoint\":" << gl::quote(label) << ",\"body_output_ablated\":" << (ablate == 1) << ",\"successes\":" << successes
                         << ",\"cases\":" << prof.probes << ",\"decisions\":" << decisions << ",\"family_successes\":"; gl::numbers(logs.evaluations, fam_success);
        logs.evaluations << ",\"family_cases\":"; gl::numbers(logs.evaluations, fam_cases); logs.evaluations << "}\n"; logs.evaluations.flush();
        std::cout << "evaluation " << label << (ablate ? " ablated " : " ") << successes << '/' << prof.probes << '\n' << std::flush;
        total[ablate] = successes;
    }
    tm.ablate_body_output = false;
    if (!eval_text.empty()) {   // held-out text, frozen, from the checkpointed state
        RunState ignored; load_checkpoint(dir / "checkpoint.state", model, tm, ignored);
        model.set_mode(false, false); model.stats() = laminar::BlockStats{};
        tm.read_text(eval_text.data(), eval_text.size());
        const double bits = model.stats().loss_sum / std::max<uint64_t>(1, model.stats().events) / 0.6931471805599453;
        logs.evaluations << "{\"checkpoint\":" << gl::quote(label) << ",\"text_bits_per_byte\":" << bits << ",\"text_bytes\":" << eval_text.size() << "}\n"; logs.evaluations.flush();
        std::cout << "evaluation " << label << " text " << bits << " bits/byte\n" << std::flush;
        model.stats() = laminar::BlockStats{};
    }
    RunState ignored; load_checkpoint(dir / "checkpoint.state", model, tm, ignored);   // live model exactly as before the probes
    return total;
}

static void build_tools(GoalControlledRelationalSession& tools, rt::Component& reader, std::ostream& bootstrap) {
    tools.knowledge = RelationalSession(1, 32768, 1024); ms::configure_initial(tools);
    auto fixture = gt::make_fixture(); fixture.lessons.apply(tools.knowledge); export_bootstrap(fixture.lessons, bootstrap);
    reader.teach(tools, bootstrap); lt::extend(tools, bootstrap, "mixed_long");
    ms::teach_initial(tools, bootstrap, "mixed_initial"); ms::extend(tools, bootstrap, "mixed_extension");
    tools.knowledge.rebuild(); tools.set_learning(false);
}

int main(int argc, char** argv) {
    try {
        gl::require(argc >= 4, "usage: model_task_learner smoke|resume|full learned|frozen_bodies|fixed_agency|readout_only DIR [--stop-after N] [key=value ...]");
        Profile prof = make_profile(argv[1], argv[2]);
        const fs::path dir = argv[3];
        size_t stop_after = SIZE_MAX;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--stop-after" && i + 1 < argc) stop_after = std::stoull(argv[++i]);
            else if (a == "--abort-during-eval" && i + 1 < argc) abort_during_eval = std::stoll(argv[++i]);
            else if (a == "--teacher" && i + 1 < argc) prof.teacher = argv[++i];
            else if (a == "--critic") prof.critic = true;
            else if (a == "--no-outcome") prof.no_outcome = true;
            else if (a == "--reencode" && i + 1 < argc) prof.reencode = (uint32_t)std::stoul(argv[++i]);
            else if (a == "--reencode-bodies") prof.reencode_bodies = true;
            else if (a == "--replay-missions" && i + 1 < argc) prof.replay_missions = (uint32_t)std::stoul(argv[++i]);
            else if (a == "--success-replay" && i + 1 < argc) prof.success_replay = (uint32_t)std::stoul(argv[++i]);
            else if (a == "--success-budget" && i + 1 < argc) prof.success_budget = (uint32_t)std::stoul(argv[++i]);
            else if (a == "--success-solo") prof.success_solo = true;
            else if (a == "--attempts" && i + 1 < argc) prof.attempts = (uint32_t)std::stoul(argv[++i]);
            else if (a == "--text-replay-kb" && i + 1 < argc) prof.text_replay_kb = (uint32_t)std::stoul(argv[++i]);
            else if (a == "--interleave" && i + 1 < argc) prof.interleave_file = argv[++i];
            else if (a == "--interleave-bytes" && i + 1 < argc) prof.interleave_bytes = std::stoull(argv[++i]);
            else if (a == "--text-eval" && i + 1 < argc) prof.text_eval_file = argv[++i];
            else if (a == "--success-prune") prof.success_prune = true;
            else if (a == "--success-stratify") prof.success_stratify = true;
            else if (a == "--withhold" && i + 1 < argc) {
                prof.withhold = argv[++i]; bool known = false; for (auto* f : family_names) known |= prof.withhold == f;
                gl::require(known, "--withhold needs a family name");
            }
            else if (a == "--rehearsal" && i + 1 < argc) prof.rehearsal = (uint32_t)std::stoul(argv[++i]);
            else if (a == "--pretrain" && i + 1 < argc) prof.pretrain_file = argv[++i];
            else if (a == "--pretrain-bytes" && i + 1 < argc) prof.pretrain_bytes = std::stoull(argv[++i]);
            else if (a.find('=') != std::string::npos) prof.extra.push_back(a);
            else throw std::runtime_error("unknown argument " + a);
        }
        if (prof.teacher != "rule") {
            gl::require(prof.teacher.rfind("policy:", 0) == 0, "teacher must be rule or policy:FILE:LABEL");
            const auto spec = prof.teacher.substr(7); const auto colon = spec.rfind(':');
            gl::require(colon != std::string::npos, "teacher policy needs FILE:LABEL");
            const auto loaded = laminar::subsystem_policy::load(spec.substr(0, colon), spec.substr(colon + 1));
            gl::require(loaded.actions == 7, "teacher policy must have seven actions");
            prof.teacher_digest = loaded.digest;
            auto policy = std::make_shared<laminar::subsystem_policy::Teacher>(loaded);
            teacher_fn = [policy](const gl::Observation& o, const std::vector<int>& m) { return (*policy)(o, m); };
            teacher_tag = "teacher_subsystem_policy";
        }
        if (!prof.interleave_file.empty()) {
            gl::require(prof.interleave_bytes > 0, "--interleave needs --interleave-bytes");
            interleave_text = read_all(prof.interleave_file); gl::require(interleave_text.size() > prof.interleave_bytes, "interleave file too short");
            prof.interleave_digest = fnv(interleave_text);
        }
        if (!prof.text_eval_file.empty()) { eval_text = read_all(prof.text_eval_file); gl::require(!eval_text.empty(), "empty text-eval file"); prof.text_eval_digest = fnv(eval_text); }
        std::vector<uint8_t> corpus;
        if (!prof.pretrain_file.empty()) {
            gl::require(prof.pretrain_bytes >= 2, "--pretrain needs --pretrain-bytes >= 2");
            std::ifstream f(prof.pretrain_file, std::ios::binary); gl::require(bool(f), "cannot open pretraining corpus");
            corpus.resize(prof.pretrain_bytes); f.read((char*)corpus.data(), (std::streamsize)corpus.size()); corpus.resize((size_t)f.gcount());
            gl::require(corpus.size() == prof.pretrain_bytes, "pretraining corpus shorter than declared");
            uint64_t h = 1469598103934665603ULL; for (uint8_t c : corpus) { h ^= c; h *= 1099511628211ULL; }
            prof.pretrain_digest = h;
        } else gl::require(prof.pretrain_bytes == 0, "--pretrain-bytes without --pretrain");
        const bool resume = fs::exists(dir / "checkpoint.state");
        fs::create_directories(dir);
        const std::string declaration = prof.declaration();
        if (fs::exists(dir / "config.json")) {
            std::ifstream f(dir / "config.json"); gl::require(std::string((std::istreambuf_iterator<char>(f)), {}) == declaration, "declared configuration changed");
            gl::require(resume, "declared run has no checkpoint: refusing to overwrite its logs");
        } else { std::ofstream f(dir / "config.json"); f << declaration; }

        const auto start = Clock::now();
        Logs logs(dir, resume);
        GoalControlledRelationalSession tools(1, 32, true); rt::Component reader;
        build_tools(tools, reader, logs.bootstrap);
        const double tools_seconds = std::chrono::duration<double>(Clock::now() - start).count();

        Organism model(model_params(prof));
        if (!resume && !corpus.empty()) {   // body pretraining: ordinary byte prediction, every learning rule and structure on
            const auto began = Clock::now(); model.set_mode(true, true);
            const size_t block = 4096; double loss = 0; uint64_t events = 0;
            for (size_t o = 0; o + 1 < corpus.size(); o += block) {   // consecutive blocks share one byte: one continuous stream
                const size_t n = std::min(block + 1, corpus.size() - o);
                model.run_block(corpus.data() + o, n); loss += model.stats().loss_sum; events += model.stats().events;
                model.end_block(); model.stats() = laminar::BlockStats{};
            }
            std::ofstream pre(dir / "pretrain.json");
            pre << "{\"bytes\":" << corpus.size() << ",\"events\":" << events << ",\"bits_per_byte\":" << loss / std::max<uint64_t>(1, events) / 0.6931471805599453
                << ",\"live_units\":" << model.live_units() << ",\"seconds\":" << std::chrono::duration<double>(Clock::now() - began).count() << "}\n";
            std::cout << "pretrained on " << corpus.size() << " bytes, " << model.live_units() << " live units\n" << std::flush;
            corpus.clear(); corpus.shrink_to_fit();
        }
        model.freeze_weights = prof.arm == "frozen_bodies";
        TaskModel tm(model, task_config(prof), 1);
        RunState rs;
        // An evaluation is due at mission 0, every eval_every missions and at the end. It always follows a checkpoint
        // (probes reload it) and is followed by one, so a crash inside an evaluation is repaired by redoing it on resume.
        auto eval_due = [&] {
            if (rs.last_eval == (int64_t)rs.next) return false;
            return rs.next == 0 || rs.next == prof.total || (prof.eval_every && rs.next % prof.eval_every == 0);
        };
        auto run_due_evaluation = [&] {
            if (!eval_due()) return;
            evaluate(prof, dir, rs.next == 0 ? std::string("initial") : "after" + std::to_string(rs.next), model, tm, tools, reader, logs, rs.next);
            model.set_mode(true, true);
            rs.last_eval = (int64_t)rs.next;
            save_checkpoint(dir, model, tm, rs, logs);   // the recorded log sizes now include this evaluation
        };
        if (resume) {
            // The current checkpoint is fsynced before it replaces the previous one; if it is nevertheless unreadable
            // (host storage failure), fall back to the previous checkpoint and redo the missions since.
            fs::path file = dir / "checkpoint.state";
            try { load_checkpoint(file, model, tm, rs); }
            catch (const std::exception& e) {
                gl::require(fs::exists(dir / "checkpoint.prev.state"), "current checkpoint unreadable and no previous checkpoint");
                std::cout << "current checkpoint unreadable (" << e.what() << "); resuming from the previous checkpoint\n" << std::flush;
                model.~Organism(); new (&model) Organism(model_params(prof)); model.freeze_weights = prof.arm == "frozen_bodies";
                tm.~TaskModel(); new (&tm) TaskModel(model, task_config(prof), 1); rs = RunState{};
                load_checkpoint(dir / "checkpoint.prev.state", model, tm, rs);
                fs::copy_file(dir / "checkpoint.prev.state", dir / "checkpoint.state", fs::copy_options::overwrite_existing);
            }
            logs.missions.close(); logs.decisions.close(); logs.evaluations.close();
            fs::resize_file(dir / "missions.jsonl", rs.log_sizes.at(0)); fs::resize_file(dir / "decisions.jsonl", rs.log_sizes.at(1)); fs::resize_file(dir / "evaluations.jsonl", rs.log_sizes.at(2));
            logs.missions.open(dir / "missions.jsonl", std::ios::app); logs.decisions.open(dir / "decisions.jsonl", std::ios::app); logs.evaluations.open(dir / "evaluations.jsonl", std::ios::app);
            for (auto* f : {&logs.missions, &logs.decisions, &logs.evaluations}) *f << std::boolalpha << std::setprecision(9);
            std::cout << "resumed at mission " << rs.next << " (tools rebuilt in " << tools_seconds << " s)\n" << std::flush;
            model.set_mode(true, true);
        } else {
            model.set_mode(true, true);
            save_checkpoint(dir, model, tm, rs, logs);
        }
        run_due_evaluation();
        model.set_mode(true, true);
        size_t done_this_process = 0;
        while (rs.next < prof.total) {
            const size_t index = rs.next;
            const size_t family = family_cycle[index % family_cycle.size()];
            size_t environment_index = 8 * rs.occurrences[family] + family; ++rs.occurrences[family];
            if (family >= 3 && (prof.name != "full" || index >= 100)) environment_index += 8000;
            MissionSpec spec; spec.index = index; spec.family = family; spec.guidance = guidance(prof, index);
            spec.phase = index < prof.full ? "guided" : index < prof.taper_end ? "taper" : "autonomous";
            if (!prof.withhold.empty() && prof.withhold == family_names[family]) spec.guidance = 0;   // no teacher in this family, ever
            spec.id = spec.phase + "_m" + std::to_string(index); spec.learning = true; spec.sample = spec.guidance < 1.0;
            spec.assistance_seed = 50000 + (uint32_t)index;
            auto reader_copy = reader.clone();
            auto env = mixed_environment(family, environment_index, false, prof.name != "full", reader_copy, logs.bootstrap, spec.id);
            tm.set_mission_tag((uint8_t)family);
            if (!interleave_text.empty()) {   // curriculum text between missions, ordinary learning, then structural maintenance
                std::vector<uint8_t> chunk(prof.interleave_bytes);
                for (uint64_t k = 0; k < prof.interleave_bytes; ++k) chunk[k] = interleave_text[(rs.text_pos + k) % interleave_text.size()];
                rs.text_pos = (rs.text_pos + prof.interleave_bytes) % interleave_text.size();
                tm.read_text(chunk.data(), chunk.size()); tm.end_mission();
                if (prof.text_replay_kb) tm.replay_extra((uint32_t)((prof.interleave_bytes + 1023) / 1024 / prof.text_replay_kb));   // defend task skills in proportion to the text
            }
            const auto began = Clock::now();
            TaskModel::GroupResult group;
            if (prof.attempts > 0 && spec.guidance == 0.0) {   // grouped outcome credit: frozen attempts at this very mission
                gl::require(prof.success_replay > 0, "--attempts needs --success-replay to rehearse what the attempts find");
                std::ostringstream discard;
                for (uint32_t k = 0; k < prof.attempts; ++k) {
                    auto attempt_reader = reader.clone();
                    auto attempt_env = mixed_environment(family, environment_index, false, prof.name != "full", attempt_reader, discard, spec.id + "_attempt" + std::to_string(k));
                    MissionSpec as = spec; as.learning = false; as.sample = true; as.id = spec.id + "_attempt" + std::to_string(k);
                    model.set_mode(false, false); model.replay = true;
                    tm.begin_attempt();
                    const auto ar = run_mission(tm, model, tools, *attempt_env, as, discard);
                    tm.end_attempt(ar.success); ++tm.metrics.attempts;
                    model.replay = false; model.set_mode(true, true);
                }
                group = tm.finish_attempts((uint8_t)family);
            }
            const auto r = run_mission(tm, model, tools, *env, spec, logs.decisions);
            rs.successes[family] += r.success;
            if (spec.phase == "autonomous") { rs.autonomous_success += r.success; gl::require(r.demonstrations == 0, "autonomous guidance"); }
            else rs.guided_success += r.success;
            ++rs.next; ++done_this_process;
            Telemetry::mission(logs.missions, spec, r, model, tm, rs);
            logs.timing << "{\"id\":" << gl::quote(spec.id) << ",\"seconds\":" << std::chrono::duration<double>(Clock::now() - began).count() << "}\n"; logs.timing.flush();
            std::cout << spec.id << " family=" << family_names[family] << " success=" << r.success << " decisions=" << r.decisions
                      << " model_exec=" << r.model_executed << " agree=" << r.agree << '/' << r.demonstrations << " units=" << model.live_units() << '\n' << std::flush;
            if (rs.next % prof.checkpoint_every == 0 || eval_due()) {
                save_checkpoint(dir, model, tm, rs, logs);
                run_due_evaluation();
                if (done_this_process >= stop_after) { std::cout << "stopping after checkpoint at mission " << rs.next << '\n'; return 0; }
            }
        }
        std::ofstream final_file(dir / "final.json");
        final_file << "{\"complete\":true,\"missions\":" << rs.next << ",\"guided_successes\":" << rs.guided_success << ",\"autonomous_successes\":" << rs.autonomous_success
                   << ",\"family_counts\":"; gl::numbers(final_file, rs.occurrences); final_file << ",\"family_successes\":"; gl::numbers(final_file, rs.successes);
        final_file << ",\"model_fingerprint\":\"" << std::hex << fingerprint(model, tm) << std::dec << "\"}\n";
        std::cout << "complete; fingerprint " << std::hex << fingerprint(model, tm) << std::dec << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
