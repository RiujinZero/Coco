#pragma once
// Experimental callable reasoning runtime. Input is a semantic goal graph and
// verified facts, not raw text. Skills, skill dispatch and call/return control
// have distinct learned local routes. No goal-specific execution recipes live here.
#include "agency.h"
#include <array>
#include <set>
#include <stdexcept>

namespace laminar::reasoning {

using Observation = std::array<float, 32>;
struct Value {
    int type = -1; // entity=0, bool=1, integer=2, absent=-1
    int id = -1;
    bool operator==(const Value&) const = default;
    bool operator<(const Value& o) const { return type == o.type ? id < o.id : type < o.type; }
    bool present() const { return type >= 0; }
};
struct Fact { int subject, relation, object; };
struct GoalNode {
    int kind; // learned semantic cue: follow=0, count=1, equal=2, literal=3
    int left = -1, right = -1, relation = -1;
    Value literal;
};
struct Goal { std::vector<GoalNode> nodes; int root = -1; std::vector<Fact> facts; };
struct Sample { Observation obs; std::vector<int> available; int action; };

struct SkillShape {
    int units=512,active=16,channels=3;
    void validate()const {
        if(units<1||units>20000||active<1||active>units||(channels!=1&&channels!=3))
            throw std::invalid_argument("invalid neural skill shape");
    }
};

// Uses Coco's actual sparse expansion, consequence routes, and local plasticity.
// fit() receives a verified/replayed action target; it is not terminal-reward RL.
struct LocalChoice {
    agency::AgencyKernel kernel;
    template<class Ar> void state_io(Ar& ar){ar(kernel);}
    explicit LocalChoice(int actions, uint64_t seed,SkillShape shape={}) {
        shape.validate();
        agency::KernelConfig c;
        c.obs_dim = 32; c.n_tendencies = actions; c.n_goals = 1; c.consequence_dim = shape.channels;
        c.n_units=shape.units;c.k_active=shape.active;
        c.baseline_cost = 0; c.focus_active = false; c.seed = seed;
        c.max_duration = 1; c.hysteresis = 0; c.commit_threshold = 0; c.commit_margin = 0;
        kernel.init(c); kernel.commitment.release_suppression = 0;
        kernel.set_goal(0, 1);
    }
    void fit(const Sample& s) {
        kernel.perceive(s.obs.data(), false);
        kernel.routes.predict_all(kernel.cur_idx, kernel.cur_val, kernel.pred_all);
        agency::EligibilityTraces immediate;
        for (int a : s.available) {
            immediate.entries.clear();
            immediate.record(a, kernel.cur_idx, kernel.cur_val);
            const float error[3] = {(a == s.action ? 1.f : 0.f) - kernel.pred_all[a * kernel.cfg.consequence_dim], 0, 0};
            agency::three_factor_update(kernel.routes, immediate, error, 0.1f);
        }
    }
    void value_update(const Observation& obs, int action, float target) {
        kernel.perceive(obs.data(), false);
        float predicted[3]; kernel.routes.predict(action, kernel.cur_idx, kernel.cur_val, predicted);
        agency::EligibilityTraces immediate;
        immediate.record(action, kernel.cur_idx, kernel.cur_val);
        const float error[3] = {target - predicted[0], 0, 0};
        agency::three_factor_update(kernel.routes, immediate, error, 0.03f);
    }
    float action_value(const Observation& obs, int action) {
        kernel.perceive(obs.data(), false);
        float predicted[3]; kernel.routes.predict(action, kernel.cur_idx, kernel.cur_val, predicted);
        return predicted[0];
    }
    void residual_update(const Observation& obs, int action, float residual) {
        kernel.perceive(obs.data(), false);
        agency::EligibilityTraces immediate;
        immediate.record(action, kernel.cur_idx, kernel.cur_val);
        const float error[3] = {residual, 0, 0};
        agency::three_factor_update(kernel.routes, immediate, error, 0.03f);
    }
    float best_value(const Observation& obs, const std::vector<int>& available) {
        kernel.perceive(obs.data(), false);
        kernel.routes.predict_all(kernel.cur_idx, kernel.cur_val, kernel.pred_all);
        float best = -1e9f;
        for (int a : available) best = std::max(best, kernel.pred_all[a * kernel.cfg.consequence_dim]);
        return best;
    }
    int choose(const Observation& obs, const std::vector<int>& available, bool pool = false, bool reset = true) {
        kernel.perceive(obs.data(), false);
        kernel.routes.predict_all(kernel.cur_idx, kernel.cur_val, kernel.pred_all);
        if (!pool) {
            int best = available.front();
            for (int a : available) if (kernel.pred_all[a * kernel.cfg.consequence_dim] > kernel.pred_all[best * kernel.cfg.consequence_dim]) best = a;
            return best;
        }
        std::fill(kernel.pool.available.begin(), kernel.pool.available.end(), 0);
        for (int a : available) kernel.pool.available[a] = 1;
        if (reset) {
            kernel.pool.reset(); kernel.commitment.tendency = -1;
            std::fill(kernel.pool.adaptation.begin(), kernel.pool.adaptation.end(), 0);
        }
        return kernel.deliberate(false, true);
    }
};

struct Scratch { std::set<Value> seen; int count = 0; };
struct MicroState {
    int task, relation; // same cues as the individual-skill probe
    Value subject, target, focus, value, output;
    int cursor = -1, predicate = -1, last = -1;
    bool exhausted = false, done = false;
    Scratch scratch;
    Observation observe() const {
        Observation o{}; o[task] = 1;
        const bool flags[] = {focus.present(), value.present(), focus == subject,
                              exhausted, value.present() && scratch.seen.count(value), scratch.count == 0};
        for (int i = 0; i < 6; ++i) o[3 + 2*i + (flags[i] ? 1 : 0)] = 1;
        o[15 + predicate + 1] = 1; o[18 + last + 1] = 1; o[31] = 1;
        return o;
    }
    std::vector<int> available() const {
        std::vector<int> a{0, 4, 9};
        if (focus.present()) a.insert(a.end(), {1, 5, 6});
        if (value.present()) a.insert(a.end(), {2, 3, 7});
        if (predicate >= 0) a.push_back(8);
        return a;
    }
    void step(int a, const std::vector<Fact>& facts) {
        if (a == 0) { focus = subject; value = {}; cursor = -1; exhausted = false; }
        if (a == 1) {
            value = {};
            // Equality is a two-value call with no fact-memory argument.
            if (task != 1 && focus.type == 0)
                for (int i = cursor + 1; i < (int)facts.size(); ++i) {
                    const auto& f = facts[i];
                    if (f.subject == focus.id && f.relation == relation) { cursor = i; value = {0, f.object}; break; }
                }
            exhausted = !value.present();
            if (exhausted) cursor = (int)facts.size();
        }
        if (a == 2) { focus = value; value = {}; cursor = -1; exhausted = false; }
        if (a == 3) scratch.seen.insert(value);
        if (a == 4) ++scratch.count;
        if (a == 5) predicate = focus == target;
        if (a == 6) output = focus;
        if (a == 7) output = value;
        if (a == 8) output = {1, predicate};
        if (a == 9) output = {2, scratch.count};
        last = a; done = a >= 6;
    }
};

struct RunOptions {
    bool pool = false;
    bool reset_competition = true;
    bool shared_scratch = false;
    bool forget_return = false;
    bool root_binding = false;
    bool record_choices = false;
    float epsilon = 0;
    int max_control_steps = 160, max_skill_steps = 256;
};
struct RunResult {
    Value output;
    int control_steps = 0, skill_steps = 0, calls = 0;
    bool timeout = false, invalid = false;
    std::vector<Sample> choices;
};

struct Runtime {
    LocalChoice skills, dispatch, control;
    Rng rng;
    explicit Runtime(uint64_t seed) : skills(10, seed), dispatch(7, seed + 100), control(7, seed + 200), rng(seed + 909) {}

    void credit_control(const RunResult& episode, float reward) {
        for (int i = (int)episode.choices.size() - 1; i >= 0; --i) {
            const auto& s = episode.choices[i];
            float target = reward;
            if (i + 1 < (int)episode.choices.size()) {
                const auto& next = episode.choices[i + 1];
                target = -0.005f + 0.99f * control.best_value(next.obs, next.available);
            }
            control.value_update(s.obs, s.action, target);
        }
    }

    void credit_monte_carlo(const RunResult& episode, float reward) {
        float target = reward;
        for (int i = (int)episode.choices.size() - 1; i >= 0; --i) {
            const auto& s = episode.choices[i];
            control.value_update(s.obs, s.action, target);
            target = -0.005f + 0.99f * target;
        }
    }

    void credit_shared_error(const RunResult& episode, float reward) {
        if (episode.choices.empty()) return;
        const auto& last = episode.choices.back();
        float last_prediction = control.action_value(last.obs, last.action);
        float target = reward;
        for (int i = (int)episode.choices.size() - 1; i >= 0; --i) {
            const auto& s = episode.choices[i];
            control.residual_update(s.obs, s.action, target - last_prediction);
            target = -0.005f + 0.99f * target;
        }
    }

    void credit_broadcast(const RunResult& episode, float reward, bool long_trace = false) {
        auto& k = control.kernel;
        k.traces.entries.clear(); k.traces.now = 0;
        k.traces.lifetime = long_trace ? 160 : 8;
        k.traces.decay = long_trace ? 0.99f : 0.85f;
        k.traces.max_entries = 160;
        for (const auto& s : episode.choices) {
            k.perceive(s.obs.data(), false);
            k.traces.record(s.action, k.cur_idx, k.cur_val);
            k.act_t = s.action; k.act_idx = k.cur_idx; k.act_val = k.cur_val; k.have_act = true;
        }
        // Exactly the kernel's last-action residual broadcast, with no successor
        // value target or separate error for each earlier action.
        const float outcome[3] = {reward, 0, 1};
        k.consequence(outcome, true);
    }

    RunResult run(const Goal& goal, const RunOptions& opt = {}) {
        RunResult result;
        const int size = (int)goal.nodes.size();
        if (goal.root < 0 || goal.root >= size) { result.invalid = true; return result; }
        for (LocalChoice* choice : {&skills, &dispatch, &control}) {
            auto& k = choice->kernel;
            k.pool.reset(); k.commitment.tendency = -1; k.commitment.age = 0;
            std::fill(k.pool.adaptation.begin(), k.pool.adaptation.end(), 0);
            k.traces.entries.clear(); k.set_goal(0, 1);
        }
        std::vector<Value> values(size);
        Value root_literal;
        for (int i = 0; i < size; ++i) if (goal.nodes[i].kind == 3) {
            values[i] = goal.nodes[i].literal;
            if (!root_literal.present()) root_literal = values[i];
        }
        int current = goal.root;
        std::vector<int> stack;
        Scratch shared;
        for (; result.control_steps < opt.max_control_steps; ++result.control_steps) {
            const auto& n = goal.nodes[current];
            if (n.left >= size || n.right >= size) { result.invalid = true; return result; }
            bool flags[] = {n.left >= 0, n.left >= 0 && values[n.left].present(),
                            n.right >= 0, n.right >= 0 && values[n.right].present(),
                            values[current].present(), !stack.empty()};
            Observation o{};
            for (int i = 0; i < 6; ++i) o[2*i + (flags[i] ? 1 : 0)] = 1;
            o[31] = 1;
            std::vector<int> available{3, 6};
            if (n.left >= 0) available.push_back(0);
            if (n.right >= 0) available.push_back(1);
            if (!stack.empty()) available.push_back(2);
            int a = control.choose(o, available, opt.pool, opt.reset_competition);
            if (opt.epsilon > 0 && rng.uniform() < opt.epsilon) a = available[rng.below((uint32_t)available.size())];
            if (opt.record_choices) result.choices.push_back(Sample{o, available, a});
            if (a == 0 || a == 1) {
                stack.push_back(current); current = a == 0 ? n.left : n.right;
            } else if (a == 2) {
                int child = current; current = stack.back(); stack.pop_back();
                if (opt.forget_return && goal.nodes[child].kind != 3) values[child] = {};
            } else if (a == 3) {
                Observation cue{}; cue[n.kind] = 1; cue[31] = 1;
                int op = dispatch.choose(cue, {3, 4, 5}, opt.pool, opt.reset_competition);
                if (n.left < 0 || !values[n.left].present() ||
                    (op == 5 && (n.right < 0 || !values[n.right].present()))) {
                    result.invalid = true; return result;
                }
                Value arg = opt.root_binding ? root_literal : values[n.left];
                if ((op == 3 || op == 4) && arg.type != 0) { result.invalid = true; return result; }
                if (op < 3 || op > 5) { result.invalid = true; return result; }
                MicroState vm;
                vm.task = op == 3 ? 0 : op == 4 ? 2 : 1;
                vm.relation = n.relation; vm.subject = arg;
                vm.target = op == 5 ? values[n.right] : Value{};
                if (opt.shared_scratch) vm.scratch = shared;
                ++result.calls;
                for (int i = 0; i < opt.max_skill_steps && !vm.done; ++i) {
                    int instruction = skills.choose(vm.observe(), vm.available(), opt.pool, opt.reset_competition);
                    if (instruction < 0) { result.invalid = true; return result; }
                    vm.step(instruction, goal.facts); ++result.skill_steps;
                }
                if (!vm.done) { result.timeout = true; return result; }
                const int expected_type = op == 3 ? 0 : op == 4 ? 2 : 1;
                if (vm.output.type != expected_type) { result.invalid = true; return result; }
                if (opt.shared_scratch) shared = vm.scratch;
                values[current] = vm.output;
            } else if (a == 6) {
                result.output = values[goal.root]; ++result.control_steps; return result;
            } else { result.invalid = true; return result; }
        }
        result.timeout = true;
        return result;
    }
};
} // namespace laminar::reasoning
