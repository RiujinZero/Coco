#pragma once
// The model as the task learner. One persistent model receives the task as a
// stream of framed events and decides every operation from its own workspace:
//
//   observe : requests, public observations, goal arrivals, retrieved evidence,
//             tool results and executed actions enter the bodies as bytes, each
//             frame tagged with its channel. Observations, teacher demonstrations
//             and generated proposals stay distinct: a demonstration is its own
//             channel, and a generated proposal is processed without learning.
//   decide  : a decision frame is processed by the bodies; a small linear readout
//             (actions x 97 weights) maps body activity -- the decision event's
//             workspace, the persistent workspace, and the workspace pooled over
//             this step's observation and decision frames -- to a distribution
//             over the legal operations. Nothing else chooses.
//   act     : the executed operation (the model's own, or a teacher demonstration
//             during guided training) is fed back as an action/demonstration frame.
//   feedback: tool outcomes and verified mission results reach the agency kernel
//             as task-level consequences; demonstrations and delayed mission
//             outcomes train the readout AND the bodies that produced the decision.
//
// Body learning combines four signals: (1) byte prediction of every observed
// frame, through the model's own local rules (L6 error, modulated hidden layers,
// structure); (2) demonstrated decisions, as an exact local output error into the
// L6 layer of the bodies active in the decision pass plus a modulated three-factor
// update of their hidden layers; (3) verified mission outcomes, as the same two
// updates weighted by the discounted advantage of each decision the model itself
// executed; (4) credit to participating bodies and routes: resistance, lock
// utility and routed-emission utility of those bodies move with that advantage.
// No gradient is propagated between body layers (no backpropagation into bodies).
//
// Division of labour with the agency kernel: the kernel never selects an
// operation and the readout never moves attention. The kernel observes public
// goal state, the last tool outcome and the readout's own confidence, commits to
// a focus over the bodies' input (which activity stays engaged, and for how
// long), and is credited by the same verified task outcomes as the readout.
#include "organism.h"
#include <array>
#include <cmath>
#include <random>
#include <sstream>
#include <cstring>
#include <string_view>

namespace laminar {

enum class TaskChannel : uint8_t {
    request = 'R', observation = 'O', arrival = 'G', evidence = 'E', tool_result = 'T',
    action = 'A', demonstration = 'D', proposal = 'P', outcome = 'V', decide = '?'
};

struct TaskModelConfig {
    int actions = 7;
    float readout_lr = 0.5f;     // demonstration cross-entropy on the readout (normalised: step / (1 + |x|^2))
    float policy_lr = 0.2f;      // outcome policy gradient on the readout (normalised likewise)
    float l6_gain = 1.f;         // task error into the L6 layer of participating bodies
    float demo_mod = 16.f;       // modulator scale: demonstration log-likelihood gain (bits) over its running mean
    float outcome_mod = 48.f;    // modulator scale: discounted outcome advantage
    float body_credit = 1.f;     // resistance / lock utility / route utility credit scale
    float route_credit = 0.02f;  // routed-emission utility change per unit advantage
    float gamma = 0.97f;         // per-decision discount of delayed outcome credit
    float baseline_rate = 0.05f; // running mission-reward baseline
    float explore = 0.1f;        // autonomous training: sample from (1-explore)*pi + explore*uniform
    bool body_learning = true;   // false: task credit reaches only the readout (frozen-body control also sets freeze_weights)
    // Outcome credit variants (off by default: the first full-scale runs used the global baseline alone).
    // critic: advantage = discounted return - V(x), with V a linear value readout on the same body features,
    //   trained on every decision of every learning mission (teacher- or model-executed). Replaces the single
    //   global baseline, whose drift can inflate every action of a rare success (see MODEL_TASK_LEARNER.md).
    // rehearsal: after each learning mission, this many stored demonstrations (features, menu, label) are
    //   replayed through the readout's cross-entropy, so demonstrated behaviour stays anchored after the teacher
    //   leaves. Readout only; stored features are those the bodies produced when the example was demonstrated.
    // reencode: after each learning mission, this many stored demonstrated situations (the last reencode_context
    //   bytes up to and including the decision frame, which of them were pooled, the menu and the teacher's label)
    //   are replayed through the CURRENT bodies with learning, structure and sequence-memory capture off. The
    //   readout is fitted to the features the bodies produce now, so it tracks body drift. With reencode_bodies the
    //   same exact L6 error also reaches the bodies active in the replay (no modulation, no outcome credit).
    //   Only verified teacher demonstrations with more than one legal operation are stored.
    bool critic = false; float critic_lr = 0.1f;
    uint32_t rehearsal = 0, rehearsal_capacity = 16384;
    uint32_t reencode = 0, reencode_context = 128; bool reencode_bodies = false;
    // replay_missions: every learning mission with at least one multi-option demonstration is recorded as a tape
    //   (every byte with its pooled flag, every task-state update and tool outcome, every decision with its menu and,
    //   when demonstrated, the teacher's label). A uniform reservoir of tape_capacity tapes is kept over the whole
    //   run. After each learning mission, this many tapes are replayed from start to end through the CURRENT model
    //   (bodies, workspace, agency kernel) with learning, structure and sequence-memory capture off, so every stored
    //   decision is re-encoded in its full mission context. The readout (and with reencode_bodies the bodies) is
    //   fitted on each demonstrated multi-option decision, exactly as in live guided training but without modulation.
    uint32_t replay_missions = 0, tape_capacity = 256;
    // success_replay: self-imitation from verified outcomes. A learning mission that the environment verifies as a
    //   success is kept as a tape whose labels include the model's OWN executed choices (teacher labels where the
    //   teacher acted). After each learning mission, this many success tapes are replayed exactly like teacher tapes.
    //   The outcome decides what is rehearsed; failures carry no credit. Uniform reservoir of success_capacity tapes.
    //   success_budget > 0 limits how often any one success tape is replayed (a success counts as a few rehearsals,
    //   not hundreds while the store is still small); tapes that used their budget stay stored but are not drawn.
    uint32_t success_replay = 0, success_capacity = 256, success_budget = 0;
    //   success_solo: store only successes in which the teacher took no action at all.
    //   success_prune: drop the model's labels on loop iterations -- a self-labelled decision whose state (the pooled
    //     observation bytes since the previous decision, plus the menu) recurs later in the same mission.
    //   success_stratify: draw success tapes with weight 1 / (1 + teacher labels of the tape's mission tag / 100), so
    //     tasks the teacher never covered dominate (tags are set by the caller per mission, e.g. the task family).
    bool success_solo = false, success_prune = false, success_stratify = false;
    template <class Ar> void state_io(Ar& ar) { ar(actions); ar(readout_lr); ar(policy_lr); ar(l6_gain); ar(demo_mod); ar(outcome_mod); ar(body_credit); ar(route_credit); ar(gamma); ar(baseline_rate); ar(explore); ar(body_learning); ar(critic); ar(critic_lr); ar(rehearsal); ar(rehearsal_capacity); ar(reencode); ar(reencode_context); ar(reencode_bodies); ar(replay_missions); ar(tape_capacity); ar(success_replay); ar(success_capacity); ar(success_budget); ar(success_solo); ar(success_prune); ar(success_stratify); }
};

class TaskModel {
public:
    static constexpr int D = 3 * Dims::F + 1;   // decision workspace, persistent workspace, pooled step workspace, bias
    static constexpr size_t max_pooled_events = 96;
    static constexpr uint8_t FRAME = 0x1E;

    struct Choice {
        int op = -1;
        std::array<float, 8> probs{};           // behaviour distribution actually sampled from (autonomous training)
        std::array<float, 8> policy{};          // readout distribution
        float confidence = 0.f, entropy = 0.f;
        bool sampled = false;
    };
    struct Metrics {
        uint64_t decisions = 0, model_executed = 0, demonstrations = 0, demo_agree = 0, bytes = 0, learned_bytes = 0;
        uint64_t l6_task_updates = 0, modulated_task_updates = 0, credited_bodies = 0, credited_routes = 0, outcomes = 0, rehearsed = 0, successes_stored = 0, labels_pruned = 0, corrections_stored = 0, attempts = 0;
        double outcome_sum = 0;
        template <class Ar> void state_io(Ar& ar) { ar(decisions); ar(model_executed); ar(demonstrations); ar(demo_agree); ar(bytes); ar(learned_bytes); ar(l6_task_updates); ar(modulated_task_updates); ar(credited_bodies); ar(credited_routes); ar(outcomes); ar(outcome_sum); ar(rehearsed); ar(successes_stored); ar(labels_pruned); ar(corrections_stored); ar(attempts); }
    };

    TaskModel(Organism& model, TaskModelConfig config, uint64_t seed) : m(model), c(config), rng_(seed) {
        if (c.actions < 1 || c.actions > 8) throw std::invalid_argument("task readout supports 1..8 actions");
        if (model.params().vocab != 256) throw std::invalid_argument("task model needs the byte vocabulary");
        W.assign((size_t)c.actions * D, 0.f);
        V.assign(D, 0.f);
        scale.assign(3 * Dims::F, 64.f);
        last_ = model.last_symbol();   // continue the model's own stream (after any pretraining)
    }

    bool learning() const { return m.params().train; }
    // Evaluation/ablation switch: the readout sees no body output (zero workspace features).
    bool ablate_body_output = false;

    // ---- observing -------------------------------------------------------------------------------------------
    void observe(TaskChannel ch, std::string_view payload, bool learn = true) {
        const bool was = m.params().train, structural = m.params().structural;
        const bool suppress = was && !learn;
        if (suppress) { m.set_mode(false, false); suppressing_ = true; }   // generated content: processed, never learned as evidence
        const bool pooled = ch == TaskChannel::observation;   // the current public state joins the step pool
        feed(FRAME, was && learn, pooled); feed((uint8_t)ch, was && learn, pooled);
        for (char x : payload) { uint8_t b = (uint8_t)x; if (b == FRAME || b == '\n') b = ' '; feed(b, was && learn, pooled); }
        feed('\n', was && learn, pooled);
        if (suppress) { m.set_mode(was, structural); suppressing_ = false; }
    }
    // Public task state for the agency kernel's task channel (eight slots).
    void set_task_state(const std::array<float, 8>& s) {
        for (int i = 0; i < 8; ++i) m.task_obs[i] = s[i];
        if (recording()) { Mark k; k.at = (uint32_t)tape_.bytes.size(); k.kind = Mark::k_state; k.state = s; tape_.marks.push_back(std::move(k)); }
    }
    // A verified tool or goal outcome, delivered to the kernel with the next event.
    void tool_outcome(float value) {
        m.post_task_outcome(value);
        if (recording()) { Mark k; k.at = (uint32_t)tape_.bytes.size(); k.kind = Mark::k_outcome; k.value = value; tape_.marks.push_back(std::move(k)); }
    }

    // ---- deciding --------------------------------------------------------------------------------------------
    // Processes the decision frame and returns the readout's choice among the legal operations. The decision
    // event's input is the frame's last byte; its known outcome is the frame terminator, so the workspace it
    // produces is the model's state after reading the whole frame.
    Choice decide(const std::vector<int>& menu, bool sample) {
        if (menu.empty()) throw std::invalid_argument("empty decision menu");
        for (int a : menu) if (a < 0 || a >= c.actions) throw std::invalid_argument("menu outside readout actions");
        const bool train = learning();
        if (recording()) { Mark k; k.at = (uint32_t)tape_.bytes.size(); k.kind = Mark::k_decision; k.menu = menu; tape_.marks.push_back(std::move(k)); }
        feed(FRAME, train, true); feed((uint8_t)TaskChannel::decide, train, true);
        for (int a : menu) feed((uint8_t)('0' + a), train, true);
        feed('\n', train, true);   // the decision event: bodies run on the last menu byte, the workspace forms the decision
        Record r;
        features(r.x);
        r.menu = menu;
        if (c.reencode && train) {   // the situation as bytes: the last reencode_context bytes, ending with this decision frame
            const size_t n = (size_t)std::min<uint64_t>(ring_n_, std::min<uint64_t>(c.reencode_context, ring_size));
            r.ctx.resize(n); r.ctx_pool.resize(n);
            for (size_t i = 0; i < n; ++i) { const size_t j = (size_t)((ring_n_ - n + i) % ring_size); r.ctx[i] = ring_[j]; r.ctx_pool[i] = ring_pool_[j]; }
        }
        r.events = std::move(pool_events_); pool_events_.clear();
        std::fill(pool_sum_, pool_sum_ + Dims::F, 0.f); pool_n_ = 0;
        Choice out;
        policy(r.x, menu, out.policy);
        out.probs = out.policy;
        int best = menu.front(); for (int a : menu) if (out.policy[a] > out.policy[best]) best = a;
        out.op = best;
        if (sample) {
            const float u = 1.f / (float)menu.size();
            for (int a : menu) out.probs[a] = (1.f - c.explore) * out.policy[a] + c.explore * u;
            float z = std::uniform_real_distribution<float>(0.f, 1.f)(rng_), acc = 0.f; out.op = menu.back();
            for (int a : menu) { acc += out.probs[a]; if (z < acc) { out.op = a; break; } }
            out.sampled = true;
        }
        out.confidence = out.policy[best];
        for (int a : menu) if (out.policy[a] > 0.f) out.entropy -= out.policy[a] * std::log(out.policy[a]);
        r.probs = out.probs; r.chosen = out.op;
        pending_ = std::move(r); has_pending_ = true;
        ++metrics.decisions;
        return out;
    }

    // ---- acting ----------------------------------------------------------------------------------------------
    // The operation that was actually executed. A demonstration (teacher-executed during guided training) trains
    // the readout and the participating bodies immediately; the model's own choice is kept for outcome credit.
    void act(int executed, bool model_executed, bool demonstrated) {
        if (!has_pending_) throw std::logic_error("act without a decision");
        Record r = std::move(pending_); has_pending_ = false;
        r.executed = executed; r.model = model_executed; r.t = mission_x_.size();
        if (learning() && c.critic) mission_x_.push_back(r.x);
        if (learning() && demonstrated && c.rehearsal) {
            Rehearsal e{r.x, {}, executed}; e.menu = r.menu;
            if (rehearsal_.size() < c.rehearsal_capacity) rehearsal_.push_back(std::move(e)); else rehearsal_[rehearsal_next_++ % c.rehearsal_capacity] = std::move(e);
        }
        if (learning() && demonstrated && c.reencode && r.menu.size() > 1) {
            Situation s{r.ctx, r.ctx_pool, r.menu, executed};
            if (situations_.size() < c.rehearsal_capacity) situations_.push_back(std::move(s)); else situations_[situation_next_++ % c.rehearsal_capacity] = std::move(s);
        }
        if (recording() && (demonstrated || (model_executed && c.success_replay))) {   // label the most recent decision event
            for (auto it = tape_.marks.rbegin(); it != tape_.marks.rend(); ++it) if (it->kind == Mark::k_decision) {
                it->label = executed; it->self = !demonstrated; tape_any_teacher_ |= demonstrated;
                if (it->menu.size() > 1) { if (demonstrated) tape_has_demo_ = true; else tape_has_self_ = true; }
                break;
            }
        }
        if (demonstrated) {
            ++metrics.demonstrations; metrics.demo_agree += r.chosen == executed;
            if (learning()) demonstrate(r, executed);
        }
        if (model_executed) ++metrics.model_executed;
        if (learning() && model_executed) episode_.push_back(std::move(r));
        const char tag[2] = {(char)('0' + executed), 0};
        observe(demonstrated ? TaskChannel::demonstration : TaskChannel::action, tag, true);
    }

    // ---- feedback --------------------------------------------------------------------------------------------
    // Delayed verified mission outcome (+1 success, -1 failure): discounted advantage credit to every decision
    // the model executed in this mission, through the readout and the bodies that produced each decision.
    void mission_outcome(float reward) {
        ++metrics.outcomes; metrics.outcome_sum += reward;
        if (!learning()) { episode_.clear(); mission_x_.clear(); return; }
        // critic values before this mission's update; then fit the critic on every decision's discounted return
        const size_t N = mission_x_.size();
        std::vector<float> value(N, 0.f);
        if (c.critic) {
            for (size_t t = 0; t < N; ++t) value[t] = dot(V, mission_x_[t]);
            for (size_t t = 0; t < N; ++t) {
                const float G = reward * std::pow(c.gamma, (float)(N - 1 - t)), step = c.critic_lr / norm2(mission_x_[t]);
                const float e = G - dot(V, mission_x_[t]);
                for (int k = 0; k < D; ++k) V[k] += step * e * mission_x_[t][k];
            }
        }
        rehearse();
        reencode();
        store_tape(reward);
        replay_tapes();
        mission_x_.clear();
        if (episode_.empty()) return;   // teacher-executed missions carry no policy credit
        if (c.policy_lr == 0.f && c.outcome_mod == 0.f && c.body_credit == 0.f) { episode_.clear(); return; }   // outcome credit disabled (--no-outcome)
        const float advantage = reward - baseline_;
        baseline_ += c.baseline_rate * (reward - baseline_);
        const size_t T = episode_.size();
        for (size_t t = 0; t < T; ++t) {
            const Record& r = episode_[t];
            const float w = c.critic ? reward * std::pow(c.gamma, (float)(N - 1 - r.t)) - value[r.t]
                                     : advantage * std::pow(c.gamma, (float)(T - 1 - t));
            if (std::fabs(w) < 1e-4f) continue;
            // loss = -w log pi(executed): the gradient pushes toward the executed op when w > 0, away when w < 0
            std::array<float, 8> err{};
            for (int a : r.menu) err[a] = w * (r.probs[a] - (a == r.executed ? 1.f : 0.f));
            const float step = c.policy_lr / norm2(r.x);
            for (int a : r.menu) { float* row = &W[(size_t)a * D]; for (int k = 0; k < D; ++k) row[k] -= step * err[a] * r.x[k]; }
            if (c.body_learning) body_update(r, err, (int32_t)std::lround(std::clamp(c.outcome_mod * w, -127.f, 127.f)), w);
        }
        episode_.clear();
    }

    // Declared structural maintenance at mission boundaries (the byte trainer does this per block).
    void end_mission() { m.end_block(); m.st = BlockStats{}; }

    // ---- state -----------------------------------------------------------------------------------------------
    template <class Ar> void state_io(Ar& ar) {
        if (has_pending_ || !episode_.empty() || !mission_x_.empty() || !attempts_.empty() || attempt_) throw std::logic_error("task model checkpoint inside a mission");
        ar.section("task_model");
        TaskModelConfig saved = c; ar(saved);
        if (Ar::loading && saved.actions != c.actions) throw std::runtime_error("task model configuration differs from checkpoint");
        if constexpr (Ar::loading) c = saved;   // the checkpoint defines the configuration (drivers also verify their declaration)
        ar(W); ar(scale); ar(baseline_); ar(demo_ema_); ar(demo_ema_init_); ar(last_); ar(metrics);
        if (c.critic || c.rehearsal) { ar(V); ar(rehearsal_); ar(rehearsal_next_); }
        if (c.reencode) { ar(situations_); ar(situation_next_); ar(ring_); ar(ring_pool_); ar(ring_n_); }
        if (c.replay_missions || c.success_replay) { ar(tapes_); ar(tapes_seen_); ar(tape_); ar(tape_has_demo_); ar(tape_has_self_); ar(successes_); ar(successes_seen_); ar(tape_any_teacher_); ar(mission_tag_); ar(teacher_labels_by_tag_); }
        { std::string s; if constexpr (!Ar::loading) { std::ostringstream o; o << rng_; s = o.str(); } ar(s); if constexpr (Ar::loading) { std::istringstream i(s); i >> rng_; } }
    }
    const std::vector<float>& readout() const { return W; }
    const std::vector<float>& feature_scale() const { return scale; }
    // Features of the most recent decision (inspection only).
    const std::array<float, D>& last_features() const { return pending_.x; }
    float baseline() const { return baseline_; }
    size_t situations() const { return situations_.size(); }
    size_t success_tapes() const { return successes_.size(); }
    // ---- grouped attempts (outcome credit without a teacher) ---------------------------------------------------
    // The caller runs the same mission several times with the model frozen (no learning, no structure, no sequence
    // capture) and sampling on, bracketing each with begin_attempt / end_attempt(verified success). finish_attempts
    // then turns the group into rehearsal material:
    //   * each successful attempt becomes a success tape (loop-pruned when success_prune), labelled with the model's
    //     own choices;
    //   * each failed attempt is compared with a successful one: at the first decision where both were in the same
    //     state but chose differently, the failed attempt's tape is labelled with the successful choice (a correction
    //     in the failure's own context) and all its other labels are cleared.
    // Nothing is learned directly: all credit flows through tape replay, like teacher demonstrations. A group with
    // no success or no failure yields at most one success tape (nothing to contrast).
    struct GroupResult { size_t successes = 0, failures = 0, stored = 0, corrections = 0; };
    void begin_attempt() { tape_.clear(); tape_has_demo_ = tape_has_self_ = tape_any_teacher_ = false; attempt_ = true; }
    void end_attempt(bool success) { attempts_.push_back({std::move(tape_), success}); tape_.clear(); tape_has_demo_ = tape_has_self_ = tape_any_teacher_ = false; attempt_ = false; }
    GroupResult finish_attempts(uint8_t tag) {
        GroupResult g; const Tape* first_success = nullptr;
        for (auto& [t, ok] : attempts_) { if (ok) { ++g.successes; if (!first_success) first_success = &t; } else ++g.failures; }
        if (first_success) {
            const auto good = decision_states(*first_success);
            for (auto& [t, ok] : attempts_) {
                if (ok) {
                    Tape copy = t; copy.tag = tag; if (c.success_prune) metrics.labels_pruned += prune_loops(copy);
                    if (c.success_budget) {   // informative weighting: budget x (1 - group success rate)
                        const double p_hat = double(g.successes) / double(attempts_.size());
                        copy.replays = c.success_budget - (uint32_t)std::lround(c.success_budget * (1.0 - p_hat));
                        if (copy.replays >= c.success_budget) continue;   // every attempt succeeded: nothing new to rehearse
                    }
                    reservoir(successes_, successes_seen_, c.success_capacity, std::move(copy)); ++g.stored; ++metrics.successes_stored;
                    if (!g.failures) break;   // all succeeded: one tape is enough
                    continue;
                }
                const auto bad = decision_states(t);
                size_t div = SIZE_MAX; int want = -1;
                for (size_t i = 0; i < bad.size() && i < good.size(); ++i) {
                    const Mark& mb = t.marks[bad[i].first]; const Mark& mg = first_success->marks[good[i].first];
                    if (bad[i].second != good[i].second) break;   // states no longer comparable
                    const int got = mb.label, ref = mg.label;
                    if (got >= 0 && ref >= 0 && got != ref) { div = bad[i].first; want = ref; break; }
                }
                if (div == SIZE_MAX) continue;
                Tape corr = std::move(t); corr.tag = tag;
                for (Mark& k : corr.marks) if (k.kind == Mark::k_decision) { k.label = -1; k.self = false; }
                corr.marks[div].label = want; corr.marks[div].self = true;
                reservoir(successes_, successes_seen_, c.success_capacity, std::move(corr)); ++g.corrections; ++metrics.corrections_stored;
            }
        }
        attempts_.clear();
        return g;
    }
    // The caller's category for the current mission (e.g. its task family); used by success_stratify.
    void set_mission_tag(uint8_t tag) { mission_tag_ = tag; }
    // Plain text between missions (curriculum mixing, text evaluation): bytes stream through the model with its
    // current learning mode, outside any task frame; never recorded into tapes, situations or the step pool.
    void read_text(const uint8_t* data, size_t n) {
        reading_text_ = true;
        for (size_t i = 0; i < n; ++i) feed(data[i], m.params().train, false);
        reading_text_ = false;
    }
    Organism& model() { return m; }
    // Answer channel: the model writes bytes from its own next-byte distribution (greedy, or sampled at a
    // temperature) until a newline or max_len, with learning, structure and placeholder capture off. The written
    // bytes become the model's history, as its own output; nothing is learned from them here.
    std::string generate(size_t max_len, bool sample, float temperature = 1.f) {
        const bool was = m.params().train, structural = m.params().structural;
        m.set_mode(false, false); m.free_run = true;
        std::string out;
        for (size_t i = 0; i < max_len; ++i) {
            m.process_event(last_, 0);
            const auto& p = m.distribution(); int v = 0;
            if (!sample) { for (int k = 1; k < 256; ++k) if (p[k] > p[v]) v = k; }
            else {
                double z = 0; std::array<double, 256> q{};
                for (int k = 0; k < 256; ++k) { q[k] = std::pow(std::max((double)p[k], 1e-12), 1.0 / temperature); z += q[k]; }
                double u = std::uniform_real_distribution<double>(0.0, z)(rng_), acc = 0; v = 255;
                for (int k = 0; k < 256; ++k) { acc += q[k]; if (u < acc) { v = k; break; } }
            }
            m.commit_outcome((uint8_t)v); last_ = (uint8_t)v; ++metrics.bytes; out += (char)v;
            if (v == '\n') break;
        }
        m.free_run = false; m.set_mode(was, structural);
        return out;
    }
    bool& agency_focus_on() { return m.agent_on; }   // inspection only
    // Inspection: replay every stored situation whose bytes contain `needle` and whose menu equals `menu`; return
    // how often each operation is the readout's argmax, and how often the teacher chose each (no updates).
    std::pair<std::array<int, 8>, std::array<int, 8>> reencode_breakdown(const std::string& needle, const std::vector<int>& menu, size_t limit) {
        std::array<int, 8> chosen{}, labels{};
        const bool was = m.params().train, structural = m.params().structural;
        float saved_sum[Dims::F]; std::memcpy(saved_sum, pool_sum_, sizeof saved_sum);
        const uint32_t saved_n = pool_n_; std::vector<PoolEvent> saved_events = std::move(pool_events_); pool_events_.clear();
        m.set_mode(false, false); m.replay = true; size_t n = 0;
        for (const Situation& s : situations_) {
            if (n >= limit || s.menu != menu) continue;
            if (std::string(s.bytes.begin(), s.bytes.end()).find(needle) == std::string::npos) continue;
            std::fill(pool_sum_, pool_sum_ + Dims::F, 0.f); pool_n_ = 0; pool_events_.clear();
            for (size_t j = 0; j < s.bytes.size(); ++j) feed(s.bytes[j], false, s.pooled[j] != 0);
            std::array<float, D> x{}; features(x); std::array<float, 8> p{}; policy(x, s.menu, p);
            int best = s.menu.front(); for (int a : s.menu) if (p[a] > p[best]) best = a;
            ++chosen[best]; ++labels[s.label]; ++n;
        }
        m.replay = false; m.set_mode(was, structural);
        std::memcpy(pool_sum_, saved_sum, sizeof saved_sum); pool_n_ = saved_n; pool_events_ = std::move(saved_events);
        return {chosen, labels};
    }
    // Inspection: re-encode the first n stored situations (or every stride-th) through the current bodies with no
    // update of any kind, and return the fraction where the readout's argmax equals the teacher's label.
    double reencode_agreement(size_t n) {
        if (situations_.empty()) return 0.0;
        const bool was = m.params().train, structural = m.params().structural;
        float saved_sum[Dims::F]; std::memcpy(saved_sum, pool_sum_, sizeof saved_sum);
        const uint32_t saved_n = pool_n_; std::vector<PoolEvent> saved_events = std::move(pool_events_); pool_events_.clear();
        m.set_mode(false, false); m.replay = true;
        const size_t stride = std::max<size_t>(1, situations_.size() / std::max<size_t>(1, n)); size_t agree = 0, total = 0;
        for (size_t i = 0; i < situations_.size() && total < n; i += stride, ++total) {
            const Situation& s = situations_[i];
            std::fill(pool_sum_, pool_sum_ + Dims::F, 0.f); pool_n_ = 0; pool_events_.clear();
            for (size_t j = 0; j < s.bytes.size(); ++j) feed(s.bytes[j], false, s.pooled[j] != 0);
            std::array<float, D> x{}; features(x); std::array<float, 8> p{}; policy(x, s.menu, p);
            int best = s.menu.front(); for (int a : s.menu) if (p[a] > p[best]) best = a;
            agree += best == s.label;
        }
        m.replay = false; m.set_mode(was, structural);
        std::memcpy(pool_sum_, saved_sum, sizeof saved_sum); pool_n_ = saved_n; pool_events_ = std::move(saved_events);
        return total ? (double)agree / (double)total : 0.0;
    }
    Metrics metrics;

private:
    struct PoolEvent { std::vector<Trace> tr; std::array<int8_t, Dims::IN> x1{}; bool novel = false; };
    struct Record {
        std::array<float, D> x{};
        std::vector<int> menu;
        std::array<float, 8> probs{};
        int chosen = -1, executed = -1; bool model = false; size_t t = 0;
        std::vector<uint8_t> ctx, ctx_pool;   // the situation as bytes (reencode only)
        std::vector<PoolEvent> events;   // bodies active in the pooled events; the decision event is last
    };
    Organism& m;
    TaskModelConfig c;
    std::vector<float> W, scale;          // readout; running RMS of each workspace feature
    float baseline_ = 0.f, demo_ema_ = 0.f; bool demo_ema_init_ = false;
    uint8_t last_ = 0;                    // matches the model's initial hist[0]
    std::mt19937_64 rng_;
    Record pending_; bool has_pending_ = false;
    float pool_sum_[Dims::F] = {0}; uint32_t pool_n_ = 0; std::vector<PoolEvent> pool_events_;
    struct Rehearsal { std::array<float, D> x{}; std::vector<int> menu; int label = -1; };
    std::vector<float> V;                                  // critic (used only when c.critic)
    std::vector<std::array<float, D>> mission_x_;          // features of this mission's decisions (critic)
    std::vector<Rehearsal> rehearsal_; uint64_t rehearsal_next_ = 0;
    struct Situation {
        std::vector<uint8_t> bytes, pooled; std::vector<int> menu; int label = -1;
        template <class Ar> void state_io(Ar& ar) { ar(bytes); ar(pooled); ar(menu); ar(label); }
    };
    static constexpr size_t ring_size = 1024;
    std::array<uint8_t, ring_size> ring_{}, ring_pool_{}; uint64_t ring_n_ = 0;   // recent bytes and their pooled flags
    std::vector<Situation> situations_; uint64_t situation_next_ = 0;
    bool capture_traces_ = false;
    struct Mark {
        enum Kind : uint8_t { k_state, k_outcome, k_decision } kind = k_state;
        uint32_t at = 0;                       // applies just before the byte at this position
        std::array<float, 8> state{}; float value = 0.f;
        std::vector<int> menu; int label = -1;
        bool self = false;                     // label is the model's own executed choice (success replay), not the teacher's
        template <class Ar> void state_io(Ar& ar) { ar(kind); ar(at); ar(state); ar(value); ar(menu); ar(label); ar(self); }
    };
    struct Tape {
        std::vector<uint8_t> bytes, pooled; std::vector<Mark> marks; uint32_t replays = 0; uint8_t tag = 0;
        template <class Ar> void state_io(Ar& ar) { ar(bytes); ar(pooled); ar(marks); ar(replays); ar(tag); }
        bool empty() const { return bytes.empty(); }
        void clear() { bytes.clear(); pooled.clear(); marks.clear(); }
    };
    Tape tape_; bool tape_has_demo_ = false, tape_has_self_ = false, suppressing_ = false;   // the mission being recorded
    bool tape_any_teacher_ = false, reading_text_ = false, attempt_ = false; uint8_t mission_tag_ = 0;
    std::vector<std::pair<Tape, bool>> attempts_;   // frozen attempts at the current mission (transient, never at a checkpoint)
    // One hash per decision: the pooled observation bytes since the previous decision, plus the menu.
    std::vector<std::pair<size_t, uint64_t>> decision_states(const Tape& t) const {
        std::vector<std::pair<size_t, uint64_t>> states; size_t from = 0;
        for (size_t i = 0; i < t.marks.size(); ++i) {
            const Mark& k = t.marks[i]; if (k.kind != Mark::k_decision) continue;
            uint64_t h = 1469598103934665603ULL;
            for (size_t p = from; p < k.at && p < t.bytes.size(); ++p) if (t.pooled[p]) { h ^= t.bytes[p]; h *= 1099511628211ULL; }
            for (int a : k.menu) { h ^= (uint64_t)(a + 1) << 8; h *= 1099511628211ULL; }
            states.push_back({i, h});
            from = k.at; while (from < t.bytes.size() && t.bytes[from] != '\n') ++from;
            if (from < t.bytes.size()) ++from;
        }
        return states;
    }
    std::array<uint64_t, 256> teacher_labels_by_tag_{};             // multi-option teacher labels seen per mission tag
    // Loop pruning: a self-labelled decision whose state recurs later in the mission was a loop iteration.
    size_t prune_loops(Tape& t) {
        const auto states = decision_states(t);
        size_t pruned = 0;
        for (size_t a = 0; a < states.size(); ++a) {
            Mark& k = t.marks[states[a].first];
            if (!k.self || k.label < 0) continue;
            for (size_t b = a + 1; b < states.size(); ++b) if (states[b].second == states[a].second) { k.label = -1; k.self = false; ++pruned; break; }
        }
        return pruned;
    }
    std::vector<Tape> tapes_; uint64_t tapes_seen_ = 0;               // teacher-labelled missions
    std::vector<Tape> successes_; uint64_t successes_seen_ = 0;       // verified successes, labelled with the model's own choices
    bool recording() const { return (c.replay_missions || c.success_replay) && ((!m.replay && m.params().train) || attempt_); }
    void reservoir(std::vector<Tape>& store, uint64_t& seen, uint32_t capacity, Tape&& t) {   // uniform over the whole run
        ++seen;
        if (store.size() < capacity) store.push_back(std::move(t));
        else { const uint64_t j = std::uniform_int_distribution<uint64_t>(0, seen - 1)(rng_); if (j < capacity) store[(size_t)j] = std::move(t); }
    }
    void store_tape(float reward) {
        if (!tape_.empty()) {
            // A verified success keeps every label, including the model's own choices: what the model did is rehearsed.
            if (c.success_replay && reward > 0.f && tape_has_self_ && (!c.success_solo || !tape_any_teacher_)) {
                Tape copy = tape_; copy.tag = mission_tag_;
                if (c.success_prune) metrics.labels_pruned += prune_loops(copy);
                reservoir(successes_, successes_seen_, c.success_capacity, std::move(copy)); ++metrics.successes_stored;
            }
            for (const Mark& k : tape_.marks) if (k.kind == Mark::k_decision && k.label >= 0 && !k.self && k.menu.size() > 1) ++teacher_labels_by_tag_[mission_tag_];
            // Teacher tapes carry only the teacher's labels; the model's own choices in a failed or mixed mission are not evidence.
            if (c.replay_missions && tape_has_demo_) {
                for (Mark& k : tape_.marks) if (k.self) { k.label = -1; k.self = false; }
                reservoir(tapes_, tapes_seen_, c.tape_capacity, std::move(tape_));
            }
        }
        tape_.clear(); tape_has_demo_ = false; tape_has_self_ = false; tape_any_teacher_ = false;
    }
public:
    // Extra teacher-tape replays outside the per-mission schedule (e.g. in proportion to interleaved text).
    void replay_extra(uint32_t n) { if (!n || tapes_.empty()) return; const uint32_t keep_t = c.replay_missions, keep_s = c.success_replay; c.replay_missions = n; c.success_replay = 0; replay_tapes(); c.replay_missions = keep_t; c.success_replay = keep_s; }
private:
    void replay_tapes() {
        if ((!c.replay_missions || tapes_.empty()) && (!c.success_replay || successes_.empty())) return;
        const bool was = m.params().train, structural = m.params().structural;
        float saved_sum[Dims::F]; std::memcpy(saved_sum, pool_sum_, sizeof saved_sum);
        const uint32_t saved_n = pool_n_; std::vector<PoolEvent> saved_events = std::move(pool_events_); pool_events_.clear();
        float saved_obs[8]; std::memcpy(saved_obs, m.task_obs, sizeof saved_obs);
        const float saved_value = m.task_value; const bool saved_reported = m.task_reported;
        m.set_mode(false, false); m.replay = true; capture_traces_ = c.reencode_bodies && c.body_learning;
        std::vector<const Tape*> chosen;
        for (uint32_t k = 0; k < c.replay_missions && !tapes_.empty(); ++k) chosen.push_back(&tapes_[std::uniform_int_distribution<size_t>(0, tapes_.size() - 1)(rng_)]);
        for (uint32_t k = 0; k < c.success_replay && !successes_.empty(); ++k) {
            std::vector<size_t> open; std::vector<double> w;
            for (size_t i = 0; i < successes_.size(); ++i) if (!c.success_budget || successes_[i].replays < c.success_budget) {
                open.push_back(i); w.push_back(c.success_stratify ? 1.0 / (1.0 + (double)teacher_labels_by_tag_[successes_[i].tag] / 100.0) : 1.0);
            }
            if (open.empty()) break;
            Tape& t = successes_[open[std::discrete_distribution<size_t>(w.begin(), w.end())(rng_)]]; ++t.replays; chosen.push_back(&t);
        }
        for (const Tape* tp : chosen) {
            const Tape& tape = *tp;
            std::fill(pool_sum_, pool_sum_ + Dims::F, 0.f); pool_n_ = 0; pool_events_.clear();
            size_t mk = 0; const Mark* pending = nullptr;
            for (size_t pos = 0; pos <= tape.bytes.size(); ++pos) {
                for (; mk < tape.marks.size() && tape.marks[mk].at == pos; ++mk) {
                    const Mark& e = tape.marks[mk];
                    if (e.kind == Mark::k_state) for (int j = 0; j < 8; ++j) m.task_obs[j] = e.state[j];
                    else if (e.kind == Mark::k_outcome) m.post_task_outcome(e.value);
                    else pending = &e;
                }
                if (pos == tape.bytes.size()) break;
                const uint8_t b = tape.bytes[pos]; feed(b, false, tape.pooled[pos] != 0);
                if (pending && b == '\n') {   // the decision frame's terminator: the decision event, as in decide()
                    Record r; features(r.x); r.menu = pending->menu; r.events = std::move(pool_events_); pool_events_.clear();
                    std::fill(pool_sum_, pool_sum_ + Dims::F, 0.f); pool_n_ = 0;
                    if (pending->label >= 0 && pending->menu.size() > 1) {
                        std::array<float, 8> p{}; policy(r.x, r.menu, p);
                        std::array<float, 8> err{}; for (int a : r.menu) err[a] = p[a] - (a == pending->label ? 1.f : 0.f);
                        if (capture_traces_) body_update(r, err, 0, 0.f);
                        const float step = c.readout_lr / norm2(r.x);
                        for (int a : r.menu) { float* row = &W[(size_t)a * D]; for (int kk = 0; kk < D; ++kk) row[kk] -= step * err[a] * r.x[kk]; }
                        ++metrics.rehearsed;
                    }
                    pending = nullptr;
                }
            }
        }
        capture_traces_ = false; m.replay = false; m.set_mode(was, structural);
        std::memcpy(m.task_obs, saved_obs, sizeof saved_obs); m.task_value = saved_value; m.task_reported = saved_reported;
        std::memcpy(pool_sum_, saved_sum, sizeof saved_sum); pool_n_ = saved_n; pool_events_ = std::move(saved_events);
    }
    void reencode() {
        if (!c.reencode || situations_.empty()) return;
        const bool was = m.params().train, structural = m.params().structural;
        // the step pool in progress belongs to the next decision: set it aside
        float saved_sum[Dims::F]; std::memcpy(saved_sum, pool_sum_, sizeof saved_sum);
        const uint32_t saved_n = pool_n_; std::vector<PoolEvent> saved_events = std::move(pool_events_); pool_events_.clear();
        m.set_mode(false, false); m.replay = true; capture_traces_ = c.reencode_bodies && c.body_learning;
        for (uint32_t i = 0; i < c.reencode; ++i) {
            const Situation& s = situations_[std::uniform_int_distribution<size_t>(0, situations_.size() - 1)(rng_)];
            std::fill(pool_sum_, pool_sum_ + Dims::F, 0.f); pool_n_ = 0; pool_events_.clear();
            for (size_t j = 0; j < s.bytes.size(); ++j) feed(s.bytes[j], false, s.pooled[j] != 0);
            Record r; features(r.x); r.menu = s.menu; r.events = std::move(pool_events_); pool_events_.clear();
            std::array<float, 8> p{}; policy(r.x, r.menu, p);
            std::array<float, 8> err{}; for (int a : r.menu) err[a] = p[a] - (a == s.label ? 1.f : 0.f);
            if (capture_traces_) body_update(r, err, 0, 0.f);
            const float step = c.readout_lr / norm2(r.x);
            for (int a : r.menu) { float* row = &W[(size_t)a * D]; for (int k = 0; k < D; ++k) row[k] -= step * err[a] * r.x[k]; }
            ++metrics.rehearsed;
        }
        capture_traces_ = false; m.replay = false; m.set_mode(was, structural);
        std::memcpy(pool_sum_, saved_sum, sizeof saved_sum); pool_n_ = saved_n; pool_events_ = std::move(saved_events);
    }
    static float dot(const std::vector<float>& w, const std::array<float, D>& x) { float s = 0.f; for (int k = 0; k < D; ++k) s += w[k] * x[k]; return s; }
    void rehearse() {
        if (!c.rehearsal || rehearsal_.empty()) return;
        for (uint32_t i = 0; i < c.rehearsal; ++i) {
            const Rehearsal& e = rehearsal_[std::uniform_int_distribution<size_t>(0, rehearsal_.size() - 1)(rng_)];
            std::array<float, 8> p{}; policy(e.x, e.menu, p);
            const float step = c.readout_lr / norm2(e.x);
            for (int a : e.menu) { const float err = p[a] - (a == e.label ? 1.f : 0.f); float* row = &W[(size_t)a * D]; for (int k = 0; k < D; ++k) row[k] -= step * err * e.x[k]; }
            ++metrics.rehearsed;
        }
    }
    std::vector<Record> episode_;

    void feed(uint8_t b, bool learned, bool pooled = false) {
        m.process_event(last_, b); last_ = b; ++metrics.bytes; metrics.learned_bytes += learned;
        if ((c.replay_missions || c.success_replay) && !reading_text_ && ((!m.replay && (m.params().train || suppressing_)) || attempt_)) { tape_.bytes.push_back(b); tape_.pooled.push_back(pooled); }
        if (c.reencode && !m.replay && !reading_text_) { ring_[ring_n_ % ring_size] = b; ring_pool_[ring_n_ % ring_size] = pooled; ++ring_n_; }
        if (!pooled) return;
        for (int k = 0; k < Dims::F; ++k) pool_sum_[k] += (float)m.ws[k];
        ++pool_n_;
        if (c.body_learning && (learned || capture_traces_) && pool_events_.size() < max_pooled_events) {
            PoolEvent e; e.tr.assign(m.traces.begin(), m.traces.begin() + m.ntrace);
            std::memcpy(e.x1.data(), m.agent_on ? m.xin_focus : m.xin, Dims::IN); e.novel = m.cur_novel;
            pool_events_.push_back(std::move(e));
        }
    }
    static float norm2(const std::array<float, D>& x) { float n = 1.f; for (float v : x) n += v * v; return n; }

    void features(std::array<float, D>& x) {
        const bool train = learning();
        for (int k = 0; k < Dims::F; ++k) {
            const float v[3] = {(float)m.ws[k], (float)m.ws_mem[k], pool_n_ ? pool_sum_[k] / (float)pool_n_ : 0.f};
            for (int j = 0; j < 3; ++j) {
                float& sc = scale[j * Dims::F + k];
                if (train) sc += 0.01f * (std::fabs(v[j]) - sc);   // normalisation statistics adapt only while learning
                x[j * Dims::F + k] = std::clamp(v[j] / (sc + 1.f), -4.f, 4.f);
            }
        }
        if (ablate_body_output) for (int k = 0; k < 3 * Dims::F; ++k) x[k] = 0.f;
        x[3 * Dims::F] = 1.f;
    }
    void policy(const std::array<float, D>& x, const std::vector<int>& menu, std::array<float, 8>& p) const {
        p.fill(0.f); float mx = -1e30f; std::array<float, 8> z{};
        for (int a : menu) { const float* row = &W[(size_t)a * D]; float s = 0.f; for (int k = 0; k < D; ++k) s += row[k] * x[k]; z[a] = s; mx = std::max(mx, s); }
        float t = 0.f; for (int a : menu) { p[a] = std::exp(z[a] - mx); t += p[a]; }
        for (int a : menu) p[a] /= t;
    }
    void demonstrate(Record& r, int label) {
        std::array<float, 8> p{}; policy(r.x, r.menu, p);
        const float logp = std::log(std::max(p[label], 1e-6f));
        if (!demo_ema_init_) { demo_ema_ = logp; demo_ema_init_ = true; }
        const float gain_bits = (logp - demo_ema_) / 0.6931471805599453f;
        demo_ema_ += 0.01f * (logp - demo_ema_);
        std::array<float, 8> err{};
        for (int a : r.menu) err[a] = p[a] - (a == label ? 1.f : 0.f);
        if (c.body_learning) body_update(r, err, (int32_t)std::lround(std::clamp(c.demo_mod * gain_bits, -127.f, 127.f)), 0.f);
        const float step = c.readout_lr / norm2(r.x);
        for (int a : r.menu) { float* row = &W[(size_t)a * D]; for (int k = 0; k < D; ++k) row[k] -= step * err[a] * r.x[k]; }
    }
    // err[a] = dLoss/dlogit[a], taken before this decision's readout update. The decision event's own
    // emissions reach the logits through the decision workspace and the persistent workspace (ws_mem = decayed
    // ws_mem + ws); every pooled event's emissions reach them through the pooled mean. Each set of bodies gets
    // its exact local output error through its L6 layer, in the byte readout's units (workspace / 128).
    void body_update(const Record& r, const std::array<float, 8>& err, int32_t M, float credit) {
        if (r.events.empty() || m.freeze_weights) return;
        float g_dec[Dims::F] = {0}, g_pool[Dims::F] = {0};
        for (int a : r.menu) { if (err[a] == 0.f) continue; const float* row = &W[(size_t)a * D];
            for (int k = 0; k < Dims::F; ++k) {
                g_dec[k] += err[a] * (row[k] / (scale[k] + 1.f) + row[Dims::F + k] / (scale[Dims::F + k] + 1.f));
                g_pool[k] += err[a] * row[2 * Dims::F + k] / (scale[2 * Dims::F + k] + 1.f);
            } }
        auto quantise = [&](const float* g) {
            bool any = false;
            for (int k = 0; k < Dims::F; ++k) {
                float q = g[k] * 128.f * m.params().err_gain * 127.f * c.l6_gain; q = std::clamp(q, -127.f, 127.f);
                m.q6[k] = (int8_t)(int)(q + (q >= 0.f ? 0.5f : -0.5f)); any |= m.q6[k] != 0;
            }
            return any;
        };
        const PoolEvent& decision = r.events.back();
        if (quantise(g_dec)) { m.apply_l6_error(decision.tr.data(), (uint32_t)decision.tr.size(), decision.novel); ++metrics.l6_task_updates; }
        // pooled mean: each event carries 1/N of the pooled gradient; kept at full scale so the INT8 error survives quantisation
        if (quantise(g_pool)) for (const auto& e : r.events) { m.apply_l6_error(e.tr.data(), (uint32_t)e.tr.size(), e.novel); ++metrics.l6_task_updates; }
        if (M) for (const auto& e : r.events) { m.modulated_on(e.tr.data(), (uint32_t)e.tr.size(), e.x1.data(), M, e.novel); ++metrics.modulated_task_updates; }
        if (credit != 0.f && c.body_credit > 0.f) {   // the bodies (and routes) that produced this decision
            const Params& p = m.params(); const float cc = c.body_credit * credit;
            for (const auto& e : r.events) for (const Trace& tr : e.tr) {
                if (!tr.emitted || !m.alive[tr.unit]) continue;
                m.resist[tr.unit] = (uint8_t)std::max(p.resist_min, std::min(p.resist_max, (int32_t)m.resist[tr.unit] - (int32_t)std::lround(cc * p.resist_down)));
                if (m.unit_lock[tr.unit] != UINT32_MAX) m.tree.locks[m.unit_lock[tr.unit]].utility += 0.1f * cc;
                ++metrics.credited_bodies;
                if (tr.source == SRC_ROUTED) { m.route_utility[tr.unit] += c.route_credit * cc; ++metrics.credited_routes; }
            }
        }
    }
};

} // namespace laminar
