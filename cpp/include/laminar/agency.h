#pragma once
// Port of the Agkern agency kernel (projects/Agkern, agkern/*.py) to plain C++.
//
// The runtime loop of docs/architecture.md section 8.1, in the same order and with each step mapped
// to the same native primitive as the Python source:
//
//   admit                                    -> (single stream here; ModalityBudget omitted)
//   settle predictions against observation   -> SensoryPredictor (its own signal, never merged)
//   update local eligibility                 -> EligibilityTraces (decay AND hard lifetime)
//   on consequences arriving:
//       update action-conditioned predictions -> ConsequenceRoutes (one plastic layer per tendency)
//       update preference / viability         -> SignalRegister::satisfy
//       apply native credit to eligible routes-> DanGate + three_factor_update
//   activate current tendencies              -> ConsequenceRoutes::effectiveness (contrast vs default)
//   bias by consequences and current needs   -> DriveMotif (shared bilinear, locally adapted)
//   run native competition                   -> CompetitionPool (one shared inhibitory pool)
//   establish / maintain commitment          -> Commitment (hysteresis + the six release rules)
//   emit through committed activity          -> a focus operation on FocusField
//
// Invariants carried over: the expansion never learns; the competition pool has no dopamine input;
// commitment never sees sensory surprise; credit touches only k_active x consequence_dim routes;
// no backward pass anywhere.
#include <cstdint>
#include <cmath>
#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <functional>
#include "rng.h"

namespace laminar::agency {

struct Gauss { Rng r; explicit Gauss(uint64_t s) : r(s) {}
    float next() { double u1 = r.uniform(), u2 = r.uniform(); if (u1 < 1e-12) u1 = 1e-12; return (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2)); } };

// ---------------------------------------------------------------------------------------------
// expansion.py: fixed random sparse projection + k-winners-take-all. Zero learned parameters.
struct SparseExpansion {
    int in_dim = 0, n_units = 0, k_active = 0, claws = 0;
    std::vector<int> idx;      // n_units * claws
    std::vector<float> w;      // n_units * claws
    std::vector<float> drive;
    template <class Ar> void state_io(Ar& ar) { ar(in_dim); ar(n_units); ar(k_active); ar(claws); ar(idx); ar(w); ar(drive); }
    void init(int in, int units, int k, int claws_, uint64_t seed) {
        in_dim = in; n_units = units; k_active = std::min(k, units); claws = std::max(1, std::min(in, claws_));
        idx.assign((size_t)units * claws, 0); w.assign((size_t)units * claws, 0.f); drive.assign(units, 0.f);
        Rng r(seed); Gauss g(seed ^ 0x5EEDULL);
        std::vector<int> perm(in);
        for (int u = 0; u < units; ++u) {
            for (int i = 0; i < in; ++i) perm[i] = i;
            for (int i = in - 1; i > 0; --i) { int j = (int)r.below((uint32_t)i + 1); std::swap(perm[i], perm[j]); }
            for (int c = 0; c < claws; ++c) { idx[(size_t)u * claws + c] = perm[c]; w[(size_t)u * claws + c] = g.next() / std::sqrt((float)claws); }
        }
    }
    // (active indices, rectified values) of the k winners; the dense vector is never materialised beyond `drive`
    void expand(const float* x, std::vector<int>& out_idx, std::vector<float>& out_val) {
        for (int u = 0; u < n_units; ++u) { float s = 0.f; for (int c = 0; c < claws; ++c) s += w[(size_t)u * claws + c] * x[idx[(size_t)u * claws + c]]; drive[u] = s > 0.f ? s : 0.f; }
        // k-WTA by insertion into a small sorted list: O(n_units * k) with k tiny, no allocation
        out_idx.assign(k_active, -1); out_val.assign(k_active, -1.f);
        for (int u = 0; u < n_units; ++u) {
            float d = drive[u]; if (d <= out_val[k_active - 1]) continue;
            int i = k_active - 1; while (i > 0 && out_val[i - 1] < d) { out_val[i] = out_val[i - 1]; out_idx[i] = out_idx[i - 1]; --i; }
            out_val[i] = d; out_idx[i] = u;
        }
        for (int i = 0; i < k_active; ++i) if (out_idx[i] < 0) { out_idx[i] = i; out_val[i] = 0.f; }   // fewer than k positive: keep zero-valued winners
    }
};

// ---------------------------------------------------------------------------------------------
// traces.py: bounded-lifetime eligibility
struct TraceEntry { int tendency; std::vector<int> idx; std::vector<float> val; double born; float strength; template <class Ar> void state_io(Ar& ar) { ar(tendency); ar(idx); ar(val); ar(born); ar(strength); } };
struct EligibilityTraces {
    float decay = 0.85f; double lifetime = 8.0; size_t max_entries = 64; double now = 0.0;
    std::deque<TraceEntry> entries;
    void record(int t, const std::vector<int>& idx, const std::vector<float>& val, float gain = 1.f) {
        if (entries.size() >= max_entries) entries.pop_front();
        // reuse a retired entry's storage when one is available (no per-event allocation on the hot path)
        if (!spare.empty()) { TraceEntry e = std::move(spare.back()); spare.pop_back(); e.tendency = t; e.idx.assign(idx.begin(), idx.end()); e.val.assign(val.begin(), val.end()); e.born = now; e.strength = gain; entries.push_back(std::move(e)); }
        else entries.push_back(TraceEntry{t, idx, val, now, gain});
    }
    std::vector<TraceEntry> spare;
    template <class Ar> void state_io(Ar& ar) { ar(decay); ar(lifetime); ar(max_entries); ar(now); std::vector<TraceEntry> e(entries.begin(), entries.end()); ar(e); entries.assign(e.begin(), e.end()); ar(spare); }
    void tick(double dt = 1.0) {
        now += dt; float f = (float)std::pow(decay, dt);
        for (auto& e : entries) e.strength *= f;
        while (!entries.empty() && (now - entries.front().born) > lifetime) { if (spare.size() < 64) spare.push_back(std::move(entries.front())); entries.pop_front(); }
    }
    template <class F> void eligible(F&& fn) const { for (auto& e : entries) if ((now - e.born) <= lifetime) fn(e); }
};

// ---------------------------------------------------------------------------------------------
// signals.py: the register of signals the architecture forbids collapsing
struct SignalRegister {
    int n_goals = 0, cdim = 0; float demand_decay = 0.95f, demand_floor = 0.f, satisfaction_gain = 0.05f;
    std::vector<float> demand, setpoint, consequence_error;
    float sensory_error = 0.f, novelty = 0.f, verified_feedback = 0.f;
    template <class Ar> void state_io(Ar& ar) { ar(n_goals); ar(cdim); ar(demand_decay); ar(demand_floor); ar(satisfaction_gain); ar(demand); ar(setpoint); ar(consequence_error); ar(sensory_error); ar(novelty); ar(verified_feedback); }
    void init(int g, int c) { n_goals = g; cdim = c; demand.assign(g, 0.f); setpoint.assign(g, 0.f); consequence_error.assign(c, 0.f); }
    void set_demand(int goal, float v) { demand[goal] = v; setpoint[goal] = v; }
    void satisfy(const float* outcome) { for (int g = 0; g < n_goals; ++g) { float a = std::max(0.f, outcome[g]); demand[g] = std::max(demand_floor, demand[g] - satisfaction_gain * a); } }
    void tick(double dt = 1.0) { float rate = 1.f - (float)std::pow(demand_decay, dt); for (int g = 0; g < n_goals; ++g) demand[g] += rate * (setpoint[g] - demand[g]); }
    void set_consequence_error(const float* observed, const float* predicted) { for (int c = 0; c < cdim; ++c) consequence_error[c] = observed[c] - predicted[c]; }
    float demand_total() const { float s = 0.f; for (float d : demand) s += d; return s; }
    int argmax_demand() const { int b = 0; for (int g = 1; g < n_goals; ++g) if (demand[g] > demand[b]) b = g; return b; }
};

// ---------------------------------------------------------------------------------------------
// prediction.py: per-tendency plastic routes; effectiveness is the contrast against the default tendency
struct ConsequenceRoutes {
    int K = 0, n_units = 0, cdim = 0, n_goals = 0, default_t = 0; uint64_t reads = 0;
    std::vector<float> W;   // K * n_units * cdim, zero-initialised
    template <class Ar> void state_io(Ar& ar) { ar(K); ar(n_units); ar(cdim); ar(n_goals); ar(default_t); ar(reads); ar(W); }
    void init(int k, int units, int c, int g) { K = k; n_units = units; cdim = c; n_goals = g; W.assign((size_t)k * units * c, 0.f); }
    void predict(int t, const std::vector<int>& idx, const std::vector<float>& val, float* out) const {
        float vs = 1e-8f; for (float v : val) vs += v;
        for (int c = 0; c < cdim; ++c) out[c] = 0.f;
        for (size_t i = 0; i < idx.size(); ++i) { const float* row = &W[((size_t)t * n_units + idx[i]) * cdim]; for (int c = 0; c < cdim; ++c) out[c] += val[i] * row[c]; }
        for (int c = 0; c < cdim; ++c) out[c] /= vs;
    }
    // (K x cdim): every tendency's prediction, the whole menu
    void predict_all(const std::vector<int>& idx, const std::vector<float>& val, std::vector<float>& out) { out.resize((size_t)K * cdim); for (int t = 0; t < K; ++t) predict(t, idx, val, &out[(size_t)t * cdim]); reads += idx.size() * K; }
    void effectiveness(const std::vector<float>& pred_all, std::vector<float>& eff) const {
        eff.resize((size_t)K * n_goals);
        for (int t = 0; t < K; ++t) for (int g = 0; g < n_goals; ++g) eff[(size_t)t * n_goals + g] = pred_all[(size_t)t * cdim + g] - pred_all[(size_t)default_t * cdim + g];
    }
};

// plasticity.py: three-factor rule on eligible routes only
inline uint64_t three_factor_update(ConsequenceRoutes& R, const EligibilityTraces& traces, const float* dan, float lr, float decay = 0.f) {
    uint64_t touched = 0;
    traces.eligible([&](const TraceEntry& e) {
        for (size_t i = 0; i < e.idx.size(); ++i) {
            float pre = e.val[i] * e.strength; float* row = &R.W[((size_t)e.tendency * R.n_units + e.idx[i]) * R.cdim];
            for (int c = 0; c < R.cdim; ++c) { row[c] += lr * pre * dan[c]; if (decay > 0.f) row[c] -= decay * lr * row[c]; }
            ++touched;
        } });
    return touched;
}

// plasticity.py: the diffuse dopaminergic broadcast, weighted by demand on goal channels only
struct DanGate {
    int n_goals = 0, cdim = 0; float gain = 1.f; uint64_t broadcasts = 0; std::vector<float> last;
    template <class Ar> void state_io(Ar& ar) { ar(n_goals); ar(cdim); ar(gain); ar(broadcasts); ar(last); }
    void init(int g, int c) { n_goals = g; cdim = c; last.assign(c, 0.f); }
    // returns false when the event may not ground consequence learning
    bool broadcast(bool verified, const SignalRegister& s) {
        if (!verified) return false;
        for (int c = 0; c < cdim; ++c) { float scale = c < n_goals ? 0.25f + s.demand[c] : 1.f; last[c] = gain * s.consequence_error[c] * scale; }
        ++broadcasts; return true;
    }
};

// prediction.py: sensory prediction over the same expansion (reconstruction mode), kept separate
struct SensoryPredictor {
    int n_units = 0, obs_dim = 0; std::vector<float> W, last_prediction, last_error;
    template <class Ar> void state_io(Ar& ar) { ar(n_units); ar(obs_dim); ar(W); ar(last_prediction); ar(last_error); }
    void init(int units, int obs) { n_units = units; obs_dim = obs; W.assign((size_t)units * obs, 0.f); last_prediction.assign(obs, 0.f); last_error.assign(obs, 0.f); }
    void predict(const std::vector<int>& idx, const std::vector<float>& val) {
        float vs = 1e-8f; for (float v : val) vs += v;
        for (int o = 0; o < obs_dim; ++o) last_prediction[o] = 0.f;
        for (size_t i = 0; i < idx.size(); ++i) { const float* row = &W[(size_t)idx[i] * obs_dim]; for (int o = 0; o < obs_dim; ++o) last_prediction[o] += val[i] * row[o]; }
        for (int o = 0; o < obs_dim; ++o) last_prediction[o] /= vs;
    }
    float learn(const std::vector<int>& idx, const std::vector<float>& val, const float* observed, float lr) {
        float n2 = 0.f;
        for (int o = 0; o < obs_dim; ++o) { last_error[o] = observed[o] - last_prediction[o]; n2 += last_error[o] * last_error[o]; }
        for (size_t i = 0; i < idx.size(); ++i) { float* row = &W[(size_t)idx[i] * obs_dim]; for (int o = 0; o < obs_dim; ++o) row[o] += lr * val[i] * last_error[o]; }
        return std::sqrt(n2);
    }
};

// motifs.py: the shared bilinear valuation motif, adapted by the same three-factor family
struct DriveMotif {
    int n_goals = 0; std::vector<float> T, elig; bool has_elig = false;
    template <class Ar> void state_io(Ar& ar) { ar(n_goals); ar(T); ar(elig); ar(has_elig); }
    void init(int g) { n_goals = g; T.assign((size_t)g * g, 0.f); for (int i = 0; i < g; ++i) T[(size_t)i * g + i] = 1.f; elig.assign((size_t)g * g, 0.f); }
    // drive_k = eff_k . (T^T d)
    void drive(const std::vector<float>& demand, const std::vector<float>& eff, int K, std::vector<float>& out) const {
        std::vector<float> td(n_goals, 0.f);
        for (int g = 0; g < n_goals; ++g) for (int h = 0; h < n_goals; ++h) td[g] += T[(size_t)h * n_goals + g] * demand[h];
        out.assign(K, 0.f); for (int k = 0; k < K; ++k) for (int g = 0; g < n_goals; ++g) out[k] += eff[(size_t)k * n_goals + g] * td[g];
    }
    void record(const std::vector<float>& demand, const float* eff_k) { for (int g = 0; g < n_goals; ++g) for (int h = 0; h < n_goals; ++h) elig[(size_t)g * n_goals + h] = demand[g] * eff_k[h]; has_elig = true; }
    void local_learn(float dan_scalar, float lr = 0.01f, float max_delta = 0.05f) {
        if (!has_elig) return; std::vector<float> d(T.size()); float n2 = 0.f;
        for (size_t i = 0; i < T.size(); ++i) { d[i] = lr * dan_scalar * elig[i]; n2 += d[i] * d[i]; }
        float n = std::sqrt(n2); if (n > max_delta) for (auto& v : d) v *= max_delta / n;
        for (size_t i = 0; i < T.size(); ++i) T[i] += d[i]; has_elig = false;
    }
};

// ---------------------------------------------------------------------------------------------
// competition.py: selection as dynamics with ONE shared inhibitory pool; no dopamine parameter exists here
struct CompetitionPool {
    int n = 0; float alpha = 1.f, gamma = 0.5f, beta = 0.6f, tau = 1.f, dt = 0.3f, noise = 0.02f, z_floor = 0.03f, adapt_recovery = 0.85f, evoke = 0.35f;
    std::vector<float> z, l, adaptation; std::vector<uint8_t> available; Gauss g{0};
    template <class Ar> void state_io(Ar& ar) { ar(n); ar(alpha); ar(gamma); ar(beta); ar(tau); ar(dt); ar(noise); ar(z_floor); ar(adapt_recovery); ar(evoke); ar(z); ar(l); ar(adaptation); ar(available); ar(g); }
    void init(int n_, float gamma_, float beta_, float baseline_cost, float dt_, float noise_, float evoke_, uint64_t seed, bool scale_to_k4 = true) {
        // The source's beta was tuned with four tendencies. The pooled inhibition on a tendency is beta*(K-1)*z,
        // so at larger K the same beta drives every activation to the floor and leaves the winner to noise.
        // Port decision: keep the pooled total at its K=4 level.
        n = n_; gamma = gamma_; beta = (scale_to_k4 && n_ > 4) ? beta_ * 3.f / (float)(n_ - 1) : beta_; dt = dt_; noise = noise_; evoke = evoke_; g = Gauss(seed);
        z.assign(n, 0.3f); l.assign(n, baseline_cost); adaptation.assign(n, 0.f); available.assign(n, 1);
    }
    void reset(float v = 0.3f) { std::fill(z.begin(), z.end(), v); }
    void step(const std::vector<float>& drive, const std::vector<float>* bias, int settle) {
        float dmean = 0.f; for (float d : drive) dmean += d; dmean /= (float)n;
        for (int s = 0; s < settle; ++s) {
            float pooled = 0.f; for (float v : z) pooled += v;
            for (int k = 0; k < n; ++k) {
                float inhibition = beta * (pooled - z[k]);
                float net = alpha * drive[k] + (bias ? (*bias)[k] : 0.f) - l[k] - adaptation[k] - gamma * z[k] - inhibition;
                float evoked = evoke * std::max(0.f, drive[k] - dmean);
                float dz = (z[k] * net) / tau + evoked;
                float nz = z[k] + dz * dt + noise * g.next() * std::sqrt(dt);
                z[k] = available[k] ? std::max(z_floor, nz) : 0.f;
            }
            for (auto& a : adaptation) a *= adapt_recovery;
        }
    }
    void suppress(int k, float amount = 0.5f) { adaptation[k] += amount; z[k] = z_floor; }
    void winner(int& idx, float& act, float& margin) const {
        int b = 0; for (int k = 1; k < n; ++k) if (z[k] > z[b]) b = k;
        float second = -1e9f; for (int k = 0; k < n; ++k) if (k != b) second = std::max(second, z[k]);
        idx = b; act = z[b]; margin = n > 1 ? z[b] - second : z[b];
    }
    float critical_effectiveness(float demand_total) const { return l[0] / (alpha * std::max(demand_total, 1e-6f)); }
};

struct Commitment {
    int n = 0; float hysteresis = 0.3f, commit_threshold = 0.08f, commit_margin = 0.02f; int patience = 3, max_duration = 200, outcompeted_patience = 2; float release_suppression = 0.5f;
    int tendency = -1, goal = -1, age = 0, invalidation = 0, outcompeted = 0; uint64_t commits = 0;
    uint64_t releases[6] = {0}; // completion, invalid_action, duration_limit, goal_change, invalidated, outcompeted
    static const char* reason_name(int r) { static const char* n[6] = {"completion", "invalid_action", "duration_limit", "goal_change", "invalidated", "outcompeted"}; return n[r]; }
    uint64_t total_age = 0;   // sum of ages at release, for mean commitment duration
    void state_bias(std::vector<float>& b) const { b.assign(n, 0.f); if (tendency >= 0) b[tendency] = hysteresis; }
    void release(int reason, CompetitionPool* pool) {
        ++releases[reason]; total_age += (uint64_t)age;
        if (pool && tendency >= 0 && (reason == 2 || reason == 4 || reason == 1)) pool->suppress(tendency, release_suppression);   // SUPPRESS_ON
        tendency = -1; goal = -1; age = 0; invalidation = 0; outcompeted = 0;
    }
    int update(CompetitionPool& pool, const SignalRegister& s, const std::vector<float>& eff, bool consequence_arrived, bool action_valid = true) {
        if (tendency >= 0) {
            ++age; int g = goal >= 0 ? goal : s.argmax_demand(); float dmax = *std::max_element(s.demand.begin(), s.demand.end());
            if (s.demand[g] < 0.1f * std::max(dmax, 1e-6f) || s.demand[g] < 0.05f) release(0, &pool);
            else if (!action_valid || !pool.available[tendency]) release(1, &pool);
            else if (age >= max_duration) release(2, &pool);
            else if (s.argmax_demand() != g && dmax > s.demand[g] + 0.1f) release(3, &pool);
            else {
                float c_crit = pool.critical_effectiveness(s.demand_total()); float c_own = eff[(size_t)tendency * s.n_goals + g];
                if (consequence_arrived && c_own < c_crit) ++invalidation; else if (consequence_arrived) invalidation = 0;
                int w; float a, m; pool.winner(w, a, m); if (w != tendency) ++outcompeted; else outcompeted = 0;
                if (invalidation >= patience) release(4, &pool); else if (outcompeted >= outcompeted_patience) release(5, &pool);
            }
        }
        if (tendency < 0) { int w; float a, m; pool.winner(w, a, m); if (a >= commit_threshold && m >= commit_margin && pool.available[w]) { tendency = w; goal = s.argmax_demand(); age = 0; ++commits; } }
        return tendency;
    }
};

// ---------------------------------------------------------------------------------------------
// focus.py: the continuous focus parameter, a Gaussian gain over the observation index space, plus the
// discrete attended stream. Attenuation, never a hard mask: the floor keeps the unattended world learnable.
struct FocusField {
    bool active = false; int dim = 0; float width = 4.f, floor = 0.15f, step = 2.f, centre = 0.f; int stream = -1;
    std::vector<float> gain_; std::vector<int32_t> gain_q8;   // the same gain in Q8 fixed point for the INT8 input path
    template <class Ar> void state_io(Ar& ar) { ar(active); ar(dim); ar(width); ar(floor); ar(step); ar(centre); ar(stream); ar(gain_); ar(gain_q8); }
    void init(int d, float w, float fl, float st, bool act) { dim = d; width = w; floor = fl; step = st; active = act; centre = d / 2.f; gain_.assign(d, 1.f); gain_q8.assign(d, 256); update(); }
    void update() {
        if (!active) { std::fill(gain_.begin(), gain_.end(), 1.f); std::fill(gain_q8.begin(), gain_q8.end(), 256); return; }
        for (int i = 0; i < dim; ++i) { float u = ((float)i - centre) / width; gain_[i] = floor + (1.f - floor) * std::exp(-0.5f * u * u); gain_q8[i] = (int32_t)std::lround(gain_[i] * 256.f); }
    }
    void move_to(float c) { centre = std::max(0.f, std::min((float)dim - 1, c)); update(); }
    void nudge(float d) { move_to(centre + d); }
    const float* gain() const { return gain_.data(); }
};

// A focus operation is a RULE that moves the field from activity already present, never a target.
struct AgencyKernel;
using FocusOp = std::function<void(FocusField&, AgencyKernel&)>;

struct KernelConfig {
    int obs_dim = 64, n_tendencies = 4, n_goals = 2, consequence_dim = 4, n_units = 512, k_active = 16, claws = 6;
    float arousal = 0.30f, beta = 0.60f, gamma = 0.50f, baseline_cost = 0.10f, dt = 0.30f; int settle = 6; float noise = 0.02f, evoke = 0.35f;
    float hysteresis = 0.30f, commit_threshold = 0.08f, commit_margin = 0.02f; int patience = 3, max_duration = 200;
    float route_lr = 0.10f, sensory_lr = 0.10f, drive_lr = 0.005f, trace_decay = 0.85f; double trace_lifetime = 8; float novelty_gain = 0.12f;
    float focus_width = 4.f, focus_step = 2.f, focus_floor = 0.15f; int focus_dim = 64; bool focus_active = true;
    int epistemic_channel = -1; float progress_fast = 0.30f, progress_slow = 0.03f;
    uint64_t seed = 0;
};

struct AgencyKernel {
    KernelConfig cfg;
    SparseExpansion expansion; ConsequenceRoutes routes; SensoryPredictor sensory; SignalRegister signals;
    EligibilityTraces traces; DanGate dan; DriveMotif drive_motif; CompetitionPool pool; Commitment commitment; FocusField focus;
    std::vector<FocusOp> ops; std::vector<std::string> op_names;
    // current condition and the context of the last emitted act
    std::vector<int> cur_idx; std::vector<float> cur_val; bool have_current = false;
    int act_t = -1; std::vector<int> act_idx; std::vector<float> act_val; bool have_act = false;
    int last_emitted = -1; bool credited_since_emit = true;
    float err_fast = 0.f, err_slow = 0.f, learning_progress = 0.f; int effort_channel = -1;
    std::vector<float> pred_all, eff, drive, bias, last_drive;
    uint64_t events = 0, actions = 0, routes_updated = 0; std::vector<uint64_t> emitted_count;
    // Operation closures are not state: the owner rebuilds them by name after init().
    template <class Ar> void state_io(Ar& ar) {
        ar(cfg); ar(expansion); ar(routes); ar(sensory); ar(signals); ar(traces); ar(dan); ar(drive_motif); ar(pool); ar(commitment); ar(focus);
        ar(op_names); ar(cur_idx); ar(cur_val); ar(have_current); ar(act_t); ar(act_idx); ar(act_val); ar(have_act); ar(last_emitted); ar(credited_since_emit);
        ar(err_fast); ar(err_slow); ar(learning_progress); ar(effort_channel); ar(pred_all); ar(eff); ar(drive); ar(bias); ar(last_drive);
        ar(events); ar(actions); ar(routes_updated); ar(emitted_count); ar(observed); ar(predicted);
    }

    void init(const KernelConfig& c) {
        cfg = c;
        expansion.init(c.obs_dim, c.n_units, c.k_active, c.claws, c.seed);
        routes.init(c.n_tendencies, c.n_units, c.consequence_dim, c.n_goals);
        sensory.init(c.n_units, c.obs_dim);
        signals.init(c.n_goals, c.consequence_dim);
        traces.decay = c.trace_decay; traces.lifetime = c.trace_lifetime;
        dan.init(c.n_goals, c.consequence_dim);
        drive_motif.init(c.n_goals);
        pool.init(c.n_tendencies, c.gamma, c.beta, c.baseline_cost, c.dt, c.noise, c.evoke, c.seed);
        commitment.n = c.n_tendencies; commitment.hysteresis = c.hysteresis; commitment.commit_threshold = c.commit_threshold; commitment.commit_margin = c.commit_margin; commitment.patience = c.patience; commitment.max_duration = c.max_duration;
        focus.init(c.focus_dim, c.focus_width, c.focus_floor, c.focus_step, c.focus_active);
        effort_channel = (c.consequence_dim > c.n_goals + 1) ? c.n_goals + 1 : -1;
        emitted_count.assign(c.n_tendencies, 0);
    }
    void set_goal(int g, float v) { signals.set_demand(g, v); }

    // perceive: admit, settle the sensory prediction, advance eligibility; leaves the condition behind
    void perceive(const float* obs, bool may_learn = true) {
        ++events;
        expansion.expand(obs, cur_idx, cur_val); have_current = true;
        sensory.predict(cur_idx, cur_val);
        if (may_learn) { signals.sensory_error = sensory.learn(cur_idx, cur_val, obs, cfg.sensory_lr); update_learning_progress(signals.sensory_error); }
        traces.tick(); signals.tick();
    }
    // deliberate: value, compete, commit, emit (event mode: every event deliberates, emission when action is due)
    int deliberate(bool consequence_arrived, bool action_due) {
        if (!have_current) return -1;
        const int K = cfg.n_tendencies;
        routes.predict_all(cur_idx, cur_val, pred_all);
        routes.effectiveness(pred_all, eff);
        drive_motif.drive(signals.demand, eff, K, drive);
        for (int k = 0; k < K; ++k) drive[k] += cfg.arousal;
        if (cfg.novelty_gain > 0.f && effort_channel >= 0) {   // 1 - predicted effort: novelty read off the same routes
            float nov_mean = 0.f;
            for (int k = 0; k < K; ++k) { float nv = std::max(0.f, std::min(1.f, 1.f - pred_all[(size_t)k * cfg.consequence_dim + effort_channel])); drive[k] += cfg.novelty_gain * nv; nov_mean += nv; }
            signals.novelty = nov_mean / (float)K;
        }
        last_drive = drive;
        commitment.state_bias(bias);
        pool.step(drive, &bias, cfg.settle);
        int committed = commitment.update(pool, signals, eff, consequence_arrived, true);
        if (committed >= 0 && action_due) return emit(committed);
        return -1;
    }
    int emit(int t) {
        bool continuing = (last_emitted == t && !credited_since_emit);
        if (!continuing) traces.record(t, cur_idx, cur_val);
        last_emitted = t; credited_since_emit = false;
        act_t = t; act_idx.assign(cur_idx.begin(), cur_idx.end()); act_val.assign(cur_val.begin(), cur_val.end()); have_act = true;
        drive_motif.record(signals.demand, &eff[(size_t)t * cfg.n_goals]);
        ++actions; ++emitted_count[t];
        if (t < (int)ops.size() && ops[t]) ops[t](focus, *this);
        return t;
    }
    // a verified consequence arrived for the last act
    std::vector<float> observed, predicted;
    // `reported` (optional, one flag per channel): a channel with no news this
    // event is set to its own prediction, so silence is neither success nor failure.
    void consequence(const float* observed_in, bool verified = true, const uint8_t* reported = nullptr) {
        observed.assign(observed_in, observed_in + cfg.consequence_dim);
        if (cfg.epistemic_channel >= 0) observed[cfg.epistemic_channel] = learning_progress;   // the kernel reports on itself
        predicted.assign(cfg.consequence_dim, 0.f);
        if (have_act) routes.predict(act_t, act_idx, act_val, predicted.data());
        if (reported) for (int c = 0; c < cfg.consequence_dim; ++c) if (!reported[c]) observed[c] = predicted[c];
        float vf = 0.f; for (int g = 0; g < cfg.n_goals; ++g) vf += observed[g]; signals.verified_feedback = vf;
        signals.set_consequence_error(observed.data(), predicted.data());
        signals.satisfy(observed.data());
        if (!dan.broadcast(verified, signals)) return;
        credited_since_emit = true;
        routes_updated += three_factor_update(routes, traces, dan.last.data(), cfg.route_lr);
        float m = 0.f; for (int g = 0; g < cfg.n_goals; ++g) m += dan.last[g];
        drive_motif.local_learn(m, cfg.drive_lr);
    }
    void update_learning_progress(float e) { err_fast += cfg.progress_fast * (e - err_fast); err_slow += cfg.progress_slow * (e - err_slow); learning_progress = std::max(0.f, err_slow - err_fast); }
    double mean_commit_age() const { uint64_t rel = 0; for (auto r : commitment.releases) rel += r; return rel ? (double)commitment.total_age / (double)rel : 0.0; }
};

// ---- the focus operations of focus.py (rules over present activity), plus region/stream ops ----
inline void op_hold(FocusField&, AgencyKernel&) {}
inline void op_toward_error(FocusField& f, AgencyKernel& k) {
    const auto& e = k.sensory.last_error; if (e.empty()) return; int t = 0; for (size_t i = 1; i < e.size() && (int)i < f.dim; ++i) if (std::fabs(e[i]) > std::fabs(e[t])) t = (int)i;
    f.nudge(f.step * ((float)t > f.centre ? 1.f : -1.f));
}
inline void op_sweep(FocusField& f, AgencyKernel&) { f.nudge(f.step); if (f.centre >= f.dim - 1 || f.centre <= 0) f.step = -f.step; }
inline FocusOp op_move_to(float centre) { return [centre](FocusField& f, AgencyKernel&) { f.move_to(centre); }; }
inline FocusOp op_attend(int stream) { return [stream](FocusField& f, AgencyKernel&) { f.stream = stream; }; }

} // namespace laminar::agency
