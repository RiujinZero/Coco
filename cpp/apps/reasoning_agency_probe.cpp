// Isolated reasoning VM wired to the EXISTING AgencyKernel, not Organism.
// Corpus/operations match python/reasoning_primitives_probe.py.
// g++ -std=c++20 -O3 -Icpp/include cpp/apps/reasoning_agency_probe.cpp -o build-wsl/reasoning_agency_probe
#include "laminar/agency.h"
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <tuple>

using namespace laminar::agency;
struct Ex {
    int task, size, subject, target, type, answer;
    std::vector<std::array<int, 3>> facts;
};
bool read(std::istream& in, Ex& x) {
    int n;
    if (!(in >> x.task >> x.size >> x.subject >> x.target >> x.type >> x.answer >> n)) return false;
    x.facts.resize(n);
    for (auto& f : x.facts) in >> f[0] >> f[1] >> f[2];
    assert(in);
    return true;
}
struct VM {
    const Ex& ex;
    int focus = -1, value = -1, cursor = -1, acc = 0, pred = -1, last = -1;
    int output_type = -1, output = -1;
    bool exhausted = false, done = false;
    std::set<int> seen;
    explicit VM(const Ex& x) : ex(x) {}
    std::vector<int> available() const {
        std::vector<int> a{0, 4, 9};
        if (focus >= 0) a.insert(a.end(), {1, 5, 6});
        if (value >= 0) a.insert(a.end(), {2, 3, 7});
        if (pred >= 0) a.push_back(8);
        return a;
    }
    std::array<float, 32> obs() const {
        std::array<float, 32> o{};
        o[ex.task] = 1;
        const bool flags[] = {focus >= 0, value >= 0, focus == ex.subject,
                              exhausted, value >= 0 && seen.count(value), acc == 0};
        for (int i = 0; i < 6; ++i) o[3 + 2*i + (flags[i] ? 1 : 0)] = 1;
        o[15 + pred + 1] = 1;
        o[18 + last + 1] = 1;
        o[31] = 1;
        return o;
    }
    void step(int a) {
        if (a == 0) { focus = ex.subject; value = -1; cursor = -1; exhausted = false; }
        if (a == 1) {
            value = -1;
            for (int i = cursor + 1; i < (int)ex.facts.size(); ++i) {
                const auto& f = ex.facts[i];
                if (f[0] == focus && f[1] == 0) { cursor = i; value = f[2]; break; }
            }
            exhausted = value < 0;
            if (exhausted) cursor = (int)ex.facts.size();
        }
        if (a == 2) { focus = value; value = -1; cursor = -1; exhausted = false; }
        if (a == 3) seen.insert(value);
        if (a == 4) ++acc;
        if (a == 5) pred = focus == ex.target;
        if (a == 6) { output_type = 0; output = focus; }
        if (a == 7) { output_type = 0; output = value; }
        if (a == 8) { output_type = 1; output = pred; }
        if (a == 9) { output_type = 2; output = acc; }
        last = a;
        done = a >= 6;
    }
};

int oracle(const VM& v) {
    if (v.focus < 0) return 0;
    if (v.ex.task == 2) {
        if (v.exhausted) return 9;
        if (v.last == 3) return 4;
        if (v.last == 4) return 1;
        return v.value >= 0 && !v.seen.count(v.value) ? 3 : 1;
    }
    if (v.exhausted) return v.ex.task == 0 ? 6 : v.pred < 0 ? 5 : 8;
    return v.value >= 0 ? 2 : 1;
}

struct Score { int correct = 0, n = 0, timeout = 0; long long steps = 0; };
int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: probe corpus_dir seed episodes default|long|oracle|taught_routes|taught_pool|taught_reset\n"; return 2; }
    const std::string dir = argv[1], mode = argv[4];
    const int seed = std::stoi(argv[2]), episodes = std::stoi(argv[3]);
    const bool taught = mode.rfind("taught_", 0) == 0;
    KernelConfig c;
    c.obs_dim = 32; c.n_tendencies = 10; c.n_goals = 1; c.consequence_dim = 3;
    c.baseline_cost = 0; c.focus_active = false; c.seed = seed;
    // One VM instruction per commitment, with operand validity as the only mask.
    c.max_duration = 1; c.hysteresis = 0; c.commit_threshold = 0; c.commit_margin = 0;
    if (mode == "long") { c.trace_lifetime = 128; c.trace_decay = 0.99f; }
    AgencyKernel k; k.init(c); k.traces.max_entries = 128;
    k.commitment.release_suppression = 0;  // expiry means instruction completion, not a failed action
    std::mt19937 rng(seed + 20);
    std::uniform_real_distribution<float> uniform(0, 1);
    std::map<std::pair<int, int>, Score> scores;
    long long invalid = 0, missing = 0;
    for (int phase = 0; phase < 2; ++phase) {
        std::ifstream in(dir + (phase ? "/test_" : "/train_") + std::to_string(seed) + ".txt");
        if (!in) return 3;
        const auto old_routes = k.routes.W, old_sensory = k.sensory.W, old_drive = k.drive_motif.T;
        Ex x; int trial = 0;
        while (read(in, x)) {
            if (!phase && trial >= episodes) break;
            const bool learn = phase == 0 && mode != "oracle";
            // Episodic transient state must not credit an unrelated prior problem.
            k.traces.entries.clear(); k.traces.now = 0;
            k.commitment.tendency = -1; k.commitment.age = 0;
            k.last_emitted = -1; k.have_act = false; k.credited_since_emit = true;
            k.pool.reset(); std::fill(k.pool.adaptation.begin(), k.pool.adaptation.end(), 0);
            k.set_goal(0, 1);
            VM v(x);
            int steps = 0;
            const int budget = phase ? 128 : 48;
            for (; steps < budget && !v.done; ++steps) {
                const auto avail = v.available();
                int a;
                if (mode == "oracle") a = oracle(v);
                else if (taught) {
                    auto o = v.obs(); k.perceive(o.data(), false);
                    std::fill(k.pool.available.begin(), k.pool.available.end(), 0);
                    for (int j : avail) k.pool.available[j] = 1;
                    k.routes.predict_all(k.cur_idx, k.cur_val, k.pred_all);
                    if (learn) {
                        // Verified one-instruction demonstrations, not delayed
                        // task reward. Use the existing local route update.
                        a = oracle(v);
                        EligibilityTraces immediate;
                        for (int j : avail) {
                            immediate.entries.clear();
                            immediate.record(j, k.cur_idx, k.cur_val);
                            float error[3] = {(j == a ? 1.f : 0.f) - k.pred_all[j * 3], 0.f, 0.f};
                            three_factor_update(k.routes, immediate, error, 0.1f);
                        }
                    } else if (mode == "taught_routes") {
                        a = avail[0];
                        for (int j : avail) if (k.pred_all[j * 3] > k.pred_all[a * 3]) a = j;
                    } else {
                        if (mode == "taught_reset") {
                            k.pool.reset(); k.commitment.tendency = -1;
                            std::fill(k.pool.adaptation.begin(), k.pool.adaptation.end(), 0);
                        }
                        a = k.deliberate(false, true);
                    }
                }
                else {
                    auto o = v.obs(); k.perceive(o.data(), learn);
                    std::fill(k.pool.available.begin(), k.pool.available.end(), 0);
                    for (int j : avail) k.pool.available[j] = 1;
                    float eps = std::max(0.04f, 0.4f * (1 - (float)trial / episodes));
                    if (learn && uniform(rng) < eps) {
                        a = avail[rng() % avail.size()];
                        k.commitment.tendency = a; k.commitment.goal = 0; k.commitment.age = 0;
                        k.routes.predict_all(k.cur_idx, k.cur_val, k.pred_all);
                        k.routes.effectiveness(k.pred_all, k.eff);
                        k.emit(a);
                    } else a = k.deliberate(false, true);
                }
                if (a < 0) { ++missing; continue; }
                if (std::find(avail.begin(), avail.end(), a) == avail.end()) { ++invalid; continue; }
                v.step(a);
            }
            const bool correct = v.done && v.output_type == x.type && v.output == x.answer;
            if (learn && !taught) {
                const float result[3] = {correct ? 1.f : -1.f, steps / 48.f, 1.f};
                k.consequence(result, true);  // the ONLY task feedback, after the answer/timeout
            }
            if (phase) {
                auto& s = scores[{x.task, x.size}]; ++s.n; s.correct += correct;
                s.steps += steps; s.timeout += !v.done;
            }
            ++trial;
        }
        if (phase) {
            // Test reads may change transient competition, never learned weights.
            assert(old_routes == k.routes.W && old_sensory == k.sensory.W && old_drive == k.drive_motif.T);
        }
    }
    const char* tasks[] = {"follow", "compare", "count"};
    std::cout << "{\"seed\":" << seed << ",\"condition\":\"agency_" << mode << "\",\"metrics\":{";
    bool first = true;
    for (const auto& [key, s] : scores) {
        if (!first) std::cout << ','; first = false;
        std::cout << '"' << tasks[key.first] << '/' << key.second << "\":{\"correct\":" << s.correct
                  << ",\"n\":" << s.n << ",\"steps\":" << s.steps << ",\"timeouts\":" << s.timeout << '}';
    }
    std::cout << "},\"invalid_actions\":" << invalid << ",\"missing_actions\":" << missing << "}\n";
}
