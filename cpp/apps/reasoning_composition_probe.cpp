#include "laminar/reasoning_runtime.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

using namespace laminar::reasoning;
std::vector<Sample> samples(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("missing " + path);
    std::vector<Sample> out;
    Sample s; int n;
    while (f >> s.action >> n) {
        s.available.resize(n); for (int& a : s.available) f >> a;
        for (float& v : s.obs) f >> v;
        assert(f); out.push_back(s);
    }
    return out;
}
struct Test { std::string family; Goal goal; Value answer; };
std::vector<Test> tests(const std::string& path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("missing " + path);
    std::vector<Test> out;
    Test x; int nodes, facts;
    while (f >> x.family >> nodes >> x.goal.root >> facts >> x.answer.type >> x.answer.id) {
        x.goal.nodes.resize(nodes); x.goal.facts.resize(facts);
        for (auto& n : x.goal.nodes) f >> n.kind >> n.left >> n.right >> n.relation >> n.literal.type >> n.literal.id;
        for (auto& r : x.goal.facts) f >> r.subject >> r.relation >> r.object;
        assert(f); out.push_back(x);
    }
    return out;
}
struct Score { int correct = 0, n = 0, timeout = 0, invalid = 0; long long steps = 0, micro = 0, calls = 0; };
int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: composition_probe corpus_dir seed output_json [replay|outcome]\n"; return 2; }
    const std::string dir = argv[1], suffix = "_" + std::string(argv[2]) + ".txt";
    int seed = std::stoi(argv[2]);
    std::string learning = argc > 4 ? argv[4] : "replay";
    Runtime r(seed);
    auto micro = samples(dir + "/micro" + suffix), dispatch = samples(dir + "/dispatch" + suffix);
    auto control = learning == "replay" ? samples(dir + "/control" + suffix) : std::vector<Sample>{};
    std::mt19937 rng(seed + 400);
    for (int pass = 0; pass < 5; ++pass) {
        for (auto* v : {&micro, &dispatch, &control}) std::shuffle(v->begin(), v->end(), rng);
        for (const auto& s : micro) r.skills.fit(s);
        for (const auto& s : dispatch) r.dispatch.fit(s);
        if (learning == "replay") for (const auto& s : control) r.control.fit(s);
    }
    if (learning != "replay") {
        auto train = tests(dir + "/train" + suffix);
        for (size_t i = 0; i < train.size(); ++i) {
            RunOptions opt; opt.record_choices = true;
            opt.epsilon = std::max(0.02f, 0.4f * (1.f - (float)i / train.size()));
            auto y = r.run(train[i].goal, opt);
            float reward = y.output == train[i].answer ? 1.f : -1.f;
            if (learning == "random_reward") reward = (rng() & 1) ? 1.f : -1.f;
            if (learning == "monte_carlo") r.credit_monte_carlo(y, reward);
            else if (learning == "shared_error") r.credit_shared_error(y, reward);
            else if (learning == "broadcast" || learning == "broadcast_long") r.credit_broadcast(y, reward, learning == "broadcast_long");
            else r.credit_control(y, reward);
        }
    }
    const auto ms = r.skills.kernel.routes.W, ds = r.dispatch.kernel.routes.W, cs = r.control.kernel.routes.W;
    const std::string test_name = argc > 5 ? argv[5] : "test";
    auto heldout = tests(dir + "/" + test_name + suffix);
    std::ofstream out(argv[3]);
    out << "{\"seed\":" << seed << ",\"learning\":\"" << learning << "\",\"training_samples\":[" << micro.size() << ',' << dispatch.size() << ',' << (learning == "replay" ? control.size() : 0) << "],\"conditions\":{";
    const std::vector<std::string> modes{"routes", "pool_reset", "pool_persistent", "no_bindings", "root_binding", "shared_scratch"};
    bool first_mode = true;
    for (const auto& mode : modes) {
        RunOptions opt;
        opt.pool = mode.rfind("pool", 0) == 0;
        opt.reset_competition = mode != "pool_persistent";
        opt.forget_return = mode == "no_bindings";
        opt.root_binding = mode == "root_binding";
        opt.shared_scratch = mode == "shared_scratch";
        std::map<std::string, Score> scores;
        for (const auto& x : heldout) {
            auto y = r.run(x.goal, opt);
            auto& s = scores[x.family]; ++s.n;
            s.correct += y.output == x.answer; s.timeout += y.timeout; s.invalid += y.invalid;
            s.steps += y.control_steps; s.micro += y.skill_steps; s.calls += y.calls;
        }
        assert(ms == r.skills.kernel.routes.W && ds == r.dispatch.kernel.routes.W && cs == r.control.kernel.routes.W);
        if (!first_mode) out << ','; first_mode = false;
        out << '"' << mode << "\":{";
        bool first = true;
        for (const auto& [family, s] : scores) {
            if (!first) out << ','; first = false;
            out << '"' << family << "\":{\"correct\":" << s.correct << ",\"n\":" << s.n
                << ",\"timeouts\":" << s.timeout << ",\"invalid\":" << s.invalid
                << ",\"steps\":" << s.steps << ",\"micro_steps\":" << s.micro << ",\"calls\":" << s.calls << '}';
        }
        out << '}';
    }
    out << "}}\n";
}
