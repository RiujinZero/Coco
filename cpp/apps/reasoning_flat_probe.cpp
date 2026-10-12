#include "laminar/procedure_memory.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
using namespace laminar::reasoning;

std::vector<Sample> read_samples(const std::string& path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("missing " + path);
    std::vector<Sample> out; Sample s; int n;
    while (f >> s.action >> n) {
        s.available.resize(n); for (int& a : s.available) f >> a;
        for (float& v : s.obs) f >> v;
        assert(f); out.push_back(s);
    }
    return out;
}
std::vector<VerifiedQuestion> read_questions(const std::string& path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("missing " + path);
    std::vector<VerifiedQuestion> out; VerifiedQuestion e; int a, b, n;
    while (f >> e.query.cue >> a >> b >> e.query.relation[0] >> e.query.relation[1] >> e.answer.type >> e.answer.id >> n) {
        e.query.a = {0, a}; e.query.b = {0, b}; e.query.facts.resize(n);
        for (auto& fact : e.query.facts) f >> fact.subject >> fact.relation >> fact.object;
        assert(f); out.push_back(e);
    }
    return out;
}
int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: flat_probe skill_corpus flat_corpus seed output_json\n"; return 2; }
    int seed = std::stoi(argv[3]); const std::string tail = "_" + std::to_string(seed) + ".txt";
    ProcedureMemory learner(seed);
    auto micro = read_samples(std::string(argv[1]) + "/micro" + tail);
    std::mt19937 rng(seed + 400);
    for (int pass = 0; pass < 5; ++pass) {
        std::shuffle(micro.begin(), micro.end(), rng);
        for (const auto& s : micro) learner.skills.fit(s);
    }
    auto training = read_questions(std::string(argv[2]) + "/train" + tail);
    auto heldout = read_questions(std::string(argv[2]) + "/test" + tail);
    const auto weights = learner.skills.kernel.routes.W;
    std::ofstream out(argv[4]); out << "{\"seed\":" << seed << ",\"conditions\":{";
    const char* families[] = {"follow", "count", "equal", "count_through", "equal_counts", "equal_counts_through", "equal_ends"};
    bool first_mode = true;
    for (const std::string mode : {"one_example", "twelve_examples", "no_bindings"}) {
        learner.procedures.clear();
        if (!first_mode) out << ','; first_mode = false;
        out << '"' << mode << "\":{";
        for (int cue = 0; cue < 7; ++cue) {
            std::vector<VerifiedQuestion> examples;
            for (const auto& e : training) if (e.query.cue == cue && (mode != "one_example" || examples.empty())) examples.push_back(e);
            auto result = learner.learn(examples, mode == "no_bindings");
            auto learned_programs = learner.procedures;
            int correct = 0, n = 0;
            for (const auto& e : heldout) if (e.query.cue == cue) {
                // No test target enters the runtime or search.
                Value answer = learner.answer(e.query);
                correct += answer == e.answer; ++n;
            }
            assert(weights == learner.skills.kernel.routes.W && learned_programs == learner.procedures);
            if (cue) out << ',';
            out << '"' << families[cue] << "\":{\"correct\":" << correct << ",\"n\":" << n
                << ",\"search_states\":" << result.states << ",\"found\":" << (result.found ? "true" : "false") << ",\"program\":[";
            for (size_t i = 0; i < result.program.size(); ++i) { if (i) out << ','; out << result.program[i]; }
            out << "]}";
        }
        out << '}';
    }
    out << "}}\n";
}
