#include "laminar/organism.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
using namespace laminar;
using namespace laminar::reasoning;

struct TextFact { std::string subject, object, text; int relation; };
struct Case { std::string group, gold, question; int family; std::vector<TextFact> facts; };
std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out; std::istringstream in(s); std::string v;
    while (std::getline(in, v, '\t')) out.push_back(v);
    return out;
}
std::vector<Case> read_cases(const std::string& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("missing " + path);
    std::vector<Case> out; std::string line;
    while (std::getline(in, line)) {
        auto h = split(line); assert(h.size() == 4);
        Case c; c.group = h[0]; c.family = std::stoi(h[1]); c.gold = h[2];
        std::getline(in, c.question);
        for (int i = 0; i < std::stoi(h[3]); ++i) {
            std::getline(in, line); auto f = split(line); assert(f.size() == 4);
            c.facts.push_back({f[0], f[2], f[3], std::stoi(f[1])});
        }
        out.push_back(c);
    }
    return out;
}
std::string prompt(const Case& c) {
    std::string s = "\n<|endofdialogue|>\nUSER: ";
    for (const auto& f : c.facts) s += f.text + "\n";
    return s + c.question + "\n\nASSISTANT:";
}
void feed(Organism& org, const std::string& text) {
    org.free_run = false;
    org.run_block((const uint8_t*)text.data(), text.size());
}
std::string generate(Organism& org) {
    org.free_run = true; std::string out;
    for (int i = 0; i < 96; ++i) {
        uint8_t pair[2] = {org.last_symbol(), 0}; org.run_block(pair, 2);
        const auto& p = org.distribution();
        int v = (int)(std::max_element(p.begin(), p.end()) - p.begin());
        org.commit_outcome((uint8_t)v); out += (char)v;
        if (v == '\n') break;
    }
    return LanguageReasoner::trim(out);
}
Params config(int seed) {
    Params p; p.seed = seed; p.units = 128; p.vocab = 256;
    p.train = false; p.structural = false; p.seq_window = 65536;
    p.seq_hash_bits = 10; p.seq_consolidate = false; p.seq_dialogues = 256;
    p.seq_chain = true; p.seq_chain_mode = 2; p.seq_max_depth = 8;
    p.seq_topic_gain = 16; p.seq_abstain = 0; p.seq_arbitrate = false;
    return p;
}
using FactKey = std::tuple<std::string, int, std::string>;
bool extraction_correct(const LanguageReasoner& r, const Case& c) {
    std::multiset<FactKey> want, got;
    for (const auto& f : c.facts) want.emplace(f.subject, f.relation, f.object);
    for (const auto& f : r.facts) got.emplace(r.names.at(f.subject), f.relation, r.names.at(f.object));
    return r.complete && want == got;
}
struct Score { int n = 0, exact = 0, extracted = 0, interpreted = 0, covered = 0; };
int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "usage: language_probe skill_corpus text_corpus seed output\n"; return 2; }
    int seed = std::stoi(argv[3]); auto reader = std::make_shared<LanguageReasoner>(seed);
    std::string tail = "_" + std::to_string(seed) + ".txt", root = std::string(argv[2]) + "/";
    auto training = read_cases(root + "train" + tail), test = read_cases(root + "test" + tail), adapt = read_cases(root + "adapt" + tail);
    { std::ifstream in(std::string(argv[1]) + "/micro" + tail); if (!in) throw std::runtime_error("missing skills");
      std::vector<Sample> samples; Sample s; int n;
      while (in >> s.action >> n) { s.available.resize(n); for (int& a : s.available) in >> a; for (float& x : s.obs) in >> x; assert(in); samples.push_back(s); }
      std::mt19937 rng(seed + 400);
      for (int pass = 0; pass < 5; ++pass) { std::shuffle(samples.begin(), samples.end(), rng); for (const auto& x : samples) reader->memory.skills.fit(x); } }
    // Ground only sentence frames in training. Evaluation annotations never go
    // to teach_fact, observe, prepare or the Organism input interface.
    for (const auto& c : training) for (const auto& f : c.facts)
        reader->teach_fact(f.text, f.subject, f.relation, f.object);
    Organism org(config(seed)), baseline(config(seed)); org.attach_reasoner(reader);
    for (const auto& c : training) {
        feed(org, prompt(c)); assert(reader->add_verified_answer(c.gold));
        feed(baseline, prompt(c) + " " + c.gold + "\n");
    }
    const size_t learned = reader->learn();
    std::map<std::string, Score> scores;
    std::ofstream examples(std::string(argv[4]) + ".examples.txt");
    auto evaluate = [&](const std::string& condition, const std::vector<Case>& cases, bool known_only = false) {
        const auto weights = reader->memory.skills.kernel.routes.W;
        const auto procedures = reader->memory.procedures; const auto forms = reader->forms;
        std::map<int, size_t> counts; for (const auto& [cue, e] : reader->evidence) counts[cue] = e.size();
        int index = 0;
        for (const auto& c : cases) {
            if (known_only && c.group != "known") continue;
            feed(org, prompt(c) + (++index % 2 ? " " : ""));
            auto& sc = scores[condition + "/" + c.group + "/" + std::to_string(c.family)]; ++sc.n;
            sc.extracted += extraction_correct(*reader, c);
            sc.interpreted += reader->prepared && reader->forms.count(reader->form);
            sc.covered += reader->status == "answer";
            const auto reply = generate(org); sc.exact += reply == c.gold;
            if (sc.n == 1) {
                examples << condition << '/' << c.group << '/' << c.family << "\n" << c.question
                    << "\nreply=" << reply << "\ngold=" << c.gold << "\nstatus=" << reader->status << "\nprogram=";
                auto it = reader->memory.procedures.find(reader->query.cue);
                if (it != reader->memory.procedures.end()) for (int a : it->second) examples << a << ' ';
                examples << "\n\n";
            }
        }
        assert(weights == reader->memory.skills.kernel.routes.W && procedures == reader->memory.procedures && forms == reader->forms);
        for (const auto& [cue, e] : reader->evidence) assert(counts.at(cue) == e.size());
        std::cerr << "seed=" << seed << " " << condition << " complete\n";
    };
    evaluate("initial", test);
    int baseline_index = 0;
    for (const auto& c : test) if (c.group == "known") {
        feed(baseline, prompt(c) + (++baseline_index % 2 ? " " : "")); auto& sc = scores["chain_baseline/known/" + std::to_string(c.family)];
        ++sc.n; sc.exact += generate(baseline) == c.gold;
    }
    auto full = reader->memory.procedures;
    reader->learn(true); evaluate("no_bindings", test, true); reader->memory.procedures = full;
    // Adapt only on new training worlds, then reevaluate the unchanged test set.
    for (const auto& c : adapt) for (const auto& f : c.facts)
        reader->teach_fact(f.text, f.subject, f.relation, f.object);
    evaluate("fact_adapted", test);
    for (const auto& c : adapt) { feed(org, prompt(c)); assert(reader->add_verified_answer(c.gold)); }
    const size_t adapted = reader->learn(); evaluate("fully_adapted", test);
    std::ofstream out(argv[4]); out << "{\"seed\":" << seed << ",\"initial_procedures\":" << learned
        << ",\"adapted_procedures\":" << adapted << ",\"fact_frames\":" << reader->frames.size() << ",\"scores\":{";
    bool first = true;
    for (const auto& [k, s] : scores) {
        if (!first) out << ','; first = false;
        out << '"' << k << "\":{\"n\":" << s.n << ",\"exact\":" << s.exact << ",\"extracted\":" << s.extracted
            << ",\"interpreted\":" << s.interpreted << ",\"covered\":" << s.covered << '}';
    }
    out << "}}\n";
    std::cout << "seed=" << seed << " learned=" << learned << " adapted=" << adapted << "\n";
}
