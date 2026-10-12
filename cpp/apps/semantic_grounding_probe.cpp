#include "laminar/organism.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
using namespace laminar;
using namespace laminar::reasoning;
struct TextFact { std::string subject, object, text; int relation; };
struct Case { std::string group, gold, expected, question; std::vector<TextFact> facts; };
std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out; std::istringstream in(s); std::string v;
    while (std::getline(in, v, '\t')) out.push_back(v); return out;
}
std::vector<Case> read_cases(const std::string& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("missing " + path);
    std::vector<Case> out; std::string line;
    while (std::getline(in, line)) {
        auto h = split(line); assert(h.size() == 4);
        Case c; c.group = h[0]; c.gold = h[1]; c.expected = h[2]; std::getline(in, c.question);
        for (int i = 0; i < std::stoi(h[3]); ++i) {
            std::getline(in, line); auto f = split(line); assert(f.size() == 4);
            c.facts.push_back({f[0], f[2], f[3], std::stoi(f[1])});
        }
        out.push_back(std::move(c));
    }
    return out;
}
std::string prompt(const Case& c) {
    std::string s = "\n<|endofdialogue|>\nUSER: ";
    for (const auto& f : c.facts) s += f.text + "\n";
    return s + c.question + "\n\nASSISTANT:";
}
void feed(Organism& org, const std::string& text) {
    org.free_run = false; org.run_block((const uint8_t*)text.data(), text.size());
}
std::string generate(Organism& org) {
    org.free_run = true; std::string out;
    for (int i = 0; i < 96; ++i) {
        uint8_t pair[2] = {org.last_symbol(), 0}; org.run_block(pair, 2);
        const auto& p = org.distribution(); int v = (int)(std::max_element(p.begin(), p.end()) - p.begin());
        org.commit_outcome((uint8_t)v); out += (char)v; if (v == '\n') break;
    }
    return LanguageReasoner::trim(out);
}
bool extraction_correct(const LanguageReasoner& r, const Case& c) {
    using Key = std::tuple<std::string, int, std::string>;
    std::multiset<Key> want, got;
    for (const auto& f : c.facts) want.emplace(f.subject, f.relation, f.object);
    for (const auto& f : r.facts) got.emplace(r.names.at(f.subject), f.relation, r.names.at(f.object));
    return r.complete && want == got;
}
struct Score { int n = 0, exact = 0, meaning = 0, extracted = 0, covered = 0, known_form = 0, ambiguous = 0, exhausted = 0; size_t max_states = 0; };
int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "usage: semantic_probe skill_corpus text_corpus seed output\n"; return 2; }
    int seed = std::stoi(argv[3]); auto reader = std::make_shared<LanguageReasoner>(seed); reader->shared_meanings = true;
    std::string tail = "_" + std::to_string(seed) + ".txt", root = std::string(argv[2]) + "/";
    auto training = read_cases(root + "train" + tail), test = read_cases(root + "test" + tail), adapt = read_cases(root + "adapt" + tail);
    { std::ifstream in(std::string(argv[1]) + "/micro" + tail); if (!in) throw std::runtime_error("missing skills");
      std::vector<Sample> samples; Sample s; int n;
      while (in >> s.action >> n) { s.available.resize(n); for (int& a : s.available) in >> a; for (float& x : s.obs) in >> x; assert(in); samples.push_back(s); }
      std::mt19937 rng(seed + 400);
      for (int pass = 0; pass < 5; ++pass) { std::shuffle(samples.begin(), samples.end(), rng); for (const auto& x : samples) reader->memory.skills.fit(x); } }
    Params p; p.seed = seed; p.units = 128; p.vocab = 256; p.train = false; p.structural = false;
    p.seq_window = 65536; p.seq_hash_bits = 10; p.seq_consolidate = false;
    Organism org(p); org.attach_reasoner(reader);
    for (const auto& c : training) for (const auto& f : c.facts) reader->teach_fact(f.text, f.subject, f.relation, f.object);
    for (const auto& c : training) { feed(org, prompt(c)); assert(reader->add_verified_answer(c.gold)); }
    size_t initial = reader->learn(), initial_rules = reader->grounding.rules.size();
    std::map<std::string, Score> scores;
    std::ofstream trace(std::string(argv[4]) + ".examples.txt");
    auto evaluate = [&](const std::string& condition) {
        const auto weights = reader->memory.skills.kernel.routes.W;
        const auto programs = reader->memory.procedures;
        const auto forms = reader->forms;
        const auto signature = reader->grounding.signature();
        std::map<int, size_t> sizes; for (const auto& [cue, e] : reader->evidence) sizes[cue] = e.size();
        int index = 0;
        for (const auto& c : test) {
            feed(org, prompt(c) + (++index % 2 ? " " : ""));
            auto& sc = scores[condition + "/" + c.group]; ++sc.n;
            sc.extracted += extraction_correct(*reader, c);
            sc.known_form += reader->forms.count(reader->form);
            sc.ambiguous += reader->interpretation.ambiguous; sc.exhausted += reader->interpretation.exhausted;
            sc.max_states = std::max(sc.max_states, reader->interpretation.states);
            const bool covered = reader->status == "answer" || reader->status == "grounded_answer";
            sc.covered += covered;
            sc.meaning += reader->interpretation.found && reader->interpretation.meaning.key() == c.expected;
            // Labels and meaning annotations are only compared AFTER interpretation.
            const auto reply = covered ? generate(org) : "<module declined>";
            sc.exact += covered && reply == c.gold;
            if (sc.n <= 2 || (sc.n <= 5 && covered && reply != c.gold)) {
                trace << condition << '/' << c.group << '\n' << c.question << "\nreply=" << reply << "\ngold=" << c.gold
                    << "\nstatus=" << reader->status << "\nmeaning=" << reader->interpretation.meaning.key()
                    << "\nexpected=" << c.expected << "\nstates=" << reader->interpretation.states << "\n\n";
            }
        }
        assert(weights == reader->memory.skills.kernel.routes.W && programs == reader->memory.procedures && forms == reader->forms);
        assert(signature == reader->grounding.signature());
        for (const auto& [cue, e] : reader->evidence) assert(sizes.at(cue) == e.size());
        std::cerr << "seed=" << seed << ' ' << condition << " complete\n";
    };
    evaluate("shared");
    reader->phrase_fragments = false; evaluate("no_fragments"); reader->phrase_fragments = true;
    reader->shared_meanings = false; evaluate("whole_question_lookup"); reader->shared_meanings = true;
    // New vocabulary is taught through ATOMIC answers, not annotated meanings,
    // and no composite question from the test set is ever a training example.
    for (const auto& c : adapt) { feed(org, prompt(c)); assert(reader->add_verified_answer(c.gold)); }
    size_t adapted = reader->learn(); evaluate("atomic_alias_teaching");
    std::ofstream out(argv[4]); out << "{\"seed\":" << seed << ",\"initial_procedures\":" << initial << ",\"initial_rules\":" << initial_rules
        << ",\"adapted_procedures\":" << adapted << ",\"adapted_rules\":" << reader->grounding.rules.size() << ",\"scores\":{";
    bool first = true;
    for (const auto& [name, sc] : scores) {
        if (!first) out << ','; first = false;
        out << '"' << name << "\":{\"n\":" << sc.n << ",\"exact\":" << sc.exact << ",\"meaning\":" << sc.meaning << ",\"extracted\":" << sc.extracted
            << ",\"covered\":" << sc.covered << ",\"known_form\":" << sc.known_form << ",\"ambiguous\":" << sc.ambiguous
            << ",\"exhausted\":" << sc.exhausted << ",\"max_states\":" << sc.max_states << '}';
    }
    out << "}}\n";
    std::cout << "seed=" << seed << " initial=" << initial << " rules=" << initial_rules << " adapted=" << adapted << '\n';
}
