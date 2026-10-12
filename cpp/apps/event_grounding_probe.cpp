#include "laminar/organism.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
using namespace laminar;
using namespace laminar::reasoning;
using World = EventGrounding::World;
std::vector<std::string> fields(const std::string& s) {
    std::vector<std::string> out; std::istringstream in(s); std::string v;
    while (std::getline(in, v, '\t')) out.push_back(v); return out;
}
std::string line(std::istream& in) { std::string s; std::getline(in, s); if (!s.empty() && s.back() == '\r') s.pop_back(); return s; }
World read_world(std::istream& in) {
    int n = std::stoi(line(in)); World out;
    for (int i = 0; i < n; ++i) { auto f = fields(line(in)); assert(f.size() == 3); out.emplace(f[0], std::stoi(f[1]), f[2]); }
    return out;
}
struct WorldObservation { std::string phase, text; uint64_t scene; World before, after; };
std::vector<WorldObservation> observations(const std::string& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("missing " + path);
    std::vector<WorldObservation> out;
    while (in.peek() != EOF) {
        auto h = fields(line(in)); assert(h.size() == 2); WorldObservation o;
        o.phase = h[0]; o.scene = std::stoull(h[1]); o.text = line(in); o.before = read_world(in); o.after = read_world(in); out.push_back(o);
    }
    return out;
}
struct Turn { std::string text, answer; World state; };
struct Case { std::string group, question, initial_text, initial_answer; World initial; std::vector<Turn> turns; };
std::vector<Case> cases(const std::string& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("missing " + path);
    std::vector<Case> out;
    while (in.peek() != EOF) {
        auto h = fields(line(in)); assert(h.size() == 3); Case c; c.group = h[0]; c.question = line(in);
        for (int i = 0; i < std::stoi(h[1]); ++i) {
            auto f = fields(line(in)); assert(f.size() == 4); c.initial.emplace(f[0], std::stoi(f[1]), f[2]); c.initial_text += f[3] + "\n";
        }
        c.initial_answer = line(in);
        for (int i = 0; i < std::stoi(h[2]); ++i) {
            // Change annotations are only read by the external audit; inference
            // gets the text. The final state is used by this scorer after reading.
            auto f = fields(line(in)); assert(f.size() == 5); Turn t; t.text = f[4]; t.answer = line(in); t.state = read_world(in); c.turns.push_back(t);
        }
        out.push_back(c);
    }
    return out;
}
void feed(Organism& org, const std::string& text) { org.free_run = false; org.run_block((const uint8_t*)text.data(), text.size()); }
std::string generate(Organism& org) {
    org.free_run = true; std::string out;
    for (int i = 0; i < 96; ++i) {
        uint8_t pair[2] = {org.last_symbol(), 0}; org.run_block(pair, 2);
        const auto& p = org.distribution(); int v = (int)(std::max_element(p.begin(), p.end()) - p.begin());
        org.commit_outcome((uint8_t)v); out += (char)v; if (v == '\n') break;
    }
    return LanguageReasoner::trim(out);
}
World state(const LanguageReasoner& r) {
    World w; for (const auto& f : r.facts) w.emplace(r.names.at(f.subject), f.relation, r.names.at(f.object)); return w;
}
struct Score { int n = 0, exact = 0, state = 0, covered = 0, declined = 0; };
int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "usage: event_probe skills atomic_questions sensors seed output\n"; return 2; }
    int seed = std::stoi(argv[4]); const std::string tail = "_" + std::to_string(seed) + ".txt";
    auto reader = std::make_shared<LanguageReasoner>(seed); reader->grounded_events = true; reader->shared_meanings = true;
    { std::ifstream in(std::string(argv[1]) + "/micro" + tail); if (!in) throw std::runtime_error("missing skills");
      std::vector<Sample> samples; Sample s; int n;
      while (in >> s.action >> n) { s.available.resize(n); for (int& a : s.available) in >> a; for (float& x : s.obs) in >> x; assert(in); samples.push_back(s); }
      std::mt19937 rng(seed + 400);
      for (int pass = 0; pass < 5; ++pass) { std::shuffle(samples.begin(), samples.end(), rng); for (const auto& x : samples) reader->memory.skills.fit(x); } }
    auto sensor = observations(std::string(argv[3]) + "/sensors" + tail);
    for (const auto& o : sensor) if (o.phase == "weak") assert(reader->events.teach_transition(o.text, o.before, o.after, o.scene));
    EventGrounding weak = reader->events;
    // Match total exposure: another 144 simple scenes versus 144 contrasting
    // scenes, both starting from the same first 144 observations.
    for (const auto& o : sensor) if (o.phase == "repeat") assert(weak.teach_transition(o.text, o.before, o.after, o.scene));
    size_t weak_grounded = weak.grounded();
    for (const auto& o : sensor) if (o.phase == "contrast") assert(reader->events.teach_transition(o.text, o.before, o.after, o.scene));
    EventGrounding grounded = reader->events;
    Params p; p.seed = seed; p.units = 128; p.vocab = 256; p.train = false; p.structural = false;
    p.seq_window = 65536; p.seq_hash_bits = 10; p.seq_consolidate = false;
    Organism org(p); org.attach_reasoner(reader);
    // Train only atomic question meanings. Fact roles and update meanings come
    // entirely from the sensory transition learner, not these file annotations.
    { std::ifstream in(std::string(argv[2]) + "/train" + tail); if (!in) throw std::runtime_error("missing questions");
      while (in.peek() != EOF) {
          auto h = fields(line(in)); assert(h.size() == 4); std::string question = line(in), text;
          for (int i = 0; i < std::stoi(h[3]); ++i) { auto f = fields(line(in)); text += f.at(3) + "\n"; }
          feed(org, "\n<|endofdialogue|>\nUSER: " + text + question + "\nASSISTANT:");
          assert(reader->add_verified_answer(h[1]));
      } }
    const auto procedures = reader->learn();
    auto test = cases(std::string(argv[3]) + "/test" + tail);
    std::map<std::string, Score> scores; std::ofstream trace(std::string(argv[5]) + ".examples.txt");
    auto evaluate = [&](const std::string& condition) {
        const auto weights = reader->memory.skills.kernel.routes.W;
        const auto programs = reader->memory.procedures; const auto forms = reader->forms;
        const auto grammar = reader->grounding.signature(), events = reader->events.signature();
        std::map<int, size_t> counts; for (const auto& [cue, e] : reader->evidence) counts[cue] = e.size();
        for (const auto& c : test) {
            auto score = [&](const std::string& phase, const std::string& gold, const World& wanted) {
                auto& sc = scores[condition + "/" + c.group + "/" + phase]; ++sc.n;
                const bool covered = reader->status == "answer" || reader->status == "grounded_answer";
                sc.covered += covered; sc.declined += !covered; sc.state += reader->complete && state(*reader) == wanted;
                const auto reply = covered ? generate(org) : "<module declined>";
                sc.exact += covered && reply == gold;
                if (sc.n <= 2) trace << condition << '/' << c.group << '/' << phase << '\n' << c.question << "\nreply=" << reply << "\ngold=" << gold << "\nstatus=" << reader->status << "\n\n";
            };
            feed(org, "\n<|endofdialogue|>\nUSER: " + c.initial_text + c.question + "\nASSISTANT:");
            score("before", c.initial_answer, c.initial);
            for (const auto& t : c.turns) {
                feed(org, "\nUSER: " + t.text + "\n" + c.question + "\nASSISTANT: ");
                score("after", t.answer, t.state);
            }
        }
        assert(weights == reader->memory.skills.kernel.routes.W && programs == reader->memory.procedures && forms == reader->forms);
        assert(grammar == reader->grounding.signature() && events == reader->events.signature());
        for (const auto& [cue, e] : reader->evidence) assert(counts.at(cue) == e.size());
        assert(reader->frames.empty()); // no aligned sentence training was used
        std::cerr << "seed=" << seed << ' ' << condition << " complete\n";
    };
    evaluate("contrast_grounded");
    reader->events = weak; evaluate("weak_scenes"); reader->events = grounded;
    for (auto& [key, f] : reader->events.frames) {
        std::set<EventGrounding::Rule> append_only;
        for (auto r : f.candidates) { r.effect = EventGrounding::ADD; append_only.insert(r); }
        f.candidates = append_only;
    }
    evaluate("append_only"); reader->events = grounded;
    std::ofstream out(argv[5]); out << "{\"seed\":" << seed << ",\"weak_frames\":" << weak_grounded << ",\"grounded_frames\":" << grounded.grounded()
        << ",\"question_procedures\":" << procedures << ",\"scores\":{";
    bool first = true;
    for (const auto& [name, sc] : scores) {
        if (!first) out << ','; first = false;
        out << '"' << name << "\":{\"n\":" << sc.n << ",\"exact\":" << sc.exact << ",\"state\":" << sc.state << ",\"covered\":" << sc.covered << ",\"declined\":" << sc.declined << '}';
    }
    out << "}}\n"; std::cout << "seed=" << seed << " grounded=" << grounded.grounded() << " atomic procedures=" << procedures << '\n';
}
