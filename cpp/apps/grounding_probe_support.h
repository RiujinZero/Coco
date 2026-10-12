#pragma once
#include "laminar/organism.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
namespace grounding_probe {
using namespace laminar;
using namespace laminar::reasoning;
using World = EventGrounding::World;
inline std::vector<std::string> fields(const std::string& s) {
    std::vector<std::string> out; std::istringstream in(s); std::string v;
    while (std::getline(in, v, '\t')) out.push_back(v); return out;
}
inline std::string line(std::istream& in) { std::string s; std::getline(in, s); if (!s.empty() && s.back() == '\r') s.pop_back(); return s; }
inline World read_world(std::istream& in) {
    int n = std::stoi(line(in)); World out;
    for (int i = 0; i < n; ++i) { auto f = fields(line(in)); assert(f.size() == 3); out.emplace(f[0], std::stoi(f[1]), f[2]); }
    return out;
}
inline void feed(Organism& org, const std::string& text) { org.free_run = false; org.run_block((const uint8_t*)text.data(), text.size()); }
inline std::string generate(Organism& org) {
    org.free_run = true; std::string out;
    for (int i = 0; i < 1024; ++i) {
        uint8_t pair[2] = {org.last_symbol(), 0}; org.run_block(pair, 2);
        const auto& p = org.distribution(); int v = (int)(std::max_element(p.begin(), p.end()) - p.begin());
        org.commit_outcome((uint8_t)v); out += (char)v; if (v == '\n') break;
    }
    return LanguageReasoner::trim(out);
}
inline World state(const LanguageReasoner& r) {
    World w; for (const auto& f : r.facts) w.emplace(r.names.at(f.subject), f.relation, r.names.at(f.object)); return w;
}
inline void train_skills(LanguageReasoner& reader, int seed) {
    std::ifstream in("logs/reasoning_composition_verified/micro_" + std::to_string(seed) + ".txt");
    if (!in) throw std::runtime_error("missing primitive demonstrations");
    std::vector<Sample> samples; Sample s; int n;
    while (in >> s.action >> n) { s.available.resize(n); for (int& a : s.available) in >> a; for (float& x : s.obs) in >> x; assert(in); samples.push_back(s); }
    std::mt19937 rng(seed + 400);
    for (int pass = 0; pass < 5; ++pass) { std::shuffle(samples.begin(), samples.end(), rng); for (const auto& x : samples) reader.memory.skills.fit(x); }
}
inline void train_events(LanguageReasoner& reader, int seed) {
    std::ifstream in("logs/event_grounding_corpus/sensors_" + std::to_string(seed) + ".txt");
    if (!in) throw std::runtime_error("missing event observations");
    while (in.peek() != EOF) {
        auto h = fields(line(in)); auto text = line(in); auto before = read_world(in), after = read_world(in);
        if (h[0] != "repeat") assert(reader.events.teach_transition(text,before,after,std::stoull(h[1])));
    }
}
inline size_t train_questions(Organism& org, LanguageReasoner& reader, int seed) {
    std::ifstream in("logs/semantic_grounding_corpus/train_" + std::to_string(seed) + ".txt");
    if (!in) throw std::runtime_error("missing atomic questions");
    while (in.peek() != EOF) {
        auto h = fields(line(in)); std::string question = line(in), text;
        for (int i = 0; i < std::stoi(h[3]); ++i) text += fields(line(in)).at(3) + "\n";
        feed(org,"\n<|endofdialogue|>\nUSER: " + text + question + "\nASSISTANT:");
        assert(reader.add_verified_answer(h[1]));
    }
    return reader.learn();
}
} // namespace grounding_probe
