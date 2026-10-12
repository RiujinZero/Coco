#include "laminar/reasoning_language.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
void input(LanguageReasoner& r, const std::string& s, bool self = false) { for (uint8_t c : s) r.observe(c, self); }
int main() {
    EventGrounding g;
    for (int i = 0; i < 4; ++i) {
        auto a = "a" + std::to_string(i), b = "b" + std::to_string(i);
        g.teach_transition(b + " was taken by " + a, {}, {{a, 1, b}}, i);
    }
    EventGrounding::Interpretation meaning;
    assert(g.grounded() == 0 && !g.interpret("key was taken by lena", meaning));
    for (int i = 0; i < 4; ++i) {
        auto a = "a" + std::to_string(i), b = "b" + std::to_string(i);
        EventGrounding::World before{{a, 1, "other"}, {"old", 1, b}}, after{{a, 1, "other"}, {a, 1, b}};
        g.teach_transition(b + " was taken by " + a, before, after, 10 + i);
    }
    assert(g.grounded() == 1 && g.interpret("key was taken by lena", meaning));
    assert(meaning.subject == "lena" && meaning.object == "key" && meaning.relation == 1 && meaning.effect == EventGrounding::SET_OBJECT);
    auto sig = g.signature();
    assert(!g.teach_transition("key was taken by lena", {}, {{"lena", 1, "key"}}, 99, true));
    assert(g.signature() == sig);
    LanguageReasoner r(0); r.grounded_events = true; r.events = g;
    input(r, "<|endofdialogue|>\nUSER: key was taken by lena. key was taken by sam.");
    assert(r.complete && r.facts.size() == 1 && r.facts[0].subject == r.entities.at("sam"));
    input(r, "key was taken by lena.", true);
    assert(r.facts[0].subject == r.entities.at("sam") && r.events.signature() == sig);
    input(r, "\n<|endofdialogue|>\nUSER: key was taken by dana.");
    assert(r.facts.size() == 1 && !r.entities.count("sam"));
    input(r, "key was mysteriously taken by sam."); assert(!r.complete);
    // Contradictory verified experience revokes an old interpretation.
    g.teach_transition("key was taken by lena", {{"old", 1, "key"}}, {{"old", 1, "key"}, {"lena", 1, "key"}}, 100);
    assert(!g.interpret("key was taken by lena", meaning));
    std::cout << "event grounding contracts passed\n";
}
