#include "laminar/semantic_grounding.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
int main() {
    SemanticGrounding g;
    auto x = Meaning::var(0), y = Meaning::var(1);
    auto follow = Meaning::call(4, {x}), count = Meaning::call(7, {x});
    g.fit({{"what is $0", x}, {"give me $0", x}, {"tell me $0", x},
           {"what is endpoint of $0", follow}, {"give me endpoint of $0", follow},
           {"what is item count of $0", count}, {"give me item count of $0", count},
           {"are $0 and $1 equal", Meaning::call(8, {x, y})}});
    auto unseen = g.parse("tell me endpoint of $0");
    assert(unseen.found && unseen.meaning.key() == follow.key());
    assert(!g.parse("tell me endpoint of $0", false).found);
    auto composed = g.parse("are item count of endpoint of $0 and item count of $1 equal");
    auto wanted = Meaning::call(8, {Meaning::call(7, {follow}), Meaning::call(7, {y})});
    if (!composed.found) std::cerr << "parse states=" << composed.states << " ambiguous=" << composed.ambiguous << " exhausted=" << composed.exhausted << '\n';
    assert(composed.found && composed.meaning.key() == wanted.key());
    assert(!g.parse("are item count of endpoint of $0 and item count of $1 equal", false).found);
    assert(!g.parse("tell me not endpoint of $0").found);
    assert(!g.parse("tell me endpoint of item count of $0").found); // wrong argument type
    assert(g.parse("tell me endpoint of $0", true, 1).exhausted);
    auto fingerprint = g.signature(); g.parse("tell me endpoint of $1"); assert(g.signature() == fingerprint);
    g.rules.push_back({{"tell", "me", "$0"}, Meaning::call(7, {x}), false});
    auto ambiguous = g.parse("tell me endpoint of $0");
    assert(ambiguous.ambiguous && !ambiguous.found);
    assert(Meaning::from_program({4, 7, 3, 7, 8, 9}, 2).key() == wanted.key());
    std::cout << "semantic grounding contracts passed; rules=" << g.rules.size() << '\n';
}
