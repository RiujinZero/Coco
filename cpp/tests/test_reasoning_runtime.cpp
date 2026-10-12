#include "laminar/procedure_memory.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
int main() {
    ProcedureMemory memory(0);
    FlatQuery q; q.cue = 0; q.a = {0, 1}; q.b = {0, 2}; q.relation[0] = 0; q.relation[1] = 1;
    memory.procedures[0] = {9};
    assert(memory.answer(q) == q.a);
    auto weights = memory.skills.kernel.routes.W;
    // Identical evidence cannot justify different answers. Failure must revoke
    // the existing procedure rather than silently retain an invalid shortcut.
    auto impossible = memory.learn({{q, q.a}, {q, q.b}}, false, 4, 32);
    assert(!impossible.found && !memory.procedures.count(0));
    assert(!memory.answer(q).present());
    assert(memory.skills.kernel.routes.W == weights);
    bool rejected = false;
    FlatQuery other = q; other.cue = 1;
    try { memory.learn({{q, q.a}, {other, other.a}}); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    Runtime runtime(0);
    Goal invalid; invalid.root = 0;
    assert(runtime.run(invalid).invalid);
    std::cout << "reasoning contracts passed\n";
}
