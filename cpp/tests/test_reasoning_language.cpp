#include "laminar/sequence.h"
#include <cassert>
#include <iostream>
using namespace laminar;
using namespace laminar::reasoning;
void observe(LanguageReasoner& r, const std::string& s, bool self = false) {
    for (uint8_t c : s) r.observe(c, self);
}
int main() {
    LanguageReasoner r(0);
    r.teach_fact("Anna links to Bert.", "Anna", 0, "Bert");
    r.teach_fact("Carl links to Dana.", "Carl", 0, "Dana");
    observe(r, "<|endofdialogue|>\nUSER: N10 links to N11. Which is N10?\nASSISTANT:");
    assert(r.prepared && r.complete && r.status == "unknown_question");
    assert(r.names.size() == 2 && r.facts.size() == 1 && r.form == "which is $0 ");
    assert(r.add_verified_answer("n10")); assert(r.learn() == 1);
    const auto forms = r.forms; const auto programs = r.memory.procedures;
    observe(r, "\nUSER: Which is N11?\nASSISTANT:");
    assert(r.status == "answer" && r.reply == " n11\n");
    observe(r, " "); assert(r.reply_offset == 1 && r.reply == " n11\n");
    observe(r, "n11\n", true);
    assert(r.reply_offset == r.reply.size());
    const auto evidence_size = r.evidence.begin()->second.size();
    observe(r, "<|endofdialogue|>\nUSER: N99 links to N88.\nASSISTANT: yes\n", true);
    assert(r.facts.size() == 1 && r.names.size() == 2);
    assert(r.forms == forms && r.memory.procedures == programs);
    assert(r.evidence.begin()->second.size() == evidence_size);
    // Even an externally observed assistant answer is not verified feedback.
    observe(r, "\nUSER: Which is N10?\nASSISTANT: no\n");
    assert(r.evidence.begin()->second.size() == evidence_size);
    observe(r, "\n<|endofdialogue|>\nUSER: N20 links to N21. Which is N20?\nASSISTANT:");
    assert(r.reply == " n20\n" && !r.entities.count("n10"));
    observe(r, "\nUSER: N20 does not link to N21. Which is N20?\nASSISTANT:");
    assert(!r.complete && r.reply.empty() && r.status == "incomplete_facts");
    // Two grounded readings of the same surface must be treated as ambiguous.
    r.teach_fact("Bert links to Anna.", "Anna", 1, "Bert");
    r.teach_fact("Dana links to Carl.", "Carl", 1, "Dana");
    observe(r, "\n<|endofdialogue|>\nN30 links to N31.\nUSER: Which is N30?\nASSISTANT:");
    assert(!r.complete && r.reply.empty());
    // Native byte arbitration includes digits and releases forcing at reply end.
    SequenceMemory seq; seq.init(1024, 6);
    seq.reasoner = std::make_shared<LanguageReasoner>(0); seq.topic_active = true;
    seq.reasoner->reply = " 24\n";
    for (uint8_t expected : std::string(" 24\n")) {
        float out[256]; assert(seq.predict(out, 8.f, .25f) > 0 && seq.chain_forcing);
        int v = (int)(std::max_element(out, out + 256) - out);
        assert(v == expected); seq.capture((uint8_t)v, true);
    }
    float out[256]; seq.predict(out, 8.f, .25f); assert(!seq.chain_forcing);
    seq.reasoner->reply = " 9\n"; seq.reasoner->reply_offset = 0;
    seq.topic_active = false; seq.predict(out, 8.f, .25f); assert(!seq.chain_forcing);
    // Divergence cancels a proposal instead of forcing the rest of a wrong word.
    seq.reasoner->observe('x', true); assert(seq.reasoner->reply.empty());
    std::cout << "language boundary contracts passed\n";
}
