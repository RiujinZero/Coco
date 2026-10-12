#include "laminar/composed_learner.h"
#include <iostream>
using namespace laminar::reasoning;
void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
int main() {
    // A literal-return fixture keeps lifecycle checks independent of costly
    // primitive search; the separate acquisition experiment exercises learning.
    ComposedLearner learner;ComposedReceipt r;r.id="literal";r.question.query.cue=9;
    r.question.query.a={2,7};r.question.query.b={0,4};r.question.query.relation[0]=0;r.question.query.relation[1]=1;r.question.answer={2,7};
    auto bad=r;bad.generated=true;check(!learner.teach(bad),"generated teaching accepted");
    bad=r;bad.question.query.knowledge=std::make_shared<FactKnowledge>();check(!learner.teach(bad),"partial-knowledge teaching accepted");
    check(learner.teach(r)&&!learner.teach(r)&&learner.answer(r.question.query)==Value{},"receipt/dirty contract failed");
    check(learner.rebuild(1)&&learner.solved()&&learner.answer(r.question.query)==r.question.answer,"consistent compilation failed");
    auto clone=learner;bad=r;bad.id="counterexample";bad.question.answer={2,6};
    check(clone.teach(bad)&&!clone.rebuild(1)&&!clone.solved()&&!clone.rebuild(1),"contradiction or repeated build became success");
    check(clone.answer(r.question.query)==Value{}&&learner.answer(r.question.query)==r.question.answer,"contradiction leaked across copies");
    auto version=clone.version();check(!clone.revise("counterexample",0,{2,7})&&version==clone.version(),"stale correction mutated model");
    check(clone.revise("counterexample",1,{2,6},false)&&clone.rebuild(1)&&clone.answer(r.question.query)==r.question.answer,"withdrawal did not restore usable knowledge");
    check(clone.revise("literal",1,{2,7},false)&&clone.rebuild(1)&&clone.memory().procedures.empty(),"empty evidence retained a program");
    std::cout<<"composed receipt validation, contradictions, withdrawal, stale versions and clone isolation passed\n";
}
