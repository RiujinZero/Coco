#include "laminar/answer_grounding.h"
#include "laminar/reasoning_language.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
void feed(LanguageReasoner& r,const std::string& s,bool self=false) { for(uint8_t c:s) r.observe(c,self); }
int main() {
    ProcedureMemory memory(0); AnswerGrounding learner; EventGrounding events;
    FlatQuery q; q.a={0,0}; q.b={0,1}; q.relation[0]=0; q.relation[1]=1; q.facts={{2,1,1},{0,1,3}};
    std::vector<std::string> names{"lena","key","sam","book"}; auto identity=Meaning::var(0);
    std::string key;
    for(int i=0;i<4;++i) key=learner.teach("lena acquired key",names,q,identity,{0,0},memory,i,0);
    learner.publish(events); auto original=events.signature(); assert(events.frames.at(key).scenes.size()==4);
    assert(events.alternatives("lena acquired key").values.size()>1);
    auto signature=learner.signature();
    assert(learner.teach("lena acquired key",names,q,identity,{},memory,99,0).empty());
    assert(learner.teach("lena acquired key",names,q,identity,{0,9},memory,99,0,true).empty());
    assert(learner.signature()==signature);
    // Explicit correction replaces one observation and can restore candidates.
    learner.teach("lena acquired key",names,q,identity,{0,9},memory,0,0); learner.publish(events);
    assert(events.alternatives("lena acquired key").conflicted);
    learner.teach("lena acquired key",names,q,identity,{0,0},memory,0,0); learner.publish(events);
    assert(events.signature()==original && learner.signature()==signature);
    learner.teach("lena acquired key",names,q,identity,{0,9},memory,100,0); learner.publish(events);
    assert(events.alternatives("lena acquired key").conflicted);
    assert(learner.retract(key,100,0)); learner.publish(events); assert(events.signature()==original);
    // Identical answers in the surviving worlds are usable; intersection facts
    // are not sufficient to establish an arbitrary answer or a training target.
    LanguageReasoner r(0); r.grounded_events=r.uncertain_events=true;
    auto& f=r.events.frames["$0 acquired $1 "]; f.pattern={"$0","acquired","$1"}; f.scenes={0,1,2,3};
    f.candidates={{1,0,1,EventGrounding::ADD},{1,0,1,EventGrounding::SET_OBJECT}};
    r.forms["what is $0 "]=0; r.memory.procedures[0]={0,9};
    int a=r.entity("lena"),b=r.entity("key"),c=r.entity("sam"),d=r.entity("book");
    r.facts={{c,1,b},{c,1,d}};
    feed(r,"USER: lena acquired key. What is lena?\nASSISTANT:");
    assert(r.status=="grounded_answer" && r.reply==" lena\n" && r.possible_worlds.size()==2);
    assert(!r.add_verified_answer("lena"));
    feed(r,"\nUSER: lena acquired book. What is lena?\nASSISTANT:");
    assert(r.possible_worlds.size()==2); // shared meaning, not four independent choices
    auto worlds=r.possible_worlds; feed(r,"lena acquired key.",true);
    assert(r.possible_worlds.size()==worlds.size());
    auto candidate_before=r.events.frames.at("$0 acquired $1 ").candidates;
    r.events.frames.at("$0 acquired $1 ").candidates={{1,0,1,EventGrounding::ADD}};
    feed(r,"\nUSER: What is lena?\nASSISTANT:");
    assert(r.status=="stale_event_model" && r.reply==" I do not know.\n");
    r.events.frames.at("$0 acquired $1 ").candidates=candidate_before;
    r.world_budget=1; feed(r,"\n<|endofdialogue|>\nUSER: lena acquired key. What is lena?\nASSISTANT:");
    assert(r.status=="world_budget" && r.reply==" I do not know.\n" && r.uncertain_replies==2);
    std::cout<<"answer grounding contracts passed\n";
}
