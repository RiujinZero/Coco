#include "../apps/grounding_probe_support.h"
using namespace grounding_probe;
void text(LanguageReasoner& r,const std::string& s,bool self=false) { for(uint8_t c:s)r.observe(c,self); }
void query(LanguageReasoner& r,const std::string& s) { text(r,"\nUSER: "+s+"?\nASSISTANT:"); }

int main() {
    FactKnowledge k;
    assert(k.truth(0,1,2)==FactKnowledge::UNKNOWN);
    assert(k.observe(0,1,2,false));assert(k.truth(0,1,2)==FactKnowledge::FALSE_VALUE);
    assert(!k.observe(0,1,2,true));assert(k.positive.empty());
    assert(k.observe_row(3,1,{}));assert(k.truth(3,1,999)==FactKnowledge::FALSE_VALUE);
    k.update(0,1,2,EventGrounding::ADD);assert(k.truth(0,1,2)==FactKnowledge::TRUE_VALUE);
    assert(k.truth(0,1,5)==FactKnowledge::UNKNOWN);
    k.update(0,1,4,EventGrounding::SET_SUBJECT);
    assert(k.truth(0,1,2)==FactKnowledge::FALSE_VALUE&&k.truth(0,1,4)==FactKnowledge::TRUE_VALUE);
    k.update(5,1,4,EventGrounding::SET_OBJECT);
    assert(k.truth(0,1,4)==FactKnowledge::FALSE_VALUE&&k.truth(5,1,4)==FactKnowledge::TRUE_VALUE);
    k.forget_column(1,4);assert(k.truth(0,1,4)==FactKnowledge::UNKNOWN);
    assert(k.closed_rows.empty()); // unknown changes invalidate intersecting coverage
    k.update(3,1,6,EventGrounding::SET_OBJECT);k.forget_row(3,1);assert(k.closed_columns.empty());

    // Discover a membership procedure from final booleans with distractors on
    // the other relation. No procedure or question recipe is a teaching input.
    ProcedureMemory memory(0);memory.lookup_enabled=true;
    std::vector<VerifiedQuestion> examples;
    for(int i=0;i<16;++i) {
        FlatQuery q;q.cue=10;q.a={0,i};q.b={0,i+100};q.relation[0]=0;q.relation[1]=1;
        if(i%2)q.facts.push_back({i,1,i+100});
        if(i%4<2)q.facts.push_back({i,0,i+100});
        q.facts.push_back({i+100,1,i});q.facts.push_back({i,1,i});
        examples.push_back({q,{1,i%2}});
    }
    auto learned=memory.learn(examples);assert(learned.found);
    assert(Meaning::from_program(learned.program,2).key()=="11($0,$1,)");
    FlatQuery q=examples.front().query;auto observation=std::make_shared<FactKnowledge>();q.knowledge=observation;
    assert(!memory.answer(q).present()&&memory.missing_evidence);
    observation->observe(0,1,100,false);assert((memory.answer(q)==Value{1,0}));
    observation->update(0,1,100,EventGrounding::ADD);assert((memory.answer(q)==Value{1,1}));
    memory.missing_evidence=true;q.cue=-1;assert(!memory.answer(q).present()&&!memory.missing_evidence);

    LanguageReasoner r(0);train_skills(r,0);
    r.partial_observation=true;r.grounded_events=true;
    // Fixture forms isolate the memory contract; membership learning above is
    // separately checked. These fixtures are not curriculum training results.
    r.forms["does $0 hold $1 "]=0;r.memory.procedures[0]={11,9};
    r.forms["how many for $0 "]=1;r.memory.procedures[1]={7,9};
    r.forms["follow $0 "]=2;r.memory.procedures[2]={4,9};
    r.entity("lena");r.entity("book");
    query(r,"Does lena hold book");assert(r.status=="missing_observation"&&r.reply==" I do not know.\n");
    assert(!r.add_verified_answer("no"));
    assert(r.observe_fact("lena",1,"book",false));query(r,"Does lena hold book");assert(r.reply==" no\n");
    query(r,"How many for lena");assert(r.status=="missing_observation");
    assert(r.observe_complete_row("lena",1,{}));query(r,"How many for lena");assert(r.reply==" 0\n");
    auto known=r.observations;assert(!r.observe_fact("lena",1,"coin",true,true));
    assert(!r.entities.count("coin")&&!(known<r.observations)&&!(r.observations<known));
    assert(r.lose_visibility("lena",1));query(r,"Does lena hold book");assert(r.status=="missing_observation");
    assert(r.observe_fact("lena",1,"book",true));query(r,"Does lena hold book");assert(r.reply==" yes\n");
    query(r,"How many for lena");assert(r.status=="missing_observation");
    assert(r.observe_complete_row("lena",1,{"book","coin"}));query(r,"How many for lena");assert(r.reply==" 2\n");
    assert(r.observe_complete_row("lena",0,{"sam"}));query(r,"Follow lena");assert(r.status=="missing_observation");
    assert(r.observe_complete_row("sam",0,{}));query(r,"Follow lena");assert(r.reply==" sam\n");
    assert(!r.observe_fact("lena",1,"book",false));query(r,"Does lena hold book");assert(r.status=="conflicting_observation");
    r.reset_episode();assert(r.observations.positive.empty()&&r.observations.negative.empty()&&r.observations.closed_rows.empty());

    // Projection of two complete but different rows is not a complete empty row.
    int a=r.entity("lena"),b=r.entity("book"),c=r.entity("coin");
    LanguageReasoner::PossibleWorld w1,w2;
    w1.knowledge.observe_row(a,1,{b});w1.facts=w1.knowledge.positive;
    w2.knowledge.observe_row(a,1,{c});w2.facts=w2.knowledge.positive;
    r.possible_worlds={w1,w2};r.project_worlds();assert(r.observations.closed_rows.empty()&&r.facts.empty());
    query(r,"How many for lena");assert(r.reply==" 1\n");
    query(r,"Does lena hold book");assert(r.status=="uncertain_answer");
    assert(r.observe_fact("lena",1,"book",true));assert(r.possible_worlds.size()==1);
    query(r,"Does lena hold book");assert(r.reply==" yes\n");
    assert(!r.observe_fact("lena",1,"coin",true));assert(r.event_issue=="conflicting_observation");

    // Aligned positive observations obey the same open-world default.
    r.reset_episode();r.grounded_events=false;
    r.teach_fact("lena owns book","lena",1,"book");r.teach_fact("sam owns coin","sam",1,"coin");
    text(r,"USER: lena owns book. Does lena hold book?\nASSISTANT:");assert(r.reply==" yes\n");
    query(r,"How many for lena");assert(r.status=="missing_observation");
    std::cout<<"partial observation contracts passed\n";
}
