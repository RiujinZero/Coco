#include "../apps/grounding_probe_support.h"
#include "laminar/observation_language_learner.h"
using namespace grounding_probe;
void input(LanguageReasoner& r,const std::string& text,bool self=false){for(uint8_t c:text)r.observe(c,self);}
LanguageReasoner fixture() {
    LanguageReasoner r(0);r.grounded_events=r.uncertain_events=r.environment_dialogue=true;
    r.forms["does $0 have the $1 "]=0;r.memory.procedures[0]={3,11,9};r.memory.lookup_enabled=true;
    for(int i=0;i<8;++i) {
        auto a="actor"+std::to_string(i),o="object"+std::to_string(i);
        World before{{o,1,"other"},{"spare",1,a},{a,0,"oldroom"}},after=before;
        after=EventGrounding::update(before,1,2,o,a);assert(r.events.teach_transition(a+" has the "+o+".",before,after,i));
        after=EventGrounding::update(before,0,2,a,std::string("room"));assert(r.events.teach_transition(a+" moved to the room.",before,after,100+i));
        before=EventGrounding::update(before,1,2,o,a);after=EventGrounding::update(before,1,2,o,std::string("recipient"));
        assert(r.events.teach_transition(a+" passed the "+o+" to recipient.",before,after,200+i));
    }
    return r;
}
std::vector<BootstrapGrounding::Episode> lessons() {
    std::vector<BootstrapGrounding::Episode> out;
    for(int i=0;i<8;++i) {
        auto a="actor"+std::to_string(i),o="object"+std::to_string(i),other="other"+std::to_string(i),spare="spare"+std::to_string(i);
        std::vector<std::string> story{a+" has the "+o+".",other+" has the "+spare+".",a+" moved to the room."};
        out.push_back({story,"Does "+a+" have the "+o+"?","yes",(uint64_t)i});
        out.push_back({story,"Does "+other+" have the "+o+"?","no",(uint64_t)i});
        story.push_back("The "+o+" was out of sight.");
        out.push_back({story,"Does "+a+" have the "+o+"?","unknown",(uint64_t)i});
        out.push_back({story,"Does "+other+" have the "+spare+"?","yes",(uint64_t)i});
    }
    return out;
}
int main() {
    ObservationLanguage lexicon;
    lexicon.frames["the $0 list is $1 "]={{"the","$0","list","is","$1"},{{3,1,0,2}},{1,2,3,4}};
    lexicon.frames["the $0 list is empty "]={{"the","$0","list","is","empty"},{{3,1,0,0}},{1,2,3,4}};
    assert(lexicon.alternatives("The key list is empty.").values.size()==1);
    assert(lexicon.alternatives("The key list is empty.",{"empty"}).values.size()==2);
    FactKnowledge k;assert(k.observe_column(1,4,{2,3}));assert(k.truth(2,1,4)==FactKnowledge::TRUE_VALUE&&k.truth(8,1,4)==FactKnowledge::FALSE_VALUE);
    auto old=k;assert(!k.observe_column(1,4,{2}));assert(!(k<old)&&!(old<k));k.forget_column(1,4);assert(k.truth(2,1,4)==FactKnowledge::UNKNOWN);
    ProcedureMemory memory(0);FlatQuery q;q.a={0,2};q.relation[0]=0;q.relation[1]=1;
    auto observed=std::make_shared<FactKnowledge>();q.knowledge=observed;
    q.facts={{2,1,3},{3,1,4}};observed->positive={{2,1,3},{3,1,4}};
    assert(!memory.execute(q,{17,9}).present()&&memory.missing_evidence);
    observed->closed_rows.insert({2,1});assert((memory.execute(q,{17,9})==Value{0,3}));
    assert(!memory.missing_evidence&&!memory.nonunique_result); // no terminal-node coverage required
    q.facts.push_back({2,1,5});observed->positive.emplace(2,1,5);
    assert(!memory.execute(q,{17,9}).present()&&memory.nonunique_result&&!memory.missing_evidence);
    q.facts.clear();observed->positive.clear();assert(!memory.execute(q,{17,9}).present()&&memory.nonunique_result);
    q.a={0,4};q.facts={{2,1,4}};observed->positive={{2,1,4}};observed->closed_columns.insert({1,4});
    assert((memory.execute(q,{19,9})==Value{0,2}));
    // Procedure search discovers one-hop selection from final answers, with a
    // second link and another relation preventing endpoint/identity shortcuts.
    memory.unique_enabled=true;std::vector<VerifiedQuestion> samples;
    for(int i=0;i<8;++i) {
        FlatQuery sample;sample.cue=7;sample.a={0,i};sample.b={0,100+i};sample.relation[0]=0;sample.relation[1]=1;
        sample.facts={{i,1,20+i},{20+i,1,30+i},{i,0,40+i},{100+i,1,50+i}};
        samples.push_back({sample,{0,20+i}});
    }
    auto found=memory.learn(samples);assert(found.found&&Meaning::from_program(found.program,2).key()=="17($0,)");
    auto prior=fixture();auto records=lessons();ObservationLanguageLearner learner;
    assert(learner.fit(records,prior));assert(learner.model.frames.size()==1);
    assert((learner.model.frames.begin()->second.candidates==std::set<ObservationLanguage::Rule>{{ObservationLanguage::FORGET_ROW,1,0,0}}));
    auto stale=prior;stale.events.frames.begin()->second.scenes.insert(9999);assert(!learner.publish(stale));
    assert(learner.publish(prior));
    input(prior,"\n<|endofdialogue|>\nCONTEXT: ada has the key.\nCONTEXT: ben has the coin.\nENVIRONMENT: The key was out of sight.\nUSER: Does ada have the key?\nASSISTANT:");
    assert(prior.reply==" I do not know.\n"&&prior.complete);
    auto knowledge=prior.observations;
    input(prior,"\nENVIRONMENT: ada has the key.\n",true);assert(!(prior.observations<knowledge)&&!(knowledge<prior.observations));
    input(prior,"\nENVIRONMENT: ada has the key.\nUSER: Does ada have the key?\nASSISTANT:");assert(prior.reply==" yes\n");
    input(prior,"\n<|endofdialogue|>\nCONTEXT: ruth passed the map to bill.\nUSER: Does ruth have the map?\nASSISTANT:");
    assert(prior.entities.count("ruth")&&prior.reply==" no\n");
    prior.observation_language.frames.begin()->second.scenes.insert(5000);
    input(prior,"\nUSER: Does ada have the key?\nASSISTANT:");assert(prior.status=="stale_observation_model");
    auto r=fixture();auto generated=records;for(auto& e:generated)e.self_generated=true;
    ObservationLanguageLearner ignored;assert(!ignored.fit(generated,r)&&ignored.accepted==0);
    auto repeats=records;for(auto& e:repeats)e.id=0;
    ObservationLanguageLearner repeated;assert(!repeated.fit(repeats,r)&&repeated.status=="insufficient_scene_support");
    ObservationLanguageLearner budget;budget.assignment_budget=0;assert(!budget.fit(records,r)&&budget.status=="insufficient_search_evidence");
    auto conflict=records;auto wrong=records.front();wrong.answer="no";conflict.push_back(wrong);
    ObservationLanguageLearner contradictory;assert(!contradictory.fit(conflict,r)&&contradictory.status=="conflicting_evidence");
    auto unknown=records;for(auto& e:unknown)e.answer="unknown";
    ObservationLanguageLearner untyped;assert(!untyped.fit(unknown,r)&&untyped.status=="untyped_unknown_question");
    // Distinct report hypotheses retain distinct worlds, not their fact union.
    r.partial_observation=true;r.entity("ada");r.entity("key");
    r.observation_language.frames["a report about $0 and $1 "]={{"a","report","about","$0","and","$1"},{{1,1,1,1},{2,1,1,1}},{1,2,3,4}};
    input(r,"CONTEXT: A report about ada and key.\nUSER: Does ada have the key?\nASSISTANT:");
    assert(r.possible_worlds.size()==2&&r.reply==" I do not know.\n");
    assert(r.observe_fact("key",1,"ada",true));assert(r.possible_worlds.size()==1);
    std::cout<<"unique reads, learned observation language, verified unknown feedback and uncertainty/provenance contracts passed\n";
}
