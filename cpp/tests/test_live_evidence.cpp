#include "laminar/live_evidence_grounding.h"
#include "laminar/online_grounding.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
void input(LanguageReasoner& r,const std::string& text,bool generated=false){for(uint8_t c:text)r.observe(c,generated);}
LanguageReasoner fixture() {
    LanguageReasoner r(0);r.environment_dialogue=r.grounded_events=r.uncertain_events=r.partial_observation=r.grounded_requests=true;r.world_budget=512;
    r.forms={{"does $0 have the $1 ",0},{"is $0 at $1 ",1},{"who has the $0 now ",2},{"where is $0 ",3}};
    r.memory.procedures={{0,{3,11,9}},{1,{10,9}},{2,{17,9}},{3,{16,9}}};
    for(int i=0;i<8;++i) {
        auto a="actor"+std::to_string(i),o="item"+std::to_string(i);
        EventGrounding::World before{{o,1,"old"},{"spare",1,a},{a,0,"oldroom"}};
        assert(r.events.teach_transition(a+" has the "+o+".",before,EventGrounding::update(before,1,2,o,a),i));
        assert(r.events.teach_transition(a+" moved to the room.",before,EventGrounding::update(before,0,2,a,std::string("room")),100+i));
    }
    // These fixtures isolate interaction contracts. The native experiment loads
    // the actual learned request and effect models from the preceding run.
    r.requests.frames["arrange for $0 to be held by $1 "]={{"arrange","for","$0","to","be","held","by","$1"},{{1,0,1,true}},{1,2,3,4},true};
    r.actuators.frames["motor"]={3,{{1,1,2,2}},{1,2,3,4},0};
    return r;
}
void setup(LanguageReasoner& r) {
    input(r,"\n<|endofdialogue|>\nCONTEXT: ada moved to the hall.\nCONTEXT: ben moved to the kitchen.\nCONTEXT: ada has the key.\nCONTEXT: ben has the coin.\nUSER: Arrange for key to be held by ben.\n");
}
std::vector<std::string> menu(const LanguageReasoner& r) {
    std::vector<std::string> out;for(const auto& a:r.names)for(const auto& b:r.names) {
        out.push_back("Does "+b+" have the "+a+"?");out.push_back("Is "+a+" at "+b+"?");
    }return out;
}
std::string gold(const LanguageReasoner& r,const FactKnowledge& k,const std::string& q) {
    auto words=EventGrounding::words(q);bool owner=words[0]=="does";
    int a=r.entities.at(owner?words.back():words[1]),b=r.entities.at(owner?words[1]:words.back());
    auto value=k.truth(a,owner?1:0,b);return value==FactKnowledge::UNKNOWN?"unknown":value==FactKnowledge::TRUE_VALUE?"yes":"no";
}
void teach(LiveEvidenceGrounding& learner,LanguageReasoner& r,const FactKnowledge& expected) {
    auto options=menu(r);
    for(int i=0;i<24;++i) {
        auto proposal=learner.propose(options);assert(proposal.index>=0);
        auto q=options[proposal.index],a=gold(r,expected,q);auto before=learner.candidates();
        assert(!learner.feedback(r,q,a,i,true)&&learner.candidates()==before);
        assert(learner.feedback(r,q,a,i));auto branches=learner.pending->branches.size();
        assert(learner.feedback(r,q,a,i)&&learner.status=="duplicate_feedback");
        assert(!learner.feedback(r,q,a=="yes"?"no":"yes",i)&&learner.status=="conflicting_feedback_id");
        assert(learner.pending->branches.size()==branches);
        if(learner.knowledge_states()==1)break;
    }
    assert(learner.knowledge_states()==1&&learner.commit(r));
    assert(!(r.observations<expected)&&!(expected<r.observations));
}
int main() {
    auto r=fixture();LiveEvidenceGrounding learner;const auto text="The key became unseen.";
    for(int scene=0;scene<4;++scene) {
        setup(r);auto expected=r.observations;expected.forget_row(r.entities.at("key"),1);
        assert(learner.begin(r,text,scene));input(r,"ENVIRONMENT: "+std::string(text)+"\n");
        assert(!learner.commit(r)&&learner.status=="unverified_observation");
        assert(learner.propose(menu(r),0).status=="question_search_budget");
        assert(!learner.feedback(r,menu(r)[0],"-1",99)&&learner.status=="invalid_answer");
        teach(learner,r,expected);
        assert(r.observation_language.frames.at("the $0 became unseen ").scenes.size()==size_t(scene+1));
    }
    assert(r.requests.evaluate(r.request,r.request_snapshot())==RequestGrounding::UNKNOWN);
    const auto goals=r.requests.signature(),motors=r.actuators.signature();OnlineGrounding control(0);control.learning=false;
    auto observed=r.request_snapshot();auto decision=control.choose(r,{{"motor",{"ada","key","ben"}}});assert(decision.index==0&&control.pending);
    assert(r.awaiting_action_observation&&!r.request_snapshot().valid);
    // Predictions establish a future effect but never fill current observations.
    assert(observed.worlds[0].facts==r.request_snapshot().worlds[0].facts);
    const auto recovery="The key came into ben's custody.";auto expected=r.observations;expected.update(r.entities.at("key"),1,r.entities.at("ben"),2);
    assert(learner.begin(r,recovery,30));input(r,"ENVIRONMENT: "+std::string(recovery)+"\n");teach(learner,r,expected);
    auto refused=control;assert(!refused.feedback_transition(r,decision.token,true,true,50)&&refused.status=="stale_observation_model");
    auto wrong=learner.receipt->before;wrong.worlds[0].closed_rows.clear();
    assert(!control.rebase_evidence_model(r,decision.token,learner.receipt->previous_model,wrong));
    assert(control.rebase_evidence_model(r,decision.token,learner.receipt->previous_model,learner.receipt->before));
    assert(control.feedback_transition(r,decision.token,true,true,50));
    assert(r.requests.signature()==goals&&r.actuators.signature()==motors);
    assert(r.requests.evaluate(r.request,r.request_snapshot())==RequestGrounding::TRUE_VALUE);
    // Scope closure must be propagated in hypothetical negative-goal outcomes.
    auto negatives=r.requests;for(auto& [key,frame]:negatives.frames)for(auto rule:std::vector<RequestGrounding::Rule>(frame.candidates.begin(),frame.candidates.end())){frame.candidates.erase(rule);rule.present=false;frame.candidates.insert(rule);}
    auto partial=r.request_snapshot();partial.worlds[0].facts.clear();partial.worlds[0].closed_rows.clear();
    assert(r.actuators.choose(negatives,"Arrange for key to be held by ada.",partial,{{"motor",{"ada","key","ben"}}}).index==0);
    auto add_only=r.actuators;add_only.frames["motor"].candidates={{1,1,2,0}};
    assert(add_only.choose(negatives,"Arrange for key to be held by ada.",partial,{{"motor",{"ada","key","ben"}}}).index<0);
    auto unchanged=partial;assert(!r.actuators.teach({"motor",{"ada","key","ben"}},partial,r.request_snapshot(),{0,1},true,60));assert(partial==unchanged);
    // A weak answer may preserve several worlds; it cannot certify a selected
    // one. They are retained with their frame bindings after publication.
    auto weak=fixture();setup(weak);LiveEvidenceGrounding uncertain;
    assert(uncertain.begin(weak,text,1));input(weak,"ENVIRONMENT: "+std::string(text)+"\n");
    assert(uncertain.feedback(weak,"Is ada at hall?","yes",0));assert(uncertain.knowledge_states()>1);
    assert(uncertain.commit(weak)&&weak.possible_worlds.size()>1);
    input(weak,"USER: Does ada have the key?\nASSISTANT:");assert(weak.reply==" I do not know.\n");
    auto stale=fixture();setup(stale);LiveEvidenceGrounding delayed;
    assert(delayed.begin(stale,text,1));input(stale,"ENVIRONMENT: "+std::string(text)+"\nUSER: A new request.\n");
    assert(!delayed.feedback(stale,"Does ada have the key?","unknown",0)&&delayed.status=="stale_observation_feedback");
    auto conflict=fixture();setup(conflict);LiveEvidenceGrounding contradiction;
    assert(contradiction.begin(conflict,text,1));input(conflict,"ENVIRONMENT: "+std::string(text)+"\n");
    assert(contradiction.feedback(conflict,"Does ada have the key?","unknown",0));
    assert(contradiction.feedback(conflict,"Does ada have the key?","yes",1));
    assert(!contradiction.commit(conflict)&&contradiction.status=="conflicting_observation_feedback");
    assert(conflict.observation_language.frames.empty());
    auto bounded=fixture();setup(bounded);LiveEvidenceGrounding budget;budget.hypothesis_budget=0;
    assert(!budget.begin(bounded,text,1)&&budget.status=="hypothesis_budget");
    // Existing alternative-history bindings survive a new report's filtering.
    auto history=fixture();setup(history);history.observation_language.frames["a note about $0 "]={{"a","note","about","$0"},{{0,-1,-1,0}},{1,2,3,4}};
    auto left=history.possible_worlds.front(),right=left;left.meanings["old"]= "left";right.meanings["old"]="right";
    right.knowledge.update(history.entities.at("coin"),1,history.entities.at("ada"),2);right.facts=right.knowledge.positive;
    history.possible_worlds={left,right};history.project_worlds();LiveEvidenceGrounding retained;retained.owned.insert("a note about $0 ");
    assert(retained.begin(history,"A note about key.",10));input(history,"ENVIRONMENT: A note about key.\n");
    assert(retained.feedback(history,"Does ben have the coin?","yes",0)&&retained.commit(history));
    assert(history.possible_worlds.size()==1&&history.possible_worlds[0].meanings.at("old")=="left");
    std::cout<<"live evidence feedback, branch preservation, partial planning and frozen actuation contracts passed\n";
}
