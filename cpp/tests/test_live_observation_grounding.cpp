#include "../apps/grounding_probe_support.h"
#include "laminar/live_observation_grounding.h"
#include "laminar/online_grounding.h"
using namespace grounding_probe;
void input(LanguageReasoner& r,const std::string& s,bool self=false) {for(uint8_t c:s)r.observe(c,self);}
LanguageReasoner fixture() {
    LanguageReasoner r(0);train_skills(r,0);r.environment_dialogue=r.grounded_events=r.uncertain_events=r.grounded_requests=true;
    r.forms["count possessions of $0 "]=0;r.memory.procedures[0]={15,9};
    r.forms["count destinations of $0 "]=1;r.memory.procedures[1]={6,9};
    for(int i=0;i<8;++i) {
        auto a="actor"+std::to_string(i),b="other"+std::to_string(i),o="object"+std::to_string(i);
        World before{{o,1,b},{a,0,"oldroom"},{b,0,"room"},{"spare",1,a}};
        auto after=EventGrounding::update(before,1,EventGrounding::SET_SUBJECT,o,a);
        assert(r.events.teach_transition(a+" has the "+o+".",before,after,i));
        after=EventGrounding::update(before,0,EventGrounding::SET_SUBJECT,a,std::string("room"));
        assert(r.events.teach_transition(a+" moved to the room.",before,after,20+i));
    }
    return r;
}
void setup(LanguageReasoner& r) {
    input(r,"\n<|endofdialogue|>\nCONTEXT: ada moved to the hall.\nCONTEXT: ben moved to the kitchen.\nCONTEXT: cam moved to the garden.\nCONTEXT: ada has the key.\nCONTEXT: cam has the coin.\nUSER: Please have ada give the key to ben.\n");
}
std::string answer(const std::string& question,const World& world) {
    auto w=EventGrounding::words(question);int count=0;
    for(const auto& [a,r,b]:world)if(w[1]=="possessions"?(r==1&&b==w.back()):(r==0&&a==w.back()))++count;
    return std::to_string(count);
}
int main() {
    auto r=fixture();LiveObservationGrounding observations;OnlineGrounding control(91);
    const std::string event="The key passed from ada into ben's keeping.";
    std::vector<std::string> menu;
    for(const std::string name:{"ada","ben","cam","key","coin","hall","kitchen","garden"})
        for(const std::string kind:{"possessions","destinations"})menu.push_back("Count "+kind+" of "+name+"?");
    for(int scene=0;scene<4;++scene) {
        setup(r);assert(control.feedback_request(r,r.request_revision,false,100+scene));
        auto issued=control.choose(r,{{"motor",{"ada","key","ben"}}});assert(issued.index==0);
        auto expected=EventGrounding::update(state(r),1,EventGrounding::SET_SUBJECT,std::string("key"),std::string("ben"));
        assert(observations.begin(r,event,scene));input(r,"ENVIRONMENT: "+event+"\n");
        assert(!observations.commit(r));assert(!r.request_snapshot().valid);
        for(int id=0;id<16;++id) {
            auto proposal=observations.propose(menu);assert(proposal.index>=0);
            auto question=menu.at(proposal.index),gold=answer(question,expected);auto before=observations.learner.signature();
            assert(!observations.feedback(r,question,gold,id,true));assert(observations.learner.signature()==before);
            assert(observations.feedback(r,question,gold,id));auto evidence=observations.learner.signature();
            assert(observations.feedback(r,question,gold,id)&&observations.status=="duplicate_feedback");
            assert(!observations.feedback(r,question,"999",id)&&observations.status=="conflicting_feedback_id");
            assert(observations.learner.signature()==evidence);
            if(observations.commit(r))break;
        }
        assert(!observations.pending&&observations.receipt&&r.complete&&state(r)==expected);
        const auto& receipt=*observations.receipt;
        auto unreconciled=control;assert(!unreconciled.feedback_transition(r,issued.token,true,true,200+scene));
        assert(unreconciled.status=="stale_observation_model");
        auto wrong=receipt.before;wrong.worlds.front().facts.clear();
        assert(!control.rebase_observation_model(r,issued.token,receipt.previous_model,wrong));
        assert(control.rebase_observation_model(r,issued.token,receipt.previous_model,receipt.before));
        assert(control.feedback_transition(r,issued.token,true,true,200+scene));
        auto frame=r.events.frames.find("the $0 passed from $1 into $2 s keeping ");assert(frame!=r.events.frames.end());
        assert(frame->second.scenes.size()==size_t(scene+1));
    }
    assert(r.events.alternatives(event).matched);
    auto old_events=r.events.signature();setup(r);assert(!observations.begin(r,event,3)&&observations.status=="reused_scene");
    // The wrong question budget cannot be used to silently choose a world.
    LiveObservationGrounding weak;auto other=fixture();setup(other);
    assert(weak.begin(other,event,10));input(other,"ENVIRONMENT: "+event+"\n");
    assert(weak.propose(menu,0).status=="question_search_budget");
    assert(weak.feedback(other,"Count destinations of ada?","1",0));assert(!weak.commit(other));
    assert(weak.worlds()>1&&!other.complete);
    input(other,"USER: A different request.\n");
    assert(!weak.feedback(other,menu[0],"0",1)&&weak.status=="stale_observation_feedback");
    auto partial=fixture();setup(partial);partial.partial_observation=true;
    LiveObservationGrounding unavailable;assert(!unavailable.begin(partial,event,30));
    assert(unavailable.status=="uncertain_before_state");
    // A verified impossible count conflicts; generated versions do not teach.
    auto bad=fixture();setup(bad);LiveObservationGrounding conflict;
    assert(conflict.begin(bad,event,40));input(bad,"ENVIRONMENT: "+event+"\n");
    assert(conflict.feedback(bad,menu[0],"999",0));assert(conflict.status=="conflicting_observation_feedback");
    assert(!conflict.commit(bad)&&!bad.complete);
    assert(r.events.signature()==old_events);
    std::cout<<"live observation QA, instance evidence, model refinement and feedback provenance contracts passed\n";
}
