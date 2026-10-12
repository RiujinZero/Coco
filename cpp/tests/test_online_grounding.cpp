#include "laminar/online_grounding.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
using World=EventGrounding::World;
void input(LanguageReasoner& r,const std::string& s,bool generated=false) {for(uint8_t c:s)r.observe(c,generated);}
LanguageReasoner fixture() {
    LanguageReasoner r(0);r.environment_dialogue=r.grounded_requests=r.grounded_events=r.uncertain_events=true;
    for(int i=0;i<8;++i) {
        auto a="actor"+std::to_string(i),b="receiver"+std::to_string(i),o="item"+std::to_string(i);
        World before{{o,1,a},{"spare",1,a},{a,0,"room"},{b,0,"roomtwo"}};
        auto next=EventGrounding::update(before,1,EventGrounding::SET_SUBJECT,o,b);
        auto prior=EventGrounding::update(before,1,EventGrounding::SET_SUBJECT,o,std::string("old"));
        assert(r.events.teach_transition(a+" has the "+o+".",prior,before,i));
        assert(r.events.teach_transition(a+" gives the "+o+" to "+b+".",before,next,20+i));
    }
    // Fixture lexical grounds names; request and actuator memories start empty.
    input(r,"CONTEXT: ada has the key.\nCONTEXT: ben has the coin.\nUSER: Please have ada give the key to ben.\n");
    assert(r.complete);return r;
}
int main() {
    auto r=fixture();OnlineGrounding loop(31);
    std::vector<GroundedActuator::Action> menu{{"opaque",{"ada","key","ben"}}};
    assert(loop.feedback_request(r,r.request_revision,false,1));
    auto goal_before=r.requests.signature();auto motor_before=r.actuators.signature();
    auto d=loop.choose(r,menu);assert(d.index==0&&d.status=="explore_actuator");
    assert(r.awaiting_action_observation&&!r.request_snapshot().valid);
    auto query_copy=r;input(query_copy,"\nUSER: Who has the key now?\nASSISTANT:");
    assert(query_copy.reply==" I do not know.\n"&&query_copy.status=="awaiting_action_observation");
    assert(loop.choose(r,menu).status=="awaiting_transition_feedback");
    assert(!loop.feedback_transition(r,d.token,true,true,2));assert(loop.status=="awaiting_observation"&&loop.pending);
    input(r,"ENVIRONMENT: ada gives the key to ben.\n",true);
    assert(!loop.feedback_transition(r,d.token,true,true,2));assert(loop.status=="awaiting_observation");
    input(r,"ENVIRONMENT: ada gives the key to ben.\n");assert(r.complete);
    assert(!r.awaiting_action_observation&&r.request_snapshot().valid);
    assert(!loop.feedback_transition(r,d.token,true,true,2,true));assert(loop.status=="self_generated_feedback"&&loop.pending);
    assert(loop.feedback_transition(r,d.token,true,true,2));assert(!loop.pending);
    assert(r.requests.signature()!=goal_before&&r.actuators.signature()!=motor_before);
    auto learned=r.requests.signature(),motors=r.actuators.signature();
    assert(!loop.feedback_transition(r,d.token,true,true,2));assert(loop.status=="unknown_feedback_token");
    assert(r.requests.signature()==learned&&r.actuators.signature()==motors);
    auto revoked=[&](const std::string& intervening,const std::string& expected) {
        r=fixture();loop=OnlineGrounding(9);assert(loop.feedback_request(r,r.request_revision,false,10));
        auto decision=loop.choose(r,menu);assert(decision.index==0);
        auto a=r.requests.signature(),b=r.actuators.signature();input(r,intervening);
        assert(!loop.feedback_transition(r,decision.token,true,true,11));assert(loop.status==expected&&!loop.pending);
        assert(r.requests.signature()==a&&r.actuators.signature()==b);
    };
    revoked("USER: A different request.\nENVIRONMENT: ada gives the key to ben.\n","stale_request_feedback");
    revoked("<|endofdialogue|>\n","stale_request_feedback");
    revoked("CONTEXT: ada gives the key to ben.\n","unmatched_observation");
    revoked("ENVIRONMENT: ada gives the key to ben.\nENVIRONMENT: ben has the key.\n","unmatched_observation");
    revoked("ENVIRONMENT: an unknown observation.\n","missing_observation");
    r=fixture();loop=OnlineGrounding(13);assert(loop.feedback_request(r,r.request_revision,false,20));
    d=loop.choose(r,menu);auto before_goal=r.requests.signature();
    input(r,"ENVIRONMENT: ada gives the key to ben.\n");
    assert(!loop.feedback_transition(r,d.token,false,true,21));assert(loop.status=="inconsistent_failure");
    assert(r.requests.signature()==before_goal&&r.actuators.frames.empty());
    // Stale request labels and evaluation mode cannot create a new meaning.
    auto revision=r.request_revision;input(r,"USER: Move ben elsewhere.\n");
    assert(!loop.feedback_request(r,revision,false,22));assert(loop.status=="stale_request_feedback");
    loop.learning=false;assert(!loop.feedback_request(r,r.request_revision,false,23));
    assert(loop.status=="learning_disabled"&&r.requests.signature()==before_goal);
    assert(loop.choose(r,menu).index<0&&!loop.pending);
    loop.learning=true;assert(loop.choose(r,menu,0).status=="action_search_budget");
    // A new request cannot make a still-unobserved action outcome certain.
    r=fixture();loop=OnlineGrounding(7);d=loop.choose(r,menu);
    input(r,"USER: Please have ada give the key to ben.\n");
    assert(loop.choose(r,menu).status=="missing_observation"&&!loop.pending);
    assert(r.awaiting_action_observation);
    input(r,"ENVIRONMENT: ada gives the key to ben.\n");assert(r.request_snapshot().valid);
    // Mere task success does not distinguish "recipient holds it" from "giver
    // lost it". Select a separating outcome even when an exploit is available.
    r=fixture();input(r,"CONTEXT: cam has the cup.\n");
    auto snap=[](const World& world){RequestGrounding::State state;state.facts=world;return RequestGrounding::Snapshot{{state},true};};
    for(int i=0;i<4;++i) {
        World before{{"key",1,"ada"},{"coin",1,"ben"},{"cup",1,"cam"}},after{{"key",1,"ben"},{"coin",1,"ben"},{"cup",1,"cam"}};
        assert(r.requests.teach(r.request,snap(before),{"ada","key","ben"},{0,1},false,40+2*i));
        assert(r.requests.teach(r.request,snap(after),{"ada","key","ben"},{0,1},true,41+2*i));
        assert(r.actuators.teach(menu[0],snap(before),snap(after),{0,1},true,50+i));
    }
    assert(r.requests.meaning(r.request).alternatives.size()==2);
    menu.push_back({"opaque",{"ada","key","cam"}});
    auto passive_reader=r;OnlineGrounding passive(3);passive.information_seeking=false;
    auto passive_choice=passive.choose(passive_reader,menu);assert(passive_choice.index==0&&passive_choice.status=="exploit");
    loop=OnlineGrounding(3);d=loop.choose(r,menu);assert(d.index==1&&d.status=="explore_meaning");
    input(r,"ENVIRONMENT: ada gives the key to cam.\n");
    assert(loop.feedback_transition(r,d.token,true,false,70));assert(r.requests.meaning(r.request).alternatives.size()==1);
    // Frozen operation still tracks an issued action and its observation, while
    // withholding all updates to request/effect meanings.
    loop.learning=false;menu={{"opaque",{"cam","key","ben"}}};
    auto frozen_goals=r.requests.signature(),frozen_motors=r.actuators.signature();
    d=loop.choose(r,menu);assert(d.index==0&&d.token&&r.awaiting_action_observation);
    input(r,"ENVIRONMENT: cam gives the key to ben.\n");
    assert(loop.feedback_transition(r,d.token,true,true,71));assert(loop.status=="observed_transition");
    assert(r.requests.signature()==frozen_goals&&r.actuators.signature()==frozen_motors);
    std::cout<<"online grounding feedback provenance and transaction contracts passed\n";
}
