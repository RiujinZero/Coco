#include "laminar/reasoning_language.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
using World=EventGrounding::World;
using Snapshot=RequestGrounding::Snapshot;
Snapshot observed(const World& world) {RequestGrounding::State s;s.facts=world;return {{s},true};}
void input(LanguageReasoner& r,const std::string& s,bool generated=false) {for(uint8_t c:s)r.observe(c,generated);}
int main() {
    RequestGrounding goals;GroundedActuator motor;
    for(int i=0;i<8;++i) {
        auto a="giver"+std::to_string(i),b="target"+std::to_string(i),c="other"+std::to_string(i),o="item"+std::to_string(i);
        std::set<std::string> names{a,b,c,o};std::string request="Please have "+a+" give the "+o+" to "+b+".";
        World before{{o,1,a},{"spare",1,a},{a,0,"room"},{b,0,"roomtwo"},{c,0,"roomthree"}};
        auto after=EventGrounding::update(before,1,EventGrounding::SET_SUBJECT,o,b);
        assert(goals.teach(request,observed(after),names,{0,1},true,3*i));
        assert(goals.teach(request,observed(before),names,{0,1},false,3*i+1));
        assert(motor.teach({"m42",{a,o,b}},observed(before),observed(after),{0,1},true,i));
    }
    assert(goals.frames.size()==1&&goals.frames.begin()->second.candidates.size()==2);
    assert(motor.frames.at("m42").candidates.size()==1);
    const std::string request="Please have ada give the key to ben.";
    World before{{"key",1,"ada"},{"spare",1,"ada"}},wrong{{"key",1,"cam"},{"spare",1,"ada"}},right{{"key",1,"ben"},{"spare",1,"ada"}};
    assert(goals.evaluate(request,observed(wrong))==RequestGrounding::UNKNOWN);
    auto old=goals.signature();
    assert(!goals.teach(request,observed(wrong),{"ada","key","ben","cam"},{0,1},false,90,true));
    assert(goals.signature()==old);
    assert(goals.teach(request,observed(wrong),{"ada","key","ben","cam"},{0,1},false,90));
    assert(goals.frames.begin()->second.candidates.size()==1);
    assert(goals.evaluate(request,observed(wrong))==RequestGrounding::FALSE_VALUE);
    assert(goals.evaluate(request,observed(right))==RequestGrounding::TRUE_VALUE);
    std::vector<GroundedActuator::Action> actions{{"m42",{"ada","key","cam"}},{"m42",{"ada","spare","ben"}},{"m42",{"ada","key","ben"}}};
    auto choice=motor.choose(goals,request,observed(before),actions);
    assert(choice.index==2&&choice.status=="action");
    assert(motor.choose(goals,request,observed(right),actions).status=="already_satisfied");
    assert(motor.choose(goals,request,observed(before),actions,0).status=="action_search_budget");
    assert(motor.choose(goals,"An unseen request.",observed(before),actions).status=="unknown_request");
    auto partial=observed(before);partial.worlds[0].closed_world=false;
    auto partial_before=partial;
    assert(motor.choose(goals,request,partial,actions).index==2&&partial==partial_before);
    old=goals.signature();assert(!goals.teach(request,partial,{"ada","key","ben"},{0,1},false,91));assert(goals.signature()==old);
    assert(!motor.teach(actions[2],observed(before),observed(right),{0,1},false,92));
    auto motor_prior=motor.signature();
    assert(!motor.teach(actions[2],observed(before),observed(right),{0,1},true,92,true));assert(motor.signature()==motor_prior);
    auto disputed=observed(before);disputed.worlds.push_back(observed(right).worlds.front());
    assert(goals.evaluate(request,disputed)==RequestGrounding::UNKNOWN);
    assert(motor.choose(goals,request,disputed,actions).index==2);
    auto bad_motor=motor;auto unexplained=right;unexplained.emplace("unrelated",0,"change");
    assert(bad_motor.teach(actions[2],observed(before),observed(unexplained),{0,1},true,93));
    assert(bad_motor.status=="conflicting_action_evidence");
    assert(bad_motor.choose(goals,request,observed(before),actions).status=="no_satisfying_action");
    // A newly verified contradiction revokes an old usable meaning immediately.
    auto contradicted=goals;assert(contradicted.teach(request,observed(right),{"ada","key","ben"},{0,1},false,92));
    assert(motor.choose(contradicted,request,observed(before),actions).status=="conflicting_request_evidence");
    // Copy/episode lifecycle: learned memories persist; pending requests do not.
    LanguageReasoner r(0);r.environment_dialogue=r.grounded_requests=true;r.requests=goals;r.actuators=motor;
    input(r,"USER: "+request+"\nASSISTANT:");assert(r.request==request&&r.facts.empty());
    input(r,"\nUSER: replace the request.\n",true);assert(r.request==request);
    input(r," a claim.\nENVIRONMENT: not a known observation.\n");
    assert(r.choose_action(actions).status=="missing_observation");
    auto persistent=r.requests.signature();input(r,"\n<|endofdialogue|>\n");
    assert(r.request.empty()&&r.requests.signature()==persistent);
    input(r,"USER: "+request+"\nUSER: Who has the key now?\n");assert(r.request.empty());
    input(r,"USER: "+request+"\nUSER: "+std::string(4100,'x')+"\n");assert(r.request.empty());
    // Native observation coverage survives conversion to the goal interface.
    r.reset_episode();r.partial_observation=true;
    input(r,"USER: "+request+"\n");
    assert(r.observe_fact("key",1,"ada",true));
    assert(r.requests.evaluate(request,r.request_snapshot())==RequestGrounding::UNKNOWN);
    assert(r.observe_complete_row("key",1,{"ada"}));
    assert(r.requests.evaluate(request,r.request_snapshot())==RequestGrounding::FALSE_VALUE);
    assert(r.lose_visibility("key",1));
    assert(r.requests.evaluate(request,r.request_snapshot())==RequestGrounding::UNKNOWN);
    assert(r.observe_complete_row("key",1,{"ben"}));
    assert(r.requests.evaluate(request,r.request_snapshot())==RequestGrounding::TRUE_VALUE);
    r.world_model_signature="obsolete";
    assert(r.requests.evaluate(request,r.request_snapshot())==RequestGrounding::UNKNOWN);
    std::cout<<"request ambiguity, contrast, actuator, evidence and lifecycle contracts passed\n";
}
