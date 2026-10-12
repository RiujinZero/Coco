#pragma once
#include "grounding_probe_support.h"
#include "laminar/incremental_grounding.h"
#include "laminar/online_grounding.h"
using namespace grounding_probe;
using Action=GroundedActuator::Action;
inline void stream(Organism& org,const std::string& text) {feed(org,std::string(1,(char)org.last_symbol())+text);}
inline std::vector<BootstrapGrounding::Episode> lessons(const std::string& path) {
    std::ifstream in(path);assert(in);std::vector<BootstrapGrounding::Episode> out;
    while(in.peek()!=EOF){auto h=fields(line(in));BootstrapGrounding::Episode e;e.id=std::stoull(h[0]);for(int i=0;i<std::stoi(h[1]);++i)e.observations.push_back(line(in));e.question=line(in);e.answer=line(in);out.push_back(std::move(e));}return out;
}
inline std::string facts(const LanguageReasoner& r) {std::string out;for(const auto& [a,rel,b]:state(r))out+=a+","+std::to_string(rel)+","+b+";";return out;}
inline std::string action_text(const Action& action) {std::string out=action.schema;for(const auto& a:action.arguments)out+=","+a;return out;}
inline std::string menu_text(const std::vector<Action>& menu) {std::string out;for(const auto& a:menu)out+=action_text(a)+";";return out;}
inline bool agree(const LanguageReasoner& r) {World w;return GroundedActuator::complete_world(r.request_snapshot(),w);}
inline std::shared_ptr<LanguageReasoner> prior(int seed) {
    auto r=std::make_shared<LanguageReasoner>(seed);train_skills(*r,seed);
    BootstrapGrounding b;assert(b.fit(lessons("logs/bootstrap_grounding/train.txt"),r->memory)&&b.publish(*r));
    if(seed%2) {
        for(auto& [key,frame]:r->events.frames){std::set<EventGrounding::Rule> rules;for(auto v:frame.candidates){v.relation^=1;rules.insert(v);}frame.candidates=rules;}
        for(auto& [cue,p]:r->memory.procedures)for(int& op:p)if((op>=4&&op<=7)||op>=10)op^=1;
    }
    IncrementalGrounding mixed,dialogue;
    assert(mixed.fit(lessons("logs/incremental_grounding/train.txt"),*r)&&mixed.publish(*r));
    assert(dialogue.fit(lessons("logs/dialogue_grounding/train.txt"),*r)&&dialogue.publish(*r));
    r->environment_dialogue=r->grounded_requests=true;r->world_budget=128;return r;
}
// Only the evaluator/environment owns goal bindings and physical state. The
// learner receives text, an unfiltered affordance menu and outcome feedback.
struct Environment {
    struct Goal {std::string text,subject,target;bool location=false,positive=true;};
    std::vector<std::string> people,objects,rooms;
    std::map<std::string,std::string> owner,place;
    int seed,version=0;
    Environment(const std::string& prefix,int s):seed(s) {
        for(int i=0;i<4;++i){people.push_back(prefix+"person"+std::to_string(i));rooms.push_back(prefix+"room"+std::to_string(i));place[people.back()]=rooms.back();}
        for(int i=0;i<3;++i){objects.push_back(prefix+"item"+std::to_string(i));owner[objects.back()]=people[i%2];}
    }
    std::string transfer() const {return "socket"+std::to_string(1000+1000*version+(seed%2?31:73));}
    std::string movement() const {return "socket"+std::to_string(1000+1000*version+(seed%2?73:31));}
    std::string initial() const {
        std::string out;for(const auto& [a,b]:place)out+="CONTEXT: "+a+" moved to the "+b+".\n";
        for(const auto& [o,p]:owner)out+="CONTEXT: "+p+" has the "+o+".\n";return out;
    }
    Goal request(int family,int serial) const {
        if(family==1) {
            auto p=people[(serial/3)%4];auto room=rooms[(serial+1)%4];
            if(place.at(p)==room)room=rooms[(serial+2)%4];
            return {"Please have "+p+" go to "+room+".",p,room,true,true};
        }
        auto o=objects[(serial/4)%3],giver=owner.at(o),target=people[(serial+1)%4];
        if(target==giver)target=people[(serial+2)%4];
        if(family==0)return {"Please have "+giver+" give the "+o+" to "+target+".",o,target,false,true};
        if(family==2)return {"Arrange for "+o+" to be held by "+target+".",o,target,false,true};
        assert(family==3);return {"Ensure "+o+" is not held by "+giver+".",o,giver,false,false};
    }
    bool satisfied(const Goal& goal) const {return ((goal.location?place:owner).at(goal.subject)==goal.target)==goal.positive;}
    std::vector<Action> menu(std::mt19937& rng) const {
        std::vector<Action> out;
        for(const auto& [o,p]:owner)for(const auto& target:people)if(target!=p)out.push_back({transfer(),{p,o,target}});
        for(const auto& [p,r]:place)for(const auto& target:rooms)if(target!=r)out.push_back({movement(),{p,target}});
        std::shuffle(out.begin(),out.end(),rng);return out;
    }
    std::pair<bool,std::string> execute(const Action& action,bool fail) {
        if(action.schema==transfer()) {
            auto a=action.arguments.at(0),o=action.arguments.at(1),b=action.arguments.at(2);assert(owner.at(o)==a&&a!=b);
            if(fail)return {false,a+" offers the "+o+" to "+b+", but "+b+" does not take it."};
            owner[o]=b;return {true,a+" gives the "+o+" to "+b+". "+b+" takes it."};
        }
        assert(action.schema==movement());auto a=action.arguments.at(0),b=action.arguments.at(1);assert(place.at(a)!=b);
        if(!fail)place[a]=b;return {!fail,a+" moved to the "+place.at(a)+"."};
    }
};
