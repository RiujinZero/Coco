#include "laminar/relational_control.h"
#include <iostream>
#include <limits>
#include <type_traits>
using namespace laminar;
using namespace laminar::relational;
void check(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
struct MissionHarness:BasicControlledRelationalSession<7> {
    using Base=BasicControlledRelationalSession<7>;
    MissionHarness():Base(7,true,false,true,{true,1,32},false,32,{},false,true){}
    using Base::select_control;using Base::execute_physical;using Base::close_episode_evidence;
    using Base::submit_mission_credit;using Base::revise_mission_credit;using Base::pending_mission_evidence;
    using Base::import_five_action_policy;using Base::valid_control_request;
    using Base::current_episode_id;using Base::current_motor_count;
};
struct FiveActionMissionHarness:ControlledRelationalSession {
    FiveActionMissionHarness():ControlledRelationalSession(11,true,false,true){}
    using ControlledRelationalSession::select_control;
    using ControlledRelationalSession::close_episode_evidence;
    using ControlledRelationalSession::submit_mission_credit;
    using ControlledRelationalSession::revise_mission_credit;
    using ControlledRelationalSession::pending_mission_evidence;
};
ControlStep physical(MissionHarness& learner,ControlOp op) {
    auto observation=learner.observe();observation[30]=1;
    return learner.execute_physical(learner.select_control(observation,{0,1,2,3,4,5,6},0,false,nullptr,ControlHint{op,"mission_fixture"}));
}
ControlStep management(MissionHarness& learner,ControlOp op) {
    reasoning::Observation obs{};obs[29]=obs[31]=1;
    auto step=learner.select_control(obs,{1,5,6},0,false,nullptr,ControlHint{op,"mission_fixture"});
    step.cost=op==ControlOp::next_goal?.002f:.01f;return step;
}
Scene add(Scene world,const std::string& predicate){world.facts.push_back({predicate,{"a","b"}});return sorted(world);}
void import_and_selection() {
    using OldValues=std::remove_cvref_t<decltype(std::declval<ControlledRelationalSession>().empirical_values())>;
    static_assert(std::is_same_v<OldValues,std::map<reasoning::Observation,std::array<float,5>>>);
    using NewValues=std::remove_cvref_t<decltype(std::declval<MissionHarness>().empirical_values())>;
    static_assert(std::is_same_v<NewValues,std::map<reasoning::Observation,std::array<float,7>>>);
    MissionHarness learner;reasoning::BackpropChoice old(5,3,32);
    for(size_t i=0;i<old.weights.size();++i)old.weights[i]+=.001f*float(int(i%7)-3);
    check(learner.import_five_action_policy(old),"five-action import rejected");
    reasoning::Observation obs{};for(size_t i=0;i<obs.size();++i)obs[i]=.03f*float(i);
    for(int a=0;a<5;++a)check(old.action_value(obs,a)==learner.policy.action_value(obs,a),"old output row/bias shifted on import");
    check(learner.policy.action_value(obs,5)==0&&learner.policy.action_value(obs,6)==0,"new output rows overwritten by old biases");
    check(learner.policy.choose(obs,{0,1,2,3,4})==old.choose(obs,{0,1,2,3,4}),"old-action-only choice changed");
    auto step=management(learner,ControlOp::next_goal);check(step.operation==ControlOp::next_goal&&!learner.active(),"agenda choice required active episode");
    check(!learner.import_five_action_policy(old),"late policy import erased history");
    reasoning::PreferenceTree tree;bool rejected=false;
    try{learner.select_control(obs,{0,1,2,3,4},0,false,&tree);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"seven-action preference-tree readout accepted");
    rejected=false;try{learner.select_control(obs,{1,1});}catch(const std::invalid_argument&){rejected=true;}check(rejected,"duplicate legal menu accepted");
    check(learner.valid_control_request("unknown a",{"a"},{{{},true},1},{})&&!learner.valid_control_request("",{"a"},{{{},true},1},{}),"structural request validator drift");
    check(learner.begin("selection","unknown a",{"a"},{{{},true},1},{},false),"selection fixture rejected");
    auto selected=learner.select_control(learner.observe(),{1,4},0,false,nullptr,ControlHint{ControlOp::reason,"mission_fixture"});
    auto forged=selected;forged.observation[30]=1;rejected=false;
    try{learner.execute_physical(forged);}catch(const std::invalid_argument&){rejected=true;}check(rejected&&learner.trace().empty(),"forged public selection accepted");
    auto done=learner.execute_physical(selected);check(done.observation==selected.observation&&done.available==selected.available,"physical execution replaced selected public inputs");
    rejected=false;try{learner.execute_physical(selected);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"selection replay accepted");
    check(learner.current_episode_id()=="selection"&&learner.current_motor_count()==0,"episode accessors drift");learner.cancel();
    learner.set_learning(false);rejected=false;
    try{management(learner,ControlOp::activate_goal);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"frozen management hint accepted");
    rejected=false;try{BasicControlledRelationalSession<7> invalid(1,true,false,true,{},true);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"seven-action model credit accepted");
}
void linked_mission() {
    MissionHarness learner;
    const Scene initial{{{"edge",{"a","old"}},{"edge",{"other","b"}},{"marked",{"a","old"}},{"marked",{"other","b"}}},true};
    const Scene complete=add(add(initial,"edge"),"marked");const std::set<std::string> names{"a","b","old","other"};
    const std::vector<Action> actions{{"edge_motor",{"a","b"}},{"mark_motor",{"a","b"}}};
    for(int i=0;i<3;++i) {
        auto world=i==0?complete:i==1?add(initial,"edge"):add(initial,"marked");
        concepts::Receipt c;c.id="concept"+std::to_string(i);c.name="joined";c.args={"a","b"};c.scene=world;c.answer=i==0;
        check(learner.knowledge.teach_concept(c),"concept rejected");
        LanguageReceipt l;l.id="language"+std::to_string(i);l.text="join a with b";l.names=names;l.scene=world;l.answer=i==0;
        check(learner.knowledge.teach_language(l),"language rejected");
    }
    for(int i=0;i<2;++i) {
        TransitionReceipt t;t.id="effect"+std::to_string(i);t.action=actions[i];t.before=initial;t.after=add(initial,i?"marked":"edge");
        check(learner.knowledge.teach_transition(t),"effect rejected");
    }
    learner.knowledge.rebuild();auto initial_weights=learner.policy.weights;std::vector<ControlStep> trace;
    trace.push_back(management(learner,ControlOp::next_goal));trace.push_back(management(learner,ControlOp::activate_goal));
    check(learner.begin("parent","join a with b",names,{initial,1},actions,true),"parent begin failed");
    trace.push_back(physical(learner,ControlOp::reason));trace.push_back(physical(learner,ControlOp::act));auto step=trace.back();
    check(step.actions.size()==1&&step.goal=="parent"&&step.observation[30]==1,"physical mission selection lost ownership/features");
    auto world=add(initial,step.actions[0].schema=="edge_motor"?"edge":"marked");
    check(learner.actions(step.token,{world},{true},2)&&learner.suspend(),"parent segment failed");
    auto segment=learner.pending_mission_evidence();check(segment.size()==1&&!segment[0].satisfaction,"interruption invented terminal satisfaction");
    trace.push_back(management(learner,ControlOp::activate_goal));
    check(learner.begin("child","join a with b",names,{complete,3},{},false),"child begin failed");
    trace.push_back(physical(learner,ControlOp::reason));trace.push_back(physical(learner,ControlOp::finish));
    check(learner.close_episode_evidence(true).has_value(),"child evidence close failed");
    check(learner.policy.weights==initial_weights&&learner.credits().empty()&&learner.empirical_values().empty(),"child completion delivered premature control credit");
    trace.push_back(management(learner,ControlOp::activate_goal));check(learner.resume("parent",{world,4}),"parent resume failed");
    trace.push_back(physical(learner,ControlOp::reason));trace.push_back(physical(learner,ControlOp::act));step=trace.back();
    auto after=add(world,step.actions[0].schema=="edge_motor"?"edge":"marked");
    check(learner.actions(step.token,{after},{true},5),"parent final action failed");
    trace.push_back(physical(learner,ControlOp::reason));trace.push_back(physical(learner,ControlOp::finish));
    check(learner.close_episode_evidence(true).has_value(),"parent evidence close failed");
    auto links=learner.pending_mission_evidence();check(links.size()==3,"mission did not retain every semantic segment");
    auto missing=links;missing.pop_back();check(!learner.submit_mission_credit("mission",trace,missing,true),"mission accepted omitted semantic link");
    check(learner.submit_mission_credit("mission",trace,links,true,128),"mission credit rejected");
    check(learner.credits().size()==1&&learner.credits()[0].trace.size()==trace.size()&&learner.updates()==128,"mission decisions credited separately");
    auto weights=learner.policy.weights;auto values=learner.empirical_values();check(weights!=initial_weights,"mission did not train shared policy");
    check(!learner.revise_credit("mission",1,false,false),"single-goal correction bypassed mission links");
    auto bad=learner.credits()[0].links;bad[0].satisfaction=false;++bad.back().version;auto version=learner.knowledge.version();
    check(!learner.revise_mission_credit("mission",1,false,bad)&&learner.knowledge.version()==version&&learner.policy.weights==weights,"failed linked correction partially mutated state");
    auto wrong=learner.credits()[0].links;for(auto& link:wrong)if(link.satisfaction)link.satisfaction=false;
    check(learner.revise_mission_credit("mission",1,false,wrong),"mission correction failed");
    auto restored=learner.credits()[0].links;for(auto& link:restored)if(link.satisfaction)link.satisfaction=true;
    check(learner.revise_mission_credit("mission",2,true,restored)&&learner.policy.weights==weights&&learner.empirical_values()==values,"mission correction did not restore exact replay");
    auto withdrawn=learner.credits()[0].links;for(auto& link:withdrawn)link.active=false;
    check(learner.revise_mission_credit("mission",3,true,withdrawn,false)&&learner.policy.weights==initial_weights&&learner.empirical_values().empty(),"withdrawn mission still trains policy");
    check(!learner.knowledge.execution_receipts().at(segment[0].id).goal_verified,"withdrawal labeled interrupted segment");
    auto active=learner.credits()[0].links;for(auto& link:active)link.active=true;
    check(learner.revise_mission_credit("mission",4,true,active)&&learner.policy.weights==weights&&learner.empirical_values()==values,"mission restoration not exact");
    check(!learner.submit_mission_credit("duplicate",trace,{},true),"reused mission trace accepted");
    auto stale=learner;auto link=stale.credits()[0].links[0];
    check(stale.knowledge.revise_language(link.id,link.version,false,true,"verified_environment"),"external stale-link fixture failed");
    bool rejected=false;try{management(stale,ControlOp::next_goal);}catch(const std::logic_error&){rejected=true;}
    check(rejected,"uncoupled semantic revision silently reused old mission credit");
}
void five_action_mission_guards() {
    FiveActionMissionHarness learner;
    check(learner.begin("ordinary","unknown a",{"a"},{{{},true},1},{},false),"ordinary fast-path fixture rejected");
    learner.choose(0,-1,false,nullptr,ControlHint{ControlOp::finish,"mission_fixture"});
    check(learner.finish(true,false,4),"ordinary fast-path credit rejected");
    check(learner.begin("five_child","unknown a",{"a"},{{{},true},1},{},false),"five-action mission fixture rejected");
    learner.choose(0,-1,false,nullptr,ControlHint{ControlOp::finish,"mission_fixture"});
    auto trace=learner.trace();check(learner.close_episode_evidence(false).has_value(),"five-action mission close rejected");
    auto updates=learner.updates();auto weights=learner.policy.weights;auto version=learner.knowledge.version();
    check(updates>0,"overflow fixture requires a nonzero existing update count");
    const size_t overflow=std::numeric_limits<size_t>::max()-updates+1;
    check(!learner.submit_mission_credit("five_mission",trace,learner.pending_mission_evidence(),true,overflow)
          &&learner.updates()==updates&&learner.policy.weights==weights&&learner.knowledge.version()==version
          &&learner.credits().size()==1&&!learner.pending_mission_evidence().empty(),
          "overflow replay request bypassed mission update bound or mutated pending credit");
    check(learner.submit_mission_credit("five_mission",trace,learner.pending_mission_evidence(),true,4),"five-action mission credit rejected");
    // Protected mission APIs remain valid on five-action subclasses. The
    // legacy fast path must stop applying once one such mission is retained.
    for(bool withdrawn:{false,true}) {
        auto copy=learner;
        if(withdrawn) {
            auto links=copy.credits().back().links;for(auto& link:links)link.active=false;
            check(copy.revise_mission_credit("five_mission",1,true,links,false),"five-action mission withdrawal rejected");
        }
        const auto link=copy.credits().back().links.front();
        check(copy.knowledge.revise_language(link.id,link.version,true,link.active,"verified_environment"),"five-action stale evidence fixture failed");
        bool selection_rejected=false,consolidation_rejected=false;
        try{copy.select_control({}, {1,4});}catch(const std::logic_error&){selection_rejected=true;}
        try{copy.consolidate(1);}catch(const std::logic_error&){consolidation_rejected=true;}
        check(selection_rejected&&consolidation_rejected,"five-action mission or withdrawn clone bypassed coupled-evidence guard");
    }
}
int main(){import_and_selection();linked_mission();five_action_mission_guards();std::cout<<"Mission substrate: seven-action import/selection, one delayed multi-goal trace, unlabeled segments, atomic correction/withdrawal/restoration and legacy value shape passed\n";}
