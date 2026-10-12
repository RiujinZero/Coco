#include "laminar/relational_control.h"
#include <iostream>
using namespace laminar::relational;
void check(bool okay,const char* why){if(!okay)throw std::runtime_error(why);}
Scene add(Scene world,const std::string& predicate) {world.facts.push_back({predicate,{"a","b"}});return sorted(world);}
const Scene initial{{{"edge",{"a","old"}},{"edge",{"other","b"}},{"marked",{"a","old"}},{"marked",{"other","b"}}},true};
const Scene complete=add(add(initial,"edge"),"marked");
const std::set<std::string> names{"a","b","old","other"};
const std::vector<Action> actions{{"edge_motor",{"a","b"}},{"mark_motor",{"a","b"}}};
ControlStep force(ControlledRelationalSession& learner,ControlOp operation) {
    // Protocol tests choose explicit operations; these are not learning scores.
    auto weights=learner.policy.weights;std::fill(learner.policy.weights.begin(),learner.policy.weights.end(),0);
    learner.policy.weights[learner.policy.output_bias()+int(operation)]=1;
    auto result=learner.choose();learner.policy.weights=weights;return result;
}
ControlledRelationalSession fixture(size_t motors=32,bool episode_return=false) {
    ControlledRelationalSession learner(7,true,false,true,{true,1,motors},false,32,{},episode_return);
    for(int i=0;i<3;++i) {
        auto world=i==0?complete:i==1?add(initial,"edge"):add(initial,"marked");
        laminar::concepts::Receipt c;c.id="concept"+std::to_string(i);c.name="joined";c.args={"a","b"};c.scene=world;c.answer=i==0;
        check(learner.knowledge.teach_concept(c),"concept rejected");
        LanguageReceipt l;l.id="language"+std::to_string(i);l.text="join a with b";l.names=names;l.scene=world;l.answer=i==0;
        check(learner.knowledge.teach_language(l),"language rejected");
    }
    for(int i=0;i<2;++i) {
        TransitionReceipt t;t.id="effect"+std::to_string(i);t.action=actions[i];t.before=initial;t.after=add(initial,i?"marked":"edge");
        check(learner.knowledge.teach_transition(t),"effect rejected");
    }
    learner.knowledge.rebuild();return learner;
}
std::pair<Scene,uint64_t> start_parent(ControlledRelationalSession& learner,const std::string& id) {
    check(learner.begin(id,"join a with b",names,{initial,1},actions,true),"parent begin failed");
    force(learner,ControlOp::reason);auto step=force(learner,ControlOp::act);
    check(step.actions.size()==1&&!learner.suspend(),"waiting goal suspended or no action");
    auto world=add(initial,step.actions[0].schema=="edge_motor"?"edge":"marked");
    check(learner.actions(step.token,{world},{true},2),"parent observation rejected");
    return {world,step.token};
}
int main() {
    auto learner=fixture(32,true);auto [world,old_token]=start_parent(learner,"parent");
    auto weights=learner.policy.weights;auto version=learner.knowledge.version();auto prefix=learner.trace();
    check(learner.suspend()&&!learner.active()&&learner.suspended_goals()==std::vector<std::string>{"parent"},"goal not retained");
    check(learner.policy.weights==weights&&learner.credits().empty()&&learner.updates()==0,"suspension invented terminal credit");
    check(learner.knowledge.version()==version+1&&learner.knowledge.execution_receipts().size()==1,"observed segment not committed");
    const auto segment=learner.knowledge.execution_receipts().begin()->second;
    check(!segment.goal_verified&&!segment.success&&!learner.knowledge.language_receipts().count(segment.id),"interruption invented satisfaction");
    LanguageReceipt collision;collision.id=segment.id;collision.text="join a with b";collision.names=names;collision.scene=complete;collision.answer=true;
    check(!learner.knowledge.teach_language(collision),"teacher claimed an execution-owned identifier");
    check(!learner.actions(old_token,{complete},{true},3)&&!learner.begin("parent","join a with b",names,{world,2},actions,true),"old callback or duplicate goal accepted");
    check(!learner.resume("parent",{world,1})&&!learner.resume("parent",{world,3},"generated"),"invalid resume accepted");
    for(int i=0;i<16;++i) {
        check(learner.begin("child"+std::to_string(i),"join a with b",names,{complete,10},{},false),"child begin failed");
        check(!learner.resume("parent",{world,3}),"resume replaced active child");
        force(learner,ControlOp::reason);force(learner,ControlOp::finish);
        check(learner.finish(true,true,32)&&learner.policy.weights!=weights,"intervening goal did not learn");
    }
    check(!learner.empirical_values().count(prefix[0].observation),"suspended parent's unlabelled decisions entered credit learning");
    world.facts.push_back({"external",{"other","a"}});world=sorted(world);
    check(learner.resume("parent",{world,11})&&learner.suspended_goals().empty(),"parent not resumed");
    check(learner.trace().size()==prefix.size()&&learner.trace()[0].token==prefix[0].token&&!learner.observe()[5]&&!learner.observe()[9],"goal trace lost or old cache reused");
    check(!learner.actions(old_token,{complete},{true},12),"old motor callback resurrected");
    force(learner,ControlOp::reason);auto step=force(learner,ControlOp::act);
    check(step.actions.size()==1,"remaining goal not replanned");
    auto after=add(world,step.actions[0].schema=="edge_motor"?"edge":"marked");
    check(learner.actions(step.token,{after},{true},12),"new segment callback rejected");
    force(learner,ControlOp::reason);check(force(learner,ControlOp::finish).reply==Answer::True,"resumed goal not achieved");
    check(learner.finish(true,true,64),"delayed parent credit rejected");
    check(learner.credits().size()==17&&learner.credits()[0].id=="child0"&&learner.credits().back().id=="parent"&&learner.credits().back().trace.size()==6,"interleaved traces or terminal credit misattributed");
    check(learner.empirical_values().at(prefix[0].observation)[int(ControlOp::reason)]>0,"terminal parent reward did not reach its pre-interruption choice");
    const auto& final=learner.knowledge.execution_receipts().at(learner.credits().back().evidence);
    check(final.goal_verified&&final.before==world&&final.actions.size()==1&&final.observed.back()==after,"external change attributed to own motor");
    weights=learner.policy.weights;
    check(learner.revise_credit("parent",1,false,false)&&learner.revise_credit("parent",2,true,true)&&learner.policy.weights==weights,"delayed credit revision not exactly replayable");
    check(!learner.knowledge.execution_receipts().at(segment.id).goal_verified,"parent correction labelled earlier fragment");
    check(learner.knowledge.revise_execution_observations(segment.id,1,segment.before,segment.observed,{true},std::nullopt),"unknown segment correction rejected");
    check(!learner.knowledge.execution_receipts().at(segment.id).goal_verified&&!learner.knowledge.language_receipts().count(segment.id),"segment correction invented goal label");
    check(learner.knowledge.revise_execution(segment.id,2,std::nullopt,false)&&learner.knowledge.revise_execution(segment.id,3,std::nullopt,true),"unknown segment withdrawal/restoration failed");
    check(!learner.knowledge.revise_execution(final.id,final.version,std::nullopt),"omitted outcome erased verified terminal label");

    auto limited=fixture(1);auto [one,ignored]=start_parent(limited,"limited");
    check(limited.suspend()&&limited.resume("limited",{one,3}),"bounded resume failed");
    force(limited,ControlOp::reason);check(limited.observe()[20]==1&&force(limited,ControlOp::act).rejected,"suspension reset motor budget");limited.cancel();

    auto frozen=fixture();frozen.set_learning(false);auto frozen_version=frozen.knowledge.version();auto frozen_weights=frozen.policy.weights;
    auto [seen,token]=start_parent(frozen,"probe");
    check(frozen.suspend()&&frozen.knowledge.version()==frozen_version&&frozen.knowledge.execution_receipts().empty()&&frozen.policy.weights==frozen_weights,"frozen suspension learned");
    auto copy=frozen;check(copy.cancel_suspended("probe")&&!copy.resume("probe",{seen,3})&&frozen.suspended_goals().size()==1,"cancellation or copy isolation failed");
    check(frozen.resume("probe",{seen,3}),"frozen goal lost");frozen.cancel();
    std::cout<<"Goal suspension: separate observed segments, delayed credit, world changes, bounds, cancellation and frozen isolation passed\n";
}
