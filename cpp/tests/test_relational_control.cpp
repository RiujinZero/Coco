#include "laminar/relational_control.h"
#include <iostream>
using namespace laminar::relational;
void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
// Force operations only to exercise protocol contracts. Restore weights before
// credit; these unit choices are never training examples for the experiment.
ControlStep force(ControlledRelationalSession& learner,ControlOp op) {
    auto weights=learner.policy.weights;
    std::fill(learner.policy.weights.begin(),learner.policy.weights.end(),0);
    learner.policy.weights[learner.policy.output_bias()+int(op)]=1;
    auto step=learner.choose();learner.policy.weights=std::move(weights);return step;
}
int main() {
  {
    ControlledRelationalSession scaled(3,true,false,true,{true,1,32,true});
    Scene visible{{{"r",{"a","b"}}},false};
    check(scaled.begin("sizes","unknown a b",{"a","b"},{visible,0},{{"motor",{"a","b"}},{"motor",{"b","a"}}},true),"quantitative episode rejected");
    auto o=scaled.observe();
    check(std::abs(o[29]-std::log2(3.f)/std::log2(129.f))<1e-6f&&std::abs(o[30]-1.f/std::log2(257.f))<1e-6f,"quantitative observation does not describe visible inventory");
    scaled.cancel();
  }
  for(bool empirical:{false,true}) {
    ControlledRelationalSession learner(17,true,false,empirical);
    Scene before{{{"edge",{"a","old"}},{"edge",{"other","b"}}},true};
    Scene after=before;after.facts.push_back({"edge",{"a","b"}});after=sorted(after);
    laminar::concepts::Receipt c;c.id="yes";c.name="linked";c.args={"a","b"};c.scene=after;c.answer=true;
    check(learner.knowledge.teach_concept(c),"concept rejected");c.id="no";c.scene=before;c.answer=false;check(learner.knowledge.teach_concept(c),"concept contrast rejected");
    LanguageReceipt l;l.id="lang_yes";l.text="join a with b";l.names={"a","b","old","other"};l.scene=after;l.answer=true;
    check(learner.knowledge.teach_language(l),"language rejected");l.id="lang_no";l.scene=before;l.answer=false;check(learner.knowledge.teach_language(l),"language contrast rejected");
    TransitionReceipt t;t.id="observed";t.action={"motor",{"a","b"}};t.before=before;t.after=after;check(learner.knowledge.teach_transition(t),"effect rejected");
    std::vector<Action> actions{t.action};std::set<std::string> names{"a","b","old","other"};
    for(bool retain:{false,true}) {
        auto bounded=learner;bounded.preserve_search(retain);
        Scene path_yes=before;path_yes.facts.push_back({"edge",{"old","b"}});path_yes=sorted(path_yes);
        Scene path_no=path_yes;path_no.facts.erase(std::find(path_no.facts.begin(),path_no.facts.end(),laminar::concepts::Atom{"edge",{"a","old"}}));
        int lesson=0;
        for(const auto& world:std::vector<Scene>{path_yes,before,path_no}) {
            laminar::concepts::Receipt c;c.id="path_c"+std::to_string(lesson);c.name="reachable";c.args={"a","b"};c.scene=world;c.answer=lesson==0;
            check(bounded.knowledge.teach_concept(c),"search retention concept rejected");
            LanguageReceipt l;l.id="path_l"+std::to_string(lesson);l.text="route a toward b";l.names=names;l.scene=world;l.answer=lesson==0;
            check(bounded.knowledge.teach_language(l),"search retention language rejected");++lesson;
        }
        auto public_names=names;std::vector<Action> distractors;
        for(int i=0;i<12;++i){auto name="o"+std::to_string(i);public_names.insert(name);distractors.push_back({"motor",{"a",name}});}
        check(bounded.begin("allowance","route a toward b",public_names,{before,0},distractors,true),"search retention episode rejected");
        check(force(bounded,ControlOp::reason).status=="search_budget","search retention fixture did not exhaust search");
        auto allowance=bounded.observe()[19];auto read=force(bounded,ControlOp::ask);
        check(bounded.read(read.token,StampedScene{before,1},"verified_environment"),"unchanged revision rejected");
        check(!bounded.observe()[5]&&!bounded.observe()[9],"revision retained stale cache or plan");
        check(std::abs(bounded.observe()[19]-(retain?allowance:2.f/15.f))<1e-6f,"unchanged scene search allowance incorrect");
        read=force(bounded,ControlOp::ask);
        check(bounded.read(read.token,StampedScene{after,2},"verified_environment"),"changed observation rejected");
        check(std::abs(bounded.observe()[19]-2.f/15.f)<1e-6f,"changed facts retained search allowance");
        bounded.cancel();
    }
    check(learner.begin("missing","join a with b",names,{{{},false},0},actions,true),"episode rejected");
    auto no_answer=force(learner,ControlOp::finish);check(no_answer.reply==Answer::Unknown&&learner.knowledge.dirty(),"answer computed outside reason");
    check(learner.finish(false,false,8),"terminal credit rejected");
    check(learner.begin("task","join a with b",names,{before,0},actions,true),"new episode rejected");
    check(force(learner,ControlOp::act).rejected,"dirty-model action was issued");
    auto reasoning=force(learner,ControlOp::reason);check(reasoning.rebuilt&&!learner.plan().actions.empty(),"reason did not rebuild and plan");
    auto issued=force(learner,ControlOp::act);check(issued.actions.size()==1,"computed action not issued");
    check(!learner.actions(issued.token,{after},{true},1,"generated"),"untrusted observation accepted");
    check(learner.actions(issued.token,{after},{true},1),"action observation rejected");
    auto read=force(learner,ControlOp::retrieve);
    check(learner.read(read.token,StampedScene{before,0},"trusted_memory"),"stale read was not processed");
    check(learner.visible().scene==after&&learner.observe()[12]==1,"stale memory overwrote observed result");
    check(force(learner,ControlOp::reason).status=="interpreted","result reasoning failed");
    check(force(learner,ControlOp::finish).reply==Answer::True,"goal not reported");
    auto version=learner.knowledge.version();check(!learner.finish(true,true,64,false,"untrusted")&&learner.knowledge.version()==version,"untrusted outcome mutated knowledge");
    check(learner.finish(true,true,64),"coupled terminal outcome rejected");
    check(learner.knowledge.execution_receipts().size()==1&&learner.credits().size()==2&&learner.updates()==72,"coupled evidence inventory wrong");
    auto learned=learner.policy.weights;
    check(!learner.revise_credit("task",0,false,false),"stale correction accepted");
    check(learner.revise_credit("task",1,false,false)&&learner.policy.weights!=learned,"old neural credit did not revise");
    check(learner.revise_credit("task",2,true,true)&&learner.policy.weights==learned,"restoring reward did not exactly replay neural updates");
    learner.knowledge.rebuild();auto model=learner.knowledge.version();
    check(learner.begin("stale","join a with b",names,{before,0},actions,true),"stale test begin failed");
    force(learner,ControlOp::reason);auto pending=force(learner,ControlOp::act);
    check(learner.knowledge.revise_concept("yes",1,true),"external concept revision failed");
    check(!learner.actions(pending.token,{after},{true},1)&&learner.policy.weights==learned,"stale semantic decision accepted feedback");learner.cancel();
    check(learner.knowledge.version()==model+1,"cancel mutated knowledge");
    auto probe=learner;probe.set_learning(false);probe.knowledge.rebuild();auto frozen=probe.policy.weights;auto counts=probe.knowledge.execution_receipts().size();
    check(probe.begin("eval","join a with b",names,{before,0},actions,true),"probe begin failed");force(probe,ControlOp::reason);auto action=force(probe,ControlOp::act);
    check(probe.actions(action.token,{after},{true},1),"probe observation failed");force(probe,ControlOp::reason);force(probe,ControlOp::finish);
    check(probe.finish(true,true)&&probe.policy.weights==frozen&&probe.knowledge.execution_receipts().size()==counts,"evaluation learned");
    check(learner.policy.weights==learned&&learner.knowledge.execution_receipts().size()==1,"clone leaked");
    ControlledRelationalSession reactive(23,true,false,empirical,{true,1,32},empirical);reactive.knowledge=learner.knowledge;
    reactive.knowledge.rebuild();auto prior_version=reactive.knowledge.version();auto prior_count=reactive.knowledge.execution_receipts().size();
    check(reactive.begin("retry","join a with b",names,{before,0},actions,true),"reactive episode rejected");
    force(reactive,ControlOp::reason);auto attempt=force(reactive,ControlOp::act);
    check(attempt.actions.size()==1&&reactive.actions(attempt.token,{before},{false},1),"failed motor observation rejected");
    check(reactive.knowledge.version()==prior_version&&reactive.knowledge.execution_receipts().size()==prior_count,"intermediate outcome was used for learning");
    check(!reactive.actions(attempt.token,{after},{true},2),"duplicate callback accepted");
    auto external=force(reactive,ControlOp::ask);
    check(!reactive.read(external.token,StampedScene{after,2},"verified_environment")&&reactive.visible().scene==sorted(before),"external world change was attributed to an own action");
    check(reactive.read(external.token,StampedScene{before,1},"verified_environment"),"consistent observation rejected");
    force(reactive,ControlOp::reason);auto retry=force(reactive,ControlOp::act);
    check(retry.actions.size()==1&&reactive.actions(retry.token,{after},{true},2),"retry after failure was prevented");
    check(reactive.knowledge.version()==prior_version,"retry learned before terminal feedback");
    force(reactive,ControlOp::reason);check(force(reactive,ControlOp::finish).reply==Answer::True,"retry did not achieve goal");
    check(reactive.finish(true,true,64),"reactive terminal feedback failed");
    const auto& credit=reactive.credits().back();const auto& execution=reactive.knowledge.execution_receipts().at(credit.evidence);
    check(execution.actions.size()==2&&execution.before==sorted(before)&&execution.observed[0]==sorted(before)&&execution.observed[1]==after,"reactive evidence chain differs from issued/observed steps");
    check(!reactive.knowledge.transition_receipts().at(credit.evidence+"_step0").executed&&reactive.knowledge.transition_receipts().at(credit.evidence+"_step1").executed,"motor failure receipt was fabricated as execution");
    if(empirical)check(reactive.consolidate(3)>0,"policy consolidation did not run");
    auto reactive_weights=reactive.policy.weights;
    check(reactive.revise_credit("retry",1,false,false)&&reactive.revise_credit("retry",2,true,true)&&reactive.policy.weights==reactive_weights,"reactive credit revision failed exact replay");
    reactive.knowledge.rebuild();check(reactive.knowledge.predict(t.action,before)==after,"failed motor execution poisoned learned effects");
  }
    std::cout<<"explicit computation, stamped reads, coupled terminal learning, stale credit, exact neural replay and frozen-probe contracts passed\n";
}
