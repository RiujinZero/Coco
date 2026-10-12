#include "laminar/composed_learner.h"
#include <iostream>
using namespace laminar::reasoning;
static void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
static PrimitiveReceipt direct(std::string id,int cue,Value subject,Value target,Value answer) {
    PrimitiveReceipt r;r.id=id;r.cue=cue;r.subject=subject;r.target=target;r.answer=answer;return r;
}
static std::vector<ComposedReceipt> count_lessons() {
    std::vector<ComposedReceipt> result;
    for(int left=0;left<3;++left)for(int right=0;right<3;++right)for(int copies=1;copies<=2;++copies) {
        ComposedReceipt r;r.id="count"+std::to_string(result.size());auto& q=r.question.query;
        q.cue=70;q.a={0,100};q.b={0,200};q.relation[0]=0;q.relation[1]=1;
        for(int n=0;n<left;++n)for(int i=0;i<copies;++i)q.facts.push_back({100,0,300+n});
        for(int n=0;n<right;++n)for(int i=0;i<copies;++i)q.facts.push_back({200,1,400+n});
        r.question.answer={1,left==right};result.push_back(std::move(r));
    }
    // Comparisons between two counts alone leave arbitrary injective numeric
    // encodings observationally equivalent. Typed numeric operands ground the
    // scale while the only teacher output remains a whole Boolean answer.
    for(int size=0;size<3;++size)for(int target=0;target<4;++target)for(int copies=1;copies<=2;++copies) {
        ComposedReceipt r;r.id="count"+std::to_string(result.size());auto& q=r.question.query;
        q.cue=72;q.a={0,100};q.b={2,target};q.relation[0]=0;q.relation[1]=1;
        for(int n=0;n<size;++n)for(int i=0;i<copies;++i)q.facts.push_back({100,0,300+n});
        r.question.answer={1,size==target};result.push_back(std::move(r));
    }
    return result;
}
static std::vector<ComposedReceipt> follow_lessons() {
    std::vector<ComposedReceipt> result;
    for(int depth:{1,2,3,12,32,48})for(int same=0;same<2;++same) {
        ComposedReceipt r;r.id="follow"+std::to_string(result.size());auto& q=r.question.query;
        q.cue=71;q.a={0,100};q.b={0,same?100+depth:500};q.relation[0]=0;q.relation[1]=1;
        for(int i=0;i<depth;++i)q.facts.push_back({100+i,0,101+i});
        r.question.answer={1,same};result.push_back(std::move(r));
    }
    return result;
}
static void outcomes(const ComposedLearner& learner,const std::vector<ComposedReceipt>& rows,const char* message) {
    for(const auto& r:rows)check(learner.answer(r.question.query)==r.question.answer,message);
}
static bool reference_feasible(const PrimitiveReceipt& r,int& steps) {
    // Evaluator-only capacity oracle. No reference decisions enter discovery,
    // rehearsal, policy warm starts or primitive teaching.
    MicroState vm;vm.task=r.cue;vm.relation=r.relation;vm.subject=r.subject;vm.target=r.target;
    for(steps=0;steps<composed_primitive_replay_limit&&!vm.done;++steps) {
        int action;
        if(!vm.focus.present())action=0;
        else if(r.cue==0)action=vm.exhausted?6:vm.value.present()?2:1;
        else if(vm.exhausted)action=9;
        else if(vm.last==3)action=4;
        else action=vm.value.present()&&!vm.scratch.seen.count(vm.value)?3:1;
        vm.step(action,r.facts);
    }
    return vm.done&&vm.output==r.answer;
}
int main() {
    PrimitiveStore priors(1);
    check(priors.teach(direct("follow_empty",0,{0,10},{},{0,10})),"follow prior rejected");
    auto one_hop=direct("follow_one",0,{0,10},{},{0,11});one_hop.facts={{10,0,11}};
    check(priors.teach(one_hop),"one-hop follow prior rejected");
    check(priors.teach(direct("count_empty",2,{0,10},{},{2,0})),"count prior rejected");
    for(int size=1;size<=2;++size) {
        auto r=direct("count_unique"+std::to_string(size),2,{0,10},{},{2,size});
        for(int i=0;i<size;++i)r.facts.push_back({10,0,20+i});
        check(priors.teach(r),"unique-record count prior rejected");
    }
    for(int type=0;type<3;++type)for(int same=0;same<2;++same)
        check(priors.teach(direct("equal"+std::to_string(type)+std::to_string(same),1,{type,0},{type,same?0:1},{1,same})),"comparison prior rejected");
    priors.rebuild(10000);
    check(priors.memory.available_skills==std::array<bool,3>{true,true,true},"underspecified prior acquisition failed");
    auto wrong_count=direct("probe_count",2,{0,10},{},{2,1});wrong_count.facts={{10,0,11},{10,0,11}};
    auto wrong_follow=direct("probe_follow",0,{0,10},{},{0,12});wrong_follow.facts={{10,0,11},{11,0,12}};
    check(!verifies_direct_outcome(priors.memory.skills,wrong_count)&&!verifies_direct_outcome(priors.memory.skills,wrong_follow),"both available priors must demonstrably be wrong");
    auto counts=count_lessons(),follows=follow_lessons();
    ComposedLearner learner(1);learner.enable_repair_bank();learner.configure_interleaving(100);
    check(learner.synchronize(priors,1,counts),"initial raw synchronization failed");
    bool first=learner.rebuild(5000);
    std::cerr<<"first accepted="<<first<<" discovered="<<learner.report().acquired<<" fit="<<learner.report().correct<<'/'<<learner.report().total<<'\n';
    check(first,"first whole-outcome repair failed");
    std::cerr<<"first repair cue="<<learner.report().cue<<" evaluations="<<learner.report().evaluations<<'\n'<<std::flush;
    check(learner.active_repairs()==std::set<int>{2},"first repair did not acquire count");outcomes(learner,counts,"first outcomes fail");
    const auto first_weights=learner.memory().skills.kernel.routes.W;
    ComposedLearner legacy(1);legacy.enable_repair();legacy.configure_interleaving(100);
    check(legacy.synchronize(priors,1,counts)&&legacy.rebuild(5000),"legacy first-repair control failed");
    check(legacy.memory().skills.kernel.routes.W==first_weights,"opt-in bank changed single-repair neural weights");
    auto all=counts;all.insert(all.end(),follows.begin(),follows.end());
    check(legacy.synchronize(priors,1,all)&&!legacy.rebuild(100),"legacy control unexpectedly fit both wrong slots");
    auto legacy_model=legacy.memory();check(!verifies_direct_outcome(legacy_model.skills,wrong_count),"legacy control did not reproduce lost first repair");
    check(learner.synchronize(priors,1,all),"second raw synchronization failed");
    bool second=learner.rebuild(5000);
    std::cerr<<"second accepted="<<second<<" discovered="<<learner.report().acquired<<" fit="<<learner.report().correct<<'/'<<learner.report().total<<" active="<<learner.active_repairs().size()<<'\n';
    for(const auto& attempt:learner.report().attempts)std::cerr<<"attempt cue="<<attempt.cue<<" fit="<<attempt.correct<<'/'<<attempt.total<<" evaluations="<<attempt.evaluations<<'\n';
    check(second,"second whole-outcome repair failed");
    std::cerr<<"second repair cue="<<learner.report().cue<<" evaluations="<<learner.report().evaluations<<'\n'<<std::flush;
    check(learner.active_repairs()==std::set<int>({0,2}),"second repair erased first bank entry");
    outcomes(learner,all,"accepted bank does not fit all outcomes");
    auto combined_weights=learner.memory().skills.kernel.routes.W;
    for(const auto& [id,r]:priors.receipts()){auto model=learner.memory();check(verifies_direct_outcome(model.skills,r),"repair changed direct prior outcome");}
    // Fresh primitive consequences are evaluator-only probes, not new teaching.
    auto model=learner.memory();
    bool counted=verifies_direct_outcome(model.skills,wrong_count),followed=verifies_direct_outcome(model.skills,wrong_follow);
    std::cerr<<"independent count="<<counted<<" follow="<<followed<<'\n';
    check(counted&&followed,"repaired policies failed independent held-out outcomes");
    int fresh_checked=0,infeasible=0,reference_max=0;
    for(int size:{0,1,4,8,16,24,32}) {
        for(int copies:{1,2,3}) {
            if(size==0&&copies>1)continue;
            auto r=direct("fresh_count",2,{0,700},{},{2,size});r.relation=17;
            for(int i=0;i<size;++i)for(int j=0;j<copies;++j)r.facts.push_back({700,17,800+i});
            int steps=0;if(!reference_feasible(r,steps)){++infeasible;continue;}reference_max=std::max(reference_max,steps);
            bool count_ok=verifies_direct_outcome(model.skills,r);if(!count_ok)std::cerr<<"failed count size="<<size<<" copies="<<copies<<'\n';
            check(count_ok,"count repair failed fresh sizes/multiplicity");++fresh_checked;
        }
        auto f=direct("fresh_follow",0,{0,900},{},{0,900+size});f.relation=23;for(int i=0;i<size;++i)f.facts.push_back({900+i,23,901+i});
        int steps=0;if(!reference_feasible(f,steps)){++infeasible;continue;}reference_max=std::max(reference_max,steps);
        bool follow_ok=verifies_direct_outcome(model.skills,f);if(!follow_ok)std::cerr<<"failed follow size="<<size<<'\n';
        check(follow_ok,"follow repair failed fresh lengths");++fresh_checked;
    }
    std::cerr<<"fresh primitive outcomes="<<fresh_checked<<" reference_infeasible="<<infeasible<<" reference_max_steps="<<reference_max<<'\n';
    check(fresh_checked==26&&infeasible==0,"fresh probe coverage changed");
    auto withdrawn=learner;
    for(const auto& r:follows)check(withdrawn.revise(r.id,1,r.question.answer,false),"follow support withdrawal rejected");
    check(withdrawn.rebuild(1)&&withdrawn.active_repairs()==std::set<int>{2},"withdrawing second support erased independently justified first repair");
    check(withdrawn.memory().skills.kernel.routes.W==first_weights,"surviving first bank differs from its original accepted model");
    outcomes(withdrawn,counts,"withdrawal damaged count outcomes");
    for(const auto& r:follows)check(withdrawn.revise(r.id,2,r.question.answer,true),"support restoration rejected");
    check(withdrawn.rebuild(1)&&withdrawn.memory().skills.kernel.routes.W==combined_weights,"support restoration lost exact combined checkpoint");
    auto conflict=learner;auto bad=follows.front();bad.id="contradiction";bad.question.answer.id=1-bad.question.answer.id;
    check(conflict.teach(bad)&&!conflict.rebuild(1)&&conflict.contradictory(),"contradiction accepted as solved");
    check(conflict.active_repairs()==std::set<int>{2},"contradiction erased unrelated supported repair");
    outcomes(conflict,counts,"contradiction damaged unrelated outcomes");
    for(const auto& r:follows)check(conflict.answer(r.question.query)==Value{},"contradictory query retained a program");
    check(conflict.revise("contradiction",1,bad.question.answer,false)&&conflict.rebuild(1),"contradiction withdrawal failed");
    check(conflict.memory().skills.kernel.routes.W==combined_weights,"correction did not restore exact combined weights");
    auto changed_priors=priors;auto direct_conflict=one_hop;direct_conflict.id="direct_contradiction";direct_conflict.answer={0,10};
    check(changed_priors.teach(direct_conflict),"contradictory direct evidence rejected instead of retained");changed_priors.rebuild(1000);
    check(!changed_priors.memory.available_skills[0],"contradictory direct prior still available");
    ComposedLearner blocked_first(1);blocked_first.enable_repair_bank();blocked_first.configure_interleaving(100);
    check(blocked_first.synchronize(changed_priors,2,counts),"contradictory baseline synchronization failed");
    check(!blocked_first.rebuild(100)&&blocked_first.report().acquired&&blocked_first.active_repairs().empty(),"empty bank installed a new repair without all direct outcomes fitting");
    auto direct_changed=learner;check(direct_changed.synchronize(changed_priors,2,all),"new direct generation rejected");
    check(!direct_changed.rebuild(1)&&direct_changed.active_repairs()==std::set<int>{2},"direct contradiction did not isolate affected repair");
    check(!direct_changed.memory().available_skills[0],"direct contradiction retained invalid follow policy");
    outcomes(direct_changed,counts,"direct contradiction erased independent count repair");
    auto only_count=direct_changed.memory();check(verifies_direct_outcome(only_count.skills,wrong_count),"direct contradiction lost repaired count semantics");
    check(changed_priors.revise("direct_contradiction",1,{0,10},false),"direct contradiction withdrawal rejected");changed_priors.rebuild(1000);
    check(direct_changed.synchronize(changed_priors,3,all)&&direct_changed.rebuild(1),"direct support restoration did not recover bank");
    outcomes(direct_changed,all,"direct restoration lost composed outcomes");
    bool rejected=false;try{learner.enable_repair_bank();}catch(const std::logic_error&){rejected=true;}check(rejected,"late bank configuration accepted");
    std::cout<<"two available wrong priors repaired successively from final composed outcomes; joint priors, retention and evidence lifecycle passed\n";
}
