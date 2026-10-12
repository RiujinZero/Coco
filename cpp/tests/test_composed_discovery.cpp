#include "laminar/composed_skill_discovery.h"
#include <iostream>
using namespace laminar::reasoning;
void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
// A unit fixture only. This known comparator is never used by acquisition runs.
struct ComparatorFixture {
    int choose(const Observation& o,const std::vector<int>&) {
        if(!o[1])return -1;
        if(o[16]||o[17])return 8;
        return o[4]?5:0;
    }
};
struct Withhold {int choose(const Observation&,const std::vector<int>&){return -1;}};
int main() {
    ProcedureMemory memory(0);memory.available_skills={false,true,false};ComparatorFixture comparator;
    VerifiedQuestion yes;yes.query.cue=91;yes.query.a={0,10};yes.query.b={0,10};yes.answer={1,1};
    auto learned=memory.learn_with({yes},comparator);
    check(learned.found&&learned.correct==1&&learned.total==1&&memory.procedures.count(91),"complete outcome proof was not installed");
    auto contradiction=yes;contradiction.answer={1,0};
    auto partial=memory.learn_with({yes,contradiction},comparator);
    check(!partial.found&&partial.program.empty()&&partial.correct==1&&partial.total==2&&!partial.best_program.empty(),"partial outcome evidence was not distinguished from proof");
    check(!memory.procedures.count(91),"failed replacement retained a stale procedure");
    auto before=memory.skill_steps;Withhold unavailable;
    check(memory.execute_with(yes.query,{8,9},unavailable)==Value{}&&memory.skill_steps==before,"unavailable proposal executed an invalid microinstruction");
    auto weights=memory.skills.kernel.routes.W;memory.learn_with({yes},comparator);
    check(memory.skills.kernel.routes.W==weights,"candidate procedure search changed neural weights");
    // Outcome retention must permit replacing a prior instruction sequence.
    // This explicit fixture tests rehearsal, not autonomous repair performance.
    PrimitiveStore prior;PrimitiveReceipt receipt;receipt.id="empty_count";receipt.cue=2;receipt.subject={0,0};receipt.answer={2,0};
    check(prior.teach(receipt),"repair fixture rejected");prior.rebuild(1000);
    check(prior.memory.available_skills[2],"empty-count prior unavailable");
    ComposedDiscoveryResult repaired;repaired.acquired=true;repaired.cue=2;repaired.repaired=true;
    MicroState state;state.task=2;state.relation=0;state.subject={0,0};
    for(int instruction:{0,1,9}) {
        auto obs=state.observe();repaired.rehearsal[policy_state_key(obs)]={obs,state.available(),instruction};state.step(instruction,{});
    }
    repaired.policy.seal(repaired.rehearsal);auto original=prior.memory.skills.kernel.routes.W;
    ComposedSkillDiscovery discovery;auto model=discovery.rehearse(prior,repaired,0);
    state={};state.task=2;state.relation=0;state.subject={0,0};
    for(int n=0;n<10&&!state.done;++n)state.step(model.skills.choose(state.observe(),state.available()),{});
    check(state.done&&state.output==receipt.answer&&prior.memory.skills.kernel.routes.W==original,"repair rehearsal lost a verified outcome or mutated the prior");
    auto wrong=repaired;wrong.rehearsal.clear();state={};state.task=2;state.relation=0;state.subject={0,0};
    for(int instruction:{4,9}){auto obs=state.observe();wrong.rehearsal[policy_state_key(obs)]={obs,state.available(),instruction};state.step(instruction,{});}
    wrong.policy.seal(wrong.rehearsal);bool rejected=false;
    try{discovery.rehearse(prior,wrong,0);}catch(const std::runtime_error&){rejected=true;}
    check(rejected&&prior.memory.skills.kernel.routes.W==original,"repair overwrote a verified primitive outcome");
    ComposedSkillDiscovery interleaved;interleaved.repair_available=true;interleaved.interleave_quantum=3;interleaved.evaluation_budget=9;
    VerifiedQuestion zero;zero.query.cue=73;zero.query.a={0,0};zero.query.b={0,1};zero.query.relation[0]=0;zero.query.relation[1]=1;zero.answer={2,0};
    std::ostringstream curve;auto shared=interleaved.learn(prior,{{zero}},0,curve);
    check(shared.acquired&&shared.cue==2&&shared.attempts.size()==3,"interleaved discovery lost a candidate slot");
    check(shared.attempts[0].evaluations==1&&shared.attempts[1].evaluations==1&&shared.attempts[2].evaluations==1,"interleaved discovery hid proposal costs or missed first fit");
    check(prior.memory.skills.kernel.routes.W==original,"interleaving changed live neural weights");
    auto memoized=interleaved;memoized.assessment_cache_entries=8;
    std::ostringstream memo_curve,memo_progress;memoized.progress.out=&memo_progress;
    auto cached=memoized.learn(prior,{{zero}},0,memo_curve);
    check(cached.acquired==shared.acquired&&cached.cue==shared.cue&&cached.programs==shared.programs&&cached.policy.rules==shared.policy.rules&&cached.policy.seed==shared.policy.seed,"memoized assessment changed accepted computation");
    check(memo_curve.str()==curve.str()&&cached.attempts.size()==shared.attempts.size(),"memoized assessment changed search trajectory");
    check(memo_progress.str().find("assessment_cache_totals\t2\t1\t")!=std::string::npos,"assessment cache did not exercise verified replay reuse");
    for(size_t i=0;i<cached.attempts.size();++i) {
        const auto& a=cached.attempts[i];const auto& b=shared.attempts[i];
        check(a.correct==b.correct&&a.total==b.total&&a.evaluations==b.evaluations&&a.proposal_steps==b.proposal_steps&&a.program_states==b.program_states,"memoization changed logical search accounting");
    }
    std::cout<<"whole-outcome proof, partial-score withholding, invalid proposal and frozen-search contracts passed\n";
}
