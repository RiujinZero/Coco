#include "laminar/policy_discovery.h"
#include <sstream>
#include <iostream>
using namespace laminar::reasoning;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main(){
    Observation x{},y{},novel{};x[0]=1;y[1]=1;novel[2]=1;
    PolicyDiscovery search;search.evaluation_budget=1000;
    auto assess=[&](DiscoveredPolicy& p){
        PolicyAssessment e;e.total=2;e.steps=4;
        int a=p.choose(x,{0,1}),b=p.choose(y,{0,1});
        // Both outputs are checked only after the two decisions have executed.
        e.correct=2*int(a==1&&b==0);
        e.visited[policy_state_key(x)]={x,{0,1},a};e.visited[policy_state_key(y)]={y,{0,1},b};return e;
    };
    std::ostringstream curve;auto result=search.learn(9,assess,curve);
    check(result.evaluation.correct==2&&result.evaluations==1000,"terminal policy induction failed");
    result.policy.seal(result.evaluation.visited);auto before=result.policy.rules;
    check(result.policy.choose(x,{0,1})==1&&result.policy.choose(y,{0,1})==0,"sealing changed observed decisions");
    check(result.policy.choose(novel,{0,1})==-1&&result.policy.choose(x,{0})==-1,"sealed policy guessed at missing state/operand");
    check(result.policy.rules==before,"inference changed rules");
    auto seed_policy=result.policy;seed_policy.rules[policy_state_key(x)]=0;auto untouched=seed_policy.rules;
    auto repaired=search.learn(9,assess,curve,&seed_policy);
    check(repaired.evaluation.correct==2&&seed_policy.rules==untouched&&seed_policy.sealed,"warm search failed to revise a policy or mutated the supplied prior");
    // A scheduler may share computation among hypotheses, but pausing must not
    // reset exploration or change the proposals of any individual search.
    search.restart_every=13;
    std::vector<int> whole_choices,chunked_choices;std::ostringstream whole_curve,chunked_curve;
    auto evaluator=[&](std::vector<int>& choices){return [&](DiscoveredPolicy& p){
        auto e=assess(p);e.total=3;choices.push_back(p.choose(x,{0,1})*2+p.choose(y,{0,1}));return e;
    };};
    auto whole=search.learn(9,evaluator(whole_choices),whole_curve,&seed_policy);
    auto continued=search.start(9,&seed_policy);
    for(int n:{7,311,1,81,600})continued.advance(n,evaluator(chunked_choices),chunked_curve);
    auto chunked=continued.result();
    check(continued.done()&&!continued.solved()&&whole_choices==chunked_choices&&whole_curve.str()==chunked_curve.str(),"resumed search changed proposal sequence or restart behavior");
    check(whole.policy.rules==chunked.policy.rules&&whole.policy.seed==chunked.policy.seed&&whole.evaluations==chunked.evaluations,"resumed search changed its accepted candidate");
    search.stop_at_first=true;auto stopping=search.start(9,&result.policy);
    stopping.advance(1000,assess,curve);auto count=stopping.count();stopping.advance(1000,assess,curve);
    check(count==1&&stopping.count()==count&&stopping.solved()&&stopping.done(),"finished search continued sampling");
    auto clone=result.policy;clone.rules[policy_state_key(x)]=0;check(result.policy.choose(x,{0,1})==1,"copy changed original policy");
    bool rejected=false;auto bad=result.evaluation.visited;bad.begin()->second.action=7;
    try{result.policy.seal(bad);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected&&result.policy.rules==before,"invalid seal was not atomic");
    rejected=false;try{search.learn(0,[](auto&){PolicyAssessment e;e.total=1;e.correct=2;return e;},curve);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"impossible fitness accepted");
    rejected=false;Observation continuous{};continuous[0]=0.25f;try{policy_state_key(continuous);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"unsupported observation accepted");
    std::cout<<"terminal policy search, atomic sealing, unknown-state withholding and clone contracts passed\n";
}
