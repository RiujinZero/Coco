#pragma once
// Outcome-based finite policy induction. A supplied search algorithm proposes
// reusable decisions; its evaluator supplies only terminal correctness, cost,
// and public observations/actions encountered during execution.
#include "reasoning_runtime.h"
#include <map>
#include <random>
#include <functional>
#include <ostream>
#include <optional>
namespace laminar::reasoning {
inline uint32_t policy_state_key(const Observation& o) {
    uint32_t key=0;
    for(int i=0;i<32;++i){if(o[i]!=0&&o[i]!=1)throw std::invalid_argument("nonbinary policy feature");if(o[i])key|=uint32_t(1)<<i;}
    return key;
}
inline uint64_t policy_mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
struct DiscoveredPolicy {
    uint64_t seed=0;
    bool sealed=false;
    std::map<uint32_t,int> rules;
    int choose(const Observation& obs,const std::vector<int>& available)const {
        if(available.empty())return -1;
        auto key=policy_state_key(obs);auto it=rules.find(key);
        if(it!=rules.end()&&std::find(available.begin(),available.end(),it->second)!=available.end())return it->second;
        // Exploration is legal while proposing a candidate. Once accepted,
        // unobserved states and unavailable operations withhold, never guess.
        return sealed?-1:available[policy_mix(key^seed)%available.size()];
    }
    void seal(const std::map<uint32_t,Sample>& observed) {
        std::map<uint32_t,int> next;
        for(const auto& [key,d]:observed){
            if(key!=policy_state_key(d.obs)||std::find(d.available.begin(),d.available.end(),d.action)==d.available.end())throw std::invalid_argument("invalid observed policy decision");
            next[key]=d.action;
        }
        rules=std::move(next);sealed=true;
    }
};
struct PolicyAssessment {
    int correct=0,total=0,steps=0,timeouts=0;
    std::map<uint32_t,Sample> visited;
    uint64_t behavior=0;
    void fingerprint(){behavior=0;for(const auto& [key,d]:visited)behavior=policy_mix(behavior^policy_mix(key)^policy_mix(d.action+1));}
};
struct PolicyCandidate {DiscoveredPolicy policy;PolicyAssessment evaluation;};
struct PolicyDiscoveryResult {DiscoveredPolicy policy;PolicyAssessment evaluation;int evaluations=0,first_solution=-1;};
class PolicySearch {
    int evaluation_budget,population_budget,restart_every;
    bool stop_at_first,stopped=false;
    std::mt19937 rng;
    std::vector<PolicyCandidate> pool;
    std::set<uint64_t> behaviors;
    PolicyCandidate best;
    int best_correct=-1,expected_total=-1,evaluations=0,first_solution=-1;
    std::optional<DiscoveredPolicy> initial;
    bool better(const PolicyCandidate& a,const PolicyCandidate& b)const {
        return a.evaluation.correct>b.evaluation.correct||(best_correct==b.evaluation.total&&a.evaluation.correct==b.evaluation.correct&&a.evaluation.steps<b.evaluation.steps);
    }
public:
    PolicySearch(uint64_t seed,int budget,int population,int restart,bool first,const DiscoveredPolicy* warm)
        :evaluation_budget(budget),population_budget(population),restart_every(restart),stop_at_first(first),rng(seed+701) {
        if(evaluation_budget<1||population_budget<1||restart_every<1)throw std::invalid_argument("invalid discovery budget");
        if(warm)initial=*warm;
    }
    bool done()const{return stopped||evaluations==evaluation_budget;}
    bool solved()const{return first_solution>=0;}
    int count()const{return evaluations;}
    void advance(int proposals,const std::function<PolicyAssessment(DiscoveredPolicy&)>& evaluate,std::ostream& curve) {
        if(proposals<1)throw std::invalid_argument("invalid search quantum");
        int end=evaluations+std::min(proposals,evaluation_budget-evaluations);
        while(!done()&&evaluations<end) {
            int iteration=evaluations;
            if(iteration&&iteration%restart_every==0&&best_correct<best.evaluation.total){pool.clear();behaviors.clear();}
            PolicyCandidate next;
            if(iteration==0&&initial){next.policy=*initial;next.policy.sealed=false;}
            else if(pool.empty()||rng()%20==0)next.policy.seed=(uint64_t(rng())<<32)|rng();
            else {
                int parent=rng()%pool.size();for(int k=0;k<2;++k){int rival=rng()%pool.size();if(better(pool[rival],pool[parent]))parent=rival;}
                next.policy=pool[parent].policy;
                const auto& visited=pool[parent].evaluation.visited;
                if(!visited.empty())for(int n=1+int(rng()%4);n;--n){auto it=visited.begin();std::advance(it,rng()%visited.size());const auto& a=it->second.available;next.policy.rules[it->first]=a[rng()%a.size()];}
            }
            next.evaluation=evaluate(next.policy);next.evaluation.fingerprint();++evaluations;
            if(expected_total<0)expected_total=next.evaluation.total;
            if(next.evaluation.total<1||next.evaluation.total!=expected_total||next.evaluation.correct<0||next.evaluation.correct>next.evaluation.total||next.evaluation.steps<0||next.evaluation.timeouts<0||next.evaluation.timeouts>next.evaluation.total)throw std::invalid_argument("invalid terminal assessment");
            for(const auto& [key,d]:next.evaluation.visited)if(key!=policy_state_key(d.obs)||std::find(d.available.begin(),d.available.end(),d.action)==d.available.end())throw std::invalid_argument("invalid observed decision");
            if(best_correct<0||better(next,best)) {
                best_correct=next.evaluation.correct;best=next;
                curve<<iteration<<'\t'<<best_correct<<'\t'<<best.evaluation.total<<'\t'<<best.evaluation.steps<<'\n';curve.flush();
                if(best_correct==best.evaluation.total&&first_solution<0)first_solution=iteration;
            }
            if(stop_at_first&&first_solution>=0){stopped=true;break;}
            if(behaviors.count(next.evaluation.behavior))continue;
            if(int(pool.size())<population_budget){behaviors.insert(next.evaluation.behavior);pool.push_back(std::move(next));}
            else {
                int victim=rng()%pool.size();for(int k=0;k<3;++k){int other=rng()%pool.size();if(better(pool[victim],pool[other]))victim=other;}
                if(!better(pool[victim],next)){behaviors.erase(pool[victim].evaluation.behavior);behaviors.insert(next.evaluation.behavior);pool[victim]=std::move(next);}
            }
        }
    }
    PolicyDiscoveryResult result()const{return {best.policy,best.evaluation,evaluations,first_solution};}
};
struct PolicyDiscovery {
    int evaluation_budget=100000,population_budget=128,restart_every=20000;
    bool stop_at_first=false;
    PolicySearch start(uint64_t seed,const DiscoveredPolicy* initial=nullptr)const {
        return {seed,evaluation_budget,population_budget,restart_every,stop_at_first,initial};
    }
    // Feedback provides whole outcomes and visited public states. Pausing a
    // search retains its population and RNG, without introducing new targets.
    PolicyDiscoveryResult learn(uint64_t seed,const std::function<PolicyAssessment(DiscoveredPolicy&)>& evaluate,
                                std::ostream& curve,const DiscoveredPolicy* initial=nullptr)const {
        auto search=start(seed,initial);search.advance(evaluation_budget,evaluate,curve);return search.result();
    }
};
}
