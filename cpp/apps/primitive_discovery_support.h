#pragma once
// Evaluator fixtures shared by discovery/transfer probes, not production inference.
#include "laminar/policy_discovery.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <iomanip>
using namespace laminar::reasoning;
inline void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
struct Problem {int task,relation;Value subject,target;std::vector<Fact> facts;};
struct Example {Problem input;Value answer;int size;};
inline MicroState start(const Problem& p){MicroState v;v.task=p.task;v.relation=p.relation;v.subject=p.subject;v.target=p.target;return v;}
struct Rollout {Value output;int steps=0;bool timeout=false,unavailable=false;std::vector<Sample> decisions;};
template<class P> Rollout run(P& policy,const Problem& p,int budget,bool record=false,int disabled=-1) {
    auto vm=start(p);Rollout r;
    while(!vm.done&&r.steps<budget){
        auto obs=vm.observe();auto available=vm.available();std::erase(available,disabled);int a=policy.choose(obs,available);
        if(record)r.decisions.push_back({obs,available,a});
        if(a<0){r.unavailable=true;break;}
        check(std::find(available.begin(),available.end(),a)!=available.end(),"invalid policy action");
        if(disabled==-2&&a==1)vm.cursor=-1;
        vm.step(a,p.facts);++r.steps;
    }
    r.output=vm.output;r.timeout=!vm.done&&!r.unavailable;return r;
}
inline Example generate(std::mt19937& rng,int task,int size,int serial,int duplicate=2) {
    std::vector<int> ids(3*size+40);std::iota(ids.begin(),ids.end(),serial*1000);std::shuffle(ids.begin(),ids.end(),rng);
    Example e;e.size=size;e.input={task,int(rng()%17),{0,ids[0]},{0,ids[size+2]}, {}};
    auto& p=e.input;
    if(task==0){for(int i=0;i<size;++i)p.facts.push_back({ids[i],p.relation,ids[i+1]});e.answer={0,ids[size]};}
    else if(task==1){
        int type=rng()%3;
        p.subject={type,type==0?ids[0]:type==1?int(rng()%2):int(rng()%(size+5))};
        p.target=(rng()%2)?p.subject:Value{type,type==1?1-p.subject.id:p.subject.id+1};
        if(rng()%4==0)p.target={int((type+1)%3),p.subject.id};
        e.answer={1,p.subject==p.target};
    }else{
        for(int i=1;i<=size;++i)for(int j=0,n=1+int(rng()%(duplicate+1));j<n;++j)p.facts.push_back({ids[0],p.relation,ids[i]});
        e.answer={2,size};
    }
    for(int i=0;i<4;++i){p.facts.push_back({ids[size+3+i],p.relation,ids[size+12+i]});p.facts.push_back({ids[0],p.relation+31,ids[size+20+i]});}
    std::shuffle(p.facts.begin(),p.facts.end(),rng);return e;
}
inline PolicyAssessment assess(DiscoveredPolicy& p,const std::vector<Example>& examples,int budget,int credit_window=0) {
    PolicyAssessment a;a.total=int(examples.size());
    for(const auto& e:examples){
        auto r=run(p,e.input,budget,true);a.correct+=r.output==e.answer;a.steps+=r.steps;a.timeouts+=r.timeout;
        size_t begin=credit_window? r.decisions.size()-std::min(r.decisions.size(),size_t(credit_window)):0;
        for(size_t i=begin;i<r.decisions.size();++i){const auto& d=r.decisions[i];if(d.action>=0)a.visited.emplace(policy_state_key(d.obs),d);}
    }
    a.fingerprint();return a;
}
inline void save_policy(const DiscoveredPolicy& p,const std::string& path) {
    std::ofstream out(path);check(bool(out),"open policy");out<<p.seed<<' '<<p.sealed<<'\n';for(auto [key,action]:p.rules)out<<key<<' '<<action<<'\n';
}
inline DiscoveredPolicy load_policy(const std::string& path) {
    std::ifstream in(path);check(bool(in),"missing discovered policy");DiscoveredPolicy p;check(bool(in>>p.seed>>p.sealed),"policy header");uint32_t key;int a;
    while(in>>key>>a){check(a>=0&&a<10&&!p.rules.count(key),"policy record");p.rules[key]=a;}
    check(in.eof()&&p.sealed,"unsealed policy");return p;
}
struct PrimitiveTrace {
    std::ofstream cases,decisions;int serial=0;
    explicit PrimitiveTrace(const std::string& prefix):cases(prefix+".cases.tsv"),decisions(prefix+".decisions.tsv"){
        check(bool(cases)&&bool(decisions),"open primitive trace");
        cases<<"case\tphase\tdisabled\ttask\tsize\trelation\tsubject_type\tsubject\ttarget_type\ttarget\tfacts\tgold_type\tgold\toutput_type\toutput\tsteps\ttimeout\tunavailable\n";
        decisions<<"case\tstep\tstate\taction\n";
    }
    void add(const Example& e,const Rollout& r,const std::string& phase,int disabled=-1){
        const auto& p=e.input;cases<<serial<<'\t'<<phase<<'\t'<<disabled<<'\t'<<p.task<<'\t'<<e.size<<'\t'<<p.relation<<'\t'<<p.subject.type<<'\t'<<p.subject.id<<'\t'<<p.target.type<<'\t'<<p.target.id<<'\t';
        for(auto f:p.facts)cases<<f.subject<<','<<f.relation<<','<<f.object<<';';
        cases<<'\t'<<e.answer.type<<'\t'<<e.answer.id<<'\t'<<r.output.type<<'\t'<<r.output.id<<'\t'<<r.steps<<'\t'<<r.timeout<<'\t'<<r.unavailable<<'\n';
        for(size_t i=0;i<r.decisions.size();++i)decisions<<serial<<'\t'<<i<<'\t'<<policy_state_key(r.decisions[i].obs)<<'\t'<<r.decisions[i].action<<'\n';
        ++serial;
    }
};
template<class P> void evaluate(P& policy,int seed,int task,const std::string& phase,PrimitiveTrace& trace,bool ablate=true) {
    std::mt19937 rng(5000+seed);std::map<int,std::pair<int,int>> score;
    for(int size:{0,1,2,3,4,8,12,24,64})for(int i=0;i<100;++i){
        auto e=generate(rng,task,size,100000+task*10000+size*100+i,4);auto r=run(policy,e.input,1024,true);trace.add(e,r,phase);
        auto& s=score[size];s.first+=r.output==e.answer;++s.second;
    }
    std::cout<<"seed="<<seed<<" task="<<task<<" phase="<<phase;for(auto [size,s]:score)std::cout<<" size"<<size<<'='<<s.first<<'/'<<s.second;std::cout<<'\n';
    if(ablate)for(int disabled:{0,1,2,3,4,5,-2})for(int size:{4,24})for(int i=0;i<10;++i){
        auto e=generate(rng,task,size,140000+task*10000+size*100+i,4);auto r=run(policy,e.input,1024,true,disabled);trace.add(e,r,phase,disabled);
    }
}
