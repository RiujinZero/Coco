#pragma once
// Versioned primitive evidence and outcome-based discovery in a live learner.
// No instruction demonstrations or pretrained policy files are accepted.
#include "policy_discovery.h"
#include "procedure_memory.h"
#include <sstream>

namespace laminar::reasoning {
struct PrimitiveReceipt {
    std::string id,source="verified_teacher";
    int cue=0,relation=0;
    Value subject,target,answer;
    std::vector<Fact> facts;
    bool active=true,generated=false;
    unsigned version=1;
};
struct PrimitiveAcquisition {
    bool acquired=false;
    int correct=0,total=0,evaluations=0,first_solution=-1;
    uint64_t search_steps=0;
    size_t rehearsed_states=0;
};
class PrimitiveStore {
    unsigned seed_;
    std::map<std::string,PrimitiveReceipt> receipts_;
    std::array<DiscoveredPolicy,3> policies_;
    std::array<bool,3> changed_{false,false,false};
    std::array<PrimitiveAcquisition,3> acquisition_;
    std::vector<float> initial_;
    bool dirty_=false;
public:
    template<class Ar> void state_io(Ar& ar){ar(seed_);ar(receipts_);ar(policies_);ar(changed_);ar(acquisition_);ar(initial_);ar(dirty_);}
private:
    static constexpr size_t capacity_=4096;
    static bool value_ok(const Value& v,bool absent=false) {
        return (absent&&v.type==-1)||(v.type>=0&&v.type<=2&&v.id>=0&&(v.type!=1||v.id<=1));
    }
    static MicroState start(const PrimitiveReceipt& r) {
        MicroState v;v.task=r.cue;v.relation=r.relation;v.subject=r.subject;v.target=r.target;return v;
    }
    template<class Policy> static Value run(Policy& p,const PrimitiveReceipt& r,int budget,std::vector<Sample>* trace=nullptr) {
        auto v=start(r);
        for(int n=0;n<budget&&!v.done;++n) {
            auto obs=v.observe();auto available=v.available();int a=p.choose(obs,available);
            if(a<0||std::find(available.begin(),available.end(),a)==available.end())return {};
            if(trace)trace->push_back({obs,available,a});
            v.step(a,r.facts);
        }
        return v.done?v.output:Value{};
    }
public:
    ProcedureMemory memory;
    explicit PrimitiveStore(unsigned seed=0,SkillShape shape={}):seed_(seed),memory(seed,shape) {
        memory.available_skills.fill(false);initial_=memory.skills.kernel.routes.W;
    }
    const auto& receipts()const{return receipts_;}
    const auto& policies()const{return policies_;}
    const auto& acquisition()const{return acquisition_;}
    bool dirty()const{return dirty_;}
    bool teach(PrimitiveReceipt r) {
        if(r.id.empty()||r.id.size()>128||receipts_.count(r.id)||receipts_.size()>=capacity_||r.version!=1||!r.active||r.generated||r.source!="verified_teacher"
           ||r.cue<0||r.cue>2||r.relation<0||r.facts.size()>512||!value_ok(r.subject)||!value_ok(r.target,true)||!value_ok(r.answer))return false;
        for(const auto& f:r.facts)if(f.subject<0||f.relation<0||f.object<0)return false;
        changed_[r.cue]=true;memory.available_skills[r.cue]=false;receipts_.emplace(r.id,std::move(r));dirty_=true;memory.procedures.clear();return true;
    }
    bool revise(const std::string& id,unsigned expected,Value answer,bool active=true,const std::string& source="verified_teacher") {
        auto found=receipts_.find(id);
        if(found==receipts_.end()||found->second.version!=expected||expected==~0u||source!=found->second.source||!value_ok(answer))return false;
        auto& r=found->second;r.answer=answer;r.active=active;++r.version;changed_[r.cue]=true;memory.available_skills[r.cue]=false;dirty_=true;memory.procedures.clear();return true;
    }
    void rebuild(int evaluation_budget=100000,bool rehearsal=true) {
        if(!dirty_)return;
        if(evaluation_budget<1)throw std::invalid_argument("invalid primitive discovery budget");
        for(int cue=0;cue<3;++cue)if(changed_[cue]) {
            std::vector<const PrimitiveReceipt*> examples;
            for(const auto& [id,r]:receipts_)if(r.active&&r.cue==cue)examples.push_back(&r);
            acquisition_[cue]={};policies_[cue]={};memory.available_skills[cue]=false;
            if(examples.empty())continue;
            auto assess=[&](DiscoveredPolicy& policy) {
                PolicyAssessment a;a.total=int(examples.size());
                for(const auto* r:examples) {
                    std::vector<Sample> trace;auto output=run(policy,*r,128,&trace);
                    a.correct+=output==r->answer;a.steps+=int(trace.size());a.timeouts+=!output.present();
                    for(const auto& s:trace)a.visited.emplace(policy_state_key(s.obs),s);
                }
                acquisition_[cue].search_steps+=uint64_t(a.steps);return a;
            };
            PolicyDiscovery discovery;discovery.evaluation_budget=evaluation_budget;std::ostringstream curve;
            auto result=discovery.learn(seed_+unsigned(cue)*1009,assess,curve);
            auto& report=acquisition_[cue];report.correct=result.evaluation.correct;report.total=result.evaluation.total;
            report.evaluations=result.evaluations;report.first_solution=result.first_solution;
            if(report.correct==report.total) {
                auto verified=assess(result.policy);result.policy.seal(verified.visited);policies_[cue]=std::move(result.policy);report.acquired=true;
            }
        }
        // Neural rehearsal is regenerated from the current raw evidence and
        // accepted policies. Retraction cannot leave old instruction targets.
        memory.skills.kernel.routes.W=initial_;memory.procedures.clear();
        std::map<uint32_t,Sample> bank;
        for(int cue=0;cue<3;++cue) {
            acquisition_[cue].rehearsed_states=0;
            memory.available_skills[cue]=acquisition_[cue].acquired;
            if(!acquisition_[cue].acquired||(!rehearsal&&!changed_[cue]))continue;
            size_t before=bank.size();
            for(const auto& [id,r]:receipts_)if(r.active&&r.cue==cue) {
                std::vector<Sample> trace;
                if(run(policies_[cue],r,128,&trace)!=r.answer)throw std::runtime_error("unverified self-generated rehearsal");
                for(const auto& s:trace) {
                    auto key=policy_state_key(s.obs);auto old=bank.find(key);
                    if(old!=bank.end()&&old->second.action!=s.action)throw std::runtime_error("conflicting self-generated decisions");
                    bank[key]=s;
                }
            }
            acquisition_[cue].rehearsed_states=bank.size()-before;
        }
        std::vector<Sample> fit;for(const auto& [key,s]:bank)fit.push_back(s);
        std::mt19937 rng(seed_+8100);
        for(int epoch=0;epoch<128;++epoch) {
            std::shuffle(fit.begin(),fit.end(),rng);for(const auto& s:fit)memory.skills.fit(s);
        }
        for(int cue=0;cue<3;++cue)if(memory.available_skills[cue]) {
            for(const auto& [id,r]:receipts_)if(r.active&&r.cue==cue&&run(memory.skills,r,128)!=r.answer) {memory.available_skills[cue]=false;break;}
        }
        changed_.fill(false);dirty_=false;
    }
};
} // namespace laminar::reasoning
