#pragma once
// Versioned whole-question evidence over the existing native primitive learner.
// Candidate discovery/rehearsal is private; callers provide facts and final
// answers, never a missing cue, micro policy or macro program.
#include "composed_skill_discovery.h"
#include <optional>

namespace laminar::reasoning {
struct ComposedReceipt {
    std::string id,source="verified_teacher";
    VerifiedQuestion question;
    bool active=true,generated=false,complete=true;
    unsigned version=1;
};
class ComposedLearner {
    unsigned seed_;
    PrimitiveStore priors_;
    mutable ProcedureMemory memory_;
    std::optional<ProcedureMemory> cache_;
    std::map<std::string,ComposedReceipt> receipts_,witnesses_;
    uint64_t version_=0,prior_generation_=0,cache_prior_generation_=0;
    bool dirty_=false,solved_=true;
    ComposedDiscoveryResult report_;
    bool reused_=false,contradictory_=false;
    bool synchronized_=false;
    bool repair_available_=false;
    int interleave_quantum_=0;
    SearchProgress progress_;
    int cache_cue_=-1;
    std::map<uint32_t,Sample> cache_rehearsal_;
    bool repair_bank_enabled_=false;
    RepairBank cache_bank_;
    std::set<int> active_repairs_;
    std::set<int> conflicting_cues_;
public:
    // progress_ is only an optional log sink, not learning state. The cached ProcedureMemory has no default
    // constructor, so it is rebuilt from the learner's seed before its state is read.
    template<class Ar> void state_io(Ar& ar) {
        ar(seed_);ar(priors_);ar(memory_);
        uint8_t has=cache_.has_value();ar(has);
        if constexpr(Ar::loading){if(has){cache_.emplace(seed_);ar(*cache_);}else cache_.reset();}
        else if(has)ar(*cache_);
        ar(receipts_);ar(witnesses_);ar(version_);ar(prior_generation_);ar(cache_prior_generation_);ar(dirty_);ar(solved_);ar(report_);
        ar(reused_);ar(contradictory_);ar(synchronized_);ar(repair_available_);ar(interleave_quantum_);ar(cache_cue_);ar(cache_rehearsal_);
        ar(repair_bank_enabled_);ar(cache_bank_);ar(active_repairs_);ar(conflicting_cues_);
    }
private:
    static bool value_ok(Value v) {return v.type>=0&&v.type<=2&&v.id>=0&&(v.type!=1||v.id<=1);}
    static bool query_ok(const FlatQuery& q) {
        if(q.cue<0||!value_ok(q.a)||!value_ok(q.b)||q.relation[0]<0||q.relation[1]<0||q.facts.size()>512||q.knowledge)return false;
        for(const auto& f:q.facts)if(f.subject<0||f.relation<0||f.object<0)return false;
        return true;
    }
    static std::vector<int> input_key(const FlatQuery& q) {
        std::vector<int> k{q.cue,q.a.type,q.a.id,q.b.type,q.b.id,q.relation[0],q.relation[1]};
        for(const auto& f:q.facts){k.push_back(f.subject);k.push_back(f.relation);k.push_back(f.object);}return k;
    }
    static bool same_content(const ComposedReceipt& a,const ComposedReceipt& b) {
        return a.active&&b.active&&a.source==b.source&&a.question.answer==b.question.answer&&input_key(a.question.query)==input_key(b.question.query);
    }
    bool compile(ProcedureMemory& memory,const std::vector<std::vector<VerifiedQuestion>>& groups)const {
        auto previous=memory.procedures;memory.procedures.clear();bool all=true;
        // Compilation freezes weights. A public VM observation and its legal
        // actions fully determine this feed-forward policy's choice. Keep the
        // cache local to this compilation so rehearsal cannot leave stale hits.
        struct FrozenChoices {
            LocalChoice& policy;
            std::map<std::pair<uint32_t,std::vector<int>>,int> cache;
            int choose(const Observation& obs,const std::vector<int>& available) {
                auto key=std::make_pair(policy_state_key(obs),available);
                auto found=cache.find(key);if(found!=cache.end())return found->second;
                int action=policy.choose(obs,available);cache.emplace(std::move(key),action);return action;
            }
        } choices{memory.skills,{}};
        for(const auto& rows:groups) {
            int cue=rows.front().query.cue;auto found=previous.find(cue);bool valid=found!=previous.end();
            if(valid)for(const auto& e:rows)if(memory.execute(e.query,found->second)!=e.answer){valid=false;break;}
            if(valid)memory.procedures[cue]=found->second;
            else {
                progress_.emit("compile_begin",cue);
                auto steps=memory.skill_steps;
                auto result=interleave_quantum_?memory.learn_with(rows,choices,false,10,50000,&progress_):memory.learn(rows,false,10,50000,&progress_);
                progress_.emit("compile_end",cue,0,result.states,memory.skill_steps-steps);
                if(!result.found)all=false;
            }
        }
        return all;
    }
    bool direct_fit(ProcedureMemory& model,bool available_only=false)const {
        for(const auto& [id,r]:priors_.receipts())if(r.active&&(!available_only||model.available_skills[r.cue])) {
            if(!verifies_direct_outcome(model.skills,r))return false;
        }
        return true;
    }
    bool supported_rehearsal()const {
        if(!cache_||cache_cue_<0||cache_rehearsal_.empty())return false;
        auto model=*cache_;std::set<uint32_t> covered;
        struct Cover {
            LocalChoice& policy;int cue;const std::map<uint32_t,Sample>& required;
            std::set<uint32_t>& covered;
            int choose(const Observation& obs,const std::vector<int>& available) {
                int action=policy.choose(obs,available);auto key=policy_state_key(obs);
                auto found=required.find(key);
                if(obs[cue]&&found!=required.end()&&found->second.action==action)covered.insert(key);
                return action;
            }
        } choices{model.skills,cache_cue_,cache_rehearsal_,covered};
        // Revalidate only unchanged original outcome witnesses. A changed or
        // retracted answer cannot justify its own old rehearsal targets.
        for(const auto& [id,r]:witnesses_) {
            auto current=receipts_.find(id);
            if(current==receipts_.end()||!same_content(r,current->second))continue;
            auto program=model.procedures.find(r.question.query.cue);
            if(program==model.procedures.end()||model.execute_with(r.question.query,program->second,choices)!=r.question.answer)return false;
        }
        if(covered.empty())return false;
        // Available-skill repair also includes verified direct outcomes in its
        // accepted rehearsal bank. Prior generation is checked by the caller.
        for(const auto& [id,r]:priors_.receipts())if(r.active&&r.cue==cache_cue_) {
            if(!verifies_direct_outcome(choices,r))return false;
        }
        return covered.size()==cache_rehearsal_.size();
    }
    RepairBank covered_repairs(ProcedureMemory& model,const RepairBank& bank,
                               const std::map<std::string,ComposedReceipt>& witnesses)const {
        std::map<int,std::set<uint32_t>> covered,composed;
        struct Cover {
            LocalChoice& policy;const RepairBank& required;std::map<int,std::set<uint32_t>> visited;
            int choose(const Observation& obs,const std::vector<int>& available) {
                int action=policy.choose(obs,available);auto key=policy_state_key(obs);
                for(const auto& [cue,repair]:required)if(obs[cue]) {
                    auto found=repair.rehearsal.find(key);
                    if(found!=repair.rehearsal.end()&&found->second.action==action&&found->second.available==available)visited[cue].insert(key);
                }
                return action;
            }
        } choices{model.skills,bank,{}};
        auto credit=[&](bool is_composed) {
            for(const auto& [cue,states]:choices.visited) {
                covered[cue].insert(states.begin(),states.end());
                if(is_composed)composed[cue].insert(states.begin(),states.end());
            }
        };
        for(const auto& [id,r]:witnesses) {
            if(conflicting_cues_.count(r.question.query.cue))continue;
            auto current=receipts_.find(id);if(current==receipts_.end()||!same_content(r,current->second))continue;
            auto program=model.procedures.find(r.question.query.cue);if(program==model.procedures.end())continue;
            choices.visited.clear();
            if(model.execute_with(r.question.query,program->second,choices)==r.question.answer)credit(true);
        }
        std::set<int> direct_failed;
        for(const auto& [id,r]:priors_.receipts())if(r.active&&bank.count(r.cue)) {
            choices.visited.clear();if(verifies_direct_outcome(choices,r))credit(false);else direct_failed.insert(r.cue);
        }
        RepairBank result;
        for(const auto& [cue,repair]:bank)if(!composed[cue].empty()&&!direct_failed.count(cue)&&covered[cue].size()==repair.rehearsal.size())result.emplace(cue,repair);
        return result;
    }
    std::pair<ProcedureMemory,RepairBank> supported_bank()const {
        if(!cache_||cache_bank_.empty())return {priors_.memory,{}};
        RepairBank bank=cache_bank_;
        // New direct evidence can refute one cached cue without erasing the
        // independently supported repairs in other cues.
        for(auto it=bank.begin();it!=bank.end();) {
            bool fits=true;for(const auto& [id,r]:priors_.receipts())if(r.active&&r.cue==it->first&&!verifies_direct_outcome(it->second.policy,r)){fits=false;break;}
            if(!fits)it=bank.erase(it);else ++it;
        }
        if(bank.empty())return {priors_.memory,{}};
        ComposedSkillDiscovery discovery;auto model=*cache_;
        if(bank.size()!=cache_bank_.size()||cache_prior_generation_!=prior_generation_) {
            // Rehearsal reads the store but does not mutate it. Preserve the
            // old programs solely as replay witnesses, never as assumed truth.
            auto priors=priors_;model=discovery.rehearse_bank(priors,bank,seed_);model.procedures=cache_->procedures;
        }
        for(size_t pass=0;pass<=cache_bank_.size();++pass) {
            auto supported=covered_repairs(model,bank,witnesses_);
            if(supported.size()==bank.size()) {
                if(direct_fit(model,true))return {std::move(model),std::move(bank)};
                return {priors_.memory,{}};
            }
            bank=std::move(supported);if(bank.empty())return {priors_.memory,{}};
            auto priors=priors_;model=discovery.rehearse_bank(priors,bank,seed_);model.procedures=cache_->procedures;
        }
        throw std::logic_error("repair support did not converge");
    }
    bool rebuild_bank(const std::vector<std::vector<VerifiedQuestion>>& groups,int evaluation_budget) {
        auto [base,bank]=supported_bank();memory_=base;
        for(const auto& [cue,repair]:bank)active_repairs_.insert(cue);
        if(compile(memory_,groups)&&direct_fit(memory_)) {
            reused_=!bank.empty();dirty_=false;solved_=!contradictory_;
            // Keep the full accepted checkpoint through partial withdrawal.
            // Restoration can then recover the exact earlier neural weights.
            if(cache_&&!bank.empty()&&bank.size()==cache_bank_.size())cache_=memory_;
            return solved_;
        }
        if(contradictory_){dirty_=false;return false;}
        ComposedSkillDiscovery discovery;discovery.evaluation_budget=evaluation_budget;discovery.repair_available=true;
        discovery.interleave_quantum=interleave_quantum_;discovery.progress=progress_;if(interleave_quantum_)discovery.assessment_cache_entries=128;
        std::ostringstream curve;progress_.emit("discovery_begin");
        report_=discovery.learn(priors_,groups,seed_,curve,bank.empty()?nullptr:&base,bank.empty()?nullptr:&bank);
        progress_.emit("discovery_end");
        if(report_.acquired) {
            progress_.emit("rehearsal_begin");auto candidate=discovery.rehearse(priors_,report_,seed_,true,&bank);
            progress_.emit("rehearsal_end");
            RepairBank next=bank;next[report_.cue]={report_.policy,report_.rehearsal};
            if(compile(candidate,groups)&&direct_fit(candidate)&&covered_repairs(candidate,next,receipts_).size()==next.size()) {
                memory_=std::move(candidate);cache_=memory_;cache_bank_=std::move(next);cache_prior_generation_=prior_generation_;
                cache_cue_=report_.cue;cache_rehearsal_=report_.rehearsal;
                witnesses_.clear();for(const auto& [id,r]:receipts_)if(r.active)witnesses_[id]=r;
                active_repairs_.clear();for(const auto& [cue,repair]:cache_bank_)active_repairs_.insert(cue);
                dirty_=false;solved_=true;return true;
            }
        }
        dirty_=false;return false;
    }
    void changed(){++version_;dirty_=true;}
public:
    explicit ComposedLearner(unsigned seed=0,SkillShape shape={}):seed_(seed),priors_(seed,shape),memory_(seed,shape){memory_.available_skills.fill(false);}
    const auto& receipts()const{return receipts_;}
    const auto& priors()const{return priors_;}
    const auto& memory()const{return memory_;}
    const auto& report()const{return report_;}
    uint64_t version()const{return version_;}
    bool dirty()const{return dirty_;}
    bool solved()const{return !dirty_&&solved_;}
    bool reused()const{return reused_;}
    bool contradictory()const{return contradictory_;}
    const auto& active_repairs()const{return active_repairs_;}
    void search_progress(std::ostream* out){progress_.out=out;}
    void configure_interleaving(int quantum) {
        if(quantum<0||version_||synchronized_)throw std::invalid_argument("interleaving must be configured before learning");
        interleave_quantum_=quantum;
    }
    void enable_repair() {
        if(version_||synchronized_)throw std::logic_error("repair must be configured before composed learning");
        repair_available_=true;
    }
    void enable_repair_bank() {
        if(version_||synchronized_)throw std::logic_error("repair bank must be configured before composed learning");
        repair_available_=repair_bank_enabled_=true;
    }
    // Adapter for a host's already-validated, versioned raw journal. The host
    // advances generation only for direct primitive evidence; its live memory
    // may contain derived skills, so it must not replace our direct baseline
    // when that generation is unchanged. The supplied store must be the host's
    // direct-evidence learner, not a restored external policy checkpoint.
    bool synchronize(const PrimitiveStore& priors,uint64_t generation,const std::vector<ComposedReceipt>& evidence) {
        if(priors.dirty())throw std::logic_error("synchronizing dirty direct priors");
        std::map<std::string,ComposedReceipt> next;
        for(const auto& r:evidence) {
            if(r.id.empty()||r.id.size()>128||r.generated||!r.complete||!r.active||!r.version
               ||(r.source!="verified_teacher"&&r.source!="verified_environment")
               ||!query_ok(r.question.query)||!value_ok(r.question.answer)||!next.emplace(r.id,r).second)return false;
        }
        if(next.size()>4096)return false;
        bool different=next.size()!=receipts_.size();
        if(!different)for(const auto& [id,r]:next) {
            auto old=receipts_.find(id);
            if(old==receipts_.end()||old->second.version!=r.version||!same_content(old->second,r)){different=true;break;}
        }
        if(!synchronized_||generation!=prior_generation_) {
            priors_=priors;prior_generation_=generation;synchronized_=true;different=true;
        }
        if(different){receipts_=std::move(next);changed();}
        return true;
    }
    bool teach_primitive(PrimitiveReceipt r) {
        if(!priors_.teach(std::move(r)))return false;
        ++prior_generation_;changed();return true;
    }
    bool revise_primitive(const std::string& id,unsigned expected,Value answer,bool active=true,const std::string& source="verified_teacher") {
        if(!priors_.revise(id,expected,answer,active,source))return false;
        ++prior_generation_;changed();return true;
    }
    bool teach(ComposedReceipt r) {
        if(r.id.empty()||r.id.size()>128||receipts_.count(r.id)||receipts_.size()>=4096||r.source!="verified_teacher"||r.generated||!r.complete||!r.active||r.version!=1
           ||!query_ok(r.question.query)||!value_ok(r.question.answer))return false;
        receipts_[r.id]=std::move(r);changed();return true;
    }
    bool revise(const std::string& id,unsigned expected,Value answer,bool active=true,const std::string& source="verified_teacher") {
        auto found=receipts_.find(id);
        if(found==receipts_.end()||found->second.version!=expected||expected==~0u||source!=found->second.source||!value_ok(answer))return false;
        auto& r=found->second;r.question.answer=answer;r.active=active;++r.version;changed();return true;
    }
    Value answer(const FlatQuery& q)const {return dirty_||!query_ok(q)?Value{}:memory_.answer(q);}
    bool rebuild(int evaluation_budget=100000) {
        if(!dirty_)return solved_;
        if(evaluation_budget<1)throw std::invalid_argument("invalid composed search budget");
        progress_.emit("composed_begin");
        priors_.rebuild();memory_=priors_.memory;report_={};reused_=contradictory_=solved_=false;active_repairs_.clear();conflicting_cues_.clear();
        std::map<int,std::vector<VerifiedQuestion>> by_cue;std::map<std::vector<int>,Value> facts;
        for(const auto& [id,r]:receipts_)if(r.active) {
            auto k=input_key(r.question.query);auto [p,inserted]=facts.emplace(k,r.question.answer);
            if(!inserted&&p->second!=r.question.answer){contradictory_=true;conflicting_cues_.insert(r.question.query.cue);}
            by_cue[r.question.query.cue].push_back(r.question);
        }
        if(contradictory_&&!repair_bank_enabled_){dirty_=false;return false;}
        std::vector<std::vector<VerifiedQuestion>> groups;for(auto& [cue,rows]:by_cue)if(!conflicting_cues_.count(cue))groups.push_back(std::move(rows));
        if(groups.empty()){dirty_=false;solved_=!contradictory_;return solved_;}
        if(repair_bank_enabled_)return rebuild_bank(groups,evaluation_budget);
        bool supported=cache_&&cache_prior_generation_==prior_generation_&&!witnesses_.empty();
        if(supported) {
            bool all=true;
            for(const auto& [id,r]:witnesses_) {
                auto current=receipts_.find(id);if(current==receipts_.end()||!same_content(r,current->second)){all=false;break;}
            }
            if(!all)supported=supported_rehearsal();
        }
        if(supported) {
            auto candidate=*cache_;
            if(compile(candidate,groups)&&direct_fit(candidate)) {
                memory_=std::move(candidate);cache_=memory_;reused_=true;dirty_=false;solved_=true;return true;
            }
        }
        if(compile(memory_,groups)){dirty_=false;solved_=true;return true;}
        ComposedSkillDiscovery discovery;discovery.evaluation_budget=evaluation_budget;discovery.repair_available=repair_available_;std::ostringstream curve;
        discovery.interleave_quantum=interleave_quantum_;discovery.progress=progress_;
        if(interleave_quantum_)discovery.assessment_cache_entries=128;
        progress_.emit("discovery_begin");
        report_=discovery.learn(priors_,groups,seed_,curve);
        progress_.emit("discovery_end");
        if(report_.acquired) {
            progress_.emit("rehearsal_begin");
            auto candidate=discovery.rehearse(priors_,report_,seed_);
            progress_.emit("rehearsal_end");
            if(compile(candidate,groups)&&direct_fit(candidate)) {
                memory_=std::move(candidate);cache_=memory_;cache_prior_generation_=prior_generation_;
                cache_cue_=report_.cue;cache_rehearsal_=report_.rehearsal;
                witnesses_.clear();for(const auto& [id,r]:receipts_)if(r.active)witnesses_[id]=r;
                dirty_=false;solved_=true;return true;
            }
        }
        dirty_=false;return false;
    }
};
}
