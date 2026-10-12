#pragma once
// Joint micro-policy / macro-program induction. Only whole-question answers
// score candidate pairs. No caller supplies the missing cue or a program.
#include "primitive_store.h"
#include "search_progress.h"

namespace laminar::reasoning {
inline constexpr int composed_primitive_replay_limit=256;
template<class Choice> bool verifies_direct_outcome(Choice& choices,const PrimitiveReceipt& r,
                                                   int limit=composed_primitive_replay_limit) {
    MicroState vm;vm.task=r.cue;vm.relation=r.relation;vm.subject=r.subject;vm.target=r.target;
    for(int n=0;n<limit&&!vm.done;++n) {
        auto available=vm.available();int action=choices.choose(vm.observe(),available);
        if(std::find(available.begin(),available.end(),action)==available.end())return false;
        vm.step(action,r.facts);
    }
    return vm.done&&vm.output==r.answer;
}
struct ComposedAttempt {
    int cue=0,correct=0,total=0,evaluations=0,first_solution=-1;
    bool previously_available=false;
    uint64_t proposal_steps=0,program_states=0;
};
struct ComposedDiscoveryResult {
    bool acquired=false;
    int cue=-1,correct=0,total=0,evaluations=0,first_solution=-1;
    uint64_t proposal_steps=0,program_states=0;
    int direct_correct=0,direct_total=0;
    bool repaired=false;
    std::vector<ComposedAttempt> attempts;
    DiscoveredPolicy policy;
    std::map<int,Program> programs;
    std::map<uint32_t,Sample> rehearsal;
};
struct VerifiedRepair {
    DiscoveredPolicy policy;
    std::map<uint32_t,Sample> rehearsal;
};
using RepairBank=std::map<int,VerifiedRepair>;
struct ComposedSkillDiscovery {
    int evaluation_budget=10000,max_instructions=8;
    // Zero preserves the historical sequential search and refinement budget.
    // Positive quanta share proposals among candidate slots and stop at a full fit.
    int interleave_quantum=0;
    size_t assessment_cache_entries=0;
    SearchProgress progress;
    bool repair_available=false;
    size_t max_states=1000;
    // Rebuild one neural model from direct evidence and every independently
    // supported repair. The caller must prove outcome support before passing
    // a bank; this method checks consistency and all direct replay outcomes.
    ProcedureMemory rehearse_bank(PrimitiveStore& store,const RepairBank& repairs,unsigned seed,bool retain=true)const {
        if(repairs.empty())return store.memory;
        ProcedureMemory model=store.memory;std::map<uint32_t,Sample> bank;
        auto add=[&](uint32_t key,const Sample& sample) {
            if(key!=policy_state_key(sample.obs)||std::find(sample.available.begin(),sample.available.end(),sample.action)==sample.available.end())
                throw std::invalid_argument("invalid repair bank sample");
            auto prior=bank.find(key);
            if(prior!=bank.end()&&(prior->second.action!=sample.action||prior->second.available!=sample.available))
                throw std::runtime_error("conflicting repair bank rehearsal");
            bank[key]=sample;
        };
        for(const auto& [cue,repair]:repairs) {
            if(cue<0||cue>2||!repair.policy.sealed||repair.rehearsal.empty())throw std::invalid_argument("invalid accepted repair");
            for(const auto& [key,sample]:repair.rehearsal) {
                if(!sample.obs[cue]||repair.policy.choose(sample.obs,sample.available)!=sample.action)
                    throw std::invalid_argument("repair bank policy differs from support");
                add(key,sample);
            }
            model.available_skills[cue]=true;
        }
        if(retain)for(const auto& [id,r]:store.receipts())if(r.active&&model.available_skills[r.cue]) {
            auto found=repairs.find(r.cue);
            const auto& policy=found==repairs.end()?store.policies()[r.cue]:found->second.policy;
            MicroState vm;vm.task=r.cue;vm.relation=r.relation;vm.subject=r.subject;vm.target=r.target;
            for(int n=0;n<composed_primitive_replay_limit&&!vm.done;++n) {
                auto obs=vm.observe();auto available=vm.available();int action=policy.choose(obs,available);
                if(action<0)throw std::runtime_error("repair bank lacks direct outcome support");
                add(policy_state_key(obs),{obs,available,action});vm.step(action,r.facts);
            }
            if(!vm.done||vm.output!=r.answer)throw std::runtime_error("repair bank changes direct outcome");
        }
        std::vector<Sample> fit;for(const auto& [key,sample]:bank)fit.push_back(sample);std::mt19937 rng(seed+9100);
        for(int epoch=0;epoch<128;++epoch){std::shuffle(fit.begin(),fit.end(),rng);for(const auto& sample:fit)model.skills.fit(sample);}
        return model;
    }
    // Materialize a candidate in the same native neural ProcedureMemory. Direct
    // receipts support old skills; new instruction targets come exclusively
    // from executions whose complete composed answers passed verification.
    ProcedureMemory rehearse(PrimitiveStore& store,const ComposedDiscoveryResult& result,
                             unsigned seed,bool retain=true,const RepairBank* retained_bank=nullptr)const {
        if(!result.acquired||result.cue<0||result.cue>2)throw std::invalid_argument("no composed discovery to rehearse");
        if(retained_bank) {
            RepairBank bank=retain?*retained_bank:RepairBank{};
            bank[result.cue]={result.policy,result.rehearsal};auto model=rehearse_bank(store,bank,seed,retain);
            for(const auto& [cue,program]:result.programs)model.procedures[cue]=program;
            return model;
        }
        ProcedureMemory model=store.memory;
        std::map<uint32_t,Sample> bank=result.rehearsal;
        if(retain)for(const auto& [id,r]:store.receipts())if(r.active) {
            MicroState vm;vm.task=r.cue;vm.relation=r.relation;vm.subject=r.subject;vm.target=r.target;
            for(int n=0;n<composed_primitive_replay_limit&&!vm.done;++n) {
                auto obs=vm.observe();auto available=vm.available();
                // A repaired skill keeps old verified outcomes, not the old
                // instruction targets that caused its new counterexample.
                int a=(r.cue==result.cue?result.policy:store.policies()[r.cue]).choose(obs,available);
                if(a<0)throw std::runtime_error("old primitive cannot generate retention evidence");
                auto key=policy_state_key(obs);auto prior=bank.find(key);
                if(prior!=bank.end()&&prior->second.action!=a)throw std::runtime_error("conflicting composed rehearsal");
                bank[key]={obs,available,a};vm.step(a,r.facts);
            }
            if(!vm.done||vm.output!=r.answer)throw std::runtime_error("old rehearsal lacks verified outcome");
        }
        std::vector<Sample> fit;for(const auto& [key,s]:bank)fit.push_back(s);std::mt19937 rng(seed+9100);
        for(int epoch=0;epoch<128;++epoch) {
            std::shuffle(fit.begin(),fit.end(),rng);for(const auto& s:fit)model.skills.fit(s);
        }
        model.available_skills[result.cue]=true;
        for(const auto& [cue,p]:result.programs)model.procedures[cue]=p;
        return model;
    }
    struct Choice {
        LocalChoice& base;DiscoveredPolicy& candidate;int cue;
        std::map<uint32_t,int>& base_cache;
        std::map<uint32_t,Sample>* visited=nullptr;
        int choose(const Observation& obs,const std::vector<int>& available) {
            auto key=policy_state_key(obs);
            if(obs[cue]) {
                int a=candidate.choose(obs,available);
                if(visited&&a>=0)(*visited)[key]={obs,available,a};
                return a;
            }
            auto found=base_cache.find(key);
            if(found!=base_cache.end()&&std::find(available.begin(),available.end(),found->second)!=available.end())return found->second;
            int a=base.choose(obs,available);base_cache[key]=a;return a;
        }
    };
    ComposedDiscoveryResult learn(PrimitiveStore& store,const std::vector<std::vector<VerifiedQuestion>>& groups,unsigned seed,std::ostream& curve,
                                  const ProcedureMemory* supported_base=nullptr,const RepairBank* retained_bank=nullptr)const {
        ComposedDiscoveryResult best;
        if(store.dirty())throw std::logic_error("direct primitive evidence must be rebuilt first");
        if((supported_base||retained_bank)&&!repair_available)throw std::invalid_argument("repair bank requires available-skill repair");
        const ProcedureMemory& base=supported_base?*supported_base:store.memory;
        std::set<int> cues;
        for(const auto& rows:groups) {
            if(rows.empty()||!cues.insert(rows.front().query.cue).second)throw std::invalid_argument("invalid composed evidence groups");
            best.total+=int(rows.size());
        }
        if(groups.empty())return best;
        const int question_total=best.total;
        const auto original_programs=store.memory.procedures;
        const auto original_direct_available=store.memory.available_skills;
        const auto original_available=base.available_skills;
        std::map<uint32_t,int> base_cache;
        std::vector<ComposedAttempt> attempts;
        if(interleave_quantum<0)throw std::invalid_argument("negative proposal quantum");
        struct Attempt {
            int cue;ProcedureMemory scratch;PolicySearch search;
            std::map<int,Program> programs;
            uint64_t proposal_steps=0,program_states=0;
            std::ostringstream curve;
            std::optional<ComposedDiscoveryResult> finished;
            std::function<PolicyAssessment(DiscoveredPolicy&)> assess;
            std::function<ComposedDiscoveryResult(PolicyDiscoveryResult)> finish;
            struct CachedAssessment {
                PolicyAssessment score;std::map<int,Program> programs;
                uint64_t proposal_steps,program_states,skill_steps;
            };
            std::deque<CachedAssessment> cache;
            uint64_t cache_hits=0,skipped_states=0,skipped_steps=0;
            Attempt(int cue,const ProcedureMemory& memory,PolicySearch search):cue(cue),scratch(memory),search(std::move(search)){scratch.available_skills[cue]=true;}
        };
        std::vector<std::unique_ptr<Attempt>> candidates;
        for(int cue=0;cue<3;++cue)if(!original_available[cue]||repair_available) {
            // A hypothesis may execute the unavailable slot. Nothing becomes
            // available in the live learner unless the whole evidence fits.
            PolicyDiscovery search;search.evaluation_budget=evaluation_budget;search.stop_at_first=interleave_quantum>0;
            const DiscoveredPolicy* warm=repair_available&&original_available[cue]?&store.policies()[cue]:nullptr;
            if(retained_bank){auto found=retained_bank->find(cue);if(found!=retained_bank->end())warm=&found->second.policy;}
            candidates.push_back(std::make_unique<Attempt>(cue,base,search.start(seed+unsigned(cue)*1009,warm)));
            auto* context=candidates.back().get();
            auto direct=[&,context](DiscoveredPolicy& candidate,std::map<uint32_t,Sample>* visited=nullptr) {
                PolicyAssessment score;
                for(const auto& [id,r]:store.receipts())if(r.active&&r.cue==context->cue) {
                    MicroState vm;vm.task=r.cue;vm.relation=r.relation;vm.subject=r.subject;vm.target=r.target;
                    for(int n=0;n<composed_primitive_replay_limit&&!vm.done;++n) {
                        auto obs=vm.observe();auto available=vm.available();int a=candidate.choose(obs,available);
                        if(a<0)break;
                        if(visited)(*visited)[policy_state_key(obs)]={obs,available,a};
                        vm.step(a,r.facts);++score.steps;
                    }
                    ++score.total;score.correct+=vm.done&&vm.output==r.answer;score.timeouts+=!vm.done;
                }
                return score;
            };
            auto assess=[&,context,direct](DiscoveredPolicy& candidate) {
                auto& scratch=context->scratch;auto& programs=context->programs;
                auto& proposal_steps=context->proposal_steps;auto& program_states=context->program_states;
                if(context->search.count()%100==0)progress.emit("proposal_begin",context->cue,context->search.count(),program_states,proposal_steps);
                // Matching every decision encountered by a deterministic
                // assessment proves that its whole execution will be identical.
                // Unvisited policy rules cannot affect that execution. Caches
                // are per slot and per learn() call; evidence/base weights never
                // change during their lifetime. Keep logical work accounting
                // identical and report avoided physical simulation separately.
                for(auto it=context->cache.rbegin();it!=context->cache.rend();++it) {
                    bool same=true;
                    for(const auto& [key,s]:it->score.visited)if(candidate.choose(s.obs,s.available)!=s.action){same=false;break;}
                    if(same) {
                        ++context->cache_hits;context->skipped_states+=it->program_states;context->skipped_steps+=it->proposal_steps;
                        proposal_steps+=it->proposal_steps;program_states+=it->program_states;scratch.skill_steps+=it->skill_steps;programs=it->programs;
                        if(context->search.count()%100==0)progress.emit("assessment_cache_hit",context->cue,context->search.count(),it->program_states,it->proposal_steps);
                        return it->score;
                    }
                }
                const auto initial_proposals=proposal_steps,initial_states=program_states,initial_steps=scratch.skill_steps;
                auto remember=[&](PolicyAssessment a) {
                    if(assessment_cache_entries) {
                        context->cache.push_back({a,programs,proposal_steps-initial_proposals,program_states-initial_states,scratch.skill_steps-initial_steps});
                        if(context->cache.size()>assessment_cache_entries)context->cache.pop_front();
                    }
                    return a;
                };
                PolicyAssessment a;
                programs.clear();
                if(repair_available) {
                    auto retained=direct(candidate,&a.visited);proposal_steps+=retained.steps;
                    a.correct=retained.correct;a.total=retained.total;a.steps=retained.steps;a.timeouts=retained.timeouts;
                    // Verified prior outcomes are constraints. Reject a repair
                    // that breaks them before paying for macro synthesis.
                    // Unexamined composed cases contribute no positive credit.
                    if(retained.correct!=retained.total){a.total+=question_total;return remember(std::move(a));}
                }
                Choice choices{scratch.skills,candidate,context->cue,base_cache,&a.visited};
                uint64_t before=scratch.skill_steps;
                for(const auto& rows:groups) {
                    auto result=scratch.learn_with(rows,choices,false,max_instructions,max_states);
                    a.correct+=int(result.correct);a.total+=int(result.total);program_states+=result.states;
                    programs[rows.front().query.cue]=result.best_program;
                }
                proposal_steps+=scratch.skill_steps-before;
                // Refinement charges executions of the proposed complete
                // programs, separately from the supplied synthesis work.
                choices.visited=nullptr;before=scratch.skill_steps;
                for(const auto& rows:groups)for(const auto& e:rows) {
                    auto output=scratch.execute_with(e.query,programs.at(e.query.cue),choices);
                    a.timeouts+=!output.present();
                }
                int replay_steps=int(scratch.skill_steps-before);a.steps+=replay_steps;proposal_steps+=replay_steps;
                return remember(std::move(a));
            };
            context->assess=assess;
            context->finish=[&,context,direct,assess](PolicyDiscoveryResult result) {
            const int cue=context->cue;auto& scratch=context->scratch;auto& programs=context->programs;
            auto verified=assess(result.policy);
            progress.emit("assessment_cache_totals",cue,context->cache_hits,context->skipped_states,context->skipped_steps);
            ComposedDiscoveryResult r;r.cue=cue;r.correct=verified.correct;r.total=verified.total;
            r.evaluations=result.evaluations;r.first_solution=result.first_solution;
            r.proposal_steps=context->proposal_steps;r.program_states=context->program_states;r.programs=programs;
            r.repaired=original_available[cue];
            if(repair_available) {auto retained=direct(result.policy);r.direct_correct=retained.correct;r.direct_total=retained.total;}
            if(r.correct==r.total) {
                Choice choices{scratch.skills,result.policy,cue,base_cache,&r.rehearsal};
                for(const auto& rows:groups)for(const auto& e:rows)
                    if(scratch.execute_with(e.query,r.programs.at(e.query.cue),choices)!=e.answer)throw std::runtime_error("composed discovery failed replay");
                if(repair_available) {
                    auto retained=direct(result.policy,&r.rehearsal);
                    if(retained.correct!=retained.total)throw std::runtime_error("repair changed a verified primitive outcome");
                }
                if(!r.rehearsal.empty()){result.policy.seal(r.rehearsal);r.policy=std::move(result.policy);r.acquired=true;}
            }
            return r;
            };
        }
        auto record=[&](ComposedDiscoveryResult r) {
            attempts.push_back({r.cue,r.correct,r.total,r.evaluations,r.first_solution,original_available[r.cue],r.proposal_steps,r.program_states});
            // Different slots may have different numbers of direct priors.
            // A partial fit with more raw correct rows must never replace a
            // complete, outcome-verified acquisition from another slot.
            if(best.cue<0||(r.acquired!=best.acquired?r.acquired:r.correct>best.correct))best=std::move(r);
        };
        if(interleave_quantum) {
            bool worked,solved=false;
            do {
                worked=false;
                for(auto& candidate:candidates)if(!candidate->search.done()) {
                    progress.emit("slot_begin",candidate->cue,candidate->search.count(),candidate->program_states,candidate->proposal_steps);
                    worked=true;candidate->search.advance(interleave_quantum,candidate->assess,candidate->curve);
                    progress.emit("slot_budget",candidate->cue,candidate->search.count(),candidate->program_states,candidate->proposal_steps);
                    if(candidate->search.solved()) {
                        candidate->finished=candidate->finish(candidate->search.result());
                        // A complete macro fit that never used this candidate
                        // supplies no learned primitive targets. Try other slots.
                        if(candidate->finished->acquired){solved=true;break;}
                    }
                }
            }while(worked&&!solved);
            // Curves remain grouped by slot for independent replay, while every
            // started candidate is accounted for even if another slot succeeded.
            for(auto& candidate:candidates)if(candidate->search.count()) {
                curve<<candidate->curve.str();record(candidate->finished?std::move(*candidate->finished):candidate->finish(candidate->search.result()));
            }
        } else for(auto& candidate:candidates) {
            // Bounded advances preserve sequential search exactly, while an
            // interrupted process retains diagnostic budget checkpoints.
            do {
                progress.emit("slot_begin",candidate->cue,candidate->search.count(),candidate->program_states,candidate->proposal_steps);
                candidate->search.advance(std::min(1000,evaluation_budget),candidate->assess,curve);
                progress.emit("slot_budget",candidate->cue,candidate->search.count(),candidate->program_states,candidate->proposal_steps);
            }while(!candidate->search.done());
            record(candidate->finish(candidate->search.result()));if(best.acquired)break;
        }
        if(store.memory.procedures!=original_programs||store.memory.available_skills!=original_direct_available||base.available_skills!=original_available)throw std::runtime_error("candidate search mutated live knowledge");
        best.attempts=std::move(attempts);
        return best;
    }
};
}
