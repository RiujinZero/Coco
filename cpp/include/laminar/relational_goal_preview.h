#pragma once
// Opt-in, supplied consequence/proposal computation over learned positive goals.
// Nothing here acts, teaches, schedules, or receives an evaluator's useful-goal
// label. All bounded searches either finish exactly or expose an invalid result.
#include "relational_session.h"
#include <cmath>
#include <limits>

namespace laminar::relational::goal_preview {
struct Limits {
    size_t work=2000000,plan_predictions=4096,candidates=64;
    int plan_depth=8,normalization_atoms=64;
    auto operator<=>(const Limits&)const=default;
};
struct Prices {
    double prediction=1,grounding=1,atom_check=1,answer_call=1,planner_call=1;
    auto operator<=>(const Prices&)const=default;
};
struct Inputs {
    std::string parent_text,candidate_text;
    Scene world;
    std::vector<Action> actions;
    std::vector<std::string> names;
    uint64_t observation_revision=0,parent_revision=0,candidate_revision=0,affordance_revision=0;
};
struct Definition {
    std::string frame,predicate;
    std::vector<std::string> args;
    std::vector<int> slots;
    concepts::Formula formula;
};
struct Work {
    size_t work=0,predictions=0,groundings=0,atom_checks=0,answer_calls=0,planner_calls=0;
    double cost=0;
};
struct Result:Work {
    Answer answer=Answer::Unknown;
    Plan plan;
    size_t planner_allowance=0; // Actual supplied planner bound after metered pre-call work; zero if no call.
    bool valid=false;
    std::string status="uncomputed";
    int before_missing=-1,after_missing=-1,delta=0;
    double normalized_delta=0;
    uint64_t knowledge_version=0,vm_steps=0;
    Inputs inputs;
    Limits limits;
    Prices prices;
    Definition parent,candidate;
    Scene predicted;
    std::vector<Atom> potential;
    template<class Ar> void state_io(Ar& ar) {
        laminar::state::io(ar,static_cast<Work&>(*this));
        ar(answer);ar(plan);ar(planner_allowance);ar(valid);ar(status);ar(before_missing);ar(after_missing);ar(delta);ar(normalized_delta);
        ar(knowledge_version);ar(vm_steps);ar(inputs);ar(limits);ar(prices);ar(parent);ar(candidate);ar(predicted);ar(potential);
    }
};
struct ProposalInputs {
    std::string parent_text;
    Scene world;
    std::vector<Action> actions;
    std::vector<std::string> names;
    uint64_t observation_revision=0,affordance_revision=0,order_seed=0;
    int minimum_atoms=1;
};
struct Proposal {
    std::string text,predicate,frame;
    std::vector<std::string> names,args;
    concepts::Formula definition;
    // A potential-domain witness is a public derivation, not a feasible plan.
    std::vector<Atom> witness;
};
struct ProposalResult:Work {
    std::vector<Proposal> candidates;
    bool complete=false,truncated=false;
    size_t enumerated_candidates=0,total_candidates=0;
    std::string status="uncomputed";
    uint64_t knowledge_version=0;
    ProposalInputs inputs;
    Limits limits;
    Prices prices;
    std::vector<Atom> potential;
    template<class Ar> void state_io(Ar& ar) {
        laminar::state::io(ar,static_cast<Work&>(*this));
        ar(candidates);ar(complete);ar(truncated);ar(enumerated_candidates);ar(total_candidates);ar(status);ar(knowledge_version);ar(inputs);ar(limits);ar(prices);ar(potential);
    }
};

namespace detail {
struct Stop {std::string status;};
enum class Unit {prediction,grounding,atom_check,answer_call,planner_call};
struct Meter {
    Work& out;const Limits& limits;const Prices& prices;
    size_t remaining()const{return limits.work-out.work;}
    void tick(Unit unit,size_t n=1) {
        if(n>remaining())throw Stop{"work_budget"};
        out.work+=n;
        switch(unit) {
            case Unit::prediction:out.predictions+=n;out.cost+=prices.prediction*n;break;
            case Unit::grounding:out.groundings+=n;out.cost+=prices.grounding*n;break;
            case Unit::atom_check:out.atom_checks+=n;out.cost+=prices.atom_check*n;break;
            case Unit::answer_call:out.answer_calls+=n;out.cost+=prices.answer_call*n;break;
            case Unit::planner_call:out.planner_calls+=n;out.cost+=prices.planner_call*n;break;
        }
    }
};
inline bool valid(const Limits& limits,const Prices& prices) {
    auto price=[](double p){return std::isfinite(p)&&p>=0;};
    return limits.plan_depth>=0&&limits.plan_depth<=16&&limits.normalization_atoms>0&&
        price(prices.prediction)&&price(prices.grounding)&&price(prices.atom_check)&&
        price(prices.answer_call)&&price(prices.planner_call);
}
inline bool same_actions(const std::vector<Action>& a,const std::vector<Action>& b) {
    return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin(),same_action);
}
inline bool same(const Inputs& a,const Inputs& b) {
    return a.parent_text==b.parent_text&&a.candidate_text==b.candidate_text&&a.world==b.world&&
        same_actions(a.actions,b.actions)&&a.names==b.names&&
        a.observation_revision==b.observation_revision&&a.parent_revision==b.parent_revision&&
        a.candidate_revision==b.candidate_revision&&a.affordance_revision==b.affordance_revision;
}
inline bool same(const ProposalInputs& a,const ProposalInputs& b) {
    return a.parent_text==b.parent_text&&a.world==b.world&&same_actions(a.actions,b.actions)&&
        a.names==b.names&&a.observation_revision==b.observation_revision&&
        a.affordance_revision==b.affordance_revision&&a.order_seed==b.order_seed&&a.minimum_atoms==b.minimum_atoms;
}
inline std::optional<Definition> resolve(const RelationalSession& session,const std::string& text) {
    std::optional<Definition> result;size_t matches=0;
    for(const auto& [name,frame]:session.frames())if(auto b=bind(frame.pattern,text)) {
        ++matches;
        if(frame.meanings.size()!=1||frame.meanings.front().negate)continue;
        const auto& meaning=frame.meanings.front();
        auto it=session.concepts().models().find(meaning.predicate);
        if(it==session.concepts().models().end()||it->second.hypotheses.size()!=1)continue;
        Definition def{name,meaning.predicate,{},meaning.slots,concepts::expanded_clauses(it->second.hypotheses.front())};
        for(int slot:meaning.slots) {
            if(slot<0||size_t(slot)>=b->args.size())return {};
            def.args.push_back(b->args[slot]);
        }
        if(def.args.size()!=size_t(it->second.arity))return {};
        result=std::move(def);
    }
    for(const auto& [name,frame]:session.computational_frames())
        if(bind(frame.pattern,text,true))++matches;
    return matches==1?result:std::nullopt;
}
inline std::set<std::string> referents(const Scene& world,const std::vector<Action>& actions,
                                      const std::vector<std::string>& names) {
    std::set<std::string> result(names.begin(),names.end());
    for(const auto& atom:world.facts)result.insert(atom.args.begin(),atom.args.end());
    for(const auto& action:actions)result.insert(action.arguments.begin(),action.arguments.end());
    return result;
}
inline void insert_scene(std::set<Atom>& domain,const Scene& scene,Meter& meter) {
    if(!scene.complete)throw Stop{"partial_world"};
    for(const auto& atom:scene.facts){meter.tick(Unit::atom_check);domain.insert(atom);}
}
inline std::vector<Atom> potential(const RelationalSession& session,const Scene& before,
                                    const std::optional<Scene>& after,const std::vector<Action>& actions,Meter& meter) {
    std::set<Atom> domain;
    // Both scenes are deliberately used even when equal. This keeps the public
    // cost contract literal and prevents a before/after domain asymmetry.
    auto extend=[&](const Scene& world) {
        insert_scene(domain,world,meter);
        for(const auto& action:actions) {
            meter.tick(Unit::prediction);auto next=session.predict(action,world);
            if(!next)throw Stop{"domain_prediction_failed"};
            insert_scene(domain,*next,meter);
        }
    };
    extend(before);if(after)extend(*after);
    return {domain.begin(),domain.end()};
}
using Index=std::map<std::pair<std::string,size_t>,std::vector<const Atom*>>;
inline Index index(const std::vector<Atom>& domain) {
    Index out;for(const auto& atom:domain)out[{atom.predicate,atom.args.size()}].push_back(&atom);return out;
}
inline std::vector<Atom> grounded(const concepts::Clause& clause,const std::vector<std::string>& binding) {
    std::set<Atom> atoms;
    for(const auto& literal:clause.body) {
        Atom atom{literal.predicate,{}};
        for(int variable:literal.vars)atom.args.push_back(binding.at(variable));
        atoms.insert(std::move(atom));
    }
    return {atoms.begin(),atoms.end()};
}
// A complete relational join over the potential domain, with no best-so-far
// early return. The most-bound next literal reduces irrelevant products but
// does not alter the admissible groundings. Every fact match and recursive
// grounding expansion is paid, including failed matches and empty joins.
template<class Emit>
void enumerate(const concepts::Clause& clause,const Index& domain,std::vector<std::string> binding,
               std::vector<bool> done,Meter& meter,const std::set<std::string>& names,Emit& emit) {
    meter.tick(Unit::grounding);
    size_t at=clause.body.size(),smallest=std::numeric_limits<size_t>::max();int most_bound=-1;
    for(size_t i=0;i<clause.body.size();++i)if(!done[i]) {
        const auto& literal=clause.body[i];int count=0;
        for(int v:literal.vars)if(!binding.at(v).empty())++count;
        auto found=domain.find({literal.predicate,literal.vars.size()});
        size_t available=found==domain.end()?0:found->second.size();
        if(count>most_bound||(count==most_bound&&available<smallest)) {at=i;most_bound=count;smallest=available;}
    }
    if(at==clause.body.size()) {
        // Unused head positions are still grounded only in public referents.
        for(int v=0;v<clause.arity;++v)if(binding[v].empty()) {
            for(const auto& name:names) {auto next=binding;next[v]=name;enumerate(clause,domain,std::move(next),done,meter,names,emit);}
            return;
        }
        emit(binding,grounded(clause,binding));return;
    }
    const auto& literal=clause.body[at];auto found=domain.find({literal.predicate,literal.vars.size()});
    if(found==domain.end())return;
    done[at]=true;
    for(const auto* atom:found->second) {
        meter.tick(Unit::atom_check);auto next=binding;bool compatible=true;
        for(size_t i=0;i<literal.vars.size();++i) {
            auto& value=next.at(literal.vars[i]);
            if(!value.empty()&&value!=atom->args[i]){compatible=false;break;}
            value=atom->args[i];
        }
        if(compatible)enumerate(clause,domain,std::move(next),done,meter,names,emit);
    }
}
inline std::pair<int,int> minima(const Definition& definition,const std::vector<Atom>& potential,
                                 const Scene& before,const Scene& after,Meter& meter,
                                 const std::set<std::string>& names) {
    auto facts=index(potential);const auto old=sorted(before),next=sorted(after);
    int a=std::numeric_limits<int>::max(),b=a;
    for(const auto& clause:definition.formula) {
        std::vector<std::string> binding(concepts::variables(clause));
        if(definition.args.size()!=size_t(clause.arity))throw Stop{"invalid_definition"};
        std::copy(definition.args.begin(),definition.args.end(),binding.begin());
        auto emit=[&](const auto&,const std::vector<Atom>& atoms) {
            int x=0,y=0;
            for(const auto& atom:atoms) {
                meter.tick(Unit::atom_check);x+=!std::binary_search(old.facts.begin(),old.facts.end(),atom);
                meter.tick(Unit::atom_check);y+=!std::binary_search(next.facts.begin(),next.facts.end(),atom);
            }
            a=std::min(a,x);b=std::min(b,y);
        };
        enumerate(clause,facts,std::move(binding),std::vector<bool>(clause.body.size()),meter,names,emit);
    }
    if(a==std::numeric_limits<int>::max())throw Stop{"no_compatible_grounding"};
    return {a,b};
}
inline uint64_t order_key(const std::string& text,uint64_t seed) {
    // Fixed FNV-1a bytes plus SplitMix64; unlike std::hash this order is portable.
    uint64_t h=14695981039346656037ull;
    for(unsigned char c:text){h^=c;h*=1099511628211ull;}
    h+=seed+0x9e3779b97f4a7c15ull;h=(h^(h>>30))*0xbf58476d1ce4e5b9ull;
    h=(h^(h>>27))*0x94d049bb133111ebull;return h^(h>>31);
}
} // namespace detail

inline bool current(const RelationalSession& session,const Inputs& inputs,const Result& result) {
    return session.ready()&&session.version()==result.knowledge_version&&detail::same(inputs,result.inputs);
}
inline bool current(const RelationalSession& session,const ProposalInputs& inputs,const ProposalResult& result) {
    return session.ready()&&session.version()==result.knowledge_version&&detail::same(inputs,result.inputs);
}
inline Result analyze(const RelationalSession& session,const Inputs& inputs,Limits limits={},Prices prices={}) {
    Result out;out.inputs=inputs;out.limits=limits;out.prices=prices;out.knowledge_version=session.version();
    detail::Meter meter{out,limits,prices};auto steps=session.primitive_steps();
    try {
        if(!detail::valid(limits,prices))throw detail::Stop{"invalid_limits_or_prices"};
        if(!session.ready())throw detail::Stop{"knowledge_dirty"};
        if(!inputs.world.complete)throw detail::Stop{"partial_world"};
        auto parent=detail::resolve(session,inputs.parent_text),candidate=detail::resolve(session,inputs.candidate_text);
        if(!parent||!candidate)throw detail::Stop{"unresolved_positive_meaning"};
        out.parent=*parent;out.candidate=*candidate;
        for(const auto& branch:parent->formula)if(branch.body.size()>size_t(limits.normalization_atoms))
            throw detail::Stop{"normalization_bound"};
        // These are calls into the existing bounded supplied interpreter and
        // planner. Their internal joins are NOT claimed as separately metered.
        meter.tick(detail::Unit::answer_call);out.answer=session.answer(inputs.candidate_text,inputs.world);
        if(out.answer==Answer::Unknown)throw detail::Stop{"unavailable_candidate_answer"};
        meter.tick(detail::Unit::planner_call);
        out.planner_allowance=std::min(limits.plan_predictions,meter.remaining());
        out.plan=session.plan(inputs.candidate_text,inputs.world,inputs.actions,out.planner_allowance,limits.plan_depth);
        meter.tick(detail::Unit::prediction,out.plan.predictions);
        if(out.plan.status!="searched"&&out.plan.status!="procedure"&&out.plan.status!="satisfied")
            throw detail::Stop{"candidate_"+out.plan.status};
        out.predicted=sorted(inputs.world);
        for(const auto& action:out.plan.actions) {
            meter.tick(detail::Unit::prediction);auto next=session.predict(action,out.predicted);
            if(!next)throw detail::Stop{"plan_prediction_failed"};
            out.predicted=std::move(*next);
        }
        out.potential=detail::potential(session,inputs.world,out.predicted,inputs.actions,meter);
        auto missing=detail::minima(*parent,out.potential,inputs.world,out.predicted,meter,
            detail::referents(inputs.world,inputs.actions,inputs.names));
        out.before_missing=missing.first;out.after_missing=missing.second;
        out.delta=missing.first-missing.second;out.normalized_delta=double(out.delta)/limits.normalization_atoms;
        out.valid=true;out.status="exact";
    }catch(const detail::Stop& stop){out.status=stop.status;}
    out.vm_steps=session.primitive_steps()-steps;
    return out;
}

inline ProposalResult propose(const RelationalSession& session,const ProposalInputs& inputs,Limits limits={},Prices prices={}) {
    ProposalResult out;out.inputs=inputs;out.limits=limits;out.prices=prices;out.knowledge_version=session.version();
    detail::Meter meter{out,limits,prices};
    try {
        if(!detail::valid(limits,prices))throw detail::Stop{"invalid_limits_or_prices"};
        if(inputs.minimum_atoms<1||inputs.minimum_atoms>64)throw detail::Stop{"invalid_minimum_atoms"};
        if(!session.ready())throw detail::Stop{"knowledge_dirty"};
        if(!inputs.world.complete)throw detail::Stop{"partial_world"};
        if(!detail::resolve(session,inputs.parent_text))throw detail::Stop{"unresolved_positive_parent"};
        out.potential=detail::potential(session,inputs.world,std::nullopt,inputs.actions,meter);
        auto domain=detail::index(out.potential);auto names=detail::referents(inputs.world,inputs.actions,inputs.names);
        std::map<std::string,Proposal> found;
        for(const auto& [frame_name,frame]:session.frames()) {
            if(frame.meanings.size()!=1||frame.meanings.front().negate)continue;
            const auto& meaning=frame.meanings.front();auto model=session.concepts().models().find(meaning.predicate);
            if(model==session.concepts().models().end()||model->second.hypotheses.size()!=1)continue;
            auto formula=concepts::expanded_clauses(model->second.hypotheses.front());
            for(const auto& branch:formula) {
                auto emit=[&](const std::vector<std::string>& binding,const std::vector<Atom>& witness) {
                    std::vector<std::string> arguments(meaning.slots.size());
                    if(binding.size()<meaning.slots.size())throw detail::Stop{"invalid_definition"};
                    for(size_t role=0;role<meaning.slots.size();++role) {
                        int slot=meaning.slots[role];
                        if(slot<0||size_t(slot)>=arguments.size())throw detail::Stop{"invalid_definition"};
                        arguments[slot]=binding[role];
                    }
                    std::string text;size_t at=0;
                    for(const auto& word:frame.pattern) {
                        if(!text.empty())text+=' ';
                        if(!word.empty()&&word[0]=='$') {
                            if(at>=arguments.size()||word!="$"+std::to_string(at))throw detail::Stop{"invalid_definition"};
                            text+=arguments[at++];
                        }else text+=word;
                    }
                    if(text==inputs.parent_text||found.count(text))return;
                    // Rendering can collide with another learned language
                    // pattern: use the same public parse ambiguity boundary.
                    auto resolved=detail::resolve(session,text);
                    if(!resolved||resolved->predicate!=meaning.predicate)return;
                    Proposal p{text,meaning.predicate,frame_name,arguments,resolved->args,formula,witness};
                    found.emplace(text,std::move(p));
                };
                detail::enumerate(branch,domain,std::vector<std::string>(concepts::variables(branch)),
                    std::vector<bool>(branch.body.size()),meter,names,emit);
            }
        }
        out.enumerated_candidates=found.size();
        // Optional supplied structural granularity, never a usefulness filter.
        // Finish and charge the entire ordinary enumeration before filtering.
        for(auto& [text,candidate]:found)
            if(!candidate.definition.empty()&&std::all_of(candidate.definition.begin(),candidate.definition.end(),
                [&](const auto& branch){return branch.body.size()>=size_t(inputs.minimum_atoms);}))
                out.candidates.push_back(std::move(candidate));
        std::sort(out.candidates.begin(),out.candidates.end(),[&](const Proposal& a,const Proposal& b){
            auto ka=detail::order_key(a.text,inputs.order_seed),kb=detail::order_key(b.text,inputs.order_seed);
            return ka==kb?a.text<b.text:ka<kb;
        });
        out.total_candidates=out.candidates.size();out.truncated=out.total_candidates>limits.candidates;
        if(out.truncated)out.candidates.resize(limits.candidates);
        out.complete=true;out.status=out.truncated?"candidate_limit":"exact";
    }catch(const detail::Stop& stop) {
        // A work-exhausted prefix is not an accepted proposal inventory.
        out.candidates.clear();out.status=stop.status;
    }
    return out;
}
} // namespace laminar::relational::goal_preview
