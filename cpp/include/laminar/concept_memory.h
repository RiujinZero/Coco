#pragma once
// Bounded relational concept induction. The supplied substrate is conjunction,
// variable binding, existential quantification and shortest-consistent search.
// No task-specific predicates or concept definitions occur in this learner.
#include <algorithm>
#include <cctype>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace laminar::concepts {
struct Atom { std::string predicate; std::vector<std::string> args; auto operator<=>(const Atom&) const = default; };
struct Scene { std::vector<Atom> facts; bool complete = true; auto operator<=>(const Scene&) const = default; };
struct Literal { std::string predicate; std::vector<int> vars; auto operator<=>(const Literal&) const = default; };
struct Clause { int arity = 0; std::vector<Literal> body; auto operator<=>(const Clause&) const = default; };
enum class Answer { Unknown = -1, False = 0, True = 1 };
struct Limits {
    int body = 3, variables = 10, arity = 8, facts = 256, entities = 32;
    size_t receipts = 1024, schemas = 128, candidates = 200000, work = 20000000;
    // One clause preserves the original conjunction learner exactly.
    int clauses = 1, expanded_clauses = 64;
};
struct Receipt {
    std::string id, name, source = "verified_teacher";
    unsigned version = 1;
    Scene scene;
    std::vector<std::string> args;
    bool answer = false, active = true, generated = false;
};
struct Hypothesis {
    Clause local, expanded;
    // Branches within a hypothesis are OR; hypotheses are unresolved meanings.
    std::vector<Clause> local_extra, expanded_extra;
};
using Formula = std::vector<Clause>;
inline Formula local_clauses(const Hypothesis& h) {
    Formula result{h.local}; result.insert(result.end(),h.local_extra.begin(),h.local_extra.end()); return result;
}
inline Formula expanded_clauses(const Hypothesis& h) {
    Formula result{h.expanded}; result.insert(result.end(),h.expanded_extra.begin(),h.expanded_extra.end()); return result;
}
struct Model {
    int arity = 0;
    std::string status = "untrained";
    std::vector<Hypothesis> hypotheses;
    size_t candidates = 0, work = 0;
};
struct Exhausted {};
struct OutsideClass {};
struct Meter {
    size_t used = 0, limit;
    explicit Meter(size_t n): limit(n) {}
    void tick(size_t n = 1) { if(n > limit || used > limit - n) throw Exhausted{}; used += n; }
};

inline bool symbol(const std::string& s) {
    return !s.empty() && s.size() <= 80 && std::all_of(s.begin(),s.end(),[](unsigned char c){return std::isalnum(c)||c=='_';});
}
inline int variables(const Clause& c) {
    int n=c.arity; for(const auto& a:c.body) for(int v:a.vars) n=std::max(n,v+1); return n;
}
inline std::string key(const Clause& c) {
    std::string out=std::to_string(c.arity)+":";
    for(const auto& a:c.body) { out+=a.predicate+"("; for(int v:a.vars) out+=std::to_string(v)+","; out+=")"; }
    return out;
}
// Exact conjunction-order / alpha-renaming canonicalization, not a general
// conjunctive-query equivalence solver. Every permutation consumes work budget.
inline Clause canonical(const Clause& c, Meter& meter) {
    std::set<int> existing; for(const auto& a:c.body) for(int v:a.vars) if(v>=c.arity) existing.insert(v);
    std::vector<int> names(existing.begin(),existing.end()), permutation;
    for(size_t i=0;i<names.size();++i) permutation.push_back(c.arity+int(i));
    Clause best; std::string best_key;
    do {
        meter.tick(); Clause x=c;
        for(auto& a:x.body) for(int& v:a.vars) if(v>=c.arity) v=permutation[std::lower_bound(names.begin(),names.end(),v)-names.begin()];
        std::sort(x.body.begin(),x.body.end()); x.body.erase(std::unique(x.body.begin(),x.body.end()),x.body.end());
        auto k=key(x); if(best_key.empty()||k<best_key) { best_key=k; best=std::move(x); }
    } while(std::next_permutation(permutation.begin(),permutation.end()));
    return best;
}

// Relational join. Query values enter only through head bindings. Callback can
// stop after a proof; materialization visits all proofs and deduplicates tuples.
inline void join(const Clause& c, const Scene& s, std::vector<std::string> binding,
                 size_t at, Meter& meter, const std::function<bool(const std::vector<std::string>&)>& emit, bool& stop) {
    if(stop) return;
    meter.tick();
    if(at==c.body.size()) { stop=emit(binding); return; }
    const auto& literal=c.body[at];
    for(const auto& atom:s.facts) {
        meter.tick(); if(atom.predicate!=literal.predicate || atom.args.size()!=literal.vars.size()) continue;
        auto next=binding; bool match=true;
        for(size_t i=0;i<atom.args.size();++i) {
            auto& v=next[literal.vars[i]];
            if(!v.empty()&&v!=atom.args[i]) { match=false; break; }
            v=atom.args[i];
        }
        if(match) join(c,s,std::move(next),at+1,meter,emit,stop);
        if(stop) return;
    }
}
inline bool proves(const Clause& c,const Scene& s,const std::vector<std::string>& args,Meter& meter) {
    if(int(args.size())!=c.arity) return false;
    std::vector<std::string> binding(variables(c)); std::copy(args.begin(),args.end(),binding.begin()); bool stop=false;
    join(c,s,std::move(binding),0,meter,[](const auto&){return true;},stop); return stop;
}

// A conjunctive query implies another iff the latter has a head-preserving
// homomorphism into the former's canonical database. Remove redundant literals
// before alpha-normalization so equivalent macro expansions do not create false
// ambiguity (e.g. an existential witness already supplied by a head argument).
inline Clause minimize(Clause c,Meter& meter) {
    std::vector<std::string> head;
    for(int i=0;i<c.arity;++i) head.push_back("v"+std::to_string(i));
    bool changed=true;
    while(changed) {
        changed=false;
        for(size_t i=0;i<c.body.size();++i) {
            Clause smaller=c; smaller.body.erase(smaller.body.begin()+i); Scene database;
            for(const auto& a:smaller.body) {
                Atom fact{a.predicate,{}}; for(int v:a.vars) fact.args.push_back("v"+std::to_string(v)); database.facts.push_back(std::move(fact));
            }
            if(proves(c,database,head,meter)) { c=std::move(smaller); changed=true; break; }
        }
    }
    return canonical(c,meter);
}

inline bool implies(const Clause& stronger,const Clause& weaker,Meter& meter) {
    if(stronger.arity!=weaker.arity)return false;
    Scene database;std::vector<std::string> head;
    for(int i=0;i<stronger.arity;++i)head.push_back("v"+std::to_string(i));
    for(const auto& a:stronger.body) {
        Atom fact{a.predicate,{}};for(int v:a.vars)fact.args.push_back("v"+std::to_string(v));database.facts.push_back(std::move(fact));
    }
    return proves(weaker,database,head,meter);
}
inline Formula normalize_formula(Formula formula,Meter& meter) {
    for(auto& c:formula)c=minimize(std::move(c),meter);
    std::sort(formula.begin(),formula.end(),[](const auto& a,const auto& b){return key(a)<key(b);});
    std::vector<bool> redundant(formula.size(),false);
    for(size_t i=0;i<formula.size();++i)for(size_t j=0;j<formula.size();++j)if(i!=j&&!redundant[j]&&implies(formula[i],formula[j],meter)) {
        if(j<i||!implies(formula[j],formula[i],meter)){redundant[i]=true;break;}
    }
    Formula result;for(size_t i=0;i<formula.size();++i)if(!redundant[i])result.push_back(std::move(formula[i]));return result;
}
inline std::string formula_key(const Formula& formula) {
    std::string result;for(const auto& c:formula)result+="["+key(c)+"]";return result;
}
inline bool proves(const Formula& formula,const Scene& scene,const std::vector<std::string>& args,Meter& meter) {
    for(const auto& c:formula)if(proves(c,scene,args,meter))return true;
    return false;
}
inline Hypothesis hypothesis(Formula local,Formula expanded) {
    if(local.empty()||expanded.empty())throw std::logic_error("empty learned rule union");
    Hypothesis h{std::move(local.front()),std::move(expanded.front()),{}, {}};
    h.local_extra.assign(std::make_move_iterator(local.begin()+1),std::make_move_iterator(local.end()));
    h.expanded_extra.assign(std::make_move_iterator(expanded.begin()+1),std::make_move_iterator(expanded.end()));return h;
}

class ConceptMemory {
    Limits limits_;
    bool reuse_, dirty_=false;
    std::map<std::string,int> base_, heads_;
    std::vector<std::string> order_;
    std::map<std::string,Receipt> receipts_;
    std::map<std::string,Model> models_;
public:
    template<class Ar> void state_io(Ar& ar){ar(limits_);ar(reuse_);ar(dirty_);ar(base_);ar(heads_);ar(order_);ar(receipts_);ar(models_);}
private:

    bool scene_ok(const Scene& s,const std::vector<std::string>& args,
                  std::map<std::string,int>& schema,const std::string& new_head="") const {
        if(s.facts.size()>size_t(limits_.facts)) return false;
        std::set<std::string> entities;
        for(const auto& a:args) { if(!symbol(a)) return false; entities.insert(a); }
        for(const auto& a:s.facts) {
            if(!symbol(a.predicate)||heads_.count(a.predicate)||a.predicate==new_head||a.args.empty()||a.args.size()>size_t(limits_.arity)) return false;
            auto it=schema.find(a.predicate);
            if(it!=schema.end() && it->second!=int(a.args.size())) return false;
            schema[a.predicate]=int(a.args.size());
            for(const auto& v:a.args) { if(!symbol(v)) return false; entities.insert(v); }
        }
        return entities.size()<=size_t(limits_.entities)&&schema.size()+heads_.size()<=limits_.schemas;
    }
    Clause expand(const Clause& local,Meter& meter) const {
        Clause out{local.arity,{}}; int fresh=variables(local);
        for(const auto& a:local.body) {
            auto it=models_.find(a.predicate);
            if(it==models_.end()) { out.body.push_back(a); continue; }
            if(it->second.hypotheses.size()!=1) throw std::logic_error("ambiguous dependency in candidate");
            const Clause& def=it->second.hypotheses[0].expanded;
            std::vector<int> map(variables(def));
            for(int i=0;i<def.arity;++i) map[i]=a.vars[i];
            for(size_t i=def.arity;i<map.size();++i) map[i]=fresh++;
            for(auto b:def.body) { for(int& v:b.vars) v=map[v]; out.body.push_back(std::move(b)); }
        }
        std::set<int> used;
        for(const auto& a:out.body) for(int v:a.vars) used.insert(v);
        if(used.size()>size_t(limits_.variables)) throw OutsideClass{};
        return minimize(std::move(out),meter);
    }
    Formula expand_union(const Clause& local,Meter& meter)const {
        Formula products{{local.arity,{}}};int fresh=variables(local);
        for(const auto& literal:local.body) {
            Formula options;auto it=models_.find(literal.predicate);
            if(it==models_.end())options={{local.arity,{literal}}};
            else {
                if(it->second.hypotheses.size()!=1)throw std::logic_error("ambiguous union dependency");
                int used=0;
                for(const auto& definition:expanded_clauses(it->second.hypotheses.front())) {
                    Clause option{local.arity,{}};std::vector<int> map(variables(definition));
                    for(int i=0;i<definition.arity;++i)map[i]=literal.vars.at(i);
                    for(size_t i=definition.arity;i<map.size();++i)map[i]=fresh+int(i)-definition.arity;
                    used=std::max(used,int(map.size())-definition.arity);
                    for(auto atom:definition.body){for(int& v:atom.vars)v=map[v];option.body.push_back(std::move(atom));}
                    options.push_back(std::move(option));
                }
                fresh+=used;
            }
            if(products.size()*options.size()>size_t(limits_.expanded_clauses))throw OutsideClass{};
            Formula next;
            for(const auto& product:products)for(const auto& option:options) {
                meter.tick();Clause joined=product;joined.body.insert(joined.body.end(),option.body.begin(),option.body.end());next.push_back(std::move(joined));
            }
            products=std::move(next);
        }
        for(const auto& branch:products) {
            std::set<int> used;for(const auto& atom:branch.body)used.insert(atom.vars.begin(),atom.vars.end());
            if(used.size()>size_t(limits_.variables))throw OutsideClass{};
        }
        return normalize_formula(std::move(products),meter);
    }
    std::vector<Atom> available(const Scene& scene,Meter& meter) const {
        std::set<Atom> facts(scene.facts.begin(),scene.facts.end());
        if(reuse_) for(const auto& [name,model]:models_) if(model.hypotheses.size()==1) {
            for(const auto& c:expanded_clauses(model.hypotheses[0])) {
                bool stop=false;
                join(c,scene,std::vector<std::string>(variables(c)),0,meter,[&](const auto& b){
                    facts.insert({name,std::vector<std::string>(b.begin(),b.begin()+c.arity)});
                    if(facts.size()>size_t(limits_.facts)) throw Exhausted{};
                    return false;
                },stop);
            }
        }
        return {facts.begin(),facts.end()};
    }
    Model induce(const std::string& name) const {
        Model model; model.arity=heads_.at(name); Meter meter(limits_.work);
        try {
            std::vector<const Receipt*> examples;
            for(const auto& [id,r]:receipts_) if(r.active&&r.name==name) examples.push_back(&r);
            if(examples.empty()) return model;
            for(size_t i=0;i<examples.size();++i) for(size_t j=0;j<i;++j) {
                meter.tick();
                if(examples[i]->args==examples[j]->args && examples[i]->scene==examples[j]->scene && examples[i]->answer!=examples[j]->answer) {
                    model.status="conflicting_evidence"; model.work=meter.used; return model;
                }
            }
            const Receipt* witness=nullptr;
            for(const auto* r:examples) if(r->answer && std::set<std::string>(r->args.begin(),r->args.end()).size()==r->args.size()
                && (!witness || r->scene.facts.size()<witness->scene.facts.size())) witness=r;
            if(!witness) { model.status="no_distinct_positive_witness"; model.work=meter.used; return model; }
            auto atoms=available(witness->scene,meter);
            // Enumerate generic connected ground subgraphs of the smallest
            // positive witness. This bounds the hypothesis class explicitly.
            std::vector<size_t> choice;
            std::set<std::string> expanded_seen;
            for(int depth=1;depth<=limits_.body;++depth) {
                std::map<std::string,Hypothesis> consistent;
                std::function<void(size_t)> enumerate=[&](size_t start) {
                    if(int(choice.size())!=depth) {
                        for(size_t i=start;i<atoms.size();++i) { choice.push_back(i); enumerate(i+1); choice.pop_back(); }
                        return;
                    }
                    if(++model.candidates>limits_.candidates) throw Exhausted{};
                    meter.tick();
                    std::map<std::string,int> ids;
                    for(int i=0;i<model.arity;++i) ids[witness->args[i]]=i;
                    Clause local{model.arity,{}}; std::set<int> covered;
                    for(size_t i:choice) {
                        Literal a{atoms[i].predicate,{}};
                        for(const auto& v:atoms[i].args) { auto [it,inserted]=ids.emplace(v,int(ids.size())); a.vars.push_back(it->second); covered.insert(it->second); }
                        local.body.push_back(std::move(a));
                    }
                    for(int i=0;i<model.arity;++i) if(!covered.count(i)) return;
                    if(int(ids.size())>limits_.variables) return;
                    // Every component must be connected to at least one head.
                    std::set<int> reached; for(int i=0;i<model.arity;++i) reached.insert(i);
                    bool changed=true;
                    while(changed) { changed=false; for(const auto& a:local.body) if(std::any_of(a.vars.begin(),a.vars.end(),[&](int v){return reached.count(v);}))
                        for(int v:a.vars) changed|=reached.insert(v).second; }
                    if(reached.size()!=ids.size()) return;
                    local=canonical(local,meter);
                    Clause expanded;
                    try { expanded=expand(local,meter); } catch(const OutsideClass&) { return; }
                    auto k=key(expanded);
                    if(!expanded_seen.insert(k).second) return;
                    // Reject on a counterexample; all remaining candidates are
                    // still visited before accepting this shortest layer.
                    for(const auto* r:examples) if(proves(expanded,r->scene,r->args,meter)!=r->answer) return;
                    consistent[k]={std::move(local),std::move(expanded),{}, {}};
                };
                enumerate(0);
                if(!consistent.empty()) {
                    for(auto& [k,h]:consistent) model.hypotheses.push_back(std::move(h));
                    model.status=model.hypotheses.size()==1?"unique":"ambiguous";
                    model.work=meter.used; return model;
                }
            }
            model.status="no_bounded_hypothesis";
        } catch(const Exhausted&) { model.hypotheses.clear(); model.status="resource_exhausted"; }
        model.work=meter.used; return model;
    }
    Model induce_union(const std::string& name)const {
        Model model;model.arity=heads_.at(name);Meter meter(limits_.work);
        try {
            std::vector<const Receipt*> examples,positives,witnesses;
            for(const auto& [id,r]:receipts_)if(r.active&&r.name==name)examples.push_back(&r);
            if(examples.empty())return model;
            for(size_t i=0;i<examples.size();++i) {
                const auto* r=examples[i];
                for(size_t j=0;j<i;++j){meter.tick();if(r->args==examples[j]->args&&r->scene==examples[j]->scene&&r->answer!=examples[j]->answer){model.status="conflicting_evidence";model.work=meter.used;return model;}}
                if(r->answer){positives.push_back(r);if(std::set<std::string>(r->args.begin(),r->args.end()).size()==r->args.size())witnesses.push_back(r);}
            }
            if(witnesses.empty()){model.status="no_distinct_positive_witness";model.work=meter.used;return model;}
            struct Candidate { Clause local;Formula expanded;std::vector<bool> covers;size_t cost; };
            std::map<std::string,Candidate> unique;std::set<std::string> local_seen;
            for(const auto* witness:witnesses) {
                auto atoms=available(witness->scene,meter);std::vector<size_t> choice;
                for(int depth=1;depth<=limits_.body;++depth) {
                    std::function<void(size_t)> enumerate=[&](size_t start) {
                        if(int(choice.size())!=depth){for(size_t i=start;i<atoms.size();++i){choice.push_back(i);enumerate(i+1);choice.pop_back();}return;}
                        if(++model.candidates>limits_.candidates)throw Exhausted{};
                        meter.tick();
                        std::map<std::string,int> ids;for(int i=0;i<model.arity;++i)ids[witness->args[i]]=i;
                        Clause local{model.arity,{}};std::set<int> covered;
                        for(size_t i:choice){Literal atom{atoms[i].predicate,{}};for(const auto& value:atoms[i].args){auto [it,inserted]=ids.emplace(value,int(ids.size()));atom.vars.push_back(it->second);covered.insert(it->second);}local.body.push_back(std::move(atom));}
                        for(int i=0;i<model.arity;++i)if(!covered.count(i))return;
                        if(int(ids.size())>limits_.variables)return;
                        std::set<int> reached;for(int i=0;i<model.arity;++i)reached.insert(i);
                        bool changed=true;while(changed){changed=false;for(const auto& atom:local.body)if(std::any_of(atom.vars.begin(),atom.vars.end(),[&](int v){return reached.count(v);}))for(int v:atom.vars)changed|=reached.insert(v).second;}
                        if(reached.size()!=ids.size())return;
                        local=canonical(local,meter);if(!local_seen.insert(key(local)).second)return;
                        Formula expanded;try{expanded=expand_union(local,meter);}catch(const OutsideClass&){return;}
                        for(const auto* r:examples)if(!r->answer&&proves(expanded,r->scene,r->args,meter))return;
                        std::vector<bool> covers;bool any=false;
                        for(const auto* r:positives){bool yes=proves(expanded,r->scene,r->args,meter);covers.push_back(yes);any|=yes;}
                        if(!any)return;
                        auto k=formula_key(expanded);size_t cost=local.body.size()+1;auto old=unique.find(k);
                        if(old==unique.end()||cost<old->second.cost||(cost==old->second.cost&&key(local)<key(old->second.local)))unique[k]={std::move(local),std::move(expanded),std::move(covers),cost};
                    };
                    enumerate(0);
                }
            }
            std::vector<Candidate> candidates;for(auto& [key,candidate]:unique)candidates.push_back(std::move(candidate));
            size_t best_cost=~size_t{0};std::map<std::string,Hypothesis> consistent;std::set<std::vector<size_t>> visited;
            std::function<void(std::vector<size_t>,std::vector<bool>,size_t)> cover=[&](std::vector<size_t> selected,std::vector<bool> covered,size_t cost) {
                if(cost>best_cost||!visited.insert(selected).second)return;
                if(++model.candidates>limits_.candidates)throw Exhausted{};
                meter.tick();
                auto missing=std::find(covered.begin(),covered.end(),false);
                if(missing==covered.end()) {
                    Formula local,expanded;for(size_t i:selected){local.push_back(candidates[i].local);expanded.insert(expanded.end(),candidates[i].expanded.begin(),candidates[i].expanded.end());}
                    if(expanded.size()>size_t(limits_.expanded_clauses))return;
                    expanded=normalize_formula(std::move(expanded),meter);
                    std::sort(local.begin(),local.end(),[](const auto& a,const auto& b){return key(a)<key(b);});
                    if(cost<best_cost){best_cost=cost;consistent.clear();}
                    auto k=formula_key(expanded);if(!consistent.count(k))consistent[k]=hypothesis(std::move(local),std::move(expanded));return;
                }
                if(selected.size()>=size_t(limits_.clauses)||cost>=best_cost)return;
                size_t at=size_t(missing-covered.begin());
                for(size_t i=0;i<candidates.size();++i)if(candidates[i].covers[at]&&!std::binary_search(selected.begin(),selected.end(),i)) {
                    meter.tick();auto next=selected;next.insert(std::lower_bound(next.begin(),next.end(),i),i);auto coverage=covered;
                    for(size_t j=0;j<coverage.size();++j)coverage[j]=coverage[j]||candidates[i].covers[j];
                    cover(std::move(next),std::move(coverage),cost+candidates[i].cost);
                }
            };
            cover({},std::vector<bool>(positives.size(),false),0);
            for(auto& [key,h]:consistent)model.hypotheses.push_back(std::move(h));
            model.status=model.hypotheses.empty()?"no_bounded_hypothesis":model.hypotheses.size()==1?"unique":"ambiguous";
        }catch(const Exhausted&){model.hypotheses.clear();model.status="resource_exhausted";}
        model.work=meter.used;return model;
    }
public:
    explicit ConceptMemory(bool reuse=true,Limits limits={}):limits_(limits),reuse_(reuse) {
        if(limits.body<1||limits.variables<1||limits.arity<1||limits.facts<1||limits.entities<1||limits.clauses<1||limits.expanded_clauses<limits.clauses||!limits.receipts||!limits.schemas||!limits.candidates||!limits.work)
            throw std::invalid_argument("invalid concept limits");
    }
    const auto& models() const { return models_; }
    const auto& receipts() const { return receipts_; }
    const auto& base_schemas() const { return base_; }
    const auto& head_schemas() const { return heads_; }
    size_t schemas() const { return base_.size()+heads_.size(); }
    bool dirty() const { return dirty_; }
    bool teach(Receipt r) {
        if(!symbol(r.id)||!symbol(r.name)||receipts_.count(r.id)||receipts_.size()>=limits_.receipts||r.source!="verified_teacher"||r.generated||r.version!=1||!r.active
            ||!r.scene.complete||r.args.empty()||r.args.size()>size_t(limits_.arity)||base_.count(r.name)) return false;
        auto h=heads_.find(r.name); if(h!=heads_.end()&&h->second!=int(r.args.size())) return false;
        auto schema=base_; if(!scene_ok(r.scene,r.args,schema,r.name)||schema.size()+heads_.size()+size_t(h==heads_.end())>limits_.schemas) return false;
        std::sort(r.scene.facts.begin(),r.scene.facts.end()); r.scene.facts.erase(std::unique(r.scene.facts.begin(),r.scene.facts.end()),r.scene.facts.end());
        if(h==heads_.end()) { heads_[r.name]=int(r.args.size()); order_.push_back(r.name); }
        base_=std::move(schema); receipts_[r.id]=std::move(r); dirty_=true; return true;
    }
    // Optimistic receipt versioning: reject stale or untrusted edits atomically.
    bool revise(const std::string& id,unsigned expected,bool answer,bool active=true,const std::string& source="verified_teacher") {
        auto it=receipts_.find(id);
        if(it==receipts_.end()||it->second.version!=expected||source!="verified_teacher"||it->second.version==~0u) return false;
        it->second.answer=answer; it->second.active=active; ++it->second.version; dirty_=true; return true;
    }
    void rebuild() {
        models_.clear();
        for(const auto& name:order_) models_[name]=limits_.clauses==1?induce(name):induce_union(name);
        dirty_=false;
    }
    Answer predict(const std::string& name,const Scene& scene,const std::vector<std::string>& args) const {
        auto it=models_.find(name); if(dirty_||it==models_.end()||it->second.hypotheses.empty()||it->second.arity!=int(args.size())) return Answer::Unknown;
        auto schema=base_; if(!scene_ok(scene,args,schema)) return Answer::Unknown;
        Meter meter(limits_.work); int answer=-2;
        try {
            for(const auto& h:it->second.hypotheses) {
                int value=proves(expanded_clauses(h),scene,args,meter)?1:(scene.complete?0:-1);
                if(answer!=-2&&answer!=value) return Answer::Unknown;
                answer=value;
            }
        } catch(const Exhausted&) { return Answer::Unknown; }
        return Answer(answer);
    }
};
} // namespace laminar::concepts
