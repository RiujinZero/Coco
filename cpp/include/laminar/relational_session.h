#pragma once
// A continuing, versioned relational learner. Raw language and transition
// evidence is replayed against changing concept definitions; successful own
// action traces yield revalidated procedural candidates. Search/exploration
// remain supplied mechanisms, not a claim of learned computation control.
#include "concept_memory.h"
#include "request_grounding.h"
#include "composed_learner.h"
#include "state_io.h"
#include <array>
#include <optional>
#include <random>
#include <sstream>

namespace laminar::relational {
using concepts::Atom; using concepts::Scene; using concepts::Answer;
using reasoning::GroundedActuator; using reasoning::EventGrounding;
using Action=GroundedActuator::Action;
inline bool same_action(const Action& a,const Action& b) { return a.schema==b.schema&&a.arguments==b.arguments; }
inline Scene sorted(Scene s) {
    std::sort(s.facts.begin(),s.facts.end()); s.facts.erase(std::unique(s.facts.begin(),s.facts.end()),s.facts.end()); return s;
}
inline std::vector<std::string> words(const std::string& text,bool signs=false) {
    std::vector<std::string> out; std::string word;
    for(unsigned char c:text) {
        if(std::isalnum(c)||c=='_') word+=char(c);
        else {
            if(!word.empty()){out.push_back(word);word.clear();}
            if(signs&&(c=='-'||c=='+'))out.push_back(std::string(1,char(c)));
        }
    }
    if(!word.empty())out.push_back(word);
    return out;
}
struct Binding { std::string frame; std::vector<std::string> pattern,args,relations; std::vector<int> arg_types; };
inline std::optional<int> quantity(const std::string& s) {
    if(s.empty()||s.size()>9)return {};
    int n=0;for(unsigned char c:s){if(c<'0'||c>'9')return {};n=n*10+c-'0';}return n;
}
inline Binding abstract(const std::string& text,const std::set<std::string>& names) {
    Binding b; b.pattern=words(text);
    for(auto& w:b.pattern) if(names.count(w)) { b.args.push_back(w);w="$"+std::to_string(b.args.size()-1); }
    for(const auto& w:b.pattern)b.frame+=w+" ";
    return b;
}
inline Binding abstract_computation(const std::string& text,const std::set<std::string>& names,const std::set<std::string>& relations,const std::set<std::string>& quantities={}) {
    Binding b;b.pattern=words(text,true);b.frame="computed:";
    for(auto& w:b.pattern) {
        if(names.count(w)){b.args.push_back(w);b.arg_types.push_back(0);w="$"+std::to_string(b.args.size()-1);}
        else if(quantities.count(w)){b.args.push_back(w);b.arg_types.push_back(2);w="#"+std::to_string(b.args.size()-1);}
        else if(relations.count(w)){b.relations.push_back(w);w="%"+std::to_string(b.relations.size()-1);}
    }
    for(const auto& w:b.pattern)b.frame+=w+" ";
    return b;
}
inline std::optional<Binding> bind(const std::vector<std::string>& pattern,const std::string& text,bool computed=false) {
    Binding b; b.pattern=pattern; auto input=words(text,computed); if(input.size()!=pattern.size())return {};
    if(computed)b.frame="computed:";
    for(size_t i=0;i<pattern.size();++i) {
        if(computed&&!pattern[i].empty()&&pattern[i][0]=='#') {
            if(!quantity(input[i])||pattern[i]!="#"+std::to_string(b.args.size()))return {};
            b.args.push_back(input[i]);b.arg_types.push_back(2);
        } else if(!pattern[i].empty()&&pattern[i][0]=='$') {
            if(!concepts::symbol(input[i])||pattern[i]!="$"+std::to_string(b.args.size()))return {};
            b.args.push_back(input[i]);
            if(computed)b.arg_types.push_back(0);
        } else if(computed&&!pattern[i].empty()&&pattern[i][0]=='%') {
            if(!concepts::symbol(input[i])||pattern[i]!="%"+std::to_string(b.relations.size()))return {};
            b.relations.push_back(input[i]);
        } else if(input[i]!=pattern[i])return {};
        b.frame+=pattern[i]+" ";
    }
    return b;
}
struct LanguageReceipt {
    std::string id,text,source="verified_teacher";
    std::set<std::string> names;
    Scene scene;
    bool answer=false,active=true,generated=false;
    unsigned version=1;
    // Supplied typed mentions, never a program or a family/operation label.
    bool computational=false;
    std::set<std::string> relations;
    std::set<std::string> quantities;
};
struct TransitionReceipt {
    std::string id,source="verified_environment";
    Action action;
    Scene before,after;
    bool executed=true,active=true;
    unsigned version=1;
};
struct Interpretation {
    std::string predicate;
    std::vector<int> slots;
    bool negate=false;
    auto operator<=>(const Interpretation&)const=default;
};
struct Frame { std::vector<std::string> pattern; std::vector<Interpretation> meanings; size_t evidence=0; };
struct ComputationalFrame { std::vector<std::string> pattern; int cue=0; size_t evidence=0,states=0; bool learned=false; };
struct Operand {
    int argument=-1; // direct utterance argument, or unique binary lookup below
    std::string predicate;
    int source_slot=0,target_slot=1;
    auto operator<=>(const Operand&)const=default;
};
struct Instruction {
    std::string schema;
    std::vector<Operand> operands;
    auto operator<=>(const Instruction&)const=default;
};
struct Procedure {
    std::string frame;
    std::vector<Instruction> code;
    auto operator<=>(const Procedure&)const=default;
};
struct ExecutionReceipt {
    std::string id,text;
    Scene before;
    std::vector<Action> actions;
    std::vector<Scene> observed;
    bool success=false,active=true;
    unsigned version=1;
    // Interrupted action segments have observed transitions but no verified
    // terminal goal label. They must never become failed/successful goal lessons.
    bool goal_verified=true;
};
struct Plan {
    uint64_t token=0,version=0;
    std::string status;
    std::vector<Action> actions;
    size_t predictions=0;
    bool reused=false;
};

template<size_t ActionCount> class BasicControlledRelationalSession;
class RelationalSession {
    template<size_t ActionCount> friend class BasicControlledRelationalSession;
    concepts::ConceptMemory concepts_;
    std::map<std::string,int> schema_,binary_ids_;
    std::map<std::string,LanguageReceipt> language_;
    std::map<std::string,TransitionReceipt> transitions_;
    std::map<std::string,ExecutionReceipt> executions_;
    std::map<std::string,std::string> transition_owner_;
    std::map<std::string,Frame> frames_;
    std::map<std::string,ComputationalFrame> computational_frames_;
    // Inference updates only scratch buffers and the work counter. Value
    // ownership ensures evaluation clones cannot share mutable neural state.
    mutable reasoning::PrimitiveStore primitives_;
    reasoning::ComposedLearner composed_;
    uint64_t primitive_generation_=0;
    int composed_budget_=0;
    reasoning::SearchProgress search_progress_;
    bool observed_relation_types_=false;
    std::map<std::string,int> computation_cues_;
    GroundedActuator motors_;
    std::set<Procedure> procedures_;
    std::mt19937 rng_;
    bool dirty_=false;
    uint64_t version_=0,next_token_=1;
    struct Pending { Plan plan; std::string text; Scene before; };
    std::optional<Pending> pending_;
    size_t evidence_budget=4096,procedure_budget=256;
public:
    // search_progress_ is only an optional log sink, not learning state.
    template<class Ar> void state_io(Ar& ar) {
        ar(concepts_);ar(schema_);ar(binary_ids_);ar(language_);ar(transitions_);ar(executions_);ar(transition_owner_);ar(frames_);
        ar(computational_frames_);ar(primitives_);ar(composed_);ar(primitive_generation_);ar(composed_budget_);ar(observed_relation_types_);
        ar(computation_cues_);ar(motors_);ar(procedures_);ar(rng_);ar(dirty_);ar(version_);ar(next_token_);ar(pending_);ar(evidence_budget);ar(procedure_budget);
    }
private:
    static constexpr size_t language_work=1000000;

    bool validate(const Scene& scene,const std::vector<std::string>& args,std::map<std::string,int>& schemas) const {
        if(scene.facts.size()>256||args.size()>32)return false;
        std::set<std::string> entities(args.begin(),args.end());
        for(const auto& s:args)if(!concepts::symbol(s))return false;
        for(const auto& a:scene.facts) {
            if(!concepts::symbol(a.predicate)||concepts_.head_schemas().count(a.predicate)||a.args.empty()||a.args.size()>8)return false;
            auto old=schemas.find(a.predicate);if(old!=schemas.end()&&old->second!=int(a.args.size()))return false;
            schemas[a.predicate]=int(a.args.size());
            for(const auto& x:a.args) {if(!concepts::symbol(x))return false;entities.insert(x);}
        }
        return entities.size()<=32&&schemas.size()+concepts_.head_schemas().size()<=128;
    }
    void changed(std::map<std::string,int> schemas) {
        schema_=std::move(schemas);
        for(const auto& [name,n]:schema_) if(n==2&&!binary_ids_.count(name))binary_ids_[name]=int(binary_ids_.size());
        ++version_;dirty_=true; // pending decisions retain their old version and become unusable
    }
    reasoning::RequestGrounding::Snapshot snapshot(const Scene& scene)const {
        reasoning::RequestGrounding::State state;state.closed_world=scene.complete;
        for(const auto& a:scene.facts)if(a.args.size()==2&&binary_ids_.count(a.predicate))state.facts.emplace(a.args[0],binary_ids_.at(a.predicate),a.args[1]);
        return {{state},true};
    }
    std::optional<Binding> parse(const std::string& text)const {
        std::optional<Binding> result;
        for(const auto& [name,f]:frames_)if(auto b=bind(f.pattern,text)) {if(result)return {};result=std::move(b);}
        for(const auto& [name,f]:computational_frames_)if(auto b=bind(f.pattern,text,true)) {if(result)return {};result=std::move(b);}
        return result;
    }
    std::optional<Binding> parse_evidence(const std::string& text)const {
        // Mention types are observed teaching data. They are available before
        // choosing to rebuild the semantics, and cannot depend on knowing the
        // answer. Duplicate lessons of one pattern are one structural match.
        std::map<std::string,Binding> matches;
        for(const auto& [id,r]:language_)if(r.active) {
            auto pattern=r.computational?abstract_computation(r.text,r.names,r.relations,r.quantities):abstract(r.text,r.names);
            if(auto b=laminar::relational::bind(pattern.pattern,text,r.computational))matches.emplace(b->frame,*b);
        }
        return matches.size()==1?std::optional<Binding>{matches.begin()->second}:std::nullopt;
    }
    bool computation_ok(const LanguageReceipt& r,const Binding& b,std::map<std::string,int>& schemas)const {
        if(!r.computational)return r.relations.empty()&&r.quantities.empty();
        if(b.args.size()!=2||b.relations.size()!=2)return false;
        for(const auto& n:r.quantities)if(!quantity(n)||r.names.count(n)||r.relations.count(n)||std::find(b.args.begin(),b.args.end(),n)==b.args.end())return false;
        for(const auto& name:r.relations) {
            if(!concepts::symbol(name)||r.names.count(name))return false;
            auto head=concepts_.head_schemas().find(name);
            if(head!=concepts_.head_schemas().end()){if(head->second!=2)return false;}
            else {auto old=schemas.find(name);if(old!=schemas.end()&&old->second!=2)return false;schemas[name]=2;}
        }
        return schemas.size()+concepts_.head_schemas().size()<=128;
    }
    std::optional<reasoning::FlatQuery> computational_query(const Binding& b,const Scene& world,int cue)const {
        if(!world.complete||b.args.size()!=2||b.relations.size()!=2)return {};
        std::set<std::string> names;
        for(size_t i=0;i<b.args.size();++i)if(b.arg_types.at(i)==0)names.insert(b.args[i]);
        for(const auto& a:world.facts)names.insert(a.args.begin(),a.args.end());
        if(names.size()>32)return {};
        std::map<std::string,int> entities,relations;
        for(const auto& n:names)entities[n]=int(entities.size());
        for(const auto& n:b.relations)if(!relations.count(n))relations[n]=int(relations.size());
        auto value=[&](size_t i){return b.arg_types.at(i)==2?reasoning::Value{2,*quantity(b.args[i])}:reasoning::Value{0,entities.at(b.args[i])};};
        reasoning::FlatQuery q;q.cue=cue;q.a=value(0);q.b=value(1);
        q.relation[0]=relations.at(b.relations[0]);q.relation[1]=relations.at(b.relations[1]);
        for(const auto& [name,r]:relations) {
            auto model=concepts_.models().find(name);
            if(model!=concepts_.models().end()) {
                if(model->second.arity!=2)return {};
                // Derived tuples are temporary computation inputs. They never
                // enter the observed scene or the raw evidence journal.
                for(const auto& [a,x]:entities)for(const auto& [b,y]:entities) {
                    auto truth=concepts_.predict(name,world,{a,b});
                    if(truth==Answer::Unknown)return {};
                    if(truth==Answer::True)q.facts.push_back({x,r,y});
                }
            } else {
                auto schema=schema_.find(name);
                bool binary=schema!=schema_.end()&&schema->second==2;
                // A typed public tuple can ground a new raw predicate name.
                // It cannot define an absent predicate or bypass an unresolved
                // learned concept. This read does not create teaching evidence.
                if(schema==schema_.end()&&observed_relation_types_&&!concepts_.head_schemas().count(name))
                    for(const auto& atom:world.facts)if(atom.predicate==name&&atom.args.size()==2){binary=true;break;}
                if(!binary)return {};
                for(const auto& a:world.facts)if(a.predicate==name)q.facts.push_back({entities.at(a.args[0]),r,entities.at(a.args[1])});
            }
        }
        return q;
    }
    void rebuild_computations() {
        search_progress_.emit("materialization_begin");
        std::map<std::string,reasoning::Program> previous;
        for(const auto& [name,f]:computational_frames_) {
            auto p=primitives_.memory.procedures.find(f.cue);
            if(p!=primitives_.memory.procedures.end())previous[name]=p->second;
        }
        computational_frames_.clear();primitives_.memory.procedures.clear();
        std::map<std::string,std::vector<std::pair<const LanguageReceipt*,Binding>>> groups;
        for(const auto& [id,r]:language_)if(r.active&&r.computational) {
            auto b=abstract_computation(r.text,r.names,r.relations,r.quantities);groups[b.frame].push_back({&r,std::move(b)});
        }
        std::vector<reasoning::ComposedReceipt> composed_evidence;
        for(const auto& [name,rows]:groups) {
            ComputationalFrame f;f.pattern=rows.front().second.pattern;f.evidence=rows.size();
            if(composed_budget_) {
                if(!computation_cues_.count(name))computation_cues_[name]=int(computation_cues_.size());
                f.cue=computation_cues_.at(name);
            } else f.cue=int(computational_frames_.size());
            std::vector<reasoning::VerifiedQuestion> examples;
            for(const auto& [r,b]:rows) {
                // Keep raw unsupported lessons for later revision/rebuild, but
                // do not let one unresolved relation erase other usable lessons
                // of this generic language pattern. The query adapter still
                // withholds whenever the requested relations are unresolved.
                auto q=computational_query(b,r->scene,f.cue);if(!q)continue;
                examples.push_back({*q,{1,int(r->answer)}});
                if(composed_budget_) {
                    reasoning::ComposedReceipt derived;derived.id=r->id;derived.source=r->source;derived.version=r->version;derived.question=examples.back();
                    composed_evidence.push_back(std::move(derived));
                }
            }
            if(!composed_budget_&&!examples.empty()) {
                auto prior=previous.find(name);bool valid=prior!=previous.end();
                if(valid)for(const auto& e:examples)if(primitives_.memory.execute(e.query,prior->second)!=e.answer){valid=false;break;}
                if(valid){primitives_.memory.procedures[f.cue]=prior->second;f.learned=true;}
                else {auto result=primitives_.memory.learn(examples,false,10,50000);f.states=result.states;f.learned=result.found;}
            }
            computational_frames_[name]=std::move(f);
        }
        if(composed_budget_) {
            if(!composed_.synchronize(primitives_,primitive_generation_,composed_evidence))throw std::runtime_error("invalid composed language evidence");
            search_progress_.emit("materialization_end");
            composed_.rebuild(composed_budget_);primitives_.memory=composed_.memory();
            search_progress_.emit("rebuild_end");
            for(auto& [name,f]:computational_frames_)f.learned=primitives_.memory.procedures.count(f.cue);
        }
    }
    void type_observed_language(LanguageReceipt& r,const Binding& b)const {
        r.computational=!b.relations.empty();r.relations={b.relations.begin(),b.relations.end()};
        if(r.computational){r.names.clear();r.quantities.clear();}
        for(size_t i=0;i<b.args.size();++i) {
            if(r.computational&&b.arg_types.at(i)==2)r.quantities.insert(b.args[i]);else r.names.insert(b.args[i]);
        }
        for(const auto& name:r.relations)r.names.erase(name);
    }
    // Sound relevance pruning for this model class: effects have no learned
    // preconditions and concepts are positive rule unions (optionally negated
    // by a language frame). An action that cannot change any matching goal
    // literal cannot enable satisfaction indirectly through another action.
    bool contributes(const Action& action,const Binding& binding)const {
        auto frame=motors_.frames.find(action.schema);
        if(frame==motors_.frames.end()||frame->second.arity!=action.arguments.size())return false;
        // Computational goals need conservative search: a count or composition
        // can depend on effects that conjunctive-literal pruning would discard.
        if(computational_frames_.count(binding.frame))return true;
        for(const auto& meaning:frames_.at(binding.frame).meanings) {
            std::vector<std::string> head;for(int slot:meaning.slots)head.push_back(binding.args[slot]);
            for(const auto& hypothesis:concepts_.models().at(meaning.predicate).hypotheses)
            for(const auto& clause:concepts::expanded_clauses(hypothesis))for(const auto& literal:clause.body) {
                if(literal.vars.size()!=2||!binary_ids_.count(literal.predicate))continue;
                for(const auto& effect:frame->second.candidates)if(effect.relation==binary_ids_.at(literal.predicate)) {
                    if((!meaning.negate&&effect.effect==EventGrounding::REMOVE)||(meaning.negate&&effect.effect==EventGrounding::ADD))continue;
                    std::array<std::string,2> values{action.arguments[effect.subject],action.arguments[effect.object]};
                    if(meaning.negate&&effect.effect==EventGrounding::SET_SUBJECT)values[1].clear();
                    if(meaning.negate&&effect.effect==EventGrounding::SET_OBJECT)values[0].clear();
                    bool match=true;
                    for(int side=0;side<2;++side)if(!values[side].empty()&&literal.vars[side]<clause.arity&&head[literal.vars[side]]!=values[side])match=false;
                    if(match)return true;
                }
            }
        }
        return false;
    }
    int evaluate(const Interpretation& m,const Scene& scene,const std::vector<std::string>& args)const {
        std::vector<std::string> head;for(int slot:m.slots)head.push_back(args.at(slot));
        int value=int(concepts_.predict(m.predicate,scene,head)); return value<0?value:(m.negate?1-value:value);
    }
    std::string meaning_key(const Interpretation& m)const {
        const auto& definitions=concepts_.models().at(m.predicate).hypotheses;
        if(definitions.size()!=1)return {};
        concepts::Meter meter(language_work);
        std::vector<std::string> branches;
        for(auto c:concepts::expanded_clauses(definitions.front())) {
            for(auto& a:c.body)for(int& v:a.vars)if(v<c.arity)v=m.slots[v];
            branches.push_back(concepts::key(concepts::canonical(c,meter)));
        }
        std::sort(branches.begin(),branches.end());
        std::string result=std::to_string(m.negate);
        for(size_t i=0;i<branches.size();++i){if(i)result+='|';result+=branches[i];}
        return result;
    }
    void rebuild_language() {
        frames_.clear();std::map<std::string,std::vector<std::pair<const LanguageReceipt*,Binding>>> groups;
        for(const auto& [id,r]:language_)if(r.active&&!r.computational) {auto b=abstract(r.text,r.names);groups[b.frame].push_back({&r,std::move(b)});}
        for(const auto& [name,rows]:groups) {
            Frame f;f.pattern=rows.front().second.pattern;f.evidence=rows.size();
            {
                size_t work=0;std::map<std::string,Interpretation> unique;
                try {
                    int n=int(rows.front().second.args.size());
                    // A new unresolved definition is not an interpretation of
                    // every existing frame. Ground language only in resolved
                    // definitions; query-time agreement still handles multiple
                    // surviving language interpretations.
                    for(const auto& [predicate,model]:concepts_.models())if(model.arity==n&&model.hypotheses.size()==1) {
                        std::vector<int> perm;for(int i=0;i<n;++i)perm.push_back(i);
                        do { for(bool negate:{false,true}) {
                            Interpretation m{predicate,perm,negate};bool fits=true;
                            for(const auto& [r,b]:rows) {
                                if(++work>language_work)throw concepts::Exhausted{};
                                if(evaluate(m,r->scene,b.args)!=int(r->answer)){fits=false;break;}
                            }
                            if(fits)unique.emplace(meaning_key(m),m);
                        }}while(std::next_permutation(perm.begin(),perm.end()));
                    }
                    for(auto& [key,m]:unique)f.meanings.push_back(std::move(m));
                }catch(const concepts::Exhausted&) {f.meanings.clear();}
            }
            frames_[name]=std::move(f);
        }
    }
    std::optional<std::string> resolve(const Operand& o,const Binding& b,const Scene& scene)const {
        if(o.argument<0||size_t(o.argument)>=b.args.size())return {};
        if(o.predicate.empty())return b.args[o.argument];
        if(!scene.complete)return {};
        std::set<std::string> values;
        for(const auto& a:scene.facts)if(a.predicate==o.predicate&&a.args.size()==2&&a.args[o.source_slot]==b.args[o.argument])values.insert(a.args[o.target_slot]);
        if(values.size()!=1)return {};
        return *values.begin();
    }
    std::optional<Operand> derive(const std::string& value,const Binding& b,const Scene& scene)const {
        for(size_t i=0;i<b.args.size();++i)if(b.args[i]==value)return Operand{int(i),"",0,1};
        for(const auto& [pred,n]:schema_)if(n==2)for(size_t i=0;i<b.args.size();++i)for(int side:{0,1}) {
            Operand o{int(i),pred,side,1-side};auto result=resolve(o,b,scene);if(result&&*result==value)return o;
        }
        return {};
    }
    void rebuild_procedures() {
        procedures_.clear();
        for(const auto& [id,r]:executions_)if(r.active&&r.goal_verified&&r.success&&!r.actions.empty()) {
            auto b=parse(r.text);if(!b||answer(r.text,r.observed.back())!=Answer::True)continue;
            Scene world=r.before;Procedure p{b->frame,{}};bool valid=true;
            for(size_t i=0;i<r.actions.size();++i) {
                if(!transitions_.at(id+"_step"+std::to_string(i)).executed) {
                    if(world!=r.observed[i]){valid=false;break;}
                    continue;
                }
                auto predicted=predict(r.actions[i],world);
                if(!predicted||*predicted!=r.observed[i]) {valid=false;break;}
                Instruction op;op.schema=r.actions[i].schema;
                for(const auto& arg:r.actions[i].arguments) {
                    auto operand=derive(arg,*b,world);if(!operand){valid=false;break;}op.operands.push_back(*operand);
                }
                if(!valid)break;
                p.code.push_back(std::move(op));world=r.observed[i];
            }
            if(valid&&!p.code.empty())procedures_.insert(std::move(p));
            if(procedures_.size()>procedure_budget){procedures_.clear();return;}
        }
    }
    bool transition_ok(const TransitionReceipt& r,std::map<std::string,int>& schemas)const {
        if(!concepts::symbol(r.id)||!concepts::symbol(r.action.schema)||r.action.arguments.empty()||r.action.arguments.size()>4||!r.before.complete||!r.after.complete
           ||!validate(r.before,r.action.arguments,schemas)||!validate(r.after,r.action.arguments,schemas))return false;
        std::set<Atom> before,after;
        for(const auto& a:r.before.facts)if(a.args.size()!=2)before.insert(a);
        for(const auto& a:r.after.facts)if(a.args.size()!=2)after.insert(a);
        return before==after&&(r.executed||sorted(r.before)==sorted(r.after));
    }
    bool affordances_ok(const Scene& world,const std::vector<Action>& available)const {
        if(available.size()>128)return false;
        auto schemas=schema_;
        for(const auto& a:available)if(!concepts::symbol(a.schema)||a.arguments.empty()||a.arguments.size()>4||!validate(world,a.arguments,schemas))return false;
        return true;
    }
public:
    explicit RelationalSession(unsigned seed=0,size_t receipts=4096,size_t procedures=256,reasoning::SkillShape shape={}):primitives_(seed,shape),composed_(seed,shape),rng_(seed),evidence_budget(receipts),procedure_budget(procedures) {
        if(!receipts||!procedures)throw std::invalid_argument("invalid session capacity");
    }
    const auto& concepts()const{return concepts_;}
    const auto& language_receipts()const{return language_;}
    const auto& transition_receipts()const{return transitions_;}
    const auto& execution_receipts()const{return executions_;}
    const auto& frames()const{return frames_;}
    const auto& computational_frames()const{return computational_frames_;}
    const auto& primitives()const{return primitives_;}
    const auto& composed()const{return composed_;}
    void enable_disjunctive_concepts(concepts::Limits limits) {
        if(version_)throw std::logic_error("concept class must be configured before evidence");
        if(limits.clauses<2)throw std::invalid_argument("disjunctive concepts need at least two clauses");
        concepts_=concepts::ConceptMemory(true,limits);
    }
    void search_progress(std::ostream* out){search_progress_.out=out;composed_.search_progress(out);}
    void enable_observed_relation_types() {
        if(version_)throw std::logic_error("relation typing must be configured before evidence");
        observed_relation_types_=true;
    }
    void enable_composed_learning(int budget=100000,bool repair_available=false,int interleave_quantum=0,bool repair_bank=false) {
        if(budget<1||composed_budget_)throw std::invalid_argument("invalid composed learning configuration");
        composed_.configure_interleaving(interleave_quantum);
        if(repair_bank)composed_.enable_repair_bank();
        else if(repair_available)composed_.enable_repair();
        composed_budget_=budget;changed(schema_);
    }
    uint64_t primitive_steps()const{return primitives_.memory.skill_steps;}
    std::optional<reasoning::FlatQuery> computation_input(const std::string& text,const Scene& world)const {
        if(!ready())return {};
        auto schemas=schema_;if(!validate(world,{},schemas))return {};
        auto b=parse(text);if(!b)return {};
        auto f=computational_frames_.find(b->frame);if(f==computational_frames_.end())return {};
        return computational_query(*b,world,f->second.cue);
    }
    const auto& motors()const{return motors_;}
    const auto& procedures()const{return procedures_;}
    const auto& binary_ids()const{return binary_ids_;}
    uint64_t version()const{return version_;}
    bool dirty()const{return dirty_;}
    // Rebuilt does not mean every concept is known. Each language frame and
    // computational relation checks its own supporting meanings.
    bool ready()const{return !dirty_;}
    bool teach_primitive(reasoning::PrimitiveReceipt r) {
        if(!primitives_.teach(std::move(r)))return false;
        ++primitive_generation_;
        changed(schema_);return true;
    }
    bool revise_primitive(const std::string& id,unsigned expected,reasoning::Value answer,bool active=true,const std::string& source="verified_teacher") {
        if(!primitives_.revise(id,expected,answer,active,source))return false;
        ++primitive_generation_;
        changed(schema_);return true;
    }
    bool teach_concept(concepts::Receipt r) {
        auto schemas=schema_;if(schemas.count(r.name)||!validate(r.scene,r.args,schemas)||!concepts_.teach(std::move(r)))return false;
        changed(std::move(schemas));return true;
    }
    bool revise_concept(const std::string& id,unsigned expected,bool value,bool active=true,const std::string& source="verified_teacher") {
        if(!concepts_.revise(id,expected,value,active,source))return false;
        changed(schema_);return true;
    }
    bool teach_language(LanguageReceipt r) {
        auto schemas=schema_;auto b=r.computational?abstract_computation(r.text,r.names,r.relations,r.quantities):abstract(r.text,r.names);
        if(!concepts::symbol(r.id)||r.text.size()>2048||b.pattern.size()>64||b.args.empty()||b.args.size()>8||r.version!=1||!r.active||r.generated||r.source!="verified_teacher"
           ||language_.count(r.id)||executions_.count(r.id)||language_.size()>=evidence_budget||!r.scene.complete||!validate(r.scene,{r.names.begin(),r.names.end()},schemas)||!computation_ok(r,b,schemas))return false;
        // Keep the actual teaching records, including repetitions and order.
        // Canonicalizing here hides duplicate-handling errors from primitive
        // acquisition even though answer() can receive a noncanonical scene.
        language_[r.id]=std::move(r);changed(std::move(schemas));return true;
    }
    bool revise_language(const std::string& id,unsigned expected,bool value,bool active=true,const std::string& source="verified_teacher") {
        auto it=language_.find(id);if(executions_.count(id)||it==language_.end()||it->second.version!=expected||expected==~0u||it->second.source!=source)return false;
        it->second.answer=value;it->second.active=active;++it->second.version;changed(schema_);return true;
    }
    // Externally observed transitions can teach effects, never procedures. Only
    // feedback for a live decision issued below can create an execution receipt.
    bool teach_transition(TransitionReceipt r) {
        auto schemas=schema_;
        if(r.source!="verified_environment"||r.version!=1||!r.active||transitions_.count(r.id)||transitions_.size()>=evidence_budget||!transition_ok(r,schemas))return false;
        r.before=sorted(r.before);r.after=sorted(r.after);transitions_[r.id]=std::move(r);changed(std::move(schemas));return true;
    }
    bool revise_transition(const std::string& id,unsigned expected,const Scene& after,bool active=true,const std::string& source="verified_environment") {
        auto it=transitions_.find(id);if(transition_owner_.count(id)||it==transitions_.end()||it->second.version!=expected||expected==~0u||it->second.source!=source)return false;
        auto r=it->second;auto schemas=schema_;r.after=sorted(after);r.active=active;
        if(!transition_ok(r,schemas))return false;
        ++r.version;it->second=std::move(r);changed(std::move(schemas));return true;
    }
    void rebuild() {
        concepts_.rebuild();
        primitives_.rebuild();
        motors_={};std::set<int> relations;for(const auto& [name,id]:binary_ids_)relations.insert(id);
        uint64_t scene=1;
        for(const auto& [id,r]:transitions_)if(r.active)motors_.teach(r.action,snapshot(r.before),snapshot(r.after),relations,r.executed,scene++);
        rebuild_language();rebuild_computations();dirty_=false;rebuild_procedures();
    }
    Answer answer(const std::string& text,const Scene& world)const {
        if(!ready())return Answer::Unknown;
        auto schemas=schema_;if(!validate(world,{},schemas))return Answer::Unknown;
        auto b=parse(text);if(!b)return Answer::Unknown;
        auto computation=computational_frames_.find(b->frame);
        if(computation!=computational_frames_.end()) {
            auto q=computational_query(*b,world,computation->second.cue);if(!q)return Answer::Unknown;
            auto value=primitives_.memory.answer(*q);
            return value.type==1&&(value.id==0||value.id==1)?Answer(value.id):Answer::Unknown;
        }
        const auto& meanings=frames_.at(b->frame).meanings;int result=-2;
        for(const auto& m:meanings) {int value=evaluate(m,world,b->args);if(value<0||(result!=-2&&result!=value))return Answer::Unknown;result=value;}
        return result==-2?Answer::Unknown:Answer(result);
    }
    std::optional<Scene> predict(const Action& action,const Scene& before)const {
        if(dirty_||!before.complete)return {};
        auto schema=schema_;if(!validate(before,action.arguments,schema))return {};
        auto f=motors_.frames.find(action.schema);
        if(f==motors_.frames.end()||f->second.arity!=action.arguments.size()||f->second.candidates.empty())return {};
        std::optional<Scene> result;
        for(const auto& rule:f->second.candidates) {
            auto updated=EventGrounding::update(snapshot(before).worlds.front().facts,rule.relation,rule.effect,action.arguments[rule.subject],action.arguments[rule.object]);
            Scene next;next.complete=true;for(const auto& a:before.facts)if(a.args.size()!=2||!binary_ids_.count(a.predicate))next.facts.push_back(a);
            for(const auto& [a,r,b]:updated)for(const auto& [name,id]:binary_ids_)if(id==r){next.facts.push_back({name,{a,b}});break;}
            next=sorted(next);if(result&&*result!=next)return {};result=std::move(next);
        }
        return result;
    }
    Plan plan(const std::string& text,const Scene& initial,const std::vector<Action>& available,size_t budget=32768,int depth=6,bool reuse=true)const {
        Plan out;out.version=version_;
        if(depth<0||depth>16||!affordances_ok(initial,available)){out.status="invalid_search_request";return out;}
        auto truth=answer(text,initial);if(truth==Answer::True){out.status="satisfied";return out;}
        if(truth==Answer::Unknown||!initial.complete){out.status="unavailable_meaning_or_world";return out;}
        auto binding=parse(text);if(!binding){out.status="unknown_language";return out;}
        if(reuse)for(const auto& p:procedures_)if(p.frame==binding->frame&&int(p.code.size())<=depth) {
            Scene world=sorted(initial);std::vector<Action> actions;bool okay=true;
            for(const auto& op:p.code) {
                Action action;action.schema=op.schema;
                for(const auto& operand:op.operands){auto v=resolve(operand,*binding,world);if(!v){okay=false;break;}action.arguments.push_back(*v);}
                if(!okay||std::none_of(available.begin(),available.end(),[&](const auto& a){return same_action(a,action);})){okay=false;break;}
                if(out.predictions>=budget){out.status="search_budget";return out;}++out.predictions;
                auto next=predict(action,world);if(!next){okay=false;break;}
                // A completed prefix must not be re-issued forever when a
                // reactive body executes one action and then replans. Effects
                // have no hidden side effects in this model; unchanged complete
                // worlds make these instructions observational no-ops.
                if(*next!=world)actions.push_back(std::move(action));
                world=std::move(*next);
            }
            if(okay&&answer(text,world)==Answer::True){out.actions=std::move(actions);out.reused=true;out.status="procedure";return out;}
        }
        struct Node {Scene world;std::vector<Action> actions;};
        std::vector<Node> queue{{sorted(initial),{}}};std::set<Scene> visited{sorted(initial)};
        for(size_t i=0;i<queue.size();++i) {
            // Copy before append: vector growth must not invalidate the node.
            auto current=queue[i];if(int(current.actions.size())>=depth)continue;
            for(const auto& action:available) {
                if(!contributes(action,*binding))continue;
                if(out.predictions>=budget){out.status="search_budget";return out;}++out.predictions;
                auto next=predict(action,current.world);if(!next||!visited.insert(*next).second)continue;
                auto actions=current.actions;actions.push_back(action);
                if(answer(text,*next)==Answer::True){out.actions=std::move(actions);out.status="searched";return out;}
                if(queue.size()>=4096){out.status="state_budget";return out;}
                queue.push_back({std::move(*next),std::move(actions)});
            }
        }
        out.status="no_plan";return out;
    }
    Plan exploration(const Scene& world,const std::vector<Action>& available) {
        Plan result;auto schema=schema_;
        if(dirty_||!world.complete||available.empty()||!validate(world,{},schema)||!affordances_ok(world,available)){result.status="unavailable_exploration";return result;}
        result.version=version_;result.status="explore";
        size_t least=~size_t(0);std::vector<size_t> choices;
        for(size_t i=0;i<available.size();++i) {
            auto f=motors_.frames.find(available[i].schema);size_t n=f==motors_.frames.end()?0:f->second.scenes.size();
            if(n<least){least=n;choices.clear();}if(n==least)choices.push_back(i);
        }
        result.actions.push_back(available[choices[rng_()%choices.size()]]);return result;
    }
    Plan issue(const std::string& text,const Scene& world,const std::vector<Action>& available,bool explore=false,size_t budget=32768,int depth=6,bool reuse=true) {
        Plan result;
        if(pending_&&pending_->plan.version==version_){result.status="awaiting_feedback";return result;}
        pending_.reset();
        if(explore)result=exploration(world,available);
        else result=plan(text,world,available,budget,depth,reuse);
        if(result.actions.empty())return result;
        result.token=next_token_++;pending_=Pending{result,text,sorted(world)};return result;
    }
    bool feedback(uint64_t token,const std::vector<Scene>& observed,const std::vector<bool>& executed,std::optional<bool> success,
                  const std::string& source="verified_environment",bool generated=false) {
        if(!pending_||pending_->plan.token!=token||pending_->plan.version!=version_||source!="verified_environment"||generated)return false;
        const auto& p=*pending_;
        if(observed.size()!=p.plan.actions.size()||executed.size()!=observed.size()||observed.empty()||executions_.size()>=evidence_budget||transitions_.size()+observed.size()>evidence_budget)return false;
        auto schemas=schema_;Scene before=p.before;std::vector<TransitionReceipt> transitions;
        std::vector<Scene> canonical;
        for(size_t i=0;i<observed.size();++i) {
            TransitionReceipt r;r.id="execution"+std::to_string(token)+"_step"+std::to_string(i);r.action=p.plan.actions[i];r.before=before;r.after=sorted(observed[i]);r.executed=executed[i];
            if(transitions_.count(r.id)||!transition_ok(r,schemas))return false;
            before=r.after;canonical.push_back(before);transitions.push_back(std::move(r));
        }
        const std::string id="execution"+std::to_string(token);
        if(executions_.count(id)||language_.count(id))return false;
        // The host's terminal satisfaction bit is also raw language evidence.
        // It labels the observed resulting world, never the predicted world.
        std::optional<LanguageReceipt> language;
        if(auto binding=parse(p.text);binding&&success.has_value()) {
            if(language_.size()>=evidence_budget)return false;
            LanguageReceipt r;r.id=id;r.text=p.text;r.source=source;r.scene=canonical.back();r.answer=*success;
            r.names.insert(binding->args.begin(),binding->args.end());
            for(const auto& atom:r.scene.facts)r.names.insert(atom.args.begin(),atom.args.end());
            type_observed_language(r,*binding);
            if(!validate(r.scene,{r.names.begin(),r.names.end()},schemas)||!computation_ok(r,*binding,schemas))return false;
            language=std::move(r);
        }
        executions_[id]={id,p.text,p.before,p.plan.actions,std::move(canonical),success.value_or(false),true,1,success.has_value()};
        for(auto& r:transitions){transition_owner_[r.id]=id;transitions_[r.id]=std::move(r);}
        if(language)language_[id]=std::move(*language);
        pending_.reset();changed(std::move(schemas));return true;
    }
    bool revise_execution(const std::string& id,unsigned expected,std::optional<bool> success,bool active=true,const std::string& source="verified_environment") {
        auto it=executions_.find(id);if(it==executions_.end()||it->second.version!=expected||expected==~0u||source!="verified_environment"||(!success.has_value()&&(it->second.goal_verified||language_.count(id))))return false;
        it->second.success=success.value_or(false);it->second.goal_verified=success.has_value();it->second.active=active;++it->second.version;
        auto language=language_.find(id);
        if(language!=language_.end()){language->second.answer=*success;language->second.active=active;++language->second.version;}
        changed(schema_);return true;
    }
    // An observation correction to an own execution is a bundle: update the
    // observed states, each dependent transition pre-state, and the paired
    // terminal language receipt together. Editing one derived step in isolation
    // would otherwise leave contradictory copies of its world in the journal.
    bool revise_execution_observations(const std::string& id,unsigned expected,const Scene& before,
            const std::vector<Scene>& observed,const std::vector<bool>& executed,std::optional<bool> success,
            const std::string& source="verified_environment") {
        auto it=executions_.find(id);
        if(it==executions_.end()||it->second.version!=expected||expected==~0u||source!="verified_environment"
           ||observed.size()!=it->second.actions.size()||executed.size()!=observed.size()||observed.empty()||(!success.has_value()&&(it->second.goal_verified||language_.count(id))))return false;
        auto updated=it->second;auto schemas=schema_;updated.before=sorted(before);updated.observed.clear();
        Scene prior=updated.before;std::vector<TransitionReceipt> changes;
        for(size_t i=0;i<observed.size();++i) {
            const auto tid=id+"_step"+std::to_string(i);auto old=transitions_.find(tid);
            if(old==transitions_.end()||old->second.version==~0u||transition_owner_.at(tid)!=id)return false;
            auto r=old->second;r.before=prior;r.after=sorted(observed[i]);r.executed=executed[i];++r.version;
            if(!transition_ok(r,schemas))return false;
            prior=r.after;updated.observed.push_back(prior);changes.push_back(std::move(r));
        }
        std::optional<LanguageReceipt> language;
        auto old_language=language_.find(id);
        if(old_language!=language_.end()) {
            if(old_language->second.version==~0u)return false;
            auto r=old_language->second;auto binding=r.computational?abstract_computation(r.text,r.names,r.relations,r.quantities):abstract(r.text,r.names);auto mentions=binding.args;
            r.names={mentions.begin(),mentions.end()};r.scene=updated.observed.back();r.answer=*success;++r.version;
            for(const auto& a:r.scene.facts)r.names.insert(a.args.begin(),a.args.end());
            type_observed_language(r,binding);
            if(!validate(r.scene,{r.names.begin(),r.names.end()},schemas)||!computation_ok(r,binding,schemas))return false;
            language=std::move(r);
        }
        ++updated.version;updated.success=success.value_or(false);updated.goal_verified=success.has_value();
        it->second=std::move(updated);
        for(auto& r:changes)transitions_[r.id]=std::move(r);
        if(language)language_[id]=std::move(*language);
        changed(std::move(schemas));return true;
    }
};
} // namespace laminar::relational
