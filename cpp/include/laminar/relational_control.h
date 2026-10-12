#pragma once
// Learned operation selection over the changing relational session. Only public
// observations, cached computations and prices enter the neural policy. The
// evaluator supplies terminal outcomes, never the correct next operation.
#include "relational_session.h"
#include "backprop_choice.h"
#include "preference_tree.h"
#include <cmath>

namespace laminar::relational {
enum class ControlOp { retrieve, reason, ask, act, finish, next_goal, activate_goal };
struct ControlPrices {
    std::array<float,5> operation{0.01f,0.005f,0.06f,0.01f,0};
    float prediction=0.00001f,motor=0.02f,rebuild=0.005f;
    void validate()const {
        for(float value:operation)if(!std::isfinite(value)||value<0||value>1)throw std::invalid_argument("invalid operation price");
        for(float value:{prediction,motor,rebuild})if(!std::isfinite(value)||value<0||value>1)throw std::invalid_argument("invalid work price");
    }
};
struct StampedScene {Scene scene;uint64_t revision=0;};
struct ControlBody {
    bool reactive=false;
    size_t batch=1,motors=32;
    bool quantitative=false;
    bool preserve_search=false;
};
struct ControlStep {
    uint64_t token=0;
    ControlOp operation=ControlOp::finish;
    reasoning::Observation observation{};
    std::vector<int> available;
    std::vector<Action> actions;
    size_t predictions=0;
    uint64_t primitive_steps=0;
    float cost=0;
    Answer reply=Answer::Unknown;
    bool rebuilt=false,reused=false,rejected=false;
    std::string status;
    // External bootstrap interventions remain distinguishable from neural choices.
    std::string teacher;
    std::string goal; // execution ownership metadata, never a policy feature
    // Set when an external learner (the model) made this choice; the controller's
    // own policy was not evaluated for the step.
    std::string decider;
};
struct ControlHint {ControlOp operation;std::string source;};
struct ControlEvidenceLink {
    std::string id;
    unsigned version=1;
    bool execution=false;
    std::optional<bool> satisfaction;
    bool active=true;
};
struct ControlCredit {
    std::string id,evidence;
    std::vector<ControlStep> trace;
    float reward=0;
    bool satisfaction=false,execution=false;
    unsigned version=1,evidence_version=1;
    bool active=true;
    std::vector<ControlEvidenceLink> links;
    bool mission=false;
};

template<size_t ActionCount=5> class BasicControlledRelationalSession {
    static_assert(ActionCount==5||ActionCount==7,"unsupported control action count");
    struct Episode {
        std::string id,text;
        std::set<std::string> names;
        std::vector<Action> available;
        StampedScene known;
        ControlPrices prices;
        uint64_t model_version=0,step_token=0,act_token=0;
        size_t next_budget=4;
        size_t earlier_motors=0;
        bool effect=false,active=false,waiting=false,done=false,timed_out=false;
        bool cached=false,read_failed=false,ask_failed=false,last_rejected=false;
        Answer answer=Answer::Unknown;
        Plan plan;
        std::vector<Scene> observed;
        std::vector<bool> executed;
        std::vector<ControlStep> trace;
    } episode_;
    std::map<std::string,Episode> suspended_;
    std::mt19937 rng_;
    std::vector<float> initial_weights_;
    reasoning::BackpropChoice target_policy_;
    size_t applied_updates_=0;
    std::map<reasoning::Observation,std::array<float,ActionCount>> empirical_values_;
    std::vector<ControlCredit> credits_;
    std::map<std::string,size_t> credit_index_;
    std::map<std::string,ControlEvidenceLink> uncredited_evidence_;
    std::set<uint64_t> mission_tokens_;
    bool has_mission_receipts_=false;
    struct Update {size_t receipt,step;bool fit_only=false,learn_model=false,demonstration=false;};
    std::vector<Update> updates_;
    uint64_t next_step_token_=1;
    std::optional<ControlStep> selected_control_;
    uint64_t selected_model_version_=0;
    std::string selected_episode_id_;
    static constexpr size_t trace_budget_=256,credit_budget_=8192,update_budget_=2097152;
    bool learning_=true,terminal_only_=false,empirical_=false,model_based_=false;
    bool episodic_return_=false,greedy_trace_credit_=false;
    bool preserve_observation_records_=false,observation_semantics_locked_=false;
    Scene observation_scene(Scene scene)const {
        if(!preserve_observation_records_)return sorted(std::move(scene));
        std::sort(scene.facts.begin(),scene.facts.end());return scene;
    }
    ControlBody body_;
    struct Consequences {
        size_t count=0;
        double immediate=0;
        std::map<size_t,size_t> next;
    };
    struct ControlState {reasoning::Observation observation;std::array<Consequences,ActionCount> consequences;};
    std::map<reasoning::Observation,size_t> control_states_;
    std::vector<ControlState> control_model_;
    size_t control_state(const reasoning::Observation& obs) {
        auto found=control_states_.find(obs);if(found!=control_states_.end())return found->second;
        if(control_model_.size()>=32768)throw std::runtime_error("control-state capacity exhausted");
        size_t id=control_model_.size();control_states_[obs]=id;control_model_.push_back({obs,{}});return id;
    }
    void learn_control_model(const ControlCredit& receipt) {
        // Estimate operation consequences from the observed trace. No semantic
        // action names, intended route or oracle next-operation label is used.
        for(size_t i=0;i<receipt.trace.size();++i) {
            if(terminal_only_&&i+1!=receipt.trace.size())continue;
            const auto& step=receipt.trace[i];size_t state=control_state(step.observation);
            bool terminal=i+1==receipt.trace.size();size_t next=terminal?0:control_state(receipt.trace[i+1].observation);
            auto& row=control_model_[state].consequences[int(step.operation)];++row.count;
            row.immediate+=double(terminal?receipt.reward:0)-step.cost;
            if(!terminal)++row.next[next];
        }
        std::vector<std::array<float,ActionCount>> values(control_model_.size());
        for(size_t i=0;i<values.size();++i)values[i]=empirical_values_[control_model_[i].observation];
        for(size_t sweep=0;sweep<256;++sweep) {
            std::vector<float> best(values.size());for(size_t i=0;i<values.size();++i)best[i]=*std::max_element(values[i].begin(),values[i].end());
            float delta=0;
            for(size_t i=0;i<values.size();++i)for(size_t a=0;a<ActionCount;++a) {
                const auto& row=control_model_[i].consequences[a];if(!row.count)continue;
                double total=row.immediate;for(const auto& [next,count]:row.next)total+=.99*double(count)*best[next];
                float q=float(total/double(row.count));delta=std::max(delta,std::abs(values[i][a]-q));values[i][a]=q;
            }
            if(delta<1e-6f)break;
        }
        for(size_t i=0;i<values.size();++i)empirical_values_[control_model_[i].observation]=values[i];
    }
    bool valid_scene(const StampedScene& world)const {
        auto schemas=knowledge.schema_;
        return knowledge.validate(world.scene,{episode_.names.begin(),episode_.names.end()},schemas);
    }
    bool current()const{return episode_.active&&episode_.model_version==knowledge.version();}
    size_t motors()const{return episode_.earlier_motors+episode_.observed.size();}
    void invalidate_cache() {
        episode_.cached=false;episode_.answer=Answer::Unknown;episode_.plan={};episode_.next_budget=4;
    }
    void accept_world(const StampedScene& incoming) {
        auto next=incoming;next.scene=observation_scene(next.scene);
        if(next.revision!=episode_.known.revision||next.scene!=episode_.known.scene) {
            auto allowance=episode_.next_budget;
            bool same=next.scene==episode_.known.scene;
            invalidate_cache();
            if(body_.preserve_search&&same)episode_.next_budget=allowance;
        }
        episode_.known=std::move(next);
    }
    void apply_update(const Update& update) {
        const auto& r=credits_.at(update.receipt);const auto& step=r.trace.at(update.step);
        if(!r.active)return;
        if(update.learn_model){learn_control_model(r);return;}
        if constexpr(ActionCount==7)if(update.demonstration) {
            // Explicitly requested bootstrap supervision. The source-tagged
            // demonstrated action is distinct from reward-derived preferences.
            // Recorded update order makes withdrawal/restoration replay exact.
            if(step.teacher.empty())throw std::logic_error("unattributed demonstration update");
            std::vector<float> gradient(policy.weights.size());
            for(int action:step.available) {
                auto g=policy.gradient(step.observation,action,action==int(step.operation)?1.f:0.f);
                for(size_t j=0;j<g.size();++j)gradient[j]+=g[j];
            }
            const float rate=.002f*(32.f/float(policy.hidden));
            for(size_t j=0;j<gradient.size();++j)policy.weights[j]-=rate*gradient[j];
            return;
        }
        float target=r.reward;
        if(empirical_) {
            // An episodic value table avoids bootstrapping through a changing
            // neural approximation. It sees only the same public observations
            // and experienced terminal outcomes. A declared memory-policy arm
            // may also use these values directly for operation selection.
            if(episodic_return_) {
                // A completed goal supplies one observed return for each of
                // its own decisions, even if unrelated goals intervened.
                // No child reward or teacher action label enters this target.
                for(size_t i=r.trace.size();i-->update.step+1;)target=.99f*(target-r.trace[i].cost);
            } else if(greedy_trace_credit_) {
                // Watkins-style goal-scoped credit: continue an observed
                // return while the next recorded action is greedy under the
                // current empirical values. At a nongreedy action, bootstrap
                // from that public state's best available value instead of
                // charging earlier choices for a later exploratory detour.
                // Tied actions all qualify; no teacher label, child reward or
                // hidden task state selects the trace or the cutoff.
                for(size_t i=r.trace.size();i-->update.step+1;) {
                    const auto& next=r.trace[i];const auto& values=empirical_values_[next.observation];
                    float best=-1e9f;for(int a:next.available)best=std::max(best,values[a]);
                    if(values[int(next.operation)]<best)target=.99f*best;
                    else target=.99f*(target-next.cost);
                }
            } else if(update.step+1<r.trace.size()) {
                const auto& next=r.trace[update.step+1];const auto& values=empirical_values_[next.observation];
                float best=-1e9f;for(int a:next.available)best=std::max(best,values[a]);target=.99f*best;
            }
            auto& values=empirical_values_[step.observation];int chosen=int(step.operation);
            if(!update.fit_only)values[chosen]+=.2f*(target-step.cost-values[chosen]);
            int best=step.available.front();for(int a:step.available)if(values[a]>values[best])best=a;
            // Distill the operation preference, rather than tiny differences
            // between almost equal Q values. These labels are self-generated.
            std::vector<float> gradient(policy.weights.size());
            for(int a:step.available) {
                auto g=policy.gradient(step.observation,a,a==best?1.f:0.f);
                for(size_t j=0;j<g.size();++j)gradient[j]+=g[j];
            }
            // Preserve the baseline function-space step when widening the
            // hidden layer; otherwise summed output gradients grow with width.
            float rate=.002f*(32.f/float(policy.hidden));
            for(size_t j=0;j<gradient.size();++j)policy.weights[j]-=rate*gradient[j];
            return;
        }
        if(update.step+1<r.trace.size()) {
            const auto& next=r.trace[update.step+1];
            // Double Q: select with the live network and evaluate with a delayed
            // copy. A repeated read must not bootstrap its own changing estimate.
            int selected=policy.choose(next.observation,next.available);
            target=0.99f*target_policy_.action_value(next.observation,selected);
        }
        auto gradient=policy.gradient(step.observation,int(step.operation),target-step.cost);
        for(size_t i=0;i<gradient.size();++i)policy.weights[i]-=0.001f*gradient[i];
        if(++applied_updates_%256==0)target_policy_.weights=policy.weights;
    }
    bool record_answer(const std::string& id,bool satisfaction) {
        if(!episode_.known.scene.complete)return true; // no invented complete teaching world
        LanguageReceipt r;r.id=id;r.text=episode_.text;r.source="verified_environment";
        r.names=episode_.names;r.scene=episode_.known.scene;r.answer=satisfaction;
        for(const auto& a:r.scene.facts)r.names.insert(a.args.begin(),a.args.end());
        if(auto parsed=knowledge.parse_evidence(r.text))knowledge.type_observed_language(r,*parsed);
        auto schemas=knowledge.schema_;auto binding=r.computational?abstract_computation(r.text,r.names,r.relations,r.quantities):abstract(r.text,r.names);
        if(!concepts::symbol(id)||binding.args.empty()||binding.args.size()>8||binding.pattern.size()>64||r.text.size()>2048
           ||knowledge.language_.count(id)||knowledge.language_.size()>=knowledge.evidence_budget
           ||!knowledge.validate(r.scene,{r.names.begin(),r.names.end()},schemas)||!knowledge.computation_ok(r,binding,schemas))return false;
        knowledge.language_[id]=std::move(r);knowledge.changed(std::move(schemas));return true;
    }
public:
    RelationalSession knowledge;
    reasoning::BackpropChoice policy;
    // Complete learning state (laminar/state_io.h). Configuration limits that are compile-time constants are not
    // state; everything a mission, credit or update can change is.
    template<class Ar> void state_io(Ar& ar) {
        ar.section("control");
        ar(episode_);ar(suspended_);ar(rng_);ar(initial_weights_);ar(target_policy_);ar(applied_updates_);
        ar(empirical_values_);ar(credits_);ar(credit_index_);ar(uncredited_evidence_);ar(mission_tokens_);ar(has_mission_receipts_);
        ar(updates_);ar(next_step_token_);ar(selected_control_);ar(selected_model_version_);ar(selected_episode_id_);
        ar(learning_);ar(terminal_only_);ar(empirical_);ar(model_based_);ar(episodic_return_);ar(greedy_trace_credit_);
        ar(preserve_observation_records_);ar(observation_semantics_locked_);ar(body_);ar(control_states_);ar(control_model_);
        ar.section("knowledge");ar(knowledge);
        ar.section("policy");ar(policy);
    }
    explicit BasicControlledRelationalSession(unsigned seed=0,bool learning=true,bool terminal_only=false,bool empirical=false,ControlBody body={},bool model_based=false,int controller_hidden=32,reasoning::SkillShape shape={},bool episodic_return=false,bool greedy_trace_credit=false)
        :rng_(seed+17011),target_policy_(ActionCount,seed,controller_hidden),learning_(learning),terminal_only_(terminal_only),empirical_(empirical||model_based),model_based_(model_based),episodic_return_(episodic_return),greedy_trace_credit_(greedy_trace_credit),body_(body),knowledge(seed,body.reactive?32768:8192,1024,shape),policy(ActionCount,seed,controller_hidden) {
        if(!body.batch||!body.motors||body.batch>body.motors||body.motors>256)throw std::invalid_argument("invalid control body limits");
        if(episodic_return_&&(!empirical_||model_based_))throw std::invalid_argument("episode return requires empirical non-model credit");
        if(greedy_trace_credit_&&(!empirical_||model_based_||episodic_return_))throw std::invalid_argument("greedy trace requires empirical non-model TD credit");
        if constexpr(ActionCount!=5)if(model_based_)throw std::invalid_argument("extended controller does not support model-based credit");
        initial_weights_=policy.weights;
    }
    const auto& credits()const{return credits_;}
    void preserve_observation_records() {
        if(observation_semantics_locked_)throw std::logic_error("observation semantics must be configured before episodes");
        preserve_observation_records_=true;
    }
    const auto& initial_weights()const{return initial_weights_;}
    const auto& empirical_values()const{return empirical_values_;}
    size_t updates()const{return updates_.size();}
    size_t consolidate(size_t rounds) {
        if(episode_.active||!learning_||!empirical_)throw std::logic_error("invalid policy consolidation");
        if(!mission_evidence_current())throw std::logic_error("mission evidence changed outside coupled revision");
        std::map<reasoning::Observation,Update> representatives;
        for(size_t r=0;r<credits_.size();++r)for(size_t s=0;s<credits_[r].trace.size();++s) {
            if(!credits_[r].active)continue;
            if(terminal_only_&&s+1!=credits_[r].trace.size())continue;
            representatives.try_emplace(credits_[r].trace[s].observation,Update{r,s,true});
        }
        if(rounds&&representatives.size()>(update_budget_-updates_.size())/rounds)throw std::runtime_error("consolidation update budget exhausted");
        std::vector<Update> order;for(const auto& [obs,update]:representatives)order.push_back(update);
        for(size_t round=0;round<rounds;++round) {
            std::shuffle(order.begin(),order.end(),rng_);
            for(const auto& update:order){updates_.push_back(update);apply_update(update);}
        }
        return order.size()*rounds;
    }
    bool learning()const{return learning_;}
    bool reactive()const{return body_.reactive;}
    void preserve_search(bool enabled){if(episode_.active)throw std::logic_error("cannot change search retention during an episode");body_.preserve_search=enabled;}
    void set_learning(bool enabled){if(episode_.active)throw std::logic_error("cannot change learning during an episode");learning_=enabled;}
    bool done()const{return episode_.done;}
    bool timed_out()const{return episode_.timed_out;}
    bool active()const{return episode_.active;}
    bool waiting()const{return episode_.waiting;}
    Answer reply()const{return episode_.done&&!episode_.timed_out?episode_.answer:Answer::Unknown;}
    const auto& trace()const{return episode_.trace;}
    std::vector<std::string> suspended_goals()const {
        std::vector<std::string> ids;for(const auto& [id,e]:suspended_)ids.push_back(id);return ids;
    }
    const auto& visible()const{return episode_.known;}
    const auto& plan()const{return episode_.plan;}
    uint64_t expected_version()const{return episode_.model_version;}
    bool begin(const std::string& id,const std::string& text,const std::set<std::string>& names,
               const StampedScene& visible,const std::vector<Action>& actions,bool effect,ControlPrices prices={}) {
        prices.validate();
        if(episode_.active||(knowledge.pending_&&knowledge.pending_->plan.version==knowledge.version())||!concepts::symbol(id)||credit_index_.count(id)||suspended_.count(id)||text.empty()||text.size()>2048||names.size()>32||actions.size()>128)return false;
        auto schemas=knowledge.schema_;
        if(!knowledge.validate(visible.scene,{names.begin(),names.end()},schemas)||!knowledge.affordances_ok(visible.scene,actions))return false;
        observation_semantics_locked_=true;
        episode_={};episode_.id=id;episode_.text=text;episode_.names=names;episode_.available=actions;
        episode_.known={observation_scene(visible.scene),visible.revision};episode_.effect=effect;episode_.prices=prices;
        episode_.active=true;episode_.model_version=knowledge.version();return true;
    }
    reasoning::Observation observe()const {
        const auto& e=episode_;reasoning::Observation o{};
        o[0]=e.known.scene.complete;o[1]=!e.known.scene.facts.empty();o[2]=knowledge.dirty();o[3]=knowledge.ready();o[4]=e.effect;
        o[5]=e.cached;o[6]=e.cached&&e.answer==Answer::True;o[7]=e.cached&&e.answer==Answer::False;o[8]=e.cached&&e.answer==Answer::Unknown;
        o[9]=!e.plan.actions.empty();o[10]=e.plan.status=="search_budget";o[11]=e.plan.status=="state_budget"||e.plan.status=="no_plan"||e.plan.status=="unavailable_meaning_or_world";
        o[12]=e.read_failed;o[13]=e.ask_failed;o[14]=e.act_token!=0;o[15]=!e.available.empty();
        o[16]=e.last_rejected;o[17]=e.plan.reused;o[18]=std::min(4.f,float(e.plan.actions.size()))/4.f;
        o[19]=float(std::log2(double(e.next_budget)))/15.f;
        if(body_.reactive)o[20]=motors()>=body_.motors;
        for(int a=0;a<5;++a)o[21+a]=20*e.prices.operation[a];
        o[26]=20*e.prices.prediction;o[27]=20*e.prices.motor;o[28]=20*e.prices.rebuild;
        if(body_.quantitative) {
            o[29]=std::log2(1.f+float(e.available.size()))/std::log2(129.f);
            o[30]=std::log2(1.f+float(e.known.scene.facts.size()))/std::log2(257.f);
        }
        o[31]=1;return o;
    }
    ControlStep choose(float epsilon=0,int disabled=-1,bool memory_policy=false,const reasoning::PreferenceTree* readout=nullptr,
                       std::optional<ControlHint> hint=std::nullopt) {
        if(!current())throw std::runtime_error("stale controlled episode");
        if(episode_.waiting||episode_.done||episode_.trace.size()>=trace_budget_)throw std::runtime_error("no available decision");
        std::vector<int> available;for(int a=0;a<5;++a)if(a!=disabled)available.push_back(a);
        return execute_physical(select_control(observe(),available,epsilon,memory_policy,readout,hint));
    }
protected:
    bool valid_control_request(const std::string& text,const std::set<std::string>& names,
                               const StampedScene& visible,const std::vector<Action>& actions)const {
        if(text.empty()||text.size()>2048||names.size()>32||actions.size()>128)return false;
        auto schemas=knowledge.schema_;
        return knowledge.validate(visible.scene,{names.begin(),names.end()},schemas)&&knowledge.affordances_ok(visible.scene,actions);
    }
    const std::string& current_episode_id()const{return episode_.id;}
    size_t current_motor_count()const{return motors();}
    bool import_five_action_policy(const reasoning::BackpropChoice& source) {
        if(source.actions!=5||source.hidden!=policy.hidden||next_step_token_!=1||!credits_.empty()||!updates_.empty()
           ||episode_.active||!suspended_.empty())return false;
        for(float value:source.weights)if(!std::isfinite(value))return false;
        if(source.weights.size()!=size_t(source.hidden*source.input+source.hidden+5*source.hidden+5))return false;
        std::copy_n(source.weights.begin(),source.output_weights(),policy.weights.begin());
        for(int action=0;action<5;++action) {
            std::copy_n(source.weights.begin()+source.output_weights()+action*source.hidden,source.hidden,
                        policy.weights.begin()+policy.output_weights()+action*policy.hidden);
            policy.weights[policy.output_bias()+action]=source.weights[source.output_bias()+action];
        }
        initial_weights_=policy.weights;target_policy_.weights=policy.weights;return true;
    }
    ControlStep select_control(const reasoning::Observation& observation,const std::vector<int>& available,
                               float epsilon=0,bool memory_policy=false,const reasoning::PreferenceTree* readout=nullptr,
                               std::optional<ControlHint> hint=std::nullopt,std::optional<ControlHint> decided=std::nullopt) {
        if(!mission_evidence_current())throw std::logic_error("mission evidence changed outside coupled revision");
        if(available.empty())throw std::invalid_argument("empty control menu");
        std::set<int> unique;
        for(int action:available)if(action<0||size_t(action)>=ActionCount||!unique.insert(action).second)throw std::invalid_argument("invalid control menu");
        for(float value:observation)if(!std::isfinite(value))throw std::invalid_argument("invalid control observation");
        if constexpr(ActionCount!=5)if(readout)throw std::invalid_argument("preference tree requires five actions");
        if(hint&&(!learning_||!concepts::symbol(hint->source)||!unique.count(int(hint->operation))))
            throw std::invalid_argument("invalid teacher intervention or guidance during frozen evaluation");
        if(decided&&(hint||readout||memory_policy||epsilon>0||!concepts::symbol(decided->source)||!unique.count(int(decided->operation))))
            throw std::invalid_argument("external decision must be the only selection source");
        ControlStep step;step.token=next_step_token_++;step.observation=observation;step.available=available;
        if(decided) {
            // The external learner owns this choice: the controller's policy is not consulted.
            step.operation=decided->operation;step.decider=decided->source;
            selected_control_=step;selected_model_version_=knowledge.version();selected_episode_id_=episode_.active?episode_.id:std::string{};
            return step;
        }
        int selected=policy.choose(step.observation,step.available);
        if(readout)selected=readout->choose(step.observation,step.available);
        if(memory_policy) {
            auto found=empirical_values_.find(step.observation);
            if(found!=empirical_values_.end()) {
                selected=step.available.front();
                for(int a:step.available)if(found->second[a]>found->second[selected])selected=a;
            }
        }
        if(std::generate_canonical<float,24>(rng_)<epsilon)selected=step.available[rng_()%step.available.size()];
        if(hint){selected=int(hint->operation);step.teacher=hint->source;}
        step.operation=ControlOp(selected);
        selected_control_=step;selected_model_version_=knowledge.version();selected_episode_id_=episode_.active?episode_.id:std::string{};
        return step;
    }
    ControlStep execute_physical(ControlStep step) {
        if(!current())throw std::runtime_error("stale controlled episode");
        if(episode_.waiting||episode_.done||episode_.trace.size()>=trace_budget_)throw std::runtime_error("no available decision");
        if(!selected_control_||step.token!=selected_control_->token||step.operation!=selected_control_->operation
           ||step.observation!=selected_control_->observation||step.available!=selected_control_->available
           ||step.teacher!=selected_control_->teacher||step.decider!=selected_control_->decider||selected_model_version_!=knowledge.version()||selected_episode_id_!=episode_.id
           ||int(step.operation)<0||int(step.operation)>=5)throw std::invalid_argument("invalid or stale physical selection");
        // Only selection fields are accepted from the caller; work, effects,
        // reply and cost are computed by the same physical protocol as before.
        step=*selected_control_;selected_control_.reset();step.cost=episode_.prices.operation[int(step.operation)];
        episode_.step_token=step.token;episode_.last_rejected=false;
        if(step.operation==ControlOp::reason) {
            if(knowledge.dirty()){knowledge.rebuild();step.rebuilt=true;step.cost+=episode_.prices.rebuild;}
            // Acquisition/rehearsal belongs to rebuild; count actual neural VM
            // inference separately and charge the public computation work rate.
            auto primitive_before=knowledge.primitive_steps();
            episode_.answer=knowledge.answer(episode_.text,episode_.known.scene);episode_.cached=true;
            if(episode_.effect&&episode_.answer==Answer::False&&episode_.known.scene.complete&&(!episode_.act_token||body_.reactive)) {
                episode_.plan=knowledge.plan(episode_.text,episode_.known.scene,episode_.available,episode_.next_budget,8,true);
                step.predictions=episode_.plan.predictions;step.reused=episode_.plan.reused;
                step.cost+=episode_.prices.prediction*float(step.predictions);
                if(episode_.plan.status=="search_budget")episode_.next_budget=std::min<size_t>(32768,episode_.next_budget*2);
            }
            step.primitive_steps=knowledge.primitive_steps()-primitive_before;
            step.cost+=episode_.prices.prediction*float(step.primitive_steps);
            step.status=episode_.plan.status.empty()?"interpreted":episode_.plan.status;
        } else if(step.operation==ControlOp::retrieve||step.operation==ControlOp::ask) {
            episode_.waiting=true;step.status="awaiting_read";
        } else if(step.operation==ControlOp::act) {
            if(((episode_.act_token||episode_.earlier_motors)&&!body_.reactive)||motors()>=body_.motors||knowledge.dirty()||!episode_.known.scene.complete||episode_.available.empty()) {
                step.rejected=true;episode_.last_rejected=true;step.status="unavailable_action";
            } else {
                Plan issued;
                if(!episode_.plan.actions.empty()&&episode_.plan.version==knowledge.version()) {
                    // This private ticket was computed by reason for this exact
                    // model/world. World changes invalidate it before reaching here.
                    issued=episode_.plan;
                } else issued=knowledge.exploration(episode_.known.scene,episode_.available);
                if(!issued.actions.empty()) {
                    if(body_.reactive)issued.actions.resize(std::min({issued.actions.size(),body_.batch,body_.motors-motors()}));
                    if(episode_.act_token) {
                        // This episode owns the pending execution. Observed motor
                        // steps are appended, with one terminal label for the whole
                        // attempt; there is no intermediate success supervision.
                        if(!knowledge.pending_||knowledge.pending_->plan.token!=episode_.act_token||episode_.observed.empty()
                           ||sorted(episode_.known.scene)!=sorted(episode_.observed.back()))throw std::runtime_error("noncontiguous controlled execution");
                        issued.token=episode_.act_token;
                        auto& all=knowledge.pending_->plan.actions;all.insert(all.end(),issued.actions.begin(),issued.actions.end());
                    } else {
                        issued.token=knowledge.next_token_++;
                        knowledge.pending_=RelationalSession::Pending{issued,episode_.text,episode_.known.scene};
                    }
                    episode_.act_token=issued.token;step.actions=issued.actions;step.reused=issued.reused;
                    episode_.waiting=true;step.status="awaiting_actions";
                    step.cost+=episode_.prices.motor*float(step.actions.size());
                } else {step.rejected=true;episode_.last_rejected=true;step.status=issued.status;}
            }
        } else {
            episode_.done=true;step.reply=episode_.cached?episode_.answer:Answer::Unknown;
            episode_.answer=step.reply;step.status="finished";
        }
        if constexpr(ActionCount!=5)step.goal=episode_.id;
        episode_.trace.push_back(step);return step;
    }
    std::optional<ControlEvidenceLink> evidence_link(const std::string& id,bool execution)const {
        if(execution) {
            auto found=knowledge.executions_.find(id);if(found==knowledge.executions_.end())return {};
            const auto& r=found->second;
            return ControlEvidenceLink{id,r.version,true,r.goal_verified?std::optional<bool>(r.success):std::nullopt,r.active};
        }
        auto found=knowledge.language_.find(id);
        if(found==knowledge.language_.end()||knowledge.executions_.count(id)||found->second.source!="verified_environment")return {};
        const auto& r=found->second;return ControlEvidenceLink{id,r.version,false,r.answer,r.active};
    }
    std::vector<ControlEvidenceLink> pending_mission_evidence()const {
        std::vector<ControlEvidenceLink> result;for(const auto& [id,link]:uncredited_evidence_)result.push_back(link);return result;
    }
    bool mission_evidence_current()const {
        // Ordinary five-action episodes never create mission receipts. Keep
        // their per-decision path independent of the accumulated credit count.
        if(!has_mission_receipts_)return true;
        for(const auto& credit:credits_)if(credit.mission)for(const auto& link:credit.links) {
            auto actual=evidence_link(link.id,link.execution);
            if(!actual||actual->version!=link.version||actual->active!=link.active||actual->satisfaction!=link.satisfaction)return false;
        }
        return true;
    }
    std::optional<ControlEvidenceLink> close_episode_evidence(bool satisfaction,
            const std::string& source="verified_environment",bool generated=false) {
        if(!current()||!episode_.done||episode_.waiting||episode_.trace.empty()||source!="verified_environment"||generated)return {};
        ControlEvidenceLink link;
        bool execution=episode_.act_token!=0;
        if(learning_) {
            if(execution) {
                link.id="execution"+std::to_string(episode_.act_token);
                if(!knowledge.feedback(episode_.act_token,episode_.observed,episode_.executed,satisfaction))return {};
            } else {
                link.id="controlled_"+episode_.id;
                if(!record_answer(link.id,satisfaction))return {};
                if(!episode_.known.scene.complete)link.id.clear();
            }
            if(!link.id.empty()) {
                link=*evidence_link(link.id,execution);uncredited_evidence_[link.id]=link;
            }
        } else if(execution&&knowledge.pending_&&knowledge.pending_->plan.token==episode_.act_token)knowledge.pending_.reset();
        episode_.active=false;selected_control_.reset();return link;
    }
    bool submit_mission_credit(const std::string& id,const std::vector<ControlStep>& trace,
            const std::vector<ControlEvidenceLink>& links,bool correct,size_t replay_updates=128,bool random_credit=false,
            const std::string& source="verified_environment",bool generated=false) {
        if(episode_.active||!suspended_.empty()||!concepts::symbol(id)||credit_index_.count(id)||trace.empty()
           ||trace.size()>trace_budget_||source!="verified_environment"||generated||model_based_)return false;
        if(!mission_evidence_current())return false;
        if(learning_&&(credits_.size()>=credit_budget_||updates_.size()>update_budget_
           ||replay_updates>update_budget_-updates_.size()))return false;
        uint64_t previous=0;
        for(const auto& step:trace) {
            if(step.token<=previous||step.token>=next_step_token_||mission_tokens_.count(step.token)||!std::isfinite(step.cost)||step.cost<0
               ||int(step.operation)<0||size_t(step.operation)>=ActionCount||step.available.empty())return false;
            previous=step.token;std::set<int> available;
            for(int action:step.available)if(action<0||size_t(action)>=ActionCount||!available.insert(action).second)return false;
            if(!available.count(int(step.operation)))return false;
            for(float value:step.observation)if(!std::isfinite(value))return false;
        }
        std::set<std::string> linked;
        if(links.size()!=uncredited_evidence_.size())return false;
        for(const auto& link:links) {
            if(link.id.empty()||!uncredited_evidence_.count(link.id)||!linked.insert(link.id).second)return false;
            auto actual=evidence_link(link.id,link.execution);
            if(!actual||actual->version!=link.version||actual->satisfaction!=link.satisfaction||actual->active!=link.active||!link.active)return false;
            for(const auto& old:credits_) {
                if(old.evidence==link.id)return false;
                for(const auto& owned:old.links)if(owned.id==link.id)return false;
            }
        }
        for(const auto& [pending,link]:uncredited_evidence_)if(!linked.count(pending))return false;
        if(!learning_)return links.empty();
        float reward=correct?1.f:-1.f;if(random_credit)reward=rng_()%2?1.f:-1.f;
        ControlCredit credit;credit.id=id;credit.trace=trace;credit.reward=reward;credit.satisfaction=correct;credit.links=links;credit.mission=true;
        credit_index_[id]=credits_.size();credits_.push_back(std::move(credit));has_mission_receipts_=true;uncredited_evidence_.clear();
        for(const auto& step:trace)mission_tokens_.insert(step.token);
        for(size_t i=0;i<replay_updates;++i) {
            size_t receipt=i<trace.size()?credits_.size()-1:rng_()%credits_.size();const auto& replay=credits_[receipt].trace;
            size_t decision=terminal_only_?replay.size()-1:(i<trace.size()?replay.size()-1-i:rng_()%replay.size());
            Update update{receipt,decision,false,false};updates_.push_back(update);apply_update(update);
        }
        return true;
    }
    bool revise_mission_credit(const std::string& id,unsigned expected,bool correct,
            const std::vector<ControlEvidenceLink>& replacements,bool active=true,
            const std::string& source="verified_environment") {
        auto found=credit_index_.find(id);
        if(episode_.active||!suspended_.empty()||!uncredited_evidence_.empty()||found==credit_index_.end()
           ||expected==~0u||source!="verified_environment")return false;
        const auto& before=credits_[found->second];
        if(before.version!=expected||!before.mission||before.links.size()!=replacements.size())return false;
        auto candidate=*this;auto& revised=candidate.credits_[found->second];
        for(size_t i=0;i<replacements.size();++i) {
            const auto& update=replacements[i];const auto& old=before.links[i];
            if(update.id!=old.id||update.version!=old.version||update.execution!=old.execution||update.active!=active)return false;
            bool okay=update.execution?candidate.knowledge.revise_execution(update.id,update.version,update.satisfaction,active,source)
                :update.satisfaction&&candidate.knowledge.revise_language(update.id,update.version,*update.satisfaction,active,source);
            if(!okay)return false;
            revised.links[i]=update;++revised.links[i].version;
        }
        ++revised.version;revised.reward=correct?1.f:-1.f;revised.satisfaction=correct;revised.active=active;
        if(!candidate.mission_evidence_current())return false;
        candidate.policy.weights=candidate.initial_weights_;candidate.target_policy_.weights=candidate.initial_weights_;
        candidate.applied_updates_=0;candidate.empirical_values_.clear();candidate.control_model_.clear();candidate.control_states_.clear();
        for(const auto& update:candidate.updates_)candidate.apply_update(update);
        *this=std::move(candidate);return true;
    }
    size_t fit_demonstration_rows(size_t rounds,const std::string& source) {
        if constexpr(ActionCount!=7)throw std::logic_error("demonstration fitting requires the experimental controller");
        if(episode_.active||!suspended_.empty()||!uncredited_evidence_.empty()||!learning_||!concepts::symbol(source))
            throw std::logic_error("invalid demonstration fitting boundary or source");
        if(!mission_evidence_current())throw std::logic_error("mission evidence changed outside coupled revision");
        using Key=std::pair<reasoning::Observation,std::vector<int>>;
        std::map<Key,Update> representatives;
        for(size_t receipt=0;receipt<credits_.size();++receipt)if(credits_[receipt].active)
            for(size_t step=0;step<credits_[receipt].trace.size();++step) {
                const auto& row=credits_[receipt].trace[step];if(row.teacher!=source)continue;
                if(int(row.operation)<0||size_t(row.operation)>=ActionCount||
                   std::find(row.available.begin(),row.available.end(),int(row.operation))==row.available.end())
                    throw std::invalid_argument("demonstration label is outside its recorded menu");
                auto [found,inserted]=representatives.try_emplace(Key{row.observation,row.available},Update{receipt,step,false,false,true});
                if(!inserted&&credits_[found->second.receipt].trace[found->second.step].operation!=row.operation)
                    throw std::invalid_argument("conflicting demonstration labels for the same public observation/menu");
            }
        if(updates_.size()>update_budget_||rounds&&representatives.size()>(update_budget_-updates_.size())/rounds)
            throw std::runtime_error("demonstration update budget exhausted");
        if(representatives.empty()||!rounds)return 0;
        std::vector<Update> order;for(const auto& [key,update]:representatives)order.push_back(update);
        for(size_t round=0;round<rounds;++round) {
            std::shuffle(order.begin(),order.end(),rng_);
            for(const auto& update:order){updates_.push_back(update);apply_update(update);}
        }
        return order.size()*rounds;
    }
    size_t fit_balanced_demonstration_rows(size_t steps,const std::string& source) {
        if constexpr(ActionCount!=7)throw std::logic_error("balanced demonstration fitting requires the experimental controller");
        if(episode_.active||!suspended_.empty()||!uncredited_evidence_.empty()||!learning_||!concepts::symbol(source))
            throw std::logic_error("invalid demonstration fitting boundary or source");
        if(!mission_evidence_current())throw std::logic_error("mission evidence changed outside coupled revision");
        using Key=std::pair<reasoning::Observation,std::vector<int>>;
        std::map<Key,Update> representatives;
        for(size_t receipt=0;receipt<credits_.size();++receipt)if(credits_[receipt].active)
            for(size_t step=0;step<credits_[receipt].trace.size();++step) {
                const auto& row=credits_[receipt].trace[step];if(row.teacher!=source)continue;
                if(int(row.operation)<0||size_t(row.operation)>=ActionCount||
                   std::find(row.available.begin(),row.available.end(),int(row.operation))==row.available.end())
                    throw std::invalid_argument("demonstration label is outside its recorded menu");
                auto [found,inserted]=representatives.try_emplace(Key{row.observation,row.available},Update{receipt,step,false,false,true});
                if(!inserted&&credits_[found->second.receipt].trace[found->second.step].operation!=row.operation)
                    throw std::invalid_argument("conflicting demonstration labels for the same public observation/menu");
            }
        if(representatives.empty()||!steps)return 0;
        if(updates_.size()>update_budget_||steps>update_budget_-updates_.size())
            throw std::runtime_error("demonstration update budget exhausted");
        // Balance observed (phase, demonstrated action), then unique states.
        // Arrival handling is public execution context, not a task identity.
        std::map<std::pair<int,int>,std::vector<Update>> grouped;
        for(const auto& [key,update]:representatives) {
            const auto& row=credits_[update.receipt].trace[update.step];
            const int phase=row.observation[29]?0:row.observation[30]?1:2;
            grouped[{phase,int(row.operation)}].push_back(update);
        }
        std::vector<std::vector<Update>> groups;for(auto& [key,rows]:grouped)groups.push_back(std::move(rows));
        std::vector<size_t> order;for(size_t i=0;i<groups.size();++i)order.push_back(i);
        for(size_t step=0;step<steps;++step) {
            if(step%order.size()==0)std::shuffle(order.begin(),order.end(),rng_);
            const auto& group=groups[order[step%order.size()]];
            const auto update=group[std::uniform_int_distribution<size_t>(0,group.size()-1)(rng_)];
            updates_.push_back(update);apply_update(update);
        }
        return steps;
    }
public:
    bool read(uint64_t token,const std::optional<StampedScene>& value,const std::string& source,bool generated=false) {
        if(!current()||!episode_.waiting||episode_.step_token!=token||generated||episode_.trace.empty())return false;
        auto operation=episode_.trace.back().operation;
        if((operation!=ControlOp::retrieve&&operation!=ControlOp::ask)||source!=(operation==ControlOp::retrieve?"trusted_memory":"verified_environment"))return false;
        if(value&&!valid_scene(*value))return false;
        bool accepted=value&&value->revision>=episode_.known.revision;
        // This receipt type records a contiguous own-action chain. An external
        // world change needs a new episode, not attribution to the last motor.
        if(accepted&&!episode_.observed.empty()) {
            const auto& last=episode_.observed.back();auto incoming=sorted(value->scene);
            if(incoming.complete&&incoming!=sorted(last))return false;
            for(const auto& atom:incoming.facts)if(std::find(last.facts.begin(),last.facts.end(),atom)==last.facts.end())return false;
        }
        if(accepted)accept_world(*value);
        if(operation==ControlOp::retrieve)episode_.read_failed=!accepted;else episode_.ask_failed=!accepted;
        episode_.waiting=false;return true;
    }
    bool actions(uint64_t token,const std::vector<Scene>& observed,const std::vector<bool>& executed,uint64_t revision,
                 const std::string& source="verified_environment",bool generated=false) {
        if(!current()||!episode_.waiting||episode_.step_token!=token||generated||source!="verified_environment"||episode_.trace.empty())return false;
        const auto& step=episode_.trace.back();
        if(step.operation!=ControlOp::act||step.actions.size()!=observed.size()||executed.size()!=observed.size()||observed.empty()||revision<episode_.known.revision)return false;
        auto schemas=knowledge.schema_;Scene before=episode_.known.scene;
        for(size_t i=0;i<observed.size();++i) {
            TransitionReceipt r;r.id="validation";r.action=step.actions[i];r.before=before;r.after=observed[i];r.executed=executed[i];
            if(!knowledge.transition_ok(r,schemas))return false;
            before=observed[i];
        }
        episode_.observed.insert(episode_.observed.end(),observed.begin(),observed.end());
        episode_.executed.insert(episode_.executed.end(),executed.begin(),executed.end());
        accept_world({observed.back(),revision});episode_.waiting=false;return true;
    }
    bool timeout() {
        if(!current()||episode_.waiting||episode_.done)return false;
        episode_.done=true;episode_.timed_out=true;episode_.answer=Answer::Unknown;return true;
    }
    // Persist a goal and its control trace while another request runs. Only
    // already acknowledged action transitions are committed here; no terminal
    // reward or goal-satisfaction label is invented at an interruption.
    bool suspend() {
        if(!current()||episode_.waiting||episode_.done||suspended_.size()>=64||suspended_.count(episode_.id))return false;
        if(episode_.act_token) {
            if(learning_) {
                if(!knowledge.feedback(episode_.act_token,episode_.observed,episode_.executed,std::nullopt))return false;
                if constexpr(ActionCount!=5) {
                    auto id="execution"+std::to_string(episode_.act_token);uncredited_evidence_[id]=*evidence_link(id,true);
                }
            } else if(knowledge.pending_&&knowledge.pending_->plan.token==episode_.act_token)knowledge.pending_.reset();
        }
        episode_.earlier_motors+=episode_.observed.size();episode_.observed.clear();episode_.executed.clear();episode_.act_token=0;
        invalidate_cache();episode_.active=false;episode_.model_version=knowledge.version();
        auto id=episode_.id;suspended_.emplace(std::move(id),std::move(episode_));episode_={};return true;
    }
    bool resume(const std::string& id,const StampedScene& visible,
                const std::string& source="verified_environment",bool generated=false) {
        auto found=suspended_.find(id);
        if(episode_.active||found==suspended_.end()||source!="verified_environment"||generated
           ||(knowledge.pending_&&knowledge.pending_->plan.version==knowledge.version()))return false;
        const auto& saved=found->second;auto schemas=knowledge.schema_;
        if(visible.revision<saved.known.revision||!knowledge.validate(visible.scene,{saved.names.begin(),saved.names.end()},schemas)
           ||!knowledge.affordances_ok(visible.scene,saved.available))return false;
        // Old plans and observation callbacks cannot survive an interruption.
        // The retained trace receives credit only when this goal finally ends.
        episode_=std::move(found->second);suspended_.erase(found);
        episode_.known={observation_scene(visible.scene),visible.revision};episode_.model_version=knowledge.version();
        episode_.active=true;episode_.step_token=0;episode_.read_failed=episode_.ask_failed=episode_.last_rejected=false;
        invalidate_cache();return true;
    }
    bool cancel_suspended(const std::string& id) {return suspended_.erase(id)!=0;}
    // Terminal correctness and satisfaction are distinct: the controller can
    // report an answer incorrectly even when the physical goal happens to hold.
    // Only satisfaction labels a world; only correctness trains operation choice.
    bool finish(bool correct,bool satisfaction,size_t replay_updates=128,bool random_credit=false,
                const std::string& source="verified_environment",bool generated=false) {
        if(!current()||!episode_.done||episode_.waiting||episode_.trace.empty()||source!="verified_environment"||generated)return false;
        if(learning_&&(credits_.size()>=credit_budget_||updates_.size()+replay_updates+size_t(model_based_)>update_budget_))return false;
        std::string evidence;bool execution=episode_.act_token!=0;
        if(learning_) {
            if(execution) {
                evidence="execution"+std::to_string(episode_.act_token);
                if(!knowledge.feedback(episode_.act_token,episode_.observed,episode_.executed,satisfaction))return false;
            } else {
                evidence="controlled_"+episode_.id;
                if(!record_answer(evidence,satisfaction))return false;
                if(!episode_.known.scene.complete)evidence.clear();
            }
            float reward=correct?1.f:-1.f;if(random_credit)reward=rng_()%2?1.f:-1.f;
            credit_index_[episode_.id]=credits_.size();credits_.push_back({episode_.id,evidence,episode_.trace,reward,satisfaction,execution,1,1});
            if(model_based_) {Update update{credits_.size()-1,0,false,true};updates_.push_back(update);apply_update(update);}
            for(size_t i=0;i<replay_updates;++i) {
                size_t receipt=i<episode_.trace.size()?credits_.size()-1:rng_()%credits_.size();
                const auto& trace=credits_[receipt].trace;
                size_t decision=terminal_only_?trace.size()-1:(i<episode_.trace.size()?trace.size()-1-i:rng_()%trace.size());
                Update update{receipt,decision,model_based_,false};updates_.push_back(update);apply_update(update);
            }
        } else if(execution&&knowledge.pending_&&knowledge.pending_->plan.token==episode_.act_token) {
            // Pending is evaluation framing, not learned evidence.
            knowledge.pending_.reset();
        }
        episode_.active=false;return true;
    }
    void cancel() {
        if(knowledge.pending_&&knowledge.pending_->plan.token==episode_.act_token)knowledge.pending_.reset();
        episode_.active=false;episode_.waiting=false;episode_.done=true;episode_.answer=Answer::Unknown;
    }
    bool revise_credit(const std::string& id,unsigned expected,bool correct,bool satisfaction,
                       const std::string& source="verified_environment") {
        auto found=credit_index_.find(id);
        if(found==credit_index_.end()||source!="verified_environment")return false;
        auto& r=credits_[found->second];if(r.version!=expected||expected==~0u)return false;
        if(r.mission)return false; // multi-evidence correction must be atomic
        if(!r.evidence.empty()) {
            bool accepted=r.execution?knowledge.revise_execution(r.evidence,r.evidence_version,satisfaction)
                :knowledge.revise_language(r.evidence,r.evidence_version,satisfaction,true,"verified_environment");
            if(!accepted)return false;
            ++r.evidence_version;
        }
        ++r.version;r.reward=correct?1.f:-1.f;r.satisfaction=satisfaction;
        policy.weights=initial_weights_;
        target_policy_.weights=initial_weights_;applied_updates_=0;empirical_values_.clear();control_model_.clear();control_states_.clear();
        for(const auto& update:updates_)apply_update(update);
        return true;
    }
};
using ControlledRelationalSession=BasicControlledRelationalSession<5>;
} // namespace laminar::relational
