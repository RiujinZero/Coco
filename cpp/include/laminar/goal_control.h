#pragma once
// One policy selects physical computation and goal management. Public requests
// supply outcomes to pursue, never the next goal/action or a solution sequence.
#include "relational_control.h"
#include "relational_goal_preview.h"
#include <limits>

namespace laminar::relational {
struct PublicGoal {
    std::string id,world_id,text,source="verified_request";
    std::set<std::string> names;
    StampedScene visible;
    std::vector<Action> actions;
    bool effect=false,generated=false;
    size_t deadline=0; // Public mission decision clock; zero means no deadline.
    unsigned version=1;
};
struct GoalControlLimits {
    size_t roots=3,derived=4,decisions=96,motors=32,parent_depth=2;
    size_t proposal_min_atoms=1; // Generic declared proposal filter; 1 preserves all positive goals.
    goal_preview::Limits preview;
    bool agenda_search_exhaustion=false; // Opt-in feature15 contract; default remains derived-goal membership.
};
struct GoalControlPrices {
    ControlPrices execution;
    float next_goal=.002f,activate_goal=.01f;
    goal_preview::Prices preview{.00001,.0000001,.0000001,.005,.005};
};
struct GoalControlStep:ControlStep {
    std::string inspected,world_id;
    uint64_t agenda_revision=0;
    size_t clock=0;
    template<class Ar> void state_io(Ar& ar){laminar::state::io(ar,static_cast<ControlStep&>(*this));ar(inspected);ar(world_id);ar(agenda_revision);ar(clock);}
};
struct GoalSubmission {
    std::string id,parent;
    bool correct=false,satisfaction=false;
    size_t clock=0,deadline=0;
    unsigned request_version=1;
};
struct GoalOutcomeRevision {std::string id;bool correct=false,satisfaction=false;};
struct GoalComputationReceipt {
    uint64_t token=0,model_before=0,model_after=0,primitive_steps=0;
    size_t plan_allowance=0; // Actual paid planner allowance, including the preview work cap; zero if no call.
    std::string mission,candidate,parent,status;
    goal_preview::Inputs inputs;
    goal_preview::Limits limits;
    GoalControlPrices prices;
    Answer answer=Answer::Unknown;
    Plan plan;
    std::optional<goal_preview::Result> preview;
    std::optional<goal_preview::ProposalResult> proposals;
    bool rebuilt=false,rejected=false;
    float operation_cost=0,rebuild_cost=0,planning_cost=0,primitive_cost=0,total_cost=0;
};

class GoalControlledRelationalSession:private BasicControlledRelationalSession<7> {
    using Base=BasicControlledRelationalSession<7>;
public:
    struct Entry {
        PublicGoal request;
        std::string parent,episode_id;
        uint64_t order=0,derived_version=0;
        size_t arrival=0,depth=0,motors=0,committed_motors=0;
        unsigned inspected_revision=0;
        bool started=false,complete=false,invalid=false,cached=false;
        size_t next_budget=4,last_plan_allowance=0;
        uint64_t cached_model=0,cached_world=0;
        Answer answer=Answer::Unknown;
        Plan plan;
        std::optional<goal_preview::Result> preview;
        std::optional<goal_preview::ProposalResult> proposals;
    };
private:
    std::map<std::string,Entry> goals_;
    std::map<std::string,StampedScene> worlds_;
    std::vector<GoalControlStep> mission_trace_;
    std::vector<GoalComputationReceipt> computations_;
    std::vector<ControlEvidenceLink> evidence_;
    std::vector<GoalSubmission> submissions_;
    std::map<std::string,std::string> terminal_links_;
    struct History {
        std::vector<GoalSubmission> submitted;
        std::map<std::string,std::string> terminal_links;
        std::vector<GoalComputationReceipt> computations;
        GoalControlLimits limits;
        GoalControlPrices prices;
        size_t roots=0;
    };
    std::map<std::string,History> history_;
    std::set<std::string> finished_missions_;
    std::mt19937 order_rng_;
    GoalControlLimits limits_;
    GoalControlPrices prices_;
    std::string mission_id_,active_id_,inspected_id_;
    bool mission_active_=false,agenda_=true;
    size_t mission_motors_=0,derived_serial_=0;
    uint64_t agenda_revision_=0;
    struct Activation {
        uint64_t token=0,agenda=0,model=0,world=0;
        unsigned request=0;
        std::string id;
    };
    std::optional<Activation> activation_;
    static bool finite_price(double x){return std::isfinite(x)&&x>=0&&x<=1;}
    static Scene records(Scene scene){std::sort(scene.facts.begin(),scene.facts.end());return scene;}
    static bool same_world(const StampedScene& a,const StampedScene& b) {
        return a.revision==b.revision&&records(a.scene)==records(b.scene);
    }
    const StampedScene& world(const Entry& e)const{return worlds_.at(e.request.world_id);}
    bool cache_current(const Entry& e)const {
        return e.cached&&e.cached_model==knowledge.version()&&e.cached_world==world(e).revision;
    }
    void invalidate(Entry& e) {
        e.cached=false;e.answer=Answer::Unknown;e.plan={};e.preview.reset();e.proposals.reset();e.next_budget=4;e.last_plan_allowance=0;
    }
    void synchronize_versions() {
        for(auto& [id,e]:goals_) {
            if(e.cached&&!cache_current(e))invalidate(e);
            // A derivation authorizes creation, not the lifetime of a goal that
            // has already started. Started children retain their request/trace
            // across observed model updates; resume discards their old plan and
            // interprets the request under the current learned model.
            if(!e.parent.empty()&&!e.started&&e.derived_version!=knowledge.version())e.invalid=true;
        }
    }
    bool update_world(const std::string& id,const StampedScene& incoming) {
        auto found=worlds_.find(id);
        if(found!=worlds_.end()) {
            if(incoming.revision<found->second.revision)return false;
            if(incoming.revision==found->second.revision&&!same_world(incoming,found->second))return false;
            if(same_world(incoming,found->second))return true;
        }
        worlds_[id]={records(incoming.scene),incoming.revision};
        for(auto& [key,e]:goals_)if(e.request.world_id==id)invalidate(e);
        ++agenda_revision_;return true;
    }
    bool uninspected()const {
        for(const auto& [id,e]:goals_)if(e.parent.empty()&&!e.complete&&!e.invalid&&e.inspected_revision!=e.request.version)return true;
        return false;
    }
    std::vector<std::string> order()const {
        std::vector<std::pair<uint64_t,std::string>> sorted;
        for(const auto& [id,e]:goals_)if(!e.complete&&!e.invalid)sorted.emplace_back(e.order,id);
        std::sort(sorted.begin(),sorted.end());std::vector<std::string> result;
        for(const auto& [key,id]:sorted)result.push_back(id);return result;
    }
    float slack(size_t deadline)const {
        if(!deadline)return 1.f;
        const double left=double(deadline)-double(mission_trace_.size());
        return float(std::clamp(left/double(limits_.decisions),-1.,1.));
    }
    goal_preview::Inputs preview_inputs(const Entry& e)const {
        goal_preview::Inputs input;
        input.parent_text=e.parent.empty()?e.request.text:goals_.at(e.parent).request.text;
        input.candidate_text=e.request.text;input.world=world(e).scene;input.actions=e.request.actions;
        input.names.assign(e.request.names.begin(),e.request.names.end());
        for(const auto& atom:input.world.facts)input.names.insert(input.names.end(),atom.args.begin(),atom.args.end());
        std::sort(input.names.begin(),input.names.end());input.names.erase(std::unique(input.names.begin(),input.names.end()),input.names.end());
        input.observation_revision=world(e).revision;
        input.parent_revision=e.parent.empty()?e.request.version:goals_.at(e.parent).request.version;
        input.candidate_revision=e.request.version;input.affordance_revision=e.request.version;return input;
    }
    bool all_roots_closed()const {
        bool any=false;for(const auto& [id,e]:goals_)if(e.parent.empty()){any=true;if(!e.complete)return false;}return any;
    }
    void collect_segments(const std::set<std::string>& before) {
        if(!learning())return;
        for(const auto& [id,r]:knowledge.execution_receipts())if(!before.count(id)) {
            ControlEvidenceLink link;link.id=id;link.version=r.version;link.execution=true;
            link.satisfaction=r.goal_verified?std::optional<bool>(r.success):std::nullopt;link.active=r.active;evidence_.push_back(link);
        }
    }
    bool suspend_active() {
        if(active_id_.empty())return true;
        std::set<std::string> before;for(const auto& [id,r]:knowledge.execution_receipts())before.insert(id);
        const auto motors=current_motor_count();
        if(!Base::suspend())return false;
        auto& e=goals_.at(active_id_);e.motors=e.committed_motors=motors;
        collect_segments(before);active_id_.clear();synchronize_versions();return true;
    }
    bool restart_active_segment() {
        if(Base::waiting()||Base::done())return false;
        auto id=active_id_;if(!suspend_active())return false;
        auto& e=goals_.at(id);if(!Base::resume(e.episode_id,world(e)))return false;
        active_id_=id;return true;
    }
    bool reconcile_active_world() {
        if(active_id_.empty()||same_world(Base::visible(),world(goals_.at(active_id_))))return true;
        auto candidate=*this;if(!candidate.restart_active_segment())return false;
        *this=std::move(candidate);return true;
    }
    bool flush_active_segment() {
        if(active_id_.empty()||current_motor_count()==goals_.at(active_id_).committed_motors)return true;
        // A paid next_goal closes only the acknowledged transition segment.
        // No satisfaction or control reward is supplied. Updating the model
        // before inspecting candidates prevents a later activation from
        // invalidating a newly proposed candidate as a side effect of switching.
        auto candidate=*this;if(!candidate.restart_active_segment())return false;
        *this=std::move(candidate);return true;
    }
    void propose_children(Entry& parent,ControlStep& step,std::optional<goal_preview::ProposalResult>& issued) {
        if(!parent.request.effect||parent.depth>=limits_.parent_depth||parent.proposals)return;
        size_t active=0;for(const auto& [id,e]:goals_)active+=!e.parent.empty()&&!e.invalid&&!e.complete;
        if(active>=limits_.derived)return;
        goal_preview::ProposalInputs input;input.parent_text=parent.request.text;input.world=world(parent).scene;
        input.actions=parent.request.actions;input.names.assign(parent.request.names.begin(),parent.request.names.end());
        input.observation_revision=world(parent).revision;input.affordance_revision=parent.request.version;input.order_seed=order_rng_();
        input.minimum_atoms=limits_.proposal_min_atoms;
        auto bounds=limits_.preview;bounds.candidates=std::min(bounds.candidates,limits_.derived-active);
        auto result=goal_preview::propose(knowledge,input,bounds,prices_.preview);
        step.cost+=float(result.cost);step.predictions+=result.predictions;
        if(result.complete)for(const auto& proposal:result.candidates) {
            if(proposal.text==parent.request.text)continue;
            bool duplicate=false;for(const auto& [id,e]:goals_)if(!e.invalid&&!e.complete&&e.parent==parent.request.id&&e.request.text==proposal.text)duplicate=true;
            if(duplicate)continue;
            Entry child;child.request=parent.request;child.request.id="derived_"+std::to_string(++derived_serial_);
            while(goals_.count(child.request.id))child.request.id="derived_"+std::to_string(++derived_serial_);
            child.request.text=proposal.text;child.request.names={proposal.names.begin(),proposal.names.end()};
            child.request.visible=world(parent);child.request.version=1;child.parent=parent.request.id;
            child.episode_id=mission_id_+"_g"+std::to_string(goals_.size());
            child.depth=parent.depth+1;child.arrival=clock();child.order=order_rng_();child.derived_version=knowledge.version();
            if(valid_control_request(child.request.text,child.request.names,child.request.visible,child.request.actions)) {
                auto id=child.request.id;goals_.emplace(id,std::move(child));++agenda_revision_;
            }
        }
        issued=result;parent.proposals=std::move(result);
    }
    GoalControlStep append(ControlStep step,const std::string& goal,const std::string& inspected) {
        GoalControlStep result;static_cast<ControlStep&>(result)=std::move(step);
        result.goal=goal;result.inspected=inspected;result.agenda_revision=agenda_revision_;
        result.clock=mission_trace_.size()+1;
        if(!goal.empty())result.world_id=goals_.at(goal).request.world_id;
        else if(!inspected.empty())result.world_id=goals_.at(inspected).request.world_id;
        mission_trace_.push_back(result);return result;
    }
    void next_candidate() {
        auto candidates=order();if(candidates.empty()){inspected_id_.clear();agenda_=true;return;}
        auto current=std::find(candidates.begin(),candidates.end(),inspected_id_);
        inspected_id_=current==candidates.end()||++current==candidates.end()?candidates.front():*current;
        auto& e=goals_.at(inspected_id_);e.inspected_revision=e.request.version;agenda_=true;++agenda_revision_;
    }
    void reason_candidate(ControlStep& step) {
        if(inspected_id_.empty()){step.rejected=true;step.status="no_inspected_goal";return;}
        auto& e=goals_.at(inspected_id_);
        GoalComputationReceipt receipt;receipt.token=step.token;receipt.mission=mission_id_;receipt.candidate=inspected_id_;
        receipt.parent=e.parent;receipt.model_before=knowledge.version();receipt.inputs=preview_inputs(e);
        receipt.limits=limits_.preview;receipt.limits.plan_predictions=std::min(limits_.preview.plan_predictions,e.next_budget);
        receipt.prices=prices_;receipt.operation_cost=step.cost;
        auto record=[&] {
            receipt.model_after=knowledge.version();receipt.status=step.status;receipt.rejected=step.rejected;
            receipt.rebuilt=step.rebuilt;receipt.primitive_steps=step.primitive_steps;
            receipt.answer=e.answer;receipt.plan=e.plan;receipt.total_cost=step.cost;
            computations_.push_back(std::move(receipt));
        };
        if(knowledge.dirty()){knowledge.rebuild();step.rebuilt=true;receipt.rebuild_cost=prices_.execution.rebuild;step.cost+=receipt.rebuild_cost;}
        synchronize_versions();
        if(e.invalid){step.rejected=true;step.status="stale_derived_goal";record();return;}
        const auto primitive_before=knowledge.primitive_steps();
        e.last_plan_allowance=0;
        if(!e.parent.empty()) {
            auto bounds=limits_.preview;bounds.plan_predictions=std::min(bounds.plan_predictions,e.next_budget);
            auto result=goal_preview::analyze(knowledge,preview_inputs(e),bounds,prices_.preview);
            receipt.plan_allowance=e.last_plan_allowance=result.planner_allowance;
            e.answer=result.answer;e.plan=result.plan;step.predictions=result.predictions;
            step.cost+=float(result.cost);receipt.preview=result;e.preview=std::move(result);
        } else {
            e.answer=knowledge.answer(e.request.text,world(e).scene);
            if(e.request.effect&&e.answer==Answer::False&&world(e).scene.complete) {
                receipt.plan_allowance=e.last_plan_allowance=std::min(limits_.preview.plan_predictions,e.next_budget);
                e.plan=knowledge.plan(e.request.text,world(e).scene,e.request.actions,
                    e.last_plan_allowance,limits_.preview.plan_depth,true);
                step.predictions=e.plan.predictions;step.reused=e.plan.reused;
                receipt.planning_cost=prices_.execution.prediction*float(step.predictions);step.cost+=receipt.planning_cost;
            }
        }
        e.cached=true;e.cached_model=knowledge.version();e.cached_world=world(e).revision;
        if(e.plan.status=="search_budget")e.next_budget=std::min(limits_.preview.plan_predictions,e.next_budget*2);
        if(e.answer==Answer::False&&e.plan.actions.empty())propose_children(e,step,receipt.proposals);
        step.primitive_steps=knowledge.primitive_steps()-primitive_before;
        receipt.primitive_cost=prices_.execution.prediction*float(step.primitive_steps);step.cost+=receipt.primitive_cost;
        step.status=e.plan.status.empty()?"goal_interpreted":e.plan.status;
        record();
    }
    bool commit_activation() {
        const auto ticket=*activation_;auto found=goals_.find(ticket.id);
        const bool valid=found!=goals_.end()&&!found->second.complete&&!found->second.invalid&&ticket.agenda==agenda_revision_
            &&ticket.model==knowledge.version()&&ticket.world==world(found->second).revision&&ticket.request==found->second.request.version;
        activation_.reset();
        if(!valid){mission_trace_.back().status="stale_goal_selection";return false;}
        if(active_id_==ticket.id){agenda_=false;mission_trace_.back().status="goal_active";return true;}
        if(!suspend_active()){mission_trace_.back().status="goal_suspension_rejected";return false;}
        auto& e=goals_.at(ticket.id);
        if(!e.parent.empty()&&!e.started&&e.derived_version!=knowledge.version()){mission_trace_.back().status="stale_derived_goal";return false;}
        const bool okay=e.started?Base::resume(e.episode_id,world(e)):Base::begin(e.episode_id,e.request.text,e.request.names,world(e),e.request.actions,e.request.effect,prices_.execution);
        if(!okay){mission_trace_.back().status="goal_activation_rejected";return false;}
        e.started=true;active_id_=ticket.id;agenda_=false;mission_trace_.back().status="goal_active";++agenda_revision_;return true;
    }
    bool commit_mission(size_t replay_updates,bool random_credit) {
        if(!active_id_.empty()&&!suspend_active())return false;
        for(const auto& id:suspended_goals())Base::cancel_suspended(id);
        bool correct=all_roots_closed();
        for(const auto& r:submissions_)if(r.parent.empty())correct&=r.correct&&(!r.deadline||r.clock<=r.deadline);
        std::vector<ControlStep> trace;for(const auto& step:mission_trace_)trace.push_back(step);
        if(!submit_mission_credit(mission_id_,trace,evidence_,correct,replay_updates,random_credit))return false;
        History retained;retained.submitted=submissions_;retained.terminal_links=terminal_links_;
        retained.computations=computations_;retained.limits=limits_;retained.prices=prices_;
        for(const auto& [id,e]:goals_)retained.roots+=e.parent.empty();history_[mission_id_]=std::move(retained);
        finished_missions_.insert(mission_id_);mission_active_=false;return true;
    }
public:
    // Expose model setup and read-only execution diagnostics, while keeping the
    // single-episode lifecycle inaccessible through this mission controller.
    using Base::knowledge;
    using Base::policy;
    using Base::credits;
    using Base::updates;
    using Base::initial_weights;
    using Base::empirical_values;
    using Base::learning;
    using Base::active;
    using Base::done;
    using Base::reply;
    using Base::trace;
    using Base::visible;
    using Base::plan;
    using Base::suspended_goals;
    // Complete learning state of the goal controller and everything it owns (laminar/state_io.h).
    template<class Ar> void state_io(Ar& ar) {
        Base::state_io(ar);
        ar.section("goal_control");
        ar(goals_);ar(worlds_);ar(mission_trace_);ar(computations_);ar(evidence_);ar(submissions_);ar(terminal_links_);
        ar(history_);ar(finished_missions_);ar(order_rng_);ar(limits_);ar(prices_);ar(mission_id_);ar(active_id_);ar(inspected_id_);
        ar(mission_active_);ar(agenda_);ar(mission_motors_);ar(derived_serial_);ar(agenda_revision_);ar(activation_);
    }
    explicit GoalControlledRelationalSession(unsigned seed=0,int hidden=32,bool learning=true)
        :Base(seed,learning,false,true,ControlBody{true,1,32,false,true},false,hidden,{},false,true),order_rng_(seed+7919) {preserve_observation_records();}
    void set_learning(bool enabled) {
        if(mission_active_)throw std::logic_error("cannot change learning during a mission");
        Base::set_learning(enabled);
    }
    size_t consolidate(size_t rounds) {
        if(mission_active_)throw std::logic_error("cannot consolidate during a mission");
        return Base::consolidate(rounds);
    }
    size_t fit_demonstrations(size_t rounds,const std::string& source) {
        if(mission_active_)throw std::logic_error("cannot fit demonstrations during a mission");
        return fit_demonstration_rows(rounds,source);
    }
    size_t fit_balanced_demonstrations(size_t steps,const std::string& source) {
        if(mission_active_)throw std::logic_error("cannot fit demonstrations during a mission");
        return fit_balanced_demonstration_rows(steps,source);
    }
    bool revise_credit(const std::string& id,unsigned expected,bool correct,bool satisfaction,
                       const std::string& source="verified_environment") {
        if(mission_active_)return false;
        return Base::revise_credit(id,expected,correct,satisfaction,source);
    }
    const auto& goals()const{return goals_;}
    const auto& submissions()const{return submissions_;}
    const auto& mission_trace()const{return mission_trace_;}
    const auto& mission_computations()const{return computations_;}
    const auto& computation_history(const std::string& mission)const{return history_.at(mission).computations;}
    const auto& mission_evidence()const{return evidence_;}
    const std::string& active_goal()const{return active_id_;}
    const std::string& inspected_goal()const{return inspected_id_;}
    bool in_agenda()const{return agenda_;}
    bool mission_active()const{return mission_active_;}
    size_t clock()const{return mission_trace_.size();}
    size_t acknowledged_motors()const{return mission_motors_;}
    uint64_t agenda_revision()const{return agenda_revision_;}
    bool waiting()const{return activation_.has_value()||Base::waiting();}
    bool begin_mission(const std::string& id,GoalControlLimits limits={},GoalControlPrices prices={}) {
        if(mission_active_||Base::active()||!suspended_goals().empty()||!concepts::symbol(id)||!concepts::symbol(id+"_g9999")||finished_missions_.count(id))return false;
        if(!limits.roots||limits.roots>16||limits.derived>32||!limits.decisions||limits.decisions>256
           ||!limits.motors||limits.motors>32||limits.parent_depth>4||!limits.preview.work||!limits.preview.plan_predictions
           ||!limits.proposal_min_atoms||limits.proposal_min_atoms>64
           ||!finite_price(prices.next_goal)||!finite_price(prices.activate_goal)
           ||!finite_price(prices.preview.prediction)||!finite_price(prices.preview.grounding)||!finite_price(prices.preview.atom_check)
           ||!finite_price(prices.preview.answer_call)||!finite_price(prices.preview.planner_call)
           ||!goal_preview::detail::valid(limits.preview,prices.preview))return false;
        try{prices.execution.validate();}catch(const std::invalid_argument&){return false;}
        goals_.clear();worlds_.clear();mission_trace_.clear();computations_.clear();evidence_.clear();submissions_.clear();terminal_links_.clear();
        active_id_.clear();inspected_id_.clear();activation_.reset();mission_motors_=derived_serial_=0;
        mission_id_=id;limits_=limits;prices_=prices;mission_active_=true;agenda_=true;++agenda_revision_;return true;
    }
    bool offer_goal(PublicGoal request) {
        if(!mission_active_||request.source!="verified_request"||request.generated||request.version!=1
           ||!concepts::symbol(request.id)||!concepts::symbol(request.world_id)||goals_.count(request.id)
           ||!valid_control_request(request.text,request.names,request.visible,request.actions))return false;
        size_t roots=0;for(const auto& [id,e]:goals_)roots+=e.parent.empty();if(roots>=limits_.roots)return false;
        auto internal_id=mission_id_+"_g"+std::to_string(goals_.size());if(!concepts::symbol(internal_id))return false;
        auto current=worlds_.find(request.world_id);
        if(current==worlds_.end()||request.visible.revision>=current->second.revision)
            if(!update_world(request.world_id,request.visible))return false;
        Entry entry;entry.request=std::move(request);entry.arrival=clock();entry.order=order_rng_();
        entry.episode_id=std::move(internal_id);
        auto id=entry.request.id;goals_.emplace(id,std::move(entry));++agenda_revision_;return true;
    }
    bool revise_deadline(const std::string& id,unsigned expected,size_t deadline,
                         const std::string& source="verified_request",bool generated=false) {
        auto found=goals_.find(id);
        if(!mission_active_||source!="verified_request"||generated||found==goals_.end()||found->second.complete
           ||!found->second.parent.empty()||found->second.request.version!=expected||expected==~0u)return false;
        std::set<std::string> descendants{id};
        for(size_t depth=0;depth<=limits_.parent_depth;++depth)
            for(const auto& [key,e]:goals_)if(descendants.count(e.parent))descendants.insert(key);
        for(const auto& key:descendants) {
            auto& e=goals_.at(key);if(e.complete)continue;
            e.request.deadline=deadline;++e.request.version;invalidate(e);
        }
        ++agenda_revision_;return true;
    }
    reasoning::Observation observe()const {
        if(!mission_active_)throw std::logic_error("no active mission");
        if(!agenda_) {auto result=Base::observe();result[30]=uninspected();return result;}
        reasoning::Observation o{};o[29]=o[31]=1;o[2]=knowledge.dirty();o[3]=knowledge.ready();
        if(!inspected_id_.empty()) {
            const auto& e=goals_.at(inspected_id_);const auto& visible=world(e);
            const bool current=cache_current(e);
            o[0]=visible.scene.complete;o[1]=!visible.scene.facts.empty();o[4]=e.request.effect;
            o[5]=current;o[6]=current&&e.answer==Answer::True;o[7]=current&&e.answer==Answer::False;o[8]=current&&e.answer==Answer::Unknown;
            o[9]=current&&!e.plan.actions.empty();o[10]=current&&e.plan.status=="search_budget";
            o[11]=e.started;o[12]=!active_id_.empty();o[13]=!e.parent.empty();o[14]=e.request.id==active_id_;o[15]=!e.parent.empty();
            if(limits_.agenda_search_exhaustion)
                o[15]=current&&e.plan.status=="search_budget"&&e.last_plan_allowance==limits_.preview.plan_predictions;
            o[16]=active_id_.empty()?1.f:slack(goals_.at(active_id_).request.deadline);o[17]=slack(e.request.deadline);
            bool preview=e.preview&&current&&goal_preview::current(knowledge,preview_inputs(e),*e.preview);
            o[18]=preview&&e.preview->valid;o[19]=o[18]?float(e.preview->normalized_delta):0;
            o[20]=float(e.request.id==active_id_?current_motor_count():e.motors)/32.f;
            o[30]=std::min(1.f,float(clock()-e.arrival)/float(limits_.decisions));
        }
        for(int i=0;i<5;++i)o[21+i]=20*prices_.execution.operation[i];
        o[26]=20*prices_.next_goal;o[27]=20*prices_.activate_goal;
        if(!active_id_.empty())o[28]=Base::plan().status=="no_plan"||Base::plan().status=="state_budget"||Base::plan().status=="search_budget";
        return o;
    }
    GoalControlStep choose(float epsilon=0,bool memory_policy=false,std::optional<ControlHint> hint={},std::optional<ControlHint> decided={}) {
        if(!mission_active_||goals_.empty()||waiting()||Base::done()&&!active_id_.empty()||clock()>=limits_.decisions||all_roots_closed())throw std::logic_error("no available mission decision");
        if(!reconcile_active_world())throw std::logic_error("cannot reconcile public world during a pending execution");
        synchronize_versions();
        std::vector<int> legal;
        if(agenda_) {legal={5};if(!inspected_id_.empty()&&!goals_.at(inspected_id_).invalid)legal={1,5,6};}
        else {legal={0,1,2,3,4,5};if(mission_motors_>=limits_.motors)legal.erase(std::find(legal.begin(),legal.end(),3));}
        auto step=select_control(observe(),legal,epsilon,memory_policy,nullptr,hint,decided);
        const std::string prior_active=active_id_,prior_inspected=inspected_id_;
        if(step.operation==ControlOp::next_goal) {
            step.cost=prices_.next_goal;
            if(flush_active_segment()){next_candidate();step.status="goal_inspected";}
            else {step.rejected=true;step.status="goal_segment_rejected";}
        } else if(step.operation==ControlOp::activate_goal) {
            const auto& e=goals_.at(inspected_id_);step.cost=prices_.activate_goal;
            activation_=Activation{step.token,agenda_revision_,knowledge.version(),world(e).revision,e.request.version,inspected_id_};
            step.status="awaiting_goal_activation";
        } else if(agenda_) {
            step.cost=prices_.execution.operation[1];reason_candidate(step);
        } else step=execute_physical(std::move(step));
        return append(std::move(step),prior_active,prior_inspected);
    }
    bool activate(uint64_t token) {
        if(!mission_active_||!activation_||activation_->token!=token)return false;
        auto candidate=*this;
        if(candidate.commit_activation()){*this=std::move(candidate);return true;}
        activation_.reset();mission_trace_.back().rejected=true;
        mission_trace_.back().status=candidate.mission_trace_.back().status;return false;
    }
    bool read(uint64_t token,const std::optional<StampedScene>& value,const std::string& source,bool generated=false) {
        if(!mission_active_||agenda_||active_id_.empty())return false;
        if(value) {
            const auto& known=world(goals_.at(active_id_));
            if(value->revision==known.revision&&!same_world(*value,known))return false;
            if(value->revision<known.revision)return Base::read(token,std::nullopt,source,generated);
        }
        if(!Base::read(token,value,source,generated))return false;
        if(Base::visible().revision<world(goals_.at(active_id_)).revision)return true;
        return update_world(goals_.at(active_id_).request.world_id,Base::visible());
    }
    bool actions(uint64_t token,const std::vector<Scene>& observed,const std::vector<bool>& executed,uint64_t revision,
                 const std::string& source="verified_environment",bool generated=false) {
        if(!mission_active_||agenda_||active_id_.empty()||mission_motors_+observed.size()>limits_.motors)return false;
        if(!observed.empty()) {
            const auto& known=world(goals_.at(active_id_));
            if(revision==known.revision&&!same_world({observed.back(),revision},known))return false;
        }
        if(!Base::actions(token,observed,executed,revision,source,generated))return false;
        mission_motors_+=observed.size();goals_.at(active_id_).motors=current_motor_count();
        if(Base::visible().revision<world(goals_.at(active_id_)).revision)return true;
        return update_world(goals_.at(active_id_).request.world_id,Base::visible());
    }
    bool close_goal(uint64_t token,bool correct,bool satisfaction,const std::string& source="verified_environment",bool generated=false) {
        if(!mission_active_||agenda_||active_id_.empty()||waiting()||!Base::done()||mission_trace_.empty()
           ||mission_trace_.back().token!=token||mission_trace_.back().operation!=ControlOp::finish)return false;
        auto link=close_episode_evidence(satisfaction,source,generated);if(!link)return false;
        if(!link->id.empty()){evidence_.push_back(*link);terminal_links_[active_id_]=link->id;}
        auto& e=goals_.at(active_id_);e.complete=true;
        submissions_.push_back({active_id_,e.parent,correct,satisfaction,clock(),e.request.deadline,e.request.version});
        active_id_.clear();inspected_id_.clear();agenda_=true;++agenda_revision_;synchronize_versions();return true;
    }
    bool finish_mission(size_t replay_updates=128,bool random_credit=false) {
        if(!mission_active_||waiting()||mission_trace_.empty()||(!all_roots_closed()&&clock()<limits_.decisions))return false;
        auto candidate=*this;if(!candidate.commit_mission(replay_updates,random_credit))return false;
        *this=std::move(candidate);return true;
    }
    bool revise_mission(const std::string& id,unsigned expected,const std::vector<GoalOutcomeRevision>& outcomes,
                        bool active=true,const std::string& source="verified_environment",bool generated=false) {
        if(mission_active_||generated||source!="verified_environment")return false;
        auto old=history_.find(id);if(old==history_.end()||outcomes.size()!=old->second.submitted.size())return false;
        auto receipt=std::find_if(credits().begin(),credits().end(),[&](const auto& r){return r.id==id&&r.mission;});
        if(receipt==credits().end())return false;
        auto revised=old->second;std::set<std::string> seen;auto links=receipt->links;
        for(auto& link:links)link.active=active;
        for(const auto& update:outcomes) {
            if(!seen.insert(update.id).second)return false;
            auto row=std::find_if(revised.submitted.begin(),revised.submitted.end(),[&](const auto& r){return r.id==update.id;});
            if(row==revised.submitted.end())return false;
            row->correct=update.correct;row->satisfaction=update.satisfaction;
            auto linked=revised.terminal_links.find(update.id);
            if(linked!=revised.terminal_links.end()) {
                auto link=std::find_if(links.begin(),links.end(),[&](const auto& r){return r.id==linked->second;});
                if(link==links.end())return false;link->satisfaction=update.satisfaction;
            }
        }
        size_t closed=0;bool correct=true;
        for(const auto& row:revised.submitted)if(row.parent.empty()){++closed;correct&=row.correct&&(!row.deadline||row.clock<=row.deadline);}
        correct&=closed==revised.roots;
        if(!revise_mission_credit(id,expected,correct,links,active,source))return false;
        history_[id]=std::move(revised);
        if(id==mission_id_) {
            submissions_=history_.at(id).submitted;
            auto updated=std::find_if(credits().begin(),credits().end(),[&](const auto& r){return r.id==id;});evidence_=updated->links;
        }
        return true;
    }
    bool import_warm_policy(const reasoning::BackpropChoice& source) {
        if(mission_active_)return false;return import_five_action_policy(source);
    }
};
} // namespace laminar::relational
