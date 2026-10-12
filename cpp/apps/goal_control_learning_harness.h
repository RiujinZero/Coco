#pragma once
// Evaluator/task plumbing only. The continuing controller owns every decision,
// request activation, execution token, semantic receipt and policy update.
#include "laminar/goal_control.h"
#include <bit>
#include <iomanip>
#include <sstream>

namespace laminar::goal_learning {
using namespace relational;
using Observation=reasoning::Observation;
inline void require(bool okay,const char* message) {if(!okay)throw std::runtime_error(message);}
inline std::string quote(const std::string& value) {
    std::ostringstream out;out<<'"';for(unsigned char c:value) {
        if(c=='"'||c=='\\')out<<'\\'<<c;
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<c;
    }out<<'"';return out.str();
}
inline void numbers(std::ostream& out,const auto& values) {
    out<<'[';bool first=true;for(const auto& value:values){if(!first)out<<',';first=false;out<<value;}out<<']';
}
inline void strings(std::ostream& out,const auto& values) {
    out<<'[';bool first=true;for(const auto& value:values){if(!first)out<<',';first=false;out<<quote(value);}out<<']';
}
inline void scene(std::ostream& out,const StampedScene& value) {
    out<<"{\"revision\":"<<value.revision<<",\"complete\":"<<value.scene.complete<<",\"facts\":[";
    bool first=true;for(const auto& atom:value.scene.facts) {
        if(!first)out<<',';first=false;out<<"{\"predicate\":"<<quote(atom.predicate)<<",\"args\":";strings(out,atom.args);out<<'}';
    }out<<"]}";
}
inline void actions(std::ostream& out,const std::vector<Action>& values) {
    out<<'[';bool first=true;for(const auto& action:values) {
        if(!first)out<<',';first=false;out<<"{\"schema\":"<<quote(action.schema)<<",\"args\":";strings(out,action.arguments);out<<'}';
    }out<<']';
}
inline std::string observation_bits(const Observation& o) {
    std::ostringstream key;key<<std::hex;for(float value:o)key<<std::setw(8)<<std::setfill('0')<<std::bit_cast<uint32_t>(value);
    return key.str();
}
inline int public_reference(const Observation& o,const std::vector<int>& menu) {
    // Declared protocol assistance only. Never called for autonomous/eval rows.
    // These parameters are the complete information boundary: no goal IDs,
    // evaluator truth, private clock, session object, or desired next action.
    auto legal=[&](int op){return std::find(menu.begin(),menu.end(),op)!=menu.end();};
    int op=5;
    if(o[29]) {if(legal(6))op=(!o[4]||!o[12]||!o[14])?6:5;}
    else if(o[30])op=5;
    else if(!o[5])op=1;
    else if(!o[4]||o[6])op=4;
    else if(o[9])op=3;
    else if(o[10])op=1;
    else if(o[8])op=2;
    else op=4;
    require(legal(op),"protocol reference selected unavailable operation");return op;
}

struct ScheduledRequest {size_t clock=0;PublicGoal request;};
// A later long-route family replaces this environment, not the controller or
// its feature contract. Evaluator truth is accessible only to outcome handling.
class Environment {
public:
    virtual ~Environment()=default;
    virtual const std::vector<ScheduledRequest>& schedule()const=0;
    virtual StampedScene read(const std::string& world)const=0;
    virtual StampedScene execute(const std::string& world,const Action& action)=0;
    virtual bool satisfaction(const PublicGoal& request)const=0;
    virtual const char* family()const=0;
};
struct MissionOptions {
    std::string id,arm;
    size_t index=0,terminal_updates=128;
    float epsilon=0;
    double hint_probability=0;
    // Explicit curriculum hook. Null retains the original protocol examples;
    // source is recorded on every assisted decision and fitted only on request.
    int (*demonstration)(const Observation&,const std::vector<int>&)=nullptr;
    std::string demonstration_source="protocol_bootstrap";
    bool random_credit=false,memory_policy=false;
    uint32_t assistance_seed=0,task_order_seed=0;
    GoalControlLimits limits;
    GoalControlPrices prices;
};
struct MissionResult {
    std::string id,arm;
    size_t index=0,decisions=0,motors=0,predictions=0,hints=0,posted=0;
    size_t updates_before=0,updates_after=0,credits_before=0,credits_after=0;
    uint64_t primitive_steps=0,model_before=0,model_after=0;
    bool success=false,interrupted=false,resumed=false,arrival_during_execution=false;
    bool learning=false,random_credit=false;
    double cost=0,discounted_cost=0,utility=0,observed_reward=-1;
    std::optional<double> credited_reward;
};
inline bool closed(const GoalControlledRelationalSession& learner) {
    for(const auto& [id,e]:learner.goals())if(e.parent.empty()&&!e.complete)return false;
    return !learner.goals().empty();
}
inline std::vector<int> public_menu(const GoalControlledRelationalSession& learner,const GoalControlLimits& limits) {
    if(learner.in_agenda()) {
        if(!learner.inspected_goal().empty()&&!learner.goals().at(learner.inspected_goal()).invalid)return {1,5,6};
        return {5};
    }
    if(learner.acknowledged_motors()>=limits.motors)return {0,1,2,4,5};
    return {0,1,2,3,4,5};
}
inline void weights(std::ostream& out,const GoalControlledRelationalSession& learner,
                    const std::string& label,const std::string& arm) {
    out<<"{\"label\":"<<quote(label)<<",\"arm\":"<<quote(arm)<<",\"updates\":"<<learner.updates()
       <<",\"credits\":"<<learner.credits().size()<<",\"model_version\":"<<learner.knowledge.version()
       <<",\"hidden\":"<<learner.policy.hidden<<",\"actions\":"<<learner.policy.actions<<",\"weights\":";
    numbers(out,learner.policy.weights);out<<"}\n";out.flush();
}
inline void result_json(std::ostream& out,const MissionResult& r) {
    out<<"{\"id\":"<<quote(r.id)<<",\"arm\":"<<quote(r.arm)<<",\"index\":"<<r.index
       <<",\"success\":"<<r.success<<",\"learning\":"<<r.learning<<",\"random_credit\":"<<r.random_credit
       <<",\"decisions\":"<<r.decisions<<",\"motors\":"<<r.motors<<",\"predictions\":"<<r.predictions
       <<",\"primitive_steps\":"<<r.primitive_steps<<",\"hints\":"<<r.hints<<",\"posted_requests\":"<<r.posted
       <<",\"interrupted\":"<<r.interrupted<<",\"resumed\":"<<r.resumed<<",\"arrival_during_execution\":"<<r.arrival_during_execution
       <<",\"updates_before\":"<<r.updates_before<<",\"updates_after\":"<<r.updates_after
       <<",\"credits_before\":"<<r.credits_before<<",\"credits_after\":"<<r.credits_after
       <<",\"model_before\":"<<r.model_before<<",\"model_after\":"<<r.model_after
       <<",\"cost\":"<<r.cost<<",\"discounted_cost\":"<<r.discounted_cost<<",\"observed_reward\":"<<r.observed_reward
       <<",\"credited_reward\":";
    if(r.credited_reward)out<<*r.credited_reward;else out<<"null";
    out<<",\"observed_utility\":"<<r.utility<<'}';
}
inline MissionResult run_mission(GoalControlledRelationalSession& learner,Environment& environment,
                                 const MissionOptions& options,std::ostream& log,std::ostream& results) {
    require(!learner.mission_active(),"mission boundary violated");
    require(options.hint_probability>=0&&options.hint_probability<=1,"invalid hint probability");
    require(options.hint_probability==0||concepts::symbol(options.demonstration_source),"invalid demonstration source");
    require(learner.learning()||options.hint_probability==0,"evaluation requested assistance");
    MissionResult out;out.id=options.id;out.arm=options.arm;out.index=options.index;
    out.learning=learner.learning();out.random_credit=options.random_credit;
    out.model_before=learner.knowledge.version();out.updates_before=learner.updates();out.credits_before=learner.credits().size();
    auto policy_before=learner.policy.weights;std::mt19937 assistance(options.assistance_seed);
    require(learner.begin_mission(options.id,options.limits,options.prices),"mission begin rejected");
    const auto& schedule=environment.schedule();require(!schedule.empty()&&schedule.front().clock==0,"missing initial public request");
    for(size_t i=1;i<schedule.size();++i)require(schedule[i].clock>=schedule[i-1].clock,"unsorted public schedule");
    log<<"{\"mission\":"<<quote(out.id)<<",\"event\":\"mission_begin\",\"arm\":"<<quote(out.arm)
       <<",\"family\":"<<quote(environment.family())<<",\"index\":"<<out.index<<",\"learning\":"<<out.learning
       <<",\"epsilon\":"<<options.epsilon<<",\"hint_probability\":"<<options.hint_probability
       <<",\"demonstration_source\":"<<quote(options.demonstration_source)
       <<",\"memory_policy\":"<<options.memory_policy
       <<",\"assistance_seed\":"<<options.assistance_seed<<",\"task_order_seed\":"<<options.task_order_seed
       <<",\"terminal_updates\":"<<options.terminal_updates<<",\"random_credit\":"<<options.random_credit<<"}\n";
    std::set<std::string> interrupted_goals;
    auto post=[&]() {
        while(out.posted<schedule.size()&&schedule[out.posted].clock==learner.clock()) {
            auto goal=schedule[out.posted].request;
            goal.visible=environment.read(goal.world_id);
            out.arrival_during_execution|=out.posted>0&&!learner.in_agenda();
            require(learner.offer_goal(goal),"scheduled public input rejected");++out.posted;
            log<<"{\"mission\":"<<quote(out.id)<<",\"event\":\"public_request\",\"clock\":"<<learner.clock()
               <<",\"id\":"<<quote(goal.id)<<",\"world_id\":"<<quote(goal.world_id)<<",\"text\":"<<quote(goal.text)
               <<",\"source\":"<<quote(goal.source)<<",\"generated\":"<<goal.generated<<",\"effect\":"<<goal.effect
               <<",\"deadline\":"<<goal.deadline<<",\"request_version\":"<<goal.version
               <<",\"agenda_order_key_audit_only\":"<<learner.goals().at(goal.id).order<<",\"names\":";strings(log,goal.names);
            log<<",\"visible\":";scene(log,goal.visible);log<<",\"actions\":";actions(log,goal.actions);log<<"}\n";
        }
    };
    post();
    while(learner.clock()<options.limits.decisions&&!closed(learner)) {
        post();const auto observation=learner.observe();const auto menu=public_menu(learner,options.limits);
        std::optional<ControlHint> hint;
        if(options.hint_probability>0&&std::generate_canonical<double,53>(assistance)<options.hint_probability) {
            const auto demonstrate=options.demonstration?options.demonstration:public_reference;
            hint=ControlHint{ControlOp(demonstrate(observation,menu)),options.demonstration_source};++out.hints;
        }
        const auto active_before=learner.active_goal(),inspected_before=learner.inspected_goal();
        const auto model_before=learner.knowledge.version();
        const auto issued=learner.choose(options.epsilon,options.memory_policy,hint);
        require(issued.observation==observation&&issued.available==menu,"public observation/menu reconstruction drift");
        require(issued.teacher==(hint?hint->source:std::string{}),"unlogged control assistance");
        std::string callback_source;bool correct=false,satisfaction=false;
        std::vector<StampedScene> callback_worlds;
        if(issued.operation==ControlOp::activate_goal) {
            require(learner.activate(issued.token),"owned activation rejected");callback_source="owned_activation_token";
            if(!active_before.empty()&&active_before!=learner.active_goal()) {
                interrupted_goals.insert(active_before);out.interrupted=true;
            }
            out.resumed|=interrupted_goals.count(learner.active_goal())!=0;
        } else if(issued.operation==ControlOp::act&&!issued.actions.empty()) {
            const auto& world=learner.goals().at(active_before).request.world_id;
            std::vector<Scene> observations;std::vector<bool> executed;
            for(const auto& action:issued.actions) {
                callback_worlds.push_back(environment.execute(world,action));observations.push_back(callback_worlds.back().scene);executed.push_back(true);
            }
            require(learner.actions(issued.token,observations,executed,callback_worlds.back().revision),"owned physics callback rejected");
            callback_source="verified_environment";
        } else if(issued.operation==ControlOp::ask||issued.operation==ControlOp::retrieve) {
            callback_worlds.push_back(environment.read(learner.goals().at(active_before).request.world_id));
            callback_source=issued.operation==ControlOp::ask?"verified_environment":"trusted_memory";
            require(learner.read(issued.token,callback_worlds.back(),callback_source),"owned read callback rejected");
        } else if(issued.operation==ControlOp::finish) {
            satisfaction=environment.satisfaction(learner.goals().at(active_before).request);
            correct=issued.reply==(satisfaction?Answer::True:Answer::False);
            if(learner.goals().at(active_before).request.effect)correct&=satisfaction;
            require(learner.close_goal(issued.token,correct,satisfaction),"owned goal label rejected");callback_source="verified_environment";
        }
        const auto& row=learner.mission_trace().back();
        out.cost+=row.cost;out.discounted_cost+=std::pow(.99,double(row.clock-1))*row.cost;
        out.predictions+=row.predictions;out.primitive_steps+=row.primitive_steps;
        log<<"{\"mission\":"<<quote(out.id)<<",\"event\":\"decision\",\"clock\":"<<row.clock<<",\"token\":"<<row.token
           <<",\"observation\":";numbers(log,observation);log<<",\"observation_bits\":"<<quote(observation_bits(observation))
           <<",\"menu\":";numbers(log,menu);log<<",\"operation\":"<<int(row.operation)<<",\"teacher\":"<<quote(row.teacher)
           <<",\"active_before\":"<<quote(active_before)<<",\"inspected_before\":"<<quote(inspected_before)
           <<",\"active_after\":"<<quote(learner.active_goal())<<",\"inspected_after\":"<<quote(learner.inspected_goal())
           <<",\"model_before\":"<<model_before<<",\"model_after\":"<<learner.knowledge.version()
           <<",\"agenda_revision\":"<<row.agenda_revision<<",\"cost\":"<<row.cost<<",\"predictions\":"<<row.predictions
           <<",\"primitive_steps\":"<<row.primitive_steps<<",\"reply\":"<<int(row.reply)<<",\"status\":"<<quote(row.status)
           <<",\"rejected\":"<<row.rejected<<",\"rebuilt\":"<<row.rebuilt<<",\"reused\":"<<row.reused<<",\"actions\":";actions(log,row.actions);
        log<<",\"callback_source\":"<<quote(callback_source)<<",\"callback_generated\":false,\"observed_worlds\":[";
        bool first=true;for(const auto& world:callback_worlds){if(!first)log<<',';first=false;scene(log,world);}log<<']';
        if(issued.operation==ControlOp::act&&!issued.actions.empty()) {
            log<<",\"executed\": [";for(size_t i=0;i<issued.actions.size();++i){if(i)log<<',';log<<"true";}log<<']';
        }
        if(issued.operation==ControlOp::finish)log<<",\"terminal_correct\":"<<correct<<",\"terminal_satisfaction\":"<<satisfaction;
        log<<"}\n";
        require(learner.policy.weights==policy_before&&learner.updates()==out.updates_before&&learner.credits().size()==out.credits_before,
            "policy changed before mission terminal");
        if(!out.learning)require(learner.knowledge.version()==out.model_before&&learner.mission_evidence().empty(),"frozen evaluation changed semantics");
    }
    out.decisions=learner.clock();out.motors=learner.acknowledged_motors();
    bool known_correct=closed(learner);
    for(const auto& s:learner.submissions())if(s.parent.empty())known_correct&=s.correct&&(!s.deadline||s.clock<=s.deadline);
    // Public schedules must preclude successful termination before an unseen
    // required root. An incorrect early close is an ordinary mission failure.
    require(!known_correct||out.posted==schedule.size(),"successful early close omitted a required future request");
    out.success=known_correct&&out.posted==schedule.size();out.observed_reward=out.success?1:-1;
    out.utility=std::pow(.99,double(out.decisions-1))*out.observed_reward-out.discounted_cost;
    require(learner.finish_mission(options.terminal_updates,options.random_credit),"mission credit rejected");
    out.model_after=learner.knowledge.version();out.updates_after=learner.updates();out.credits_after=learner.credits().size();
    require(out.updates_after-out.updates_before==(out.learning?options.terminal_updates:0),"unexpected policy update count");
    require(out.credits_after-out.credits_before==size_t(out.learning),"unexpected mission credit count");
    if(!out.learning||options.terminal_updates==0)require(learner.policy.weights==policy_before,"frozen-policy control changed weights");
    if(out.learning) {
        const auto& credit=learner.credits().back();out.credited_reward=credit.reward;
        require(credit.satisfaction==out.success,"mission evaluator disagrees with actual correctness receipt");
        if(!options.random_credit)require(credit.reward==out.observed_reward,"factual terminal credit changed reward");
        log<<"{\"mission\":"<<quote(out.id)<<",\"event\":\"mission_credit\",\"source\":\"verified_environment\",\"version\":"<<credit.version
           <<",\"reward\":"<<credit.reward<<",\"actual_correct\":"<<credit.satisfaction<<",\"random_credit\":"<<options.random_credit<<",\"links\":[";
        bool first=true;for(const auto& link:credit.links) {
            if(!first)log<<',';first=false;log<<"{\"id\":"<<quote(link.id)<<",\"version\":"<<link.version
                <<",\"execution\":"<<link.execution<<",\"active\":"<<link.active<<",\"satisfaction\":";
            if(link.satisfaction)log<<*link.satisfaction;else log<<"null";log<<'}';
        }log<<"]}\n";
    }
    log<<"{\"mission\":"<<quote(out.id)<<",\"event\":\"mission_end\",\"result\":";result_json(log,out);log<<"}\n";log.flush();
    result_json(results,out);results<<'\n';results.flush();return out;
}
} // namespace laminar::goal_learning
