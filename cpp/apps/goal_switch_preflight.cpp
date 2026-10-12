// Evaluator feasibility witness, never a policy teacher or training campaign.
#include "laminar/goal_control.h"
#include "../tests/goal_control_fixture.h"
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace gt=laminar::goal_test;
using namespace laminar::relational;
using Observation=laminar::reasoning::Observation;
static constexpr size_t decision_limit=96,motor_limit=32,urgent_slack=12;
static constexpr double discount=.99,required_margin=.10;
static const std::vector<unsigned> order_seeds{7,11,19};
static const std::vector<size_t> arrival_clocks{5,11,17,23};

// This entire function has only the actual public policy inputs. No ID, clock,
// evaluator, session, planner, answer label or trace can enter this interface.
static int reference_choice(const Observation& o,const std::vector<int>& menu) {
    auto legal=[&](int op){return std::find(menu.begin(),menu.end(),op)!=menu.end();};
    int operation=5;
    if(o[29]) {
        if(legal(6))operation=(!o[4]||!o[12]||!o[14])?6:5;
    } else if(o[30])operation=5;
    else if(!o[5])operation=1;
    else if(!o[4]||o[6])operation=4;
    else if(o[9])operation=3;
    else if(o[10])operation=1;
    else if(o[8])operation=2;
    else operation=4;
    gt::require(legal(operation),"reference chose an unavailable operation");return operation;
}

static GoalControlLimits limits() {
    GoalControlLimits result;result.roots=2;result.derived=0;
    result.decisions=decision_limit;result.motors=motor_limit;return result;
}
static GoalControlPrices prices() {
    // Fixed before all measurements. Every other field is the declared default.
    GoalControlPrices result;
    result.execution.operation={.01f,.005f,.06f,.01f,0.f};
    result.execution.prediction=.00001f;result.execution.motor=.02f;result.execution.rebuild=.005f;
    result.next_goal=.002f;result.activate_goal=.01f;
    result.preview={.00001,.0000001,.0000001,.005,.005};return result;
}
static PublicGoal request(const gt::PublicRequestFixture& source) {
    PublicGoal out;out.id=source.id;out.world_id=source.world_id;out.text=source.text;
    out.names=source.names;out.visible=source.visible;out.actions=source.actions;
    out.effect=source.effect;out.deadline=source.deadline;return out;
}
static std::vector<int> public_menu(const GoalControlledRelationalSession& learner) {
    // Affordance reconstruction only. Assert equality with choose()'s own menu
    // on every row. IDs are used here solely to access public validity metadata.
    if(learner.in_agenda()) {
        if(!learner.inspected_goal().empty()&&!learner.goals().at(learner.inspected_goal()).invalid)return {1,5,6};
        return {5};
    }
    if(learner.acknowledged_motors()>=motor_limit)return {0,1,2,4,5};
    return {0,1,2,3,4,5};
}
static GoalControlStep prefer(GoalControlledRelationalSession& learner,int operation) {
    auto saved=learner.policy.weights;
    std::fill(learner.policy.weights.begin(),learner.policy.weights.end(),0);
    learner.policy.weights[learner.policy.output_bias()+operation]=1;
    try {auto step=learner.choose();learner.policy.weights=std::move(saved);return step;}
    catch(...){learner.policy.weights=std::move(saved);throw;}
}
static std::string quoted(const std::string& value) {
    std::ostringstream out;out<<'"';for(unsigned char c:value) {
        if(c=='"'||c=='\\')out<<'\\'<<c;
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<c;
    }out<<'"';return out.str();
}
template<class Range> static void numbers(std::ostream& out,const Range& values) {
    out<<'[';bool first=true;for(const auto& value:values){if(!first)out<<',';first=false;out<<value;}out<<']';
}
static void strings(std::ostream& out,const auto& values) {
    out<<'[';bool first=true;for(const auto& value:values){if(!first)out<<',';first=false;out<<quoted(value);}out<<']';
}
static void scene(std::ostream& out,const StampedScene& value) {
    out<<"{\"revision\":"<<value.revision<<",\"complete\":"<<value.scene.complete<<",\"facts\":[";
    bool first=true;for(const auto& atom:value.scene.facts) {
        if(!first)out<<',';first=false;out<<"{\"predicate\":"<<quoted(atom.predicate)<<",\"args\":";strings(out,atom.args);out<<'}';
    }out<<"]}";
}
static void actions(std::ostream& out,const std::vector<Action>& values) {
    out<<'[';bool first=true;for(const auto& a:values) {
        if(!first)out<<',';first=false;out<<"{\"schema\":"<<quoted(a.schema)<<",\"args\":";strings(out,a.arguments);out<<'}';
    }out<<']';
}
struct AliasClass {size_t rows=0;std::set<int> operations;std::set<std::string> contexts,runs;};
static std::map<std::string,AliasClass> aliases;
static std::string observation_key(const Observation& o,const std::vector<int>& menu) {
    std::ostringstream key;key<<std::hex;for(float v:o)key<<std::setw(8)<<std::setfill('0')<<std::bit_cast<uint32_t>(v);
    key<<':';for(int op:menu)key<<op;return key.str();
}
struct Run {
    std::string id,strategy;
    unsigned seed=0;size_t arrival=0,decisions=0,motors=0,predictions=0;
    uint64_t primitive_steps=0;
    bool success=false,urgent_posted=false,arrival_during_execution=false,arrival_flag_only=false,semantic_learning=false;
    bool interrupted=false,resumed=false,physical_satisfied=false,urgent_correct=false;
    size_t physical_motors_at_arrival=0,physical_close=0,urgent_close=0;
    uint64_t model_start=0,model_end=0;
    size_t final_credit_receipts=0;
    double reward=-1,cost=0,discounted_cost=0,utility=0;
};
static bool roots_closed(const GoalControlledRelationalSession& learner) {
    for(const auto& [id,entry]:learner.goals())if(entry.parent.empty()&&!entry.complete)return false;
    return !learner.goals().empty();
}
static Run run(const RelationalSession& knowledge,unsigned seed,size_t arrival,
               const std::string& strategy,std::ostream& trace) {
    const bool semantic_learning=strategy=="reference_semantic";
    const bool reference=strategy=="reference"||semantic_learning;
    auto fixture=gt::make_fixture();GoalControlledRelationalSession learner(seed,32,semantic_learning);
    learner.knowledge=knowledge;auto weights=learner.policy.weights;
    const auto initial_version=learner.knowledge.version();
    Run result;result.strategy=strategy;result.seed=seed;result.arrival=arrival;result.semantic_learning=semantic_learning;
    result.model_start=initial_version;
    result.id=strategy+"_s"+std::to_string(seed)+"_a"+std::to_string(arrival);
    gt::require(learner.begin_mission(result.id,limits(),prices()),"mission begin failed");
    auto physical=request(fixture.request("physical"));
    gt::require(learner.offer_goal(physical),"physical root rejected");
    auto emit_request=[&](const PublicGoal& goal) {
        trace<<"{\"run\":"<<quoted(result.id)<<",\"event\":\"public_request\",\"clock\":"<<learner.clock()
             <<",\"id\":"<<quoted(goal.id)<<",\"world_id\":"<<quoted(goal.world_id)<<",\"text\":"<<quoted(goal.text)
             <<",\"effect\":"<<goal.effect<<",\"deadline\":"<<goal.deadline
             <<",\"candidate_order_key_audit_only\":"<<learner.goals().at(goal.id).order<<",\"names\":";strings(trace,goal.names);
        trace<<",\"visible\":";scene(trace,goal.visible);trace<<",\"actions\":";actions(trace,goal.actions);trace<<"}\n";
    };
    emit_request(physical);
    while(learner.clock()<decision_limit&&!roots_closed(learner)) {
        if(learner.clock()==arrival) {
            auto before=learner.observe();auto urgent=request(fixture.request("arrival"));
            urgent.visible=fixture.evaluator.read(urgent.world_id);urgent.deadline=arrival+urgent_slack;
            result.arrival_during_execution=!learner.in_agenda()&&learner.active_goal()=="physical";
            result.physical_motors_at_arrival=learner.acknowledged_motors();
            gt::require(learner.offer_goal(urgent),"urgent root rejected");result.urgent_posted=true;
            auto after=learner.observe();auto expected=before;expected[30]=1;
            result.arrival_flag_only=result.arrival_during_execution&&before[30]==0&&after==expected;
            emit_request(urgent);
        }
        auto observation=learner.observe();auto menu=public_menu(learner);
        const auto active_before=learner.active_goal(),inspected_before=learner.inspected_goal();
        const auto version_before=learner.knowledge.version();
        int selected=-1;
        if(reference)selected=reference_choice(observation,menu);
        else if(strategy=="delayed_failure")selected=5;
        else if(strategy=="immediate_failure")selected=learner.in_agenda()?
            (std::find(menu.begin(),menu.end(),6)!=menu.end()?6:5):4;
        auto issued=selected<0?learner.choose():prefer(learner,selected);
        gt::require(issued.observation==observation&&issued.available==menu,"public input/menu reconstruction drift");
        gt::require(selected<0||int(issued.operation)==selected,"reference/control choice was not issued");
        const auto operation=issued.operation;
        bool callback=false,terminal_correct=false,terminal_satisfaction=false;
        StampedScene callback_world;
        if(operation==ControlOp::activate_goal) {
            gt::require(learner.activate(issued.token),"owned activation rejected");callback=true;
            result.interrupted|=active_before=="physical"&&learner.active_goal()=="arrival";
            result.resumed|=result.interrupted&&learner.active_goal()=="physical";
        } else if(operation==ControlOp::act&&!issued.actions.empty()) {
            const auto world_id=learner.goals().at(active_before).request.world_id;
            std::vector<Scene> observed;std::vector<bool> executed;
            for(const auto& a:issued.actions) {
                callback_world=fixture.evaluator.execute(world_id,a);observed.push_back(callback_world.scene);executed.push_back(true);
            }
            gt::require(learner.actions(issued.token,observed,executed,callback_world.revision),"owned action callback rejected");callback=true;
        } else if(operation==ControlOp::ask||operation==ControlOp::retrieve) {
            callback_world=fixture.evaluator.read(learner.goals().at(active_before).request.world_id);
            gt::require(learner.read(issued.token,callback_world,operation==ControlOp::ask?"verified_environment":"trusted_memory"),"owned read callback rejected");callback=true;
        } else if(operation==ControlOp::finish) {
            terminal_satisfaction=fixture.evaluator.satisfied(active_before);
            terminal_correct=issued.reply==(terminal_satisfaction?Answer::True:Answer::False);
            if(learner.goals().at(active_before).request.effect)terminal_correct&=terminal_satisfaction;
            gt::require(learner.close_goal(issued.token,terminal_correct,terminal_satisfaction),"owned terminal callback rejected");callback=true;
            if(active_before=="physical")result.physical_close=learner.clock();
            if(active_before=="arrival"){result.urgent_close=learner.clock();result.urgent_correct=terminal_correct;}
        }
        const auto& row=learner.mission_trace().back();
        result.cost+=row.cost;result.discounted_cost+=std::pow(discount,double(row.clock-1))*row.cost;
        result.predictions+=row.predictions;result.primitive_steps+=row.primitive_steps;
        if(reference) {
            auto& group=aliases[observation_key(observation,menu)];++group.rows;group.operations.insert(int(row.operation));
            group.contexts.insert(active_before+"/"+inspected_before);group.runs.insert(result.id);
        }
        trace<<"{\"run\":"<<quoted(result.id)<<",\"event\":\"decision\",\"clock\":"<<row.clock<<",\"token\":"<<row.token
             <<",\"observation\":";numbers(trace,observation);trace<<",\"observation_bits\":"<<quoted(observation_key(observation,menu))
             <<",\"menu\":";numbers(trace,menu);trace<<",\"operation\":"<<int(row.operation)
             <<",\"assistance\":"<<quoted(strategy=="initial_policy"?"none":"evaluator_only_"+strategy)
             <<",\"teacher\":"<<quoted(row.teacher)<<",\"active_before\":"<<quoted(active_before)
             <<",\"inspected_before\":"<<quoted(inspected_before)<<",\"active_after\":"<<quoted(learner.active_goal())
             <<",\"inspected_after\":"<<quoted(learner.inspected_goal())<<",\"model_before\":"<<version_before
             <<",\"model_after\":"<<learner.knowledge.version()<<",\"agenda_revision\":"<<row.agenda_revision
             <<",\"cost\":"<<row.cost<<",\"predictions\":"<<row.predictions<<",\"primitive_steps\":"<<row.primitive_steps
             <<",\"reply\":"<<int(row.reply)<<",\"status\":"<<quoted(row.status)<<",\"rejected\":"<<row.rejected
             <<",\"actions\":";actions(trace,row.actions);trace<<",\"callback\":"<<callback;
        if(operation==ControlOp::act&&!row.actions.empty()||operation==ControlOp::ask||operation==ControlOp::retrieve) {
            trace<<",\"observed_world\":";scene(trace,callback_world);
        }
        if(operation==ControlOp::finish)trace<<",\"terminal_correct\":"<<terminal_correct<<",\"terminal_satisfaction\":"<<terminal_satisfaction;
        trace<<"}\n";
        gt::require(learner.policy.weights==weights&&learner.updates()==0&&learner.credits().empty(),"preflight changed policy state");
        if(!semantic_learning)gt::require(learner.mission_evidence().empty()&&learner.knowledge.version()==initial_version,
            "frozen preflight changed semantic state");
    }
    result.decisions=learner.clock();result.motors=learner.acknowledged_motors();
    result.physical_satisfied=fixture.evaluator.satisfied("physical");
    result.success=result.urgent_posted&&roots_closed(learner);
    for(const auto& row:learner.submissions())result.success&=row.correct&&(!row.deadline||row.clock<=row.deadline);
    gt::require(learner.finish_mission(0),"mission finish rejected");
    gt::require(learner.policy.weights==weights&&learner.updates()==0&&learner.credits().size()==size_t(semantic_learning),
        "zero-gradient mission changed policy");
    result.model_end=learner.knowledge.version();result.final_credit_receipts=learner.credits().size();
    result.reward=result.success?1:-1;
    result.utility=std::pow(discount,double(result.decisions-1))*result.reward-result.discounted_cost;
    return result;
}
static void emit_run(std::ostream& out,const Run& r) {
    out<<"{\"id\":"<<quoted(r.id)<<",\"strategy\":"<<quoted(r.strategy)<<",\"seed\":"<<r.seed<<",\"arrival\":"<<r.arrival
       <<",\"success\":"<<r.success<<",\"semantic_learning\":"<<r.semantic_learning
       <<",\"model_start\":"<<r.model_start<<",\"model_end\":"<<r.model_end<<",\"final_credit_receipts\":"<<r.final_credit_receipts
       <<",\"decisions\":"<<r.decisions<<",\"motors\":"<<r.motors
       <<",\"predictions\":"<<r.predictions<<",\"primitive_steps\":"<<r.primitive_steps
       <<",\"arrival_during_execution\":"<<r.arrival_during_execution<<",\"arrival_flag_only\":"<<r.arrival_flag_only
       <<",\"motors_at_arrival\":"<<r.physical_motors_at_arrival<<",\"interrupted\":"<<r.interrupted<<",\"resumed\":"<<r.resumed
       <<",\"physical_satisfied\":"<<r.physical_satisfied<<",\"urgent_correct\":"<<r.urgent_correct
       <<",\"physical_close\":"<<r.physical_close<<",\"urgent_close\":"<<r.urgent_close
       <<",\"reward\":"<<r.reward<<",\"cost\":"<<r.cost<<",\"discounted_cost\":"<<r.discounted_cost<<",\"utility\":"<<r.utility<<'}';
}
int main(int argc,char** argv) {
    try {
        gt::require(argc==2,"usage: goal_switch_preflight OUTPUT_PREFIX");
        std::string prefix=argv[1];std::ofstream trace(prefix+"_trace.jsonl"),summary(prefix+"_summary.json"),alias(prefix+"_aliases.jsonl");
        gt::require(trace&&summary&&alias,"cannot open preflight output");
        for(auto* out:{&trace,&summary,&alias})*out<<std::boolalpha<<std::setprecision(17);
        auto fixture=gt::make_fixture();
        std::vector<Run> runs;
        for(unsigned seed:order_seeds) {
            // Each seed rebuilds identical raw lessons in a new semantic session;
            // no fixed-seed exploration RNG is inadvertently cloned across seeds.
            auto knowledge=fixture.make_knowledge(seed);
            for(const auto& lesson:fixture.lessons.language)
                gt::require(knowledge.answer(lesson.text,lesson.scene)==(lesson.answer?Answer::True:Answer::False),
                    "seed changed raw-lesson interpretation");
            for(size_t arrival:arrival_clocks)
            for(const auto& strategy:{"reference","reference_semantic","initial_policy","immediate_failure","delayed_failure"}) {
                runs.push_back(run(knowledge,seed,arrival,strategy,trace));trace.flush();
                const auto& r=runs.back();std::cout<<r.id<<" success="<<r.success<<" T="<<r.decisions<<" motors="<<r.motors<<" G="<<r.utility<<'\n'<<std::flush;
            }
        }
        size_t references=0,successes=0,semantic_references=0,semantic_successes=0,baseline_successes=0,arrival_signals=0,resumes=0,aliased=0,cross_context=0,conflicts=0;
        double worst_margin=1e30,worst_bound_margin=1e30,strongest_checked=-1e30;
        const double failure_bound=-std::pow(discount,double(decision_limit-1));
        for(const auto& r:runs)if(r.strategy=="immediate_failure"||r.strategy=="delayed_failure")strongest_checked=std::max(strongest_checked,r.utility);
        for(const auto& r:runs) {
            if(r.strategy=="reference"||r.strategy=="reference_semantic") {
                ++references;successes+=r.success;arrival_signals+=r.arrival_flag_only;resumes+=r.resumed;
                semantic_references+=r.semantic_learning;semantic_successes+=r.semantic_learning&&r.success;
                worst_margin=std::min(worst_margin,r.utility-strongest_checked);worst_bound_margin=std::min(worst_bound_margin,r.utility-failure_bound);}
            if(r.strategy=="initial_policy")baseline_successes+=r.success;
        }
        for(const auto& [key,group]:aliases) {
            aliased+=group.rows>1;cross_context+=group.contexts.size()>1;conflicts+=group.operations.size()>1;
            alias<<"{\"key\":"<<quoted(key)<<",\"rows\":"<<group.rows<<",\"operations\":";numbers(alias,group.operations);
            alias<<",\"contexts\":";strings(alias,group.contexts);alias<<",\"runs\":";strings(alias,group.runs);alias<<"}\n";
        }
        const bool feasible=successes==references&&arrival_signals==references&&resumes==references;
        const bool utility_ok=worst_margin>=required_margin&&worst_bound_margin>=required_margin;
        summary<<"{\n\"schema\":\"goal_switch_preflight_v2\",\n\"training_run\":false,\n\"policy_learning\":false,\n"
            <<"\"semantic_learning_reference_control\":true,\n\"warm_control_policy\":false,\n\"semantic_seed_matches_order_seed\":true,\n\"order_seeds\":";numbers(summary,order_seeds);
        summary<<",\n\"arrival_clocks\":";numbers(summary,arrival_clocks);
        summary<<",\n\"limits\":{\"roots\":2,\"derived\":0,\"decisions\":96,\"motors\":32,\"urgent_slack\":12},\n"
            <<"\"prices\":{\"execution\":[0.01,0.005,0.06,0.01,0],\"prediction_and_primitive\":0.00001,\"motor\":0.02,\"rebuild\":0.005,\"next_goal\":0.002,\"activate_goal\":0.01,\"preview\":[0.00001,0.0000001,0.0000001,0.005,0.005]},\n"
            <<"\"discount\":0.99,\n\"required_margin\":0.10,\n\"reference_runs\":"<<references<<",\n\"reference_successes\":"<<successes
            <<",\n\"semantic_reference_runs\":"<<semantic_references<<",\n\"semantic_reference_successes\":"<<semantic_successes
            <<",\n\"initial_policy_successes\":"<<baseline_successes<<",\n\"arrival_flag_only_runs\":"<<arrival_signals
            <<",\n\"resumed_runs\":"<<resumes<<",\n\"alias_classes\":"<<aliases.size()<<",\n\"repeated_alias_classes\":"<<aliased
            <<",\n\"cross_context_alias_classes\":"<<cross_context<<",\n\"reference_action_conflicts\":"<<conflicts
            <<",\n\"strongest_checked_failure_utility\":"<<strongest_checked<<",\n\"optimistic_zero_cost_failure_bound\":"<<failure_bound
            <<",\n\"worst_reference_margin_vs_checked\":"<<worst_margin<<",\n\"worst_reference_margin_vs_bound\":"<<worst_bound_margin
            <<",\n\"feasibility_gate\":"<<feasible<<",\n\"utility_gate\":"<<utility_ok<<",\n\"preflight_pass\":"<<(feasible&&utility_ok&&conflicts==0&&baseline_successes==0)
            <<",\n\"scope\":\"visited-support feature-only evaluator witness; initial zero-output policy deficit, not post-warm acquisition\",\n\"runs\":[\n";
        bool first=true;for(const auto& r:runs){if(!first)summary<<",\n";first=false;emit_run(summary,r);}summary<<"\n]}\n";
        std::cout<<"feasibility="<<feasible<<" utility="<<utility_ok<<" margin="<<worst_margin<<" bound_margin="<<worst_bound_margin<<'\n';
        return 0; // A failed scientific gate is a result, not a tool failure.
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
