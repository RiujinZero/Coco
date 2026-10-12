// A bounded, declared protocol-learning pilot. No forced output weights.
#include "goal_control_learning_harness.h"
#include "../tests/goal_control_fixture.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace gl=laminar::goal_learning;
namespace gt=laminar::goal_test;
using namespace laminar::relational;
static constexpr unsigned seed=1;
static constexpr size_t warm_missions=100,autonomous_missions=200;
static const std::vector<size_t> training_arrivals{3,5,7},evaluation_arrivals{4,6,8};
static const std::vector<unsigned> evaluation_orders{41,43,47,53};
static GoalControlLimits pilot_limits() {
    GoalControlLimits limits;limits.roots=2;limits.derived=0;limits.decisions=256;limits.motors=32;return limits;
}
static GoalControlPrices pilot_prices() {
    GoalControlPrices prices;prices.execution.operation.fill(0);prices.execution.prediction=prices.execution.motor=prices.execution.rebuild=0;
    prices.next_goal=prices.activate_goal=0;prices.preview={0,0,0,0,0};return prices;
}
static std::string configuration() {
    return R"CONFIG({
  "schema": "goal_control_learning_pilot_v1",
  "seed": 1,
  "hidden": 32,
  "receipt_capacity": 32768,
  "raw_semantic_teaching": "once before all missions; goal_control_fixture raw receipts",
  "limits": {"roots": 2, "derived": 0, "decisions": 256, "motors": 32},
  "prices": {"execution": [0,0,0,0,0], "prediction": 0, "motor": 0, "rebuild": 0, "next_goal": 0, "activate_goal": 0, "preview": [0,0,0,0,0]},
  "discount": 0.99,
  "warm_missions": 100,
  "warm_hint_schedule": "mission index0..49 probability1; index50..99 probability(99-index)/50, therefore .98 to0",
  "warm_epsilon": 0,
  "autonomous_missions_per_arm": 200,
  "autonomous_epsilon": 0.10,
  "terminal_updates": 128,
  "memory_policy": false,
  "training_arrivals": [3,5,7],
  "training_affordance_order_seed": "1000+mission_index; same index100..299 across arms",
  "assistance_rng_seed": "50000+mission_index; independent of task and learner RNG",
  "physical_deadline": 256,
  "urgent_deadline": "arrival+12",
  "evaluation_arrivals": [4,6,8],
  "evaluation_affordance_order_seeds": [41,43,47,53],
  "evaluation": "one frozen checkpoint clone sequentially across12 cases; epsilon0, no hints or semantic/policy updates",
  "agenda_order": "controller RNG continues normally; actual keys logged; no independent reseeding claim",
  "checkpoints": ["before", "warm100", "adaptive300", "frozen_policy300", "random_credit300"],
  "controls": ["frozen policy after warm: semantic learning on, finish_mission0", "random credit after warm: true semantic outcomes, independently randomized +/-1 terminal policy reward,128updates"],
  "control_rng_caveat": "random-credit draws use the controller RNG and can shift later epsilon draws; task schedules and initial warm state are matched",
  "task": "six necessary absent edges plus disconnected distractor affordances; same-world urgent grounded query",
  "early_close": "correct physical completion requires at least9 decisions, after latest declared arrival8; incorrect early termination is failure",
  "claim": "protocol-learning groundwork; no post-bootstrap acquisition claim if warm100 already solves evaluation; no long-route/derived-goal closure"
}
)CONFIG";
}
static PublicGoal convert(const gt::PublicRequestFixture& from) {
    PublicGoal goal;goal.id=from.id;goal.world_id=from.world_id;goal.text=from.text;goal.names=from.names;
    goal.visible=from.visible;goal.actions=from.actions;goal.effect=from.effect;goal.deadline=from.deadline;return goal;
}
class SixEdgeEnvironment final:public gl::Environment {
    gt::Fixture fixture_=gt::make_fixture();
    std::vector<gl::ScheduledRequest> requests_;
    std::vector<Action> allowed_;
    size_t motors_=0;
public:
    SixEdgeEnvironment(size_t arrival,unsigned order_seed) {
        gl::require(arrival>0&&arrival<9,"arrival can follow earliest correct completion");
        auto physical=convert(fixture_.request("physical"));physical.deadline=256;
        std::mt19937 order(order_seed);std::shuffle(physical.actions.begin(),physical.actions.end(),order);allowed_=physical.actions;
        auto urgent=convert(fixture_.request("arrival"));urgent.deadline=arrival+12;
        requests_.push_back({0,std::move(physical)});requests_.push_back({arrival,std::move(urgent)});
        const auto initial=fixture_.evaluator.read("shared").scene;
        for(int i=0;i<6;++i)gl::require(std::find(initial.facts.begin(),initial.facts.end(),Atom{"edge",{"n"+std::to_string(i),"n"+std::to_string(i+1)}})==initial.facts.end(),
            "physical fixture contains an initially present required edge");
    }
    const std::vector<gl::ScheduledRequest>& schedule()const override{return requests_;}
    StampedScene read(const std::string& world)const override{return fixture_.evaluator.read(world);}
    StampedScene execute(const std::string& world,const Action& action)override {
        gl::require(std::any_of(allowed_.begin(),allowed_.end(),[&](const auto& candidate) {
            return candidate.schema==action.schema&&candidate.arguments==action.arguments;
        }),"physics received nonpublic affordance");
        auto before=read(world);auto after=fixture_.evaluator.execute(world,action);++motors_;
        gl::require(after.scene.facts.size()<=before.scene.facts.size()+1,"motor added more than one edge");return after;
    }
    bool satisfaction(const PublicGoal& goal)const override {
        const bool satisfied=fixture_.evaluator.satisfied(goal.id);
        if(goal.id=="physical"&&satisfied) {
            gl::require(motors_>=6,"physical root completed without six actual motors");
            auto world=read("shared").scene;
            for(int i=0;i<6;++i)gl::require(std::find(world.facts.begin(),world.facts.end(),Atom{"edge",{"n"+std::to_string(i),"n"+std::to_string(i+1)}})!=world.facts.end(),
                "independent satisfaction omitted required edge");
        }
        return satisfied;
    }
    const char* family()const override{return "raw_six_edge_public_root_protocol";}
};
struct Outputs {
    std::ofstream trace,missions,parameters,evaluations,bootstrap;
    // mode=append reopens existing evidence after a checkpoint resume; the default truncates as before.
    explicit Outputs(const std::string& prefix,std::ios::openmode mode=std::ios::trunc):trace(prefix+"_trace.jsonl",std::ios::out|mode),missions(prefix+"_missions.jsonl",std::ios::out|mode),
        parameters(prefix+"_weights.jsonl",std::ios::out|mode),evaluations(prefix+"_evaluations.jsonl",std::ios::out|mode),bootstrap(prefix+"_bootstrap.jsonl",std::ios::out|mode) {
        for(auto* out:{&trace,&missions,&parameters,&evaluations,&bootstrap}) {
            gl::require(bool(*out),"cannot open pilot output");*out<<std::boolalpha<<std::setprecision(17);
        }
    }
};
static void export_bootstrap(const gt::RawLessons& lessons,std::ostream& out) {
    for(const auto& receipt:lessons.concepts) {
        out<<"{\"kind\":\"concept\",\"id\":"<<gl::quote(receipt.id)<<",\"name\":"<<gl::quote(receipt.name)
           <<",\"source\":"<<gl::quote(receipt.source)<<",\"version\":"<<receipt.version<<",\"active\":"<<receipt.active
           <<",\"generated\":"<<receipt.generated<<",\"answer\":"<<receipt.answer<<",\"args\":";gl::strings(out,receipt.args);
        out<<",\"scene\":";gl::scene(out,{receipt.scene,0});out<<"}\n";
    }
    for(const auto& receipt:lessons.language) {
        out<<"{\"kind\":\"language\",\"id\":"<<gl::quote(receipt.id)<<",\"text\":"<<gl::quote(receipt.text)
           <<",\"source\":"<<gl::quote(receipt.source)<<",\"version\":"<<receipt.version<<",\"active\":"<<receipt.active
           <<",\"answer\":"<<receipt.answer<<",\"names\":";gl::strings(out,receipt.names);
        out<<",\"scene\":";gl::scene(out,{receipt.scene,0});out<<"}\n";
    }
    for(const auto& receipt:lessons.transitions) {
        out<<"{\"kind\":\"transition\",\"id\":"<<gl::quote(receipt.id)<<",\"source\":"<<gl::quote(receipt.source)
           <<",\"version\":"<<receipt.version<<",\"active\":"<<receipt.active<<",\"action\":";
        gl::actions(out,{receipt.action});out<<",\"before\":";gl::scene(out,{receipt.before,0});out<<",\"after\":";gl::scene(out,{receipt.after,0});out<<"}\n";
    }out.flush();
}
static gl::MissionOptions options(const std::string& arm,size_t index,bool learning=true) {
    gl::MissionOptions out;out.id=arm+"_m"+std::to_string(index);out.arm=arm;out.index=index;
    out.limits=pilot_limits();out.prices=pilot_prices();out.terminal_updates=learning?128:0;
    out.assistance_seed=50000+unsigned(index);out.task_order_seed=1000+unsigned(index);return out;
}
static size_t evaluate(const GoalControlledRelationalSession& source,const std::string& label,Outputs& output) {
    auto frozen=source;frozen.set_learning(false);
    const auto weights=source.policy.weights;const auto updates=source.updates(),credits=source.credits().size(),version=source.knowledge.version();
    size_t successes=0,index=0;
    for(unsigned order:evaluation_orders)for(size_t arrival:evaluation_arrivals) {
        SixEdgeEnvironment environment(arrival,order);auto config=options("eval_"+label,index++,false);config.task_order_seed=order;
        auto result=gl::run_mission(frozen,environment,config,output.trace,output.missions);successes+=result.success;
        gl::require(result.cost==0,"declared zero-price evaluation charged work");
        gl::require(result.hints==0&&result.updates_before==result.updates_after,"evaluation assistance or updates");
    }
    gl::require(frozen.policy.weights==weights&&frozen.updates()==updates&&frozen.credits().size()==credits&&frozen.knowledge.version()==version,
        "evaluation clone mutated learned state");
    gl::require(source.policy.weights==weights&&source.updates()==updates&&source.credits().size()==credits&&source.knowledge.version()==version,
        "evaluation changed live learner");
    output.evaluations<<"{\"checkpoint\":"<<gl::quote(label)<<",\"successes\":"<<successes<<",\"cases\":12,\"updates\":"<<updates
        <<",\"credits\":"<<credits<<",\"model_version\":"<<version<<",\"hints\":0,\"epsilon\":0,\"frozen\":true}\n";output.evaluations.flush();
    std::cout<<"evaluation "<<label<<" "<<successes<<"/12 updates="<<updates<<'\n'<<std::flush;return successes;
}
static void train(GoalControlledRelationalSession& learner,const std::string& arm,size_t first,size_t count,Outputs& output) {
    size_t successes=0;
    for(size_t index=first;index<first+count;++index) {
        auto config=options(arm,index);config.epsilon=first==0?0.f:.10f;
        if(first==0)config.hint_probability=index<50?1.:double(99-index)/50.;
        if(arm=="frozen_policy")config.terminal_updates=0;
        config.random_credit=arm=="random_credit";
        SixEdgeEnvironment environment(training_arrivals[index%training_arrivals.size()],config.task_order_seed);
        auto result=gl::run_mission(learner,environment,config,output.trace,output.missions);successes+=result.success;
        gl::require(result.cost==0,"declared zero-price mission charged work");
        if(first!=0)gl::require(result.hints==0,"autonomous mission received protocol guidance");
        gl::weights(output.parameters,learner,config.id,arm);
        if((index-first+1)%10==0||index+1==first+count)
            std::cout<<arm<<" missions="<<(index-first+1)<<" successes="<<successes<<" updates="<<learner.updates()<<'\n'<<std::flush;
    }
}
int main(int argc,char** argv) {
    try {
        gl::require(argc==3,"usage: goal_control_learning_pilot --declare|--run OUTPUT_PREFIX");
        const std::string mode=argv[1],prefix=argv[2];
        if(mode=="--declare") {
            gl::require(!std::filesystem::exists(prefix+"_config.json"),"declared configuration already exists");
            std::ofstream file(prefix+"_config.json");gl::require(bool(file),"cannot declare pilot configuration");file<<configuration();return 0;
        }
        gl::require(mode=="--run","unknown pilot mode");std::ifstream declaration(prefix+"_config.json");
        std::string declared((std::istreambuf_iterator<char>(declaration)),std::istreambuf_iterator<char>());
        gl::require(declared==configuration(),"pilot configuration missing or changed since declaration");
        for(const auto& suffix:{"_trace.jsonl","_missions.jsonl","_weights.jsonl","_evaluations.jsonl","_bootstrap.jsonl","_summary.json"})
            gl::require(!std::filesystem::exists(prefix+suffix),"pilot output exists; refusing to overwrite evidence");
        Outputs output(prefix);auto fixture=gt::make_fixture();GoalControlledRelationalSession learner(seed,32,true);
        // Explicit one-time capacity configuration; no replacement after teaching.
        learner.knowledge=RelationalSession(seed,32768,1024);
        fixture.lessons.apply(learner.knowledge);export_bootstrap(fixture.lessons,output.bootstrap);
        gl::weights(output.parameters,learner,"initial","shared");
        const auto before=evaluate(learner,"before",output);
        train(learner,"warm",0,warm_missions,output);gl::weights(output.parameters,learner,"warm100","shared");
        const auto warm=evaluate(learner,"warm100",output);
        auto frozen=learner,random=learner;const auto warm_weights=learner.policy.weights;
        train(learner,"adaptive",warm_missions,autonomous_missions,output);
        const auto adaptive=evaluate(learner,"adaptive300",output);
        gl::weights(output.parameters,frozen,"branch_start","frozen_policy");
        train(frozen,"frozen_policy",warm_missions,autonomous_missions,output);
        gl::require(frozen.policy.weights==warm_weights,"frozen-policy arm changed warm weights");
        const auto control=evaluate(frozen,"frozen_policy300",output);
        gl::weights(output.parameters,random,"branch_start","random_credit");
        train(random,"random_credit",warm_missions,autonomous_missions,output);
        const auto random_success=evaluate(random,"random_credit300",output);
        std::ofstream summary(prefix+"_summary.json");summary<<std::boolalpha;
        summary<<"{\n\"schema\":\"goal_control_learning_pilot_result_v1\",\n\"complete\":true,\n\"seed\":1,\n"
            <<"\"warm_missions\":100,\n\"autonomous_missions_per_arm\":200,\n\"before\":"<<before<<",\n\"warm100\":"<<warm
            <<",\n\"adaptive300\":"<<adaptive<<",\n\"frozen_policy300\":"<<control<<",\n\"random_credit300\":"<<random_success
            <<",\n\"evaluation_cases_per_checkpoint\":12,\n\"warm_already_solves_all\":"<<(warm==12)
            <<",\n\"post_warm_improvement_observed\":"<<(adaptive>warm)
            <<",\n\"scope\":\"single-seed protocol-learning groundwork; no derived-goal or long-route closure\"\n}\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
