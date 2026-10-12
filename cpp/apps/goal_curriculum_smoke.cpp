// Finite environment/demonstration contract, without neural fitting.
#include "goal_stopping_environment.h"
#include "goal_long_environment.h"
#include <fstream>
#include <iostream>
namespace gl=laminar::goal_learning;
namespace stop=gl::stopping;
namespace lt=gl::long_task;
namespace gt=laminar::goal_test;
using namespace laminar::relational;
struct Smoke {bool success=false,interrupted=false,resumed=false;size_t decisions=0,motors=0,reads=0,finishes=0;};
static Smoke run(GoalControlledRelationalSession learner,gl::Environment& environment,const std::string& id,
                 GoalControlLimits limits,std::ostream& trace) {
    const auto weights=learner.policy.weights;const auto updates=learner.updates();
    gl::require(learner.begin_mission(id,limits,stop::prices()),"curriculum smoke begin rejected");
    size_t posted=0;Smoke result;std::set<std::string> interrupted;
    auto post=[&]() {while(posted<environment.schedule().size()&&environment.schedule()[posted].clock==learner.clock()) {
        auto request=environment.schedule()[posted++].request;request.visible=environment.read(request.world_id);
        gl::require(learner.offer_goal(request),"curriculum public request rejected");
        trace<<"{\"mission\":"<<gl::quote(id)<<",\"event\":\"request\",\"clock\":"<<learner.clock()
             <<",\"id\":"<<gl::quote(request.id)<<",\"text\":"<<gl::quote(request.text)<<",\"world\":";gl::scene(trace,request.visible);
        trace<<",\"actions\":";gl::actions(trace,request.actions);trace<<"}\n";
    }};
    post();
    while(learner.clock()<limits.decisions&&!gl::closed(learner)) {
        post();const auto o=learner.observe();const auto menu=gl::public_menu(learner,limits);
        const int demonstrated=gl::curriculum_demonstration(o,menu);
        gl::require(stop::demonstration(o,menu)==demonstrated&&lt::demonstration(o,menu)==demonstrated,"family-dependent example label");
        const auto active=learner.active_goal();auto step=learner.choose(0,false,ControlHint{ControlOp(demonstrated),gl::curriculum_demonstration_source});
        gl::require(step.operation==ControlOp(demonstrated),"explicit example hint not selected");
        bool satisfaction=false,correct=false;std::vector<StampedScene> observed;
        if(step.operation==ControlOp::activate_goal) {
            gl::require(learner.activate(step.token),"curriculum activation rejected");
            if(!active.empty()&&active!=learner.active_goal()){interrupted.insert(active);result.interrupted=true;}
            result.resumed|=interrupted.count(learner.active_goal())!=0;
        } else if(step.operation==ControlOp::act&&!step.actions.empty()) {
            const auto& request=learner.goals().at(active).request;std::vector<Scene> observations;
            for(const auto& action:step.actions){observed.push_back(environment.execute(request.world_id,action));observations.push_back(observed.back().scene);}
            gl::require(learner.actions(step.token,observations,std::vector<bool>(observations.size(),true),observed.back().revision),"curriculum physical callback rejected");
        } else if(step.operation==ControlOp::retrieve||step.operation==ControlOp::ask) {
            const auto& request=learner.goals().at(active).request;observed.push_back(environment.read(request.world_id));++result.reads;
            gl::require(learner.read(step.token,observed.back(),step.operation==ControlOp::retrieve?"trusted_memory":"verified_environment"),"curriculum read rejected");
        } else if(step.operation==ControlOp::finish) {
            const auto& request=learner.goals().at(active).request;satisfaction=environment.satisfaction(request);
            correct=step.reply==(satisfaction?Answer::True:Answer::False)&&(!request.effect||satisfaction);++result.finishes;
            gl::require(learner.close_goal(step.token,correct,satisfaction),"curriculum finish rejected");
        }
        trace<<"{\"mission\":"<<gl::quote(id)<<",\"event\":\"decision\",\"clock\":"<<learner.clock()<<",\"observation\":";gl::numbers(trace,o);
        trace<<",\"menu\":";gl::numbers(trace,menu);trace<<",\"operation\":"<<demonstrated<<",\"teacher\":"<<gl::quote(step.teacher)
             <<",\"active\":"<<gl::quote(active)<<",\"reply\":"<<int(step.reply)<<",\"satisfaction\":"<<satisfaction<<",\"correct\":"<<correct
             <<",\"observed\":[";for(size_t i=0;i<observed.size();++i){if(i)trace<<',';gl::scene(trace,observed[i]);}trace<<"]}\n";
        gl::require(learner.policy.weights==weights&&learner.updates()==updates,"smoke example fitted a policy");
    }
    result.success=gl::closed(learner)&&posted==environment.schedule().size();
    for(const auto& submission:learner.submissions())if(submission.parent.empty())result.success&=submission.correct&&(!submission.deadline||submission.clock<=submission.deadline);
    result.decisions=learner.clock();result.motors=learner.acknowledged_motors();
    gl::require(learner.finish_mission(0),"curriculum terminal evidence rejected");
    gl::require(learner.policy.weights==weights&&learner.updates()==updates,"curriculum smoke changed weights");return result;
}
static void result(std::ostream& out,const std::string& id,const Smoke& r) {
    out<<"{\"id\":"<<gl::quote(id)<<",\"success\":"<<r.success<<",\"decisions\":"<<r.decisions<<",\"motors\":"<<r.motors
       <<",\"read_operations\":"<<r.reads<<",\"finish_operations\":"<<r.finishes<<",\"interrupted\":"<<r.interrupted<<",\"resumed\":"<<r.resumed<<'}';
}
int main(int argc,char** argv) {
    gl::require(argc==2,"usage: goal_curriculum_smoke OUTPUT_PREFIX");const std::string prefix=argv[1];
    std::ofstream out(prefix+"_summary.json"),trace(prefix+"_trace.jsonl"),evidence(prefix+"_extension.jsonl");
    for(auto* stream:{&out,&trace,&evidence}){gl::require(bool(*stream),"curriculum output unavailable");*stream<<std::boolalpha<<std::setprecision(17);}
    GoalControlledRelationalSession base(7,32,true);gt::make_raw_lessons().apply(base.knowledge);
    lt::extend(base,evidence,"curriculum_smoke_extension");
    out<<"{\"schema\":\"goal_curriculum_smoke_v1\",\"policy_training\":false,\"teacher\":"<<gl::quote(gl::curriculum_demonstration_source)
       <<",\"training_examples\":[";bool first=true;
    for(const auto& spec:stop::training_cases()){stop::Environment env(spec);if(!first)out<<',';first=false;result(out,spec.id,run(base,env,spec.id,stop::limits(),trace));out.flush();}
    out<<"],\"heldout_environment_reference_checks\":[";first=true;
    for(const auto& spec:stop::heldout_cases()){stop::Environment env(spec);if(!first)out<<',';first=false;result(out,spec.id,run(base,env,spec.id,stop::limits(),trace));out.flush();}
    // Explicitly test a positive partial query that stays incomplete forever:
    // the learned positive entailment must permit finish without forced reads.
    gl::Observation partial{};partial[0]=0;partial[5]=0;partial[31]=1;
    const std::vector<int> execution_menu{0,1,2,3,4,5};gl::require(gl::curriculum_demonstration(partial,execution_menu)==1,"partial query did not reason first");
    partial[5]=partial[6]=1;gl::require(gl::curriculum_demonstration(partial,execution_menu)==4,"known positive partial query did not finish");
    partial[6]=0;partial[8]=1;gl::require(gl::curriculum_demonstration(partial,execution_menu)==2,"unresolved partial query did not retrieve");
    out<<"],\"partial_feature_contract\":true,\"long_examples\":[";
    first=true;
    for(unsigned order:{0,1}) {lt::LongEnvironment env(8,order,"smoke"+std::to_string(order));if(!first)out<<',';first=false;
        result(out,"long_"+std::to_string(order),run(base,env,"long_"+std::to_string(order),lt::limits(true),trace));out.flush();}
    out<<"],\"family_label_consistent\":true,\"policy_updates\":0,\"weights_unchanged\":true}\n";
    std::cout<<"Curriculum environment smoke recorded: "<<prefix<<"_summary.json\n";
}
