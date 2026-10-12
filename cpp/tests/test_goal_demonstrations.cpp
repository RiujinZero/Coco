#include "laminar/goal_control.h"
#include "goal_control_fixture.h"
#include <iostream>
#include <limits>

using namespace laminar::relational;
using laminar::goal_test::require;

static PublicGoal query() {
    auto fixture=laminar::goal_test::make_fixture();const auto& original=fixture.request("arrival");
    PublicGoal result;result.id="query";result.world_id=original.world_id;result.text=original.text;
    result.names=original.names;result.visible=original.visible;return result;
}
static GoalControlLimits limits() {
    GoalControlLimits result;result.roots=1;result.derived=0;result.decisions=32;return result;
}
static GoalControlStep demonstrate(GoalControlledRelationalSession& learner,ControlOp operation,const std::string& source) {
    return learner.choose(0,false,ControlHint{operation,source});
}
static void lesson(GoalControlledRelationalSession& learner,const std::string& id,const std::string& source,bool conflict=false) {
    require(learner.begin_mission(id,limits())&&learner.offer_goal(query()),"query demonstration setup failed");
    bool rejected=false;try{learner.fit_demonstrations(1,source);}catch(const std::logic_error&){rejected=true;}
    require(rejected,"active mission accepted demonstration fitting");
    rejected=false;try{learner.fit_balanced_demonstrations(1,source);}catch(const std::logic_error&){rejected=true;}
    require(rejected,"active mission accepted balanced fitting");
    demonstrate(learner,ControlOp::next_goal,source);
    auto token=demonstrate(learner,ControlOp::activate_goal,source);
    require(learner.activate(token.token),"demonstrated activation rejected");
    demonstrate(learner,ControlOp::reason,source);
    require(learner.observe()[7]==1,"independent fixture did not produce known False query");
    if(conflict)demonstrate(learner,ControlOp::reason,source);
    auto completed=demonstrate(learner,ControlOp::finish,source);
    require(completed.reply==Answer::False&&learner.close_goal(completed.token,true,false)&&learner.finish_mission(0),
            "demonstrated correct False completion failed");
}
static double error(const GoalControlledRelationalSession& learner,const std::string& source) {
    double result=0;
    for(const auto& credit:learner.credits())if(credit.active)for(const auto& row:credit.trace)if(row.teacher==source)
        for(int action:row.available) {
            const double residual=learner.policy.action_value(row.observation,action)-(action==int(row.operation)?1.:0.);
            result+=residual*residual;
        }
    return result;
}
static std::vector<int> probe_rng(GoalControlledRelationalSession learner) {
    require(learner.begin_mission("rng_probe",limits())&&learner.offer_goal(query()),"RNG probe failed");
    std::vector<int> result;
    for(int i=0;i<8;++i) {
        auto choice=learner.choose(1);result.push_back(int(choice.operation));
        if(choice.operation==ControlOp::activate_goal)break;
    }
    return result;
}
int main() {
    GoalControlledRelationalSession learner(19);
    laminar::goal_test::make_raw_lessons().apply(learner.knowledge);
    const auto initial_weights=learner.policy.weights;
    for(int i=0;i<4;++i)lesson(learner,"lesson"+std::to_string(i),"stopping_examples");
    require(learner.policy.weights==initial_weights&&learner.updates()==0&&learner.empirical_values().empty(),
            "guidance itself secretly fitted a policy");
    const auto before_error=error(learner,"stopping_examples");
    const auto version=learner.knowledge.version(),credits=learner.credits().size();
    const auto fitted=learner.fit_demonstrations(1024,"stopping_examples");
    require(fitted>0&&learner.updates()==fitted&&learner.policy.weights!=initial_weights,
            "explicit demonstration fitting did not update policy");
    require(learner.empirical_values().empty()&&learner.knowledge.version()==version&&learner.credits().size()==credits,
            "demonstration fitting changed rewards, values or semantic evidence");
    require(error(learner,"stopping_examples")<before_error*.05,"recorded demonstrated choices did not fit");
    for(const auto& credit:learner.credits())for(const auto& row:credit.trace)
        require(learner.policy.choose(row.observation,row.available)==int(row.operation),"demonstrated choice not learned");
    auto frozen=learner;frozen.set_learning(false);bool rejected=false;
    try{frozen.fit_demonstrations(1,"stopping_examples");}catch(const std::logic_error&){rejected=true;}
    require(rejected&&frozen.policy.weights==learner.policy.weights,"frozen learner fitted demonstrations");
    auto before=learner;
    require(learner.fit_demonstrations(std::numeric_limits<size_t>::max(),"absent_source")==0,
            "absent source produced training updates");
    rejected=false;try{learner.fit_demonstrations(std::numeric_limits<size_t>::max(),"stopping_examples");}
    catch(const std::runtime_error&){rejected=true;}
    require(rejected&&learner.policy.weights==before.policy.weights&&learner.updates()==before.updates()&&probe_rng(learner)==probe_rng(before),
            "overflow rejection or absent source changed state/RNG");
    const auto saved_weights=learner.policy.weights;
    require(learner.revise_mission("lesson0",1,{{"query",true,false}},false),"demonstration mission withdrawal failed");
    require(learner.policy.weights!=saved_weights&&learner.updates()==fitted,"withdrawal failed to replay associated demonstration fits");
    require(learner.revise_mission("lesson0",2,{{"query",true,false}},true),"demonstration mission restoration failed");
    require(learner.policy.weights==saved_weights&&learner.empirical_values().empty(),"demonstration restoration was not exact");
    const auto previous_updates=learner.updates();
    const auto balanced_version=learner.knowledge.version();
    require(learner.fit_balanced_demonstrations(257,"stopping_examples")==257&&learner.updates()==previous_updates+257,
            "balanced fitter did not apply exact step budget");
    require(learner.empirical_values().empty()&&learner.knowledge.version()==balanced_version&&learner.credits().size()==credits,
            "balanced fitting changed semantic or value state");
    const auto balanced_weights=learner.policy.weights;
    require(learner.revise_mission("lesson0",3,{{"query",true,false}},false),"balanced withdrawal failed");
    require(learner.revise_mission("lesson0",4,{{"query",true,false}},true)&&learner.policy.weights==balanced_weights,
            "balanced restoration was not exact");
    frozen=learner;frozen.set_learning(false);rejected=false;
    try{frozen.fit_balanced_demonstrations(1,"stopping_examples");}catch(const std::logic_error&){rejected=true;}
    require(rejected&&frozen.policy.weights==learner.policy.weights,"frozen balanced fit accepted");
    before=learner;
    require(learner.fit_balanced_demonstrations(std::numeric_limits<size_t>::max(),"absent_source")==0,"absent balanced source fitted");
    rejected=false;try{learner.fit_balanced_demonstrations(std::numeric_limits<size_t>::max(),"stopping_examples");}
    catch(const std::runtime_error&){rejected=true;}
    require(rejected&&learner.policy.weights==before.policy.weights&&learner.updates()==before.updates()&&probe_rng(learner)==probe_rng(before),
            "balanced overflow changed state");
    lesson(learner,"contradiction","conflicting_examples",true);before=learner;
    rejected=false;try{learner.fit_demonstrations(1,"conflicting_examples");}catch(const std::invalid_argument&){rejected=true;}
    require(rejected&&learner.policy.weights==before.policy.weights&&learner.updates()==before.updates()&&
            learner.knowledge.version()==before.knowledge.version()&&probe_rng(learner)==probe_rng(before),
            "conflicting labels were accepted or changed state/RNG");
    rejected=false;try{learner.fit_balanced_demonstrations(10,"conflicting_examples");}catch(const std::invalid_argument&){rejected=true;}
    require(rejected&&learner.policy.weights==before.policy.weights&&learner.updates()==before.updates()&&probe_rng(learner)==probe_rng(before),
            "balanced conflicting labels changed state");
    std::cout<<"Explicit source-tagged demonstration fitting, no value/semantic mutation, boundary/overflow/conflict rejection and exact replay passed\n";
}
