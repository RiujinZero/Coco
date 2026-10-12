#include "../apps/goal_long_environment.h"
#include <iostream>
namespace gl=laminar::goal_learning;
namespace lt=gl::long_task;
namespace gt=laminar::goal_test;
using namespace laminar::relational;
static void same_public_state(const RelationalSession& a,const RelationalSession& b) {
    gl::require(a.version()==b.version()&&a.dirty()==b.dirty()&&a.procedures()==b.procedures(),"rollback model state mismatch");
    auto equal=[](const auto& left,const auto& right) {
        gl::require(left.size()==right.size(),"rollback receipt count mismatch");
        for(const auto& [id,row]:left)gl::require(right.count(id)&&lt::same_receipt(row,right.at(id)),"rollback receipt content mismatch");
    };
    equal(a.concepts().receipts(),b.concepts().receipts());equal(a.language_receipts(),b.language_receipts());
    equal(a.transition_receipts(),b.transition_receipts());equal(a.execution_receipts(),b.execution_receipts());
    gl::require(a.concepts().models().size()==b.concepts().models().size(),"rollback model inventory mismatch");
    for(const auto& [name,model]:a.concepts().models()) {
        const auto& other=b.concepts().models().at(name);
        gl::require(model.status==other.status&&model.work==other.work&&model.candidates==other.candidates&&model.hypotheses.size()==other.hypotheses.size(),"rollback model search mismatch");
        for(size_t i=0;i<model.hypotheses.size();++i)
            gl::require(laminar::concepts::formula_key(laminar::concepts::expanded_clauses(model.hypotheses[i]))==
                        laminar::concepts::formula_key(laminar::concepts::expanded_clauses(other.hypotheses[i])),"rollback meaning mismatch");
    }
}
int main() {
    auto lessons=gt::make_raw_lessons();GoalControlledRelationalSession learner(19,32,true);
    learner.knowledge=RelationalSession(19,lessons.language.size()+2);lessons.apply(learner.knowledge);
    Scene world;std::vector<Action> offered{{"edge_motor",{"p0","p1"}},{"edge_motor",{"p1","p2"}},{"edge_motor",{"p2","p3"}}};
    auto issued=learner.knowledge.issue("route p0 toward p3",world,offered,true,1,8);
    gl::require(issued.actions.size()==1,"pending exploratory setup failed");
    auto before=learner.knowledge;const auto policy=learner.policy.weights;std::ostringstream evidence;
    bool rejected=false;try{lt::extend(learner,evidence,"capacity_failure");}catch(const std::runtime_error&){rejected=true;}
    gl::require(rejected&&evidence.str().empty()&&learner.policy.weights==policy&&learner.updates()==0&&learner.credits().empty(),"failed extension committed evidence or policy");
    same_public_state(learner.knowledge,before);
    gl::require(learner.knowledge.issue("route p0 toward p3",world,offered,true,1,8).status=="awaiting_feedback", "pending token lost on rollback");
    auto after=lt::execute_physics(world,issued.actions.front());
    gl::require(learner.knowledge.feedback(issued.token,{after},{true},false)&&before.feedback(issued.token,{after},{true},false),"restored pending feedback rejected");
    learner.knowledge.rebuild();before.rebuild();same_public_state(learner.knowledge,before);
    // Four further exploratory choices compare exact action and token streams;
    // the backup must restore RNG as well as public semantic records.
    for(int i=0;i<4;++i) {
        auto actual=learner.knowledge.issue("route p0 toward p3",world,offered,true,1,8);
        auto expected=before.issue("route p0 toward p3",world,offered,true,1,8);
        gl::require(actual.token==expected.token&&actual.version==expected.version&&actual.status==expected.status&&actual.actions.size()==expected.actions.size()&&
            std::equal(actual.actions.begin(),actual.actions.end(),expected.actions.begin(),same_action),"rollback RNG/token stream mismatch");
        gl::require(actual.actions.size()==1,"exploratory stream unavailable");auto observed=lt::execute_physics(world,actual.actions.front());
        // Observed interrupted segments require no new language slot, so the
        // deliberately small teaching capacity still permits all four probes.
        gl::require(learner.knowledge.feedback(actual.token,{observed},{true},std::nullopt)&&
                    before.feedback(expected.token,{observed},{true},std::nullopt),"rollback feedback stream mismatch");
        learner.knowledge.rebuild();before.rebuild();
    }
    same_public_state(learner.knowledge,before);
    GoalControlledRelationalSession success(7,32,true);lessons.apply(success.knowledge);std::ostringstream committed;
    const auto report=lt::extend(success,committed,"success_case");
    gl::require(report.concept_examples==12&&report.bootstrap_motors==3&&!committed.str().empty(),"successful extension failed");
    std::cout<<"Long extension failure atomicity: receipts, pending token, RNG and unchanged policy passed\n";
}
