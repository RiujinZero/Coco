#include "laminar/goal_control.h"
#include "goal_control_fixture.h"
#include <iostream>

using namespace laminar::relational;
using laminar::goal_test::require;

static GoalControlStep choose(GoalControlledRelationalSession& agent,ControlOp operation) {
    auto result=agent.choose(0,false,ControlHint{operation,"protocol_test"});
    require(result.operation==operation,"protocol hint did not select legal operation");
    return result;
}
static PublicGoal request(const laminar::goal_test::PublicRequestFixture& row) {
    PublicGoal result;result.id=row.id;result.world_id=row.world_id;result.text=row.text;
    result.names=row.names;result.visible=row.visible;result.actions=row.actions;
    result.effect=row.effect;result.deadline=row.deadline;return result;
}
static GoalControlLimits limits(bool enabled,size_t work=100000) {
    GoalControlLimits result;result.roots=3;result.derived=32;result.decisions=128;
    result.preview.candidates=32;result.preview.plan_predictions=16;result.preview.work=work;
    result.agenda_search_exhaustion=enabled;return result;
}
static void inspect(GoalControlledRelationalSession& agent,const std::string& id) {
    for(size_t i=0;i<=agent.goals().size();++i) {
        choose(agent,ControlOp::next_goal);
        if(agent.inspected_goal()==id)return;
    }
    throw std::runtime_error("fixture goal not inspectable");
}
static std::string child(const GoalControlledRelationalSession& agent) {
    for(const auto& [id,e]:agent.goals())
        if(e.parent=="physical"&&e.request.text=="route n0 toward n3")return id;
    throw std::runtime_error("raw learned short-route proposal absent");
}
static void no_learning(const GoalControlledRelationalSession& agent,const std::vector<float>& weights) {
    require(agent.updates()==0&&agent.credits().empty()&&agent.policy.weights==weights&&agent.acknowledged_motors()==0,
            "observing search exhaustion taught policy, added terminal credit or executed motors");
}
static void exhausted_sequence(GoalControlledRelationalSession& agent,const std::string& id,bool enabled) {
    inspect(agent,id);
    const bool derived=!agent.goals().at(id).parent.empty();
    require(agent.observe()[15]==(enabled?0:derived),"uncached exhaustion/default feature incorrect");
    std::vector<GoalComputationReceipt> retained;
    for(size_t allowance:{4,8,16}) {
        auto before=agent.observe();const auto count=agent.mission_computations().size(),clock=agent.clock();
        const auto primitives=agent.knowledge.primitive_steps();
        for(int i=0;i<3;++i)require(agent.observe()==before,"read-only observation changed features");
        require(agent.clock()==clock&&agent.mission_computations().size()==count&&agent.knowledge.primitive_steps()==primitives,
                "observing exhaustion performed free computation");
        auto step=choose(agent,ControlOp::reason);
        const auto& receipt=agent.mission_computations().back();const auto& entry=agent.goals().at(id);
        require(step.observation==before&&step.cost>0&&step.predictions>=allowance&&receipt.plan.predictions==allowance&&step.status=="search_budget",
                "fixture did not pay and exhaust the expected planner allowance");
        require(receipt.token==step.token&&receipt.candidate==id&&receipt.plan_allowance==allowance&&
                receipt.limits.plan_predictions==allowance&&receipt.total_cost==step.cost&&entry.last_plan_allowance==allowance,
                "immutable receipt and inspected cache disagree about paid allowance");
        if(derived)require(receipt.preview&&receipt.preview->planner_allowance==allowance,
                           "preview omitted actual paid planner allowance");
        require(entry.next_budget==std::min(size_t(16),allowance*2),"ordinary allowance growth changed");
        require(agent.observe()[15]==(enabled?allowance==16:derived),"exhaustion reported before paying the global allowance");
        for(const auto& old:retained) {
            auto found=std::find_if(agent.mission_computations().begin(),agent.mission_computations().end(),
                                    [&](const auto& r){return r.token==old.token;});
            require(found!=agent.mission_computations().end()&&found->plan_allowance==old.plan_allowance&&
                    found->limits.plan_predictions==old.limits.plan_predictions&&found->inputs.world==old.inputs.world&&
                    found->plan.status==old.plan.status&&found->plan.predictions==old.plan.predictions&&found->total_cost==old.total_cost,
                    "later reasoning rewrote earlier computation evidence");
        }
        retained.push_back(receipt);
    }
}
static GoalControlledRelationalSession make_agent(const laminar::goal_test::Fixture& fixture,bool enabled,size_t work=100000) {
    GoalControlledRelationalSession result(31);result.knowledge=fixture.make_knowledge(31);
    require(result.begin_mission("exhaustion",limits(enabled,work))&&result.offer_goal(request(fixture.request("physical"))),
            "exhaustion fixture setup failed");
    return result;
}
static void contract_and_default() {
    auto fixture=laminar::goal_test::make_fixture();auto opted=make_agent(fixture,true),legacy=make_agent(fixture,false);
    auto original_weights=opted.policy.weights;
    exhausted_sequence(opted,"physical",true);exhausted_sequence(legacy,"physical",false);
    const auto selected=child(opted);require(selected==child(legacy),"feature opt-in changed proposal identity/order");
    exhausted_sequence(opted,selected,true);exhausted_sequence(legacy,selected,false);
    require(opted.mission_trace().size()==legacy.mission_trace().size(),"feature opt-in changed protocol accounting");
    for(size_t i=0;i<opted.mission_trace().size();++i) {
        const auto& a=opted.mission_trace()[i];const auto& b=legacy.mission_trace()[i];
        auto observed=a.observation;observed[15]=b.observation[15];
        require(observed==b.observation&&a.operation==b.operation&&a.token==b.token&&a.clock==b.clock&&a.cost==b.cost&&
                a.predictions==b.predictions&&a.primitive_steps==b.primitive_steps&&a.available==b.available&&a.status==b.status,
                "feature15 opt-in altered another observation or paid protocol result");
    }
    no_learning(opted,original_weights);no_learning(legacy,original_weights);
    auto refresh=request(fixture.request("arrival"));refresh.id="fresh_world";refresh.visible.revision=2;
    refresh.visible.scene.facts.push_back({"edge",{"n5","n6"}});
    require(opted.offer_goal(refresh),"new public world rejected");
    require(opted.observe()[15]==0&&!opted.goals().at(selected).cached&&opted.goals().at(selected).last_plan_allowance==0&&
            opted.goals().at(selected).next_budget==4,"world invalidation retained paid exhaustion");
    choose(opted,ControlOp::reason);
    require(opted.mission_computations().back().plan_allowance==4&&opted.observe()[15]==0,
            "world invalidation did not restart paid allowance");
    inspect(opted,"fresh_world");choose(opted,ControlOp::reason);
    require(opted.observe()[10]==0&&opted.observe()[15]==0&&opted.mission_computations().back().plan_allowance==0,
            "non-planning query invented search exhaustion or allowance");
    no_learning(opted,original_weights);
    auto model_agent=make_agent(fixture,true);exhausted_sequence(model_agent,"physical",true);
    auto raw=fixture.lessons.language.front();raw.id="new_raw_example";
    require(model_agent.knowledge.teach_language(raw),"raw model-change example rejected");
    require(model_agent.observe()[15]==0,"stale model cache exposed exhaustion");
    choose(model_agent,ControlOp::reason);
    require(model_agent.goals().at("physical").last_plan_allowance==4&&model_agent.observe()[15]==0,
            "model invalidation did not clear/restart paid allowance");
    std::cout<<"paid4/8/16 boundary, optional feature15 only, default membership, world/model invalidation and no learning passed\n";
}
static void work_limited_preview() {
    auto fixture=laminar::goal_test::make_fixture();auto agent=make_agent(fixture,true,100000);
    // Controller tests above verify exact forwarding of this result metadata.
    // Independently exercise a smaller work cap than prediction cap without
    // injecting a derived goal or replacing private controller state.
    goal_preview::Inputs inputs;inputs.parent_text="span n0 toward n6";inputs.candidate_text="route n0 toward n3";
    inputs.world=fixture.request("physical").visible.scene;inputs.actions=fixture.request("physical").actions;
    inputs.names.assign(fixture.request("physical").names.begin(),fixture.request("physical").names.end());
    goal_preview::Limits bounds;bounds.plan_predictions=16;bounds.work=10;
    auto limited=goal_preview::analyze(agent.knowledge,inputs,bounds);
    require(limited.planner_calls==1&&limited.planner_allowance==8&&limited.plan.predictions==8&&
            limited.plan.status=="search_budget"&&!limited.valid,"work-limited preview claimed the global allowance");
    bounds.work=1;auto no_plan=goal_preview::analyze(agent.knowledge,inputs,bounds);
    require(no_plan.planner_calls==0&&no_plan.planner_allowance==0,"missing planner call retained an allowance");
    std::cout<<"actual preview allowance respects pre-call work and is zero without planner invocation\n";
}
int main() {
    contract_and_default();work_limited_preview();
    std::cout<<"Optional agenda search-exhaustion feature contract passed; no learned scheduling claim\n";
}
