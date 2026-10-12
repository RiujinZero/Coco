#include "laminar/goal_control.h"
#include "goal_control_fixture.h"
#include <iostream>
#include <type_traits>

using namespace laminar::relational;
using laminar::goal_test::require;

template<class T> concept PublicHostSuspend=requires(T& value){value.suspend();};
template<class T> concept PublicHostCancel=requires(T& value){value.cancel();};
template<class T> concept PublicSingleGoalFinish=requires(T& value){value.finish(true,true);};
template<class T> concept PublicSingleGoalBegin=requires(T& value) {
    value.begin(std::string{},std::string{},std::set<std::string>{},StampedScene{},std::vector<Action>{},true);
};
static_assert(!std::is_convertible_v<GoalControlledRelationalSession*,BasicControlledRelationalSession<7>*>);
static_assert(!PublicHostSuspend<GoalControlledRelationalSession>&&!PublicHostCancel<GoalControlledRelationalSession>);
static_assert(!PublicSingleGoalBegin<GoalControlledRelationalSession>&&!PublicSingleGoalFinish<GoalControlledRelationalSession>);

// Protocol choices only: the temporary output preferences are restored before
// any terminal credit. These tests make no policy-acquisition claim.
static GoalControlStep prefer(GoalControlledRelationalSession& learner,ControlOp operation) {
    auto weights=learner.policy.weights;
    std::fill(learner.policy.weights.begin(),learner.policy.weights.end(),0);
    learner.policy.weights[learner.policy.output_bias()+int(operation)]=1;
    try {
        auto step=learner.choose();learner.policy.weights=std::move(weights);return step;
    } catch(...) {learner.policy.weights=std::move(weights);throw;}
}
static GoalControlStep force(GoalControlledRelationalSession& learner,ControlOp operation) {
    auto step=prefer(learner,operation);
    require(step.operation==operation,"forced protocol operation was not legal");return step;
}
static bool choice_rejected(GoalControlledRelationalSession& learner,ControlOp operation) {
    try{force(learner,operation);}catch(const std::exception&){return true;}return false;
}
static PublicGoal request(const laminar::goal_test::PublicRequestFixture& source) {
    PublicGoal goal;goal.id=source.id;goal.world_id=source.world_id;goal.text=source.text;
    goal.names=source.names;goal.visible=source.visible;goal.actions=source.actions;
    goal.effect=source.effect;goal.deadline=source.deadline;return goal;
}
static GoalControlLimits limits(size_t decisions=96,size_t motors=32) {
    GoalControlLimits result;result.derived=0;result.decisions=decisions;result.motors=motors;
    result.preview.work=100000;return result;
}
static GoalControlPrices prices() {
    GoalControlPrices result;
    result.preview={.00001,.00001,.00001,.00001,.00001};return result;
}
static GoalControlledRelationalSession learner(const laminar::goal_test::Fixture& fixture,unsigned seed=7) {
    GoalControlledRelationalSession result(seed);result.knowledge=fixture.make_knowledge(seed);return result;
}
static void inspect(GoalControlledRelationalSession& learner,const std::string& id) {
    for(size_t n=0;n<learner.goals().size()+2;++n) {
        force(learner,ControlOp::next_goal);
        if(learner.inspected_goal()==id)return;
    }
    throw std::runtime_error("public candidate was not reachable by next_goal");
}
static uint64_t activate(GoalControlledRelationalSession& learner,const std::string& id) {
    inspect(learner,id);auto selected=force(learner,ControlOp::activate_goal);
    require(learner.waiting()&&learner.activate(selected.token),"policy-issued goal activation rejected");
    require(!learner.waiting()&&learner.active_goal()==id&&!learner.in_agenda(),"activation did not enter goal execution");
    return selected.token;
}
static void reason_until_plan(GoalControlledRelationalSession& learner) {
    for(int n=0;n<16;++n) {
        force(learner,ControlOp::reason);
        if(!learner.plan().actions.empty())return;
        require(learner.plan().status=="search_budget","fixture physical goal became unavailable");
    }
    throw std::runtime_error("fixture planner exceeded its declared allowance");
}
static void close_query(GoalControlledRelationalSession& learner,bool expected) {
    force(learner,ControlOp::reason);auto finished=force(learner,ControlOp::finish);
    require(finished.reply==(expected?Answer::True:Answer::False),"query reply disagrees with evaluator truth");
    require(learner.close_goal(finished.token,true,expected),"query terminal evidence rejected");
}

static void continuing_mission() {
    auto fixture=laminar::goal_test::make_fixture();auto agent=learner(fixture);
    auto initial_weights=agent.policy.weights;auto initial_updates=agent.updates();
    require(agent.begin_mission("continuing",limits(),prices()),"mission begin failed");
    bool rejected=false;try{agent.set_learning(false);}catch(const std::logic_error&){rejected=true;}
    require(rejected&&agent.learning(),"mission-active learning-mode change bypassed boundary");
    rejected=false;try{agent.consolidate(1);}catch(const std::logic_error&){rejected=true;}
    require(rejected&&agent.updates()==initial_updates&&agent.policy.weights==initial_weights,
            "mission-active consolidation bypassed terminal-credit boundary");
    require(agent.offer_goal(request(fixture.request("physical"))),"physical request rejected");
    auto original_ticket=activate(agent,"physical");
    require(!agent.activate(original_ticket),"activation token replayed");
    require(agent.observe()[29]==0&&agent.observe()[30]==0,"execution phase or inspected-request flag incorrect");

    reason_until_plan(agent);auto first=force(agent,ControlOp::act);
    require(first.actions.size()==1&&agent.waiting(),"first physical callback not pending");
    const auto prefix=agent.trace();
    require(choice_rejected(agent,ControlOp::next_goal),"goal switched while motor callback was outstanding");
    // Arrivals are authenticated public input. They neither select a goal nor
    // cancel the already-owned physical callback.
    auto arriving=request(fixture.request("arrival"));arriving.deadline=agent.clock()+12;
    require(agent.offer_goal(arriving),"arrival during motor callback rejected");
    require(agent.active_goal()=="physical"&&agent.observe()[30]==1,"arrival selected a goal or hid its public flag");
    auto observed=fixture.evaluator.execute("shared",first.actions.front());
    require(!agent.actions(first.token+100,{observed.scene},{true},observed.revision),"foreign motor token accepted");
    require(!agent.actions(first.token,{observed.scene},{true},observed.revision,"generated"),"generated action observation accepted");
    auto before=agent.visible();
    require(!agent.actions(first.token,{observed.scene},{true},before.revision),"conflicting same-revision motor world accepted");
    require(agent.waiting()&&agent.acknowledged_motors()==0&&agent.visible().scene==before.scene&&agent.visible().revision==before.revision,
            "rejected motor callback mutated waiting state, budget or observation");
    require(agent.actions(first.token,{observed.scene},{true},observed.revision),"owned callback invalidated by public arrival");
    require(!agent.actions(first.token,{observed.scene},{true},observed.revision),"motor callback replayed");
    require(agent.observe()[30]==1&&agent.acknowledged_motors()==1,"acknowledgement lost arrival or motor accounting");
    require(agent.offer_goal(request(fixture.request("separate_query"))),"distinct-world request rejected");

    activate(agent,"arrival");
    require(agent.visible().scene==observed.scene&&agent.visible().revision==observed.revision,
            "same-world arrival used its stale offered snapshot");
    require(agent.knowledge.execution_receipts().size()==1,"interrupted physical segment missing");
    const auto segment=agent.knowledge.execution_receipts().begin()->second;
    require(!segment.goal_verified&&!segment.success,"interruption invented a final goal label");
    require(agent.policy.weights==initial_weights&&agent.updates()==initial_updates&&agent.credits().empty(),
            "suspension supplied child policy credit");
    force(agent,ControlOp::reason);auto arrival_finish=force(agent,ControlOp::finish);
    const bool arrival_truth=fixture.evaluator.satisfied("arrival");
    require(arrival_finish.reply==(arrival_truth?Answer::True:Answer::False),"shared-world query was not grounded");
    auto evidence_version=agent.knowledge.version();
    require(!agent.close_goal(arrival_finish.token,true,arrival_truth,"generated")&&agent.knowledge.version()==evidence_version,
            "generated terminal outcome changed evidence");
    require(agent.close_goal(arrival_finish.token,true,arrival_truth),"arrival close failed");
    require(agent.active_goal().empty()&&agent.in_agenda(),"closing child automatically resumed physical root");

    activate(agent,"separate_query");
    auto separate=fixture.evaluator.read("separate");
    require(agent.visible().scene==separate.scene&&agent.visible().revision==separate.revision,
            "shared observations crossed distinct world identities");
    close_query(agent,fixture.evaluator.satisfied("separate_query"));
    require(agent.policy.weights==initial_weights&&agent.updates()==initial_updates&&agent.credits().empty(),
            "closing public subepisodes trained before mission terminal");

    activate(agent,"physical");
    require(agent.trace().size()==prefix.size()&&agent.trace().front().token==prefix.front().token,
            "goal resume lost its pre-interruption decisions");
    require(agent.visible().scene==observed.scene&&agent.visible().revision==observed.revision&&!agent.observe()[5]&&!agent.observe()[9],
            "resume restored stale world or cached plan");
    require(!agent.actions(first.token,{observed.scene},{true},observed.revision+1),"resumption resurrected old callback");
    for(int attempt=0;attempt<8&&!fixture.evaluator.satisfied("physical");++attempt) {
        reason_until_plan(agent);auto action=force(agent,ControlOp::act);
        require(action.actions.size()==1,"reactive parent issued an unbounded batch");
        observed=fixture.evaluator.execute("shared",action.actions.front());
        require(agent.actions(action.token,{observed.scene},{true},observed.revision),"resumed physical callback rejected");
    }
    require(fixture.evaluator.satisfied("physical"),"reference physical goal did not finish");
    close_query(agent,true);
    require(agent.credits().empty()&&agent.updates()==initial_updates&&agent.policy.weights==initial_weights,
            "last goal closing supplied early policy credit");
    const auto mission_trace=agent.mission_trace();
    require(agent.finish_mission(64)&&!agent.mission_active(),"mission terminal credit rejected");
    require(agent.credits().size()==1&&agent.credits().back().mission&&agent.credits().back().reward==1&&
            agent.updates()==initial_updates+64&&agent.policy.weights!=initial_weights,"mission did not produce one terminal control receipt");
    const auto& credit=agent.credits().back();
    require(credit.trace.size()==mission_trace.size()&&credit.links.size()==agent.mission_evidence().size(),"mission omitted decisions or semantic evidence");
    uint64_t previous=0;
    for(size_t i=0;i<credit.trace.size();++i) {
        const auto& step=credit.trace[i];require(step.token>previous,"mission duplicated or reordered a decision");previous=step.token;
        require(step.token==mission_trace[i].token&&step.observation==mission_trace[i].observation&&step.cost==mission_trace[i].cost&&
                step.goal==mission_trace[i].goal,
                "mission credit differs from its experienced chronological trace");
    }
    require(!agent.finish_mission(64),"mission credited twice");

    std::vector<GoalOutcomeRevision> original_outcomes;
    for(const auto& row:agent.submissions())original_outcomes.push_back({row.id,row.correct,row.satisfaction});
    const auto accepted_weights=agent.policy.weights;const auto accepted_version=agent.knowledge.version();
    auto missing=original_outcomes;missing.pop_back();
    require(!agent.revise_mission("continuing",1,missing)&&!agent.revise_mission("continuing",0,original_outcomes)&&
            !agent.revise_mission("continuing",1,original_outcomes,true,"verified_environment",true)&&
            agent.policy.weights==accepted_weights&&agent.knowledge.version()==accepted_version,
            "incomplete, stale or generated mission correction changed state");
    require(!agent.revise_credit("continuing",1,false,false),"legacy single-goal correction accepted bundled mission");
    auto corrected=original_outcomes;corrected.front().correct=false;
    require(agent.revise_mission("continuing",1,corrected)&&agent.credits().front().reward==-1&&agent.policy.weights!=accepted_weights,
            "mission correctness correction did not replay terminal credit");
    require(agent.revise_mission("continuing",2,original_outcomes)&&agent.policy.weights==accepted_weights,
            "mission outcome restoration did not exactly replay weights");
    require(!agent.knowledge.execution_receipts().at(segment.id).goal_verified,
            "mission correction invented a label for an interrupted segment");
    require(agent.revise_mission("continuing",3,original_outcomes,false)&&agent.policy.weights==initial_weights&&
            !agent.credits().front().active,"mission withdrawal retained obsolete policy credit");
    for(const auto& link:agent.credits().front().links)require(!link.active,"mission withdrawal omitted linked semantic evidence");
    require(agent.revise_mission("continuing",4,original_outcomes,true)&&agent.policy.weights==accepted_weights,
            "mission withdrawal/restoration changed exact policy replay");

    // Public IDs may recur in a new mission; semantic ownership must use a
    // unique internal episode identity. Correctly reporting False is success.
    require(agent.begin_mission("false_query",limits(),prices()),"second mission begin failed");
    auto false_request=request(fixture.request("arrival"));false_request.id="physical";
    false_request.text="inspect n6 toward n0";false_request.visible=fixture.evaluator.read("shared");false_request.deadline=0;
    require(agent.offer_goal(false_request),"reused public goal ID rejected across missions");
    activate(agent,"physical");close_query(agent,false);
    require(agent.finish_mission(8)&&agent.credits().size()==2&&agent.credits().back().reward==1,
            "correct False query or repeated public ID broke mission credit");
    require(!agent.begin_mission("continuing",limits(),prices()),"finished mission identifier was reused");
    std::cout<<"continuing mission, arrival, world isolation, terminal-only credit and False query passed\n";
}

static void stale_selection_and_callback() {
    auto fixture=laminar::goal_test::make_fixture();auto agent=learner(fixture,11);
    require(agent.begin_mission("stale_selection",limits(),prices())&&agent.offer_goal(request(fixture.request("physical"))),"stale fixture setup failed");
    inspect(agent,"physical");auto selected=force(agent,ControlOp::activate_goal);
    require(!agent.activate(selected.token+1)&&agent.waiting(),"wrong activation token consumed valid ticket");
    require(agent.offer_goal(request(fixture.request("arrival")))&&!agent.activate(selected.token)&&!agent.waiting(),
            "new public request failed to stale activation");
    selected=force(agent,ControlOp::activate_goal);
    require(agent.revise_deadline("physical",1,90)&&!agent.activate(selected.token),"deadline revision failed to stale activation");
    require(!agent.revise_deadline("physical",1,80)&&!agent.revise_deadline("physical",2,80,"generated"),"stale/generated deadline accepted");
    selected=force(agent,ControlOp::activate_goal);
    require(agent.knowledge.revise_concept("goal_fixture_concept_0",1,false)&&!agent.activate(selected.token),
            "semantic revision failed to stale activation");
    agent.knowledge.rebuild();selected=force(agent,ControlOp::activate_goal);
    require(agent.activate(selected.token),"fresh activation did not recover");

    auto generated=request(fixture.request("separate_query"));generated.generated=true;
    auto count=agent.goals().size();require(!agent.offer_goal(generated)&&agent.goals().size()==count,"generated request accepted");
    auto read=force(agent,ControlOp::ask);auto original=agent.visible();
    auto update=request(fixture.request("separate_query"));update.id="new_shared";update.world_id="shared";
    update.visible={fixture.request("separate_query").visible.scene,5};
    require(agent.offer_goal(update),"new public observation during read rejected");
    require(agent.read(read.token,StampedScene{original.scene,2},"verified_environment")&&!agent.waiting()&&
            agent.visible().revision==original.revision&&agent.visible().scene==original.scene,
            "stale shared-world read overwrote known state instead of acknowledging failure");
    require(!agent.read(read.token,update.visible,"verified_environment"),"read callback replayed");
    read=force(agent,ControlOp::ask);
    require(agent.visible().revision==5&&agent.visible().scene==update.visible.scene,"next decision did not adopt newest public registry observation");
    require(!agent.read(read.token,update.visible,"verified_environment",true)&&agent.waiting(),"generated read consumed callback");
    require(agent.read(read.token,update.visible,"verified_environment")&&!agent.waiting()&&agent.visible().revision==5,
            "current same-world callback failed after stale attempt");
    force(agent,ControlOp::next_goal);selected=force(agent,ControlOp::activate_goal);
    auto active_before=agent.active_goal();auto trace_before=agent.trace();auto receipts_before=agent.knowledge.execution_receipts().size();
    auto inspected=agent.inspected_goal();auto revision=agent.goals().at(inspected).request.version;
    require(agent.revise_deadline(inspected,revision,90)&&!agent.activate(selected.token)&&agent.active_goal()==active_before&&
            agent.active()&&agent.trace().size()==trace_before.size()&&agent.knowledge.execution_receipts().size()==receipts_before,
            "stale activation partially changed the existing active goal");
    std::cout<<"stale activation, generated input and transactional callback rejection passed\n";
}

static void shared_budgets() {
    auto fixture=laminar::goal_test::make_fixture();auto agent=learner(fixture,13);
    require(agent.begin_mission("motor_budget",limits(96,1),prices()),"motor-budget mission begin failed");
    auto first=request(fixture.request("physical")),second=first;second.id="second_physical";
    require(agent.offer_goal(first)&&agent.offer_goal(second),"motor-budget requests rejected");
    activate(agent,first.id);reason_until_plan(agent);auto action=force(agent,ControlOp::act);
    auto seen=fixture.evaluator.execute("shared",action.actions.front());
    require(agent.actions(action.token,{seen.scene},{true},seen.revision)&&agent.acknowledged_motors()==1,"single motor acknowledgement failed");
    activate(agent,second.id);auto refused=prefer(agent,ControlOp::act);
    require(std::find(refused.available.begin(),refused.available.end(),int(ControlOp::act))==refused.available.end()&&
            refused.operation!=ControlOp::act&&agent.acknowledged_motors()==1,"switching reset the shared motor budget");

    auto bounded=learner(fixture,17);require(bounded.begin_mission("decision_budget",limits(5),prices())&&
        bounded.offer_goal(request(fixture.request("physical"))),"decision-budget setup failed");
    for(int n=0;n<5;++n)force(bounded,ControlOp::next_goal);
    require(bounded.clock()==5&&choice_rejected(bounded,ControlOp::next_goal),"agenda operations escaped shared decision budget");
    require(bounded.finish_mission(8)&&bounded.credits().back().reward==-1,"decision exhaustion did not close failed mission");

    auto late=learner(fixture,18);require(late.begin_mission("late_query",limits(),prices()),"late-query begin failed");
    auto deadline=request(fixture.request("arrival"));deadline.deadline=2;
    require(late.offer_goal(deadline),"deadline request rejected");activate(late,deadline.id);close_query(late,false);
    require(late.finish_mission(8)&&late.credits().back().reward==-1,"correct reply after deadline earned success");
    require(late.revise_mission("late_query",1,{{deadline.id,true,false}})&&late.credits().back().reward==-1,
            "outcome revision rewrote original timing or deadline");
    std::cout<<"shared motor and decision budgets passed\n";
}

static void frozen_clone_and_warm_import() {
    auto fixture=laminar::goal_test::make_fixture();auto source=learner(fixture,19);
    const auto weights=source.policy.weights;const auto version=source.knowledge.version();
    const auto languages=source.knowledge.language_receipts().size();
    auto probe=source;probe.set_learning(false);
    require(probe.begin_mission("frozen",limits(),prices()),"frozen mission begin failed");
    auto goal=request(fixture.request("arrival"));goal.deadline=0;
    require(probe.offer_goal(goal),"frozen request rejected");activate(probe,goal.id);close_query(probe,false);
    require(probe.finish_mission(16)&&probe.credits().empty()&&probe.updates()==0&&probe.policy.weights==weights&&
            probe.knowledge.version()==version&&probe.knowledge.language_receipts().size()==languages&&probe.mission_evidence().empty(),
            "frozen mission learned semantic or policy evidence");
    require(!source.mission_active()&&source.goals().empty()&&source.policy.weights==weights&&source.knowledge.version()==version,
            "clone mutated its trained source");

    ControlledRelationalSession old(23);GoalControlledRelationalSession imported(23);
    require(imported.import_warm_policy(old.policy),"five-output warm import failed");
    laminar::reasoning::Observation observation{};observation[3]=observation[12]=observation[31]=1;
    for(int action=0;action<5;++action)
        require(imported.policy.action_value(observation,action)==old.policy.action_value(observation,action),
                "warm import changed an old action prediction");
    require(imported.begin_mission("import_locked",limits(),prices())&&!imported.import_warm_policy(old.policy),
            "warm policy import remained available after mission start");
    std::cout<<"frozen clone isolation and exact old-action warm import passed\n";
}

static std::string propose_short_child(GoalControlledRelationalSession& agent) {
    inspect(agent,"physical");auto work=force(agent,ControlOp::reason);
    require(work.cost>0&&work.predictions>0,"agenda proposal computation was not charged");
    for(const auto& [id,entry]:agent.goals())if(entry.parent=="physical"&&entry.request.text=="route n0 toward n3")return id;
    throw std::runtime_error("generic public proposal inventory omitted the fixture short-route candidate");
}
static void check_computation_receipts(const GoalControlledRelationalSession& agent) {
    size_t count=0;
    for(const auto& step:agent.mission_trace())if(step.operation==ControlOp::reason&&step.observation[29]==1)++count;
    require(agent.mission_computations().size()==count,"paid agenda reasoning omitted a computation receipt");
    for(const auto& receipt:agent.mission_computations()) {
        auto step=std::find_if(agent.mission_trace().begin(),agent.mission_trace().end(),
            [&](const auto& row){return row.token==receipt.token;});
        require(step!=agent.mission_trace().end()&&step->inspected==receipt.candidate&&step->cost==receipt.total_cost&&
                step->primitive_steps==receipt.primitive_steps,"computation receipt differs from its experienced decision");
        double components=receipt.operation_cost+receipt.rebuild_cost+receipt.planning_cost+receipt.primitive_cost;
        if(receipt.preview)components+=receipt.preview->cost;
        if(receipt.proposals)components+=receipt.proposals->cost;
        require(std::isfinite(components)&&std::abs(components-receipt.total_cost)<1e-6,
                "paid computation components do not account for the full decision cost");
        require(!receipt.inputs.candidate_text.empty()&&receipt.inputs.observation_revision&&receipt.model_before==receipt.model_after,
                "computation receipt omitted exact request/world/model provenance");
    }
}
static void derived_goal_protocol() {
    auto fixture=laminar::goal_test::make_fixture();auto agent=learner(fixture,29);
    auto bounds=limits(128);bounds.derived=16;bounds.preview.candidates=32;
    require(agent.begin_mission("derived_mission",bounds,prices())&&agent.offer_goal(request(fixture.request("physical"))),
            "derived mission setup failed");
    auto weights=agent.policy.weights;auto child=propose_short_child(agent);
    const auto first_receipt=agent.mission_computations().front();
    require(first_receipt.proposals&&first_receipt.proposals->complete&&first_receipt.proposals->total_candidates,
            "paid proposal inventory missing from immutable computation receipt");
    activate(agent,child);
    require(agent.observe()[30]==0,"internal uninspected proposals impersonated an external request arrival");
    for(int attempt=0;attempt<4&&!laminar::goal_test::exact_route(fixture.evaluator.read("shared").scene,"n0","n3",3);++attempt) {
        reason_until_plan(agent);auto action=force(agent,ControlOp::act);
        auto observed=fixture.evaluator.execute("shared",action.actions.front());
        require(agent.actions(action.token,{observed.scene},{true},observed.revision),"derived child motor feedback rejected");
    }
    require(laminar::goal_test::exact_route(fixture.evaluator.read("shared").scene,"n0","n3",3),"derived child did not execute its learned goal");
    close_query(agent,true);
    require(agent.goals().at(child).complete&&!agent.goals().at("physical").complete&&agent.active_goal().empty()&&
            agent.credits().empty()&&agent.updates()==0&&agent.policy.weights==weights,
            "derived completion resumed parent or supplied separate child policy credit");
    activate(agent,"physical");
    require(agent.visible().scene==fixture.evaluator.read("shared").scene,"parent activation did not inherit child's acknowledged world");
    for(int attempt=0;attempt<4&&!fixture.evaluator.satisfied("physical");++attempt) {
        reason_until_plan(agent);auto action=force(agent,ControlOp::act);
        auto observed=fixture.evaluator.execute("shared",action.actions.front());
        require(agent.actions(action.token,{observed.scene},{true},observed.revision),"parent callback after child rejected");
    }
    require(fixture.evaluator.satisfied("physical"),"parent did not finish after derived child");close_query(agent,true);
    require(agent.finish_mission(32)&&agent.credits().size()==1&&agent.credits().back().links.size()==2&&agent.credits().back().reward==1,
            "parent and child did not share one terminal mission credit");
    check_computation_receipts(agent);
    const auto& retained=agent.computation_history("derived_mission");
    require(retained.size()==agent.mission_computations().size()&&retained.front().token==first_receipt.token&&
            retained.front().inputs.world==first_receipt.inputs.world&&
            retained.front().proposals->total_candidates==first_receipt.proposals->total_candidates&&
            retained.front().proposals->cost==first_receipt.proposals->cost,
            "cache invalidation or terminal credit erased experienced proposal evidence");
    std::vector<GoalOutcomeRevision> outcomes;
    for(const auto& row:agent.submissions())outcomes.push_back({row.id,row.correct,row.satisfaction});
    require(agent.revise_mission("derived_mission",1,outcomes)&&
            agent.computation_history("derived_mission").front().model_before==first_receipt.model_before,
            "mission correction rewrote experienced computation provenance");
    require(agent.begin_mission("after_receipts",bounds,prices())&&agent.mission_computations().empty()&&
            agent.computation_history("derived_mission").front().token==first_receipt.token,
            "new mission erased prior computation history");

    auto rejected_fixture=laminar::goal_test::make_fixture();auto rejected=learner(rejected_fixture,31);
    require(rejected.begin_mission("derived_activation",bounds,prices())&&rejected.offer_goal(request(rejected_fixture.request("physical"))),
            "derived activation setup failed");
    auto proposed=propose_short_child(rejected);activate(rejected,"physical");
    reason_until_plan(rejected);auto action=force(rejected,ControlOp::act);
    auto observed=rejected_fixture.evaluator.execute("shared",action.actions.front());
    require(rejected.actions(action.token,{observed.scene},{true},observed.revision),"parent motor before derived activation rejected");
    auto version=rejected.knowledge.version();auto trace=rejected.trace();
    force(rejected,ControlOp::next_goal);
    require(rejected.goals().at(proposed).invalid&&rejected.inspected_goal()!=proposed&&rejected.knowledge.version()>version&&
            rejected.knowledge.execution_receipts().size()==1&&rejected.mission_evidence().size()==1,
            "agenda inspection did not commit observed segment and invalidate old proposals");
    const auto& interrupted=rejected.knowledge.execution_receipts().begin()->second;
    require(!interrupted.goal_verified&&interrupted.observed.back()==observed.scene&&rejected.trace().size()==trace.size()&&
            rejected.policy.weights==rejected.initial_weights()&&rejected.updates()==0,
            "agenda segment boundary changed the trace or invented terminal credit");
    for(int n=0;n<3;++n) {
        force(rejected,ControlOp::next_goal);
        require(rejected.inspected_goal()!=proposed,"stale derived candidate remained inspectable");
    }
    auto receipts=rejected.knowledge.execution_receipts().size();auto current_goal=rejected.active_goal();
    require(!rejected.activate(action.token)&&rejected.active_goal()==current_goal&&rejected.knowledge.execution_receipts().size()==receipts,
            "non-activation token changed goal/evidence state");

    // A selected/started subgoal is an owned continuing goal. New evidence may
    // stale its plan, but must not turn its own observed segment into a reason
    // to discard the goal or make its preserved trace impossible to resume.
    auto resumed_fixture=laminar::goal_test::make_fixture();auto resumed=learner(resumed_fixture,37);
    require(resumed.begin_mission("derived_resume",bounds,prices())&&resumed.offer_goal(request(resumed_fixture.request("physical"))),
            "derived resume setup failed");
    auto continuing=propose_short_child(resumed);activate(resumed,continuing);reason_until_plan(resumed);
    auto issued=force(resumed,ControlOp::act);auto public_world=resumed_fixture.evaluator.execute("shared",issued.actions.front());
    require(resumed.actions(issued.token,{public_world.scene},{true},public_world.revision),"derived first motor rejected");
    auto prefix=resumed.trace();auto urgent=request(resumed_fixture.request("arrival"));urgent.deadline=resumed.clock()+16;
    require(resumed.offer_goal(urgent),"urgent request during derived goal rejected");
    activate(resumed,"arrival");
    require(resumed.goals().at(continuing).started&&!resumed.goals().at(continuing).invalid&&
            resumed.goals().at(continuing).motors==1,"segment flush discarded or reset an instantiated derived goal");
    close_query(resumed,resumed_fixture.evaluator.satisfied("arrival"));activate(resumed,continuing);
    require(resumed.active_goal()==continuing&&resumed.trace().size()==prefix.size()&&
            resumed.trace().front().token==prefix.front().token&&resumed.acknowledged_motors()==1,
            "same derived goal did not resume with its original decisions and motor use");
    for(int attempt=0;attempt<3&&!laminar::goal_test::exact_route(resumed_fixture.evaluator.read("shared").scene,"n0","n3",3);++attempt) {
        reason_until_plan(resumed);auto next=force(resumed,ControlOp::act);
        public_world=resumed_fixture.evaluator.execute("shared",next.actions.front());
        require(resumed.actions(next.token,{public_world.scene},{true},public_world.revision),"resumed derived callback rejected");
    }
    require(laminar::goal_test::exact_route(public_world.scene,"n0","n3",3),"resumed derived goal did not finish");close_query(resumed,true);
    require(resumed.credits().empty()&&resumed.updates()==0,"interrupted derived completion generated early credit");
    activate(resumed,"physical");
    for(int attempt=0;attempt<4&&!resumed_fixture.evaluator.satisfied("physical");++attempt) {
        reason_until_plan(resumed);auto next=force(resumed,ControlOp::act);
        public_world=resumed_fixture.evaluator.execute("shared",next.actions.front());
        require(resumed.actions(next.token,{public_world.scene},{true},public_world.revision),"parent callback after derived resume rejected");
    }
    require(resumed_fixture.evaluator.satisfied("physical"),"parent after derived resume did not finish");close_query(resumed,true);
    require(resumed.finish_mission(32)&&resumed.credits().size()==1&&resumed.credits().back().reward==1&&
            resumed.credits().back().links.size()==4,"interrupted derived trace/evidence did not join one mission credit");
    std::cout<<"paid generic proposal, derived execution/resume, bundled credit and stale proposal invalidation passed\n";
}

int main() {
    continuing_mission();stale_selection_and_callback();shared_budgets();frozen_clone_and_warm_import();derived_goal_protocol();
    std::cout<<"Goal control protocol: learner-issued switching/resume, public arrivals, ownership, shared budgets and one mission terminal credit passed\n";
}
