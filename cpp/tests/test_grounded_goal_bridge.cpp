#include "laminar/grounded_relational_bridge.h"
#include "laminar/goal_control.h"
#include <iostream>

using namespace laminar::reasoning;
using namespace laminar::relational;

static void require(bool value,const std::string& message) {
    if(!value)throw std::runtime_error(message);
}
static void feed(GroundedSession& session,const std::string& text,bool generated=false) {
    for(uint8_t byte:text)session.observe(byte,generated);
}
static std::shared_ptr<GroundedSession> reader(int channel) {
    auto model=std::make_shared<LanguageReasoner>(channel);
    model->environment_dialogue=model->grounded_requests=model->grounded_events=
        model->uncertain_events=model->partial_observation=true;
    model->memory.unique_enabled=true;model->world_budget=4096;
    // Declared existing atomic-question bootstrap. No program or successful
    // interpretation for the unfamiliar "logged beside" report is supplied.
    model->forms={{"who has the $0 now ",0}};
    model->memory.procedures={{0,{16+channel,9}}};
    for(int i=0;i<8;++i) {
        const auto person="actor"+std::to_string(i),object="item"+std::to_string(i);
        EventGrounding::World before{{object,channel,"old"},{"spare",channel,person}};
        require(model->events.teach_transition(person+" has the "+object+".",before,
            EventGrounding::update(before,channel,2,object,person),i),"familiar observation bootstrap rejected");
    }
    auto session=std::make_shared<GroundedSession>(model);session->teachers={"teacher"};
    feed(*session,"\n<|endofdialogue|>\nCONTEXT: ada has the key.\nCONTEXT: ben has the coin.\nCONTEXT: cy has the pen.\n");
    return session;
}
static GroundedRelationalEvidence capture(const GroundedSession& session) {
    auto captured=GroundedRelationalBridge::capture(session);
    require(captured.value.has_value(),"reader capture rejected: "+captured.status);
    require(captured.value->alternatives()==session.reader->request_snapshot(),"capture lost reader alternatives");
    return *captured.value;
}
static StampedScene project(const GroundedSession& session,const GroundedRelationalEvidence& evidence,uint64_t revision) {
    auto projected=GroundedRelationalBridge::project_current(session,evidence,revision);
    require(projected.value.has_value(),"current projection rejected: "+projected.status);
    require(!projected.value->scene.complete,"partial reader output was promoted to a complete teaching world");
    return *projected.value;
}
static void teach_query(GoalControlledRelationalSession& learner,int channel) {
    const auto predicate="reader_r"+std::to_string(channel);
    Scene no{{{predicate,{"key","ben"}},{predicate,{"spare","cy"}}},true};
    auto yes=no;yes.facts.push_back({predicate,{"key","cy"}});
    // Separate structured query bootstrap from raw Boolean outcomes. The
    // reader's unfamiliar report is never used as a labeled teaching scene.
    for(bool answer:{false,true}) {
        laminar::concepts::Receipt concept_row;
        concept_row.id=answer?"query_concept_yes":"query_concept_no";concept_row.name="linked";
        concept_row.args={"key","cy"};concept_row.scene=answer?yes:no;concept_row.answer=answer;
        require(learner.knowledge.teach_concept(concept_row),"query concept bootstrap rejected");
        LanguageReceipt language;language.id=answer?"query_language_yes":"query_language_no";
        language.text="join key with cy";language.names={"key","cy","ben","spare"};
        language.scene=answer?yes:no;language.answer=answer;
        require(learner.knowledge.teach_language(language),"query language bootstrap rejected");
    }
    learner.knowledge.rebuild();
}
static PublicGoal goal(const std::string& id,const std::string& world,const StampedScene& visible) {
    PublicGoal request;request.id=id;request.world_id=world;request.text="join key with cy";
    request.names={"key","cy"};request.visible=visible;return request;
}
// Force protocol choices and restore weights immediately. This is a wiring
// test, not a learned scheduler or an acquisition experiment.
static GoalControlStep force(GoalControlledRelationalSession& learner,ControlOp operation) {
    auto saved=learner.policy.weights;
    std::fill(learner.policy.weights.begin(),learner.policy.weights.end(),0);
    learner.policy.weights[learner.policy.output_bias()+int(operation)]=1;
    try {
        auto step=learner.choose();learner.policy.weights=saved;
        require(step.operation==operation,"requested protocol operation was not legal");return step;
    } catch(...) {learner.policy.weights=std::move(saved);throw;}
}
static void activate(GoalControlledRelationalSession& learner,const std::string& id) {
    for(size_t attempt=0;attempt<learner.goals().size()+1;++attempt) {
        force(learner,ControlOp::next_goal);
        if(learner.inspected_goal()==id) {
            auto selected=force(learner,ControlOp::activate_goal);
            require(learner.activate(selected.token),"reader-backed public goal activation rejected");return;
        }
    }
    throw std::runtime_error("public reader goal was not reachable in agenda");
}
static Answer reason(GoalControlledRelationalSession& learner) {
    force(learner,ControlOp::reason);
    const auto state=learner.observe();
    return state[6]?Answer::True:state[7]?Answer::False:Answer::Unknown;
}
static void close(GoalControlledRelationalSession& learner,Answer answer,bool correct,bool satisfaction) {
    auto finished=force(learner,ControlOp::finish);
    require(finished.reply==answer&&learner.close_goal(finished.token,correct,satisfaction),"reader-backed goal close failed");
}

static void integrated_worlds(int channel) {
    auto first=reader(channel),second=reader(channel);
    feed(*first,"ENVIRONMENT: The key was logged beside cy.\n");
    feed(*second,"ENVIRONMENT: ben has the key.\n");
    auto ambiguous=capture(*first),other=capture(*second);
    require(ambiguous.alternatives().worlds.size()>1,"unfamiliar report was supplied an interpretation");
    GroundedRelationalBridge first_dependencies,second_dependencies;
    require(first_dependencies.bind_dependency("first_view",ambiguous)&&
        second_dependencies.bind_dependency("second_view",other),"reader dependencies rejected");
    auto first_view=project(*first,ambiguous,1),second_view=project(*second,other,1);

    GoalControlledRelationalSession learner(37+channel);teach_query(learner,channel);
    const auto initial_weights=learner.policy.weights;
    const auto language_count=learner.knowledge.language_receipts().size();
    const auto semantic_version=learner.knowledge.version();
    GoalControlLimits limits;limits.roots=2;limits.derived=0;limits.decisions=64;
    require(learner.begin_mission("reader_mission",limits),"reader mission begin rejected");
    auto first_goal=goal("first","reader_world_a",first_view);
    auto generated=first_goal;generated.generated=true;
    require(!learner.offer_goal(generated)&&learner.goals().empty(),"generated reader request became public input");
    require(learner.offer_goal(first_goal)&&learner.offer_goal(goal("second","reader_world_b",second_view)),"reader projections rejected as PublicGoal inputs");
    activate(learner,"first");
    require(learner.visible().scene==first_view.scene&&reason(learner)==Answer::Unknown,"ambiguous reader output invented a goal answer");
    auto pending=force(learner,ControlOp::ask);
    require(!learner.read(pending.token,first_view,"verified_environment",true)&&learner.waiting(),"generated projected read consumed callback");

    // The unfamiliar report is learned from a raw final answer, without a
    // report operator label, a supplied parse, or a goal-selection hint.
    auto anchor=first->feedback_anchor(1);require(anchor.has_value(),"novel report has no feedback anchor");
    require(first->teach(*anchor,0,"Who has the key now?","cy","teacher"),"raw final-answer teaching rejected");
    auto learned=capture(*first);
    require(!GroundedRelationalBridge::project_current(*first,ambiguous,2).value&&
        first_dependencies.invalidated_dependencies(*first)==std::vector<std::string>{"first_view"},"reader teaching retained stale projection authority");
    require(second_dependencies.invalidated_dependencies(*second).empty(),"first reader teaching invalidated distinct-world evidence");
    auto learned_view=project(*first,learned,2);
    require(!learner.read(pending.token+1,learned_view,"verified_environment")&&learner.waiting(),"foreign reader callback token accepted");
    require(learner.read(pending.token,learned_view,"verified_environment")&&reason(learner)==Answer::True,"seven-action controller could not use learned raw report");
    require(first_dependencies.forget_dependency("first_view")&&first_dependencies.bind_dependency("first_view",learned),"dependency refresh rejected");

    feed(*first,"\n<|endofdialogue|>\nENVIRONMENT: ben has the key.\nUSER: withdraw that report\nASSISTANT:",true);
    require(GroundedRelationalBridge::current(*first,learned.stamp()),"generated bytes changed reader evidence");
    require(!first->teach(*first->feedback_anchor(1),9,"Who has the key now?","ben","teacher",GroundedSession::ADD_FEEDBACK,0,true),"generated final answer taught reader");
    require(GroundedRelationalBridge::current(*first,learned.stamp()),"generated feedback changed evidence stamp");

    auto correction_copy=first->clone();
    require(correction_copy->teach(*correction_copy->feedback_anchor(1),0,"Who has the key now?","unknown","teacher",GroundedSession::WITHDRAW_FEEDBACK),"reader feedback withdrawal rejected");
    auto withdrawn=capture(*correction_copy);
    require(withdrawn.stamp()!=learned.stamp()&&!GroundedRelationalBridge::project_current(*correction_copy,learned,3).value,
        "reader withdrawal retained projection authority");
    require(first_dependencies.invalidated_dependencies(*correction_copy)==std::vector<std::string>{"first_view"}&&
        GroundedRelationalBridge::current(*first,learned.stamp()),"withdrawal failed dependency invalidation or leaked across clone");
    require(correction_copy->teach(*correction_copy->feedback_anchor(1),0,"Who has the key now?","cy","teacher",GroundedSession::REVISE_FEEDBACK,1),"reader feedback restoration rejected");
    auto restored=capture(*correction_copy);
    require(restored.alternatives()==learned.alternatives()&&restored.stamp()!=learned.stamp()&&restored.stamp()!=withdrawn.stamp(),
        "restoration lost worlds or reused old evidence identity");
    pending=force(learner,ControlOp::ask);
    auto restored_view=project(*correction_copy,restored,3);
    require(learner.read(pending.token,restored_view,"verified_environment")&&reason(learner)==Answer::True,"restored current reader projection failed goal read");
    close(learner,Answer::True,true,true);
    require(learner.credits().empty()&&learner.updates()==0&&learner.policy.weights==initial_weights,
        "reader-backed child close supplied early control reward");

    activate(learner,"second");
    require(learner.visible().scene==second_view.scene&&learner.visible().revision==1&&reason(learner)==Answer::Unknown,
        "first reader's facts or revision crossed world identity");
    require(RequestGrounding::truth(other.alternatives(),{"key","cy",channel,true})==RequestGrounding::FALSE_VALUE,
        "fixture did not retain scoped reader negative");
    // Canonical partial Scene cannot express that negative. A later actual raw
    // report supplies a positive observation through the same pending-read API.
    pending=force(learner,ControlOp::ask);feed(*second,"ENVIRONMENT: cy has the key.\n");
    auto changed=capture(*second);auto changed_view=project(*second,changed,2);
    require(second_dependencies.invalidated_dependencies(*second)==std::vector<std::string>{"second_view"},"new observation retained old dependency");
    require(learner.read(pending.token,changed_view,"verified_environment")&&reason(learner)==Answer::True,"second reader's current report failed goal read");
    close(learner,Answer::True,true,true);
    require(learner.knowledge.language_receipts().size()==language_count&&learner.knowledge.version()==semantic_version&&learner.mission_evidence().empty(),
        "partial reader worlds became complete semantic teaching evidence");
    require(learner.finish_mission(16)&&learner.credits().size()==1&&learner.credits().back().mission&&
        learner.credits().back().reward==1&&learner.updates()==16,"reader-backed roots did not share one terminal mission credit");

    // Continue the same controller: absence in a partial reader view still does
    // not become False merely because an earlier mission succeeded.
    feed(*first,"ENVIRONMENT: ben has the key.\n");auto negative=capture(*first);
    require(learner.begin_mission("reader_followup",limits)&&learner.offer_goal(goal("first","reader_world_a",project(*first,negative,4))),
        "continuing reader mission rejected");
    activate(learner,"first");require(reason(learner)==Answer::Unknown,"continued reader mission invented closed-world False");
    close(learner,Answer::Unknown,false,false);
    require(learner.finish_mission(8)&&learner.credits().size()==2&&learner.credits().back().reward==-1&&learner.updates()==24,
        "incomplete follow-up failed to retain a separate failed mission receipt");
    std::cout<<"reader channel "<<channel<<": raw final-answer learning, two worlds, ambiguity, generated filtering, correction stamps and continuing mission credit passed\n";
}

int main() {
    for(int channel:{0,1})integrated_worlds(channel);
    std::cout<<"Grounded reader to seven-action goal controller bridge protocol passed; no learned scheduling or automatic derived-evidence replay claim\n";
}
