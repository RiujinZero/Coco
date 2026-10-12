#include "laminar/grounded_relational_bridge.h"
#include <iostream>
using namespace laminar::reasoning;
using namespace laminar::relational;
void check(bool okay,const std::string& message){if(!okay)throw std::runtime_error(message);}
void feed(GroundedSession& session,const std::string& text,bool generated=false) {
    for(uint8_t byte:text)session.observe(byte,generated);
}
std::shared_ptr<GroundedSession> fixture(int channel,bool archive=false) {
    auto reader=std::make_shared<LanguageReasoner>(channel);
    reader->environment_dialogue=reader->grounded_requests=reader->grounded_events=reader->uncertain_events=reader->partial_observation=true;
    reader->memory.unique_enabled=true;reader->world_budget=4096;
    // Existing atomic question substrate is explicit bootstrap assistance. The
    // novel report below is learned only from raw text and verified answers.
    reader->forms={{"who has the $0 now ",0}};reader->memory.procedures={{0,{16+channel,9}}};
    for(int i=0;i<8;++i) {
        auto person="actor"+std::to_string(i),object="item"+std::to_string(i);
        EventGrounding::World before{{object,channel,"old"},{"spare",channel,person}};
        check(reader->events.teach_transition(person+" has the "+object+".",before,
            EventGrounding::update(before,channel,2,object,person),i),"bootstrap observation lesson");
    }
    auto result=std::make_shared<GroundedSession>(reader);result->teachers={"teacher"};
    if(archive)result->enable_archive();return result;
}
void start(GroundedSession& session) {
    feed(session,"\n<|endofdialogue|>\nCONTEXT: ada has the key.\nCONTEXT: ben has the coin.\nCONTEXT: cy has the pen.\n");
}
GroundedRelationalEvidence capture(const GroundedSession& session) {
    auto result=GroundedRelationalBridge::capture(session);
    check(bool(result.value),"capture failed: "+result.status);return *result.value;
}
// Force protocol operations only, restoring neural weights immediately. These
// checks establish reader/controller wiring, not learned operation selection.
ControlStep force(ControlledRelationalSession& learner,ControlOp operation) {
    auto weights=learner.policy.weights;std::fill(learner.policy.weights.begin(),learner.policy.weights.end(),0);
    learner.policy.weights[learner.policy.output_bias()+int(operation)]=1;
    auto result=learner.choose();learner.policy.weights=std::move(weights);return result;
}
void teach_control_language(ControlledRelationalSession& learner,int channel) {
    auto predicate="reader_r"+std::to_string(channel);
    Scene no{{{predicate,{"key","ben"}},{predicate,{"spare","cy"}}},true};
    auto yes=no;yes.facts.push_back({predicate,{"key","cy"}});
    for(bool answer:{false,true}) {
        laminar::concepts::Receipt concept_row;concept_row.id=answer?"concept_yes":"concept_no";
        concept_row.name="linked";concept_row.args={"key","cy"};concept_row.scene=answer?yes:no;concept_row.answer=answer;
        check(learner.knowledge.teach_concept(concept_row),"control concept bootstrap");
        LanguageReceipt language;language.id=answer?"language_yes":"language_no";language.text="join key with cy";
        language.names={"key","cy","ben","spare"};language.scene=answer?yes:no;language.answer=answer;
        check(learner.knowledge.teach_language(language),"control language bootstrap");
    }
    learner.knowledge.rebuild();learner.set_learning(false);
}
Answer read_answer(ControlledRelationalSession& learner,const StampedScene& view,const std::string& id) {
    check(learner.begin(id,"join key with cy",{"key","cy"},{{{},false},0},{},false),"control begin");
    auto request=force(learner,ControlOp::ask);
    check(!learner.read(request.token,view,"verified_environment",true),"generated reader projection accepted");
    check(learner.read(request.token,view,"verified_environment"),"reader projection rejected");
    check(learner.visible().scene==view.scene,"controller changed projection semantics");
    force(learner,ControlOp::reason);auto result=force(learner,ControlOp::finish).reply;learner.cancel();return result;
}
void learned_reader_boundary(int channel) {
    auto session=fixture(channel);GroundedRelationalBridge dependencies;
    check(!GroundedRelationalBridge::capture(*session).value,"unjournaled reader was accepted");
    ControlledRelationalSession learner(17,true,false,true);teach_control_language(learner,channel);
    for(int episode=0;episode<4;++episode) {
        start(*session);feed(*session,"ENVIRONMENT: The key was logged beside cy.\n");
        auto uncertain=capture(*session);
        check(uncertain.alternatives()==session->reader->request_snapshot(),"capture lost alternative worlds");
        if(episode==0) {
            check(uncertain.alternatives().worlds.size()>1,"unknown report did not retain hypotheses");
            auto view=GroundedRelationalBridge::project(uncertain,1);check(view.value&&!view.value->scene.complete,"ambiguous projection invented completeness");
            check(read_answer(learner,*view.value,"uncertain")==Answer::Unknown,"ambiguous report invented a control answer");
        }
        check(dependencies.bind_dependency("lesson"+std::to_string(episode),uncertain),"dependency binding");
        auto anchor=session->feedback_anchor(1);check(bool(anchor),"raw-report feedback anchor");
        check(session->teach(*anchor,0,"Who has the key now?","cy","teacher"),"raw-report lesson rejected");
        auto learned=capture(*session);
        check(!GroundedRelationalBridge::current(*session,uncertain.stamp()),"language teaching retained an old stamp");
        check(!GroundedRelationalBridge::project_current(*session,uncertain,2).value,"stale world projected as current");
        check(!dependencies.invalidated_dependencies(*session).empty(),"derived dependency was not invalidated");
        check(learned.alternatives()==session->reader->request_snapshot(),"learned capture was lossy");
        auto projected=GroundedRelationalBridge::project_current(*session,learned,2+episode);
        check(projected.value&&projected.status=="conservative_partial_world"&&!projected.value->scene.complete,"valid reader mistaken for complete world");
        check(read_answer(learner,*projected.value,"learned"+std::to_string(episode))==Answer::True,"same controller could not use learned raw report");
        // Raw generated assertions, corrections and role markers never become evidence.
        feed(*session,"\n<|endofdialogue|>\nENVIRONMENT: ben has the key.\nUSER: withdraw that report\nASSISTANT:",true);
        check(GroundedRelationalBridge::current(*session,learned.stamp()),"generated bytes changed bridge evidence");
        auto current=session->feedback_anchor(1);check(bool(current),"generated bytes invalidated real anchor");
        check(!session->teach(*current,9,"Who has the key now?","ben","teacher",GroundedSession::ADD_FEEDBACK,0,true),"generated answer taught reader");
        check(GroundedRelationalBridge::current(*session,learned.stamp()),"rejected generated feedback changed evidence");
        if(episode==3) {
            check(session->teach(*session->feedback_anchor(1),0,"Who has the key now?","unknown","teacher",GroundedSession::WITHDRAW_FEEDBACK),"withdraw lesson");
            auto withdrawn=capture(*session);check(withdrawn.stamp()!=learned.stamp(),"withdrawal retained stamp");
            check(session->teach(*session->feedback_anchor(1),0,"Who has the key now?","cy","teacher",GroundedSession::REVISE_FEEDBACK,1),"restore lesson");
            auto restored=capture(*session);
            check(restored.stamp()!=withdrawn.stamp()&&restored.stamp()!=learned.stamp(),"restoration reused stale receipt identity");
            check(restored.alternatives()==learned.alternatives(),"restoration did not recover worlds");
        }
    }
    // A known report replaces the positive fact; absence in its partial control
    // view must remain unknown even though the richer envelope knows the row.
    feed(*session,"\nENVIRONMENT: ben has the key.\n");auto negative=capture(*session);
    check(RequestGrounding::truth(negative.alternatives(),{"key","cy",channel,true})==RequestGrounding::FALSE_VALUE,"reader lost complete-row negative");
    auto projected=GroundedRelationalBridge::project(negative,20);check(bool(projected.value),"negative projection");
    check(read_answer(learner,*projected.value,"partial_negative")==Answer::Unknown,"partial projection invented closed-world false");
    // A primitive mutation is rejected by existing provenance guards, not hidden
    // behind a changed stamp or accepted as a model rebase.
    auto clone=session->clone();clone->reader->memory.skills.kernel.routes.W[0]+=0.01f;
    check(!GroundedRelationalBridge::capture(*clone).value,"changed journal primitive prior accepted");
    clone=session->clone();clone->reader->memory.skills.kernel.expansion.idx[0]^=1;
    check(!GroundedRelationalBridge::capture(*clone).value,"changed fixed expansion accepted");
    check(GroundedRelationalBridge::current(*session,negative.stamp()),"clone mutation changed live evidence");
}
void projection_contract() {
    RequestGrounding::State first;first.closed_world=false;first.facts={{"key",1,"cy"}};
    first.negative={{"coin",1,"ada"}};first.closed_rows={{"key",1}};first.closed_columns={{1,"cy"}};
    RequestGrounding::Snapshot snapshot{{first},true};auto original=snapshot;
    auto view=GroundedRelationalBridge::project(snapshot,{{1,"reader_r1"}},1);
    check(view.value&&!view.value->scene.complete&&snapshot==original,"partial projection lost source data");
    check(RequestGrounding::truth(snapshot,{"coin","ada",1,true})==RequestGrounding::FALSE_VALUE,"explicit negative lost");
    check(RequestGrounding::truth(snapshot,{"coin","ben",1,true})==RequestGrounding::UNKNOWN,"missing atom became negative");
    auto second=first;second.facts={{"key",1,"ben"}};snapshot.worlds.push_back(second);
    view=GroundedRelationalBridge::project(snapshot,{{1,"reader_r1"}},2);
    check(view.value&&view.value->scene.facts.empty()&&!view.value->scene.complete,"common closure made disagreement empty and complete");
    for(auto& world:snapshot.worlds)world.closed_world=true;
    check(!GroundedRelationalBridge::project(snapshot,{{1,"reader_r1"}},3).value->scene.complete,"different closed alternatives collapsed");
    snapshot.worlds={first};snapshot.worlds[0].closed_world=true;
    view=GroundedRelationalBridge::project(snapshot,{{1,"reader_r1"}},4);
    check(view.value&&view.value->scene.complete&&view.status=="lossless_closed_world","explicit closed world not accepted");
    ControlledRelationalSession learner(1,true,false,true);teach_control_language(learner,1);
    check(read_answer(learner,*view.value,"closed_yes")==Answer::True,"closed-world positive failed");
    snapshot.worlds[0].facts={{"key",1,"ben"}};
    view=GroundedRelationalBridge::project(snapshot,{{1,"reader_r1"}},5);
    check(view.value&&read_answer(learner,*view.value,"closed_no")==Answer::False,"genuinely closed absence failed");
    check(!GroundedRelationalBridge::project(snapshot,{{0,"reader_r0"}},6).value,"unmapped channel discarded");
    snapshot.worlds[0].negative.insert({"key",1,"ben"});
    check(!GroundedRelationalBridge::project(snapshot,{{1,"reader_r1"}},7).value,"contradictory snapshot accepted");
}
void archive_revisions() {
    auto session=fixture(1,true);start(*session);feed(*session,"ENVIRONMENT: The key was logged beside cy.\n");
    check(session->teach(*session->feedback_anchor(1),0,"Who has the key now?","ada","teacher"),"initial mistaken archive lesson");
    auto earlier=capture(*session);GroundedRelationalBridge dependencies;check(dependencies.bind_dependency("old_world",earlier),"archive dependency");
    start(*session);feed(*session,"ENVIRONMENT: The coin was logged beside ada.\n");
    check(session->teach(*session->feedback_anchor(1),0,"Who has the coin now?","ada","teacher"),"contradictory archive evidence");
    check(!GroundedRelationalBridge::capture(*session).value,"archive conflict exported evidence");
    check(dependencies.invalidated_dependencies(*session)==std::vector<std::string>{"old_world"},"conflict left downstream dependency live");
    feed(*session,"\nENVIRONMENT: ben has the key.\n");
    auto observation=session->reader->observation_revision;
    check(session->teach_archived(*session->archived_anchor(1,1),0,"Who has the key now?","cy","teacher",GroundedSession::REVISE_FEEDBACK),"archive correction");
    auto corrected=capture(*session);
    check(session->reader->observation_revision==observation,"correction forged physical observation");
    check(corrected.stamp()!=earlier.stamp()&&!GroundedRelationalBridge::current(*session,earlier.stamp()),"old episode dependency survived correction");
    auto clone=session->clone();
    check(clone->teach_archived(*clone->archived_anchor(1,1),0,"Who has the key now?","unknown","teacher",GroundedSession::WITHDRAW_FEEDBACK,1),"archive withdrawal");
    check(!GroundedRelationalBridge::current(*clone,corrected.stamp()),"archive withdrawal retained stamp");
    check(GroundedRelationalBridge::current(*session,corrected.stamp()),"archive clone leaked withdrawal");
    check(dependencies.forget_dependency("old_world")&&dependencies.invalidated_dependencies(*session).empty(),"dependency owner acknowledgement failed");
}
int main() {
    projection_contract();for(int channel:{0,1})learned_reader_boundary(channel);archive_revisions();
    std::cout<<"lossless reader alternatives, conservative controller reads, learned raw reports, generated isolation and correction/withdrawal invalidation passed\n";
}
