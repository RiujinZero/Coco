#pragma once
// The actual trainable raw-report reader participates in the continuing learner.
// Its lossless evidence envelope is retained; only conservative views cross to
// the controller. No projected world is used as a concept teaching receipt.
#include "goal_control_learning_harness.h"
#include "laminar/grounded_relational_bridge.h"

namespace laminar::goal_learning::reader_task {
namespace gr=reasoning;
inline constexpr const char* teacher_source="reader_verified_answers";
inline void feed(gr::GroundedSession& reader,const std::string& text,bool generated=false) {
    for(uint8_t byte:text)reader.observe(byte,generated);
}
inline void reset(gr::GroundedSession& reader,const std::string& object="key",
                  const std::string& actor="cy") {
    feed(reader,"\n<|endofdialogue|>\nCONTEXT: ada has the "+object+".\nCONTEXT: ben has the coin.\nCONTEXT: "+actor+" has the pen.\n");
}
inline void export_snapshot(std::ostream& out,const gr::GroundedRelationalEvidence& evidence,
                            const gr::GroundedControlProjection& projection,const std::string& label) {
    out<<"{\"event\":\"reader_snapshot\",\"label\":"<<quote(label)
       <<",\"episode\":"<<evidence.stamp().episode<<",\"observation_revision\":"<<evidence.stamp().observation
       <<",\"history_revision\":"<<evidence.stamp().history<<",\"archive_revision\":"<<evidence.stamp().archive
       <<",\"projection_status\":"<<quote(projection.status)<<",\"worlds\":[";
    bool first=true;for(const auto& world:evidence.alternatives().worlds) {
        if(!first)out<<',';first=false;out<<"{\"closed_world\":"<<world.closed_world<<",\"facts\":[";
        bool atom_first=true;for(const auto& [a,r,b]:world.facts) {
            if(!atom_first)out<<',';atom_first=false;out<<'['<<quote(a)<<','<<r<<','<<quote(b)<<']';
        }
        out<<"],\"negative\":[";atom_first=true;for(const auto& [a,r,b]:world.negative) {
            if(!atom_first)out<<',';atom_first=false;out<<'['<<quote(a)<<','<<r<<','<<quote(b)<<']';
        }
        out<<"],\"closed_rows\":[";atom_first=true;for(const auto& [a,r]:world.closed_rows) {
            if(!atom_first)out<<',';atom_first=false;out<<'['<<quote(a)<<','<<r<<']';
        }
        out<<"],\"closed_columns\":[";atom_first=true;for(const auto& [r,b]:world.closed_columns) {
            if(!atom_first)out<<',';atom_first=false;out<<'['<<r<<','<<quote(b)<<']';
        }out<<"]}";
    }
    out<<"],\"projection\":";if(projection.value)scene(out,*projection.value);else out<<"null";out<<"}\n";
}
class Component {
    std::shared_ptr<gr::GroundedSession> session_;
    uint64_t revision_=0;
    std::optional<gr::GroundedRelationalEvidence> initially_ambiguous_;
public:
    Component() {
        auto language=std::make_shared<gr::LanguageReasoner>(1);
        language->environment_dialogue=language->grounded_requests=language->grounded_events=
            language->uncertain_events=language->partial_observation=true;
        language->memory.unique_enabled=true;language->world_budget=4096;
        // Existing atomic question program is declared substrate assistance.
        language->forms={{"who has the $0 now ",0}};language->memory.procedures={{0,{17,9}}};
        for(int i=0;i<8;++i) {
            const auto actor="actor"+std::to_string(i),object="item"+std::to_string(i);
            gr::EventGrounding::World before{{object,1,"old"},{"spare",1,actor}};
            require(language->events.teach_transition(actor+" has the "+object+".",before,
                gr::EventGrounding::update(before,1,2,object,actor),i),"reader atomic observation lesson");
        }
        session_=std::make_shared<gr::GroundedSession>(language);session_->teachers={teacher_source};
        session_->enable_archive();
    }
    Component clone()const {auto copy=*this;copy.session_=session_->clone();return copy;}
    // Complete reader state for subsystem checkpoints (laminar/state_io.h). The session is owned by this component.
    template<class Ar> void state_io(Ar& ar){ar(session_);ar(revision_);ar(initially_ambiguous_);}
    const auto& session()const{return *session_;}
    void teach(GoalControlledRelationalSession& learner,std::ostream& out) {
        require(!learner.mission_active()&&learner.learning(),"reader teaching boundary");
        out<<"{\"event\":\"reader_substrate\",\"channel\":1,\"atomic_question_form\":\"who has the $0 now \","
              "\"atomic_question_program\":[17,9],\"atomic_has_transition_examples\":8,\"source\":\"explicit_fixture_setup\"}\n";
        for(int i=0;i<8;++i) {
            const auto actor="actor"+std::to_string(i),object="item"+std::to_string(i);
            out<<"{\"event\":\"reader_atomic_transition\",\"source\":\"explicit_fixture_setup\",\"text\":"
               <<quote(actor+" has the "+object+".")<<",\"before\":[["<<quote(object)<<",1,\"old\"],[\"spare\",1,"<<quote(actor)
               <<"]],\"after\":[["<<quote(object)<<",1,"<<quote(actor)<<"],[\"spare\",1,"<<quote(actor)<<"]]}\n";
        }
        for(int episode=0;episode<4;++episode) {
            reset(*session_);feed(*session_,"ENVIRONMENT: The key was logged beside cy.\n");
            auto before=gr::GroundedRelationalBridge::capture(*session_);require(bool(before.value),"reader prelesson capture");
            if(!episode)initially_ambiguous_=*before.value;
            export_snapshot(out,*before.value,gr::GroundedRelationalBridge::project(*before.value,++revision_),"before_answer_"+std::to_string(episode));
            const auto anchor=session_->feedback_anchor(1);require(bool(anchor),"reader feedback anchor");
            require(session_->teach(*anchor,0,"Who has the key now?","cy",teacher_source),"raw report answer teaching");
            out<<"{\"event\":\"reader_raw_answer\",\"episode\":"<<session_->resets
               <<",\"report\":\"The key was logged beside cy.\",\"question\":\"Who has the key now?\","
                 "\"answer\":\"cy\",\"source\":"<<quote(teacher_source)<<",\"generated\":false}\n";
            auto after=gr::GroundedRelationalBridge::capture(*session_);require(bool(after.value),"reader learned capture");
            export_snapshot(out,*after.value,gr::GroundedRelationalBridge::project(*after.value,++revision_),"after_answer_"+std::to_string(episode));
        }
        // Structured language learns from independently authored raw scenes and
        // labels, never from a guessed or intersected reader projection.
        for(bool answer:{false,true}) {
            Scene world{{{"reader_r1",{"key","ben"}},{"reader_r1",{"spare","cy"}}},true};
            if(answer)world.facts.push_back({"reader_r1",{"key","cy"}});
            const auto id=std::string("reader_link_")+(answer?"yes":"no");
            concepts::Receipt row;row.id=id;row.name="reader_linked";row.args={"key","cy"};row.scene=world;row.answer=answer;
            LanguageReceipt language;language.id=id;language.text="locate key with cy";language.names={"key","cy","ben","spare"};language.scene=world;language.answer=answer;
            require(learner.knowledge.teach_concept(row)&&learner.knowledge.teach_language(language),"reader controller language lesson");
            out<<"{\"event\":\"reader_control_language_teaching\",\"id\":"<<quote(id)<<",\"source\":"<<quote(row.source)
               <<",\"language_source\":"<<quote(language.source)<<",\"answer\":"<<answer
               <<",\"text\":\"locate key with cy\",\"concept\":\"reader_linked\",\"args\":[\"key\",\"cy\"],\"scene\":";scene(out,{world,0});out<<"}\n";
        }
        learner.knowledge.rebuild();out.flush();
    }
    void boundary_probes(const GoalControlledRelationalSession& learner,std::ostream& out,const std::string& label)const {
        require(initially_ambiguous_.has_value(),"reader probes precede teaching");
        auto ambiguous=gr::GroundedRelationalBridge::project(*initially_ambiguous_,0);
        require(ambiguous.value&&!ambiguous.value->scene.complete,"ambiguous reader lost uncertainty");
        const auto ambiguous_answer=learner.knowledge.answer("locate key with cy",ambiguous.value->scene);
        require(ambiguous_answer==Answer::Unknown,"ambiguous reader produced Boolean conclusion");
        auto copy=session_->clone();reset(*copy);feed(*copy,"ENVIRONMENT: ben has the key.\n");
        auto captured=gr::GroundedRelationalBridge::capture(*copy);require(bool(captured.value),"reader negative capture");
        auto negative=gr::GroundedRelationalBridge::project(*captured.value,0);
        require(negative.value&&!negative.value->scene.complete,"reader negative invented closure");
        const auto negative_answer=learner.knowledge.answer("locate key with cy",negative.value->scene);
        require(negative_answer==Answer::Unknown,"partial negative became structured false");
        const auto stamp=captured.value->stamp();feed(*copy,"ENVIRONMENT: cy has the key.\n",true);
        require(gr::GroundedRelationalBridge::current(*copy,stamp),"generated raw text changed reader evidence");
        out<<"{\"event\":\"reader_uncertainty_probe\",\"label\":"<<quote(label)
           <<",\"initial_alternatives\":"<<initially_ambiguous_->alternatives().worlds.size()
           <<",\"ambiguous_answer\":"<<int(ambiguous_answer)<<",\"partial_negative_answer\":"<<int(negative_answer)
           <<",\"generated_isolated\":true,\"control_or_semantic_update\":false}\n";
        export_snapshot(out,*captured.value,negative,label+"_partial_negative");out.flush();
    }
    StampedScene observe(const std::string& object,const std::string& actor,std::ostream& out,const std::string& label) {
        reset(*session_,object,actor);feed(*session_,"ENVIRONMENT: The "+object+" was logged beside "+actor+".\n");
        auto capture=gr::GroundedRelationalBridge::capture(*session_);require(bool(capture.value),"reader mission capture");
        auto projected=gr::GroundedRelationalBridge::project_current(*session_,*capture.value,++revision_);
        require(bool(projected.value),"reader mission projection");export_snapshot(out,*capture.value,projected,label);out.flush();
        return *projected.value;
    }
};
class Environment final:public goal_learning::Environment {
    StampedScene view_;
    std::vector<ScheduledRequest> requests_;
public:
    Environment(Component& reader,const std::string& object,const std::string& actor,std::ostream& evidence,const std::string& label)
        :view_(reader.observe(object,actor,evidence,label)) {
        PublicGoal request;request.id="reader_query";request.world_id="reader_world";request.text="locate "+object+" with "+actor;
        request.names={object,actor};request.visible=view_;request.effect=false;request.deadline=256;requests_.push_back({0,std::move(request)});
    }
    const std::vector<ScheduledRequest>& schedule()const override{return requests_;}
    StampedScene read(const std::string& world)const override {require(world=="reader_world","unknown reader world");return view_;}
    StampedScene execute(const std::string&,const Action&)override{throw std::logic_error("reader query has no physical affordances");}
    bool satisfaction(const PublicGoal& request)const override {
        require(request.id=="reader_query"&&request.world_id=="reader_world","unknown reader request");
        // The independently supplied report describes this positive relation.
        // This label does not inspect the inferred snapshot to invent truth.
        return true;
    }
    const char* family()const override{return "learned_raw_report_positive_projection";}
};
} // namespace laminar::goal_learning::reader_task
