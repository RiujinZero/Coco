#pragma once
// One optional byte-stream session for grounded questions and evidence edits.
// Source identities are host metadata; role-like text inside a message is not.
#include "correction_language_grounding.h"
#include "grounded_archive.h"
#include <charconv>
#include <memory>

namespace laminar::reasoning {
struct GroundedSession {
    std::shared_ptr<LanguageReasoner> reader;
    HistoryReferenceGrounding references;
    CorrectionLanguageGrounding corrections;
    RevisableGrounding journal;
    GroundedCorrectionDialogue dialogue;
    std::string report_source="trusted_stream",user_source="trusted_stream",pending_source;
    std::string pending_user,status="session_ready",route;
    uint64_t next_report=1,requests=0,questions=0,edits=0,clarifications=0,refusals=0,reports=0,resets=0;
    bool user_ready=false;
    int mention_field=0,source_field=1;
    struct FeedbackAnchor {
        uint64_t episode=0,report=0,revision=0,history=0;
        std::string model;
    };
    enum FeedbackKind {ADD_FEEDBACK,REVISE_FEEDBACK,WITHDRAW_FEEDBACK};
    std::set<std::string> teachers,initial_frames,acquired_frames;
    uint64_t next_learning_scene=1,feedback_accepted=0,feedback_rejected=0,feedback_duplicates=0;
    std::string teaching_status;
    std::optional<GroundedArchive> archive;
    std::vector<std::string> initial_context;

    GroundedSession() = default;   // checkpoint restore target
    template<class Ar> void state_io(Ar& ar) {
        ar(reader);ar(references);ar(corrections);ar(journal);ar(dialogue);ar(report_source);ar(user_source);ar(pending_source);
        ar(pending_user);ar(status);ar(route);ar(next_report);ar(requests);ar(questions);ar(edits);ar(clarifications);ar(refusals);ar(reports);ar(resets);
        ar(user_ready);ar(mention_field);ar(source_field);ar(teachers);ar(initial_frames);ar(acquired_frames);
        ar(next_learning_scene);ar(feedback_accepted);ar(feedback_rejected);ar(feedback_duplicates);ar(teaching_status);ar(archive);ar(initial_context);
    }
    explicit GroundedSession(std::shared_ptr<LanguageReasoner> r):reader(std::move(r)) {
        if(!reader||!reader->environment_dialogue||!reader->partial_observation||!reader->grounded_requests)
            throw std::invalid_argument("grounded session requires the environment dialogue reader");
        for(const auto& [key,frame]:reader->observation_language.frames) {
            initial_frames.insert(key);
            for(uint64_t scene:frame.scenes)next_learning_scene=std::max(next_learning_scene,scene+1);
        }
    }
    std::shared_ptr<GroundedSession> clone() const {
        auto copy=std::make_shared<GroundedSession>(*this);copy->reader=std::make_shared<LanguageReasoner>(*reader);return copy;
    }
    void enable_archive() {
        if(resets||reports||journal.base||!initial_context.empty())throw std::invalid_argument("enable archive before session input");
        archive.emplace(*reader);
    }
    bool synchronize_archive() {
        if(!archive)return true;
        archive->capture(resets,initial_context,journal,reader->names);
        bool okay=archive->refresh(*reader,journal,resets);track_learning();status=archive->status;return okay;
    }
    void finish_archived_correction() {
        if(!archive)return;
        auto text=reader->reply,code=status;
        if(synchronize_archive()){reader->reply=text;reader->reply_offset=0;reader->status=code;status=code;}
        else reply("The remaining evidence conflicts.",archive->status);
    }
    std::optional<GroundedArchive::Anchor> archived_anchor(uint64_t episode,uint64_t report) {
        if(!archive||episode>=resets)return {};
        return archive->anchor(episode,report);
    }
    bool teach_archived(const GroundedArchive::Anchor& anchor,uint64_t id,const std::string& question,const std::string& answer,
                        const std::string& source,FeedbackKind kind=ADD_FEEDBACK,uint64_t previous_revision=0,bool generated=false) {
        auto reject=[&](const std::string& why){teaching_status=why;++feedback_rejected;return false;};
        if(generated)return reject("self_generated_feedback");
        if(!teachers.count(source))return reject("unverified_teacher");
        if(!archive||anchor.episode>=resets)return reject("unknown_archived_episode");
        if(!archive->teach(anchor,id,question,answer,source,int(kind),previous_revision))return reject(archive->status);
        if(archive->status=="duplicate_feedback"){++feedback_duplicates;teaching_status=archive->status;return true;}
        ++feedback_accepted;
        // A revision of remote teaching evidence invalidates pending local
        // actions/clarifications even when the projected facts happen to agree.
        ++reader->history_revision;journal.history_revision=reader->history_revision;
        synchronize_archive();teaching_status=archive->status;return true;
    }
    void reset_history(const std::vector<std::string>& previous_names={}) {
        if(archive&&(!initial_context.empty()||journal.base))archive->capture(resets,initial_context,journal,previous_names);
        // An unresolved contradiction must not export the last apparently
        // successful narrowing into the next episode.
        if(!archive&&journal.base&&!journal.consistent)reader->observation_language=journal.base->observation_language;
        std::erase_if(acquired_frames,[&](const std::string& key){return !reader->observation_language.frames.count(key);});
        journal=RevisableGrounding{};dialogue=GroundedCorrectionDialogue{};next_report=1;
        pending_user.clear();pending_source.clear();user_ready=false;route.clear();status="episode_reset";++resets;
        initial_context.clear();
    }
    bool start_history() {
        if(journal.base)return true;
        // An archive can reconstruct an initially uncertain context from raw
        // evidence. Start its provenance container even when the live projection
        // is unavailable; only the subsequent full replay may authorize answers.
        auto initial=*reader;
        if(archive){initial.complete=true;initial.event_issue.clear();initial.awaiting_action_observation=false;}
        if(journal.start(initial)){journal.trainable_frames=acquired_frames;synchronize_archive();return true;}
        status=journal.status;return false;
    }
    void track_learning() {
        for(const auto& [key,frame]:reader->observation_language.frames)if(!initial_frames.count(key))acquired_frames.insert(key);
        journal.trainable_frames=acquired_frames;
    }
    std::optional<FeedbackAnchor> feedback_anchor(uint64_t id) {
        auto record=journal.find(id);if(!record||!record->active||!journal.guard(*reader))return {};
        return FeedbackAnchor{reader->episode_revision,id,record->revision,reader->history_revision,reader->observation_language.signature()};
    }
    // This is a host-verified evidence API. No dialogue role or generated text
    // grants teacher authority, and anchors bind delayed feedback to its history.
    bool teach(const FeedbackAnchor& anchor,uint64_t id,const std::string& question,const std::string& answer,
               const std::string& source,FeedbackKind kind=ADD_FEEDBACK,uint64_t previous_revision=0,bool generated=false) {
        auto reject=[&](const std::string& why){teaching_status=why;++feedback_rejected;return false;};
        if(generated)return reject("self_generated_feedback");
        if(!teachers.count(source))return reject("unverified_teacher");
        auto current=feedback_anchor(anchor.report);
        if(!current||current->episode!=anchor.episode||current->revision!=anchor.revision||current->history!=anchor.history||current->model!=anchor.model)
            return reject("stale_teaching_anchor");
        bool accepted=false;auto changes=journal.changes.size();
        if(kind==ADD_FEEDBACK)accepted=journal.feedback(*reader,anchor.report,id,question,answer,source);
        else if(kind==REVISE_FEEDBACK||kind==WITHDRAW_FEEDBACK) {
            auto record=journal.find(anchor.report);auto evidence=record->evidence.find(id);
            if(evidence==record->evidence.end()||evidence->second.question!=question)return reject("unknown_feedback_question");
            accepted=journal.revise_feedback(*reader,anchor.report,id,previous_revision,answer,source,kind==REVISE_FEEDBACK);
        } else return reject("unknown_feedback_kind");
        if(!accepted)return reject(journal.status);
        teaching_status=journal.status;status=journal.status;
        if(journal.changes.size()==changes)++feedback_duplicates;else ++feedback_accepted;
        track_learning();if(archive&&journal.changes.size()!=changes){synchronize_archive();teaching_status=archive->status;}return true;
    }
    void reply(const std::string& text,const std::string& code) {
        status=code;GroundedHistoryCorrection::reply(*reader,text,code);
    }
    static std::optional<uint64_t> choice(const std::string& text) {
        if(text.empty())return {};uint64_t value=0;
        const auto result=std::from_chars(text.data(),text.data()+text.size(),value);
        if(result.ec!=std::errc{}||result.ptr!=text.data()+text.size())return {};
        return value;
    }
    void dispatch() {
        auto text=pending_user;pending_user.clear();user_ready=false;++requests;route.clear();
        if(!start_history()){++refusals;reply("The evidence history is unavailable.",status);return;}
        auto records=HistoryReferenceGrounding::records(journal,*reader,mention_field,source_field);
        auto selected=choice(text);
        if(selected&&dialogue.correction.pending) {
            route="clarification";
            if(user_source!=pending_source){++refusals;reply("That choice has a different source.","clarification_source_mismatch");return;}
            auto token=dialogue.correction.pending->token;
            if(dialogue.choose(*reader,journal,references,corrections,records,token,*selected)){++edits;status=dialogue.status;finish_archived_correction();}
            else {
                ++refusals;status=dialogue.status;
                if(status=="invalid_report_choice")reply("Choose one of the listed reports.",status);
            }
            return;
        }
        // Routing is based on learned interpretations, never a verb list or a
        // test-case tag. A question mark is not required to recognize a question.
        const auto correction=corrections.meaning(*reader,references,records,text);
        const bool correction_known=correction.status!="unknown_correction"&&correction.status!="correction_span_budget";
        auto question=RevisableGrounding::question(*reader,text);
        if(correction_known&&question) {
            route="ambiguous";++refusals;reply("That request has more than one interpretation.","ambiguous_request_route");return;
        }
        if(correction_known) {
            route="correction";pending_source=user_source;
            bool applied=dialogue.begin(*reader,journal,references,corrections,records,text,user_source);
            if(applied)++edits;
            else if(dialogue.correction.pending&&dialogue.status=="ambiguous_reference")++clarifications;
            else if(dialogue.status!="kept_report")++refusals;
            status=dialogue.status;if(applied)finish_archived_correction();return;
        }
        if(question) {
            route="question";++questions;
            if(!journal.consistent||!journal.guard(*reader)){reply("I do not know.","unavailable_evidence_history");return;}
            reader->question=text;reader->prepare();status=reader->status;
            if(status=="answer"||status=="grounded_answer")++reader->answered;else ++reader->uncertain_replies;
            if(reader->reply.empty())reply("I do not know.",status);return;
        }
        route="unknown";++refusals;reply("I do not understand that request.","unknown_grounded_request");
    }
    void observe(uint8_t c,bool generated) {
        auto& r=*reader;
        if(generated){r.observe(c,true);return;}
        const auto old_episode=r.episode_revision,old_observation=r.observation_revision;
        const auto old_role=r.stream_role;
        const bool at_marker=c==':'&&!r.line_has_role;
        const auto marker=at_marker?LanguageReasoner::trim(r.segment+char(c)):std::string{};
        const auto text=c=='\n'?LanguageReasoner::trim(r.segment):std::string{};
        // CONTEXT is an initial-condition envelope. Once history has started,
        // new evidence must enter as ENVIRONMENT so it receives provenance.
        if(c=='\n'&&old_role==LanguageReasoner::StreamRole::CONTEXT&&journal.base&&!text.empty()) {
            r.segment.clear();r.line_has_role=false;r.reply.clear();r.reply_offset=0;
            status="late_context_rejected";return;
        }
        if(at_marker&&marker=="USER:"){pending_user.clear();user_ready=false;}
        // The session will arbitrate question/correction meanings at this
        // boundary; suppress the reader's standalone automatic question path.
        if(at_marker&&marker=="ASSISTANT:"&&user_ready)r.dialogue_question_pending=false;
        auto previous_names=archive&&c=='>'?r.names:std::vector<std::string>{};
        r.observe(c,false);
        if(r.episode_revision!=old_episode){reset_history(previous_names);return;}
        if(r.observation_revision!=old_observation&&!r.last_observation_environment&&old_role==LanguageReasoner::StreamRole::CONTEXT)
            initial_context.push_back(r.last_observation_text);
        if(at_marker&&(marker=="ENVIRONMENT:"||marker=="USER:"))start_history();
        if(r.observation_revision!=old_observation&&r.last_observation_environment) {
            if(!journal.base||!journal.append(r,next_report,report_source,false,next_learning_scene)) {
                status=journal.status;r.complete=false;r.event_issue="unrecorded_environment";
            } else {++next_report;++next_learning_scene;++reports;track_learning();status=journal.status;synchronize_archive();}
        }
        if(c=='\n'&&old_role==LanguageReasoner::StreamRole::USER&&!text.empty()&&!r.dialogue_line_overflow) {
            pending_user=text;user_ready=true;
        }
        if(at_marker&&marker=="ASSISTANT:"&&user_ready)dispatch();
    }
    std::string report() const {
        return "requests="+std::to_string(requests)+" questions="+std::to_string(questions)+
            " reports="+std::to_string(reports)+" edits="+std::to_string(edits)+
            " clarifications="+std::to_string(clarifications)+" refusals="+std::to_string(refusals)+
            " episodes="+std::to_string(resets)+" status="+status;
    }
};
} // namespace laminar::reasoning
