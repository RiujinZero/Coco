#pragma once
// Cross-episode provenance, replayed from the immutable bootstrap rather than
// from previously narrowed candidate sets. Shared possible-world bindings keep
// correlations between frame meanings across physical-world resets.
#include "revisable_grounding.h"

namespace laminar::reasoning {
struct GroundedArchive {
    using Report=RevisableGrounding::Report;
    struct Episode {
        std::vector<std::string> context,names;
        std::vector<Report> reports;
    };
    struct Anchor {uint64_t episode,report,report_revision,archive_revision;};
    LanguageReasoner prior;
    std::map<uint64_t,Episode> episodes;
    RevisableGrounding replayed;
    std::map<std::pair<uint64_t,uint64_t>,uint64_t> report_keys;
    std::map<uint64_t,uint64_t> context_keys;
    uint64_t revision=0;
    size_t episode_budget=128,record_budget=8192,text_budget=8*1024*1024;
    size_t branch_budget=4096,history_budget=65536,evaluation_budget=4194304;
    bool consistent=true,truncated=false;
    std::string status="archive_ready";

    explicit GroundedArchive(const LanguageReasoner& r):prior(r){prior.reset_episode();}
    GroundedArchive() = default;   // checkpoint restore target
    template<class Ar> void state_io(Ar& ar) {
        ar(prior);ar(episodes);ar(replayed);ar(report_keys);ar(context_keys);ar(revision);ar(episode_budget);ar(record_budget);ar(text_budget);
        ar(branch_budget);ar(history_budget);ar(evaluation_budget);ar(consistent);ar(truncated);ar(status);
    }
    static std::string signature(const Episode& episode) {
        std::ostringstream out;
        for(const auto& text:episode.context)out<<text.size()<<':'<<text;
        for(const auto& r:episode.reports) {
            out<<r.id<<','<<r.revision<<','<<r.active<<','<<r.learning_scene<<':'<<r.text.size()<<':'<<r.text<<r.source.size()<<':'<<r.source;
            for(const auto& [id,e]:r.evidence)out<<id<<','<<e.revision<<','<<e.active<<':'<<e.question.size()<<':'<<e.question<<e.answer.size()<<':'<<e.answer<<e.source.size()<<':'<<e.source;
        }
        return out.str();
    }
    static size_t storage_bytes(const Episode& episode) {
        size_t bytes=signature(episode).size();
        for(const auto& name:episode.names)bytes+=sizeof(std::string)+name.size();
        return bytes;
    }
    bool fits(uint64_t id,const Episode& next) const {
        size_t records=next.context.size()+next.reports.size()+1,bytes=storage_bytes(next);
        for(const auto& [key,e]:episodes)if(key!=id){records+=e.context.size()+e.reports.size()+1;bytes+=storage_bytes(e);}
        return episodes.size()+size_t(!episodes.count(id))<=episode_budget&&records<=record_budget&&bytes<=text_budget;
    }
    bool capture(uint64_t id,const std::vector<std::string>& context,const RevisableGrounding& journal,const std::vector<std::string>& names) {
        if(truncated)return false;
        Episode next{context,names,journal.reports};auto old=episodes.find(id);
        if(!fits(id,next)) {
            truncated=true;consistent=false;status="archive_capacity";return false;
        }
        if(old!=episodes.end()&&signature(old->second)==signature(next))return true;
        episodes[id]=std::move(next);++revision;return true;
    }
    Report* find(uint64_t episode,uint64_t report) {
        auto e=episodes.find(episode);if(e==episodes.end())return nullptr;
        for(auto& r:e->second.reports)if(r.id==report)return &r;return nullptr;
    }
    std::optional<Anchor> anchor(uint64_t episode,uint64_t report) {
        auto r=find(episode,report);if(!r||!r->active||truncated)return {};
        return Anchor{episode,report,r->revision,revision};
    }
    // kind: add=0, revise=1, withdraw=2. Authority is checked by the session;
    // existing evidence also binds its original teacher and prior revision.
    bool teach(const Anchor& expected,uint64_t id,const std::string& q,const std::string& a,
               const std::string& source,int kind,uint64_t previous_revision) {
        auto current=anchor(expected.episode,expected.report);
        if(!current||current->report_revision!=expected.report_revision||current->archive_revision!=expected.archive_revision){status="stale_archive_anchor";return false;}
        auto r=find(expected.episode,expected.report);auto old=r->evidence.find(id);
        if(kind<0||kind>2){status="unknown_feedback_kind";return false;}
        auto language=prior;language.names=episodes.at(expected.episode).names;language.entities.clear();
        for(size_t i=0;i<language.names.size();++i)language.entities[language.names[i]]=i;
        RevisableGrounding validation;
        if(kind!=2&&!validation.valid_evidence(language,q,a)){status=validation.status;return false;}
        const auto previous=*r;
        if(kind==0) {
            if(old!=r->evidence.end()) {
                bool same=old->second.active&&old->second.source==source&&old->second.question==q&&old->second.answer==a;
                status=same?"duplicate_feedback":"conflicting_feedback_id";return same;
            }
            r->evidence[id]={q,a,source,0,true};
        } else {
            if(old==r->evidence.end()||old->second.question!=q){status="unknown_feedback_question";return false;}
            auto& e=old->second;
            if(e.source!=source||e.revision!=previous_revision){status="stale_evidence_revision";return false;}
            if(kind==1)e.answer=a;e.active=kind==1;++e.revision;
        }
        // Remote receipts mutate an older episode rather than the current
        // capture. Check the whole archive before accepting the edit. A rejected
        // receipt can be retried and has not lost any previously accepted input.
        if(!fits(expected.episode,episodes.at(expected.episode))) {
            *r=previous;status="archive_capacity";return false;
        }
        ++revision;status="recorded_archive_feedback";return true;
    }
    bool refresh(LanguageReasoner& reader,RevisableGrounding& local,uint64_t current) {
        const bool history_started=bool(local.base);
        auto fail=[&](const std::string& why) {
            consistent=false;status=why;reader.complete=false;reader.event_issue=why;reader.prepared=false;reader.reply.clear();reader.reply_offset=0;reader.status=why;
            local.consistent=false;local.status=why;local.expected_state=RevisableGrounding::state_stamp(reader);
            local.observation_revision=reader.observation_revision;local.history_revision=reader.history_revision;return false;
        };
        if(truncated)return fail("archive_capacity");
        if(reader.events.signature()!=prior.events.signature()||reader.forms!=prior.forms||reader.memory.procedures!=prior.memory.procedures||
           reader.grounding.signature()!=prior.grounding.signature()||reader.memory.skills.kernel.routes.W!=prior.memory.skills.kernel.routes.W||
           reader.relations[0]!=prior.relations[0]||reader.relations[1]!=prior.relations[1])return fail("changed_archive_prior");
        auto work=prior;work.world_budget=reader.world_budget;
        RevisableGrounding combined;combined.branch_budget=branch_budget;combined.history_budget=history_budget;combined.evaluation_budget=evaluation_budget;
        if(!combined.start(work))return fail(combined.status);
        std::map<std::pair<uint64_t,uint64_t>,uint64_t> links;std::map<uint64_t,uint64_t> starts;uint64_t next=1;
        for(const auto& [id,e]:episodes) {
            Report boundary;boundary.id=next++;boundary.active=false;boundary.reset_before=true;combined.reports.push_back(boundary);starts[id]=boundary.id;
            for(const auto& text:e.context) {
                Report context;context.id=next++;context.text=text;context.source="initial_context";
                combined.reports.push_back(std::move(context));starts[id]=next-1;
            }
            for(auto record:e.reports){links[{id,record.id}]=next;record.id=next++;combined.reports.push_back(std::move(record));}
        }
        if(!combined.replay(work))return fail(combined.status);
        auto context=combined.at(starts.at(current));if(!context)return fail("missing_archive_context");
        // Only knowledge and meanings are published; byte framing, request
        // ownership, counters and outstanding physical-action state stay live.
        reader.names=work.names;reader.entities=work.entities;reader.facts=work.facts;reader.observations=work.observations;
        reader.possible_worlds=work.possible_worlds;reader.observation_language=work.observation_language;
        reader.world_model_signature=reader.events.signature();reader.observation_language_signature=reader.observation_language.signature();
        reader.complete=true;reader.event_issue.clear();reader.prepared=false;reader.reply.clear();reader.reply_offset=0;reader.status="replayed_archive";
        context->episode_revision=reader.episode_revision;
        local.base=*context;local.reports=episodes.at(current).reports;local.history.clear();
        for(const auto& r:local.reports)local.history[r.id]=combined.history.at(links.at({current,r.id}));
        local.published=reader.observation_language;local.expected_state=RevisableGrounding::state_stamp(reader);
        local.observation_revision=reader.observation_revision;local.history_revision=reader.history_revision;local.consistent=true;local.status="replayed_archive";
        if(!history_started)local.base.reset(); // remote teaching does not close a still-arriving initial context
        replayed=std::move(combined);report_keys=std::move(links);context_keys=std::move(starts);consistent=true;status="replayed_archive";return true;
    }
    std::optional<LanguageReasoner> at(uint64_t episode,uint64_t report) const {
        auto link=report_keys.find({episode,report});if(!consistent||link==report_keys.end())return {};
        return replayed.at(link->second);
    }
};
} // namespace laminar::reasoning
