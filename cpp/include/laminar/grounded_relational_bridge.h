#pragma once
// Preserve learned reader uncertainty and provenance at the structured boundary.
// A conservative Scene is only a controller view; it is never teaching evidence.
// This does not replay derived receipts, connect Organism, or plan in partial worlds.
#include "grounded_session.h"
#include "relational_control.h"
#include <bit>

namespace laminar::reasoning {
struct GroundedEvidenceStamp {
    uint64_t episode=0,observation=0,history=0,archive=0;
    std::string state,model,evidence;
    bool operator==(const GroundedEvidenceStamp&)const=default;
};

struct GroundedControlProjection {
    std::optional<relational::StampedScene> value;
    std::string status;
};

class GroundedRelationalEvidence {
    RequestGrounding::Snapshot alternatives_;
    std::vector<std::string> names_;
    std::map<int,std::string> channels_;
    GroundedEvidenceStamp stamp_;
    friend class GroundedRelationalBridge;
public:
    const auto& alternatives()const{return alternatives_;}
    const auto& names()const{return names_;}
    const auto& channels()const{return channels_;}
    const auto& stamp()const{return stamp_;}
    template<class Ar> void state_io(Ar& ar){ar(alternatives_);ar(names_);ar(channels_);ar(stamp_);}
};

class GroundedRelationalBridge {
    // This index reports dependencies requiring owner-controlled invalidation.
    // It does not certify, withdraw or rebuild a downstream teaching receipt.
    std::map<std::string,GroundedEvidenceStamp> dependencies_;
    static void text(std::ostream& out,const std::string& value){out<<value.size()<<':'<<value;}
    static std::string primitive_signature(const LanguageReasoner& reader) {
        const auto& memory=reader.memory;const auto& kernel=memory.skills.kernel;
        const auto& expansion=kernel.expansion;std::ostringstream out;
        out<<memory.lookup_enabled<<memory.inverse_enabled<<memory.unique_enabled<<';';
        for(bool available:memory.available_skills)out<<available;
        out<<';'<<kernel.cfg.obs_dim<<','<<kernel.cfg.n_tendencies<<','<<kernel.cfg.consequence_dim<<';'
           <<expansion.in_dim<<','<<expansion.n_units<<','<<expansion.k_active<<','<<expansion.claws<<';';
        for(int index:expansion.idx)out<<index<<',';
        out<<';';for(float weight:expansion.w)out<<std::bit_cast<uint32_t>(weight)<<',';
        out<<';';for(float weight:kernel.routes.W)out<<std::bit_cast<uint32_t>(weight)<<',';
        return out.str();
    }
    static GroundedEvidenceStamp stamp(const GroundedSession& session) {
        const auto& r=*session.reader;GroundedEvidenceStamp out;
        out.episode=r.episode_revision;out.observation=r.observation_revision;out.history=r.history_revision;
        out.archive=session.archive?session.archive->revision:0;
        out.state=RevisableGrounding::state_stamp(r);
        std::ostringstream model;
        text(model,r.events.signature());text(model,r.observation_language.signature());text(model,r.grounding.signature());
        text(model,r.requests.signature());text(model,r.actuators.signature());
        text(model,session.references.signature());text(model,session.corrections.signature());
        model<<r.relations[0]<<','<<r.relations[1]<<';'<<r.shared_meanings<<r.phrase_fragments
             <<r.grounded_events<<r.uncertain_events<<r.partial_observation<<r.grounded_requests
             <<r.memory.lookup_enabled<<r.memory.inverse_enabled<<r.memory.unique_enabled<<';';
        for(bool available:r.memory.available_skills)model<<available;
        for(const auto& [form,cue]:r.forms){text(model,form);model<<cue<<';';}
        for(const auto& [cue,program]:r.memory.procedures){model<<cue<<':';for(int op:program)model<<op<<',';model<<';';}
        // Bit identity avoids stream rounding hiding a small neural change.
        text(model,primitive_signature(r));
        out.model=model.str();std::ostringstream evidence;
        GroundedArchive::Episode current{session.initial_context,r.names,session.journal.reports};
        text(evidence,GroundedArchive::signature(current));evidence<<bool(session.archive)<<';';
        if(session.archive)for(const auto& [id,episode]:session.archive->episodes) {
            evidence<<id<<':';text(evidence,GroundedArchive::signature(episode));
        }
        out.evidence=evidence.str();return out;
    }
public:
    struct Capture {
        std::optional<GroundedRelationalEvidence> value;
        std::string status;
    };
    static Capture capture(const GroundedSession& session) {
        if(!session.reader)return {{},"missing_reader"};
        // Guard updates diagnostic status, so use a private journal copy.
        auto journal=session.journal;
        if(!journal.consistent)return {{},"inconsistent_journal"};
        if(!journal.guard(*session.reader))return {{},journal.status};
        if(primitive_signature(*session.reader)!=primitive_signature(*journal.base))return {{},"changed_reader_primitive_prior"};
        if(session.archive&&(!session.archive->consistent||session.archive->truncated))return {{},"unavailable_archive"};
        auto snapshot=session.reader->request_snapshot();
        if(!snapshot.valid||snapshot.worlds.empty())return {{},"unavailable_reader_world"};
        GroundedRelationalEvidence result;result.alternatives_=std::move(snapshot);result.names_=session.reader->names;
        for(int channel:session.reader->relations) {
            if(channel<0)return {{},"invalid_reader_channel"};
            result.channels_[channel]="reader_r"+std::to_string(channel);
        }
        result.stamp_=stamp(session);return {std::move(result),"captured"};
    }
    static bool current(const GroundedSession& session,const GroundedEvidenceStamp& expected) {
        auto result=capture(session);return result.value&&result.value->stamp()==expected;
    }
    bool bind_dependency(const std::string& id,const GroundedRelationalEvidence& evidence) {
        if(id.empty()||dependencies_.count(id))return false;
        dependencies_.emplace(id,evidence.stamp());return true;
    }
    bool forget_dependency(const std::string& id){return dependencies_.erase(id)>0;}
    std::vector<std::string> invalidated_dependencies(const GroundedSession& session)const {
        auto result=capture(session);std::vector<std::string> invalid;
        for(const auto& [id,expected]:dependencies_)
            if(!result.value||result.value->stamp()!=expected)invalid.push_back(id);
        return invalid;
    }
    // Generic structural projection. Calling this does not authenticate a world.
    // Negatives and coverage remain in the evidence envelope, never invented in
    // Scene. Distinct closed worlds are uncertain even if their intersection is empty.
    static GroundedControlProjection project(const RequestGrounding::Snapshot& snapshot,
        const std::map<int,std::string>& channels,uint64_t revision) {
        if(!snapshot.valid||snapshot.worlds.empty())return {{},"unavailable_reader_world"};
        std::set<std::string> predicates,names;
        for(const auto& [id,name]:channels)
            if(id<0||!concepts::symbol(name)||!predicates.insert(name).second)return {{},"invalid_channel_mapping"};
        auto atom_ok=[&](const auto& atoms){
            for(const auto& [a,r,b]:atoms) {
                if(!concepts::symbol(a)||!concepts::symbol(b)||!channels.count(r))return false;
                names.insert(a);names.insert(b);
            }
            return true;
        };
        for(const auto& world:snapshot.worlds) {
            if(!atom_ok(world.facts)||!atom_ok(world.negative))return {{},"invalid_reader_atom"};
            for(const auto& atom:world.negative)if(world.facts.count(atom))return {{},"conflicting_reader_atom"};
            for(const auto& [a,r]:world.closed_rows) {
                if(!concepts::symbol(a)||!channels.count(r))return {{},"invalid_reader_coverage"};names.insert(a);
            }
            for(const auto& [r,b]:world.closed_columns) {
                if(!concepts::symbol(b)||!channels.count(r))return {{},"invalid_reader_coverage"};names.insert(b);
            }
        }
        if(names.size()>32||channels.size()>128)return {{},"control_capacity"};
        auto common=snapshot.worlds.front().facts;bool closed=true;
        for(const auto& world:snapshot.worlds) {
            closed&=world.closed_world&&world.facts==snapshot.worlds.front().facts;
            EventGrounding::World intersection;
            std::set_intersection(common.begin(),common.end(),world.facts.begin(),world.facts.end(),std::inserter(intersection,intersection.begin()));
            common=std::move(intersection);
        }
        if(common.size()>256)return {{},"control_capacity"};
        relational::Scene scene;scene.complete=closed;
        for(const auto& [a,r,b]:common)scene.facts.push_back({channels.at(r),{a,b}});
        scene=relational::sorted(std::move(scene));
        return {relational::StampedScene{std::move(scene),revision},closed?"lossless_closed_world":"conservative_partial_world"};
    }
    // The caller owns the monotonic revision namespace of its control session;
    // a reader episode reset must not reuse old controller observation stamps.
    static GroundedControlProjection project(const GroundedRelationalEvidence& evidence,uint64_t revision) {
        if(evidence.names().size()>32)return {{},"control_capacity"};
        return project(evidence.alternatives(),evidence.channels(),revision);
    }
    static GroundedControlProjection project_current(const GroundedSession& session,
        const GroundedRelationalEvidence& evidence,uint64_t revision) {
        if(!current(session,evidence.stamp()))return {{},"stale_reader_evidence"};
        return project(evidence,revision);
    }
};
} // namespace laminar::reasoning
