#pragma once
// Verified answers travel separately from transcript bytes. Offsets refer only
// to bytes committed from the training file, excluding generated/probe bytes.
#include "grounded_bootstrap.h"
#include <limits>

namespace laminar::reasoning {
struct GroundedFeedbackStream {
    struct Row {
        uint64_t offset,episode,report,report_revision,evidence,previous_revision;
        GroundedSession::FeedbackKind kind;
        std::string source,question,answer;
    };
    std::vector<Row> rows;
    size_t cursor=0;
    static uint64_t number(const std::string& text) {
        auto value=GroundedSession::choice(text);
        GroundedBootstrap::require(bool(value),"invalid feedback integer");return *value;
    }
    void load(const std::string& path) {
        auto input=GroundedBootstrap::open(path);
        while(input.peek()!=EOF) {
            auto f=GroundedBootstrap::fields(GroundedBootstrap::line(input));
            GroundedBootstrap::require(f.size()==10,"feedback requires ten tab-separated fields");
            auto kind=f[6]=="add"?GroundedSession::ADD_FEEDBACK:f[6]=="revise"?GroundedSession::REVISE_FEEDBACK:GroundedSession::WITHDRAW_FEEDBACK;
            GroundedBootstrap::require(f[6]=="add"||f[6]=="revise"||f[6]=="withdraw","invalid feedback operation");
            Row row{number(f[0]),number(f[1]),number(f[2]),number(f[3]),number(f[4]),number(f[5]),kind,f[7],f[8],f[9]};
            GroundedBootstrap::require(row.offset>0&&(rows.empty()||row.offset>=rows.back().offset),"feedback offsets must be positive and ordered");
            GroundedBootstrap::require(kind!=GroundedSession::ADD_FEEDBACK||row.previous_revision==0,"new feedback cannot have a prior revision");
            GroundedBootstrap::require(!row.source.empty()&&!row.question.empty()&&!row.answer.empty(),"empty feedback field");
            GroundedBootstrap::require(rows.size()<1048576,"feedback row budget");rows.push_back(std::move(row));
        }
    }
    uint64_t next_offset() const {return cursor<rows.size()?rows[cursor].offset:std::numeric_limits<uint64_t>::max();}
    void deliver(uint64_t offset,GroundedSession& session) {
        while(cursor<rows.size()&&rows[cursor].offset<=offset) {
            const auto& row=rows[cursor];
            GroundedBootstrap::require(row.offset==offset,"skipped feedback byte offset");
            bool accepted=false;
            if(session.resets==row.episode) {
                auto anchor=session.feedback_anchor(row.report);
                GroundedBootstrap::require(anchor&&anchor->revision==row.report_revision,"feedback report mismatch");
                accepted=session.teach(*anchor,row.evidence,row.question,row.answer,row.source,row.kind,row.previous_revision);
            } else {
                auto anchor=session.archived_anchor(row.episode,row.report);
                GroundedBootstrap::require(anchor&&anchor->report_revision==row.report_revision,"archived feedback report mismatch");
                accepted=session.teach_archived(*anchor,row.evidence,row.question,row.answer,row.source,row.kind,row.previous_revision);
            }
            GroundedBootstrap::require(accepted,"feedback rejected: "+session.teaching_status);
            ++cursor;
        }
    }
};
} // namespace laminar::reasoning
