#pragma once
// Learn a correction construction from verified post-correction answers.
// Supplied primitives are KEEP/WITHDRAW/REPLACE and typed span substitution;
// no correction verbs, delimiters, argument offsets or target IDs are supplied.
#include "history_reference_grounding.h"

namespace laminar::reasoning {
struct CorrectionLanguageGrounding {
    enum Operation {KEEP=0,WITHDRAW=1,REPLACE=2};
    struct Parse {
        std::string frame,reference,report;
        auto key() const {return std::tie(frame,reference,report);}
        bool operator<(const Parse& p) const {return key()<p.key();}
    };
    struct Example {
        std::string context;
        std::set<int> supported;
    };
    struct Frame {
        std::set<int> candidates;
        std::set<uint64_t> scenes;
        std::map<std::pair<uint64_t,uint64_t>,Example> evidence;
    };
    struct Plan {
        int operation=KEEP;
        std::string reference,report;
        auto key() const {return std::tie(operation,reference,report);}
        bool operator<(const Plan& p) const {return key()<p.key();}
    };
    struct Meaning {std::set<Plan> plans;std::string status;};
    using QA=std::pair<std::string,std::string>;
    std::map<std::string,Frame> frames;
    std::string status;
    size_t word_budget=64,span_budget=4096,simulation_budget=256;

    static std::string span(const std::vector<std::string>& words,size_t start,size_t end) {
        std::string out;for(size_t i=start;i<end;++i){if(!out.empty())out+=' ';out+=words[i];}return out;
    }
    static bool report_span(const LanguageReasoner& r,const std::string& text) {
        std::set<std::string> known(r.names.begin(),r.names.end());
        const auto observations=r.observation_language.alternatives(text,known);
        if(observations.matched) {
            if(observations.conflicted||observations.values.empty())return false;
            for(const auto& v:observations.values)for(const auto& a:v.arguments)if(!known.count(a))return false;
            return true;
        }
        const auto events=r.events.alternatives(text);
        if(events.conflicted||events.values.empty())return false;
        for(const auto& v:events.values)for(const auto& a:v.mentions)if(!known.count(a))return false;
        return true;
    }
    std::set<Parse> parses(const LanguageReasoner& r,const HistoryReferenceGrounding& references,
                         const std::vector<HistoryReferenceGrounding::Record>& records,const std::string& text,
                         bool& over_budget) const {
        struct Span {size_t first,last;std::string text;};
        over_budget=false;const auto words=EventGrounding::words(text);
        if(words.size()>word_budget){over_budget=true;return {};}
        std::vector<Span> refs,reports;size_t evaluations=0;
        for(size_t a=0;a<words.size();++a)for(size_t b=a+1;b<=words.size();++b) {
            if(++evaluations>span_budget){over_budget=true;return {};}
            auto candidate=span(words,a,b);auto bound=HistoryReferenceGrounding::bind(candidate,records,r);
            auto ref=references.frames.find(EventGrounding::join(bound.first));
            if(ref!=references.frames.end()&&ref->second.scenes.size()>=4&&!ref->second.candidates.empty())refs.push_back({a,b,candidate});
            if(report_span(r,candidate))reports.push_back({a,b,candidate});
        }
        std::set<std::string> values(r.names.begin(),r.names.end());
        for(const auto& record:records)for(const auto& [field,v]:record.fields)values.insert(v.begin(),v.end());
        std::set<Parse> out;
        for(const auto& ref:refs) {
            auto add=[&](const Span* report) {
                if(++evaluations>span_budget){over_budget=true;return;}
                std::vector<std::string> pattern;
                for(size_t i=0;i<words.size();) {
                    if(i==ref.first){pattern.push_back("@reference");i=ref.last;}
                    else if(report&&i==report->first){pattern.push_back("@report");i=report->last;}
                    else {if(values.count(words[i]))return;pattern.push_back(words[i++]);}
                }
                out.insert({EventGrounding::join(pattern),ref.text,report?report->text+".":""});
            };
            add(nullptr);
            for(const auto& report:reports)if(report.last<=ref.first||report.first>=ref.last)add(&report);
        }
        if(over_budget)return {};return out;
    }
    static std::string context(const LanguageReasoner& r,const RevisableGrounding& journal,
                               const HistoryReferenceGrounding& refs,const std::vector<HistoryReferenceGrounding::Record>& records,
                               const std::string& text,const std::vector<QA>& answers) {
        std::string out=refs.signature()+RevisableGrounding::state_stamp(r)+text+'\n';
        out+=HistoryReferenceGrounding::context_signature(records);
        if(journal.base)out+=RevisableGrounding::state_stamp(*journal.base);
        out+=std::to_string(r.episode_revision)+":"+std::to_string(r.observation_revision)+":"+std::to_string(r.history_revision)+'\n';
        for(const auto& record:journal.reports) {
            out+=std::to_string(record.id)+":"+std::to_string(record.revision)+":"+std::to_string(record.active)+":"+record.source+":"+record.text+'\n';
            for(const auto& [id,e]:record.evidence)out+=std::to_string(id)+":"+std::to_string(e.revision)+":"+std::to_string(e.active)+":"+e.source+":"+e.question+":"+e.answer+'\n';
        }
        out+=r.events.signature()+r.observation_language.signature()+r.grounding.signature();
        std::ostringstream prior;prior<<std::hexfloat;
        for(const auto& [form,program]:r.forms)prior<<form<<':'<<program<<';';
        for(const auto& [cue,program]:r.memory.procedures){prior<<cue<<':';for(int op:program)prior<<op<<',';prior<<';';}
        for(float w:r.memory.skills.kernel.routes.W)prior<<w<<',';
        out+=prior.str();
        for(const auto& [q,a]:answers)out+=q+":"+a+'\n';
        return out;
    }
    bool teach(const LanguageReasoner& reader,RevisableGrounding& journal,const HistoryReferenceGrounding& references,
               const std::vector<HistoryReferenceGrounding::Record>& records,const std::string& utterance,
               const std::vector<QA>& answers,uint64_t scene,uint64_t evidence,bool generated=false) {
        if(generated){status="self_generated_feedback";return false;}
        if(!journal.consistent||!journal.guard(reader)){status="stale_history";return false;}
        if(answers.empty()){status="missing_correction_answers";return false;}
        for(const auto& [q,a]:answers)if(!journal.valid_evidence(reader,q,a)){status=journal.status;return false;}
        bool over=false;auto candidates=parses(reader,references,records,utterance,over);
        if(over){status="correction_span_budget";return false;}
        if(candidates.empty()){status="unbound_correction";return false;}
        // Aggregate alternative parses of one construction by existential support;
        // do not silently choose whichever span enumeration visits first.
        std::map<std::string,std::set<int>> supported;size_t simulations=0;
        for(const auto& p:candidates) {
            auto ref=references.resolve(reader,records,p.reference);
            if(!ref.resolved){status="ambiguous_teaching_reference";return false;}
            auto id=*ref.candidates.begin();const auto* record=journal.find(id);
            if(!record){status="unknown_correction_report";return false;}
            supported.try_emplace(p.frame);
            for(int op:{KEEP,WITHDRAW,REPLACE}) {
                if(op==REPLACE&&p.report.empty())continue;
                if(++simulations>simulation_budget){status="correction_simulation_budget";return false;}
                auto world=reader;auto history=journal;
                if(op!=KEEP&&(!history.revise_report(world,id,record->revision,p.report,record->source,op==REPLACE)||!history.consistent))continue;
                bool agrees=true;
                for(const auto& [q,a]:answers) {
                    auto question=RevisableGrounding::question(world,q);
                    if(!question){agrees=false;break;}
                    const auto target=RevisableGrounding::target(world,a,question->meaning.type());
                    agrees&=HistoryReferenceGrounding::consequence(world,q)==target;
                }
                if(agrees)supported[p.frame].insert(op);
            }
        }
        const auto stamp=context(reader,journal,references,records,utterance,answers);
        const auto id=std::make_pair(scene,evidence);
        // Check every frame before publishing any evidence from this example.
        bool all_duplicate=true;
        for(const auto& [key,ops]:supported) {
            auto old=frames.find(key);
            if(old==frames.end()||!old->second.evidence.count(id)){all_duplicate=false;continue;}
            if(old->second.evidence.at(id).context!=stamp){status="conflicting_feedback_id";return false;}
        }
        if(all_duplicate){status="duplicate_feedback";return true;}
        for(const auto& [key,ops]:supported) {
            auto& f=frames[key];if(f.evidence.count(id))continue;
            if(f.evidence.empty())f.candidates=ops;
            else std::erase_if(f.candidates,[&](int op){return !ops.count(op);});
            f.evidence[id]={stamp,ops};f.scenes.insert(scene);
        }
        status="learned";return true;
    }
    Meaning meaning(const LanguageReasoner& reader,const HistoryReferenceGrounding& references,
                    const std::vector<HistoryReferenceGrounding::Record>& records,const std::string& utterance) const {
        Meaning out;bool over=false,untrusted=false;
        const auto candidates=parses(reader,references,records,utterance,over);
        if(over){out.status="correction_span_budget";return out;}
        for(const auto& p:candidates) {
            auto f=frames.find(p.frame);if(f==frames.end())continue;
            if(f->second.scenes.size()<4||f->second.candidates.empty()){untrusted=true;continue;}
            for(int op:f->second.candidates)out.plans.insert({op,p.reference,op==REPLACE?p.report:""});
        }
        if(untrusted){out.plans.clear();out.status="unsupported_correction";}
        else out.status=out.plans.empty()?"unknown_correction":out.plans.size()==1?"grounded_correction":"ambiguous_correction_meaning";
        return out;
    }
    std::string signature() const {
        std::string out;for(const auto& [key,f]:frames){out+=key+":"+std::to_string(f.scenes.size())+":";for(int op:f.candidates)out+=std::to_string(op)+",";out+='\n';}return out;
    }
};

struct GroundedCorrectionDialogue {
    GroundedHistoryCorrection correction;
    std::string language_snapshot,status;
    bool begin(LanguageReasoner& reader,RevisableGrounding& journal,const HistoryReferenceGrounding& refs,
               const CorrectionLanguageGrounding& language,const std::vector<HistoryReferenceGrounding::Record>& records,
               const std::string& utterance,const std::string& source,bool generated=false) {
        if(generated){status="self_generated_correction";return false;}
        if(!journal.guard(reader)){status=journal.status;return false;}
        correction.pending.reset();auto meaning=language.meaning(reader,refs,records,utterance);status=meaning.status;
        if(meaning.plans.size()!=1){GroundedHistoryCorrection::reply(reader,"I cannot interpret that correction.",status);return false;}
        const auto& plan=*meaning.plans.begin();language_snapshot=language.signature();
        if(plan.operation==CorrectionLanguageGrounding::KEEP) {
            if(!refs.resolve(reader,records,plan.reference).resolved){status="unresolved_keep_reference";GroundedHistoryCorrection::reply(reader,"I cannot identify that report.",status);return false;}
            status="kept_report";GroundedHistoryCorrection::reply(reader,"The report is unchanged.",status);return false;
        }
        bool applied=correction.begin(reader,journal,refs,records,plan.reference,plan.report,source,plan.operation==CorrectionLanguageGrounding::REPLACE);
        status=correction.status;return applied;
    }
    bool choose(LanguageReasoner& reader,RevisableGrounding& journal,const HistoryReferenceGrounding& refs,
                const CorrectionLanguageGrounding& language,const std::vector<HistoryReferenceGrounding::Record>& records,
                uint64_t token,uint64_t record,bool generated=false) {
        if(generated){status="self_generated_clarification";return false;}
        if(language.signature()!=language_snapshot) {
            correction.pending.reset();status="stale_correction_language";GroundedHistoryCorrection::reply(reader,"That correction needs a fresh reference.",status);return false;
        }
        bool applied=correction.choose(reader,journal,refs,records,token,record);status=correction.status;return applied;
    }
};
} // namespace laminar::reasoning
