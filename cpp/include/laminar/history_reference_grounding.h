#pragma once
// Ground references to ordered records using verified historical answers.
// Generic field filters and ranks contain no English reference vocabulary.
#include "revisable_grounding.h"

namespace laminar::reasoning {
struct HistoryReferenceGrounding {
    using Answer=LiveEvidenceGrounding::Answer;
    struct Record {
        uint64_t id=0;
        bool active=true;
        std::map<int,std::set<std::string>> fields;
    };
    struct Selector {
        std::vector<int> fields; // one field per reference argument; -1 ignores it
        int active=-1,rank=-99; // -1 any state; rank -99 all, >=0 from front, <0 from back
        auto key() const {return std::tie(fields,active,rank);}
        bool operator<(const Selector& s) const{return key()<s.key();}
        bool operator==(const Selector& s) const{return key()==s.key();}
        std::string signature() const {
            std::string out;for(int f:fields)out+=std::to_string(f)+",";
            return out+"/"+std::to_string(active)+"/"+std::to_string(rank);
        }
    };
    struct Evidence {
        std::string context,question,answer;
        std::set<Selector> supported;
    };
    struct Frame {
        std::vector<std::string> pattern;
        std::map<std::pair<uint64_t,uint64_t>,Evidence> evidence;
        std::set<Selector> candidates;
        std::set<uint64_t> scenes;
    };
    struct Resolution {
        std::set<uint64_t> candidates;
        bool resolved=false;
        std::string status;
    };
    std::map<std::string,Frame> frames;
    std::string status;
    size_t candidate_budget=4096,evaluation_budget=131072;

    static std::vector<Record> records(const RevisableGrounding& journal,const LanguageReasoner& reader,
                                        int mention_field=0,int source_field=1) {
        std::vector<Record> out;
        for(const auto& report:journal.reports) {
            Record r{report.id,report.active,{}};r.fields[source_field].insert(report.source);
            for(const auto& word:EventGrounding::words(report.text))if(reader.entities.count(word))r.fields[mention_field].insert(word);
            out.push_back(std::move(r));
        }
        return out;
    }
    static std::set<uint64_t> select(const std::vector<Record>& records,const Selector& rule,const std::vector<std::string>& args) {
        std::vector<uint64_t> matches;
        for(const auto& record:records) {
            bool match=rule.active<0||record.active==bool(rule.active);
            for(size_t i=0;i<rule.fields.size();++i)if(rule.fields[i]>=0) {
                auto field=record.fields.find(rule.fields[i]);
                match&=field!=record.fields.end()&&field->second.count(args.at(i));
            }
            if(match)matches.push_back(record.id);
        }
        if(rule.rank==-99)return {matches.begin(),matches.end()};
        const int index=rule.rank<0?int(matches.size())+rule.rank:rule.rank;
        return index>=0&&index<int(matches.size())?std::set<uint64_t>{matches[index]}:std::set<uint64_t>{};
    }
    static std::string context_signature(const std::vector<Record>& records) {
        std::string out;
        for(const auto& r:records) {
            out+=std::to_string(r.id)+":"+std::to_string(r.active)+":";
            for(const auto& [f,values]:r.fields){out+=std::to_string(f)+"=";for(const auto& v:values)out+=v+",";out+=';';}
            out+='\n';
        }
        return out;
    }
    static std::pair<std::vector<std::string>,std::vector<std::string>> bind(const std::string& reference,
                                        const std::vector<Record>& records,const LanguageReasoner& reader) {
        std::set<std::string> values(reader.names.begin(),reader.names.end());
        for(const auto& r:records)for(const auto& [key,v]:r.fields)values.insert(v.begin(),v.end());
        auto words=EventGrounding::words(reference);std::vector<std::string> args;
        for(auto& word:words)if(values.count(word)){args.push_back(word);word="$"+std::to_string(args.size()-1);}
        return {words,args};
    }
    std::set<Selector> hypotheses(size_t arity,const std::vector<Record>& records) const {
        std::set<int> fields{-1};for(const auto& r:records)for(const auto& [key,values]:r.fields)fields.insert(key);
        std::vector<std::vector<int>> combinations{{}};
        for(size_t i=0;i<arity;++i) {
            std::vector<std::vector<int>> next;
            for(const auto& old:combinations)for(int f:fields){auto v=old;v.push_back(f);next.push_back(std::move(v));}
            combinations=std::move(next);if(combinations.size()>candidate_budget)return {};
        }
        std::set<Selector> out;
        for(const auto& f:combinations)for(int active:{-1,0,1})for(int rank:{-99,-2,-1,0,1}) {
            out.insert({f,active,rank});if(out.size()>candidate_budget)return {};
        }
        return out;
    }
    static Answer consequence(LanguageReasoner reader,const std::string& question) {
        auto q=RevisableGrounding::question(reader,question);if(!q)return {};
        std::set<Answer> answers;
        for(const auto& w:reader.possible_worlds) {
            auto query=q->query;query.facts.clear();for(auto [a,r,b]:w.facts)query.facts.push_back({a,r,b});
            query.knowledge=std::make_shared<FactKnowledge>(w.knowledge);
            auto value=q->meaning.evaluate(reader.memory,query);
            bool unknown=!value.present()&&(reader.memory.missing_evidence||reader.memory.nonunique_result);
            if(!value.present()&&!unknown)return {};
            answers.insert({value,unknown,true});
        }
        if(answers.empty())return {};
        return answers.size()==1?*answers.begin():Answer{{},true,true};
    }
    // Feedback is a normal grounded answer at a referenced historical state;
    // no selected record ID, selector or correct order is a teaching input.
    bool teach(const LanguageReasoner& reader,RevisableGrounding& journal,const std::vector<Record>& records,
               const std::string& reference,const std::string& question,const std::string& answer,
               uint64_t scene,uint64_t evidence,bool generated=false) {
        if(generated){status="self_generated_feedback";return false;}
        if(!journal.consistent||!journal.guard(reader)){status="stale_history";return false;}
        auto [pattern,args]=bind(reference,records,reader);
        if(args.size()>2||pattern.empty()){status="unbound_reference";return false;}
        auto parsed=RevisableGrounding::question(reader,question);if(!parsed){status="unknown_question";return false;}
        auto wanted=RevisableGrounding::target(reader,answer,parsed->meaning.type());
        if(!wanted.valid){status="invalid_answer";return false;}
        const auto key=EventGrounding::join(pattern);
        auto context=context_signature(records)+RevisableGrounding::state_stamp(*journal.base)+journal.published.signature();
        for(const auto& r:journal.reports) {
            context+=std::to_string(r.id)+":"+std::to_string(r.revision)+":"+r.text+":"+r.source+":"+std::to_string(r.active)+"\n";
            for(const auto& [id,e]:r.evidence)context+=std::to_string(id)+":"+std::to_string(e.revision)+":"+e.question+":"+e.answer+":"+std::to_string(e.active)+"\n";
        }
        auto old=frames.find(key);auto id=std::make_pair(scene,evidence);
        if(old!=frames.end()&&old->second.evidence.count(id)) {
            const auto& e=old->second.evidence.at(id);
            bool same=e.context==context&&e.question==question&&e.answer==answer;
            status=same?"duplicate_feedback":"conflicting_feedback_id";return same;
        }
        auto candidates=hypotheses(args.size(),records);if(candidates.empty()){status="candidate_budget";return false;}
        std::map<uint64_t,Answer> cache;std::set<Selector> supported;size_t evaluations=0;
        for(const auto& rule:candidates) {
            const auto selected=select(records,rule,args);bool agrees=!selected.empty();
            for(uint64_t record:selected) {
                if(++evaluations>evaluation_budget){status="reference_search_budget";return false;}
                if(!cache.count(record)) {
                    auto historical=journal.at(record);if(!historical){status="missing_historical_state";return false;}
                    // Historical entity IDs may differ from the current vocabulary.
                    auto q=RevisableGrounding::question(*historical,question);
                    if(!q){cache[record]={};continue;}
                    auto value=consequence(*historical,question);
                    if(value.valid&&value.value.type==0) {
                        const auto name=historical->names.at(value.value.id);
                        if(!reader.entities.count(name))value={};else value.value.id=reader.entities.at(name);
                    }
                    cache[record]=value;
                }
                agrees&=cache.at(record)==wanted;
            }
            if(agrees)supported.insert(rule);
        }
        auto& frame=frames[key];frame.pattern=pattern;frame.evidence[id]={context,question,answer,supported};
        if(frame.evidence.size()==1)frame.candidates=std::move(supported);
        else std::erase_if(frame.candidates,[&](const Selector& rule){return !supported.count(rule);});
        frame.scenes.insert(scene);status=frame.candidates.empty()?"conflicting_reference_evidence":"learned";return true;
    }
    Resolution resolve(const LanguageReasoner& reader,const std::vector<Record>& records,const std::string& reference) const {
        auto [pattern,args]=bind(reference,records,reader);auto frame=frames.find(EventGrounding::join(pattern));
        if(frame==frames.end())return {{},false,"unknown_reference"};
        if(frame->second.scenes.size()<4)return {{},false,"insufficient_reference_evidence"};
        if(frame->second.candidates.empty())return {{},false,"conflicting_reference_evidence"};
        Resolution result;bool missing=false;
        for(const auto& rule:frame->second.candidates) {
            auto selected=select(records,rule,args);missing|=selected.empty();result.candidates.insert(selected.begin(),selected.end());
        }
        result.resolved=!missing&&result.candidates.size()==1;
        result.status=result.resolved?"resolved_reference":result.candidates.empty()?"missing_reference":"ambiguous_reference";
        return result;
    }
    std::string signature() const {
        std::string out;
        for(const auto& [key,frame]:frames){out+=key+":"+std::to_string(frame.scenes.size())+":";for(const auto& r:frame.candidates)out+=r.signature()+";";out+='\n';}
        return out;
    }
};

// The operation (replace/withdraw) and replacement payload are supplied by the
// caller's explicit correction envelope. Only the record reference is learned.
struct GroundedHistoryCorrection {
    struct Pending {
        uint64_t token,episode,observation,history;
        std::string model,context,state,text,source;
        bool active=true;
        std::map<uint64_t,uint64_t> candidates;
    };
    std::optional<Pending> pending;
    uint64_t next_token=1;
    std::string status;
    static void reply(LanguageReasoner& r,const std::string& text,const std::string& status) {
        r.prepared=false;r.reply=" "+text+"\n";r.reply_offset=0;r.status=status;
    }
    bool begin(LanguageReasoner& reader,RevisableGrounding& journal,const HistoryReferenceGrounding& model,
               const std::vector<HistoryReferenceGrounding::Record>& records,const std::string& reference,
               const std::string& replacement,const std::string& source,bool active=true,bool generated=false) {
        if(generated){status="self_generated_correction";return false;}
        if(!journal.guard(reader)){status=journal.status;return false;}
        auto resolved=model.resolve(reader,records,reference);pending.reset();status=resolved.status;
        if(resolved.candidates.empty()){reply(reader,"I cannot identify that report.",status);return false;}
        Pending p{next_token++,reader.episode_revision,reader.observation_revision,reader.history_revision,
            model.signature(),HistoryReferenceGrounding::context_signature(records),RevisableGrounding::state_stamp(reader),replacement,source,active,{}};
        for(uint64_t id:resolved.candidates){auto entry=journal.find(id);if(!entry){status="unknown_report";return false;}p.candidates[id]=entry->revision;}
        pending=std::move(p);
        if(resolved.resolved)return choose(reader,journal,model,records,pending->token,*resolved.candidates.begin());
        std::string text="Which report? Candidates:";for(auto [id,revision]:pending->candidates)text+=" "+std::to_string(id)+".";
        reply(reader,text,status);return false;
    }
    bool choose(LanguageReasoner& reader,RevisableGrounding& journal,const HistoryReferenceGrounding& model,
                const std::vector<HistoryReferenceGrounding::Record>& records,uint64_t token,uint64_t report,bool generated=false) {
        if(generated){status="self_generated_clarification";return false;}
        if(!pending||pending->token!=token){status="unknown_correction_token";return false;}
        const auto p=*pending;
        if(p.episode!=reader.episode_revision||p.observation!=reader.observation_revision||p.history!=reader.history_revision||
           p.model!=model.signature()||p.context!=HistoryReferenceGrounding::context_signature(records)||
           p.state!=RevisableGrounding::state_stamp(reader)||!journal.guard(reader)) {
            pending.reset();status="stale_correction";reply(reader,"That correction needs a fresh reference.",status);return false;
        }
        if(!p.candidates.count(report)){status="invalid_report_choice";return false;}
        if(!journal.revise_report(reader,report,p.candidates.at(report),p.text,p.source,p.active)) {
            status=journal.status;reply(reader,"The report could not be revised.",status);return false;
        }
        pending.reset();status=journal.consistent?"revised_report":journal.status;
        reply(reader,journal.consistent?"The report has been revised.":"The remaining evidence conflicts.",status);return true;
    }
};
} // namespace laminar::reasoning
