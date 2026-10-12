#pragma once
// Retain raw reports and identified teaching evidence. Rebuild meanings and
// dependent knowledge after an explicit correction, without resetting dialogue.
// Source identities are supplied by the trusted caller, not inferred from text.
#include "live_evidence_grounding.h"

namespace laminar::reasoning {
struct RevisableGrounding {
    using Rule=ObservationLanguage::Rule;
    using World=LanguageReasoner::PossibleWorld;
    using Answer=LiveEvidenceGrounding::Answer;
    struct Evidence {
        std::string question,answer,source;
        uint64_t revision=0;
        bool active=true;
    };
    struct Report {
        uint64_t id=0,revision=0;
        std::string text,source;
        bool active=true;
        std::map<uint64_t,Evidence> evidence;
        uint64_t learning_scene=0;
        // An archive boundary clears physical evidence while retaining shared
        // language choices. It is host protocol, never an inferred report kind.
        bool reset_before=false;
    };
    struct Change {
        std::string kind,source,payload;
        uint64_t report=0,evidence=0,revision=0;
    };
    struct View {
        std::vector<std::string> names;
        std::vector<World> worlds;
    };
    std::optional<LanguageReasoner> base;
    std::vector<Report> reports;
    std::vector<Change> changes;
    std::map<uint64_t,View> history;
    ObservationLanguage published;
    // Only frames acquired by this host may keep learning across episode bases.
    // Their earlier evidence is consolidated; current-episode evidence remains
    // retractable by rebuilding from the episode's original candidates.
    std::set<std::string> trainable_frames;
    std::string expected_state,status;
    uint64_t observation_revision=0,history_revision=0;
    size_t report_budget=2048,branch_budget=4096,history_budget=16384,evaluation_budget=1048576;
    size_t evaluations=0;
    bool consistent=false;

    static std::string state_stamp(const LanguageReasoner& r) {
        std::ostringstream out;
        for(const auto& n:r.names)out<<n<<',';
        out<<'|'<<r.complete<<':'<<r.event_issue<<'|';
        auto knowledge=[&](const FactKnowledge& k) {
            for(auto [a,b,c]:k.positive)out<<"p"<<a<<','<<b<<','<<c<<';';
            for(auto [a,b,c]:k.negative)out<<"n"<<a<<','<<b<<','<<c<<';';
            for(auto [a,b]:k.closed_rows)out<<"r"<<a<<','<<b<<';';
            for(auto [a,b]:k.closed_columns)out<<"c"<<a<<','<<b<<';';
        };
        for(auto f:r.facts)out<<f.subject<<','<<f.relation<<','<<f.object<<';';
        knowledge(r.observations);
        for(const auto& w:r.possible_worlds) {
            out<<'|';knowledge(w.knowledge);
            for(auto [a,b,c]:w.facts)out<<a<<','<<b<<','<<c<<';';
            for(const auto& [k,v]:w.meanings)out<<k<<'='<<v<<';';
        }
        return out.str();
    }
    bool start(const LanguageReasoner& r) {
        if(base){status="journal_already_started";return false;}
        if(!r.environment_dialogue||!r.partial_observation||r.grounded_references||!r.request_snapshot().valid) {
            status="invalid_journal_base";return false;
        }
        base=r;published=r.observation_language;expected_state=state_stamp(r);
        observation_revision=r.observation_revision;history_revision=r.history_revision;
        consistent=true;status="journal_started";return true;
    }
    bool guard(const LanguageReasoner& r,bool next_report=false) {
        if(!base){status="missing_journal";return false;}
        const auto& b=*base;
        if(r.episode_revision!=b.episode_revision||r.history_revision!=history_revision||
           r.observation_revision!=observation_revision+uint64_t(next_report)) {
            status="stale_journal";return false;
        }
        if(!r.partial_observation||!r.environment_dialogue||r.grounded_references||
           r.events.signature()!=b.events.signature()||r.observation_language.signature()!=published.signature()||
           r.forms!=b.forms||r.memory.procedures!=b.memory.procedures||r.grounding.signature()!=b.grounding.signature()||
           r.memory.skills.kernel.routes.W!=b.memory.skills.kernel.routes.W||r.relations[0]!=b.relations[0]||r.relations[1]!=b.relations[1]) {
            status="changed_journal_prior";return false;
        }
        if(!next_report&&state_stamp(r)!=expected_state){status="unrecorded_state_change";return false;}
        return true;
    }
    Report* find(uint64_t id) {
        for(auto& r:reports)if(r.id==id)return &r;
        return nullptr;
    }
    static std::optional<LiveEvidenceGrounding::Question> question(const LanguageReasoner& r,const std::string& text) {
        auto copy=r;copy.complete=true;copy.event_issue.clear();copy.awaiting_action_observation=false;
        copy.world_model_signature=copy.events.signature();copy.observation_language_signature=copy.observation_language.signature();
        copy.question=text;copy.prepare();auto meaning=copy.question_meaning();
        if(meaning.type()<0||!copy.query.a.present())return {};
        return LiveEvidenceGrounding::Question{copy.query,meaning};
    }
    static Answer target(const LanguageReasoner& r,const std::string& text,int type) {
        const auto raw=LanguageReasoner::trim(text);const auto words=EventGrounding::words(raw);
        if(words.size()!=1)return {};
        const auto& w=words[0];
        if(w=="unknown")return {{},true,true};
        if(type==1&&(w=="yes"||w=="no"))return {{1,w=="yes"},false,true};
        if(type==2&&!raw.empty()&&raw.size()<9&&std::all_of(raw.begin(),raw.end(),[](unsigned char c){return std::isdigit(c);}))return {{2,std::stoi(raw)},false,true};
        if(type==0&&r.entities.count(w))return {{0,r.entities.at(w)},false,true};
        return {};
    }
    bool valid_evidence(const LanguageReasoner& r,const std::string& q,const std::string& a) {
        auto parsed=question(r,q);
        if(!parsed){status="unknown_question";return false;}
        if(!target(r,a,parsed->meaning.type()).valid){status="invalid_answer";return false;}
        return true;
    }
    // Called only after one real ENVIRONMENT line has passed through the reader.
    // True means the record was accepted, including contradictory evidence.
    // consistent/status report whether its consequences can be used.
    bool append(LanguageReasoner& r,uint64_t id,const std::string& source,bool generated=false,std::optional<uint64_t> learning_scene={}) {
        if(generated){status="self_generated_report";return false;}
        if(!guard(r,true))return false;
        if(!r.last_observation_environment||r.last_observation_text.empty()||source.empty()){status="invalid_report_source";return false;}
        if(find(id)){status="duplicate_report_id";return false;}
        if(reports.size()>=report_budget){status="report_budget";return false;}
        reports.push_back({id,0,r.last_observation_text,source,true,{},learning_scene.value_or(id)});
        changes.push_back({"report",source,r.last_observation_text,id,0,0});
        observation_revision=r.observation_revision;
        rebuild(r);return true;
    }
    bool feedback(LanguageReasoner& r,uint64_t report,uint64_t id,const std::string& q,const std::string& a,
                  const std::string& source,bool generated=false) {
        if(generated){status="self_generated_feedback";return false;}
        if(!guard(r))return false;
        auto entry=find(report);
        if(!entry||!entry->active||source.empty()){status="unknown_report";return false;}
        auto old=entry->evidence.find(id);
        if(old!=entry->evidence.end()) {
            bool same=old->second.active&&old->second.question==q&&old->second.answer==a&&old->second.source==source;
            status=same?"duplicate_feedback":"conflicting_feedback_id";return same;
        }
        if(!valid_evidence(r,q,a))return false;
        entry->evidence[id]={q,a,source,0,true};changes.push_back({"feedback",source,a,report,id,0});
        if(report!=reports.back().id)r.history_revision=++history_revision;
        rebuild(r);return true;
    }
    bool revise_feedback(LanguageReasoner& r,uint64_t report,uint64_t id,uint64_t previous_revision,
                         const std::string& answer,const std::string& source,bool active=true,bool generated=false) {
        if(generated){status="self_generated_feedback";return false;}
        if(!guard(r))return false;
        auto entry=find(report);
        if(!entry||!entry->evidence.count(id)){status="unknown_feedback";return false;}
        auto& e=entry->evidence.at(id);
        if(e.revision!=previous_revision||e.source!=source){status="stale_evidence_revision";return false;}
        if(active&&!valid_evidence(r,e.question,answer))return false;
        if(active)e.answer=answer;e.active=active;++e.revision;
        changes.push_back({active?"revise_feedback":"withdraw_feedback",source,answer,report,id,e.revision});
        r.history_revision=++history_revision;rebuild(r);return true;
    }
    bool revise_report(LanguageReasoner& r,uint64_t report,uint64_t previous_revision,const std::string& text,
                       const std::string& source,bool active=true,bool generated=false) {
        if(generated){status="self_generated_report";return false;}
        if(!guard(r))return false;
        auto entry=find(report);
        if(!entry){status="unknown_report";return false;}
        if(entry->revision!=previous_revision||entry->source!=source){status="stale_report_revision";return false;}
        if(active&&(text.empty()||text.size()>4096)){status="invalid_report";return false;}
        // Withdrawal changes evidential status, not the retained utterance.
        // References to inactive records still need their original content.
        if(active)entry->text=text;entry->active=active;++entry->revision;
        changes.push_back({active?"revise_report":"withdraw_report",source,text,report,0,entry->revision});
        r.history_revision=++history_revision;rebuild(r);return true;
    }
    bool replay(LanguageReasoner& r) {return guard(r)&&rebuild(r);}
private:
    bool rebuild(LanguageReasoner& r) {
        evaluations=0;consistent=false;history.clear();
        auto work=*base;work.awaiting_action_observation=false;
        if(work.possible_worlds.empty()) {
            World w;w.knowledge=work.observations;w.facts=w.knowledge.positive;work.possible_worlds.push_back(w);
        }
        std::map<std::string,ObservationLanguage::Frame> learned;size_t historical_worlds=0;
        ObservationLanguage adaptable;
        for(const auto& key:trainable_frames)if(base->observation_language.frames.count(key))
            adaptable.frames[key]=base->observation_language.frames.at(key);
        auto fail=[&](const std::string& why) {
            status=why;r.complete=false;r.event_issue=why;r.prepared=false;r.reply.clear();r.reply_offset=0;r.status=why;
            history.clear();expected_state=state_stamp(r);return false;
        };
        for(const auto& entry:reports) {
            if(entry.reset_before) {
                std::set<World> reset;
                for(auto world:work.possible_worlds){world.knowledge={};world.facts.clear();reset.insert(std::move(world));}
                work.possible_worlds.assign(reset.begin(),reset.end());work.project_worlds();
            }
            if(entry.active) {
                struct Option {Rule rule;std::vector<int> args;std::string frame,choice;};
                std::vector<Option> options;
                auto known=base->observation_language.alternatives(entry.text,{work.names.begin(),work.names.end()});
                // A partially taught frame still needs new evidence before its
                // fourth scene. It is available to teaching replay, not ordinary
                // inference, which retains the four-scene admission threshold.
                if(!known.matched)known=adaptable.alternatives(entry.text,{work.names.begin(),work.names.end()},0);
                if(!known.matched&&!learned.empty()) {
                    ObservationLanguage prefix;prefix.frames=learned;
                    known=prefix.alternatives(entry.text,{work.names.begin(),work.names.end()},0);
                }
                EventGrounding::Interpretation event;
                std::string induced;
                if(known.matched) {
                    if(known.conflicted)return fail("conflicting_prior_observation");
                    for(const auto& v:known.values) {
                        if(trainable_frames.count(v.frame)||learned.count(v.frame)) {
                            if(!learned.count(v.frame))learned[v.frame]=base->observation_language.frames.at(v.frame);
                            auto it=learned.find(v.frame);
                            for(const auto& [id,e]:entry.evidence)if(e.active)it->second.scenes.insert(entry.learning_scene);
                        }
                        std::vector<int> args;for(const auto& n:v.arguments)args.push_back(work.entity(n));
                        options.push_back({v.rule,args,"observation:"+v.frame,v.rule.signature()});
                    }
                } else if(base->events.interpret(entry.text,event)) {
                    int a=work.entity(event.subject),b=work.entity(event.object);
                    for(const auto& n:event.mentions)work.entity(n);
                    options.push_back({{7+event.effect,event.relation,0,2},{a,b},event.frame,event.choice});
                } else {
                    auto pattern=EventGrounding::words(entry.text);std::vector<int> args;
                    for(auto& word:pattern)if(work.entities.count(word)){args.push_back(work.entities.at(word));word="$"+std::to_string(args.size()-1);}
                    if(args.empty()||args.size()>4)return fail("unbound_report");
                    induced=EventGrounding::join(pattern);auto [it,created]=learned.try_emplace(induced);
                    if(created){it->second.pattern=pattern;it->second.candidates=ObservationLanguage::hypotheses(args.size(),{work.relations[0],work.relations[1]},true);}
                    for(const auto& [id,e]:entry.evidence)if(e.active)it->second.scenes.insert(entry.learning_scene);
                    for(const auto& rule:it->second.candidates)options.push_back({rule,args,"observation:"+induced,rule.signature()});
                }
                std::set<World> next;
                for(const auto& world:work.possible_worlds)for(const auto& o:options) {
                    if(++evaluations>evaluation_budget)return fail("replay_evaluation_budget");
                    auto choice=world.meanings.find(o.frame);
                    if(choice!=world.meanings.end()&&choice->second!=o.choice)continue;
                    auto candidate=world;
                    if(!ObservationLanguage::apply(candidate.knowledge,o.rule,o.args))continue;
                    candidate.facts=candidate.knowledge.positive;candidate.meanings[o.frame]=o.choice;
                    next.insert(std::move(candidate));
                    if(next.size()>branch_budget)return fail("replay_branch_budget");
                }
                if(next.empty())return fail("conflicting_report_history");
                work.possible_worlds.assign(next.begin(),next.end());work.project_worlds();
                for(const auto& [id,e]:entry.evidence)if(e.active) {
                    auto q=question(work,e.question);
                    if(!q)return fail("unbound_historical_question");
                    auto wanted=target(work,e.answer,q->meaning.type());
                    if(!wanted.valid)return fail("unbound_historical_answer");
                    std::vector<World> surviving;
                    for(const auto& candidate:work.possible_worlds) {
                        if(++evaluations>evaluation_budget)return fail("replay_evaluation_budget");
                        auto query=q->query;query.facts.clear();for(auto [a,b,c]:candidate.facts)query.facts.push_back({a,b,c});
                        query.knowledge=std::make_shared<FactKnowledge>(candidate.knowledge);
                        auto value=q->meaning.evaluate(work.memory,query);
                        bool unknown=!value.present()&&(work.memory.missing_evidence||work.memory.nonunique_result);
                        if(Answer{value,unknown,value.present()||unknown}==wanted)surviving.push_back(candidate);
                    }
                    if(surviving.empty())return fail("conflicting_journal_evidence");
                    work.possible_worlds=std::move(surviving);work.project_worlds();
                }
            }
            historical_worlds+=work.possible_worlds.size();
            if(historical_worlds>history_budget)return fail("replay_history_budget");
            history[entry.id]={work.names,work.possible_worlds};
        }
        if(work.possible_worlds.size()>r.world_budget)return fail("world_budget");
        // Later constraints also refine earlier views through consistent frame
        // choices. They never overwrite a later physical event with an old fact.
        for(auto& [id,view]:history) {
            std::vector<World> surviving;
            for(const auto& earlier:view.worlds) {
                bool supported=false;
                for(const auto& final:work.possible_worlds) {
                    if(++evaluations>evaluation_budget)return fail("replay_evaluation_budget");
                    bool compatible=true;
                    for(const auto& [key,choice]:earlier.meanings) {
                        auto it=final.meanings.find(key);compatible&=it!=final.meanings.end()&&it->second==choice;
                    }
                    if(compatible){supported=true;break;}
                }
                if(supported)surviving.push_back(earlier);
            }
            view.worlds=std::move(surviving);
        }
        auto model=base->observation_language;
        for(auto& [key,frame]:learned) {
            std::set<std::string> supported;bool unselected=false;
            for(const auto& w:work.possible_worlds) {
                auto meaning=w.meanings.find("observation:"+key);
                if(meaning==w.meanings.end())unselected=true;else supported.insert(meaning->second);
            }
            if(!unselected)std::erase_if(frame.candidates,[&](const Rule& rule){return !supported.count(rule.signature());});
            model.frames[key]=frame;
        }
        published=std::move(model);r.observation_language=published;r.entities=work.entities;r.names=work.names;
        r.possible_worlds=std::move(work.possible_worlds);r.project_worlds();r.complete=true;r.event_issue.clear();
        r.world_model_signature=r.events.signature();r.observation_language_signature=published.signature();
        r.prepared=false;r.reply.clear();r.reply_offset=0;r.status="replayed_evidence";
        expected_state=state_stamp(r);consistent=true;status="replayed_evidence";return true;
    }
public:
    std::optional<LanguageReasoner> at(uint64_t report) const {
        if(!consistent||!history.count(report))return {};
        auto reader=*base;const auto& view=history.at(report);
        reader.names=view.names;reader.entities.clear();for(size_t i=0;i<reader.names.size();++i)reader.entities[reader.names[i]]=i;
        reader.possible_worlds=view.worlds;reader.project_worlds();reader.observation_language=published;
        reader.world_model_signature=reader.events.signature();reader.observation_language_signature=published.signature();
        reader.complete=true;reader.event_issue.clear();reader.awaiting_action_observation=false;reader.prepared=false;reader.reply.clear();reader.reply_offset=0;
        return reader;
    }
};
} // namespace laminar::reasoning
