#pragma once
// Extend a learned language model from raw stories and verified answers,
// including an explicitly verified unknown target. No coverage/operator labels
// or structured knowledge states are teaching inputs.
#include "incremental_grounding.h"

namespace laminar::reasoning {
struct ObservationLanguageLearner {
    using Episode=BootstrapGrounding::Episode;
    using Rule=ObservationLanguage::Rule;
    struct Sentence {
        bool event=false;EventGrounding::Interpretation known;
        std::string frame;std::vector<std::string> args;
    };
    struct Lesson {
        std::vector<Sentence> story;std::vector<std::string> names,args;
        std::string form;Value answer;bool unknown=false;uint64_t scene=0;
    };
    ObservationLanguage model;
    std::map<std::string,std::set<Program>> questions;
    std::string status,prior_events,prior_observations,prior_grammar;
    std::map<std::string,int> prior_forms;std::map<int,Program> prior_programs;
    std::vector<float> prior_weights;std::array<int,2> relations{};
    size_t accepted=0,rejected=0,checked=0,skipped=0,assignment_budget=131072;
    int rounds=0;
    bool fit(const std::vector<Episode>& examples,LanguageReasoner& prior) {
        model=prior.observation_language;questions.clear();accepted=rejected=checked=skipped=0;rounds=0;
        status="insufficient_evidence";
        prior_events=prior.events.signature();prior_observations=prior.observation_language.signature();prior_grammar=prior.grounding.signature();
        prior_forms=prior.forms;prior_programs=prior.memory.procedures;prior_weights=prior.memory.skills.kernel.routes.W;
        relations={prior.relations[0],prior.relations[1]};
        IncrementalGrounding lex;lex.lexicalize(examples,prior);lex.surface.lexicon.erase("unknown");
        for(const auto& e:examples)for(const auto& s:e.observations)
            for(const auto& parsed:prior.observation_language.alternatives(s).values)
                lex.surface.lexicon.insert(parsed.arguments.begin(),parsed.arguments.end());
        std::vector<Lesson> lessons;std::map<std::string,int> types;
        for(const auto& e:examples) {
            if(e.self_generated||e.answer.empty()){++rejected;continue;}
            auto q=lex.symbolize(e.question);Lesson l;l.form=q.frame;l.args=q.arguments;l.scene=e.id;
            bool valid=!q.arguments.empty()&&q.arguments.size()<=2;std::set<std::string> names;
            for(const auto& text:e.observations) {
                Sentence s;
                if(prior.events.interpret(text,s.known)) {
                    s.event=true;s.args={s.known.subject,s.known.object};
                    names.insert(s.known.mentions.begin(),s.known.mentions.end());
                } else {
                    // A contradicted or ambiguous old frame is not a new word.
                    if(prior.events.alternatives(text).matched){valid=false;break;}
                    auto bound=lex.symbolize(text);s.frame=bound.frame;s.args=bound.arguments;
                    if(s.args.empty()||s.args.size()>4){valid=false;break;}
                }
                names.insert(s.args.begin(),s.args.end());l.story.push_back(std::move(s));
            }
            for(const auto& arg:l.args)valid&=names.count(arg)>0;
            l.names.assign(names.begin(),names.end());auto a=EventGrounding::words(e.answer);
            if(a.size()!=1||l.story.empty())valid=false;
            else if(a[0]=="unknown")l.unknown=true;
            else if(a[0]=="yes"||a[0]=="no")l.answer={1,a[0]=="yes"};
            else if(IncrementalGrounding::numeric(LanguageReasoner::trim(e.answer))&&a[0].size()<9)l.answer={2,std::stoi(a[0])};
            else {auto it=std::find(l.names.begin(),l.names.end(),a[0]);if(it==l.names.end())valid=false;else l.answer={0,int(it-l.names.begin())};}
            if(!valid){++rejected;continue;}
            if(!l.unknown) {
                auto [it,inserted]=types.emplace(l.form,l.answer.type);
                if(!inserted&&it->second!=l.answer.type){status="conflicting_answer_types";return false;}
            }
            for(const auto& s:l.story)if(!s.event) {
                auto [it,inserted]=model.frames.try_emplace(s.frame);
                if(inserted){it->second.pattern=SemanticGrounding::words(s.frame);it->second.candidates=ObservationLanguage::hypotheses(s.args.size(),{relations[0],relations[1]});}
                it->second.scenes.insert(l.scene);
            }
            lessons.push_back(std::move(l));++accepted;
        }
        if(lessons.empty())return false;
        for(const auto& l:lessons)if(!questions.count(l.form)) {
            if(!types.count(l.form)){status="untyped_unknown_question";return false;}
            std::set<Program> candidates=IncrementalGrounding::atomic_programs(l.args.size(),types.at(l.form));
            for(int op=16;op<=19;++op)for(const Program p:{Program{op,9},Program{3,op,9}})
                if(Meaning::from_program(p,l.args.size()).type()==types.at(l.form))candidates.insert(p);
            auto known=prior.forms.find(l.form);
            if(known!=prior.forms.end()&&prior.memory.procedures.count(known->second))candidates.insert(prior.memory.procedures.at(known->second));
            std::set<std::string> seen;
            for(const auto& p:candidates)if(seen.insert(Meaning::from_program(p,l.args.size()).key()).second)questions[l.form].insert(p);
        }
        auto complexity=[](const Lesson& l){std::set<std::string> f;for(const auto& s:l.story)if(!s.event)f.insert(s.frame);return std::make_pair(f.size(),l.story.size());};
        std::stable_sort(lessons.begin(),lessons.end(),[&](const auto& a,const auto& b){return complexity(a)<complexity(b);});
        bool converged=false;
        for(rounds=0;rounds<12;++rounds) {
            bool changed=false;checked=skipped=0;
            for(const auto& l:lessons) {
                std::set<std::string> involved;for(const auto& s:l.story)if(!s.event)involved.insert(s.frame);
                const auto& programs=questions.at(l.form);size_t combinations=programs.size();
                if(!combinations){status="conflicting_evidence";return false;}
                for(const auto& key:involved) {
                    size_t n=model.frames.at(key).candidates.size();if(!n){status="conflicting_evidence";return false;}
                    if(combinations>assignment_budget/n){combinations=assignment_budget+1;break;}combinations*=n;
                }
                if(combinations>assignment_budget){++skipped;continue;}++checked;
                std::map<std::string,int> ids;for(size_t i=0;i<l.names.size();++i)ids[l.names[i]]=i;
                FlatQuery q;q.a={0,ids.at(l.args.front())};q.b={0,ids.at(l.args.back())};q.relation[0]=relations[0];q.relation[1]=relations[1];
                std::map<std::string,Rule> assignment;std::map<std::string,std::set<Rule>> supported;
                std::set<Program> supported_programs;std::map<FactKnowledge,std::set<Program>> cache;
                std::function<void(size_t,FactKnowledge)> walk=[&](size_t index,FactKnowledge k) {
                    if(index==l.story.size()) {
                        auto found=cache.find(k);
                        if(found==cache.end()) {
                            q.facts.clear();for(auto [a,r,b]:k.positive)q.facts.push_back({a,r,b});q.knowledge=std::make_shared<FactKnowledge>(k);
                            std::set<Program> matches;
                            for(const auto& p:programs) {
                                auto value=prior.memory.execute(q,p);
                                bool match=l.unknown?(!value.present()&&(prior.memory.missing_evidence||prior.memory.nonunique_result)):value==l.answer;
                                if(match)matches.insert(p);
                            }
                            found=cache.emplace(k,std::move(matches)).first;
                        }
                        if(!found->second.empty()) {
                            supported_programs.insert(found->second.begin(),found->second.end());
                            for(const auto& [key,rule]:assignment)supported[key].insert(rule);
                        }
                        return;
                    }
                    const auto& s=l.story[index];std::vector<int> args;for(const auto& a:s.args)args.push_back(ids.at(a));
                    if(s.event){k.update(args[0],s.known.relation,args[1],s.known.effect);walk(index+1,std::move(k));return;}
                    auto advance=[&](const Rule& rule){auto next=k;if(ObservationLanguage::apply(next,rule,args))walk(index+1,std::move(next));};
                    auto bound=assignment.find(s.frame);
                    if(bound!=assignment.end())advance(bound->second);
                    else for(const auto& rule:model.frames.at(s.frame).candidates){assignment[s.frame]=rule;advance(rule);assignment.erase(s.frame);}
                };
                walk(0,{});
                if(supported_programs.empty()){status="conflicting_evidence";return false;}
                if(supported_programs.size()<programs.size()){questions[l.form]=std::move(supported_programs);changed=true;}
                for(const auto& key:involved)if(supported[key].size()<model.frames.at(key).candidates.size()) {
                    model.frames[key].candidates=std::move(supported[key]);changed=true;
                }
            }
            if(!changed){converged=true;++rounds;break;}
        }
        if(!converged||skipped||checked<4){status="insufficient_search_evidence";return false;}
        for(const auto& [form,programs]:questions)if(programs.size()!=1){status="ambiguous_questions";return false;}
        for(const auto& [key,frame]:model.frames)if(frame.scenes.size()<4){status="insufficient_scene_support";return false;}
        status="learned";return true;
    }
    bool publish(LanguageReasoner& reader) const {
        if(status!="learned"||reader.events.signature()!=prior_events||reader.observation_language.signature()!=prior_observations||
           reader.grounding.signature()!=prior_grammar||reader.forms!=prior_forms||reader.memory.procedures!=prior_programs||
           reader.memory.skills.kernel.routes.W!=prior_weights||relations!=std::array<int,2>{reader.relations[0],reader.relations[1]})return false;
        reader.reset_episode();reader.observation_language=model;reader.partial_observation=true;reader.memory.unique_enabled=true;
        int next=0;for(const auto& [form,cue]:reader.forms)next=std::max(next,cue+1);for(const auto& [cue,p]:reader.memory.procedures)next=std::max(next,cue+1);
        for(const auto& [form,programs]:questions) {
            auto [it,inserted]=reader.forms.try_emplace(form,next);if(inserted)++next;
            reader.memory.procedures[it->second]=*programs.begin();
        }
        std::vector<SemanticGrounding::Example> meanings;
        for(const auto& [form,cue]:reader.forms)if(reader.memory.procedures.count(cue))
            meanings.push_back({form,Meaning::from_program(reader.memory.procedures.at(cue),form.find("$1")==std::string::npos?1:2)});
        reader.grounding.fit(meanings);reader.shared_meanings=reader.grounded_events=reader.uncertain_events=true;return true;
    }
};
} // namespace laminar::reasoning
