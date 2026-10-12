#pragma once
// Shared observation/question induction anchored by an existing learned model.
// Teaching records contain raw text and verified final answers only. Prior
// meanings are learned anchors, not fresh teacher-supplied semantic labels.
#include "bootstrap_grounding.h"

namespace laminar::reasoning {
struct IncrementalGrounding {
    using Episode=BootstrapGrounding::Episode;
    using Sentence=BootstrapGrounding::Sentence;
    using Lesson=BootstrapGrounding::Lesson;
    using Rule=EventGrounding::Rule;
    using Facts=EventGrounding::World;
    BootstrapGrounding surface;
    std::map<std::string,std::set<Rule>> frames;
    std::map<std::string,std::set<Program>> questions;
    std::map<std::string,std::vector<std::string>> patterns;
    std::map<std::string,std::set<uint64_t>> scenes;
    std::string status,prior_events,prior_grammar;
    std::map<std::string,int> prior_forms;
    std::map<int,Program> prior_programs;
    std::vector<float> prior_weights;
    std::array<int,2> relations{};
    std::map<int,std::string> orientation_gauges;
    size_t accepted=0,rejected=0,checked=0,skipped=0;
    size_t assignment_budget=131072;
    int rounds=0;

    Sentence symbolize(const std::string& text) const {
        Sentence result;auto words=EventGrounding::words(text);
        // Argument slots describe positions, not distinct entity identities.
        // Repeated names bind the same value without creating a separate frame
        // (e.g. reflexive transfers or comparing an entity with itself).
        for(auto& word:words)if(surface.lexicon.count(word)) {
            result.arguments.push_back(word);word="$"+std::to_string(result.arguments.size()-1);
        }
        result.frame=EventGrounding::join(words);return result;
    }
    static bool numeric(const std::string& text) {
        return !text.empty()&&std::all_of(text.begin(),text.end(),[](unsigned char c){return std::isdigit(c);});
    }
    void lexicalize(const std::vector<Episode>& examples,const LanguageReasoner& prior) {
        surface.lexicon.clear();
        std::map<std::vector<std::string>,std::set<std::string>> observed;
        for(const auto& e:examples)if(!e.self_generated&&!e.answer.empty()) {
            auto q=EventGrounding::words(e.question);
            for(const auto& text:e.observations) {
                for(const auto& word:EventGrounding::words(text))observed[q].insert(word);
                // Known sentence meanings can recognize new referents. These
                // meanings come from previous learning and stay constrained.
                for(const auto& meaning:prior.events.alternatives(text).values) {
                    surface.lexicon.insert(meaning.subject);surface.lexicon.insert(meaning.object);
                    surface.lexicon.insert(meaning.mentions.begin(),meaning.mentions.end());
                }
            }
            auto answer=EventGrounding::words(e.answer);
            if(answer.size()==1&&answer[0]!="yes"&&answer[0]!="no"&&!numeric(answer[0]))surface.lexicon.insert(answer[0]);
        }
        for(const auto& [a,oa]:observed)for(const auto& [b,ob]:observed)if(a.size()==b.size()) {
            int changed=-1;bool multiple=false;
            for(size_t i=0;i<a.size();++i)if(a[i]!=b[i]){if(changed>=0){multiple=true;break;}changed=i;}
            if(!multiple&&changed>=0&&oa.count(a[changed])&&ob.count(b[changed])) {
                surface.lexicon.insert(a[changed]);surface.lexicon.insert(b[changed]);
            }
        }
    }
    static std::set<Program> atomic_programs(int arity,int type) {
        std::vector<Program> all{{0,9},{1,9}};
        for(int op:{4,5,6,7,8,10,11,12,13,14,15}){all.push_back({op,9});all.push_back({3,op,9});}
        std::set<std::string> seen;std::set<Program> out;
        for(const auto& p:all) {
            auto meaning=Meaning::from_program(p,arity);
            if(meaning.type()==type&&seen.insert(meaning.key()).second)out.insert(p);
        }
        return out;
    }
    bool fit(const std::vector<Episode>& examples,LanguageReasoner& prior) {
        frames.clear();questions.clear();patterns.clear();scenes.clear();orientation_gauges.clear();
        status="insufficient_evidence";accepted=rejected=checked=skipped=0;rounds=0;
        prior_events=prior.events.signature();prior_grammar=prior.grounding.signature();
        prior_forms=prior.forms;prior_programs=prior.memory.procedures;prior_weights=prior.memory.skills.kernel.routes.W;
        relations={prior.relations[0],prior.relations[1]};
        if(prior.partial_observation||prior.events.frames.empty()||prior.forms.empty()) {
            status="unsupported_prior";return false;
        }
        for(const auto& [key,frame]:prior.events.frames) {
            frames[key]=frame.candidates;patterns[key]=frame.pattern;scenes[key]=frame.scenes;
        }
        for(const auto& [form,cue]:prior.forms)if(prior.memory.procedures.count(cue))questions[form]={prior.memory.procedures.at(cue)};
        lexicalize(examples,prior);
        std::vector<Lesson> lessons;
        for(const auto& e:examples) {
            if(e.self_generated||e.answer.empty()){++rejected;continue;}
            auto q=symbolize(e.question);Lesson lesson;lesson.form=q.frame;lesson.arguments=q.arguments;lesson.id=e.id;
            bool valid=!q.arguments.empty()&&q.arguments.size()<=2;
            std::set<std::string> names,observed;
            for(const auto& text:e.observations) {
                for(const auto& w:EventGrounding::words(text))observed.insert(w);
                auto sentence=symbolize(text);
                if(sentence.arguments.size()<2||sentence.arguments.size()>4){valid=false;break;}
                names.insert(sentence.arguments.begin(),sentence.arguments.end());lesson.sentences.push_back(std::move(sentence));
            }
            for(const auto& argument:q.arguments)valid&=observed.count(argument)>0&&names.count(argument)>0;
            lesson.names.assign(names.begin(),names.end());auto answer=EventGrounding::words(e.answer);
            if(answer.size()!=1||lesson.sentences.empty())valid=false;
            else if(answer[0]=="yes"||answer[0]=="no")lesson.answer={1,answer[0]=="yes"};
            else if(numeric(answer[0])){if(answer[0].size()>8)valid=false;else lesson.answer={2,std::stoi(answer[0])};}
            else {
                auto found=std::find(lesson.names.begin(),lesson.names.end(),answer[0]);
                if(found==lesson.names.end())valid=false;else lesson.answer={0,(int)(found-lesson.names.begin())};
            }
            if(!valid){++rejected;continue;}
            if(!questions.count(q.frame))questions[q.frame]=atomic_programs(q.arguments.size(),lesson.answer.type);
            for(const auto& sentence:lesson.sentences) {
                patterns[sentence.frame]=SemanticGrounding::words(sentence.frame);scenes[sentence.frame].insert(lesson.id);
                if(!frames.count(sentence.frame)) {
                    auto& rules=frames[sentence.frame];int n=sentence.arguments.size();
                    for(int r:relations)for(int a=0;a<n;++a)for(int b=0;b<n;++b)for(int effect=0;effect<4;++effect)
                        rules.insert({r,a,b,effect});
                }
            }
            lessons.push_back(std::move(lesson));++accepted;
        }
        if(lessons.empty())return false;
        // Reversing every edge of a previously unused relation, swapping its
        // row/column replacement effects, and reversing every query read leaves
        // all observable answers invariant. Choose one representation; do not
        // call that choice learned argument semantics. Existing channels retain
        // their established orientation. Self-bindings are not removed.
        std::set<int> used;
        for(const auto& [key,frame]:prior.events.frames)for(const auto& r:frame.candidates)used.insert(r.relation);
        for(int relation:relations)if(!used.count(relation))for(auto& [key,rules]:frames)if(!prior.events.frames.count(key)) {
            orientation_gauges[relation]=key;
            for(auto it=rules.begin();it!=rules.end();)if(it->relation==relation&&it->subject>it->object)it=rules.erase(it);else ++it;
            break;
        }
        // Teach simple histories first, without supplying any event or question
        // family labels. Later passes revisit earlier examples after refinement.
        auto complexity=[](const Lesson& e){std::set<std::string> f;for(const auto& s:e.sentences)f.insert(s.frame);return std::make_pair(f.size(),e.sentences.size());};
        std::stable_sort(lessons.begin(),lessons.end(),[&](const auto& a,const auto& b){return complexity(a)<complexity(b);});
        bool converged=false;
        for(rounds=0;rounds<12;++rounds) {
            bool changed=false;checked=skipped=0;
            for(const auto& lesson:lessons) {
                std::set<std::string> involved;for(const auto& s:lesson.sentences)involved.insert(s.frame);
                const auto& programs=questions.at(lesson.form);size_t combinations=programs.size();
                if(combinations==0){status="conflicting_evidence";return false;}
                for(const auto& key:involved) {
                    size_t n=frames.at(key).size();if(!n){status="conflicting_evidence";return false;}
                    if(combinations>assignment_budget/n){combinations=assignment_budget+1;break;}combinations*=n;
                }
                if(combinations>assignment_budget){++skipped;continue;}++checked;
                FlatQuery query;query.relation[0]=relations[0];query.relation[1]=relations[1];
                std::map<std::string,int> ids;for(size_t i=0;i<lesson.names.size();++i)ids[lesson.names[i]]=i;
                query.a={0,ids.at(lesson.arguments[0])};query.b={0,ids.at(lesson.arguments.back())};
                std::map<std::string,Rule> assignment;std::map<std::string,std::set<Rule>> supported_frames;
                std::set<Program> supported_programs;
                std::map<Facts,std::set<Program>> cache;
                std::function<void(size_t,const Facts&)> walk=[&](size_t step,const Facts& world) {
                    if(step==lesson.sentences.size()) {
                        auto found=cache.find(world);
                        if(found==cache.end()) {
                            query.facts.clear();for(const auto& [a,r,b]:world)query.facts.push_back({ids.at(a),r,ids.at(b)});
                            std::set<Program> matches;for(const auto& p:programs)if(prior.memory.execute(query,p)==lesson.answer)matches.insert(p);
                            found=cache.emplace(world,std::move(matches)).first;
                        }
                        if(!found->second.empty()) {
                            supported_programs.insert(found->second.begin(),found->second.end());
                            for(const auto& [key,rule]:assignment)supported_frames[key].insert(rule);
                        }
                        return;
                    }
                    const auto& s=lesson.sentences[step];
                    auto advance=[&](const Rule& rule){walk(step+1,EventGrounding::update(world,rule.relation,rule.effect,
                        s.arguments.at(rule.subject),s.arguments.at(rule.object)));};
                    auto bound=assignment.find(s.frame);
                    if(bound!=assignment.end())advance(bound->second);
                    else for(const auto& rule:frames.at(s.frame)){assignment[s.frame]=rule;advance(rule);assignment.erase(s.frame);}
                };
                walk(0,{});
                if(supported_programs.empty()){status="conflicting_evidence";return false;}
                if(supported_programs.size()<programs.size()){questions[lesson.form]=std::move(supported_programs);changed=true;}
                for(const auto& key:involved)if(supported_frames[key].size()<frames.at(key).size()) {
                    frames[key]=std::move(supported_frames[key]);changed=true;
                }
            }
            if(!changed){converged=true;++rounds;break;}
        }
        if(!converged||skipped||checked<4){status="insufficient_search_evidence";return false;}
        for(const auto& [form,programs]:questions)if(programs.size()!=1){status="ambiguous_questions";return false;}
        status="learned";return true;
    }
    bool publish(LanguageReasoner& reader) const {
        if(status!="learned"||reader.partial_observation||reader.events.signature()!=prior_events||
           reader.grounding.signature()!=prior_grammar||reader.forms!=prior_forms||reader.memory.procedures!=prior_programs||
           reader.memory.skills.kernel.routes.W!=prior_weights||relations!=std::array<int,2>{reader.relations[0],reader.relations[1]})return false;
        // No partial publication: validate the prior before changing anything.
        reader.reset_episode();
        for(const auto& [key,rules]:frames) {
            auto& frame=reader.events.frames[key];frame.pattern=patterns.at(key);frame.candidates=rules;
            frame.scenes=scenes.at(key);frame.initialized=true;
        }
        int next=0;for(const auto& [form,cue]:reader.forms)next=std::max(next,cue+1);
        for(const auto& [cue,p]:reader.memory.procedures)next=std::max(next,cue+1);
        for(const auto& [form,programs]:questions) {
            auto found=reader.forms.find(form);int cue;
            if(found==reader.forms.end()){cue=next++;reader.forms[form]=cue;}else cue=found->second;
            reader.memory.procedures[cue]=*programs.begin();
        }
        std::vector<SemanticGrounding::Example> meanings;
        for(const auto& [form,cue]:reader.forms)if(reader.memory.procedures.count(cue))
            meanings.push_back({form,Meaning::from_program(reader.memory.procedures.at(cue),form.find("$1")==std::string::npos?1:2)});
        reader.grounding.fit(meanings);reader.grounded_events=reader.uncertain_events=reader.shared_meanings=true;
        return true;
    }
};
} // namespace laminar::reasoning
