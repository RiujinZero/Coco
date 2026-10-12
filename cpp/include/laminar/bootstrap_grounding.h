#pragma once
// Joint observation/question grounding from raw episodes and verified answers.
// No entity spans, fact labels, relation meanings or query programs are inputs.
// Current scope: a single atomic question template, single-token entities,
// initially empty episode memory, and the existing binary update hypothesis space.
#include "reasoning_language.h"
#include <functional>

namespace laminar::reasoning {
struct BootstrapGrounding {
    using Rule=EventGrounding::Rule;
    using Facts=EventGrounding::World;
    struct Episode { std::vector<std::string> observations;std::string question,answer;uint64_t id=0;bool self_generated=false; };
    struct Sentence { std::string frame;std::vector<std::string> arguments; };
    struct Lesson {
        std::vector<Sentence> sentences;std::string form;std::vector<std::string> arguments;
        std::vector<std::string> names;Value answer;uint64_t id;
    };
    struct Model { Program program;std::map<std::string,std::set<Rule>> frames;bool viable=true;size_t checked=0,skipped=0; };
    std::vector<Model> models;
    std::map<std::string,std::vector<std::string>> patterns;
    std::map<std::string,std::set<uint64_t>> scenes;
    std::set<std::string> lexicon;
    std::string form,status;
    size_t rejected=0,accepted=0;
    size_t assignment_budget=32768;
    int rounds=0;

    Sentence symbolize(const std::string& text) const {
        auto words=EventGrounding::words(text);Sentence result;std::map<std::string,int> ids;
        for(auto& word:words)if(lexicon.count(word)) {
            auto [it,inserted]=ids.emplace(word,ids.size());
            if(inserted)result.arguments.push_back(word);
            word="$"+std::to_string(it->second);
        }
        result.frame=EventGrounding::join(words);return result;
    }
    // Variation in questions proposes entity vocabulary; answer tokens add
    // possible referents. This is learned lexical alignment, not entity labels.
    // This is a narrow lexical heuristic, not general entity recognition.
    // Proposed query referents must also occur in the observed story, and all
    // questions must share one induced template.
    bool lexicalize(const std::vector<Episode>& examples) {
        std::set<std::vector<std::string>> questions;
        for(const auto& e:examples)if(!e.self_generated&&!e.answer.empty()) {
            questions.insert(EventGrounding::words(e.question));
            auto answer=EventGrounding::words(e.answer);
            if(answer.size()==1&&answer[0]!="yes"&&answer[0]!="no"&&
               !std::all_of(answer[0].begin(),answer[0].end(),[](unsigned char c){return std::isdigit(c);}))lexicon.insert(answer[0]);
        }
        for(const auto& a:questions)for(const auto& b:questions)if(a.size()==b.size()) {
            int changed=-1;bool multiple=false;
            for(size_t i=0;i<a.size();++i)if(a[i]!=b[i]){if(changed>=0){multiple=true;break;}changed=i;}
            if(!multiple&&changed>=0){lexicon.insert(a[changed]);lexicon.insert(b[changed]);}
        }
        for(const auto& e:examples)if(!e.self_generated&&!e.answer.empty()) {
            auto q=symbolize(e.question);
            if(q.arguments.empty()||q.arguments.size()>2)return false;
            std::set<std::string> observed;
            for(const auto& text:e.observations)for(const auto& word:EventGrounding::words(text))observed.insert(word);
            for(const auto& argument:q.arguments)if(!observed.count(argument))return false;
            if(form.empty())form=q.frame;
            if(form!=q.frame)return false;
        }
        return !form.empty();
    }
    bool fit(const std::vector<Episode>& examples,ProcedureMemory& memory) {
        models.clear();patterns.clear();scenes.clear();lexicon.clear();form.clear();
        rejected=accepted=0;rounds=0;status="insufficient_evidence";
        if(!lexicalize(examples)){status="unsupported_question_templates";return false;}
        std::vector<Lesson> lessons;
        for(const auto& e:examples) {
            if(e.self_generated||e.answer.empty()){++rejected;continue;}
            auto q=symbolize(e.question);Lesson lesson;lesson.form=q.frame;lesson.arguments=q.arguments;lesson.id=e.id;
            std::set<std::string> names(q.arguments.begin(),q.arguments.end());bool valid=true;
            for(const auto& text:e.observations) {
                auto sentence=symbolize(text);
                if(sentence.arguments.size()<2||sentence.arguments.size()>4){valid=false;break;}
                names.insert(sentence.arguments.begin(),sentence.arguments.end());lesson.sentences.push_back(sentence);
            }
            auto answer=EventGrounding::words(e.answer);
            lesson.names.assign(names.begin(),names.end());
            if(answer.size()!=1||lesson.sentences.empty())valid=false;
            else if(answer[0]=="yes"||answer[0]=="no")lesson.answer={1,answer[0]=="yes"};
            else if(std::all_of(answer[0].begin(),answer[0].end(),[](unsigned char c){return std::isdigit(c);})) {
                if(answer[0].size()>8)valid=false;else lesson.answer={2,std::stoi(answer[0])};
            } else {
                auto it=std::find(lesson.names.begin(),lesson.names.end(),answer[0]);
                if(it==lesson.names.end())valid=false;else lesson.answer={0,(int)(it-lesson.names.begin())};
            }
            if(!valid){++rejected;continue;}
            for(const auto& sentence:lesson.sentences) {
                patterns[sentence.frame]=SemanticGrounding::words(sentence.frame);scenes[sentence.frame].insert(lesson.id);
            }
            lessons.push_back(std::move(lesson));++accepted;
        }
        if(lessons.empty())return false;
        // Atomic procedures, enumerated without reference to surface words.
        // Later compositions continue to use the existing procedure/grammar learner.
        std::vector<Program> programs{{0,9},{1,9}};
        for(int op:{4,5,6,7,8,10,11}) {
            programs.push_back({op,9});
            programs.push_back({3,op,9});
        }
        std::set<std::string> meanings;
        for(const auto& program:programs) {
            auto meaning=Meaning::from_program(program,lessons.front().arguments.size());
            if(meaning.type()!=lessons.front().answer.type||!meanings.insert(meaning.key()).second)continue;
            Model model;model.program=program;
            for(const auto& [key,pattern]:patterns) {
                int arity=0;for(const auto& word:pattern)arity=std::max(arity,EventGrounding::slot(word)+1);
                for(int relation=0;relation<2;++relation)for(int a=0;a<arity;++a)for(int b=0;b<arity;++b)
                    for(int effect=0;effect<4;++effect)model.frames[key].insert({relation,a,b,effect});
            }
            models.push_back(std::move(model));
        }
        auto complexity=[](const Lesson& e){std::set<std::string> frames;for(const auto& s:e.sentences)frames.insert(s.frame);return std::make_pair(frames.size(),e.sentences.size());};
        std::stable_sort(lessons.begin(),lessons.end(),[&](const auto& a,const auto& b){return complexity(a)<complexity(b);});
        for(rounds=0;rounds<8;++rounds) {
            bool changed=false;
            for(auto& model:models)if(model.viable) {
                model.checked=model.skipped=0;
                for(const auto& lesson:lessons) {
                    std::set<std::string> involved;for(const auto& s:lesson.sentences)involved.insert(s.frame);
                    size_t combinations=1;
                    for(const auto& key:involved) {
                        auto n=model.frames.at(key).size();
                        if(n==0){model.viable=false;break;}
                        if(combinations>assignment_budget/n){combinations=assignment_budget+1;break;}combinations*=n;
                    }
                    if(!model.viable)break;
                    if(combinations>assignment_budget){++model.skipped;continue;}
                    ++model.checked;
                    FlatQuery query;query.relation[0]=0;query.relation[1]=1;
                    std::map<std::string,int> ids;for(size_t i=0;i<lesson.names.size();++i)ids[lesson.names[i]]=i;
                    query.a={0,ids.at(lesson.arguments[0])};query.b={0,ids.at(lesson.arguments.back())};
                    std::map<std::string,Rule> assignment;std::map<std::string,std::set<Rule>> supported;
                    std::map<Facts,Value> cache;bool any=false;
                    std::function<void(size_t,const Facts&)> walk=[&](size_t step,const Facts& world) {
                        if(step==lesson.sentences.size()) {
                            auto found=cache.find(world);
                            if(found==cache.end()) {
                                query.facts.clear();for(const auto& [a,r,b]:world)query.facts.push_back({ids.at(a),r,ids.at(b)});
                                found=cache.emplace(world,memory.execute(query,model.program)).first;
                            }
                            if(found->second==lesson.answer) {
                                any=true;for(const auto& [key,rule]:assignment)supported[key].insert(rule);
                            }
                            return;
                        }
                        const auto& sentence=lesson.sentences[step];
                        auto advance=[&](const Rule& rule){walk(step+1,EventGrounding::update(world,rule.relation,rule.effect,
                            sentence.arguments.at(rule.subject),sentence.arguments.at(rule.object)));};
                        auto bound=assignment.find(sentence.frame);
                        if(bound!=assignment.end())advance(bound->second);
                        else for(const auto& rule:model.frames.at(sentence.frame)) {
                            assignment[sentence.frame]=rule;advance(rule);assignment.erase(sentence.frame);
                        }
                    };
                    walk(0,{});
                    if(!any){model.viable=false;changed=true;break;}
                    // Domain projection is conservative: discard a candidate
                    // only after exhaustive enumeration proves it unsupported.
                    // Correlations are not retained between different frames.
                    for(const auto& key:involved)if(supported[key].size()<model.frames.at(key).size()) {
                        model.frames[key]=std::move(supported[key]);changed=true;
                    }
                }
            }
            if(!changed){++rounds;break;}
        }
        std::set<std::string> distinct;
        bool budget=false;
        for(const auto& model:models)if(model.viable) {
            distinct.insert(canonical(model));budget|=model.skipped>0||model.checked<4;
        }
        status=distinct.empty()?"conflicting_evidence":distinct.size()>1?"ambiguous_joint_model":"learned";
        if(!distinct.empty()&&budget)status="insufficient_search_evidence";
        return status=="learned";
    }
    static std::string encode(const Model& model,bool swap) {
        std::string result="program:";
        for(int op:model.program){if(swap&&((op>=4&&op<=7)||op>=10))op^=1;result+=std::to_string(op)+",";}
        for(const auto& [key,rules]:model.frames) {
            result+='\n'+key+":";std::set<Rule> ordered;
            for(auto rule:rules){if(swap)rule.relation^=1;ordered.insert(rule);}
            for(const auto& rule:ordered)result+=std::to_string(rule.relation)+","+std::to_string(rule.subject)+","+std::to_string(rule.object)+","+std::to_string(rule.effect)+";";
        }
        return result;
    }
    static std::string canonical(const Model& model){return std::min(encode(model,false),encode(model,true));}
    bool publish(LanguageReasoner& reader) const {
        if(status!="learned")return false;
        // This cold bootstrap has no anchors to identify its latent channels
        // with the meanings of an existing model. Do not silently merge them.
        if(!reader.events.frames.empty()||!reader.frames.empty()||!reader.forms.empty()||!reader.memory.procedures.empty())return false;
        const Model* selected=nullptr;
        for(const auto& model:models)if(model.viable&&(!selected||encode(model,false)<encode(*selected,false)))selected=&model;
        if(!selected)return false;
        reader.reset_episode();
        for(const auto& [key,rules]:selected->frames) {
            auto& frame=reader.events.frames[key];frame.pattern=patterns.at(key);frame.candidates=rules;
            frame.scenes=scenes.at(key);frame.initialized=true;
        }
        int cue;
        auto found=reader.forms.find(form);
        if(found==reader.forms.end()){cue=reader.forms.size();reader.forms[form]=cue;}else cue=found->second;
        reader.memory.procedures[cue]=selected->program;reader.relations[0]=0;reader.relations[1]=1;
        reader.grounded_events=reader.uncertain_events=true;
        // Symbolic relation channel numbering is not identifiable from language
        // alone; only global channel permutations are quotiented above.
        return true;
    }
};
} // namespace laminar::reasoning
