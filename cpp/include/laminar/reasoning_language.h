#pragma once
// Experimental grounded text interface. Fact frames are learned from aligned
// demonstrations; question forms acquire procedures from verified final answers.
// This is a deliberately narrow grammar learner, not a general language model.
#include "procedure_memory.h"
#include "semantic_grounding.h"
#include "event_grounding.h"
#include "reference_grounding.h"
#include "request_grounding.h"
#include "observation_language.h"
#include <cctype>
#include <sstream>
#include <tuple>

namespace laminar::reasoning {
struct LanguageReasoner {
    struct Frame {
        std::vector<std::string> words;
        int relation = -1;
        std::set<std::pair<std::string, std::string>> support;
    };
    ProcedureMemory memory;
    std::vector<Frame> frames;
    std::map<std::string, int> forms;
    std::map<int, std::vector<VerifiedQuestion>> evidence;
    // These are grounded relation identities, not question-specific recipes.
    int relations[2] = {0, 1};
    std::map<std::string, int> entities;
    std::vector<std::string> names;
    std::vector<Fact> facts;
    std::string segment, question, reply, form, status;
    size_t reply_offset = 0;
    bool complete = true, assistant = false, prepared = false;
    // An explicit environment protocol separates requests/claims from evidence.
    // Legacy story-QA streams retain their original USER-as-narrator convention.
    enum class StreamRole { CONTEXT, USER, ASSISTANT, ENVIRONMENT };
    bool environment_dialogue = false;
    StreamRole stream_role = StreamRole::CONTEXT;
    bool line_has_role = false, dialogue_question_pending = false;
    bool dialogue_line_overflow = false;
    // Requests are retained separately from facts and answers. The executor is
    // an explicit caller-owned actuator; reading a request never changes state.
    bool grounded_requests = false;
    std::string request;
    RequestGrounding requests;
    GroundedActuator actuators;
    // Monotonic source revisions let an external interaction controller bind
    // delayed feedback to one request and one observed transition.
    uint64_t episode_revision=0,request_revision=0,observation_revision=0;
    // An explicit revision of historical evidence invalidates in-flight credit
    // even when the physical observation count and learned model are unchanged.
    uint64_t history_revision=0;
    bool last_observation_environment=false;
    std::string last_observation_text;
    bool awaiting_action_observation=false;
    FlatQuery query;
    SemanticGrounding grounding;
    bool shared_meanings = false, phrase_fragments = true;
    SemanticGrounding::Parse interpretation;
    EventGrounding events;
    bool grounded_events = false;
    // Bounded possible worlds for ambiguous learned event meanings. This tracks
    // uncertainty within the hypothesis space, not arbitrary unseen world facts.
    bool uncertain_events = false;
    bool partial_observation = false;
    FactKnowledge observations;
    ObservationLanguage observation_language;
    std::string observation_language_signature;
    using FactSet = std::set<std::tuple<int,int,int>>;
    struct PossibleWorld {
        FactSet facts;
        std::map<std::string,std::string> meanings;
        FactKnowledge knowledge;
        bool operator<(const PossibleWorld& w) const { return std::tie(facts,meanings,knowledge)<std::tie(w.facts,w.meanings,w.knowledge); }
    };
    std::vector<PossibleWorld> possible_worlds;
    std::string event_issue, world_model_signature;
    size_t world_budget = 64;
    ReferenceGrounding references;
    bool grounded_references = false;
    ReferenceGrounding::Context reference_context, turn_events;
    ReferenceGrounding::Resolution pending_reference;
    std::string pending_text;
    bool pending_question = false;
    std::vector<std::string> deferred_facts;
    uint64_t answered = 0, abstained = 0, clarification_requests = 0, uncertain_replies = 0, forced_bytes = 0;
    explicit LanguageReasoner(uint64_t seed) : memory(seed) {}
    LanguageReasoner() : LanguageReasoner(0) {}   // checkpoint restore target: every field is then loaded
    template <class Ar> void state_io(Ar& ar) {
        ar(memory); ar(frames); ar(forms); ar(evidence); ar.array(relations, 2); ar(entities); ar(names); ar(facts);
        ar(segment); ar(question); ar(reply); ar(form); ar(status); ar(reply_offset); ar(complete); ar(assistant); ar(prepared);
        ar(environment_dialogue); ar(stream_role); ar(line_has_role); ar(dialogue_question_pending); ar(dialogue_line_overflow);
        ar(grounded_requests); ar(request); ar(requests); ar(actuators); ar(episode_revision); ar(request_revision); ar(observation_revision);
        ar(history_revision); ar(last_observation_environment); ar(last_observation_text); ar(awaiting_action_observation); ar(query);
        ar(grounding); ar(shared_meanings); ar(phrase_fragments); ar(interpretation); ar(events); ar(grounded_events); ar(uncertain_events);
        ar(partial_observation); ar(observations); ar(observation_language); ar(observation_language_signature); ar(possible_worlds);
        ar(event_issue); ar(world_model_signature); ar(world_budget); ar(references); ar(grounded_references); ar(reference_context);
        ar(turn_events); ar(pending_reference); ar(pending_text); ar(pending_question); ar(deferred_facts);
        ar(answered); ar(abstained); ar(clarification_requests); ar(uncertain_replies); ar(forced_bytes);
    }
    const ReferenceGrounding::Context& active_reference_context() const {
        return turn_events.empty() ? reference_context : turn_events;
    }

    static std::vector<std::string> tokens(const std::string& s) {
        std::vector<std::string> out; std::string w;
        for (unsigned char c : s) {
            if (std::isalnum(c)) w += (char)std::tolower(c);
            else if (!w.empty()) { out.push_back(w); w.clear(); }
        }
        if (!w.empty()) out.push_back(w);
        return out;
    }
    static std::string trim(const std::string& s) {
        const auto b = s.find_first_not_of(" \r\n\t");
        if (b == std::string::npos) return {};
        return s.substr(b, s.find_last_not_of(" \r\n\t") - b + 1);
    }
    void teach_fact(const std::string& text, const std::string& source, int relation,
                    const std::string& destination) {
        auto a = tokens(source), b = tokens(destination), words = tokens(text);
        if (a.size() != 1 || b.size() != 1 || a == b)
            throw std::invalid_argument("grounding requires two distinct single-token entities");
        int na = 0, nb = 0;
        for (auto& w : words) {
            if (w == a[0]) { w = "$subject"; ++na; }
            else if (w == b[0]) { w = "$object"; ++nb; }
        }
        if (na != 1 || nb != 1) throw std::invalid_argument("grounded spans must occur exactly once");
        for (auto& f : frames) if (f.words == words && f.relation == relation) {
            f.support.emplace(a[0], b[0]); return;
        }
        frames.push_back({words, relation, {{a[0], b[0]}}});
    }
    int entity(const std::string& name) {
        auto it = entities.find(name);
        if (it != entities.end()) return it->second;
        int id = (int)names.size(); entities[name] = id; names.push_back(name); return id;
    }
    RequestGrounding::Snapshot request_snapshot() const {
        RequestGrounding::Snapshot result;
        result.valid=complete&&!awaiting_action_observation&&pending_text.empty()&&(world_model_signature.empty()||world_model_signature==events.signature())&&
            (observation_language_signature.empty()||observation_language_signature==observation_language.signature());
        auto append=[&](const FactSet& facts,const FactKnowledge& knowledge) {
            RequestGrounding::State world;world.closed_world=!partial_observation;
            for(const auto& [a,r,b]:facts)world.facts.emplace(names.at(a),r,names.at(b));
            if(partial_observation) {
                for(const auto& [a,r,b]:knowledge.negative)world.negative.emplace(names.at(a),r,names.at(b));
                for(const auto& [a,r]:knowledge.closed_rows)world.closed_rows.emplace(names.at(a),r);
                for(const auto& [r,b]:knowledge.closed_columns)world.closed_columns.emplace(r,names.at(b));
            }
            result.worlds.push_back(std::move(world));
        };
        if(possible_worlds.empty()) {FactSet current;for(const auto& f:facts)current.emplace(f.subject,f.relation,f.object);append(current,observations);}
        else for(const auto& world:possible_worlds)append(world.facts,world.knowledge);
        return result;
    }
    bool add_verified_request_feedback(bool satisfied,uint64_t scene,bool self_generated=false) {
        if(!grounded_requests||request.empty())return false;
        return requests.teach(request,request_snapshot(),{names.begin(),names.end()},
                              {relations[0],relations[1]},satisfied,scene,self_generated);
    }
    GroundedActuator::Choice choose_action(const std::vector<GroundedActuator::Action>& available,size_t budget=8192) const {
        if(!grounded_requests||request.empty())return {-1,0,"unknown_request"};
        return actuators.choose(requests,request,request_snapshot(),available,budget);
    }
    Meaning question_meaning() const {
        auto it=forms.find(form);
        if(it!=forms.end()) {
            auto program=memory.procedures.find(it->second);
            if(program!=memory.procedures.end()) return Meaning::from_program(program->second,form.find("$1")==std::string::npos?1:2);
        }
        return interpretation.found?interpretation.meaning:Meaning{};
    }
    void project_worlds() {
        facts.clear(); if(possible_worlds.empty()) return;
        auto common=possible_worlds.front().facts;
        for(size_t i=1;i<possible_worlds.size();++i) {
            FactSet intersection;
            std::set_intersection(common.begin(),common.end(),possible_worlds[i].facts.begin(),possible_worlds[i].facts.end(),std::inserter(intersection,intersection.begin()));
            common=std::move(intersection);
        }
        for(auto [s,r,o]:common) facts.push_back({s,r,o});
        if(partial_observation) {
            observations=possible_worlds.front().knowledge;observations.positive=common;
            for(size_t i=1;i<possible_worlds.size();++i) {
                const auto& k=possible_worlds[i].knowledge;FactSet negative;
                std::set_intersection(observations.negative.begin(),observations.negative.end(),k.negative.begin(),k.negative.end(),std::inserter(negative,negative.begin()));
                observations.negative=std::move(negative);
            }
            // Closure can be projected only when every branch knows the same
            // complete set. Common closure alone does not imply an empty set.
            for(auto it=observations.closed_rows.begin();it!=observations.closed_rows.end();) {
                bool same=true;
                for(const auto& w:possible_worlds) {
                    same&=w.knowledge.closed_rows.count(*it)>0;
                    for(const auto& atom:w.facts)if(std::get<0>(atom)==it->first&&std::get<1>(atom)==it->second&&!common.count(atom))same=false;
                }
                if(!same)it=observations.closed_rows.erase(it);else ++it;
            }
            for(auto it=observations.closed_columns.begin();it!=observations.closed_columns.end();) {
                bool same=true;
                for(const auto& w:possible_worlds) {
                    same&=w.knowledge.closed_columns.count(*it)>0;
                    for(const auto& atom:w.facts)if(std::get<1>(atom)==it->first&&std::get<2>(atom)==it->second&&!common.count(atom))same=false;
                }
                if(!same)it=observations.closed_columns.erase(it);else ++it;
            }
        }
    }
    template<class Operation> bool perceive(Operation operation,bool self_generated=false) {
        if(!partial_observation||self_generated||!complete)return false;
        if(possible_worlds.empty()) {
            auto next=observations;
            if(!operation(next)){complete=false;event_issue="conflicting_observation";return false;}
            observations=std::move(next);facts.clear();for(auto [s,r,o]:observations.positive)facts.push_back({s,r,o});return true;
        }
        std::vector<PossibleWorld> surviving;
        for(auto world:possible_worlds)if(operation(world.knowledge)){world.facts=world.knowledge.positive;surviving.push_back(std::move(world));}
        if(surviving.empty()){complete=false;event_issue="conflicting_observation";return false;}
        possible_worlds=std::move(surviving);project_worlds();return true;
    }
    bool observe_fact(const std::string& subject,int relation,const std::string& object,bool present,bool self_generated=false) {
        if(self_generated||!partial_observation)return false;
        int a=entity(subject),b=entity(object);
        return perceive([&](FactKnowledge& k){return k.observe(a,relation,b,present);});
    }
    bool observe_complete_row(const std::string& subject,int relation,const std::vector<std::string>& objects,bool self_generated=false) {
        if(self_generated||!partial_observation)return false;
        int a=entity(subject);std::set<int> values;for(const auto& name:objects)values.insert(entity(name));
        return perceive([&](FactKnowledge& k){return k.observe_row(a,relation,values);});
    }
    bool lose_visibility(const std::string& name,int relation,bool column=false,bool self_generated=false) {
        if(self_generated||!partial_observation||!entities.count(name))return false;
        int id=entities.at(name);
        return perceive([&](FactKnowledge& k){if(column)k.forget_column(relation,id);else k.forget_row(id,relation);return true;});
    }
    void read_fact(const std::string& text) {
        const auto words = tokens(text);
        if (words.empty()) return;
        if(partial_observation) {
            if(!complete)return;
            const auto model=observation_language.signature();
            if((!observation_language_signature.empty()&&observation_language_signature!=model)||
               (!world_model_signature.empty()&&world_model_signature!=events.signature())) {
                complete=false;event_issue="stale_observation_model";return;
            }
            observation_language_signature=model;
            auto alternatives=observation_language.alternatives(text,{names.begin(),names.end()});
            if(alternatives.matched) {
                if(alternatives.conflicted){complete=false;event_issue="conflicting_observation_evidence";return;}
                if(possible_worlds.empty()){FactSet initial;for(const auto& f:facts)initial.emplace(f.subject,f.relation,f.object);possible_worlds.push_back({initial,{},observations});}
                std::set<PossibleWorld> next;
                for(const auto& interpretation:alternatives.values) {
                    if(facts.size()>=4096||names.size()+interpretation.arguments.size()>8190){complete=false;event_issue="world_budget";return;}
                    std::vector<int> args;for(const auto& name:interpretation.arguments)args.push_back(entity(name));
                    for(auto world:possible_worlds) {
                        const auto key="observation:"+interpretation.frame,choice=interpretation.rule.signature();
                        auto bound=world.meanings.find(key);
                        if(bound!=world.meanings.end()&&bound->second!=choice)continue;
                        world.meanings[key]=choice;
                        if(!ObservationLanguage::apply(world.knowledge,interpretation.rule,args))continue;
                        world.facts=world.knowledge.positive;next.insert(std::move(world));
                        if(next.size()>world_budget){complete=false;event_issue="world_budget";return;}
                    }
                }
                if(next.empty()){complete=false;event_issue="conflicting_observation";return;}
                possible_worlds.assign(next.begin(),next.end());project_worlds();return;
            }
        }
        if (grounded_events) {
            std::string resolved = text;
            if (grounded_references) {
                if (!pending_text.empty()) {
                    // A bare, explicitly named candidate is a single-use user
                    // clarification. It does not teach a global phrase meaning.
                    if (words.size() == 1 && entities.count(words[0]) &&
                        (pending_reference.candidates.empty() || pending_reference.candidates.count(words[0]))) {
                        auto bound = ReferenceGrounding::bind(pending_text, pending_reference.phrase, words[0]);
                        bool was_question = pending_question; pending_text.clear(); pending_reference = {};
                        if (was_question) question = bound; else read_fact(bound);
                        auto waiting = std::move(deferred_facts); deferred_facts.clear();
                        for (const auto& fact : waiting) read_fact(fact);
                    } else if (words.size() > 1) deferred_facts.push_back(text);
                    return;
                }
                auto result = references.resolve(text, active_reference_context());
                if (!result.resolved) { pending_reference = result; pending_text = text; pending_question = false; return; }
                resolved = result.text;
            }
            if(uncertain_events||partial_observation) {
                const auto current_model=events.signature();
                if(!world_model_signature.empty() && world_model_signature!=current_model) {
                    complete=false;event_issue="stale_event_model";return;
                }
                world_model_signature=current_model;
                auto alternatives=events.alternatives(resolved);
                if(!alternatives.matched || alternatives.conflicted || alternatives.values.empty()) {
                    complete=false; event_issue=alternatives.conflicted?"conflicting_event_evidence":"unknown_event"; return;
                }
                if(facts.size()>=4096 || names.size()>=8190) { complete=false; event_issue="world_budget"; return; }
                if(possible_worlds.empty()) {
                    FactSet initial; for(const auto& f:facts) initial.emplace(f.subject,f.relation,f.object);
                    possible_worlds.push_back({std::move(initial),{},observations});
                }
                std::set<PossibleWorld> next;
                for(const auto& update:alternatives.values) {
                    int a=entity(update.subject),b=entity(update.object);
                    // A participant remains a referent even when this event's
                    // final state does not contain it (e.g. a previous holder).
                    for(const auto& name:update.mentions)entity(name);
                    for(const auto& world:possible_worlds) {
                        auto bound=world.meanings.find(update.frame);
                        if(bound!=world.meanings.end() && bound->second!=update.choice) continue;
                        auto next_world=world;
                        next_world.meanings[update.frame]=update.choice;
                        next_world.facts=EventGrounding::update(world.facts,update.relation,update.effect,a,b);
                        if(partial_observation) {
                            next_world.knowledge.update(a,update.relation,b,update.effect);
                            next_world.facts=next_world.knowledge.positive;
                        }
                        next.insert(std::move(next_world));
                        if(next.size()>world_budget) { complete=false; event_issue="world_budget"; return; }
                    }
                    if(grounded_references) turn_events.push_back(update);
                }
                if(next.empty()) { complete=false;event_issue="conflicting_event_history";return; }
                possible_worlds.assign(next.begin(),next.end()); project_worlds(); return;
            }
            EventGrounding::Interpretation update;
            if (!events.interpret(resolved, update) || facts.size() >= 4096 || names.size() >= 8190) { complete = false; return; }
            int a = entity(update.subject), b = entity(update.object);
            for(const auto& name:update.mentions)entity(name);
            std::set<std::tuple<int, int, int>> state;
            for (const auto& f : facts) state.emplace(f.subject, f.relation, f.object);
            state = EventGrounding::update(std::move(state), update.relation, update.effect, a, b);
            facts.clear(); for (auto [s, r, o] : state) facts.push_back({s, r, o});
            if (grounded_references) turn_events.push_back(update);
            return;
        }
        std::set<std::tuple<std::string, int, std::string>> candidates;
        for (const auto& f : frames) {
            if (f.support.size() < 2 || words.size() != f.words.size()) continue;
            std::string a, b; bool match = true;
            for (size_t i = 0; i < words.size(); ++i) {
                if (f.words[i] == "$subject") a = words[i];
                else if (f.words[i] == "$object") b = words[i];
                else if (f.words[i] != words[i]) { match = false; break; }
            }
            if (match) candidates.emplace(a, f.relation, b);
        }
        // Missing or ambiguous facts must not masquerade as a count of zero.
        if (candidates.size() != 1 || facts.size() >= 4096 || names.size() >= 8190) {
            complete = false; return;
        }
        const auto& [a, r, b] = *candidates.begin();
        if(partial_observation) observe_fact(a,r,b,true);
        else facts.push_back({entity(a), r, entity(b)});
    }
    void reset_episode() {
        ++episode_revision;
        awaiting_action_observation=false;
        last_observation_text.clear();
        entities.clear(); names.clear(); facts.clear(); segment.clear(); question.clear();
        request.clear();
        reply.clear(); form.clear(); status.clear(); reply_offset = 0;
        complete = true; assistant = false; prepared = false; query = {};
        stream_role=StreamRole::CONTEXT;line_has_role=false;dialogue_question_pending=false;
        dialogue_line_overflow=false;
        reference_context.clear(); turn_events.clear(); pending_reference = {};
        pending_text.clear(); pending_question = false; deferred_facts.clear();
        possible_worlds.clear(); event_issue.clear(); world_model_signature.clear();
        observations={};
        observation_language_signature.clear();
    }
    std::string render(Value v) const {
        if (v.type == 0 && v.id >= 0 && v.id < (int)names.size()) return names[v.id];
        if (v.type == 1 && (v.id == 0 || v.id == 1)) return v.id ? "yes" : "no";
        if (v.type == 2 && v.id >= 0) return std::to_string(v.id);
        return {};
    }
    bool prepare() {
        prepared = false; reply.clear(); reply_offset = 0; form.clear(); query = {};
        if(awaiting_action_observation) {status="awaiting_action_observation";reply=" I do not know.\n";return true;}
        interpretation = {};
        if(partial_observation&&!observation_language_signature.empty()&&observation_language_signature!=observation_language.signature()) {
            complete=false;event_issue="stale_observation_model";
        }
        if((uncertain_events||partial_observation) && !world_model_signature.empty() && world_model_signature!=events.signature()) {
            complete=false;event_issue="stale_event_model";
        }
        if (grounded_references) {
            if (pending_text.empty()) {
                auto result = references.resolve(question, active_reference_context());
                if (!result.resolved) { pending_reference = result; pending_text = question; pending_question = true; }
                else question = result.text;
            }
            if (!pending_text.empty()) {
                reply = " Which entity do you mean by \"" + trim(pending_reference.phrase) + "\"?";
                if (!pending_reference.candidates.empty()) {
                    reply += " Candidates:"; for (const auto& name : pending_reference.candidates) reply += " " + name;
                    reply += ".";
                }
                reply += '\n'; status = "clarification"; return true;
            }
        }
        query.relation[0] = relations[0]; query.relation[1] = relations[1];
        std::vector<int> args;
        for (const auto& w : tokens(question)) {
            auto it = entities.find(w);
            if (it != entities.end()) {
                form += "$" + std::to_string(args.size()) + " "; args.push_back(it->second);
            } else form += w + " ";
        }
        if (!complete) {
            status=event_issue.empty()?"incomplete_facts":event_issue;
            if((uncertain_events||partial_observation) && !event_issue.empty()) {
                reply=status=="conflicting_event_evidence"?" The event evidence conflicts.\n":status=="conflicting_observation"?" The observations conflict.\n":" I do not know.\n"; return true;
            }
            return false;
        }
        if (args.empty() || args.size() > 2) { status = "unbound_arguments"; return false; }
        query.a = {0, args[0]}; query.b = {0, args.size() == 2 ? args[1] : args[0]};
        query.facts = facts; prepared = true;
        if(partial_observation)query.knowledge=std::make_shared<FactKnowledge>(observations);
        auto it = forms.find(form);
        if (it == forms.end()) {
            if (!shared_meanings) { status = "unknown_question"; return false; }
            interpretation = grounding.parse(form, phrase_fragments);
            if (!interpretation.found) {
                status = interpretation.exhausted ? "parse_budget" : interpretation.ambiguous ? "ambiguous_question" : "unknown_question";
                return false;
            }
            if((uncertain_events||partial_observation) && possible_worlds.size()>1) return answer_possible_worlds(interpretation.meaning);
            auto text = render(interpretation.meaning.evaluate(memory, query));
            if (text.empty()) { if(partial_observation&&(memory.missing_evidence||memory.nonunique_result))return missing_observation();status = "failed_grounded_call"; return false; }
            reply = " " + text + "\n"; status = "grounded_answer"; return true;
        }
        query.cue = it->second;
        if((uncertain_events||partial_observation) && possible_worlds.size()>1) return answer_possible_worlds(question_meaning());
        auto out = memory.answer(query); auto text = render(out);
        if (text.empty()) { if(partial_observation&&(memory.missing_evidence||memory.nonunique_result))return missing_observation();status = "no_procedure_or_failed_call"; return false; }
        reply = " " + text + "\n"; status = "answer"; return true;
    }
    bool missing_observation() { reply=" I do not know.\n";status=memory.missing_evidence?"missing_observation":"nonunique_result";prepared=false;return true; }
    bool answer_possible_worlds(const Meaning& meaning) {
        if(meaning.type()<0) { status="unknown_question"; return false; }
        std::set<Value> answers;
        for(const auto& world:possible_worlds) {
            FlatQuery q=query; q.facts.clear(); for(auto [s,r,o]:world.facts) q.facts.push_back({s,r,o});
            if(partial_observation)q.knowledge=std::make_shared<FactKnowledge>(world.knowledge);
            auto answer=meaning.evaluate(memory,q);
            if(!answer.present()) { reply=" I do not know.\n"; status="uncertain_answer"; prepared=false; return true; }
            answers.insert(answer);
        }
        if(answers.size()!=1) { reply=" I do not know.\n"; status="uncertain_answer"; prepared=false; return true; }
        reply=" "+render(*answers.begin())+"\n"; status="grounded_answer"; return true;
    }
    // Explicit verified feedback only. Merely reading or emitting an assistant
    // answer never calls this method. No test target is accepted by prepare().
    bool add_verified_answer(const std::string& text) {
        if (!prepared || possible_worlds.size()>1) return false;
        std::string answer = trim(text); Value v;
        if (answer == "yes" || answer == "no") v = {1, answer == "yes"};
        else if (!answer.empty() && answer.size() < 9 &&
                 std::all_of(answer.begin(), answer.end(), [](unsigned char c) { return std::isdigit(c); }))
            v = {2, std::stoi(answer)};
        else { auto t = tokens(answer); if (t.size() == 1 && entities.count(t[0])) v = {0, entities.at(t[0])}; }
        if (!v.present()) return false;
        auto it = forms.find(form);
        if (it == forms.end()) it = forms.emplace(form, (int)forms.size()).first;
        query.cue = it->second;
        // Any new evidence revokes a cached procedure until it is reverified.
        memory.procedures.erase(query.cue);
        grounding.rules.clear();
        evidence[query.cue].push_back({query, v}); return true;
    }
    size_t learn(bool no_bindings = false) {
        size_t found = 0;
        for (const auto& [cue, examples] : evidence) found += memory.learn(examples, no_bindings).found;
        if (shared_meanings) {
            std::vector<SemanticGrounding::Example> meanings;
            for (const auto& [form, cue] : forms) {
                auto it = memory.procedures.find(cue); if (it == memory.procedures.end()) continue;
                int arity = form.find("$1") == std::string::npos ? 1 : 2;
                meanings.push_back({form, Meaning::from_program(it->second, arity)});
            }
            grounding.fit(meanings);
        }
        return found;
    }
    void finish_segment(char delimiter) {
        std::string s = trim(segment); segment.clear();
        if (s.empty() || assistant) return;
        if (delimiter == '?') question = s;
        else read_fact(s);
    }
    void begin_dialogue_role(StreamRole role) {
        stream_role=role;assistant=role==StreamRole::ASSISTANT;
        segment.clear();line_has_role=true;
        if(role==StreamRole::USER) {
            ++request_revision;
            request.clear();
            if(pending_text.empty())question.clear();
            prepared=false;dialogue_question_pending=false;
        } else if(role==StreamRole::ASSISTANT) {
            if(dialogue_question_pending) {
                if(prepare()) {
                    if(status=="clarification")++clarification_requests;
                    else if(status=="answer"||status=="grounded_answer")++answered;
                    else ++uncertain_replies;
                } else ++abstained;
            } else {
                prepared=false;reply.clear();reply_offset=0;status="awaiting_environment";
            }
            dialogue_question_pending=false;
            if(pending_text.empty()&&!turn_events.empty()) {
                reference_context=std::move(turn_events);turn_events.clear();
            }
        }
    }
    void observe_dialogue_byte(uint8_t c) {
        if(dialogue_line_overflow) {
            if(c=='\n'){dialogue_line_overflow=false;line_has_role=false;segment.clear();}
            return;
        }
        segment+=(char)c;
        if(segment.size()>4096) {
            if(stream_role==StreamRole::CONTEXT||stream_role==StreamRole::ENVIRONMENT) {
                complete=false;event_issue="observation_budget";
            } else if(stream_role==StreamRole::USER) {
                prepared=false;dialogue_question_pending=false;request.clear();status="input_budget";
            }
            segment.clear();dialogue_line_overflow=true;return;
        }
        if(c=='>'&&trim(segment)=="<|endofdialogue|>"){reset_episode();return;}
        // Markers are envelopes at the start of a physical line. A second
        // marker inside that line remains text, including inside a request.
        if(c==':'&&!line_has_role) {
            const auto marker=trim(segment);
            if(marker=="USER:"){begin_dialogue_role(StreamRole::USER);return;}
            if(marker=="ASSISTANT:"){begin_dialogue_role(StreamRole::ASSISTANT);return;}
            if(marker=="ENVIRONMENT:"){begin_dialogue_role(StreamRole::ENVIRONMENT);return;}
            if(marker=="CONTEXT:"){begin_dialogue_role(StreamRole::CONTEXT);return;}
        }
        if(c!='\n')return;
        std::string text=trim(segment);segment.clear();line_has_role=false;
        if(text.empty())return;
        if(stream_role==StreamRole::CONTEXT||stream_role==StreamRole::ENVIRONMENT) {
            // An observation can contain multiple sentences whose references
            // belong to this one message. Its meaning is learned as a whole.
            read_fact(text);
            ++observation_revision;last_observation_environment=stream_role==StreamRole::ENVIRONMENT;
            last_observation_text=text;
            if(last_observation_environment)awaiting_action_observation=false;
        } else if(stream_role==StreamRole::USER) {
            if(text.back()=='?') {
                question=trim(text.substr(0,text.size()-1));dialogue_question_pending=true;
            } else if(grounded_references&&!pending_text.empty()&&pending_question&&tokens(text).size()==1) {
                // A user can resolve a pending question reference without
                // turning ordinary commands or claims into world observations.
                read_fact(text);dialogue_question_pending=true;
            } else if(grounded_requests) {
                request=text;
            }
        }
    }
    void observe(uint8_t c, bool self) {
        // A caller may supply a reply prefix (often the space after the colon).
        // Matching observed bytes advance it without granting learning credit.
        if (reply_offset < reply.size() && c == (uint8_t)reply[reply_offset]) ++reply_offset;
        else { reply.clear(); reply_offset = 0; }
        if (self) return; // generated facts, role markers and answers are never evidence
        if(environment_dialogue){observe_dialogue_byte(c);return;}
        segment += (char)c;
        if (segment.size() > 1024) { complete = false; segment.clear(); return; }
        if (c == '>' && trim(segment) == "<|endofdialogue|>") { reset_episode(); return; }
        if (c == ':') {
            auto marker = trim(segment);
            if (marker == "USER:") { assistant = false; segment.clear(); if (pending_text.empty()) question.clear(); prepared = false; return; }
            if (marker == "ASSISTANT:") {
                segment.clear(); assistant = true;
                if (prepare()) {
                    if (status == "clarification") ++clarification_requests;
                    else if(status=="answer"||status=="grounded_answer") ++answered;
                    else ++uncertain_replies;
                } else ++abstained;
                if (pending_text.empty() && !turn_events.empty()) { reference_context = std::move(turn_events); turn_events.clear(); }
                return;
            }
        }
        if (c == '.' || c == '?' || c == '!' || c == '\n') {
            segment.pop_back(); finish_segment((char)c);
        }
    }
    bool boost(float* out, float& total) {
        if (reply_offset >= reply.size()) return false;
        std::fill(out, out + 256, 0.f);
        total = 1.f; out[(uint8_t)reply[reply_offset]] = total;
        ++forced_bytes; return true;
    }
};
} // namespace laminar::reasoning
