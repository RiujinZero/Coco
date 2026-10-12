#pragma once
// Acquire new report meanings from verified consequence answers in a live
// episode. No action prediction, operation label or post-state is a target.
#include "answer_grounding.h"
#include "reasoning_language.h"
#include <optional>

namespace laminar::reasoning {
struct LiveObservationGrounding {
    using Rule=EventGrounding::Rule;
    using Facts=LanguageReasoner::FactSet;
    struct Question {std::string text;FlatQuery query;Meaning meaning;};
    struct Proposal {int index=-1;size_t partitions=0,score=0,evaluations=0;std::string status;};
    struct Pending {
        LanguageReasoner before;
        std::string text;
        uint64_t scene;
        AnswerGrounding::Binding binding;
        size_t feedback=0;
    };
    struct Receipt {
        std::string previous_model;
        RequestGrounding::Snapshot before;
        uint64_t episode=0,request=0,observation=0;
    };
    AnswerGrounding learner;
    std::optional<Pending> pending;
    std::optional<Receipt> receipt;
    std::map<AnswerGrounding::EvidenceId,std::pair<std::string,std::string>> verified;
    std::set<uint64_t> scenes;
    std::string status;
    bool begin(const LanguageReasoner& reader,const std::string& text,uint64_t scene) {
        if(pending){status="observation_pending";return false;}
        // These facts are explicitly the pre-action/pre-report snapshot. Clearing
        // its pending-outcome flag does not make a claim about the current world.
        auto historical=reader;historical.awaiting_action_observation=false;
        EventGrounding::World world;
        if(reader.partial_observation||!GroundedActuator::complete_world(historical.request_snapshot(),world)) {
            status="uncertain_before_state";return false;
        }
        if(scenes.count(scene)){status="reused_scene";return false;}
        auto binding=AnswerGrounding::bind_event(text,reader.names);
        if(binding.arguments.size()<2||binding.arguments.size()>4){status="unbound_arguments";return false;}
        if((reader.events.frames.count(binding.key)||reader.events.alternatives(text).matched)&&!learner.frames.count(binding.key)) {status="prior_observation";return false;}
        pending=Pending{std::move(historical),text,scene,std::move(binding),0};receipt.reset();scenes.insert(scene);
        status="awaiting_observation";return true;
    }
    std::set<Rule> candidates() const {
        if(!pending)return {};
        EventGrounding model;learner.publish(model);const auto& p=*pending;
        auto found=model.frames.find(p.binding.key);
        if(found!=model.frames.end())return found->second.candidates;
        std::set<Rule> out;int n=(int)p.binding.arguments.size();
        for(int r:p.before.relations)for(int a=0;a<n;++a)for(int b=0;b<n;++b)for(int e=0;e<4;++e)out.insert({r,a,b,e});
        return out;
    }
    Facts consequence(const Rule& rule) const {
        const auto& p=*pending;Facts before;for(const auto& f:p.before.facts)before.emplace(f.subject,f.relation,f.object);
        return EventGrounding::update(before,rule.relation,rule.effect,p.binding.arguments.at(rule.subject),p.binding.arguments.at(rule.object));
    }
    size_t worlds() const {std::set<Facts> out;for(const auto& rule:candidates())out.insert(consequence(rule));return out.size();}
    std::optional<Question> question(const std::string& text) const {
        if(!pending)return {};
        auto copy=pending->before;copy.question=text;copy.prepare();auto meaning=copy.question_meaning();
        if(!copy.prepared||meaning.type()<0)return {};
        return Question{text,copy.query,std::move(meaning)};
    }
    Value evaluate(const Question& question,const Rule& rule) {
        auto query=question.query;query.facts.clear();for(auto [s,r,o]:consequence(rule))query.facts.push_back({s,r,o});
        return question.meaning.evaluate(pending->before.memory,query);
    }
    Proposal propose(const std::vector<std::string>& menu,size_t budget=8192) {
        Proposal result;result.status="no_distinguishing_question";auto rules=candidates();
        if(rules.empty()){result.status="conflicting_observation_feedback";return result;}
        for(size_t i=0;i<menu.size();++i) {
            auto q=question(menu[i]);if(!q)continue;
            std::map<Value,size_t> groups;
            for(const auto& rule:rules) {
                if(result.evaluations>=budget)return {-1,0,0,result.evaluations,"question_search_budget"};
                ++result.evaluations;++groups[evaluate(*q,rule)];
            }
            size_t largest=0;bool answerable=false;for(const auto& [value,count]:groups){largest=std::max(largest,count);answerable|=value.present();}
            size_t score=rules.size()-largest;
            if(answerable&&(result.index<0||score>result.score)) {result.index=(int)i;result.score=score;result.partitions=groups.size();result.status="question";}
        }
        return result;
    }
    bool matches(const LanguageReasoner& reader) const {
        if(!pending)return false;const auto& p=*pending;
        return reader.episode_revision==p.before.episode_revision&&reader.request_revision==p.before.request_revision&&reader.history_revision==p.before.history_revision&&
            reader.observation_revision==p.before.observation_revision+1&&reader.last_observation_environment&&
            reader.last_observation_text==p.text&&reader.events.signature()==p.before.events.signature()&&
            reader.forms==p.before.forms&&reader.memory.procedures==p.before.memory.procedures&&
            reader.grounding.signature()==p.before.grounding.signature()&&reader.memory.skills.kernel.routes.W==p.before.memory.skills.kernel.routes.W;
    }
    bool feedback(LanguageReasoner& reader,const std::string& text,const std::string& answer,uint64_t observation,bool self_generated=false) {
        if(self_generated){status="self_generated_feedback";return false;}
        if(!matches(reader)){status="stale_observation_feedback";return false;}
        auto q=question(text);if(!q){status="unknown_question";return false;}
        Value value;const auto raw=LanguageReasoner::trim(answer);auto words=EventGrounding::words(raw);
        if(words.size()!=1){status="invalid_answer";return false;}
        const auto& word=words.front();int type=q->meaning.type();
        if(type==1&&(word=="yes"||word=="no"))value={1,word=="yes"};
        else if(type==2&&!raw.empty()&&raw.size()<9&&std::all_of(raw.begin(),raw.end(),[](unsigned char c){return std::isdigit(c);}))value={2,std::stoi(raw)};
        else if(type==0&&pending->before.entities.count(word))value={0,pending->before.entities.at(word)};
        if(!value.present()){status="invalid_answer";return false;}
        auto id=AnswerGrounding::EvidenceId{pending->scene,observation};auto old=verified.find(id);
        auto record=std::make_pair(text,word);
        if(old!=verified.end()) {status=old->second==record?"duplicate_feedback":"conflicting_feedback_id";return old->second==record;}
        auto& p=*pending;
        auto key=learner.teach(p.text,p.before.names,q->query,q->meaning,value,p.before.memory,p.scene,observation);
        if(key!=p.binding.key){status="unbound_feedback";return false;}
        verified[id]=std::move(record);++p.feedback;
        if(candidates().empty()) {reader.complete=false;reader.event_issue="conflicting_observation_feedback";status=reader.event_issue;return true;}
        status="learned";return true;
    }
    bool commit(LanguageReasoner& reader) {
        if(!matches(reader)){status="stale_observation_feedback";return false;}
        const auto rules=candidates();if(rules.empty()){status="conflicting_observation_feedback";return false;}
        auto& p=*pending;
        auto old=p.before.events.frames.find(p.binding.key);
        if(!p.feedback&&(old==p.before.events.frames.end()||old->second.scenes.size()<4)){status="unverified_observation";return false;}
        std::set<Facts> outcomes;for(const auto& rule:rules)outcomes.insert(consequence(rule));
        if(outcomes.size()!=1){status="uncertain_observation";return false;}
        auto revised=p.before.events;learner.publish(revised);
        // Live updates only refine hypotheses. Expanding a previously excluded
        // meaning after a correction requires history replay, outside this API.
        for(const auto& [key,frame]:p.before.events.frames) {
            auto next=revised.frames.find(key);
            if(next==revised.frames.end()||next->second.pattern!=frame.pattern||
               !std::includes(frame.candidates.begin(),frame.candidates.end(),next->second.candidates.begin(),next->second.candidates.end())) {
                status="nonmonotone_observation_revision";return false;
            }
        }
        receipt=Receipt{p.before.events.signature(),p.before.request_snapshot(),reader.episode_revision,reader.request_revision,reader.observation_revision};
        reader.events=std::move(revised);reader.names=p.before.names;reader.entities=p.before.entities;
        reader.facts.clear();for(auto [s,r,o]:*outcomes.begin())reader.facts.push_back({s,r,o});
        // Forgetting old meaning correlations is conservative: every prior
        // world had these same pre-state facts, and every retained current rule
        // has this same post-state. Future ambiguity must still be checked.
        reader.possible_worlds.clear();reader.observations={};reader.world_model_signature=reader.events.signature();
        reader.complete=true;reader.event_issue.clear();reader.awaiting_action_observation=false;
        reader.prepared=false;reader.reply.clear();reader.reply_offset=0;reader.status="verified_observation";
        pending.reset();status="verified_observation";return true;
    }
};
} // namespace laminar::reasoning
