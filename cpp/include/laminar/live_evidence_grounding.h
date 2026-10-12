#pragma once
// Acquire report meanings in an ongoing partially observed episode. Verified
// consequence answers constrain knowledge hypotheses; action predictions are
// never evidence. Existing alternative histories retain their correlations.
#include "reasoning_language.h"
#include <optional>

namespace laminar::reasoning {
struct LiveEvidenceGrounding {
    using Rule=ObservationLanguage::Rule;
    using World=LanguageReasoner::PossibleWorld;
    struct Branch {Rule rule;World world;};
    struct Answer {
        Value value;bool unknown=false,valid=false;
        auto key() const {return std::tie(valid,unknown,value);}
        bool operator<(const Answer& a) const {return key()<a.key();}
        bool operator==(const Answer& a) const {return key()==a.key();}
    };
    struct Question {FlatQuery query;Meaning meaning;};
    struct Proposal {int index=-1;size_t score=0,partitions=0,evaluations=0;std::string status;};
    struct Pending {
        LanguageReasoner before;std::string text,key;std::vector<std::string> pattern;std::vector<int> args;
        uint64_t scene=0;size_t feedback=0;std::vector<Branch> branches;
    };
    struct Receipt {std::string previous_model;RequestGrounding::Snapshot before;};
    std::optional<Pending> pending;std::optional<Receipt> receipt;
    std::set<std::string> owned;
    std::set<uint64_t> scenes;
    std::map<std::pair<uint64_t,uint64_t>,std::pair<std::string,Answer>> feedback_ids;
    std::string status;
    size_t hypothesis_budget=4096;

    bool begin(const LanguageReasoner& reader,const std::string& text,uint64_t scene) {
        if(pending){status="observation_pending";return false;}
        auto before=reader;before.awaiting_action_observation=false;
        if(!before.partial_observation||!before.request_snapshot().valid){status="invalid_before_state";return false;}
        if(scenes.count(scene)){status="reused_scene";return false;}
        // An old frame may have been indistinguishable in a closed world but
        // ambiguous about evidence/coverage. Reground it when its retained
        // event interpretations disagree; exact prior meanings stay anchored.
        EventGrounding::Interpretation fixed_event;
        if(reader.events.interpret(text,fixed_event)){status="prior_observation";return false;}
        auto pattern=EventGrounding::words(text);std::vector<int> args;
        for(auto& word:pattern)if(reader.entities.count(word)) {args.push_back(reader.entities.at(word));word="$"+std::to_string(args.size()-1);}
        if(args.empty()||args.size()>4){status="unbound_arguments";return false;}
        const auto key=EventGrounding::join(pattern);
        auto known=reader.observation_language.alternatives(text,{reader.names.begin(),reader.names.end()});
        if((known.matched||reader.observation_language.frames.count(key))&&!owned.count(key)){status="prior_observation";return false;}
        auto candidates=ObservationLanguage::hypotheses(args.size(),{reader.relations[0],reader.relations[1]},true);
        auto old=reader.observation_language.frames.find(key);
        if(old!=reader.observation_language.frames.end())candidates=old->second.candidates;
        auto worlds=before.possible_worlds;
        if(worlds.empty()){World initial;initial.knowledge=before.observations;initial.facts=initial.knowledge.positive;worlds.push_back(initial);}
        std::vector<Branch> branches;
        for(const auto& rule:candidates)for(auto world:worlds) {
            auto previous=world.meanings.find("observation:"+key);
            if(previous!=world.meanings.end()&&previous->second!=rule.signature())continue;
            if(!ObservationLanguage::apply(world.knowledge,rule,args))continue;
            world.meanings["observation:"+key]=rule.signature();world.facts=world.knowledge.positive;
            branches.push_back({rule,std::move(world)});
            if(branches.size()>hypothesis_budget){status="hypothesis_budget";return false;}
        }
        if(branches.empty()){status="conflicting_observation";return false;}
        pending=Pending{std::move(before),text,key,std::move(pattern),std::move(args),scene,0,std::move(branches)};
        receipt.reset();scenes.insert(scene);status="awaiting_observation";return true;
    }
    std::set<Rule> candidates() const {
        std::set<Rule> out;if(pending)for(const auto& b:pending->branches)out.insert(b.rule);return out;
    }
    size_t knowledge_states() const {
        std::set<FactKnowledge> out;if(pending)for(const auto& b:pending->branches)out.insert(b.world.knowledge);return out.size();
    }
    std::optional<Question> question(const std::string& text) const {
        if(!pending)return {};
        auto copy=pending->before;copy.question=text;copy.prepare();auto meaning=copy.question_meaning();
        if(meaning.type()<0||!copy.query.a.present())return {};
        return Question{copy.query,std::move(meaning)};
    }
    Answer evaluate(const Question& question,const Branch& branch) {
        auto q=question.query;q.facts.clear();for(auto [a,r,b]:branch.world.knowledge.positive)q.facts.push_back({a,r,b});
        q.knowledge=std::make_shared<FactKnowledge>(branch.world.knowledge);
        auto& memory=pending->before.memory;auto value=question.meaning.evaluate(memory,q);
        return {value,!value.present()&&(memory.missing_evidence||memory.nonunique_result),value.present()||memory.missing_evidence||memory.nonunique_result};
    }
    Proposal propose(const std::vector<std::string>& menu,size_t budget=131072) {
        Proposal result;result.status="no_question";
        if(!pending||pending->branches.empty()){result.status="no_pending_hypotheses";return result;}
        for(size_t i=0;i<menu.size();++i) {
            auto q=question(menu[i]);if(!q)continue;
            std::map<Answer,size_t> groups;
            for(const auto& branch:pending->branches) {
                if(result.evaluations>=budget)return {-1,0,0,result.evaluations,"question_search_budget"};
                ++result.evaluations;++groups[evaluate(*q,branch)];
            }
            size_t largest=0;bool answerable=false;
            for(const auto& [answer,n]:groups){largest=std::max(largest,n);answerable|=answer.valid;}
            size_t score=pending->branches.size()-largest;
            if(answerable&&(result.index<0||score>result.score))result={int(i),score,groups.size(),result.evaluations,"question"};
        }
        return result;
    }
    bool matches(const LanguageReasoner& reader) const {
        if(!pending)return false;const auto& p=*pending;
        return reader.partial_observation&&reader.episode_revision==p.before.episode_revision&&reader.request_revision==p.before.request_revision&&reader.history_revision==p.before.history_revision&&
            reader.observation_revision==p.before.observation_revision+1&&reader.last_observation_environment&&reader.last_observation_text==p.text&&
            reader.events.signature()==p.before.events.signature()&&reader.observation_language.signature()==p.before.observation_language.signature()&&
            reader.forms==p.before.forms&&reader.memory.procedures==p.before.memory.procedures&&reader.grounding.signature()==p.before.grounding.signature()&&
            reader.memory.skills.kernel.routes.W==p.before.memory.skills.kernel.routes.W;
    }
    bool feedback(LanguageReasoner& reader,const std::string& question_text,const std::string& text,uint64_t id,bool generated=false) {
        if(generated){status="self_generated_feedback";return false;}
        if(!matches(reader)){status="stale_observation_feedback";return false;}
        auto q=question(question_text);if(!q){status="unknown_question";return false;}
        auto raw=LanguageReasoner::trim(text);auto words=EventGrounding::words(raw);Answer answer;
        if(words.size()==1) {
            const auto& w=words[0];int type=q->meaning.type();
            if(w=="unknown")answer={{},true,true};
            else if(type==1&&(w=="yes"||w=="no"))answer={{1,w=="yes"},false,true};
            else if(type==2&&!raw.empty()&&raw.size()<9&&std::all_of(raw.begin(),raw.end(),[](unsigned char c){return std::isdigit(c);}))answer={{2,std::stoi(raw)},false,true};
            else if(type==0&&pending->before.entities.count(w))answer={{0,pending->before.entities.at(w)},false,true};
        }
        if(!answer.valid){status="invalid_answer";return false;}
        auto evidence=std::make_pair(pending->scene,id);auto record=std::make_pair(question_text,answer);auto old=feedback_ids.find(evidence);
        if(old!=feedback_ids.end()){status=old->second==record?"duplicate_feedback":"conflicting_feedback_id";return old->second==record;}
        std::vector<Branch> surviving;for(const auto& b:pending->branches)if(evaluate(*q,b)==answer)surviving.push_back(b);
        feedback_ids[evidence]=record;++pending->feedback;pending->branches=std::move(surviving);
        if(pending->branches.empty()){reader.complete=false;reader.event_issue="conflicting_observation_feedback";status=reader.event_issue;return true;}
        status="learned";return true;
    }
    bool commit(LanguageReasoner& reader) {
        if(!matches(reader)){status="stale_observation_feedback";return false;}
        auto& p=*pending;auto candidates_=candidates();if(candidates_.empty()){status="conflicting_observation_feedback";return false;}
        auto old=p.before.observation_language.frames.find(p.key);
        if(!p.feedback&&(old==p.before.observation_language.frames.end()||old->second.scenes.size()<4)){status="unverified_observation";return false;}
        std::set<World> worlds;for(const auto& b:p.branches)worlds.insert(b.world);
        if(worlds.size()>reader.world_budget){status="world_budget";return false;}
        auto revised=p.before.observation_language;auto& frame=revised.frames[p.key];
        frame.pattern=p.pattern;frame.candidates=std::move(candidates_);if(p.feedback)frame.scenes.insert(p.scene);
        receipt=Receipt{p.before.observation_language.signature(),p.before.request_snapshot()};
        reader.observation_language=std::move(revised);reader.entities=p.before.entities;reader.names=p.before.names;
        reader.possible_worlds.assign(worlds.begin(),worlds.end());reader.project_worlds();
        reader.world_model_signature=reader.events.signature();reader.observation_language_signature=reader.observation_language.signature();
        reader.complete=true;reader.event_issue.clear();reader.awaiting_action_observation=false;reader.prepared=false;reader.reply.clear();reader.reply_offset=0;
        reader.status="verified_observation";owned.insert(p.key);pending.reset();status="verified_observation";return true;
    }
};
} // namespace laminar::reasoning
