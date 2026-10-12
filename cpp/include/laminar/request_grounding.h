#pragma once
// Learn requested state predicates from explicitly verified satisfaction feedback.
// Learn actuator effects separately from observed, acknowledged transitions.
// Neither interface accepts a correct action, goal binding or language recipe.
#include "event_grounding.h"
#include "fact_knowledge.h"

namespace laminar::reasoning {
struct RequestGrounding {
    using World=EventGrounding::World;
    enum Truth { UNKNOWN=-1, FALSE_VALUE=0, TRUE_VALUE=1 };
    struct State {
        World facts,negative;
        std::set<std::pair<std::string,int>> closed_rows;
        std::set<std::pair<int,std::string>> closed_columns;
        bool closed_world=true;
        bool operator==(const State&) const=default;
        void update(const std::string& a,int r,const std::string& b,int effect) {
            FactKnowledge::update_evidence(facts,negative,closed_rows,closed_columns,a,r,b,effect);
        }
        Truth truth(const std::string& a,int r,const std::string& b) const {
            if(facts.count({a,r,b}))return TRUE_VALUE;
            if(closed_world||negative.count({a,r,b})||closed_rows.count({a,r})||closed_columns.count({r,b}))return FALSE_VALUE;
            return UNKNOWN;
        }
    };
    struct Snapshot { std::vector<State> worlds; bool valid=true;bool operator==(const Snapshot&) const=default; };
    struct Rule {
        int relation=-1,subject=-1,object=-1;bool present=true;
        auto key() const {return std::make_tuple(relation,subject,object,present);}
        bool operator<(const Rule& r) const {return key()<r.key();}
        bool operator==(const Rule& r) const {return key()==r.key();}
    };
    struct Frame {
        std::vector<std::string> pattern;
        std::set<Rule> candidates;
        std::set<uint64_t> scenes;
        bool initialized=false;
    };
    struct Literal {
        std::string subject,object;int relation=-1;bool present=true;
        bool operator<(const Literal& x) const {return std::tie(subject,relation,object,present)<std::tie(x.subject,x.relation,x.object,x.present);}
    };
    struct Meaning {std::set<Literal> alternatives;std::string status="unknown_request";};
    std::map<std::string,Frame> frames;
    std::string status;
    static bool bind(const std::vector<std::string>& pattern,const std::vector<std::string>& input,std::vector<std::string>& args) {
        if(pattern.size()!=input.size())return false;
        args.assign(4,{});
        for(size_t i=0;i<input.size();++i) {
            int slot=EventGrounding::slot(pattern[i]);
            if(slot<0){if(pattern[i]!=input[i])return false;}
            else if(args[slot].empty())args[slot]=input[i];
            else if(args[slot]!=input[i])return false;
        }
        return true;
    }
    static Truth truth(const Snapshot& snapshot,const Literal& literal) {
        if(!snapshot.valid||snapshot.worlds.empty())return UNKNOWN;
        Truth result=UNKNOWN;bool first=true;
        for(const auto& world:snapshot.worlds) {
            auto t=world.truth(literal.subject,literal.relation,literal.object);
            if(t==UNKNOWN)return UNKNOWN;
            if(!literal.present)t=t==TRUE_VALUE?FALSE_VALUE:TRUE_VALUE;
            if(!first&&t!=result)return UNKNOWN;
            result=t;first=false;
        }
        return result;
    }
    bool teach(const std::string& text,const Snapshot& observed,const std::set<std::string>& names,
               const std::set<int>& relations,bool satisfied,uint64_t scene,bool self_generated=false) {
        if(self_generated){status="self_generated_feedback";return false;}
        if(!observed.valid||observed.worlds.empty()){status="missing_observation";return false;}
        auto pattern=EventGrounding::words(text);std::vector<std::string> args;
        // Reuse a learned surface pattern so a newly named argument need not
        // already occur in the current fact set. No semantic role is supplied.
        std::vector<std::string> matches;
        for(const auto& [key,frame]:frames)if(bind(frame.pattern,pattern,args))matches.push_back(key);
        if(matches.size()>1){status="ambiguous_surface";return false;}
        std::string key;
        if(matches.size()==1) {
            key=matches.front();pattern=frames.at(key).pattern;
            bind(pattern,EventGrounding::words(text),args);
        } else {
            args.clear();
            for(auto& word:pattern)if(names.count(word)) {
                args.push_back(word);word="$"+std::to_string(args.size()-1);
            }
            if(args.size()<2||args.size()>4){status="unbound_arguments";return false;}
            key=EventGrounding::join(pattern);
        }
        Frame next;auto old=frames.find(key);
        if(old!=frames.end())next=old->second;
        else {
            next.pattern=pattern;next.initialized=true;
            for(int r:relations)for(size_t a=0;a<args.size();++a)for(size_t b=0;b<args.size();++b)
                for(bool present:{false,true})next.candidates.insert({r,(int)a,(int)b,present});
        }
        if(next.candidates.empty()){status="conflicting_request_evidence";return false;}
        std::set<Rule> supported;
        for(const auto& rule:next.candidates) {
            auto t=truth(observed,{args.at(rule.subject),args.at(rule.object),rule.relation,rule.present});
            // Unknown evidence cannot be silently treated as a negative label.
            // This small learner waits for a fully discriminating observation.
            if(t==UNKNOWN){status="missing_observation";return false;}
            if((t==TRUE_VALUE)==satisfied)supported.insert(rule);
        }
        next.candidates=std::move(supported);next.scenes.insert(scene);frames[key]=std::move(next);
        status=frames.at(key).candidates.empty()?"conflicting_request_evidence":"learned";
        return true;
    }
    Meaning meaning(const std::string& text,bool require_support=true) const {
        Meaning result;auto input=EventGrounding::words(text);std::vector<std::string> args;
        bool insufficient=false;
        for(const auto& [key,frame]:frames)if(bind(frame.pattern,input,args)) {
            if(frame.candidates.empty()){result.alternatives.clear();result.status="conflicting_request_evidence";return result;}
            if(require_support&&frame.scenes.size()<4){insufficient=true;continue;}
            for(const auto& rule:frame.candidates)result.alternatives.insert({args.at(rule.subject),args.at(rule.object),rule.relation,rule.present});
        }
        if(insufficient){result.alternatives.clear();result.status="insufficient_request_evidence";}
        else if(!result.alternatives.empty())result.status=result.alternatives.size()==1?"grounded_request":"ambiguous_request";
        return result;
    }
    Truth evaluate(const std::string& text,const Snapshot& observed) const {
        auto parsed=meaning(text);Truth result=UNKNOWN;bool first=true;
        for(const auto& literal:parsed.alternatives) {
            auto t=truth(observed,literal);
            if(t==UNKNOWN||(!first&&t!=result))return UNKNOWN;
            result=t;first=false;
        }
        return result;
    }
    std::string signature() const {
        std::string out;
        for(const auto& [key,frame]:frames) {
            out+=key+":"+std::to_string(frame.scenes.size())+":";
            for(const auto& r:frame.candidates)out+=std::to_string(r.relation)+","+std::to_string(r.subject)+","+std::to_string(r.object)+","+std::to_string(r.present)+";";
            out+='\n';
        }
        return out;
    }
};

struct GroundedActuator {
    using World=EventGrounding::World;
    struct Action {std::string schema;std::vector<std::string> arguments;};
    struct Frame {size_t arity=0;std::set<EventGrounding::Rule> candidates;std::set<uint64_t> scenes;size_t failures=0;};
    struct Choice {int index=-1;size_t predictions=0;std::string status;};
    std::map<std::string,Frame> frames;
    std::string status;
    // Read-back observations must agree on the complete state. This transition
    // learner does not invent effects from a partial/ambiguous observation.
    static bool complete_world(const RequestGrounding::Snapshot& snapshot,World& world) {
        if(!snapshot.valid||snapshot.worlds.empty())return false;
        world=snapshot.worlds.front().facts;
        for(const auto& w:snapshot.worlds)if(!w.closed_world||w.facts!=world)return false;
        return true;
    }
    bool teach(const Action& action,const RequestGrounding::Snapshot& before,const RequestGrounding::Snapshot& after,
               const std::set<int>& relations,bool executed,uint64_t scene,bool self_generated=false) {
        if(self_generated){status="self_generated_feedback";return false;}
        World prior,next;if(!complete_world(before,prior)||!complete_world(after,next)){status="missing_observation";return false;}
        if(action.arguments.empty()||action.arguments.size()>4){status="invalid_action";return false;}
        auto found=frames.find(action.schema);
        if(found!=frames.end()&&found->second.arity!=action.arguments.size()){status="invalid_action";return false;}
        if(!executed) {
            if(prior!=next){status="inconsistent_failure";return false;}
            if(found!=frames.end())++found->second.failures;
            status="observed_failure";return true;
        }
        Frame frame;if(found!=frames.end())frame=found->second;
        else {
            frame.arity=action.arguments.size();
            for(int r:relations)for(size_t a=0;a<frame.arity;++a)for(size_t b=0;b<frame.arity;++b)
                for(int effect=0;effect<4;++effect)frame.candidates.insert({r,(int)a,(int)b,effect});
        }
        for(auto it=frame.candidates.begin();it!=frame.candidates.end();) {
            if(EventGrounding::update(prior,it->relation,it->effect,action.arguments[it->subject],action.arguments[it->object])!=next)it=frame.candidates.erase(it);else ++it;
        }
        frame.scenes.insert(scene);frames[action.schema]=std::move(frame);
        status=frames.at(action.schema).candidates.empty()?"conflicting_action_evidence":"learned";return true;
    }
    Choice choose(const RequestGrounding& requests,const std::string& request,const RequestGrounding::Snapshot& observed,
                  const std::vector<Action>& available,size_t budget=8192) const {
        Choice result;auto meaning=requests.meaning(request);
        if(meaning.alternatives.empty()){result.status=meaning.status;return result;}
        if(!observed.valid||observed.worlds.empty()){result.status="missing_observation";return result;}
        if(requests.evaluate(request,observed)==RequestGrounding::TRUE_VALUE){result.status="already_satisfied";return result;}
        for(size_t i=0;i<available.size();++i) {
            const auto& action=available[i];auto frame=frames.find(action.schema);
            if(frame==frames.end()||frame->second.scenes.size()<4||frame->second.candidates.empty()||frame->second.arity!=action.arguments.size())continue;
            bool achieves=true;
            for(const auto& rule:frame->second.candidates) {
                if(result.predictions>=budget){result.status="action_search_budget";return result;}++result.predictions;
                auto predicted=observed;
                for(auto& w:predicted.worlds)w.update(action.arguments[rule.subject],rule.relation,action.arguments[rule.object],rule.effect);
                if(requests.evaluate(request,predicted)!=RequestGrounding::TRUE_VALUE){achieves=false;break;}
            }
            if(achieves){result.index=(int)i;result.status="action";return result;}
        }
        result.status="no_satisfying_action";return result;
    }
    std::string signature() const {
        std::string out;
        for(const auto& [key,frame]:frames) {
            out+=key+":"+std::to_string(frame.arity)+":"+std::to_string(frame.scenes.size())+":"+std::to_string(frame.failures)+":";
            for(const auto& r:frame.candidates)out+=std::to_string(r.relation)+","+std::to_string(r.subject)+","+std::to_string(r.object)+","+std::to_string(r.effect)+";";
            out+='\n';
        }
        return out;
    }
};
} // namespace laminar::reasoning
