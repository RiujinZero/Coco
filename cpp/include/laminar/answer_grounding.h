#pragma once
// Ground novel event wording from verified consequence answers. Question
// meanings come from the already learned reader; no post-event state is accepted.
#include "event_grounding.h"
#include "semantic_grounding.h"

namespace laminar::reasoning {
struct AnswerGrounding {
    using Rule = EventGrounding::Rule;
    using EvidenceId = std::pair<uint64_t, uint64_t>; // physical scene, observation
    struct Frame {
        std::vector<std::string> pattern;
        std::map<EvidenceId, std::set<Rule>> evidence;
    };
    std::map<std::string, Frame> frames;
    struct Binding {std::vector<std::string> pattern;std::vector<int> arguments;std::string key;};
    static Binding bind_event(const std::string& text,const std::vector<std::string>& names) {
        Binding out;out.pattern=EventGrounding::words(text);
        std::map<std::string,int> named,slots;for(size_t i=0;i<names.size();++i)named[names[i]]=(int)i;
        for(auto& word:out.pattern)if(named.count(word)) {
            auto it=slots.find(word);
            if(it==slots.end()){int slot=(int)slots.size();it=slots.emplace(word,slot).first;out.arguments.push_back(named.at(word));}
            word="$"+std::to_string(it->second);
        }
        out.key=EventGrounding::join(out.pattern);return out;
    }
    // Updating the same evidence ID is an explicit correction, not a new vote.
    // Missing feedback and generated answers are not evidence.
    std::string teach(const std::string& text, const std::vector<std::string>& names,
                      const FlatQuery& before, const Meaning& question, Value answer,
                      ProcedureMemory& memory, uint64_t scene, uint64_t observation,
                      bool self_generated = false,
                      const std::vector<EventGrounding::Interpretation>& observed_events = {}) {
        if (self_generated || !answer.present() || question.type() < 0) return {};
        std::map<std::string,int> named; for (int i=0;i<(int)names.size();++i) named[names[i]]=i;
        for(const auto& e:observed_events) if(!named.count(e.subject)||!named.count(e.object)) return {};
        auto binding=bind_event(text,names);const auto& pattern=binding.pattern;const auto& arguments=binding.arguments;
        if(arguments.size()<2 || arguments.size()>4) return {};
        std::set<int> relations; for(int r:before.relation) if(r>=0) relations.insert(r);
        std::set<std::tuple<int,int,int>> world;
        for(const auto& f:before.facts) { world.emplace(f.subject,f.relation,f.object); relations.insert(f.relation); }
        std::set<Rule> possible;
        for(int r:relations) for(int a=0;a<(int)arguments.size();++a) for(int b=0;b<(int)arguments.size();++b)
            for(int effect=0;effect<4;++effect) {
                auto after=EventGrounding::update(world,r,effect,arguments[a],arguments[b]);
                for(const auto& e:observed_events) after=EventGrounding::update(after,e.relation,e.effect,named.at(e.subject),named.at(e.object));
                FlatQuery query=before; query.facts.clear();
                for(auto [s,relation,o]:after) query.facts.push_back({s,relation,o});
                if(question.evaluate(memory,query)==answer) possible.insert({r,a,b,effect});
            }
        const auto key=EventGrounding::join(pattern); auto& frame=frames[key]; frame.pattern=pattern;
        frame.evidence[{scene,observation}]=std::move(possible); return key;
    }
    bool retract(const std::string& key,uint64_t scene,uint64_t observation) {
        auto it=frames.find(key); return it!=frames.end() && it->second.evidence.erase({scene,observation});
    }
    void publish(EventGrounding& events) const {
        for(const auto& [key,frame]:frames) {
            EventGrounding::Frame learned; learned.pattern=frame.pattern;
            for(const auto& [id,possible]:frame.evidence) {
                if(!learned.initialized) { learned.candidates=possible; learned.initialized=true; }
                else {
                    std::set<Rule> common;
                    std::set_intersection(learned.candidates.begin(),learned.candidates.end(),possible.begin(),possible.end(),std::inserter(common,common.begin()));
                    learned.candidates=std::move(common);
                }
                learned.scenes.insert(id.first);
            }
            events.frames[key]=std::move(learned);
        }
    }
    std::string signature() const {
        std::string out;
        for(const auto& [key,f]:frames) {
            out+=key+":";
            for(const auto& [id,possible]:f.evidence) {
                out+=std::to_string(id.first)+","+std::to_string(id.second)+":";
                for(const auto& r:possible) out+=std::to_string(r.relation)+","+std::to_string(r.subject)+","+std::to_string(r.object)+","+std::to_string(r.effect)+";";
                out+="/";
            }
            out+='\n';
        }
        return out;
    }
};
} // namespace laminar::reasoning
