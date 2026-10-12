#pragma once
// Generic evidence operations, with no observation-language words or question
// recipes. A learned frame can preserve alternatives until later evidence.
#include "event_grounding.h"
#include "fact_knowledge.h"

namespace laminar::reasoning {
struct ObservationLanguage {
    enum Kind {IGNORE=0,POSITIVE=1,NEGATIVE=2,ROW=3,COLUMN=4,FORGET_ROW=5,FORGET_COLUMN=6,
               UPDATE_ADD=7,UPDATE_REMOVE=8,UPDATE_ROW=9,UPDATE_COLUMN=10};
    struct Rule {
        int kind=0,relation=-1,focus=-1,mask=0;
        auto key() const {return std::tie(kind,relation,focus,mask);}
        bool operator<(const Rule& r) const {return key()<r.key();}
        bool operator==(const Rule& r) const {return key()==r.key();}
        std::string signature() const {return std::to_string(kind)+","+std::to_string(relation)+","+std::to_string(focus)+","+std::to_string(mask);}
    };
    struct Frame {std::vector<std::string> pattern;std::set<Rule> candidates;std::set<uint64_t> scenes;};
    struct Interpretation {Rule rule;std::vector<std::string> arguments;std::string frame;};
    struct Alternatives {bool matched=false,conflicted=false;std::vector<Interpretation> values;};
    std::map<std::string,Frame> frames;
    static std::set<Rule> hypotheses(int arity,const std::set<int>& relations,bool include_updates=false) {
        std::set<Rule> out{{IGNORE,-1,-1,0}};
        for(int r:relations)for(int a=0;a<arity;++a) {
            for(int b=0;b<arity;++b)for(int kind:{POSITIVE,NEGATIVE})out.insert({kind,r,a,1<<b});
            for(int mask=0;mask<(1<<arity);++mask)for(int kind:{ROW,COLUMN})out.insert({kind,r,a,mask});
            for(int kind:{FORGET_ROW,FORGET_COLUMN})out.insert({kind,r,a,0});
            if(include_updates)for(int b=0;b<arity;++b)for(int kind=UPDATE_ADD;kind<=UPDATE_COLUMN;++kind)out.insert({kind,r,a,1<<b});
        }
        return out;
    }
    static bool apply(FactKnowledge& k,const Rule& r,const std::vector<int>& args) {
        if(r.kind==IGNORE)return true;
        const int focus=args.at(r.focus);std::set<int> values;
        for(size_t i=0;i<args.size();++i)if(r.mask&(1<<i))values.insert(args[i]);
        if(r.kind==POSITIVE||r.kind==NEGATIVE)return values.size()==1&&k.observe(focus,r.relation,*values.begin(),r.kind==POSITIVE);
        if(r.kind==ROW)return k.observe_row(focus,r.relation,values);
        if(r.kind==COLUMN)return k.observe_column(r.relation,focus,values);
        if(r.kind==FORGET_ROW){k.forget_row(focus,r.relation);return true;}
        if(r.kind==FORGET_COLUMN){k.forget_column(r.relation,focus);return true;}
        if(r.kind>=UPDATE_ADD&&r.kind<=UPDATE_COLUMN&&values.size()==1){k.update(focus,r.relation,*values.begin(),r.kind-UPDATE_ADD);return true;}
        return false;
    }
    Alternatives alternatives(const std::string& text,const std::set<std::string>& known_entities={},size_t minimum_scenes=4) const {
        Alternatives result;auto input=EventGrounding::words(text);
        // Literal vocabulary is learned with the frames. Do not invent an
        // entity from a known grammatical word (e.g. an empty-list marker).
        // An independently established entity with that spelling remains a
        // possible reading, preserving genuine ambiguity.
        std::set<std::string> literals;
        for(const auto& [key,frame]:frames)for(const auto& word:frame.pattern)if(EventGrounding::slot(word)<0)literals.insert(word);
        for(const auto& [key,frame]:frames) {
            if(frame.scenes.size()<minimum_scenes||frame.pattern.size()!=input.size())continue;
            std::vector<std::string> args;bool match=true;
            for(size_t i=0;i<input.size();++i) {
                int s=EventGrounding::slot(frame.pattern[i]);
                if(s<0){if(frame.pattern[i]!=input[i]){match=false;break;}}
                else {
                    if(literals.count(input[i])&&!known_entities.count(input[i])){match=false;break;}
                    if(args.size()<=size_t(s))args.resize(s+1);
                    if(!args[s].empty()&&args[s]!=input[i]){match=false;break;}
                    args[s]=input[i];
                }
            }
            if(!match)continue;result.matched=true;
            if(frame.candidates.empty())result.conflicted=true;
            for(const auto& rule:frame.candidates)result.values.push_back({rule,args,key});
        }
        return result;
    }
    std::string signature() const {
        std::string out;
        for(const auto& [key,frame]:frames) {
            out+=key+":"+std::to_string(frame.scenes.size())+":";
            for(const auto& rule:frame.candidates)out+=rule.signature()+";";
            out+='\n';
        }
        return out;
    }
};
} // namespace laminar::reasoning
