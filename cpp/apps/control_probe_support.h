#pragma once
// Evaluator-only wire format and physics, shared with the relational-session probe.
#include "laminar/relational_session.h"
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace laminar::relational;
namespace concepts=laminar::concepts;
void require(bool okay,const std::string& message){if(!okay)throw std::runtime_error(message);}
struct World { Scene scene;std::vector<std::string> names; };
World read_world(std::istream& in) {
    World w;int complete,n;in>>complete>>n;require(n>=0&&n<=32&&(complete==0||complete==1),"bad world header");w.scene.complete=bool(complete);
    w.names.resize(n);for(auto& s:w.names)in>>s;in>>n;require(n>=0&&n<=256,"bad fact count");w.scene.facts.resize(n);
    for(auto& a:w.scene.facts){in>>a.predicate>>n;require(n>0&&n<=8,"bad atom arity");a.args.resize(n);for(auto& v:a.args)in>>v;}
    require(bool(in),"truncated world");return w;
}
void write_world(const Scene& w) {
    std::cout<<' '<<int(w.complete)<<" 0 "<<w.facts.size();
    for(const auto& a:w.facts){std::cout<<' '<<a.predicate<<' '<<a.args.size();for(const auto& v:a.args)std::cout<<' '<<v;}
}
Action read_action(std::istream& in) {
    Action a;int n;in>>a.schema>>n;require(n>0&&n<=4,"bad action arity");a.arguments.resize(n);for(auto& v:a.arguments)in>>v;return a;
}
void write_action(const Action& a) {std::cout<<' '<<a.schema<<' '<<a.arguments.size();for(const auto& v:a.arguments)std::cout<<' '<<v;}
concepts::Clause read_clause(std::istream& in) {
    concepts::Clause c;int n;in>>c.arity>>n;require(n>0&&n<=32&&c.arity>0&&c.arity<=8,"bad oracle clause");
    for(int i=0;i<n;++i){concepts::Literal a;int arity;in>>a.predicate>>arity;require(arity>0&&arity<=8,"bad literal");a.vars.resize(arity);for(int& v:a.vars){in>>v;require(v>=0&&v<32,"bad variable");}c.body.push_back(a);}return c;
}
void write_clause(const concepts::Clause& c,std::ostream& out=std::cout) {
    out<<' '<<c.arity<<' '<<c.body.size();
    for(const auto& a:c.body){out<<' '<<a.predicate<<' '<<a.vars.size();for(int v:a.vars)out<<' '<<v;}
}
// Evaluator-owned private world mechanics and terminal satisfaction. Nothing in
// this object is reachable through the learner's issue/plan/answer interface.
struct Environment {
    struct Effect {std::string predicate;int subject,object,kind;};
    struct Goal {std::vector<std::string> args;concepts::Clause clause;bool negate=false;};
    std::map<std::string,Effect> effects;
    std::map<std::string,Goal> goals;
    struct ComputationalGoal {int family=0;std::string a,b,r,s;};
    std::map<std::string,ComputationalGoal> computations;
    std::map<std::string,concepts::Clause> derived;
    explicit Environment(const std::string& file) {
        std::ifstream input(file);require(bool(input),"missing private evaluator");std::string line;
        while(std::getline(input,line)) {
            std::istringstream in(line);std::string op,name;in>>op>>name;
            if(op=="E"){Effect e;in>>e.predicate>>e.subject>>e.object>>e.kind;effects[name]=e;}
            else if(op=="G"){Goal g;int n,negate;in>>n;require(n>0&&n<=8,"bad evaluator goal");g.args.resize(n);for(auto& v:g.args)in>>v;in>>negate;g.negate=bool(negate);g.clause=read_clause(in);goals[name]=std::move(g);}
            else if(op=="H"){auto c=read_clause(in);require(c.arity==2,"nonbinary derived computation relation");derived[name]=std::move(c);}
            else if(op=="Q"){ComputationalGoal q;in>>q.family>>q.a>>q.b>>q.r>>q.s;require(q.family>=0&&q.family<3,"invalid computational goal");computations[name]=std::move(q);}
            else require(false,"bad evaluator record");
            std::string extra;require(bool(in)&&!(in>>extra),"invalid evaluator record");
        }
    }
    Scene execute(const Action& a,Scene w)const {
        const auto& effect=effects.at(a.schema);auto subject=a.arguments.at(effect.subject),object=a.arguments.at(effect.object);
        w.facts.erase(std::remove_if(w.facts.begin(),w.facts.end(),[&](const auto& atom){
            return atom.predicate==effect.predicate&&((effect.kind==1&&atom.args==std::vector<std::string>{subject,object})
                ||(effect.kind==2&&atom.args[0]==subject)||(effect.kind==3&&atom.args[1]==object));
        }),w.facts.end());
        if(effect.kind!=1)w.facts.push_back({effect.predicate,{subject,object}});
        return sorted(w);
    }
    bool satisfied(const std::string& id,const Scene& scene)const {
        auto q=computations.find(id);
        if(q!=computations.end()) {
            const auto& g=q->second;std::set<std::string> entities{g.a,g.b};
            for(const auto& atom:scene.facts)entities.insert(atom.args.begin(),atom.args.end());
            auto relation=[&](const std::string& name) {
                std::set<std::pair<std::string,std::string>> pairs;
                auto definition=derived.find(name);
                if(definition!=derived.end()) {
                    for(const auto& a:entities)for(const auto& b:entities){concepts::Meter meter(20000000);if(concepts::proves(definition->second,scene,{a,b},meter))pairs.emplace(a,b);}
                }else for(const auto& atom:scene.facts)if(atom.predicate==name&&atom.args.size()==2)pairs.emplace(atom.args[0],atom.args[1]);
                return pairs;
            };
            auto r=relation(g.r),s=relation(g.s);
            auto follow=[](const auto& pairs,std::string x)->std::optional<std::string> {
                std::set<std::string> seen;
                while(seen.insert(x).second) {
                    std::set<std::string> next;for(const auto& [a,b]:pairs)if(a==x)next.insert(b);
                    if(next.empty())return x;if(next.size()!=1)return {};x=*next.begin();
                }
                return {};
            };
            auto count=[](const auto& pairs,const std::string& x){size_t n=0;for(const auto& [a,b]:pairs)n+=a==x;return n;};
            if(g.family==0)return count(r,g.a)==count(s,g.b);
            auto a=follow(r,g.a);if(!a)return false;
            if(g.family==1){auto b=follow(s,g.b);return b&&*a==*b;}
            return count(s,*a)==count(s,g.b);
        }
        const auto& goal=goals.at(id);concepts::Meter meter(20000000);return concepts::proves(goal.clause,scene,goal.args,meter)!=goal.negate;
    }
};
void dump(const RelationalSession& s,const std::string& stage,std::ostream& out=std::cout) {
    out<<"R "<<stage<<' '<<s.version()<<' '<<s.ready()<<' '<<s.concepts().receipts().size()<<' '<<s.language_receipts().size()<<' '<<s.transition_receipts().size()<<' '<<s.execution_receipts().size()<<' '<<s.procedures().size()<<'\n';
    for(const auto& [name,model]:s.concepts().models()) {
        out<<"C "<<name<<' '<<model.status<<' '<<model.arity<<' '<<model.hypotheses.size()<<'\n';
        for(const auto& h:model.hypotheses){
            out<<"D "<<name;write_clause(h.local,out);out<<'\n';
            for(const auto& c:h.local_extra){out<<"DX "<<name;write_clause(c,out);out<<'\n';}
            out<<"E "<<name;write_clause(h.expanded,out);out<<'\n';
            for(const auto& c:h.expanded_extra){out<<"EX "<<name;write_clause(c,out);out<<'\n';}
        }
    }
    for(const auto& [name,f]:s.frames()) {
        out<<"L "<<std::quoted(name)<<' '<<f.evidence<<' '<<f.meanings.size()<<'\n';
        for(const auto& m:f.meanings){out<<"I "<<std::quoted(name)<<' '<<m.predicate<<' '<<m.negate<<' '<<m.slots.size();for(int v:m.slots)out<<' '<<v;out<<'\n';}
    }
    for(const auto& [name,f]:s.computational_frames()) {
        auto p=s.primitives().memory.procedures.find(f.cue);
        out<<"K "<<std::quoted(name)<<' '<<f.cue<<' '<<f.evidence<<' '<<f.learned<<' '<<(p==s.primitives().memory.procedures.end()?0:p->second.size());
        if(p!=s.primitives().memory.procedures.end())for(int a:p->second)out<<' '<<a;
        out<<'\n';
    }
    if(!s.primitives().receipts().empty()) {
        out<<"V "<<s.primitives().receipts().size();for(bool available:s.primitives().memory.available_skills)out<<' '<<available;out<<'\n';
    }
    for(const auto& [name,f]:s.motors().frames) {
        out<<"F "<<name<<' '<<f.arity<<' '<<f.scenes.size()<<' '<<f.candidates.size()<<'\n';
        for(const auto& r:f.candidates) {
            std::string pred;for(const auto& [p,id]:s.binary_ids())if(id==r.relation)pred=p;
            out<<"B "<<name<<' '<<pred<<' '<<r.subject<<' '<<r.object<<' '<<r.effect<<'\n';
        }
    }
    for(const auto& p:s.procedures()) {
        out<<"P "<<std::quoted(p.frame)<<' '<<p.code.size();
        for(const auto& i:p.code){out<<' '<<i.schema<<' '<<i.operands.size();for(const auto& o:i.operands)out<<' '<<o.argument<<' '<<std::quoted(o.predicate)<<' '<<o.source_slot<<' '<<o.target_slot;}
        out<<'\n';
    }
    out<<"END\n";
}
