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
void write_clause(const concepts::Clause& c) {
    std::cout<<' '<<c.arity<<' '<<c.body.size();
    for(const auto& a:c.body){std::cout<<' '<<a.predicate<<' '<<a.vars.size();for(int v:a.vars)std::cout<<' '<<v;}
}
// Evaluator-owned private world mechanics and terminal satisfaction. Nothing in
// this object is reachable through the learner's issue/plan/answer interface.
struct Environment {
    struct Effect {std::string predicate;int subject,object,kind;};
    struct Goal {std::vector<std::string> args;concepts::Clause clause;bool negate=false;};
    std::map<std::string,Effect> effects;
    std::map<std::string,Goal> goals;
    explicit Environment(const std::string& file) {
        std::ifstream input(file);require(bool(input),"missing private evaluator");std::string line;
        while(std::getline(input,line)) {
            std::istringstream in(line);std::string op,name;in>>op>>name;
            if(op=="E"){Effect e;in>>e.predicate>>e.subject>>e.object>>e.kind;effects[name]=e;}
            else if(op=="G"){Goal g;int n,negate;in>>n;require(n>0&&n<=8,"bad evaluator goal");g.args.resize(n);for(auto& v:g.args)in>>v;in>>negate;g.negate=bool(negate);g.clause=read_clause(in);goals[name]=std::move(g);}
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
        const auto& goal=goals.at(id);concepts::Meter meter(20000000);return concepts::proves(goal.clause,scene,goal.args,meter)!=goal.negate;
    }
};
void dump(const RelationalSession& s,const std::string& stage) {
    std::cout<<"R "<<stage<<' '<<s.version()<<' '<<s.ready()<<' '<<s.concepts().receipts().size()<<' '<<s.language_receipts().size()<<' '<<s.transition_receipts().size()<<' '<<s.execution_receipts().size()<<' '<<s.procedures().size()<<'\n';
    for(const auto& [name,model]:s.concepts().models()) {
        std::cout<<"C "<<name<<' '<<model.status<<' '<<model.arity<<' '<<model.hypotheses.size()<<'\n';
        for(const auto& h:model.hypotheses){std::cout<<"D "<<name;write_clause(h.local);std::cout<<'\n';std::cout<<"E "<<name;write_clause(h.expanded);std::cout<<'\n';}
    }
    for(const auto& [name,f]:s.frames()) {
        std::cout<<"L "<<std::quoted(name)<<' '<<f.evidence<<' '<<f.meanings.size()<<'\n';
        for(const auto& m:f.meanings){std::cout<<"I "<<std::quoted(name)<<' '<<m.predicate<<' '<<m.negate<<' '<<m.slots.size();for(int v:m.slots)std::cout<<' '<<v;std::cout<<'\n';}
    }
    for(const auto& [name,f]:s.motors().frames) {
        std::cout<<"F "<<name<<' '<<f.arity<<' '<<f.scenes.size()<<' '<<f.candidates.size()<<'\n';
        for(const auto& r:f.candidates) {
            std::string pred;for(const auto& [p,id]:s.binary_ids())if(id==r.relation)pred=p;
            std::cout<<"B "<<name<<' '<<pred<<' '<<r.subject<<' '<<r.object<<' '<<r.effect<<'\n';
        }
    }
    for(const auto& p:s.procedures()) {
        std::cout<<"P "<<std::quoted(p.frame)<<' '<<p.code.size();
        for(const auto& i:p.code){std::cout<<' '<<i.schema<<' '<<i.operands.size();for(const auto& o:i.operands)std::cout<<' '<<o.argument<<' '<<std::quoted(o.predicate)<<' '<<o.source_slot<<' '<<o.target_slot;}
        std::cout<<'\n';
    }
    std::cout<<"END\n";
}
int main(int argc,char** argv)try {
    require(argc==5,"usage: relational_session_probe input evaluator seed reuse|search");
    bool reuse=std::string(argv[4])=="reuse";require(reuse||std::string(argv[4])=="search","bad mode");
    RelationalSession learner(unsigned(std::stoul(argv[3])));Environment env(argv[2]);std::ifstream input(argv[1]);require(bool(input),"missing input");
    std::string line;
    while(std::getline(input,line)) {
        if(line.empty())continue;
        std::istringstream in(line);std::string op;in>>op;
        if(op=="C") {
            concepts::Receipt r;int value;in>>r.id>>r.name>>value;auto w=read_world(in);r.scene=w.scene;r.args=w.names;r.answer=bool(value);
            auto id=r.id;bool accepted=learner.teach_concept(std::move(r));std::cout<<"T C "<<id<<' '<<accepted<<' '<<learner.version()<<'\n';require(accepted,"concept receipt rejected");
        } else if(op=="L") {
            LanguageReceipt r;int value;in>>r.id>>value;auto w=read_world(in);r.scene=w.scene;r.names={w.names.begin(),w.names.end()};r.answer=bool(value);in>>std::quoted(r.text);
            auto id=r.id;bool accepted=learner.teach_language(std::move(r));std::cout<<"T L "<<id<<' '<<accepted<<' '<<learner.version()<<'\n';require(accepted,"language receipt rejected");
        } else if(op=="T") {
            TransitionReceipt r;in>>r.id;r.action=read_action(in);r.before=read_world(in).scene;r.after=read_world(in).scene;
            auto id=r.id;bool accepted=learner.teach_transition(std::move(r));std::cout<<"T T "<<id<<' '<<accepted<<' '<<learner.version()<<'\n';require(accepted,"transition receipt rejected");
        } else if(op=="R") {
            std::string stage;in>>stage;learner.rebuild();dump(learner,stage);
        } else if(op=="Q") {
            std::string id,text;in>>id;auto w=read_world(in);in>>std::quoted(text);
            std::cout<<"Q "<<id<<' '<<learner.version()<<' '<<int(learner.answer(text,w.scene))<<'\n';
        } else if(op=="G"||op=="A") {
            std::string id,text;size_t budget,n;int depth;in>>id>>budget>>depth;auto w=read_world(in);in>>std::quoted(text)>>n;require(n<=128,"too many actions");std::vector<Action> available;
            for(size_t i=0;i<n;++i)available.push_back(read_action(in));
            if(op=="G") {
                auto control=learner.plan(text,w.scene,available,budget,depth,false);auto control_world=w.scene;
                for(size_t i=0;i<control.actions.size();++i){control_world=env.execute(control.actions[i],control_world);std::cout<<"Y "<<id<<' '<<i;write_action(control.actions[i]);write_world(control_world);std::cout<<'\n';}
                std::cout<<"K "<<id<<' '<<control.status<<' '<<control.predictions<<' '<<control.actions.size()<<' '<<env.satisfied(id,control_world)<<'\n';
            }
            auto result=learner.issue(text,w.scene,available,op=="A",budget,depth,reuse);
            std::vector<Scene> observed;auto world=w.scene;
            for(size_t i=0;i<result.actions.size();++i){world=env.execute(result.actions[i],world);observed.push_back(world);std::cout<<"X "<<id<<' '<<i;write_action(result.actions[i]);write_world(world);std::cout<<'\n';}
            bool success=env.satisfied(id,world),accepted=false;
            if(result.token){accepted=learner.feedback(result.token,observed,std::vector<bool>(observed.size(),true),success);require(accepted,"own trace rejected");learner.rebuild();}
            std::cout<<"G "<<id<<' '<<result.version<<' '<<result.status<<' '<<result.reused<<' '<<result.predictions<<' '<<result.actions.size()<<' '<<success<<' '<<result.token<<' '<<learner.version()<<' '<<accepted<<'\n';
            if(accepted)dump(learner,"after_"+id);
        } else if(op=="VC"||op=="VL") {
            std::string id;unsigned version;int answer;in>>id>>version>>answer;
            bool accepted=op=="VC"?learner.revise_concept(id,version,bool(answer)):learner.revise_language(id,version,bool(answer));
            std::cout<<"V "<<op<<' '<<id<<' '<<accepted<<' '<<learner.version()<<'\n';require(accepted,"revision rejected");
        } else require(false,"unknown stream record");
        std::string extra;require(bool(in)&&!(in>>extra),"bad/trailing stream record: "+op);
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
