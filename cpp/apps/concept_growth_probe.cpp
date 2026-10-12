#include "laminar/concept_memory.h"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace laminar::concepts;
static void read_world(std::istream& in,Scene& scene,std::vector<std::string>& args) {
    int complete=0,n=0; in>>complete>>n;
    if(n<1||n>8||(complete!=0&&complete!=1)) throw std::runtime_error("bad query");
    scene.complete=bool(complete); args.resize(n); for(auto& a:args) in>>a;
    in>>n; if(n<0||n>256) throw std::runtime_error("bad scene");
    scene.facts.resize(n);
    for(auto& a:scene.facts) { in>>a.predicate>>n; if(n<1||n>8) throw std::runtime_error("bad atom"); a.args.resize(n); for(auto& v:a.args) in>>v; }
}
static void write_clause(const Clause& c) {
    std::cout<<' '<<c.arity<<' '<<c.body.size();
    for(const auto& a:c.body) { std::cout<<' '<<a.predicate<<' '<<a.vars.size(); for(int v:a.vars) std::cout<<' '<<v; }
}
int main(int argc,char** argv) {
    if(argc!=3) { std::cerr<<"usage: concept_growth_probe corpus reuse|flat|wide\n"; return 2; }
    const std::string mode=argv[2]; if(mode!="reuse"&&mode!="flat"&&mode!="wide") return 2;
    Limits limits; if(mode=="wide") limits.body=8;
    ConceptMemory memory(mode=="reuse",limits); std::ifstream input(argv[1]); if(!input) return 2;
    std::string line,stage="none"; size_t count=0;
    while(std::getline(input,line)) {
        ++count; if(line.empty()) continue;
        std::istringstream in(line); std::string op; in>>op;
        if(op=="T") {
            Receipt r; int value; in>>r.id>>r.name>>value; if(value!=0&&value!=1) return 3; r.answer=bool(value); read_world(in,r.scene,r.args);
            std::cout<<"T "<<r.id<<' '<<memory.teach(std::move(r))<<'\n';
        } else if(op=="V") {
            std::string id; unsigned version; int value,active; in>>id>>version>>value>>active;
            if((value!=0&&value!=1)||(active!=0&&active!=1)) return 3;
            std::cout<<"V "<<id<<' '<<memory.revise(id,version,bool(value),bool(active))<<'\n';
        } else if(op=="R") {
            in>>stage; memory.rebuild(); std::cout<<"R "<<stage<<' '<<memory.schemas()<<' '<<memory.receipts().size()<<'\n';
            for(const auto& [name,m]:memory.models()) {
                std::cout<<"M "<<stage<<' '<<name<<' '<<m.status<<' '<<m.arity<<' '<<m.hypotheses.size()<<' '<<m.candidates<<' '<<m.work<<'\n';
                for(size_t i=0;i<m.hypotheses.size();++i) {
                    std::cout<<"D "<<stage<<' '<<name<<' '<<i; write_clause(m.hypotheses[i].local); std::cout<<'\n';
                    std::cout<<"E "<<stage<<' '<<name<<' '<<i; write_clause(m.hypotheses[i].expanded); std::cout<<'\n';
                }
            }
        } else if(op=="Q") {
            std::string id,name; Scene s; std::vector<std::string> args; in>>id>>name; read_world(in,s,args);
            std::cout<<"A "<<stage<<' '<<id<<' '<<int(memory.predict(name,s,args))<<'\n';
        } else throw std::runtime_error("unknown input operation");
        std::string extra;
        if(!in || (in>>extra)) throw std::runtime_error("invalid/trailing input at line "+std::to_string(count));
    }
}
