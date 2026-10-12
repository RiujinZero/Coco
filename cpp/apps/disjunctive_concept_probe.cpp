#include "laminar/concept_memory.h"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace laminar::concepts;
static void require(bool okay){if(!okay)throw std::runtime_error("invalid probe input");}
static void world(std::istream& input,Scene& scene,std::vector<std::string>& args) {
    int complete=0,n=0;input>>complete>>n;require((complete==0||complete==1)&&n>0&&n<=8);
    scene.complete=bool(complete);args.resize(n);for(auto& a:args)input>>a;
    input>>n;require(n>=0&&n<=256);scene.facts.resize(n);
    for(auto& atom:scene.facts){input>>atom.predicate>>n;require(n>0&&n<=8);atom.args.resize(n);for(auto& a:atom.args)input>>a;}
}
static void write(const Formula& formula) {
    std::cout<<' '<<formula.size();
    for(const auto& clause:formula){std::cout<<' '<<clause.arity<<' '<<clause.body.size();for(const auto& atom:clause.body){std::cout<<' '<<atom.predicate<<' '<<atom.vars.size();for(int v:atom.vars)std::cout<<' '<<v;}}
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;std::string mode=argv[2];Limits limits;limits.body=2;limits.clauses=2;limits.expanded_clauses=8;
    if(mode=="wide"){limits.body=8;limits.clauses=1;}
    else if(mode=="wide_union"){limits.body=4;limits.clauses=4;}
    else if(mode!="union"&&mode!="flat")return 2;
    ConceptMemory memory(mode=="union",limits);std::ifstream file(argv[1]);if(!file)return 2;
    std::string line,stage="initial";
    while(std::getline(file,line)) {
        if(line.empty())continue;std::istringstream input(line);std::string op;input>>op;
        if(op=="T") {
            Receipt r;int answer;input>>r.id>>r.name>>answer;require(answer==0||answer==1);r.answer=bool(answer);world(input,r.scene,r.args);
            std::string id=r.id;std::cout<<"T "<<id<<' '<<memory.teach(std::move(r))<<'\n';
        }else if(op=="V") {
            std::string id;unsigned version;int answer,active;input>>id>>version>>answer>>active;require((answer==0||answer==1)&&(active==0||active==1));
            std::cout<<"V "<<id<<' '<<memory.revise(id,version,bool(answer),bool(active))<<'\n';
        }else if(op=="R") {
            input>>stage;memory.rebuild();std::cout<<"R "<<stage<<'\n';
            for(const auto& [name,model]:memory.models()) {
                std::cout<<"M "<<stage<<' '<<name<<' '<<model.status<<' '<<model.hypotheses.size()<<' '<<model.candidates<<' '<<model.work<<'\n';
                for(size_t i=0;i<model.hypotheses.size();++i){const auto& h=model.hypotheses[i];std::cout<<"D "<<stage<<' '<<name<<' '<<i;write(local_clauses(h));std::cout<<'\n';std::cout<<"E "<<stage<<' '<<name<<' '<<i;write(expanded_clauses(h));std::cout<<'\n';}
            }
        }else if(op=="Q") {
            std::string id,name;Scene scene;std::vector<std::string> args;input>>id>>name;world(input,scene,args);
            std::cout<<"A "<<stage<<' '<<id<<' '<<int(memory.predict(name,scene,args))<<'\n';
        }else throw std::runtime_error("unknown probe operation");
        std::string extra;require(bool(input)&&!(input>>extra));
    }
}
