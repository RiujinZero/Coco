// Evaluation only: use archived neural weights and learned macro programs.
// Raw verified teaching supplies the language adapter's existing typed frames;
// no primitive is acquired; evaluated programs come only from the archive.
#include "control_probe_support.h"
int main(int argc,char** argv)try {
    require(argc==6,"usage: frozen_repair_language_probe seed weights teaching cases observed_types");
    RelationalSession adapter;
    if(std::stoi(argv[5]))adapter.enable_observed_relation_types();
    std::ifstream teaching(argv[3]);require(bool(teaching),"missing teaching file");std::string line;
    while(std::getline(teaching,line)) {
        std::istringstream in(line);std::string op,id;in>>op>>id;
        if((op!="M"&&op!="N")||!id.starts_with("repair_composed"))continue;
        LanguageReceipt r;r.id=id;r.computational=true;int value;in>>value;r.answer=bool(value);
        auto world=read_world(in);r.scene=world.scene;r.names={world.names.begin(),world.names.end()};in>>std::quoted(r.text);
        size_t n;in>>n;for(size_t i=0;i<n;++i){std::string name;in>>name;r.relations.insert(name);}
        if(op=="N"){in>>n;for(size_t i=0;i<n;++i){std::string name;in>>name;r.quantities.insert(name);r.names.erase(name);}}
        require(bool(in)&&adapter.teach_language(r),"invalid frame evidence");
    }
    adapter.rebuild();
    laminar::reasoning::ProcedureMemory model(std::stoul(argv[1]));
    std::ifstream weights(argv[2]);require(bool(weights),"missing weights");
    for(auto& w:model.skills.kernel.routes.W)require(bool(weights>>w)&&std::isfinite(w),"invalid weights");
    float extra;require(!(weights>>extra)&&weights.eof(),"trailing weights");
    auto frozen=model.skills.kernel.routes.W;auto version=adapter.version();
    std::ifstream cases(argv[4]);require(bool(cases),"missing cases");
    std::cout<<"case\toutput\tsteps\n";
    while(std::getline(cases,line)) {
        std::istringstream in(line);int id;in>>id;auto world=read_world(in);std::string text;in>>std::quoted(text);
        size_t n;in>>n;laminar::reasoning::Program program(n);for(int& op:program)in>>op;
        require(bool(in),"invalid evaluation case");
        auto q=adapter.computation_input(text,world.scene);laminar::reasoning::Value result;auto before=model.skill_steps;
        if(q)result=model.execute(*q,program);
        std::cout<<id<<'\t'<<(result.type==1?result.id:-1)<<'\t'<<model.skill_steps-before<<'\n';
    }
    require(frozen==model.skills.kernel.routes.W&&version==adapter.version(),"evaluation changed learned state");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
