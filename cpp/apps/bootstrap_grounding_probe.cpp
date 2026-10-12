#include "grounding_probe_support.h"
#include "laminar/bootstrap_grounding.h"
using namespace grounding_probe;
using Learner=BootstrapGrounding;
std::vector<Learner::Episode> read(const std::string& path,bool targets) {
    std::ifstream in(path);if(!in)throw std::runtime_error("missing "+path);
    std::vector<Learner::Episode> examples;
    while(in.peek()!=EOF) {
        auto h=fields(line(in));assert(h.size()==2);Learner::Episode e;e.id=std::stoull(h[0]);
        for(int i=0;i<std::stoi(h[1]);++i)e.observations.push_back(line(in));e.question=line(in);
        if(targets)e.answer=line(in);examples.push_back(std::move(e));
    }
    return examples;
}
int main(int argc,char** argv) {
    assert(argc==2||argc==3);int seed=std::stoi(argv[1]);
    std::string directory="logs/bootstrap_grounding";if(argc==3)directory+="/"+std::string(argv[2]);
    auto reader=std::make_shared<LanguageReasoner>(seed);train_skills(*reader,seed);
    auto train=read(directory+"/train.txt",true);Learner bootstrap;
    bool learned=bootstrap.fit(train,reader->memory);
    const auto root=directory+"/"+std::to_string(seed);
    std::ofstream model(root+".model.txt");
    model<<"status="<<bootstrap.status<<" accepted="<<bootstrap.accepted<<" rejected="<<bootstrap.rejected<<" rounds="<<bootstrap.rounds<<'\n';
    for(const auto& m:bootstrap.models) {
        model<<"viable="<<m.viable<<" checked="<<m.checked<<" skipped="<<m.skipped<<'\n';
        model<<Learner::encode(m,false)<<'\n';
    }
    std::cerr<<"seed="<<seed<<" status="<<bootstrap.status<<" accepted="<<bootstrap.accepted<<" rounds="<<bootstrap.rounds<<'\n';
    if(!learned)return 1;
    assert(bootstrap.publish(*reader));
    std::ofstream learned_out(root+".learned.tsv");
    learned_out<<"question\t"<<bootstrap.form<<'\t';for(int op:reader->memory.procedures.begin()->second)learned_out<<op<<',';learned_out<<'\n';
    for(const auto& [key,frame]:reader->events.frames)for(const auto& rule:frame.candidates)
        learned_out<<"event\t"<<key<<'\t'<<rule.relation<<','<<rule.subject<<','<<rule.object<<','<<rule.effect<<'\n';
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p);org.attach_reasoner(reader);
    auto test=read(directory+"/test.txt",false);const auto base=reader->events;
    std::ofstream trace(root+".trace.tsv");trace<<"mode\tid\tstatus\treply\tfacts\n";
    for(const std::string mode:{"learned","append_only","no_observation_grounding"}) {
        reader->events=base;
        if(mode=="append_only")for(auto& [key,frame]:reader->events.frames) {
            std::set<EventGrounding::Rule> rules;for(auto rule:frame.candidates){rule.effect=EventGrounding::ADD;rules.insert(rule);}frame.candidates=rules;
        }
        if(mode=="no_observation_grounding")reader->events.frames.clear();
        const auto events=reader->events.signature();const auto procedures=reader->memory.procedures;
        const auto forms=reader->forms;const auto weights=reader->memory.skills.kernel.routes.W;
        for(const auto& e:test) {
            std::string prompt="\n<|endofdialogue|>\n";
            for(const auto& s:e.observations)prompt+=s+'\n';prompt+="USER: "+e.question+"\n\nASSISTANT:";
            feed(org,prompt);auto status=reader->status;auto before=state(*reader);
            auto reply=reader->reply.empty()?"<no proposal>":generate(org);assert(before==state(*reader));
            std::string facts;for(const auto& [a,r,b]:before)facts+=a+","+std::to_string(r)+","+b+";";
            trace<<mode<<'\t'<<e.id<<'\t'<<status<<'\t'<<reply<<'\t'<<facts<<'\n';
        }
        assert(events==reader->events.signature()&&procedures==reader->memory.procedures&&forms==reader->forms);
        assert(weights==reader->memory.skills.kernel.routes.W);
        std::cerr<<"seed="<<seed<<" "<<mode<<" complete\n";
    }
}
