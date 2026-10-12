#include "grounding_probe_support.h"
#include "laminar/incremental_grounding.h"
using namespace grounding_probe;
using Episode=BootstrapGrounding::Episode;
std::vector<Episode> read(const std::string& path,bool targets) {
    std::ifstream in(path);if(!in)throw std::runtime_error("missing "+path);std::vector<Episode> examples;
    while(in.peek()!=EOF){auto h=fields(line(in));Episode e;e.id=10000+std::stoull(h[0]);for(int i=0;i<std::stoi(h[1]);++i)e.observations.push_back(line(in));e.question=line(in);if(targets)e.answer=line(in);examples.push_back(std::move(e));}return examples;
}
int main(int argc,char** argv) {
    assert(argc==2);int seed=std::stoi(argv[1]);auto reader=std::make_shared<LanguageReasoner>(seed);train_skills(*reader,seed);
    BootstrapGrounding bootstrap;assert(bootstrap.fit(read("logs/bootstrap_grounding/train.txt",true),reader->memory));assert(bootstrap.publish(*reader));
    if(seed%2) {
        for(auto& [key,frame]:reader->events.frames){std::set<EventGrounding::Rule> rules;for(auto r:frame.candidates){r.relation^=1;rules.insert(r);}frame.candidates=rules;}
        for(auto& [cue,p]:reader->memory.procedures)for(int& op:p)if((op>=4&&op<=7)||op>=10)op^=1;
    }
    auto old_forms=reader->forms;auto old_programs=reader->memory.procedures;
    auto old_events=reader->events;IncrementalGrounding learner;
    bool learned=learner.fit(read("logs/incremental_grounding/train.txt",true),*reader);
    std::string root="logs/incremental_grounding/"+std::to_string(seed);std::ofstream model(root+".model.tsv");
    model<<"status\t"<<learner.status<<"\taccepted\t"<<learner.accepted<<"\trejected\t"<<learner.rejected<<"\tchecked\t"<<learner.checked<<"\tskipped\t"<<learner.skipped<<"\trounds\t"<<learner.rounds<<'\n';
    for(const auto& [form,programs]:learner.questions)for(const auto& p:programs){model<<"question\t"<<form<<'\t';for(int op:p)model<<op<<',';model<<'\n';}
    for(const auto& [key,rules]:learner.frames)for(const auto& r:rules)model<<"event\t"<<key<<'\t'<<r.relation<<','<<r.subject<<','<<r.object<<','<<r.effect<<'\n';
    for(const auto& [relation,key]:learner.orientation_gauges)model<<"orientation\t"<<relation<<'\t'<<key<<'\n';
    model.flush();std::cerr<<"seed="<<seed<<" "<<learner.status<<" accepted="<<learner.accepted<<" rejected="<<learner.rejected<<" checked="<<learner.checked<<" skipped="<<learner.skipped<<'\n';
    if(!learned)return 1;assert(learner.publish(*reader));
    for(const auto& [f,c]:old_forms)assert(reader->forms.at(f)==c&&reader->memory.procedures.at(c)==old_programs.at(c));
    for(const auto& [key,frame]:old_events.frames)assert(reader->events.frames.at(key).candidates==frame.candidates);
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p);org.attach_reasoner(reader);auto test=read("logs/incremental_grounding/test.txt",false);
    const auto grammar=reader->grounding.signature(),events=reader->events.signature();const auto forms=reader->forms;const auto programs=reader->memory.procedures;const auto weights=reader->memory.skills.kernel.routes.W;
    std::ofstream trace(root+".trace.tsv");trace<<"mode\tid\tstatus\treply\tfacts\n";
    for(const std::string mode:{"learned","no_shared_meanings"}) {
      reader->shared_meanings=mode=="learned";
      for(const auto& e:test) {
        std::string prompt="\n<|endofdialogue|>\n";for(const auto& s:e.observations)prompt+=s+'\n';prompt+="USER: "+e.question+"\n\nASSISTANT:";
        feed(org,prompt);auto status=reader->status;auto before=state(*reader);std::string reply=reader->reply.empty()?"<no proposal>":generate(org);assert(before==state(*reader));
        std::string facts;for(const auto& [a,r,b]:before)facts+=a+","+std::to_string(r)+","+b+";";
        trace<<mode<<'\t'<<e.id-10000<<'\t'<<status<<'\t'<<reply<<'\t'<<facts<<'\n';
      }
    }
    assert(grammar==reader->grounding.signature()&&events==reader->events.signature()&&forms==reader->forms&&programs==reader->memory.procedures&&weights==reader->memory.skills.kernel.routes.W);
    std::cerr<<"seed="<<seed<<" mixed evaluation complete\n";
}
