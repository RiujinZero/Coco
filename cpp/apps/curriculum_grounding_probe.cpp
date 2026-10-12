// Interface readiness, not a full curriculum training claim. Warm-start the
// existing grounded learner; consume only prompt bytes on held-out evaluation.
#include "grounding_probe_support.h"
using namespace grounding_probe;

template<class Callback> void records(const std::string& path, Callback callback) {
    std::ifstream in(path, std::ios::binary);
    if(!in) throw std::runtime_error("missing " + path);
    while(in.peek()!=EOF) {
        auto header=fields(line(in)); assert(header.size()==2);
        size_t n=std::stoull(header[1]); std::string prompt(n,'\0');
        in.read(prompt.data(),n); assert(in && in.get()=='\n');
        callback(std::stoi(header[0]),prompt);
    }
}

int main() {
    auto reader=std::make_shared<LanguageReasoner>(0);
    reader->grounded_events=reader->shared_meanings=true;
    train_skills(*reader,0); train_events(*reader,0);
    Params p; p.seed=0;p.units=128;p.vocab=256;p.train=false;p.structural=false;
    p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p);org.attach_reasoner(reader);assert(train_questions(org,*reader,0)==18);
    // Establish that this executable still answers a learned question before
    // attributing abstentions to a curriculum interface mismatch.
    feed(org,"\n<|endofdialogue|>\nUSER: lena holds book. What is lena?\nASSISTANT:");
    assert(reader->status=="answer"||reader->status=="grounded_answer");
    assert(generate(org)=="lena");
    const auto events=reader->events.signature(),grammar=reader->grounding.signature();
    const auto programs=reader->memory.procedures; const auto forms=reader->forms;
    const auto weights=reader->memory.skills.kernel.routes.W;
    size_t evidence=0;for(const auto& [cue,examples]:reader->evidence)evidence+=examples.size();
    // Native boundary, same role markers as laminar_babi. No target file is
    // opened in this phase, and neither learning method is called.
    std::ofstream trace("logs/curriculum_grounding/native.tsv");
    trace<<"id\tstatus\tcomplete\tfacts\treply\n";
    size_t n=0;
    records("logs/curriculum_grounding/heldout.inputs",[&](int id,const std::string& prompt) {
        feed(org,prompt); auto status=reader->status;
        std::string reply=reader->reply.empty()?"<no grounded proposal>":generate(org);
        assert(reply.find_first_of("\t\r\n")==std::string::npos);
        trace<<id<<'\t'<<status<<'\t'<<reader->complete<<'\t'<<reader->facts.size()<<'\t'<<reply<<'\n';
        if(++n%200==0)std::cerr<<"held-out prompts="<<n<<'\n';
    });
    assert(events==reader->events.signature()&&grammar==reader->grounding.signature());
    assert(programs==reader->memory.procedures&&forms==reader->forms&&weights==reader->memory.skills.kernel.routes.W);
    size_t after=0;for(const auto& [cue,examples]:reader->evidence)after+=examples.size();assert(evidence==after);
    std::ifstream targets("logs/curriculum_grounding/training.targets");assert(targets);
    size_t accepted=0,attempts=0;
    records("logs/curriculum_grounding/training.inputs",[&](int,const std::string& prompt) {
        auto target=line(targets); feed(org,prompt);
        accepted+=reader->add_verified_answer(target);++attempts;
    });
    std::ofstream result("logs/curriculum_grounding/training_acceptance.json");
    result<<"{\"attempts\":"<<attempts<<",\"accepted\":"<<accepted<<",\"pretrained_procedures\":"<<programs.size()<<"}\n";
    std::cout<<"curriculum prompts="<<n<<" training feedback accepted="<<accepted<<'/'<<attempts<<'\n';
}
