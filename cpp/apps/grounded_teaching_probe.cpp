#include "grounded_feedback.h"
#include "laminar/organism.h"
#include <iostream>
using namespace laminar;
using namespace laminar::reasoning;
void check(bool good,const std::string& message){if(!good)throw std::runtime_error(message);}
std::string knowledge(const LanguageReasoner& r) {
    std::string out;const auto& k=r.observations;
    for(auto [a,b,c]:k.positive)out+="p,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b,c]:k.negative)out+="n,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b]:k.closed_rows)out+="r,"+r.names[a]+","+std::to_string(b)+";";
    for(auto [a,b]:k.closed_columns)out+="c,"+std::to_string(a)+","+r.names[b]+";";return out;
}
std::string generate(Organism& org) {
    org.free_run=true;std::string out;
    for(int i=0;i<256;++i) {
        uint8_t pair[]={org.last_symbol(),0};org.run_block(pair,2);const auto& p=org.distribution();
        auto c=uint8_t(std::max_element(p.begin(),p.end())-p.begin());org.commit_outcome(c);out+=char(c);if(c=='\n')break;
    }
    org.free_run=false;return LanguageReasoner::trim(out);
}
int main(int argc,char** argv) try {
    check(argc==2||argc==3,"expected seed [no-feedback]");int seed=std::stoi(argv[1]);bool learning=argc==2;
    auto s=GroundedBootstrap::load("logs/grounded_session/bootstrap.paths",seed);s->teachers={"verified_teacher"};
    auto refs=s->references.signature(),commands=s->corrections.signature(),events=s->reader->events.signature(),grammar=s->reader->grounding.signature();
    auto prior=s->reader->observation_language;auto forms=s->reader->forms;auto programs=s->reader->memory.procedures;auto weights=s->reader->memory.skills.kernel.routes.W;
    GroundedFeedbackStream teaching;if(learning)teaching.load("logs/grounded_teaching/feedback.tsv");
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=true;p.structural=true;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p);org.attach_grounded_observer(s->reader,[s](uint8_t c,bool generated){s->observe(c,generated);});
    size_t chunk=std::vector<size_t>{1,7,113}.at(seed);uint64_t offset=0;
    std::string root="logs/grounded_teaching/"+std::to_string(seed)+(learning?"":".no-feedback");
    std::ofstream trace(root+".turns.tsv"),states(root+".states.tsv"),models(root+".models.tsv");
    trace<<"turn\toffset\treply\troute\tstatus\tconsistent\tfeedback\n";
    states<<"turn\tscope\tknowledge\n";models<<"turn\tframe\tsupport\trules\n";
    auto in=GroundedBootstrap::open("logs/grounded_teaching/turns.inputs");int count=std::stoi(GroundedBootstrap::line(in));
    for(int t=0;t<count;++t) {
        size_t length=std::stoull(GroundedBootstrap::line(in));std::string text(length,'\0');in.read(text.data(),length);check(bool(in)&&in.get()=='\n',"invalid turn");
        auto requests=s->requests;
        for(size_t pos=0;pos<text.size();) {
            size_t n=std::min({uint64_t(chunk),uint64_t(text.size()-pos),teaching.next_offset()-offset});
            std::string data(1,char(org.last_symbol()));data+=text.substr(pos,n);org.run_block((const uint8_t*)data.data(),data.size());org.end_block();
            pos+=n;offset+=n;teaching.deliver(offset,*s);
        }
        auto reply=s->requests!=requests?generate(org):"";
        trace<<t<<'\t'<<offset<<'\t'<<reply<<'\t'<<s->route<<'\t'<<s->status<<'\t'<<s->journal.consistent<<'\t'<<s->feedback_accepted<<'\n';
        states<<t<<"\tcurrent\t"<<knowledge(*s->reader)<<'\n';
        for(const auto& record:s->journal.reports)if(auto historical=s->journal.at(record.id))states<<t<<'\t'<<record.id<<'\t'<<knowledge(*historical)<<'\n';
        for(const auto& key:s->acquired_frames) {
            const auto& f=s->reader->observation_language.frames.at(key);models<<t<<'\t'<<key<<'\t'<<f.scenes.size()<<'\t';
            for(const auto& rule:f.candidates)models<<rule.signature()<<';';models<<'\n';
        }
        if(learning)check(s->reader->complete,"incomplete turn "+std::to_string(t)+": "+s->status);
        for(const auto& [key,f]:prior.frames)check(s->reader->observation_language.frames.at(key).candidates==f.candidates,"prior observation changed");
    }
    check(teaching.cursor==teaching.rows.size(),"unused feedback");
    check(refs==s->references.signature()&&commands==s->corrections.signature()&&events==s->reader->events.signature()&&grammar==s->reader->grounding.signature()&&forms==s->reader->forms&&programs==s->reader->memory.procedures&&weights==s->reader->memory.skills.kernel.routes.W,"prior skill changed");
    std::ofstream model(root+".model.txt");model<<s->reader->observation_language.signature();
    std::cout<<"seed="<<seed<<" feedback="<<s->feedback_accepted<<" new_frames="<<s->acquired_frames.size()<<" byte_units="<<org.live_units()<<" "<<s->report()<<'\n';
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
