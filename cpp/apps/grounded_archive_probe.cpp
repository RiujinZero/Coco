#include "grounded_feedback.h"
#include "laminar/organism.h"
#include <iostream>
using namespace laminar;
using namespace laminar::reasoning;
void check(bool okay,const std::string& message){if(!okay)throw std::runtime_error(message);}
std::string knowledge(const LanguageReasoner& r) {
    if(!r.complete)return "unavailable";std::string out;const auto& k=r.observations;
    for(auto [a,b,c]:k.positive)out+="p,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b,c]:k.negative)out+="n,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b]:k.closed_rows)out+="r,"+r.names[a]+","+std::to_string(b)+";";
    for(auto [a,b]:k.closed_columns)out+="c,"+std::to_string(a)+","+r.names[b]+";";return out;
}
std::string generate(Organism& org) {
    org.free_run=true;std::string out;
    for(int i=0;i<256;++i){uint8_t pair[]={org.last_symbol(),0};org.run_block(pair,2);const auto& p=org.distribution();auto c=uint8_t(std::max_element(p.begin(),p.end())-p.begin());org.commit_outcome(c);out+=char(c);if(c=='\n')break;}
    org.free_run=false;return LanguageReasoner::trim(out);
}
int main(int argc,char** argv) try {
    check(argc==2,"expected seed");int seed=std::stoi(argv[1]);size_t chunk=std::vector<size_t>{1,7,113}.at(seed);
    auto s=GroundedBootstrap::load("logs/grounded_session/bootstrap.paths",seed);s->teachers={"verified_teacher"};s->enable_archive();
    auto events=s->reader->events.signature(),grammar=s->reader->grounding.signature(),refs=s->references.signature(),commands=s->corrections.signature();
    auto forms=s->reader->forms;auto programs=s->reader->memory.procedures;auto weights=s->reader->memory.skills.kernel.routes.W;
    GroundedFeedbackStream teaching;teaching.load("logs/grounded_archive/feedback.tsv");
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=true;p.structural=true;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p);org.attach_grounded_observer(s->reader,[s](uint8_t c,bool generated){s->observe(c,generated);});
    std::string root="logs/grounded_archive/"+std::to_string(seed);uint64_t offset=0;
    std::ofstream trace(root+".turns.tsv"),states(root+".states.tsv"),models(root+".models.tsv"),evidence(root+".evidence.tsv");
    trace<<"turn\toffset\treply\troute\tstatus\tconsistent\tfeedback\tarchive_revision\n";
    states<<"turn\tscope\tknowledge\n";models<<"turn\tframe\tsupport\trules\n";
    evidence<<"turn\tepisode\treport\tid\trevision\tactive\tquestion\tanswer\tsource\n";
    auto input=GroundedBootstrap::open("logs/grounded_archive/turns.inputs");int turns=std::stoi(GroundedBootstrap::line(input));
    for(int t=0;t<turns;++t) {
        size_t length=std::stoull(GroundedBootstrap::line(input));std::string text(length,'\0');input.read(text.data(),length);check(bool(input)&&input.get()=='\n',"invalid input turn");
        auto requests=s->requests;
        for(size_t pos=0;pos<text.size();) {
            size_t n=std::min({uint64_t(chunk),uint64_t(text.size()-pos),teaching.next_offset()-offset});
            std::string bytes(1,char(org.last_symbol()));bytes+=text.substr(pos,n);org.run_block((const uint8_t*)bytes.data(),bytes.size());org.end_block();
            pos+=n;offset+=n;teaching.deliver(offset,*s);
        }
        auto reply=s->requests!=requests?generate(org):"";
        trace<<t<<'\t'<<offset<<'\t'<<reply<<'\t'<<s->route<<'\t'<<s->status<<'\t'<<s->archive->consistent<<'\t'<<s->feedback_accepted<<'\t'<<s->archive->revision<<'\n';
        states<<t<<"\tcurrent\t"<<knowledge(*s->reader)<<'\n';
        for(const auto& [id,episode]:s->archive->episodes)for(const auto& record:episode.reports) {
            if(auto historical=s->archive->at(id,record.id))states<<t<<'\t'<<id<<':'<<record.id<<'\t'<<knowledge(*historical)<<'\n';
            for(const auto& [key,e]:record.evidence)evidence<<t<<'\t'<<id<<'\t'<<record.id<<'\t'<<key<<'\t'<<e.revision<<'\t'<<e.active<<'\t'<<e.question<<'\t'<<e.answer<<'\t'<<e.source<<'\n';
        }
        for(const auto& [key,f]:s->reader->observation_language.frames)if(!s->initial_frames.count(key)) {
            models<<t<<'\t'<<key<<'\t'<<f.scenes.size()<<'\t';for(const auto& rule:f.candidates)models<<rule.signature()<<';';models<<'\n';
        }
    }
    check(teaching.cursor==teaching.rows.size()&&s->archive->consistent,"incomplete teaching stream");
    check(events==s->reader->events.signature()&&grammar==s->reader->grounding.signature()&&refs==s->references.signature()&&commands==s->corrections.signature()&&forms==s->reader->forms&&programs==s->reader->memory.procedures&&weights==s->reader->memory.skills.kernel.routes.W,"prior skill changed");
    std::ofstream model(root+".model.txt");model<<s->reader->observation_language.signature();
    std::cout<<"seed="<<seed<<" feedback="<<s->feedback_accepted<<" archived="<<s->archive->episodes.size()<<" "<<s->report()<<'\n';
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
