#include "grounded_bootstrap.h"
#include "laminar/organism.h"
#include <iostream>
using namespace laminar;
using namespace laminar::reasoning;
void check(bool okay,const std::string& message){if(!okay)throw std::runtime_error(message);}
void attach(Organism& org,const std::shared_ptr<GroundedSession>& s){org.attach_grounded_observer(s->reader,[s](uint8_t c,bool self){s->observe(c,self);});}
void feed(Organism& org,const std::string& text,size_t chunk) {
    org.free_run=false;
    for(size_t pos=0;pos<text.size();pos+=chunk) {
        std::string data(1,char(org.last_symbol()));data+=text.substr(pos,chunk);org.run_block((const uint8_t*)data.data(),data.size());
    }
}
std::string generate(Organism& org) {
    org.free_run=true;std::string out;
    for(int i=0;i<1024;++i){uint8_t pair[]={org.last_symbol(),0};org.run_block(pair,2);const auto& p=org.distribution();auto c=uint8_t(std::max_element(p.begin(),p.end())-p.begin());org.commit_outcome(c);out+=char(c);if(c=='\n')break;}
    org.free_run=false;return LanguageReasoner::trim(out);
}
std::string knowledge(const LanguageReasoner& r) {
    std::string out;const auto& k=r.observations;
    for(auto [a,b,c]:k.positive)out+="p,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b,c]:k.negative)out+="n,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b]:k.closed_rows)out+="r,"+r.names[a]+","+std::to_string(b)+";";
    for(auto [a,b]:k.closed_columns)out+="c,"+std::to_string(a)+","+r.names[b]+";";return out;
}
int main(int argc,char** argv) {
    check(argc==2,"expected seed");int seed=std::stoi(argv[1]);size_t chunk=std::vector<size_t>{1,7,113}.at(seed);
    auto session=GroundedBootstrap::load("logs/grounded_session/bootstrap.paths",seed);
    const auto refs=session->references.signature(),commands=session->corrections.signature(),events=session->reader->events.signature(),observations=session->reader->observation_language.signature(),grammar=session->reader->grounding.signature();
    const auto forms=session->reader->forms;
    const auto programs=session->reader->memory.procedures;
    const auto weights=session->reader->memory.skills.kernel.routes.W;
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p);attach(org,session);
    const auto root="logs/grounded_session/"+std::to_string(seed);
    std::ofstream trace(root+".turns.tsv"),states(root+".states.tsv"),records(root+".records.tsv"),models(root+".model.txt");
    trace<<"episode\tturn\treply\troute\tstatus\trequests\tquestions\treports\tedits\tclarifications\trefusals\tresets\tobservation\thistory\tpending\tprobe_isolated\n";
    states<<"episode\tturn\tscope\tknowledge\n";records<<"episode\tturn\tid\trevision\tactive\tsource\ttext\n";models<<refs<<commands;
    auto in=GroundedBootstrap::open("logs/grounded_session/dialogues.inputs");int episodes=std::stoi(GroundedBootstrap::line(in));
    for(int e=0;e<episodes;++e) {
        auto header=GroundedBootstrap::fields(GroundedBootstrap::line(in));int turns=std::stoi(header[1]);
        for(int t=0;t<turns;++t) {
            auto h=GroundedBootstrap::fields(GroundedBootstrap::line(in));size_t length=std::stoull(h[2]);std::string text(length,'\0');in.read(text.data(),length);check(bool(in)&&in.get()=='\n',"bad input framing");
            session->report_source=session->user_source=h[0];auto before=session->requests;
            if(h[1]=="1") {
                org.free_run=true;for(uint8_t c:text){uint8_t pair[]={org.last_symbol(),0};org.run_block(pair,2);org.commit_outcome(c);}org.free_run=false;
            } else feed(org,text,chunk);
            std::string reply;if(session->requests!=before)reply=generate(org);
            bool isolated=false;
            if(t==13) {
                const auto stamp=RevisableGrounding::state_stamp(*session->reader);auto changes=session->journal.changes.size();auto pending=session->dialogue.correction.pending;
                auto probe=session->clone();attach(org,probe);
                feed(org,"\n<|endofdialogue|>\nCONTEXT: evalperson has the evalitem.\nUSER: Who has the evalitem now?\nASSISTANT:",chunk);
                check(generate(org)=="evalperson","probe answer");attach(org,session);
                isolated=RevisableGrounding::state_stamp(*session->reader)==stamp&&session->journal.changes.size()==changes&&pending&&session->dialogue.correction.pending&&pending->token==session->dialogue.correction.pending->token;
                check(isolated,"probe contaminated training history");
            }
            trace<<e<<'\t'<<t<<'\t'<<reply<<'\t'<<session->route<<'\t'<<session->status<<'\t'<<session->requests<<'\t'<<session->questions<<'\t'<<session->reports<<'\t'<<session->edits<<'\t'<<session->clarifications<<'\t'<<session->refusals<<'\t'<<session->resets<<'\t'<<session->reader->observation_revision<<'\t'<<session->reader->history_revision<<'\t'<<bool(session->dialogue.correction.pending)<<'\t'<<isolated<<'\n';
            states<<e<<'\t'<<t<<"\tcurrent\t"<<knowledge(*session->reader)<<'\n';
            for(const auto& record:session->journal.reports) {
                records<<e<<'\t'<<t<<'\t'<<record.id<<'\t'<<record.revision<<'\t'<<record.active<<'\t'<<record.source<<'\t'<<record.text<<'\n';
                auto historical=session->journal.at(record.id);check(bool(historical),"missing historical state");states<<e<<'\t'<<t<<'\t'<<record.id<<'\t'<<knowledge(*historical)<<'\n';
            }
            check(refs==session->references.signature()&&commands==session->corrections.signature()&&events==session->reader->events.signature()&&observations==session->reader->observation_language.signature()&&grammar==session->reader->grounding.signature()&&forms==session->reader->forms&&programs==session->reader->memory.procedures&&weights==session->reader->memory.skills.kernel.routes.W,"inference changed learned models");
        }
    }
    std::cerr<<"seed="<<seed<<" chunk="<<chunk<<" "<<session->report()<<'\n';
}
