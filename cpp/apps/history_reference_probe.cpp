#include "interaction_probe_support.h"
#include "laminar/observation_language_learner.h"
#include "laminar/history_reference_grounding.h"

std::string knowledge(const LanguageReasoner& r) {
    std::string out;const auto& k=r.observations;
    for(auto [a,b,c]:k.positive)out+="p,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b,c]:k.negative)out+="n,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b]:k.closed_rows)out+="r,"+r.names[a]+","+std::to_string(b)+";";
    for(auto [a,b]:k.closed_columns)out+="c,"+std::to_string(a)+","+r.names[b]+";";
    return out;
}
std::string rules(const std::set<HistoryReferenceGrounding::Selector>& values) {
    std::string out;for(const auto& rule:values)out+=rule.signature()+";";return out;
}
std::string ids(const std::set<uint64_t>& values) {std::string out;for(auto id:values)out+=std::to_string(id)+",";return out;}
std::vector<std::string> questions(const Environment& env) {
    std::vector<std::string> out;for(const auto& o:env.objects)out.push_back("Who has the "+o+" now?");
    for(const auto& p:env.people)out.push_back("Where is "+p+"?");return out;
}
void populate(Organism& org,LanguageReasoner& reader,RevisableGrounding& journal,std::ifstream& input,int count) {
    struct Retire{uint64_t id;std::string source;};std::vector<Retire> retired;
    for(int i=0;i<count;++i) {
        auto f=fields(line(input));assert(f.size()==4);auto id=std::stoull(f[0]);
        stream(org,"\nENVIRONMENT: "+f[3]+"\n");assert(journal.append(reader,id,f[1])&&journal.consistent);
        if(f[2]=="0")retired.push_back({id,f[1]});
    }
    for(const auto& r:retired)assert(journal.revise_report(reader,r.id,0,"",r.source,false)&&journal.consistent);
}
int main(int argc,char** argv) {
    assert(argc==2);const int seed=std::stoi(argv[1]);auto prior_reader=prior(seed);
    ObservationLanguageLearner prior_language;assert(prior_language.fit(lessons("logs/observation_language/train.txt"),*prior_reader)&&prior_language.publish(*prior_reader));
    const auto events=prior_reader->events.signature(),grammar=prior_reader->grounding.signature(),observations=prior_reader->observation_language.signature();
    const auto programs=prior_reader->memory.procedures;const auto weights=prior_reader->memory.skills.kernel.routes.W;
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p),qa(p);HistoryReferenceGrounding learner;
    const std::string root="logs/history_reference/"+std::to_string(seed);
    std::ofstream teaching(root+".teaching.tsv"),model(root+".model.txt"),cases(root+".cases.tsv"),answers(root+".answers.tsv"),states(root+".states.tsv"),records_out(root+".records.tsv");
    teaching<<"scene\tevidence\treference\tquestion\tanswer\tframe\tsupport\trules\n";
    cases<<"case\tgroup\tresolution\tcandidates\tapplied\tselected\tstatus\tinitial_reply\tfinal_reply\tinvalid_preserved\tepisode_before\tepisode_after\thistory_before\thistory_after\n";
    answers<<"case\tscope\tquestion\treply\tstatus\n";
    states<<"case\tphase\tscope\tknowledge\n";
    records_out<<"case\tphase\tid\trevision\tactive\tsource\ttext\n";
    auto ask=[&](const LanguageReasoner& reader,const std::string& question) {
        auto copy=std::make_shared<LanguageReasoner>(reader);qa.attach_reasoner(copy);
        stream(qa,"\nUSER: "+question+"\nASSISTANT:");auto answer=generate(qa);return std::make_pair(answer,copy->status);
    };
    std::ifstream train("logs/history_reference/train.txt");assert(train);const int scenes=std::stoi(line(train));size_t lessons=0;
    for(int i=0;i<scenes;++i) {
        auto h=fields(line(train));assert(h.size()==4);auto reader=std::make_shared<LanguageReasoner>(*prior_reader);org.attach_reasoner(reader);
        Environment env(h[1],seed);stream(org,"\n<|endofdialogue|>\n"+env.initial());RevisableGrounding journal;assert(journal.start(*reader));
        populate(org,*reader,journal,train,std::stoi(h[2]));auto descriptors=HistoryReferenceGrounding::records(journal,*reader,seed%2,1-seed%2);
        for(int q=0;q<std::stoi(h[3]);++q) {
            auto lesson=fields(line(train));assert(lesson.size()==3);
            stream(org,"\nUSER: "+lesson[0]+"\nASSISTANT:");
            assert(learner.teach(*reader,journal,descriptors,reader->request,lesson[1],lesson[2],std::stoull(h[0]),q));
            if(learner.status!="learned"){std::cerr<<"teaching "<<h[0]<<" "<<q<<" "<<learner.status<<'\n';return 2;}
            auto bound=HistoryReferenceGrounding::bind(reader->request,descriptors,*reader);auto key=EventGrounding::join(bound.first);const auto& frame=learner.frames.at(key);
            teaching<<h[0]<<'\t'<<q<<'\t'<<lesson[0]<<'\t'<<lesson[1]<<'\t'<<lesson[2]<<'\t'<<key<<'\t'<<frame.scenes.size()<<'\t'<<rules(frame.candidates)<<'\n';++lessons;
        }
    }
    const auto frozen=learner.signature();model<<frozen;
    std::ifstream test("logs/history_reference/test.txt");assert(test);int total=std::stoi(line(test)),applied_count=0;
    for(int i=0;i<total;++i) {
        auto h=fields(line(test));assert(h.size()==4);const auto& group=h[3];auto reader=std::make_shared<LanguageReasoner>(*prior_reader);org.attach_reasoner(reader);
        Environment env(h[1],seed);stream(org,"\n<|endofdialogue|>\n"+env.initial());RevisableGrounding journal;assert(journal.start(*reader));
        populate(org,*reader,journal,test,std::stoi(h[2]));auto request=fields(line(test));if(request.size()==3)request.push_back("");assert(request.size()==4);
        auto capture=[&](const std::string& phase) {
            states<<i<<'\t'<<phase<<"\tcurrent\t"<<knowledge(*reader)<<'\n';
            for(const auto& r:journal.reports) {
                records_out<<i<<'\t'<<phase<<'\t'<<r.id<<'\t'<<r.revision<<'\t'<<r.active<<'\t'<<r.source<<'\t'<<r.text<<'\n';
                auto historical=journal.at(r.id);assert(historical);states<<i<<'\t'<<phase<<'\t'<<r.id<<'\t'<<knowledge(*historical)<<'\n';
            }
        };
        capture("before");auto descriptors=HistoryReferenceGrounding::records(journal,*reader,seed%2,1-seed%2);
        stream(org,"\nUSER: "+request[0]+"\nASSISTANT:");const auto resolution=learner.resolve(*reader,descriptors,reader->request);
        if(resolution.resolved&&group!="generated") {
            auto historical=journal.at(*resolution.candidates.begin());assert(historical);
            for(const auto& q:questions(env)){auto [reply,status]=ask(*historical,q);answers<<i<<"\treference\t"<<q<<'\t'<<reply<<'\t'<<status<<'\n';}
        }
        const auto episode=reader->episode_revision,revision=reader->history_revision;GroundedHistoryCorrection correction;
        bool applied=correction.begin(*reader,journal,learner,descriptors,reader->request,request[3],request[1],std::stoi(request[2]),group=="generated");
        std::string initial_reply=group=="generated"?"":generate(org),final_reply;bool invalid_preserved=false;
        // The clarification ID is not read until after reference resolution and
        // the initial reply. Definite test inputs carry no correct target ID.
        const auto choice=std::stoll(line(test));const auto stale_text=line(test);
        if(choice>=0) {
            assert(correction.pending&&!resolution.resolved);const auto token=correction.pending->token;
            const auto before=RevisableGrounding::state_stamp(*reader),signature=learner.signature();auto changes=journal.changes.size();
            assert(!correction.choose(*reader,journal,learner,descriptors,token,99999));
            invalid_preserved=correction.pending.has_value()&&journal.changes.size()==changes&&RevisableGrounding::state_stamp(*reader)==before&&learner.signature()==signature;
            assert(invalid_preserved);
            if(group=="stale") {
                stream(org,"\nENVIRONMENT: "+stale_text+"\n");assert(journal.append(*reader,99999,"late_sensor"));
                descriptors=HistoryReferenceGrounding::records(journal,*reader,seed%2,1-seed%2);
            }
            stream(org,"\nUSER: "+std::to_string(choice)+"\nASSISTANT:");
            applied=correction.choose(*reader,journal,learner,descriptors,token,choice);final_reply=generate(org);
        }
        const auto selected=applied?std::to_string(journal.changes.back().report):std::string("-1");applied_count+=applied;
        cases<<i<<'\t'<<group<<'\t'<<resolution.status<<'\t'<<ids(resolution.candidates)<<'\t'<<applied<<'\t'<<selected<<'\t'<<correction.status<<'\t'<<initial_reply<<'\t'<<final_reply<<'\t'<<invalid_preserved<<'\t'<<episode<<'\t'<<reader->episode_revision<<'\t'<<revision<<'\t'<<reader->history_revision<<'\n';
        capture("after");for(const auto& q:questions(env)){auto [reply,status]=ask(*reader,q);answers<<i<<"\tcurrent\t"<<q<<'\t'<<reply<<'\t'<<status<<'\n';}
        assert(learner.signature()==frozen&&reader->events.signature()==events&&reader->grounding.signature()==grammar&&reader->observation_language.signature()==observations&&reader->memory.procedures==programs&&reader->memory.skills.kernel.routes.W==weights);
    }
    std::cerr<<"seed="<<seed<<" scenes="<<scenes<<" lessons="<<lessons<<" frames="<<learner.frames.size()<<" cases="<<total<<" applied="<<applied_count<<'\n';
}
