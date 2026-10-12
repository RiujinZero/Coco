#include "interaction_probe_support.h"
#include "laminar/observation_language_learner.h"
#include "laminar/correction_language_grounding.h"

std::string knowledge(const LanguageReasoner& r) {
    std::string out;const auto& k=r.observations;
    for(auto [a,b,c]:k.positive)out+="p,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b,c]:k.negative)out+="n,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b]:k.closed_rows)out+="r,"+r.names[a]+","+std::to_string(b)+";";
    for(auto [a,b]:k.closed_columns)out+="c,"+std::to_string(a)+","+r.names[b]+";";
    return out;
}
std::vector<std::string> questions(const Environment& env) {
    std::vector<std::string> out;for(const auto& o:env.objects)out.push_back("Who has the "+o+" now?");
    for(const auto& p:env.people)out.push_back("Where is "+p+"?");return out;
}
void populate(Organism& org,LanguageReasoner& reader,RevisableGrounding& journal,std::ifstream& input,int count) {
    std::vector<std::pair<uint64_t,std::string>> retired;
    for(int i=0;i<count;++i) {
        auto f=fields(line(input));assert(f.size()==4);auto id=std::stoull(f[0]);
        stream(org,"\nENVIRONMENT: "+f[3]+"\n");assert(journal.append(reader,id,f[1])&&journal.consistent);
        if(f[2]=="0")retired.emplace_back(id,f[1]);
    }
    for(const auto& [id,source]:retired)assert(journal.revise_report(reader,id,0,"",source,false)&&journal.consistent);
}
int main(int argc,char** argv) {
    assert(argc==2);const int seed=std::stoi(argv[1]);auto base=prior(seed);
    ObservationLanguageLearner observations;assert(observations.fit(lessons("logs/observation_language/train.txt"),*base)&&observations.publish(*base));
    const auto event_signature=base->events.signature(),grammar=base->grounding.signature(),observation_signature=base->observation_language.signature();
    const auto programs=base->memory.procedures;const auto weights=base->memory.skills.kernel.routes.W;
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p),qa(p);HistoryReferenceGrounding refs;
    auto setup=[&](std::ifstream& in,const std::vector<std::string>& h,RevisableGrounding& j) {
        auto r=std::make_shared<LanguageReasoner>(*base);org.attach_reasoner(r);Environment env(h[1],seed);
        stream(org,"\n<|endofdialogue|>\n"+env.initial());assert(j.start(*r));populate(org,*r,j,in,std::stoi(h[2]));return r;
    };
    std::ifstream reference_train("logs/history_reference/train.txt");assert(reference_train);int scenes=std::stoi(line(reference_train));
    for(int i=0;i<scenes;++i) {
        auto h=fields(line(reference_train));RevisableGrounding j;auto r=setup(reference_train,h,j);
        auto records=HistoryReferenceGrounding::records(j,*r,seed%2,1-seed%2);
        for(int n=0;n<std::stoi(h[3]);++n) {
            auto lesson=fields(line(reference_train));assert(refs.teach(*r,j,records,lesson[0],lesson[1],lesson[2],std::stoull(h[0]),n));
        }
    }
    const auto frozen_refs=refs.signature();CorrectionLanguageGrounding language;
    const auto root="logs/correction_language/"+std::to_string(seed);
    std::ofstream teaching(root+".teaching.tsv"),model(root+".model.txt"),cases(root+".cases.tsv"),states(root+".states.tsv"),records_out(root+".records.tsv"),answers(root+".answers.tsv");
    teaching<<"scene\tutterance\tframe\tsupport\toperations\n";
    cases<<"case\tgroup\tmeaning\toperation\treference\tpayload\tresolution\tcandidates\tapplied\tselected\tstatus\tinitial_reply\tfinal_reply\tinvalid_preserved\tepisode_before\tepisode_after\thistory_before\thistory_after\n";
    states<<"case\tphase\tscope\tknowledge\n";records_out<<"case\tphase\tid\trevision\tactive\tsource\ttext\n";answers<<"case\tquestion\treply\tstatus\n";
    std::ifstream train("logs/correction_language/train.txt");assert(train);int training=std::stoi(line(train));
    for(int i=0;i<training;++i) {
        auto h=fields(line(train));RevisableGrounding j;auto r=setup(train,h,j);auto utterance=line(train);
        std::vector<CorrectionLanguageGrounding::QA> feedback;
        for(int n=0;n<std::stoi(h[3]);++n){auto row=fields(line(train));feedback.emplace_back(row[0],row[1]);}
        auto records=HistoryReferenceGrounding::records(j,*r,seed%2,1-seed%2);
        stream(org,"\nUSER: "+utterance+"\nASSISTANT:");auto state=RevisableGrounding::state_stamp(*r);auto changes=j.changes.size();
        assert(language.teach(*r,j,refs,records,r->request,feedback,std::stoull(h[0]),0));
        assert(RevisableGrounding::state_stamp(*r)==state&&j.changes.size()==changes&&refs.signature()==frozen_refs);
        bool over=false;auto parses=language.parses(*r,refs,records,r->request,over);assert(!over);
        for(const auto& parsed:parses) {
            const auto& frame=language.frames.at(parsed.frame);teaching<<h[0]<<'\t'<<utterance<<'\t'<<parsed.frame<<'\t'<<frame.scenes.size()<<'\t';
            for(int op:frame.candidates)teaching<<op<<',';teaching<<'\n';
        }
    }
    const auto frozen=language.signature();model<<frozen;
    std::ifstream test("logs/correction_language/test.txt");assert(test);int total=std::stoi(line(test)),applied_count=0;
    for(int i=0;i<total;++i) {
        auto h=fields(line(test));const auto& group=h[3];RevisableGrounding j;auto r=setup(test,h,j);auto request=fields(line(test));assert(request.size()==2);
        auto records=HistoryReferenceGrounding::records(j,*r,seed%2,1-seed%2);
        auto capture=[&](const std::string& phase) {
            states<<i<<'\t'<<phase<<"\tcurrent\t"<<knowledge(*r)<<'\n';
            for(const auto& record:j.reports) {
                records_out<<i<<'\t'<<phase<<'\t'<<record.id<<'\t'<<record.revision<<'\t'<<record.active<<'\t'<<record.source<<'\t'<<record.text<<'\n';
                auto historical=j.at(record.id);assert(historical);states<<i<<'\t'<<phase<<'\t'<<record.id<<'\t'<<knowledge(*historical)<<'\n';
            }
        };
        capture("before");stream(org,"\nUSER: "+request[1]+"\nASSISTANT:");
        HistoryReferenceGrounding no_refs;CorrectionLanguageGrounding no_language;
        const auto& active_refs=group=="no_reference_model"?no_refs:refs;const auto& active_language=group=="no_command_model"?no_language:language;
        auto meaning=active_language.meaning(*r,active_refs,records,r->request);
        CorrectionLanguageGrounding::Plan plan{-1,"",""};HistoryReferenceGrounding::Resolution resolution;
        if(meaning.plans.size()==1){plan=*meaning.plans.begin();resolution=active_refs.resolve(*r,records,plan.reference);}
        const auto episode=r->episode_revision,revision=r->history_revision;GroundedCorrectionDialogue dialogue;
        bool applied=dialogue.begin(*r,j,active_refs,active_language,records,r->request,request[0],group=="generated");
        std::string initial=group=="generated"?"":generate(org),final;bool invalid_preserved=false;
        // No reference span, operation, payload boundary or definite target ID
        // was supplied above. Read a later user clarification only after output.
        auto choice=std::stoll(line(test));auto stale_text=line(test);
        if(choice>=0) {
            assert(dialogue.correction.pending);auto token=dialogue.correction.pending->token;
            auto state=RevisableGrounding::state_stamp(*r);auto count=j.changes.size();
            assert(!dialogue.choose(*r,j,active_refs,active_language,records,token,99999));
            invalid_preserved=dialogue.correction.pending.has_value()&&j.changes.size()==count&&RevisableGrounding::state_stamp(*r)==state;assert(invalid_preserved);
            if(group=="stale"){stream(org,"\nENVIRONMENT: "+stale_text+"\n");assert(j.append(*r,99999,"late_sensor"));records=HistoryReferenceGrounding::records(j,*r,seed%2,1-seed%2);}
            stream(org,"\nUSER: "+std::to_string(choice)+"\nASSISTANT:");applied=dialogue.choose(*r,j,active_refs,active_language,records,token,choice);final=generate(org);
        }
        applied_count+=applied;auto selected=applied?std::to_string(j.changes.back().report):"-1";
        cases<<i<<'\t'<<group<<'\t'<<meaning.status<<'\t'<<plan.operation<<'\t'<<plan.reference<<'\t'<<plan.report<<'\t'<<resolution.status<<'\t';
        for(auto id:resolution.candidates)cases<<id<<',';
        cases<<'\t'<<applied<<'\t'<<selected<<'\t'<<dialogue.status<<'\t'<<initial<<'\t'<<final<<'\t'<<invalid_preserved<<'\t'<<episode<<'\t'<<r->episode_revision<<'\t'<<revision<<'\t'<<r->history_revision<<'\n';
        capture("after");Environment env(h[1],seed);
        for(const auto& question:questions(env)) {
            auto copy=std::make_shared<LanguageReasoner>(*r);qa.attach_reasoner(copy);stream(qa,"\nUSER: "+question+"\nASSISTANT:");
            answers<<i<<'\t'<<question<<'\t'<<generate(qa)<<'\t'<<copy->status<<'\n';
        }
        assert(language.signature()==frozen&&refs.signature()==frozen_refs&&r->events.signature()==event_signature&&r->grounding.signature()==grammar&&r->observation_language.signature()==observation_signature&&r->memory.procedures==programs&&r->memory.skills.kernel.routes.W==weights);
    }
    std::cerr<<"seed="<<seed<<" scenes="<<training<<" frames="<<language.frames.size()<<" requests="<<total<<" applied="<<applied_count<<'\n';
}
