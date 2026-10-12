#include "interaction_probe_support.h"
#include "laminar/observation_language_learner.h"
#include "laminar/revisable_grounding.h"

std::string knowledge(const LanguageReasoner& r,const FactKnowledge& k) {
    std::string out;
    for(auto [a,b,c]:k.positive)out+="p,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b,c]:k.negative)out+="n,"+r.names[a]+","+std::to_string(b)+","+r.names[c]+";";
    for(auto [a,b]:k.closed_rows)out+="r,"+r.names[a]+","+std::to_string(b)+";";
    for(auto [a,b]:k.closed_columns)out+="c,"+std::to_string(a)+","+r.names[b]+";";
    return out;
}
std::string bindings(const LanguageReasoner::PossibleWorld& w) {
    std::string out;for(const auto& [key,choice]:w.meanings)out+=key+"#"+choice+";";return out;
}
std::vector<std::string> queries(const LanguageReasoner& r) {
    std::vector<std::string> out;
    for(const auto& a:r.names)for(const auto& b:r.names) {
        out.push_back("Does "+b+" have the "+a+"?");out.push_back("Is "+a+" at "+b+"?");
    }
    for(const auto& a:r.names){out.push_back("Who has the "+a+" now?");out.push_back("Where is "+a+"?");}
    return out;
}
int main(int argc,char** argv) {
    assert(argc==2);int seed=std::stoi(argv[1]);auto reader=prior(seed);
    ObservationLanguageLearner bootstrap;assert(bootstrap.fit(lessons("logs/observation_language/train.txt"),*reader)&&bootstrap.publish(*reader));
    reader->world_budget=4096;
    const auto events=reader->events.signature(),grammar=reader->grounding.signature();
    const auto forms=reader->forms;const auto programs=reader->memory.procedures;
    const auto weights=reader->memory.skills.kernel.routes.W;
    const auto prior_model=reader->observation_language;
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p),qa(p);org.attach_reasoner(reader);Environment env("revision",seed);
    stream(org,"\n<|endofdialogue|>\n"+env.initial());RevisableGrounding journal;assert(journal.start(*reader));
    const auto episode=reader->episode_revision;
    const std::string root="logs/revision_grounding/"+std::to_string(seed);
    std::ofstream steps(root+".steps.tsv"),models(root+".models.tsv"),worlds(root+".worlds.tsv"),answers(root+".answers.tsv"),changes(root+".changes.tsv"),held(root+".frozen.tsv");
    steps<<"step\top\tconsistent\tstatus\thistory_revision\tepisode_revision\treports\tworlds\tevaluations\n";
    models<<"step\tframe\tsupport\trules\n";
    worlds<<"step\tscope\tknowledge\tbindings\n";
    answers<<"step\tscope\tquestion\treply\tstatus\n";
    changes<<"kind\treport\tevidence\trevision\tsource\tpayload\n";
    held<<"case\tphase\tscope\tquestion\treply\tstatus\tknowledge\n";
    auto ask=[&](const LanguageReasoner& r,const std::string& question) {
        auto copy=std::make_shared<LanguageReasoner>(r);qa.attach_reasoner(copy);
        stream(qa,"\nUSER: "+question+"\nASSISTANT:");auto text=generate(qa);
        return std::make_pair(text,copy->status);
    };
    std::ifstream input("logs/revision_grounding/train.tsv");assert(input);line(input);int step=0,conflicts=0;
    while(input.peek()!=EOF) {
        auto f=fields(line(input));assert(f.size()==9);const auto& op=f[0];uint64_t report=std::stoull(f[1]),evidence=std::stoull(f[2]),revision=std::stoull(f[3]);bool active=std::stoi(f[8]);
        bool accepted=false;
        if(op=="report") {
            stream(org,"\nENVIRONMENT: "+f[5]+"\n");accepted=journal.append(*reader,report,f[4]);
        } else if(op=="feedback")accepted=journal.feedback(*reader,report,evidence,f[6],f[7],f[4]);
        else if(op=="revise_feedback")accepted=journal.revise_feedback(*reader,report,evidence,revision,f[7],f[4],active);
        else if(op=="revise_report")accepted=journal.revise_report(*reader,report,revision,f[5],f[4],active);
        if(!accepted){std::cerr<<"rejected step="<<step<<" status="<<journal.status<<'\n';return 2;}
        assert(reader->episode_revision==episode);conflicts+=!journal.consistent;
        steps<<step<<'\t'<<op<<'\t'<<journal.consistent<<'\t'<<journal.status<<'\t'<<reader->history_revision<<'\t'<<reader->episode_revision<<'\t'<<journal.reports.size()<<'\t'<<reader->possible_worlds.size()<<'\t'<<journal.evaluations<<'\n';
        for(const auto& [key,frame]:reader->observation_language.frames)if(!prior_model.frames.count(key)) {
            models<<step<<'\t'<<key<<'\t'<<frame.scenes.size()<<'\t';for(const auto& rule:frame.candidates)models<<rule.signature()<<';';models<<'\n';
        }
        auto capture=[&](const LanguageReasoner& r,const std::string& scope,const std::vector<std::string>& menu) {
            if(journal.consistent)for(const auto& w:r.possible_worlds)worlds<<step<<'\t'<<scope<<'\t'<<knowledge(r,w.knowledge)<<'\t'<<bindings(w)<<'\n';
            for(const auto& q:menu) {auto [reply,status]=ask(r,q);answers<<step<<'\t'<<scope<<'\t'<<q<<'\t'<<reply<<'\t'<<status<<'\n';}
        };
        capture(*reader,"current",queries(*reader));
        for(const auto& entry:journal.reports)if(auto history=journal.at(entry.id))capture(*history,std::to_string(entry.id),{"Who has the revisionitem0 now?","Where is revisionperson2?"});
        ++step;
    }
    assert(journal.consistent);
    for(const auto& c:journal.changes)changes<<c.kind<<'\t'<<c.report<<'\t'<<c.evidence<<'\t'<<c.revision<<'\t'<<c.source<<'\t'<<c.payload<<'\n';
    // Freeze meanings; new worlds receive corrections to identified raw reports,
    // with no question targets or report-language relearning.
    const auto learned=reader->observation_language.signature();
    for(int c=0;c<48;++c) {
        auto r=std::make_shared<LanguageReasoner>(*reader);org.attach_reasoner(r);Environment test("revheld"+std::to_string(c),seed);
        stream(org,"\n<|endofdialogue|>\n"+test.initial());RevisableGrounding replay;assert(replay.start(*r));
        auto object=test.objects[c%3],actor=test.people[c%4];
        auto recipient=[&](int delta){return test.people[(c+delta)%4];};
        auto text=[&](int delta){return "The "+object+" was logged beside "+recipient(delta)+".";};
        const auto gap="The location record for "+actor+" went dark.";
        for(int phase=0;phase<8;++phase) {
            if(phase==0||phase==1||phase==3) {
                auto report=phase==0?text(1):phase==1?gap:recipient(3)+" has the "+object+".";
                stream(org,"\nENVIRONMENT: "+report+"\n");assert(replay.append(*r,phase,"sensor"));
            } else if(phase==2)assert(replay.revise_report(*r,0,0,text(2),"sensor"));
            else if(phase==4)assert(replay.revise_report(*r,0,1,text(0),"sensor"));
            else if(phase==5)assert(replay.revise_report(*r,1,0,"","sensor",false));
            else if(phase==6)assert(replay.revise_report(*r,3,0,"","sensor",false));
            else assert(replay.revise_report(*r,1,1,gap,"sensor"));
            assert(replay.consistent&&r->observation_language.signature()==learned);
            for(const auto& scope:{std::string("current"),std::string("0")}) {
                auto historical=replay.at(0);const auto& state=scope=="current"?*r:*historical;
                for(const auto& q:{"Who has the "+object+" now?","Where is "+actor+"?"}) {
                    auto [reply,status]=ask(state,q);held<<c<<'\t'<<phase<<'\t'<<scope<<'\t'<<q<<'\t'<<reply<<'\t'<<status<<'\t'<<knowledge(state,state.observations)<<'\n';
                }
            }
        }
    }
    assert(reader->events.signature()==events&&reader->grounding.signature()==grammar&&reader->forms==forms&&reader->memory.procedures==programs&&reader->memory.skills.kernel.routes.W==weights);
    std::cerr<<"seed="<<seed<<" operations="<<step<<" withheld_conflict_states="<<conflicts<<" final_frames="<<reader->observation_language.frames.size()-prior_model.frames.size()<<" frozen_questions=1536\n";
}
