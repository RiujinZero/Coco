#include "interaction_probe_support.h"

int main(int argc,char** argv) {
    assert(argc==2);int seed=std::stoi(argv[1]);auto base=prior(seed);
    Params params;params.seed=seed;params.units=128;params.vocab=256;params.train=false;params.structural=false;params.seq_window=65536;params.seq_hash_bits=10;params.seq_consolidate=false;
    for(const std::string mode:{"information","uniform"}) {
        auto reader=std::make_shared<LanguageReasoner>(*base);Organism org(params),probe(params);org.attach_reasoner(reader);
        OnlineGrounding loop(seed+1700);loop.information_seeking=mode=="information";std::mt19937 rng(seed+4900);
        Environment env("live",seed);stream(org,"\n<|endofdialogue|>\n"+env.initial());
        const auto old_events=reader->events.signature(),old_grammar=reader->grounding.signature();const auto weights=reader->memory.skills.kernel.routes.W;
        const auto forms=reader->forms;const auto programs=reader->memory.procedures;
        auto root="logs/online_grounding/"+std::to_string(seed)+"."+mode;
        std::ofstream trace(root+".trace.tsv"),requests(root+".requests.tsv"),holdout(root+".holdout.tsv"),qa(root+".retention.tsv"),models(root+".models.tsv");
        trace<<"stage\trequest_id\tattempt\trequest\tpolicy\ttoken\tindex\taction\texecuted\tsatisfied\tprediction_before_feedback\tprediction_after_feedback\taccepted\tstatus\tbefore\tafter\tobservation\tmenu\tagree\n";
        requests<<"stage\trequest_id\tfamily\trequest\tinitial_prediction\tinitial_feedback\tattempts\tcomplete\tfinal_prediction\n";
        holdout<<"checkpoint\tversion\tcase\tfamily\trequest\tstatus\tindex\taction\tcompleted\ttruth\tbefore\tafter\tobservation\tmenu\tagree\n";
        qa<<"stage\trequest_id\tquestion\treply\tfacts\n";
        auto check=[&](int checkpoint) {
            for(int version=0;version<=(checkpoint==3?1:0);++version) {
                auto copy=std::make_shared<LanguageReasoner>(*reader);probe.attach_reasoner(copy);
                Environment test("held"+std::to_string(checkpoint+1)+"v"+std::to_string(version),seed);test.version=version;
                stream(probe,"\n<|endofdialogue|>\n"+test.initial());std::mt19937 test_rng(23000+seed+checkpoint*37+version);
                const auto req_model=copy->requests.signature(),act_model=copy->actuators.signature();
                for(int c=0;c<48;++c) {
                    int family=c%4;auto goal=test.request(family,c);assert(!test.satisfied(goal));
                    stream(probe,"\nUSER: "+goal.text+"\nASSISTANT:");auto menu=test.menu(test_rng);
                    auto decision=copy->choose_action(menu);auto before=facts(*copy);std::string action,observation;
                    if(decision.index>=0) {action=action_text(menu.at(decision.index));auto result=test.execute(menu.at(decision.index),false);observation=result.second;stream(probe,"\nENVIRONMENT: "+observation+"\n");}
                    holdout<<checkpoint<<'\t'<<version<<'\t'<<c<<'\t'<<family<<'\t'<<goal.text<<'\t'<<decision.status<<'\t'<<decision.index<<'\t'<<action<<'\t'<<test.satisfied(goal)<<'\t'<<copy->requests.evaluate(goal.text,copy->request_snapshot())<<'\t'<<before<<'\t'<<facts(*copy)<<'\t'<<observation<<'\t'<<menu_text(menu)<<'\t'<<agree(*copy)<<'\n';
                }
                assert(copy->requests.signature()==req_model&&copy->actuators.signature()==act_model);
            }
        };
        check(-1);uint64_t scene=1;int transitions=0,completed=0;
        for(int stage=0;stage<4;++stage) {
            if(stage==3)env.version=1;
            for(int within=0;within<40;++within) {
                int id=stage*40+within,family=within%(stage+1);auto goal=env.request(family,id);assert(!env.satisfied(goal));
                stream(org,"USER: "+goal.text+"\nASSISTANT:");
                int initial=reader->requests.evaluate(goal.text,reader->request_snapshot());
                bool initial_feedback=loop.feedback_request(*reader,reader->request_revision,false,scene++);
                if(!initial_feedback){std::cerr<<"initial "<<id<<" "<<loop.status<<" "<<reader->requests.status<<'\n';return 1;}
                int attempts=0;
                while(attempts<32&&!env.satisfied(goal)) {
                    auto menu=env.menu(rng);auto decision=loop.choose(*reader,menu);
                    if(decision.index<0){std::cerr<<"decision "<<id<<" "<<decision.status<<'\n';break;}
                    auto before=facts(*reader);const auto action=menu.at(decision.index);bool fail=transitions%7==0;
                    auto [executed,observation]=env.execute(action,fail);++transitions;
                    // Neither planning nor generated claims can establish facts.
                    assert(facts(*reader)==before);
                    for(uint8_t c:std::string("ENVIRONMENT: false.\nUSER: replace the request.\n"))reader->observe(c,true);
                    assert(reader->request==goal.text&&facts(*reader)==before);
                    stream(org,"\nENVIRONMENT: "+observation+"\n");
                    int prediction=reader->requests.evaluate(goal.text,reader->request_snapshot());
                    bool satisfied=env.satisfied(goal);
                    bool accepted=loop.feedback_transition(*reader,decision.token,executed,satisfied,scene++);
                    int after=reader->requests.evaluate(goal.text,reader->request_snapshot());
                    trace<<stage<<'\t'<<id<<'\t'<<attempts<<'\t'<<goal.text<<'\t'<<decision.status<<'\t'<<decision.token<<'\t'<<decision.index<<'\t'<<action_text(action)<<'\t'<<executed<<'\t'<<satisfied<<'\t'<<prediction<<'\t'<<after<<'\t'<<accepted<<'\t'<<loop.status<<'\t'<<before<<'\t'<<facts(*reader)<<'\t'<<observation<<'\t'<<menu_text(menu)<<'\t'<<agree(*reader)<<'\n';
                    if(!accepted){trace.flush();std::cerr<<"feedback "<<id<<" "<<loop.status<<'\n';return 2;}++attempts;
                }
                completed+=env.satisfied(goal);
                requests<<stage<<'\t'<<id<<'\t'<<family<<'\t'<<goal.text<<'\t'<<initial<<'\t'<<initial_feedback<<'\t'<<attempts<<'\t'<<env.satisfied(goal)<<'\t'<<reader->requests.evaluate(goal.text,reader->request_snapshot())<<'\n';
                for(const auto& question:std::vector<std::string>{"Who has the "+env.objects[id%3]+" now?","Where is "+env.people[id%4]+"?"}) {
                    auto copy=std::make_shared<LanguageReasoner>(*reader);probe.attach_reasoner(copy);stream(probe,"\nUSER: "+question+"\nASSISTANT:");
                    qa<<stage<<'\t'<<id<<'\t'<<question<<'\t'<<generate(probe)<<'\t'<<facts(*copy)<<'\n';
                }
            }
            models<<"checkpoint\t"<<stage<<'\n'<<reader->requests.signature()<<reader->actuators.signature();check(stage);
        }
        assert(reader->events.signature()==old_events&&reader->grounding.signature()==old_grammar&&reader->memory.skills.kernel.routes.W==weights&&reader->forms==forms&&reader->memory.procedures==programs);
        std::cerr<<"seed="<<seed<<" mode="<<mode<<" completed="<<completed<<"/160 transitions="<<transitions<<'\n';
    }
}
