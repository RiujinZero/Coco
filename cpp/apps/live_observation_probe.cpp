#include "interaction_probe_support.h"
#include "laminar/live_observation_grounding.h"
std::string report(const Environment& env,const Action& action,bool executed,const std::string& familiar,int variant) {
    if(!executed)return familiar;
    if(action.schema==env.transfer()) {
        const auto& a=action.arguments[0];const auto& o=action.arguments[1];const auto& b=action.arguments[2];
        return variant? a+" relinquished the "+o+" into "+b+"'s care.":"The "+o+" passed from "+a+" into "+b+"'s keeping.";
    }
    const auto& a=action.arguments[0];const auto& b=action.arguments[1];
    return variant?"The destination of "+a+" became "+b+".":a+" relocated into "+b+".";
}
std::vector<std::string> question_menu(const Environment& env,const LanguageReasoner& reader) {
    std::vector<std::string> out;
    for(const auto& name:reader.names) {
        out.push_back("What is the number of items held by "+name+"?");
        out.push_back("What is the number of destinations for "+name+"?");
    }
    // The environment offers well-defined, previously grounded questions. It
    // never chooses which question would distinguish the current hypotheses.
    for(const auto& person:env.people)out.push_back("Where is "+person+"?");return out;
}
std::string verified_answer(const Environment& env,const std::string& text) {
    const auto words=EventGrounding::words(text);const auto& name=words.back();
    if(words.front()=="where")return env.place.at(name);
    if(text.find("items held")!=std::string::npos) {
        size_t count=0;for(const auto& [o,p]:env.owner)count+=p==name;return std::to_string(count);
    }
    assert(text.find("destinations")!=std::string::npos);return std::to_string(env.place.count(name));
}
std::string rules(const std::set<EventGrounding::Rule>& candidates) {
    std::string out;for(const auto& r:candidates)out+=std::to_string(r.relation)+","+std::to_string(r.subject)+","+std::to_string(r.object)+","+std::to_string(r.effect)+";";return out;
}
int main(int argc,char** argv) {
    assert(argc==2);int seed=std::stoi(argv[1]);auto reader=prior(seed);
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p),probe(p);org.attach_reasoner(reader);Environment env("evolving",seed);std::mt19937 rng(seed+29000);
    OnlineGrounding control(seed+6100);LiveObservationGrounding observation;
    const auto grammar=reader->grounding.signature();const auto forms=reader->forms;const auto programs=reader->memory.procedures;const auto weights=reader->memory.skills.kernel.routes.W;
    const auto prior_events=reader->events;
    auto root="logs/live_observation_grounding/"+std::to_string(seed);
    std::ofstream trace(root+".trace.tsv"),answers(root+".answers.tsv"),requests(root+".requests.tsv"),qa(root+".retention.tsv"),models(root+".models.tsv"),holdout(root+".holdout.tsv"),predictions(root+".predictions.tsv");
    trace<<"request_id\tattempt\tevent_id\trequest\tpolicy\tindex\taction\texecuted\treport\tstaged\tinitial_status\tquestions\tframe\tbefore\tafter\tmenu\tsatisfied\tprediction\taccepted\tagree\n";
    answers<<"event_id\tanswer_id\tquestion\tanswer\tchoice\tscore\tpartitions\tbefore_rules\tafter_rules\tbefore_worlds\tafter_worlds\tmenu\n";
    predictions<<"event_id\tanswer_id\tquestion_index\trule\ttype\tvalue\n";
    requests<<"request_id\tfamily\trequest\tattempts\tcomplete\tinitial_prediction\n";
    qa<<"request_id\tquestion\treply\tfacts\n";
    holdout<<"mode\tversion\tcase\trequest\tindex\taction\treport\tcompleted\ttruth\tstatus\tbefore\tafter\tmenu\tagree\n";
    stream(org,"\n<|endofdialogue|>\n"+env.initial());uint64_t scene=1;int transitions=0,completed=0,question_total=0;
    for(int id=0;id<160;++id) {
        int stage=id/40,family=(id%40)%(stage+1);if(stage==3)env.version=1;
        auto goal=env.request(family,id);assert(!env.satisfied(goal));stream(org,"USER: "+goal.text+"\nASSISTANT:");
        int initial=reader->requests.evaluate(goal.text,reader->request_snapshot());
        assert(control.feedback_request(*reader,reader->request_revision,false,scene++));int attempt=0;
        while(attempt<32&&!env.satisfied(goal)) {
            auto menu=env.menu(rng);auto choice=control.choose(*reader,menu);
            if(choice.index<0){std::cerr<<"choice "<<id<<" "<<choice.status<<'\n';return 1;}
            const auto action=menu.at(choice.index);auto before=facts(*reader);int event_id=transitions++;
            auto [executed,familiar]=env.execute(action,event_id%7==0);
            auto text=report(env,action,executed,familiar,event_id%2);
            bool staged=observation.begin(*reader,text,10000+event_id);
            if(!staged)assert(observation.status=="prior_observation");
            stream(org,"\nENVIRONMENT: "+text+"\n");
            std::string frame,initial_status="prior_observation";int questions=0;
            if(staged) {
                frame=observation.pending->binding.key;
                bool committed=observation.commit(*reader);initial_status=observation.status;
                while(!committed&&questions<16) {
                    const auto candidates=observation.candidates();const auto worlds=observation.worlds();
                    auto options=question_menu(env,*reader);auto selected=observation.propose(options);
                    if(selected.index<0){std::cerr<<"question "<<event_id<<" "<<selected.status<<'\n';return 2;}
                    auto question=options.at(selected.index),answer=verified_answer(env,question);
                    // Audit the actual bounded learned-policy predictions. A
                    // hypothetical world can fall outside its demonstrated
                    // domain; absence must remain visible, not be replaced by
                    // an ideal interpreter's answer in the reported score.
                    for(size_t qi=0;qi<options.size();++qi) {
                        auto parsed=observation.question(options[qi]);assert(parsed);
                        for(const auto& rule:candidates) {
                            auto value=observation.evaluate(*parsed,rule);
                            std::string rendered=value.type==0?reader->names.at(value.id):value.present()?std::to_string(value.id):"";
                            predictions<<event_id<<'\t'<<questions<<'\t'<<qi<<'\t'<<rules({rule})<<'\t'<<value.type<<'\t'<<rendered<<'\n';
                        }
                    }
                    const auto evidence=observation.learner.signature();
                    assert(!observation.feedback(*reader,question,answer,questions,true));assert(observation.learner.signature()==evidence);
                    assert(observation.feedback(*reader,question,answer,questions));
                    std::string offered;for(const auto& q:options)offered+=q+"|";
                    answers<<event_id<<'\t'<<questions<<'\t'<<question<<'\t'<<answer<<'\t'<<selected.index<<'\t'<<selected.score<<'\t'<<selected.partitions<<'\t'<<rules(candidates)<<'\t'<<rules(observation.candidates())<<'\t'<<worlds<<'\t'<<observation.worlds()<<'\t'<<offered<<'\n';
                    ++questions;committed=observation.commit(*reader);
                }
                if(!committed){std::cerr<<"unresolved "<<event_id<<" "<<observation.status<<'\n';return 3;}
                const auto& proof=*observation.receipt;
                assert(control.rebase_observation_model(*reader,choice.token,proof.previous_model,proof.before));
            }
            int prediction=reader->requests.evaluate(goal.text,reader->request_snapshot());
            bool accepted=control.feedback_transition(*reader,choice.token,executed,env.satisfied(goal),scene++);
            trace<<id<<'\t'<<attempt<<'\t'<<event_id<<'\t'<<goal.text<<'\t'<<choice.status<<'\t'<<choice.index<<'\t'<<action_text(action)<<'\t'<<executed<<'\t'<<text<<'\t'<<staged<<'\t'<<initial_status<<'\t'<<questions<<'\t'<<frame<<'\t'<<before<<'\t'<<facts(*reader)<<'\t'<<menu_text(menu)<<'\t'<<env.satisfied(goal)<<'\t'<<prediction<<'\t'<<accepted<<'\t'<<agree(*reader)<<'\n';
            if(!accepted){trace.flush();std::cerr<<"feedback "<<event_id<<" "<<control.status<<'\n';return 4;}
            question_total+=questions;++attempt;
        }
        completed+=env.satisfied(goal);requests<<id<<'\t'<<family<<'\t'<<goal.text<<'\t'<<attempt<<'\t'<<env.satisfied(goal)<<'\t'<<initial<<'\n';
        for(const auto& question:std::vector<std::string>{"Who has the "+env.objects[id%3]+" now?","Where is "+env.people[id%4]+"?"}) {
            auto copy=std::make_shared<LanguageReasoner>(*reader);probe.attach_reasoner(copy);stream(probe,"\nUSER: "+question+"\nASSISTANT:");
            qa<<id<<'\t'<<question<<'\t'<<generate(probe)<<'\t'<<facts(*copy)<<'\n';
        }
        if(id%40==39) {
            models<<"checkpoint\t"<<stage<<'\n';
            for(const auto& [key,f]:observation.learner.frames) {const auto& learned=reader->events.frames.at(key);models<<key<<':'<<learned.scenes.size()<<':'<<rules(learned.candidates)<<'\n';}
        }
    }
    const auto events=reader->events.signature(),requests_model=reader->requests.signature(),effects=reader->actuators.signature(),evidence=observation.learner.signature();
    for(const std::string mode:{"grounded","no_new_reports"})for(int version=0;version<2;++version)for(int c=0;c<48;++c) {
        auto copy=std::make_shared<LanguageReasoner>(*reader);copy->reset_episode();
        if(mode=="no_new_reports")for(const auto& [key,f]:observation.learner.frames)copy->events.frames.erase(key);
        probe.attach_reasoner(copy);Environment test("newreport"+std::to_string(version)+"case"+std::to_string(c),seed);test.version=version;std::mt19937 eval_rng(seed+33000+version*100+c);
        stream(probe,"\n<|endofdialogue|>\n"+test.initial());auto goal=test.request(c%4,c);
        stream(probe,"USER: "+goal.text+"\nASSISTANT:");auto offered=test.menu(eval_rng);auto selected=copy->choose_action(offered);assert(selected.index>=0);
        auto before=facts(*copy);const auto action=offered.at(selected.index);auto result=test.execute(action,false);
        auto text=report(test,action,result.first,result.second,(c/4)%2);stream(probe,"\nENVIRONMENT: "+text+"\n");
        holdout<<mode<<'\t'<<version<<'\t'<<c<<'\t'<<goal.text<<'\t'<<selected.index<<'\t'<<action_text(action)<<'\t'<<text<<'\t'<<test.satisfied(goal)<<'\t'<<copy->requests.evaluate(goal.text,copy->request_snapshot())<<'\t'<<(copy->complete?"complete":copy->event_issue)<<'\t'<<before<<'\t'<<facts(*copy)<<'\t'<<menu_text(offered)<<'\t'<<agree(*copy)<<'\n';
    }
    assert(reader->events.signature()==events&&reader->requests.signature()==requests_model&&reader->actuators.signature()==effects&&observation.learner.signature()==evidence);
    assert(reader->grounding.signature()==grammar&&reader->forms==forms&&reader->memory.procedures==programs&&reader->memory.skills.kernel.routes.W==weights);
    for(const auto& [key,f]:prior_events.frames)assert(reader->events.frames.at(key).candidates==f.candidates);
    std::cerr<<"seed="<<seed<<" completed="<<completed<<"/160 transitions="<<transitions<<" observation_answers="<<question_total<<'\n';
}
