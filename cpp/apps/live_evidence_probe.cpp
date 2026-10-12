#include "interaction_probe_support.h"
#include "laminar/observation_language_learner.h"
#include "laminar/live_evidence_grounding.h"

std::vector<std::string> split(const std::string& text,char delimiter) {
    std::istringstream input(text);std::vector<std::string> out;std::string word;
    while(std::getline(input,word,delimiter))out.push_back(word);return out;
}
void restore_frozen_actions(LanguageReasoner& reader,int seed) {
    // Reload the final models learned in the earlier online experiment. The
    // export retains support counts, not evidence IDs; these memories stay frozen.
    std::ifstream in("logs/online_grounding/"+std::to_string(seed)+".information.models.tsv");assert(in);
    int checkpoint=-1;
    while(in.peek()!=EOF) {
        auto text=line(in);
        if(text.starts_with("checkpoint\t")){checkpoint=std::stoi(text.substr(11));continue;}
        if(checkpoint!=3)continue;auto f=split(text,':');
        if(f.size()==3) {
            auto& frame=reader.requests.frames[f[0]];frame.pattern=SemanticGrounding::words(f[0]);frame.initialized=true;
            for(int i=0;i<std::stoi(f[1]);++i)frame.scenes.insert(i);
            for(const auto& rule:split(f[2],';')){auto v=split(rule,',');assert(v.size()==4);frame.candidates.insert({std::stoi(v[0]),std::stoi(v[1]),std::stoi(v[2]),bool(std::stoi(v[3]))});}
        } else {
            assert(f.size()==5);auto& frame=reader.actuators.frames[f[0]];frame.arity=std::stoi(f[1]);frame.failures=std::stoi(f[3]);
            for(int i=0;i<std::stoi(f[2]);++i)frame.scenes.insert(i);
            for(const auto& rule:split(f[4],';')){auto v=split(rule,',');assert(v.size()==4);frame.candidates.insert({std::stoi(v[0]),std::stoi(v[1]),std::stoi(v[2]),std::stoi(v[3])});}
        }
    }
    assert(reader.requests.frames.size()==4&&reader.actuators.frames.size()==4);
}
std::string encode(const LanguageReasoner& r,const FactKnowledge& k) {
    std::string out;
    for(auto [a,rel,b]:k.positive)out+="p,"+r.names[a]+","+std::to_string(rel)+","+r.names[b]+";";
    for(auto [a,rel,b]:k.negative)out+="n,"+r.names[a]+","+std::to_string(rel)+","+r.names[b]+";";
    for(auto [a,rel]:k.closed_rows)out+="r,"+r.names[a]+","+std::to_string(rel)+";";
    for(auto [rel,b]:k.closed_columns)out+="c,"+std::to_string(rel)+","+r.names[b]+";";
    return out;
}
std::string rule_set(const std::set<ObservationLanguage::Rule>& rules) {std::string out;for(const auto& r:rules)out+=r.signature()+";";return out;}
std::vector<std::string> questions(const LanguageReasoner& r) {
    std::vector<std::string> out;
    for(const auto& a:r.names)for(const auto& b:r.names) {
        out.push_back("Does "+b+" have the "+a+"?");out.push_back("Is "+a+" at "+b+"?");
    }
    for(const auto& a:r.names){out.push_back("Who has the "+a+" now?");out.push_back("Where is "+a+"?");}return out;
}
// The environment's record of what it has made observable is independent of
// the learner. Hidden physical changes are not answers to these questions.
struct PublicEvidence {
    std::map<std::pair<std::string,int>,std::string> rows;
    void set(const std::string& a,int r,const std::string& b){rows[{a,r}]=b;}
    std::string answer(const std::string& question) const {
        const auto w=EventGrounding::words(question);bool membership=w[0]=="does"||w[0]=="is";
        int rel=w[0]=="is"||w[0]=="where"?0:1;
        auto subject=w[0]=="does"?w.back():w[0]=="who"?w[3]:w.back();
        if(w[0]=="is")subject=w[1];
        auto it=rows.find({subject,rel});if(it==rows.end())return "unknown";
        if(!membership)return it->second;
        auto object=w[0]=="does"?w[1]:w.back();return it->second==object?"yes":"no";
    }
};
int main(int argc,char** argv) {
    assert(argc==2);int seed=std::stoi(argv[1]);auto reader=prior(seed);
    ObservationLanguageLearner prior_learning;assert(prior_learning.fit(lessons("logs/observation_language/train.txt"),*reader)&&prior_learning.publish(*reader));
    restore_frozen_actions(*reader,seed);reader->world_budget=512;
    const auto initial_events=reader->events.signature(),requests=reader->requests.signature(),actuators=reader->actuators.signature(),grammar=reader->grounding.signature();
    const auto programs=reader->memory.procedures;const auto weights=reader->memory.skills.kernel.routes.W;
    const auto initial_observations=reader->observation_language;
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p),query_org(p);org.attach_reasoner(reader);Environment env("livepartial",seed);PublicEvidence public_state;
    for(const auto& [a,b]:env.place)public_state.set(a,0,b);for(const auto& [o,a]:env.owner)public_state.set(o,1,a);
    stream(org,"\n<|endofdialogue|>\n"+env.initial());auto names=reader->names;
    OnlineGrounding control(seed+913);control.learning=false;LiveEvidenceGrounding learner;std::mt19937 rng(seed+7382);
    std::string root="logs/live_evidence/"+std::to_string(seed);
    std::ofstream reports(root+".reports.tsv"),answers(root+".answers.tsv"),retention(root+".retention.tsv"),actions(root+".actions.tsv"),models(root+".models.txt"),holdout(root+".holdout.tsv");
    reports<<"event\trequest_id\ttype\treport\tstaged\tquestions\tbefore\tafter\tworlds\tframe\trules\tsupport\n";
    answers<<"event\tanswer_id\tquestion\tanswer\tchoice\tscore\tpartitions\tbefore_rules\tafter_rules\tbefore_branches\tafter_branches\n";
    retention<<"event\tquestion\treply\tstatus\n";
    actions<<"request_id\tattempt\trequest\tbefore_truth\tindex\taction\tmenu\texecuted\treport_event\tsatisfied\tafter_truth\tbefore\tafter\n";
    holdout<<"case\tphase\tquestion\treply\tstatus\tknowledge\n";
    int event=0,attempt_total=0,total_answers=0,completed=0,unknown_before=0;
    auto receive=[&](const std::string& text,int request_id,const std::string& kind) {
        int id=event++;auto before=encode(*reader,reader->observations);
        bool staged=learner.begin(*reader,text,10000+id);if(!staged)assert(learner.status=="prior_observation");
        stream(org,"\nENVIRONMENT: "+text+"\n");int n=0;std::string frame,rules;size_t support=0;
        if(staged) {
            frame=learner.pending->key;auto options=questions(*reader);
            auto old=learner.pending->before.observation_language.frames.find(frame);
            bool verified=old!=learner.pending->before.observation_language.frames.end()&&old->second.scenes.size()>=4;
            while((!verified||learner.knowledge_states()>1)&&n<32) {
                const auto candidates=rule_set(learner.candidates());size_t branches=learner.pending->branches.size();
                auto selected=learner.propose(options,1<<20);assert(selected.index>=0);
                auto q=options.at(selected.index),gold=public_state.answer(q);
                assert(!learner.feedback(*reader,q,gold,n,true));assert(learner.pending->branches.size()==branches);
                assert(learner.feedback(*reader,q,gold,n));
                answers<<id<<'\t'<<n<<'\t'<<q<<'\t'<<gold<<'\t'<<selected.index<<'\t'<<selected.score<<'\t'<<selected.partitions<<'\t'<<candidates<<'\t'<<rule_set(learner.candidates())<<'\t'<<branches<<'\t'<<learner.pending->branches.size()<<'\n';
                if(learner.pending->branches.empty()){answers.flush();std::cerr<<"conflict "<<id<<" "<<text<<" / "<<q<<" / "<<gold<<'\n';std::exit(2);}
                ++n;verified=true;
                if(!selected.score&&learner.knowledge_states()>1)break;
            }
            if(!learner.commit(*reader)){std::cerr<<"commit "<<id<<" "<<learner.status<<'\n';std::exit(3);}
            rules=rule_set(reader->observation_language.frames.at(frame).candidates);support=reader->observation_language.frames.at(frame).scenes.size();
        }
        reports<<id<<'\t'<<request_id<<'\t'<<kind<<'\t'<<text<<'\t'<<staged<<'\t'<<n<<'\t'<<before<<'\t'<<encode(*reader,reader->observations)<<'\t'<<reader->possible_worlds.size()<<'\t'<<frame<<'\t'<<rules<<'\t'<<support<<'\n';
        // Full current-knowledge answers, never fed back as teaching evidence.
        for(const auto& q:questions(*reader)) {
            auto copy=std::make_shared<LanguageReasoner>(*reader);query_org.attach_reasoner(copy);stream(query_org,"\nUSER: "+q+"\nASSISTANT:");
            auto reply=generate(query_org);retention<<id<<'\t'<<q<<'\t'<<reply<<'\t'<<copy->status<<'\n';
            const auto expected=public_state.answer(q);
            if(reply!=(expected=="unknown"?"I do not know.":expected)) {
                reports.flush();answers.flush();retention.flush();
                std::cerr<<"retention "<<id<<" report="<<text<<" question="<<q<<" expected="<<expected<<" reply="<<reply<<" status="<<copy->status<<" worlds="<<reader->possible_worlds.size()<<'\n';std::exit(4);
            }
        }
        total_answers+=n;return std::make_pair(id,staged);
    };
    for(int id=0;id<160;++id) {
        env.version=id>=80;
        // The hidden holder/location really changes. Only a loss-of-visibility
        // report reaches the learner; the physical value stays in the evaluator.
        if(id%2==0) {
            auto object=env.objects[(id/2)%3],old=env.owner[object];auto target=env.people[(id/6+seed)%4];
            if(target==old)target=env.people[(id/6+seed+1)%4];env.owner[object]=target;public_state.rows.erase({object,1});
            receive("The holder of "+object+" went unobserved.",id,"hidden_owner");
        } else {
            auto actor=env.people[(id/2)%4],old=env.place[actor];auto room=env.rooms[(id/8+seed)%4];
            if(room==old)room=env.rooms[(id/8+seed+1)%4];env.place[actor]=room;public_state.rows.erase({actor,0});
            receive("The whereabouts of "+actor+" were unobserved.",id,"hidden_location");
        }
        auto goal=env.request(id%4,id);assert(!env.satisfied(goal));stream(org,"\nUSER: "+goal.text+"\nASSISTANT:");int attempt=0;
        while(!env.satisfied(goal)&&attempt<3) {
            int truth=reader->requests.evaluate(goal.text,reader->request_snapshot());assert(truth!=RequestGrounding::TRUE_VALUE);unknown_before+=truth==RequestGrounding::UNKNOWN;
            auto before=encode(*reader,reader->observations);auto offered=env.menu(rng);auto choice=control.choose(*reader,offered);assert(choice.index>=0);
            const auto action=offered.at(choice.index);auto [executed,familiar]=env.execute(action,attempt_total++%7==0);
            std::string text;
            if(action.schema==env.transfer()) {
                const auto& a=action.arguments;public_state.set(a[1],1,env.owner.at(a[1]));
                text=executed?"The "+a[1]+" passed from "+a[0]+" into "+a[2]+"'s keeping.":familiar;
            } else {
                const auto& a=action.arguments;public_state.set(a[0],0,env.place.at(a[0]));
                text=executed?a[0]+" relocated into "+a[1]+".":familiar;
            }
            auto [report_id,staged]=receive(text,id,"action");
            if(staged)assert(control.rebase_evidence_model(*reader,choice.token,learner.receipt->previous_model,learner.receipt->before));
            assert(control.feedback_transition(*reader,choice.token,executed,env.satisfied(goal),20000+report_id));
            int after=reader->requests.evaluate(goal.text,reader->request_snapshot());
            assert(after==int(env.satisfied(goal)));
            actions<<id<<'\t'<<attempt++<<'\t'<<goal.text<<'\t'<<truth<<'\t'<<choice.index<<'\t'<<action_text(action)<<'\t'<<menu_text(offered)<<'\t'<<executed<<'\t'<<report_id<<'\t'<<env.satisfied(goal)<<'\t'<<after<<'\t'<<before<<'\t'<<encode(*reader,reader->observations)<<'\n';
        }
        completed+=env.satisfied(goal);
    }
    for(const auto& key:learner.owned){const auto& f=reader->observation_language.frames.at(key);models<<key<<':'<<f.scenes.size()<<':'<<rule_set(f.candidates)<<'\n';}
    // Frozen fresh-name histories test learned visibility and event reports
    // without the live learner, external questions, or goal satisfaction bits.
    for(int c=0;c<48;++c) {
        auto copy=std::make_shared<LanguageReasoner>(*reader);query_org.attach_reasoner(copy);Environment test("unseen"+std::to_string(c),seed);
        stream(query_org,"\n<|endofdialogue|>\n"+test.initial());auto object=test.objects[c%3],actor=test.people[c%4],room=test.rooms[(c+1)%4],receiver=test.people[(c+2)%4];
        const auto owner=test.owner[object];
        std::vector<std::pair<std::string,std::string>> stages{
            {"The holder of "+object+" went unobserved.","Who has the "+object+" now?"},
            {"The whereabouts of "+actor+" were unobserved.","Where is "+actor+"?"},
            {"The "+object+" passed from "+owner+" into "+receiver+"'s keeping.","Who has the "+object+" now?"},
            {actor+" relocated into "+room+".","Where is "+actor+"?"}};
        for(size_t phase=0;phase<stages.size();++phase) {
            const auto& [report,q]=stages[phase];stream(query_org,"\nENVIRONMENT: "+report+"\nUSER: "+q+"\nASSISTANT:");
            holdout<<c<<'\t'<<phase<<'\t'<<q<<'\t'<<generate(query_org)<<'\t'<<copy->status<<'\t'<<encode(*copy,copy->observations)<<'\n';
        }
    }
    assert(reader->events.signature()==initial_events&&reader->requests.signature()==requests&&reader->actuators.signature()==actuators&&reader->grounding.signature()==grammar&&reader->memory.procedures==programs&&reader->memory.skills.kernel.routes.W==weights);
    for(const auto& [key,frame]:initial_observations.frames)assert(reader->observation_language.frames.at(key).candidates==frame.candidates);
    std::cerr<<"seed="<<seed<<" completed="<<completed<<" actions="<<attempt_total<<" unknown_goal_before="<<unknown_before<<" reports="<<event<<" verified_answers="<<total_answers<<'\n';
}
