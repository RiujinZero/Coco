// Evaluator-only worlds and whole-query Boolean answers. No missing skill cue,
// macro program or primitive output is passed to composed acquisition.
#include "primitive_discovery_support.h"
#include "laminar/composed_skill_discovery.h"
#include <filesystem>
VerifiedQuestion composed(std::mt19937& rng,int family,int n,int other,int depth,int serial) {
    VerifiedQuestion e;auto& q=e.query;q.cue=71+family;
    std::vector<int> ids(5*n+5*depth+100);std::iota(ids.begin(),ids.end(),serial*1000);std::shuffle(ids.begin(),ids.end(),rng);
    q.a={0,ids[0]};q.b=family==1?Value{2,other}:Value{0,ids[1]};q.relation[0]=int(rng()%17);q.relation[1]=q.relation[0]+31;
    int fresh=2;auto add=[&](int s,int r,int o){q.facts.push_back({s,r,o});};
    int endpoint=q.a.id,right_endpoint=q.b.id;
    if(family!=0)for(int i=0;i<depth;++i){int next=ids[fresh++];add(endpoint,q.relation[0],next);endpoint=next;}
    if(family==3)for(int i=0;i<depth;++i){int next=ids[fresh++];add(right_endpoint,q.relation[0],next);right_endpoint=next;}
    int relation=q.relation[family==0?0:1];
    for(int i=0;i<n;++i){int x=ids[fresh++];for(int j=0,copies=1+int(rng()%3);j<copies;++j)add(endpoint,relation,x);}
    if(family==0||family==3)for(int i=0;i<other;++i){int x=ids[fresh++];for(int j=0,copies=1+int(rng()%3);j<copies;++j)add(right_endpoint,q.relation[1],x);}
    if(family!=0&&depth) { // root counts must not substitute for the endpoint count
        int root_count=int(rng()%6);
        for(int i=0;i<root_count;++i)add(q.a.id,q.relation[1],ids[fresh++]);
        if(family==3)for(int i=0,c=int(rng()%6);i<c;++i)add(q.b.id,q.relation[1],ids[fresh++]);
    }
    for(int i=0;i<3;++i){add(ids[fresh++],relation,ids[fresh++]);add(q.a.id,relation+41,ids[fresh++]);}
    std::shuffle(q.facts.begin(),q.facts.end(),rng);e.answer=family==2?Value{2,n}:Value{1,n==other};return e;
}
void question(std::ostream& out,const VerifiedQuestion& e,const std::string& phase,int id,int size) {
    const auto& q=e.query;out<<id<<' '<<phase<<' '<<size<<' '<<q.cue<<' '<<q.a.type<<' '<<q.a.id<<' '<<q.b.type<<' '<<q.b.id<<' '<<q.relation[0]<<' '<<q.relation[1]<<' '<<e.answer.type<<' '<<e.answer.id<<' '<<q.facts.size();
    for(const auto& f:q.facts)out<<' '<<f.subject<<' '<<f.relation<<' '<<f.object;out<<'\n';
}
template<class Choice> struct RecordingChoice {
    Choice& choices;std::ostream& out;int id;std::string model;size_t step=0;
    int choose(const Observation& obs,const std::vector<int>& available) {
        int a=choices.choose(obs,available);out<<id<<' '<<model<<' '<<step++<<' '<<policy_state_key(obs)<<' '<<a<<'\n';return a;
    }
};
void weights(const ProcedureMemory& model,const std::string& path) {
    std::ofstream f(path);f<<std::setprecision(9);for(float v:model.skills.kernel.routes.W)f<<v<<'\n';
}
int main(int argc,char** argv)try {
    check(argc==5||argc==6,"usage: composed_credit_probe seed output_directory evaluations boolean|numeric|repair [proposal_quantum]");
    bool numeric=std::string(argv[4])=="numeric",repair=std::string(argv[4])=="repair";
    check(numeric||repair||std::string(argv[4])=="boolean","invalid curriculum profile");
    unsigned seed=std::stoul(argv[1]);std::filesystem::create_directories(argv[2]);std::string prefix=std::string(argv[2])+"/"+std::to_string(seed);
    PrimitiveStore store(seed);PrimitiveTrace prior_trace(prefix+".prior");
    for(int cue=0;cue<(repair?3:2);++cue) {
        std::mt19937 rng(1000+seed);
        for(int size:{0,1,2,3,5,8})for(int i=0;i<(size<=3?8:4);++i) {
            auto e=generate(rng,cue,size,size*100+i,repair&&cue==2?0:2);PrimitiveReceipt r;
            r.id="skill"+std::to_string(cue)+"x"+std::to_string(size*8+i);r.cue=cue;r.relation=e.input.relation;
            r.subject=e.input.subject;r.target=e.input.target;r.facts=e.input.facts;r.answer=e.answer;
            if(r.target.type==1)r.target.id%=2;if(cue==1)r.answer={1,r.subject==r.target};
            check(store.teach(r),"direct prior rejected");
        }
        store.rebuild();check(store.memory.available_skills[cue],"prior acquisition failed");
        save_policy(store.policies()[cue],prefix+".prior"+std::to_string(cue)+".policy.txt");
        std::cout<<"seed="<<seed<<" prior="<<cue<<" acquired\n"<<std::flush;
    }
    for(const auto& [id,r]:store.receipts()) {
        Example e;e.input={r.cue,r.relation,r.subject,r.target,r.facts};e.answer=r.answer;e.size=r.cue==2?r.answer.id:0;
        prior_trace.add(e,run(store.memory.skills,e.input,256,true),"prior");
    }
    check(store.memory.available_skills[2]==repair,"unexpected counting prior availability");
    if(repair) {
        PrimitiveTrace warm(prefix+".warm");std::mt19937 test_rng(39000+seed);int correct=0,total=0;
        for(int size:{0,1,2,3,4,8,12,24,64})for(int i=0;i<12;++i) {
            auto e=generate(test_rng,2,size,210000+size*100+i,4);auto r=run(store.memory.skills,e.input,1024,true);
            correct+=r.output==e.answer;++total;warm.add(e,r,"before_repair");
        }
        std::cout<<"seed="<<seed<<" available_count_before_repair="<<correct<<'/'<<total<<'\n'<<std::flush;
    }
    std::ofstream inputs(prefix+".questions.txt"),curve(prefix+".curve.tsv"),programs(prefix+".programs.txt");
    std::vector<std::vector<VerifiedQuestion>> groups(numeric?1:2);std::mt19937 rng(seed+28000);int serial=1;
    for(int family=numeric?2:0;family<(numeric?3:2);++family)for(int n=0;n<4;++n)for(int other=0;other<4;++other)for(int repeat=0;repeat<2;++repeat) {
        auto e=composed(rng,family,n,other,1+repeat,serial);question(inputs,e,"train",serial,n);++serial;groups[numeric?0:family].push_back(e);
    }
    auto initial=store.memory.skills.kernel.routes.W;
    weights(store.memory,prefix+".initial.weights.txt");
    ComposedSkillDiscovery discovery;discovery.evaluation_budget=std::stoi(argv[3]);discovery.repair_available=repair;
    discovery.interleave_quantum=argc==6?std::stoi(argv[5]):0;
    check(discovery.interleave_quantum>=0&&(!discovery.interleave_quantum||repair),"invalid interleaved repair configuration");
    auto result=discovery.learn(store,groups,seed,curve);
    if(repair) {
        std::ofstream attempts(prefix+".attempts.tsv");attempts<<"cue\tavailable\tcorrect\ttotal\tevaluations\tfirst\tsteps\tstates\n";
        for(const auto& a:result.attempts)attempts<<a.cue<<'\t'<<a.previously_available<<'\t'<<a.correct<<'\t'<<a.total<<'\t'<<a.evaluations<<'\t'<<a.first_solution<<'\t'<<a.proposal_steps<<'\t'<<a.program_states<<'\n';
    }
    check(store.memory.skills.kernel.routes.W==initial&&store.memory.available_skills[2]==repair,"hypothesis search leaked into live policy");
    std::cout<<"seed="<<seed<<" acquired="<<result.acquired<<" cue="<<result.cue<<" correct="<<result.correct<<'/'<<result.total<<" proposals="<<result.evaluations<<" first="<<result.first_solution<<" microsteps="<<result.proposal_steps<<" macro_states="<<result.program_states<<'\n'<<std::flush;
    if(result.acquired)save_policy(result.policy,prefix+".policy.txt");
    std::ofstream summary(prefix+".summary.txt");summary<<result.acquired<<' '<<result.cue<<' '<<result.correct<<' '<<result.total<<' '<<result.evaluations<<' '<<result.first_solution<<' '<<result.proposal_steps<<' '<<result.program_states<<'\n';
    if(result.acquired) {
        std::map<int,std::vector<VerifiedQuestion>> teaching;
        for(const auto& rows:groups)teaching[rows.front().query.cue]=rows;
        for(int family=0;family<4;++family)if(!teaching.count(71+family))
            for(int n=0;n<4;++n)for(int other=0;other<4;++other)for(int repeat=0;repeat<2;++repeat) {
                auto e=composed(rng,family,n,other,1+repeat,serial);question(inputs,e,"transfer_train",serial,n);++serial;teaching[e.query.cue].push_back(e);
            }
        std::vector<std::pair<int,VerifiedQuestion>> tests;
        for(int family=0;family<4;++family)for(int size:{0,1,2,3,4,8,12})for(int i=0;i<20;++i) {
            auto e=composed(rng,family,size,i%2?size:(size+1)%13,4+4*(i%3),serial);question(inputs,e,"test",serial,size);tests.push_back({serial++,e});
        }
        std::ofstream micro(prefix+".micro.txt"),answers(prefix+".answers.txt"),samples(prefix+".derived_samples.txt");
        for(const auto& [key,s]:result.rehearsal){samples<<key<<' '<<s.action<<' '<<s.available.size();for(int a:s.available)samples<<' '<<a;samples<<'\n';}
        for(const std::string name:{"symbolic","rehearsal","no_rehearsal","unacquired"}) {
            auto model=name=="rehearsal"||name=="no_rehearsal"?discovery.rehearse(store,result,seed,name=="rehearsal"):store.memory;
            std::map<uint32_t,int> base_cache;ComposedSkillDiscovery::Choice symbolic{model.skills,result.policy,result.cue,base_cache};
            if(name=="symbolic")model.available_skills[result.cue]=true;
            for(const auto& [cue,rows]:teaching) {
                auto learned=name=="symbolic"?model.learn_with(rows,symbolic,false,10,50000):model.learn(rows,false,10,50000);
                programs<<name<<' '<<cue<<' '<<learned.found<<' '<<learned.states<<' '<<learned.program.size();for(int op:learned.program)programs<<' '<<op;programs<<'\n';
            }
            weights(model,prefix+"."+name+".weights.txt");auto frozen=model.skills.kernel.routes.W;
            int train_id=1,correct=0,total=0;
            auto evaluate=[&](int id,const VerifiedQuestion& e) {
                auto found=model.procedures.find(e.query.cue);Value y;auto before=model.skill_steps;
                if(found!=model.procedures.end()) {
                    if(name=="symbolic") {RecordingChoice<ComposedSkillDiscovery::Choice> record{symbolic,micro,id,name};y=model.execute_with(e.query,found->second,record);}
                    else {RecordingChoice<LocalChoice> record{model.skills,micro,id,name};y=model.execute_with(e.query,found->second,record);}
                }
                answers<<id<<' '<<name<<' '<<y.type<<' '<<y.id<<' '<<(model.skill_steps-before)<<'\n';return y==e.answer;
            };
            for(const auto& rows:groups)for(const auto& e:rows)evaluate(train_id++,e);
            for(const auto& [id,e]:tests){correct+=evaluate(id,e);++total;}
            PrimitiveTrace primitive_trace(prefix+"."+name);
            std::mt19937 test_rng(39000+seed);
            for(int task=0;task<3;++task)for(int size:{0,1,2,3,4,8,12,24,64})for(int i=0;i<12;++i) {
                auto e=generate(test_rng,task,size,200000+task*10000+size*100+i,4);
                if(e.input.target.type==1)e.input.target.id%=2;if(task==1)e.answer={1,e.input.subject==e.input.target};
                auto r=name=="symbolic"?run(symbolic,e.input,1024,true):run(model.skills,e.input,1024,true);
                primitive_trace.add(e,r,"heldout");
            }
            check(model.skills.kernel.routes.W==frozen,"evaluation changed primitive weights");
            std::cout<<"seed="<<seed<<" model="<<name<<" composed="<<correct<<'/'<<total<<'\n'<<std::flush;
        }
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
