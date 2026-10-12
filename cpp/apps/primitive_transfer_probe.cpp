#include "primitive_discovery_support.h"
#include "laminar/procedure_memory.h"
std::vector<VerifiedQuestion> read_questions(const std::string& path) {
    std::ifstream f(path);check(bool(f),"missing flat corpus");std::vector<VerifiedQuestion> out;VerifiedQuestion e;int a,b,n;
    while(f>>e.query.cue>>a>>b>>e.query.relation[0]>>e.query.relation[1]>>e.answer.type>>e.answer.id>>n){
        check(n>=0&&n<10000,"invalid fact count");e.query.a={0,a};e.query.b={0,b};e.query.facts.resize(n);
        for(auto& x:e.query.facts)check(bool(f>>x.subject>>x.relation>>x.object),"truncated flat corpus");out.push_back(e);
    }
    check(f.eof(),"malformed flat corpus");return out;
}
void save_weights(const LocalChoice& policy,const std::string& path){
    std::ofstream out(path);check(bool(out),"open neural weights");out<<std::setprecision(9);
    for(float v:policy.kernel.routes.W){check(std::isfinite(v),"nonfinite route");out<<v<<'\n';}
}
int main(int argc,char** argv)try{
    check(argc==3,"usage: primitive_transfer_probe seed replay|no_replay");
    int seed=std::stoi(argv[1]);std::string mode=argv[2];check(mode=="replay"||mode=="no_replay","unknown mode");
    const auto prefix="logs/primitive_discovery/"+std::to_string(seed)+"."+mode;
    std::array<DiscoveredPolicy,3> policies;for(int t=0;t<3;++t)policies[t]=load_policy("logs/primitive_discovery/"+std::to_string(seed)+"."+std::to_string(t)+".search.policy.txt");
    ProcedureMemory memory(seed);PrimitiveTrace trace(prefix);
    std::vector<Sample> prior;
    std::ofstream samples(prefix+".self_samples.tsv");check(bool(samples),"open self samples");samples<<"stage\tstate\taction\tavailable\n";
    std::mt19937 shuffle_rng(8100+seed);
    save_weights(memory.skills,prefix+".initial.weights.txt");
    for(int task:{0,1,2}){
        // Data generation must not depend on how much rehearsal the learner
        // performs. Both arms receive byte-identical new-skill examples.
        std::mt19937 data_rng(7700+seed+task*100);
        std::map<uint32_t,Sample> unique;
        for(int i=0;i<256;++i){
            // Fresh short training worlds; the only external teaching signal is
            // whether the discovered policy's complete answer was correct.
            auto e=generate(data_rng,task,data_rng()%4,200000+task*1000+i);
            auto r=run(policies[task],e.input,128,true);
            check(r.output==e.answer,"discovered policy failed self-generation training validation");
            trace.add(e,r,"self_generation");
            for(const auto& s:r.decisions){
                auto key=policy_state_key(s.obs);auto old=unique.find(key);
                check(old==unique.end()||old->second.action==s.action,"inconsistent self-generated policy");unique[key]=s;
            }
        }
        std::vector<Sample> bank=mode=="replay"?prior:std::vector<Sample>{};
        for(auto [key,s]:unique){bank.push_back(s);prior.push_back(s);samples<<task<<'\t'<<key<<'\t'<<s.action<<'\t';for(int a:s.available)samples<<a<<',';samples<<'\n';}
        for(int epoch=0;epoch<128;++epoch){std::shuffle(bank.begin(),bank.end(),shuffle_rng);for(const auto& s:bank)memory.skills.fit(s);}
        save_weights(memory.skills,prefix+".stage"+std::to_string(task)+".weights.txt");
        auto before=memory.skills.kernel.routes.W;
        for(int old=0;old<=task;++old)evaluate(memory.skills,seed,old,"after_"+std::to_string(task),trace,task==2);
        check(before==memory.skills.kernel.routes.W,"primitive evaluation changed weights");
    }
    // Same native ProcedureMemory used by the grounded reader. Its primitive
    // weights now derive solely from the learner's outcome-discovered policies.
    // No files of teacher micro-instructions are loaded.
    const auto tail="_"+std::to_string(seed)+".txt";
    auto train=read_questions("logs/reasoning_flat_corpus/train"+tail);
    auto tests=read_questions("logs/reasoning_flat_corpus/test"+tail);
    std::ofstream programs(prefix+".programs.tsv"),answers(prefix+".composition.tsv");
    check(bool(programs)&&bool(answers),"open composition logs");
    programs<<"cue\tfound\tstates\tinstructions\n";answers<<"case\tcue\toutput_type\toutput\n";
    const auto weights=memory.skills.kernel.routes.W;
    for(int cue=0;cue<7;++cue){
        std::vector<VerifiedQuestion> examples;for(const auto& e:train)if(e.query.cue==cue)examples.push_back(e);
        auto result=memory.learn(examples);programs<<cue<<'\t'<<result.found<<'\t'<<result.states<<'\t';for(int a:result.program)programs<<a<<',';programs<<'\n';
    }
    auto frozen_programs=memory.procedures;std::map<int,std::pair<int,int>> counts;
    for(size_t i=0;i<tests.size();++i){
        const auto& e=tests[i];auto y=memory.answer(e.query);answers<<i<<'\t'<<e.query.cue<<'\t'<<y.type<<'\t'<<y.id<<'\n';
        auto& c=counts[e.query.cue];c.first+=y==e.answer;++c.second;
    }
    check(weights==memory.skills.kernel.routes.W&&frozen_programs==memory.procedures,"composition test changed model");
    std::cout<<"seed="<<seed<<" mode="<<mode<<" composition";for(auto [cue,c]:counts)std::cout<<" cue"<<cue<<'='<<c.first<<'/'<<c.second;std::cout<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
