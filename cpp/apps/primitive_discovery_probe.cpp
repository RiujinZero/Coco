// No teacher instructions/templates/intermediate correctness labels.
#include "primitive_discovery_support.h"
#include "laminar/backprop_choice.h"
int main(int argc,char** argv)try {
    check(argc==5,"usage: primitive_discovery_probe seed task search|first|terminal|random|frozen|backprop evaluations");
    int seed=std::stoi(argv[1]),task=std::stoi(argv[2]),budget=std::stoi(argv[4]);std::string mode=argv[3];
    check(seed>=0&&task>=0&&task<3&&budget>0,"invalid arguments");
    check(mode=="search"||mode=="first"||mode=="terminal"||mode=="random"||mode=="frozen"||mode=="backprop","unknown mode");
    std::string prefix="logs/primitive_discovery/"+std::to_string(seed)+"."+std::to_string(task)+"."+mode;
    PrimitiveTrace trace(prefix);std::mt19937 rng(1000+seed);std::vector<Example> train;
    for(int size=0;size<=3;++size)for(int i=0;i<8;++i)train.push_back(generate(rng,task,size,size*100+i));
    std::ofstream config(prefix+".config.tsv");check(bool(config),"open config");config<<"seed\ttask\tmode\tbudget\n"<<seed<<'\t'<<task<<'\t'<<mode<<'\t'<<budget<<'\n';
    if(mode!="backprop"){
        std::ofstream curve(prefix+".curve.tsv");check(bool(curve),"open curve");curve<<"candidate\tcorrect\ttotal\tsteps\n";
        PolicyDiscovery learner;learner.evaluation_budget=mode=="frozen"?1:budget;learner.stop_at_first=mode=="first";
        std::mt19937 rewards(seed+881);
        auto result=learner.learn(seed,[&](DiscoveredPolicy& p){
            auto a=assess(p,train,48,mode=="terminal"?1:0);
            if(mode=="random"){a.correct=0;for(int n=0;n<a.total;++n)a.correct+=int(rewards()%2);}
            return a;
        },curve);
        auto actual=assess(result.policy,train,48);
        result.policy.seal(actual.visited);save_policy(result.policy,prefix+".policy.txt");
        for(const auto& e:train){auto r=run(result.policy,e.input,48,true);trace.add(e,r,"training");}
        auto before=result.policy.rules;evaluate(result.policy,seed,task,mode,trace);
        check(before==result.policy.rules&&result.policy.sealed,"evaluation changed policy");
    }else{
        BackpropChoice policy(10,seed);
        for(int i=0;i<budget;++i) {
            auto e=generate(rng,task,rng()%4,i);auto v=start(e.input);std::vector<Sample> history;
            for(int t=0;t<48&&!v.done;++t){
                auto obs=v.observe();auto available=v.available();int a=policy.choose(obs,available);
                if(rng()%100<20)a=available[rng()%available.size()];
                history.push_back({obs,available,a});v.step(a,e.input.facts);
            }
            float reward=v.output==e.answer?1.f:-1.f;
            for(int j=int(history.size())-1;j>=0;--j){
                const auto& s=history[j];float target=j+1==int(history.size())?reward:0.99f*policy.best_value(history[j+1].obs,history[j+1].available)-0.002f;
                policy.value_update(s.obs,s.action,target);
            }
        }
        std::ofstream weights(prefix+".weights.txt");check(bool(weights),"open weights");weights<<std::setprecision(9);
        for(float w:policy.weights){check(std::isfinite(w),"nonfinite weight");weights<<w<<'\n';}
        auto before=policy.weights;evaluate(policy,seed,task,mode,trace);check(before==policy.weights,"evaluation changed weights");
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
