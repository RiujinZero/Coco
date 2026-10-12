#include "laminar/computation_control.h"
#include "laminar/backprop_choice.h"
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <iomanip>
using namespace laminar::reasoning;
void check(bool v,const char* why){if(!v)throw std::runtime_error(why);}

// Evaluator/environment data. None of this type is passed into the policy.
struct World {
    int root=0,goal=0,marker=-1;
    bool effect=false;
    std::map<int,int> truth,archive;
};
struct Machine {
    int focus=0,next=-1,scope=0;
    bool done=false;
    World& world;
    explicit Machine(World& w):focus(w.root),world(w){}
    void read(const std::map<int,int>& store) {
        auto it=store.find(focus);
        if(it==store.end()){if(scope!=1&&scope!=2)scope=3;return;}
        next=it->second;scope=next<0?2:1;
    }
    ComputationStatus status(int price)const {
        ComputationStatus s;s.scope=scope;s.effect_requested=world.effect;s.effect_at_focus=world.marker==focus;
        s.costs={price?0.05f:0.01f,0.005f,price?0.01f:0.05f,0.02f,0};return s;
    }
    void step(ComputationOp op) {
        switch(op) {
        case ComputationOp::retrieve:read(world.archive);break;
        case ComputationOp::ask:read(world.truth);break; // one requested row, never the final answer
        case ComputationOp::reason:check(scope==1,"missing operand");focus=next;next=-1;scope=0;break;
        case ComputationOp::act:world.marker=focus;break;
        case ComputationOp::finish:done=true;break;
        }
    }
    bool correct()const{return done&&focus==world.goal&&(!world.effect||world.marker==world.goal);}
};
World make_world(std::mt19937& r,int length,bool effect,int base,int missing) {
    World w;w.effect=effect;std::vector<int> ids(length+9);std::iota(ids.begin(),ids.end(),base);std::shuffle(ids.begin(),ids.end(),r);
    w.root=ids[0];w.goal=ids[length];
    for(int i=0;i<=length;++i){w.truth[ids[i]]=i==length?-1:ids[i+1];if(int(r()%100)>=missing)w.archive[ids[i]]=w.truth[ids[i]];}
    for(int i=length+1;i<int(ids.size());++i){w.truth[ids[i]]=ids.back();w.archive[ids[i]]=ids.back();}
    return w;
}
struct Result {bool correct=false,timeout=false;int steps=0;float cost=0;std::array<int,5> actions{};};
template<class Learner> Result run(Learner& learner,World& world,int price,bool cached,int budget,float epsilon,
           std::ofstream* log=nullptr,int trial=0,int disabled=-1) {
    Machine m(world);if(cached)m.read(world.truth);learner.begin();Result out;
    for(int t=0;t<budget&&!m.done;++t) {
        auto status=m.status(price);auto op=learner.choose(status,epsilon,disabled);int a=int(op);
        if(log)*log<<trial<<'\t'<<t<<'\t'<<m.focus<<'\t'<<m.scope<<'\t'<<m.next<<'\t'<<world.marker<<'\t'<<a<<'\t'<<status.costs[a]<<'\n';
        m.step(op);++out.steps;++out.actions[a];out.cost+=status.costs[a];
    }
    out.correct=m.correct();out.timeout=!m.done;return out;
}
template<class Learner> const std::vector<float>& weights(const Learner& learner) {
    if constexpr(requires{learner.policy.kernel.routes.W;})return learner.policy.kernel.routes.W;
    else return learner.policy.weights;
}
template<class Learner> void experiment(int seed,int episodes,const std::string& mode,const std::string& dir) {
    Learner learner(seed);std::mt19937 rng(seed+9021),credit_rng(seed+519);
    const bool frozen=mode.find("frozen")!=std::string::npos,shuffled=mode.find("shuffled")!=std::string::npos,terminal=mode.find("terminal")!=std::string::npos;
    const auto initial=weights(learner);
    for(int i=0;i<episodes;++i) {
        // One mixed stream; no skill demonstrations or next-step targets.
        int length=rng()%4,price=rng()%2;bool effect=rng()%2,cached=rng()%4==0;
        auto w=make_world(rng,length,effect,0,30);
        float epsilon=std::max(0.05f,0.35f*(1.f-float(i)/episodes));
        auto result=run(learner,w,price,cached,32,epsilon);
        float outcome=result.correct?1.f:-1.f;
        if(shuffled)outcome=credit_rng()%2?1.f:-1.f;
        if(!frozen)learner.credit(outcome,terminal);
    }
    const auto learned=weights(learner);double delta=0;
    for(size_t i=0;i<learned.size();++i){check(std::isfinite(learned[i]),"nonfinite learned weight");delta+=std::abs(learned[i]-initial[i]);}
    check(frozen?delta==0:delta>0,"weight change contract");
    std::string prefix=dir+"/"+std::to_string(seed)+"."+mode;
    std::ofstream cases(prefix+".cases.tsv"),steps(prefix+".steps.tsv"),weight_file(prefix+".weights.txt");
    check(bool(cases)&&bool(steps)&&bool(weight_file),"open output failed");
    cases<<"trial\tlength\teffect\tprice\tcached\tdisabled\troot\tgoal\ttruth\tarchive\tcorrect\ttimeout\tsteps\tcost\n";
    steps<<"trial\tstep\tfocus\tscope\tnext\tmarker\taction\tcost\n";
    weight_file<<std::setprecision(9);for(auto v:learned)weight_file<<v<<'\n';
    std::ofstream initial_file(prefix+".initial_weights.txt");check(bool(initial_file),"open initial weights failed");
    initial_file<<std::setprecision(9);for(auto v:initial)initial_file<<v<<'\n';
    std::mt19937 test_rng(77871+seed);int trial=0,correct=0,total=0;double cost=0;std::array<int,5> acts{};
    // Identical test worlds across arms, fresh entity IDs, unseen path lengths,
    // complete/missing memory, cached operands, both intents and cost regimes.
    for(int disabled:{-1,0,1,2,3})for(int length:{0,1,4,12,32})for(int effect=0;effect<2;++effect)for(int price=0;price<2;++price)for(int rep=0;rep<12;++rep) {
        bool cached=rep%3==0;int missing=rep%3==1?0:40;
        auto w=make_world(test_rng,length,effect,100000+trial*100,missing);int budget=4*(length+1)+8;
        auto r=run(learner,w,price,cached,budget,0,&steps,trial,disabled);
        cases<<trial<<'\t'<<length<<'\t'<<effect<<'\t'<<price<<'\t'<<cached<<'\t'<<disabled<<'\t'<<w.root<<'\t'<<w.goal<<'\t';
        for(auto [a,b]:w.truth)cases<<a<<':'<<b<<';';cases<<'\t';for(auto [a,b]:w.archive)cases<<a<<':'<<b<<';';
        cases<<'\t'<<r.correct<<'\t'<<r.timeout<<'\t'<<r.steps<<'\t'<<r.cost<<'\n';
        if(disabled==-1){correct+=r.correct;++total;cost+=r.cost;for(int a=0;a<5;++a)acts[a]+=r.actions[a];}
        ++trial;
    }
    check(learned==weights(learner),"evaluation modified weights");
    std::cout<<"seed="<<seed<<" mode="<<mode<<" correct="<<correct<<'/'<<total<<" cost="<<cost<<" weight_l1_change="<<delta<<" actions=";
    for(auto a:acts)std::cout<<a<<',';std::cout<<'\n';
}
int main(int argc,char** argv)try {
    check(argc>=4,"usage: computation_control_probe seed episodes learned|frozen|shuffled|terminal|backprop|backprop_frozen|backprop_shuffled|backprop_terminal [output_dir]");
    int seed=std::stoi(argv[1]),episodes=std::stoi(argv[2]);std::string mode=argv[3],dir=argc>4?argv[4]:"logs/computation_control";
    check(mode=="learned"||mode=="frozen"||mode=="shuffled"||mode=="terminal"||mode=="backprop"||mode=="backprop_frozen"||mode=="backprop_shuffled"||mode=="backprop_terminal","unknown mode");
    if(mode.rfind("backprop",0)==0)experiment<BasicComputationControl<BackpropChoice>>(seed,episodes,mode,dir);
    else experiment<ComputationControl>(seed,episodes,mode,dir);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
