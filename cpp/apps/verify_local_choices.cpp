// Reload a frozen native policy and verify logged public-state choices.
#include "laminar/reasoning_runtime.h"
#include <fstream>
#include <iostream>
using namespace laminar::reasoning;
int main(int argc,char** argv)try {
    bool export_choices=argc>3&&std::string(argv[argc-1])=="--export";
    int options=argc-int(export_choices);
    if(options!=3&&options!=6)throw std::runtime_error("usage: verify_local_choices seed weights [units active channels] [--export]");
    SkillShape shape;if(options==6)shape={std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5])};
    LocalChoice policy(10,std::stoul(argv[1]),shape);std::ifstream weights(argv[2]);
    for(auto& w:policy.kernel.routes.W)if(!(weights>>w)||!std::isfinite(w))throw std::runtime_error("invalid weight file");
    float extra;if(weights>>extra)throw std::runtime_error("trailing weight data");
    if(export_choices) {
        // Enumerate the public observation abstraction, including unreachable
        // combinations. Independent VM replay selects only reachable states.
        for(int task=0;task<3;++task)for(int flags=0;flags<64;++flags)
        for(int predicate=-1;predicate<=1;++predicate)for(int last=-1;last<=9;++last) {
            Observation obs{};obs[task]=obs[31]=1;
            for(int i=0;i<6;++i)obs[3+2*i+((flags>>i)&1)]=1;
            obs[16+predicate]=obs[19+last]=1;
            std::vector<int> available{0,4,9};
            if(flags&1)available.insert(available.end(),{1,5,6});
            if(flags&2)available.insert(available.end(),{2,3,7});
            if(predicate>=0)available.push_back(8);
            uint32_t key=0;for(int i=0;i<32;++i)if(obs[i])key|=uint32_t(1)<<i;
            std::cout<<key<<' '<<policy.choose(obs,available)<<'\n';
        }
        return 0;
    }
    uint32_t key;int expected,n;size_t checked=0;
    while(std::cin>>key>>expected>>n) {
        if(n<1||n>10)throw std::runtime_error("invalid available count");std::vector<int> available(n);
        for(auto& a:available)if(!(std::cin>>a)||a<0||a>9)throw std::runtime_error("invalid available operation");
        Observation obs{};for(int i=0;i<32;++i)obs[i]=float((key>>i)&1);
        if(policy.choose(obs,available)!=expected)throw std::runtime_error("logged choice differs from saved neural weights");++checked;
    }
    if(!std::cin.eof())throw std::runtime_error("invalid choice data");std::cout<<checked<<" frozen public-state choices verified\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
