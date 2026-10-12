#include "laminar/continual_training.h"
#include <iostream>
using namespace laminar::relational;
void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
int main() {
    for(bool terminal:{false,true}) {
        ControlledRelationalSession learner(5,true,terminal,true,{},false,32,{},true);
        check(learner.begin("return_test","unknown a",{"a"},{{{},false},0},{},false),"begin failed");
        auto read=learner.choose(0,-1,false,nullptr,ControlHint{ControlOp::ask,"credit_fixture"});
        check(learner.read(read.token,StampedScene{{{},true},1},"verified_environment"),"read failed");
        learner.choose(0,-1,false,nullptr,ControlHint{ControlOp::reason,"credit_fixture"});
        learner.choose(0,-1,false,nullptr,ControlHint{ControlOp::finish,"credit_fixture"});
        auto trace=learner.trace();check(learner.finish(false,false,3),"failed goal feedback rejected");
        if(terminal) {
            check(!learner.empirical_values().count(trace[0].observation),"terminal-only control credited first action");
            continue;
        }
        // Independent discounted sum, not the implementation's backwards loop.
        for(size_t i=0;i<trace.size();++i) {
            double expected=-std::pow(.99,double(trace.size()-1-i));
            for(size_t j=i;j<trace.size();++j)expected-=std::pow(.99,double(j-i))*trace[j].cost;
            auto value=learner.empirical_values().at(trace[i].observation)[int(trace[i].operation)];
            check(std::abs(value-.2*expected)<1e-6,"episode return disagrees with discounted observed reward/cost sum");
        }
        auto before=learner.policy.weights;
        check(learner.revise_credit("return_test",1,true,true)&&learner.revise_credit("return_test",2,false,false)&&before==learner.policy.weights,"episode-return correction failed exact replay");
    }
    for(const auto& profile:{"controller_return","controller50k_return"}) {
        auto config=laminar::training::ContinualTrainingConfig::named(profile);auto learner=config.create(1);
        check(config.episodic_return&&learner.policy.weights.size()==size_t(38*config.controller_hidden+5),"return profile shape drift");
    }
    std::cout<<"Episode returns match independently summed outcomes/costs; truncated control and correction replay passed\n";
}
