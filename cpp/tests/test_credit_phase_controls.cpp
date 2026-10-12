#include "laminar/continual_training.h"
#include <iostream>
using namespace laminar;
void check(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
void episode(relational::ControlledRelationalSession& learner,const std::string& id) {
    using namespace relational;
    check(learner.begin(id,"unknown a",{"a"},{{{},false},0},{},false),"phase begin failed");
    auto read=learner.choose(0,-1,false,nullptr,ControlHint{ControlOp::ask,"phase_fixture"});
    check(learner.read(read.token,StampedScene{{{},true},1},"verified_environment"),"phase read failed");
    learner.choose(0,-1,false,nullptr,ControlHint{ControlOp::reason,"phase_fixture"});
    learner.choose(0,-1,false,nullptr,ControlHint{ControlOp::finish,"phase_fixture"});
}
int main() {
    for(const auto& base:{std::string("controller50k"),std::string("controller50k_trace")}) {
        auto normal=training::ContinualTrainingConfig::named(base);
        check(normal.json(1).find("control_ablation_after")==std::string::npos,"default profile fingerprint changed");
        for(const auto& suffix:{std::string("_frozen_after400"),std::string("_shuffled_after400")}) {
            auto config=training::ContinualTrainingConfig::named(base+suffix);
            check(config.control_ablation_after==400&&!config.control_frozen(399)&&!config.credit_shuffled(399),"ablation started during warm start");
            check(config.control_frozen(400)==config.freeze_control&&config.credit_shuffled(400)==config.shuffled_credit,"ablation did not start at episode401");
            check(config.greedy_trace_credit==normal.greedy_trace_credit&&config.controller_hidden==normal.controller_hidden,"phase profile changed baseline learner");
            auto altered=config;altered.control_ablation_after=399;check(config.json(1)!=altered.json(1),"phase boundary absent from fingerprint");
            // Use the same mechanism with a short boundary for an exact
            // matched-state contract. The named file-run profiles remain400.
            config.control_ablation_after=4;config.controller_hidden=32;
            auto baseline=normal;baseline.controller_hidden=32;
            auto reference=baseline.create(5),control=config.create(5);
            std::vector<float> at_freeze;size_t boundary_updates=0;bool shuffled_outcome=false;
            for(size_t completed=0;completed<12;++completed) {
                auto id="phase"+std::to_string(completed);episode(reference,id);episode(control,id);
                if(completed==4){at_freeze=control.policy.weights;boundary_updates=control.updates();}
                bool correct=completed%2==0;
                check(reference.finish(correct,false,16)&&control.finish(correct,false,config.control_frozen(completed)?0:16,config.credit_shuffled(completed)),"phase credit failed");
                if(completed<4)check(reference.policy.weights==control.policy.weights&&reference.empirical_values()==control.empirical_values(),"warm state differs before phase boundary");
                if(completed>=4&&config.freeze_control)check(control.policy.weights==at_freeze&&control.updates()==boundary_updates,"frozen control learned after phase boundary");
                if(completed>=4&&config.shuffled_credit)shuffled_outcome|=control.credits().back().reward!=(correct?1.f:-1.f);
            }
            check(control.knowledge.version()==reference.knowledge.version()&&control.credits().size()==12,"credit control stopped ordinary evidence learning");
            if(config.shuffled_credit)check(shuffled_outcome&&reference.policy.weights!=control.policy.weights,"shuffled phase did not change terminal credit");
        }
    }
    std::cout<<"Credit phase controls: exact warm-state parity, boundary activation, later frozen weights, shuffled outcomes, continued evidence and configuration fingerprints passed\n";
}
