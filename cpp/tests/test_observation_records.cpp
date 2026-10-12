#include "laminar/continual_training.h"
#include <iostream>
using namespace laminar::relational;
void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
int main() {
    Scene records{{{"edge",{"a","b"}},{"edge",{"a","b"}}},true};
    ControlledRelationalSession baseline,learner;
    learner.preserve_observation_records();
    check(baseline.begin("base","query a with b",{"a","b"},{records,1},{},false),"default begin rejected");
    check(learner.begin("parent","query a with b",{"a","b"},{records,1},{},false),"records begin rejected");
    check(baseline.visible().scene.facts.size()==1&&learner.visible().scene.facts.size()==2,"observation multiplicity boundary wrong");
    auto step=learner.choose(0,-1,false,nullptr,ControlHint{ControlOp::ask,"protocol_test"});
    records.facts.push_back(records.facts.front());
    check(learner.read(step.token,StampedScene{records,2},"verified_environment")&&learner.visible().scene.facts.size()==3,"read normalized records");
    check(learner.suspend()&&learner.resume("parent",{records,3})&&learner.visible().scene.facts.size()==3,"resume normalized records");
    bool rejected=false;try{learner.preserve_observation_records();}catch(const std::logic_error&){rejected=true;}
    check(rejected,"late semantic configuration accepted");
    auto config=laminar::training::ContinualTrainingConfig::named("controller50k_repair_records");
    check(config.observation_records&&config.repair_available&&config.interleave_quantum==1000,"records profile wrong");
    check(config.json(1).find("sorted_with_multiplicity")!=std::string::npos,"record semantics missing from fingerprint");
    std::cout<<"observation records retained across begin/read/resume; default set semantics unchanged\n";
}
