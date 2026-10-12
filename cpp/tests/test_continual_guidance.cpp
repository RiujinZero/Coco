#include "../apps/continual_guidance.h"
#include <iostream>
using namespace laminar::training;
using namespace laminar::relational;
void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
int main() {
    GuidanceSchedule g;g.configured=true;g.source="public_control_v1";g.full=4;g.fade=4;
    check(g.rate(0)==1&&g.rate(4)==1&&g.rate(6)==.5&&g.rate(8)==0,"taper boundary error");
    size_t hints=0;for(size_t s=0;s<1000;++s) {
        check(g.intervene(0,s)&&!g.intervene(8,s),"guidance escaped phase");
        hints+=g.intervene(6,s);
    }
    check(hints>400&&hints<600,"taper lottery biased");
    ControlledRelationalSession a(1),b(1);
    auto begin=[](auto& learner){return learner.begin("lesson","query u",{"u"},{{{},true},1},{},false);};
    check(begin(a)&&begin(b),"begin failed");
    try {a.choose(0,-1,false,nullptr,ControlHint{ControlOp::finish,"invalid source"});check(false,"invalid source accepted");}catch(const std::invalid_argument&){}
    auto first=a.choose(0,-1,false,nullptr,ControlHint{ControlOp::finish,"verified_demo"});
    auto other=b.choose(0,-1,false,nullptr,ControlHint{ControlOp::finish,"verified_demo"});
    check(first.token==other.token&&first.teacher=="verified_demo"&&first.operation==ControlOp::finish,"hint attribution or atomic rejection failed");
    check(a.finish(false,false,0),"verified failure rejected");
    check(a.credits().back().trace.front().teacher=="verified_demo"&&a.credits().back().reward==-1,"teacher trace lost or assumed correct");
    ControlledRelationalSession frozen(1);frozen.set_learning(false);check(begin(frozen),"frozen begin failed");
    try {frozen.choose(0,-1,false,nullptr,ControlHint{ControlOp::finish,"verified_demo"});check(false,"frozen hint accepted");}catch(const std::invalid_argument&){}
    check(frozen.trace().empty(),"frozen rejection mutated trace");
    std::cout<<"Guidance taper, attributed credit, atomic rejection and frozen evaluation guard passed\n";
}
