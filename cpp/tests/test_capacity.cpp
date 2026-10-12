#include "laminar/continual_training.h"
#include <iostream>
using namespace laminar::reasoning;
void check(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
int main() {
    // Dropping the two unused outcome channels must preserve every learned
    // scalar prediction and decision at the original expansion size.
    LocalChoice original(10,1),compact(10,1,{512,16,1});
    for(int epoch=0;epoch<4;++epoch)for(int i=0;i<31;++i) {
        Sample s;s.obs[31]=1;s.obs[i]=1;s.available={0,2,4,9};s.action=s.available[i%4];
        original.fit(s);compact.fit(s);
        for(int a:s.available)check(original.action_value(s.obs,a)==compact.action_value(s.obs,a),"channel compaction changed prediction");
        check(original.choose(s.obs,s.available)==compact.choose(s.obs,s.available),"channel compaction changed choice");
        check(original.best_value(s.obs,s.available)==compact.best_value(s.obs,s.available),"channel compaction changed best value");
    }
    for(size_t i=0;i<compact.kernel.routes.W.size();++i)check(compact.kernel.routes.W[i]==original.kernel.routes.W[i*3],"channel compaction changed plasticity");
    for(const auto& profile:{"controller50k","controller100k","agkern50k","agkern100k","both50k","both100k","controller50k_repair"}) {
        auto c=laminar::training::ContinualTrainingConfig::named(profile);auto learner=c.create(7);
        check(c.repair_available==(std::string(profile)=="controller50k_repair"),"repair silently changed an existing profile");
        check(learner.policy.weights.size()==size_t(38*c.controller_hidden+5),"wrong controller capacity");
        auto& skill=learner.knowledge.primitives().memory.skills;
        check(skill.kernel.routes.W.size()==size_t(10*c.skill_shape.units*c.skill_shape.channels),"lost skill shape in session");
        check(learner.knowledge.composed().memory().skills.kernel.routes.W.size()==skill.kernel.routes.W.size(),"lost skill shape in composed learner");
        if(c.controller_hidden==32)continue;
        auto& p=learner.policy;Observation o{};for(int i=0;i<32;++i)o[i]=float(i%5-2)/7;
        for(size_t i=p.output_weights();i<p.weights.size();++i)p.weights[i]=float(int(i%7)-3)/1000;
        auto g=p.gradient(o,2,0.7f);
        auto loss=[&](){double d=p.action_value(o,2)-0.7;return 0.5*d*d;};
        for(size_t i:{size_t(0),size_t(17),size_t(p.h_bias()),size_t(p.h_bias()+p.hidden-1),size_t(p.output_weights()+2*p.hidden+19),size_t(p.output_bias()+2)}) {
            float before=p.weights[i];p.weights[i]=before+.001f;double plus=loss();p.weights[i]=before-.001f;double minus=loss();p.weights[i]=before;
            check(std::abs((plus-minus)/.002-g[i])<.001,"wide controller gradient mismatch");
        }
        double before=loss();float rate=.002f*(32.f/p.hidden);
        for(size_t i=0;i<g.size();++i)p.weights[i]-=rate*g[i];
        check(loss()<before,"wide update did not reduce loss");
    }
    std::cout<<"Capacity shapes, unused-channel equivalence and wide gradients passed\n";
}
