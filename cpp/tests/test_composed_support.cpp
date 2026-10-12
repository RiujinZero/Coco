#include "laminar/composed_learner.h"
#include <iostream>
using namespace laminar::reasoning;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main() {
    ComposedLearner learner(1);
    for(int duplicate=0;duplicate<2;++duplicate)for(int copy=0;copy<2;++copy)for(int size=0;size<4;++size) {
        ComposedReceipt r;r.id="row"+std::to_string(duplicate)+std::to_string(copy)+std::to_string(size);
        auto& q=r.question.query;q.cue=17;q.a={0,100};q.b={0,200};q.relation[0]=0;q.relation[1]=1;
        for(int i=0;i<size;++i)for(int j=0;j<=duplicate;++j)q.facts.push_back({100,0,300+i});
        r.question.answer={2,size};check(learner.teach(r),"count outcome rejected");
    }
    bool acquired=learner.rebuild(100000);
    std::cerr<<"support fixture acquired="<<acquired<<" cue="<<learner.report().cue<<" fit="<<learner.report().correct<<'/'<<learner.report().total<<'\n';
    check(acquired&&learner.report().acquired&&learner.report().cue==2,"count acquisition fixture failed");
    const auto learned=learner.memory().skills.kernel.routes.W;
    auto reduced=learner;
    for(int duplicate=0;duplicate<2;++duplicate)for(int size=0;size<4;++size)
        check(reduced.revise("row"+std::to_string(duplicate)+"1"+std::to_string(size),1,{2,size},false),"redundant withdrawal rejected");
    check(reduced.rebuild(1)&&reduced.reused()&&reduced.report().evaluations==0,"redundant withdrawal discarded supported rehearsal");
    check(reduced.memory().skills.kernel.routes.W==learned,"supported withdrawal changed neural primitives");
    // Removing every duplicate-sensitive outcome leaves part of the learned
    // state/action bank unsupported, even though the remaining answers fit.
    auto insufficient=learner;
    for(int copy=0;copy<2;++copy)for(int size=0;size<4;++size)
        check(insufficient.revise("row1"+std::to_string(copy)+std::to_string(size),1,{2,size},false),"critical withdrawal rejected");
    insufficient.rebuild(1);check(!insufficient.reused(),"unsupported state/action targets survived withdrawal");
    auto empty=learner;
    for(const auto& [id,r]:learner.receipts())check(empty.revise(id,1,r.question.answer,false),"full withdrawal rejected");
    check(empty.rebuild(1)&&!empty.memory().available_skills[2]&&empty.memory().procedures.empty(),"fully withdrawn outcome support retained derived skill");
    std::cout<<"Composed support: redundant evidence retained exact weights; missing state coverage and full withdrawal invalidate cache\n";
}
