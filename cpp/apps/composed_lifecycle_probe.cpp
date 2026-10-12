#include "laminar/composed_learner.h"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace laminar::reasoning;
void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
int main(int argc,char** argv)try {
    check(argc==4,"usage: composed_lifecycle_probe seed raw_receipts output_prefix");
    ComposedLearner learner(std::stoul(argv[1]));std::ifstream input(argv[2]);check(bool(input),"missing raw receipts");
    std::string op;std::vector<ComposedReceipt> early,later;
    while(input>>op) {
        if(op=="P") {
            PrimitiveReceipt r;size_t n;input>>r.id>>r.cue>>r.relation>>r.subject.type>>r.subject.id>>r.target.type>>r.target.id>>r.answer.type>>r.answer.id>>n;
            check(n<=512,"too many prior facts");r.facts.resize(n);for(auto& f:r.facts)input>>f.subject>>f.relation>>f.object;
            check(learner.teach_primitive(r),"prior receipt rejected");
        } else {
            check(op=="Q"||op=="L","unknown receipt");ComposedReceipt r;auto& e=r.question;auto& q=e.query;size_t n;
            input>>r.id>>q.cue>>q.a.type>>q.a.id>>q.b.type>>q.b.id>>q.relation[0]>>q.relation[1]>>e.answer.type>>e.answer.id>>n;
            check(n<=512,"too many composed facts");q.facts.resize(n);for(auto& f:q.facts)input>>f.subject>>f.relation>>f.object;
            (op=="Q"?early:later).push_back(r);
        }
        check(bool(input),"truncated receipt");
    }
    check(!early.empty()&&!later.empty(),"missing lifecycle curriculum");
    auto weights=[&](const std::string& stage) {
        std::ofstream f(std::string(argv[3])+"."+stage+".weights.txt");f<<std::setprecision(9);
        for(float value:learner.memory().skills.kernel.routes.W)f<<value<<'\n';
    };
    for(const auto& r:early)check(learner.teach(r),"composed receipt rejected");
    check(learner.answer(early.front().question.query)==Value{},"dirty evidence served a cached answer");
    check(learner.rebuild(20000)&&learner.report().acquired,"whole-outcome acquisition failed");weights("acquired");
    auto learned=learner.memory().skills.kernel.routes.W;
    auto verify=[&](){for(const auto& [id,r]:learner.receipts())if(r.active)check(learner.answer(r.question.query)==r.question.answer,"current evidence no longer fits");};verify();
    std::cout<<"acquired cue="<<learner.report().cue<<" examples="<<learner.report().total<<" first="<<learner.report().first_solution<<'\n'<<std::flush;
    auto version=learner.version();auto invalid=early.front();invalid.id="untrusted";invalid.source="model";
    check(!learner.teach(invalid)&&learner.version()==version,"untrusted outcome changed evidence");
    invalid.source="verified_teacher";invalid.generated=true;check(!learner.teach(invalid),"generated output became teaching");
    invalid.generated=false;invalid.complete=false;check(!learner.teach(invalid),"partial world became complete teaching");
    check(!learner.revise(early.front().id,0,{1,0})&&learner.version()==version,"stale revision changed evidence");
    for(const auto& r:later)check(learner.teach(r),"later composition rejected");
    check(learner.rebuild(1)&&learner.reused()&&learner.memory().skills.kernel.routes.W==learned,"new procedure required hidden neural relearning");verify();weights("extended");
    auto conflict=early.front();conflict.id="conflict";conflict.question.answer.id=1-conflict.question.answer.id;
    check(learner.teach(conflict)&&!learner.rebuild(1)&&learner.contradictory()&&!learner.rebuild(1),"contradictory final answers became a proof");
    check(learner.answer(early.front().question.query)==Value{}&&!learner.memory().available_skills[2],"contradiction left unsupported computation available");weights("contradiction");
    check(learner.revise("conflict",1,conflict.question.answer,false)&&learner.rebuild(1)&&learner.reused(),"withdrawal did not restore current proof");verify();weights("restored");
    check(learner.memory().skills.kernel.routes.W==learned,"restoration changed the retained primitive policy");
    auto evidence=learner.receipts();
    for(const auto& [id,r]:evidence)if(r.active)check(learner.revise(id,r.version,r.question.answer,false),"full withdrawal rejected");
    check(learner.rebuild(1)&&!learner.memory().available_skills[2]&&learner.memory().procedures.empty(),"unsupported derived skill survived full withdrawal");weights("withdrawn");
    for(const auto& [id,r]:evidence)if(r.active)check(learner.revise(id,r.version+1,r.question.answer,true),"full restoration rejected");
    check(learner.rebuild(1)&&learner.reused()&&learner.memory().skills.kernel.routes.W==learned,"restored evidence did not revalidate cached learning");verify();weights("final");
    std::cout<<"new composition, trusted outcomes, dirty withholding, stale revisions, contradiction, withdrawal and restoration contracts passed; version="<<learner.version()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
