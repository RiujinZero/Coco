#include "laminar/relational_session.h"
#include <iostream>
using namespace laminar::reasoning;
using namespace laminar::relational;
static void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
static PrimitiveReceipt primitive(std::string id,int cue,Value subject,Value target,Value answer) {
    PrimitiveReceipt r;r.id=id;r.cue=cue;r.subject=subject;r.target=target;r.answer=answer;return r;
}
static LanguageReceipt language(std::string id,int kind,int size,int other,int copies=1,int depth=0,int offset=0) {
    LanguageReceipt r;r.id=id;r.computational=true;r.relations={"left","right"};r.scene.complete=true;
    auto entity=[&](int n){return "e"+std::to_string(offset+n);};
    std::string a=entity(100),b=kind==1?std::to_string(other):kind==2?entity(other?100+depth:500):entity(200);
    r.names.insert(a);if(kind==1)r.quantities.insert(b);else r.names.insert(b);
    std::string word=kind==0?"loma":kind==1?"navi":"telo";
    r.text=word+" "+a+" beside "+b+" across left and right";
    if(kind==2) {
        for(int i=0;i<depth;++i)r.scene.facts.push_back({"left",{entity(100+i),entity(101+i)}});
        r.answer=bool(other);
    }else {
        for(int n=0;n<size;++n)for(int copy=0;copy<copies;++copy)r.scene.facts.push_back({"left",{a,entity(300+n)}});
        if(kind==0)for(int n=0;n<other;++n)for(int copy=0;copy<copies;++copy)r.scene.facts.push_back({"right",{b,entity(400+n)}});
        r.answer=size==other;
    }
    return r;
}
static void count_evaluation(const RelationalSession& session,int& checked) {
    for(int size=0;size<=5;++size)for(int other=0;other<=5;++other)for(int kind:{0,1}) {
        auto r=language("unused",kind,size,other,3,0,10000+checked*100);
        check(session.answer(r.text,r.scene)==Answer(r.answer),"fresh count language answer failed");++checked;
    }
}
static void primitive_evaluation(const RelationalSession& session) {
    auto model=session.primitives().memory;int checked=0;
    for(int size:{0,1,4,8,16,24,32}) {
        auto follow=primitive("unused",0,{0,9000},{},{0,9000+size});follow.relation=71;
        for(int i=0;i<size;++i)follow.facts.push_back({9000+i,71,9001+i});
        check(verifies_direct_outcome(model.skills,follow),"session follow primitive failed independent length");++checked;
        for(int copies:{1,2,3}) {
            if(size==0&&copies>1)continue;
            auto count=primitive("unused",2,{0,8000},{},{2,size});count.relation=73;
            for(int i=0;i<size;++i)for(int copy=0;copy<copies;++copy)count.facts.push_back({8000,73,8100+i});
            check(verifies_direct_outcome(model.skills,count),"session count primitive failed independent multiplicity");++checked;
        }
    }
    std::cerr<<"session primitive probes="<<checked<<'\n';
}
int main() {
    RelationalSession session(1);session.enable_composed_learning(5000,true,100,true);
    std::vector<PrimitiveReceipt> priors;
    priors.push_back(primitive("follow_empty",0,{0,10},{},{0,10}));
    auto one=primitive("follow_one",0,{0,10},{},{0,11});one.facts={{10,0,11}};priors.push_back(one);
    for(int size=0;size<=2;++size) {
        auto r=primitive("count"+std::to_string(size),2,{0,10},{},{2,size});
        for(int i=0;i<size;++i)r.facts.push_back({10,0,20+i});priors.push_back(r);
    }
    for(int type=0;type<3;++type)for(int same=0;same<2;++same)
        priors.push_back(primitive("equal"+std::to_string(type)+std::to_string(same),1,{type,0},{type,same?0:1},{1,same}));
    for(const auto& r:priors)check(session.teach_primitive(r),"raw primitive prior rejected");
    int serial=0;std::vector<LanguageReceipt> duplicate_rows,follow_rows;
    for(int left=0;left<3;++left)for(int right=0;right<3;++right)for(int copies:{1,2}) {
        auto r=language("row"+std::to_string(serial++),0,left,right,copies);
        if(copies==1)check(session.teach_language(r),"warm count language rejected");else duplicate_rows.push_back(r);
    }
    for(int size=0;size<3;++size)for(int target=0;target<4;++target)for(int copies:{1,2}) {
        auto r=language("row"+std::to_string(serial++),1,size,target,copies);
        if(copies==1)check(session.teach_language(r),"warm numeric language rejected");else duplicate_rows.push_back(r);
    }
    for(int depth:{0,1,2,3,12,20,24})for(int same:{0,1}) {
        auto r=language("row"+std::to_string(serial++),2,0,same,1,depth);
        if(depth<=1)check(session.teach_language(r),"warm follow language rejected");else follow_rows.push_back(r);
    }
    session.rebuild();check(session.primitives().memory.available_skills==std::array<bool,3>{true,true,true},"wrong priors not already available");
    check(session.composed().active_repairs().empty(),"warm language unexpectedly repaired a primitive");
    auto count_error=language("unused",1,1,1,2,0,5000),follow_error=language("unused",2,0,1,1,8,5000);
    check(session.answer(count_error.text,count_error.scene)==Answer::False,"count error not visible through learned language");
    check(session.answer(follow_error.text,follow_error.scene)==Answer::False,"follow error not visible through learned language");
    for(const auto& r:duplicate_rows)check(session.teach_language(r),"composed count counterexample rejected");session.rebuild();
    std::cerr<<"session count repair solved="<<session.composed().solved()<<" cue="<<session.composed().report().cue<<" proposals="<<session.composed().report().evaluations<<'\n';
    check(session.composed().solved()&&session.composed().active_repairs()==std::set<int>{2},"language count repair failed");
    check(session.answer(count_error.text,count_error.scene)==Answer::True&&session.answer(follow_error.text,follow_error.scene)==Answer::False,"first repair changed wrong language family");
    auto first=session.primitives().memory.skills.kernel.routes.W;int count_checked=0;count_evaluation(session,count_checked);
    for(const auto& r:follow_rows)check(session.teach_language(r),"composed follow counterexample rejected");session.rebuild();
    std::cerr<<"session follow repair solved="<<session.composed().solved()<<" cue="<<session.composed().report().cue<<" proposals="<<session.composed().report().evaluations<<'\n';
    check(session.composed().solved()&&session.composed().active_repairs()==std::set<int>({0,2}),"language second repair lost first");
    check(session.answer(follow_error.text,follow_error.scene)==Answer::True,"follow language error not repaired");
    auto combined=session.primitives().memory.skills.kernel.routes.W;count_evaluation(session,count_checked);int follow_checked=0;
    for(int depth:{0,1,4,8,16,24})for(int same:{0,1}) {
        auto r=language("unused",2,0,same,1,depth,20000+follow_checked*100);
        check(session.answer(r.text,r.scene)==Answer(r.answer),"fresh follow language failed");++follow_checked;
    }
    primitive_evaluation(session);
    auto model=session.primitives().memory;for(const auto& r:priors)check(verifies_direct_outcome(model.skills,r),"language repair changed original direct prior");
    for(const auto& r:follow_rows)check(session.revise_language(r.id,1,r.answer,false),"language support withdrawal rejected");session.rebuild();
    check(session.composed().active_repairs()==std::set<int>{2}&&session.primitives().memory.skills.kernel.routes.W==first,"language withdrawal erased independent first repair");
    count_evaluation(session,count_checked);
    for(const auto& r:follow_rows)check(session.revise_language(r.id,2,r.answer,true),"language support restoration rejected");session.rebuild();
    check(session.primitives().memory.skills.kernel.routes.W==combined&&session.composed().active_repairs()==std::set<int>({0,2}),"language restoration lost exact combined policy");
    std::cout<<"successive language-grounded repairs: count checks="<<count_checked<<" follow checks="<<follow_checked<<"; primitive outcomes, priors and lifecycle passed\n";
}
