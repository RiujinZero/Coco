#include "laminar/concept_memory.h"
#include <iostream>
using namespace laminar::concepts;
static void check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
static Receipt lesson(std::string id,std::string name,bool answer,std::vector<Atom> facts,std::vector<std::string> args={"a","b"}) {
    Receipt r; r.id=id; r.name=name; r.answer=answer; r.args=args; r.scene.facts=facts; return r;
}
int main() {
    ConceptMemory m;
    auto positive=lesson("p","link",true,{{"edge",{"a","b"}},{"other",{"a","b"}}});
    check(m.teach(positive),"valid evidence rejected"); m.rebuild();
    check(m.models().at("link").hypotheses.size()==2,"alternatives collapsed");
    Scene only_edge{{{"edge",{"z","y"}}},true};
    check(m.predict("link",only_edge,{"z","y"})==Answer::Unknown,"ambiguous evidence guessed");
    check(m.teach(lesson("n","link",false,{{"other",{"a","b"}}})),"contrast rejected");
    check(m.predict("link",only_edge,{"z","y"})==Answer::Unknown,"dirty model answered"); m.rebuild();
    check(m.models().at("link").hypotheses.size()==1 && m.predict("link",only_edge,{"z","y"})==Answer::True,"contrast failed");
    auto old=m.models(); check(m.predict("link",only_edge,{"y","z"})==Answer::False,"role order lost");
    only_edge.complete=false;
    check(m.predict("link",only_edge,{"y","z"})==Answer::Unknown && m.predict("link",only_edge,{"z","y"})==Answer::True,"partial scene treated as complete");
    auto bad=positive; bad.id="bad"; bad.generated=true; check(!m.teach(bad),"self-generated receipt accepted");
    bad.generated=false; bad.scene.facts={{"edge",{"a","b","c"}}}; check(!m.teach(bad),"arity mismatch accepted");
    bad.scene.facts={{"link",{"a","b"}}}; check(!m.teach(bad),"derived predicate injected");
    check(!m.revise("p",0,false)&&!m.dirty(),"stale edit mutated model");
    check(!m.revise("p",1,false,true,"generated"),"untrusted edit accepted");
    auto copy=m; check(copy.revise("p",1,false),"edit rejected"); copy.rebuild();
    check(m.predict("link",only_edge,{"z","y"})==Answer::True,"clone shares mutable state");
    auto contradictory=positive; contradictory.id="conflict"; contradictory.answer=false;
    check(m.teach(contradictory),"conflicting receipt rejected instead of retained"); m.rebuild();
    check(m.models().at("link").status=="conflicting_evidence" && m.predict("link",only_edge,{"z","y"})==Answer::Unknown,"conflict guessed");
    check(m.revise("conflict",1,true),"correction rejected"); m.rebuild();
    check(m.predict("link",only_edge,{"z","y"})==Answer::True,"replay failed to recover");
    check(m.revise("n",1,false,false),"withdrawal rejected"); m.rebuild();
    check(m.models().at("link").hypotheses.size()==2,"withdrawal did not restore alternatives");
    Limits limit; limit.candidates=1; ConceptMemory bounded(true,limit);
    check(bounded.teach(positive),"bounded teach failed"); bounded.rebuild();
    check(bounded.models().at("link").status=="resource_exhausted" && bounded.models().at("link").hypotheses.empty(),"partial search accepted a prefix");
    Meter budget(100000);
    Clause c{2,{{"r",{0,2}},{"r",{1,2}}}},d{2,{{"r",{1,7}},{"r",{0,7}}}};
    check(key(canonical(c,budget))==key(canonical(d,budget)),"alpha normalization differs");
    Scene shared{{{"r",{"a","x"}},{"r",{"b","x"}}},true},separate{{{"r",{"a","x"}},{"r",{"b","y"}}},true};
    check(proves(c,shared,{"a","b"},budget)&&!proves(c,separate,{"a","b"},budget),"existential join lost identity");
    Clause redundant{3,{{"r",{0,2}},{"s",{1,2}},{"r",{0,3}},{"s",{1,3}}}};
    Clause core{3,{{"r",{0,2}},{"s",{1,2}}}};
    check(minimize(redundant,budget)==canonical(core,budget),"redundant existential witness created false ambiguity");
    Limits short_rules; short_rules.body=2;
    ConceptMemory growing(true,short_rules),flat(false,short_rules);
    auto first=lesson("first","pair",true,{{"r",{"a","z"}},{"s",{"z","b"}}});
    auto chain=lesson("second","chain",true,{{"r",{"a","x"}},{"s",{"x","b"}},{"r",{"b","y"}},{"s",{"y","c"}}},{"a","b","c"});
    for(auto* learner:{&growing,&flat}) {
        check(learner->teach(first)&&learner->teach(chain),"growing evidence rejected");
        for(int i=0;i<4;++i) { auto negative=chain; negative.id="negative"+std::to_string(i); negative.answer=false; negative.scene.facts.erase(negative.scene.facts.begin()+i); check(learner->teach(negative),"chain contrast rejected"); }
        learner->rebuild();
    }
    check(growing.models().at("chain").hypotheses.size()==1 && growing.predict("chain",chain.scene,chain.args)==Answer::True,"new role structure not learned");
    check(flat.predict("chain",chain.scene,chain.args)==Answer::Unknown,"flat control escaped local body bound");
    auto conflict=first; conflict.id="old_wrong"; conflict.answer=false;
    check(growing.teach(conflict),"old evidence conflict rejected"); growing.rebuild();
    check(growing.predict("chain",chain.scene,chain.args)==Answer::Unknown,"invalid dependency survived rebuild");
    check(growing.revise("old_wrong",1,true),"old evidence correction rejected"); growing.rebuild();
    check(growing.predict("chain",chain.scene,chain.args)==Answer::True,"dependency failed to recover");
    std::cout<<"concept alternatives, role binding, partial knowledge, evidence revision, recovery and budget contracts passed\n";
}
