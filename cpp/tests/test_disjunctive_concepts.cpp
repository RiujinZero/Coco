#include "laminar/concept_memory.h"
#include <iostream>
using namespace laminar::concepts;
static void check(bool okay,const char* why){if(!okay)throw std::runtime_error(why);}
static Receipt receipt(std::string id,std::string name,bool value,Scene scene,std::vector<std::string> args={"a","b"}) {
    Receipt r;r.id=id;r.name=name;r.answer=value;r.scene=std::move(scene);r.args=std::move(args);return r;
}
static Scene edge(int branch,std::string a,std::string b,std::string z) {
    if(branch)return Scene{{{"t",{a,b}}},true};
    return Scene{{{"r",{a,z}},{"s",{z,b}}},true};
}
static Scene chain(int first,int second) {
    auto world=edge(first,"a","b","u"),tail=edge(second,"b","c","v");
    world.facts.insert(world.facts.end(),tail.facts.begin(),tail.facts.end());return world;
}
int main() {
    Limits limits;limits.body=2;limits.clauses=2;limits.expanded_clauses=8;
    ConceptMemory growing(true,limits),flat(false,limits);
    Limits wide_limits;wide_limits.body=8;ConceptMemory wide(false,wide_limits);
    for(auto* learner:{&growing,&flat,&wide}) {
        check(learner->teach(receipt("p0","link",true,edge(0,"a","b","z"))),"path teaching rejected");
        check(learner->teach(receipt("p1","link",true,edge(1,"a","b","z"))),"direct teaching rejected");
        check(learner->teach(receipt("n0","link",false,Scene{{{"r",{"a","z"}},{"s",{"q","b"}}},true})),"split witness rejected");
        learner->rebuild();
    }
    check(growing.models().at("link").hypotheses.size()==1,"union remained ambiguous");
    check(expanded_clauses(growing.models().at("link").hypotheses.front()).size()==2,"union lost a branch");
    check(wide.models().at("link").status=="no_bounded_hypothesis","wider conjunction represented impossible union");
    for(int branch:{0,1}) {
        auto world=edge(branch,"fresh_first","fresh_second","fresh_middle");
        check(growing.predict("link",world,{"fresh_first","fresh_second"})==Answer::True,"renamed branch failed");
        check(growing.predict("link",world,{"fresh_second","fresh_first"})==Answer::False,"role order ignored");
        world.complete=false;
        check(growing.predict("link",world,{"fresh_first","fresh_second"})==Answer::True,"partial proof withheld");
        check(growing.predict("link",world,{"fresh_second","fresh_first"})==Answer::Unknown,"partial absence guessed");
    }
    int id=0;
    for(auto* learner:{&growing,&flat}) {
        for(int first:{0,1})for(int second:{0,1}) {
            auto world=chain(first,second);
            check(learner->teach(receipt("chain_p"+std::to_string(id++),"relay",true,world,{"a","b","c"})),"composition teaching rejected");
            for(size_t i=0;i<world.facts.size();++i){auto missing=world;missing.facts.erase(missing.facts.begin()+i);check(learner->teach(receipt("chain_n"+std::to_string(id++),"relay",false,missing,{"a","b","c"})),"composition contrast rejected");}
        }
        learner->rebuild();
    }
    check(growing.models().at("relay").hypotheses.size()==1,"union composition not learned");
    const auto& learned=growing.models().at("relay").hypotheses.front();
    check(local_clauses(learned).size()==1&&learned.local.body.size()==2&&learned.local.body[0].predicate=="link"&&learned.local.body[1].predicate=="link","new relation was not reused");
    check(expanded_clauses(learned).size()==4,"Cartesian expansion lost mixed branches");
    for(int first:{0,1})for(int second:{0,1}) {
        check(growing.predict("relay",chain(first,second),{"a","b","c"})==Answer::True,"reused branch failed");
        check(flat.predict("relay",chain(first,second),{"a","b","c"})==Answer::Unknown,"flat control escaped body bound");
    }
    check(growing.predict("link",edge(0,"later_a","later_b","later_z"),{"later_a","later_b"})==Answer::True,"later learning lost old relation");
    auto conflict=receipt("conflict","link",false,edge(1,"a","b","z"));
    check(growing.teach(conflict),"conflict refused");growing.rebuild();
    check(growing.predict("link",edge(1,"a","b","z"),{"a","b"})==Answer::Unknown,"conflicting union answered");
    check(growing.predict("relay",chain(0,0),{"a","b","c"})==Answer::Unknown,"invalid union dependency survived");
    check(growing.revise("conflict",1,true),"correction rejected");growing.rebuild();
    check(growing.predict("relay",chain(0,0),{"a","b","c"})==Answer::True,"corrected union did not restore dependent concept");
    check(growing.revise("conflict",2,true,false)&&growing.revise("p1",1,true,false),"withdrawal rejected");growing.rebuild();
    check(expanded_clauses(growing.models().at("link").hypotheses.front()).size()==1,"withdrawn branch stayed learned without support");
    check(growing.predict("link",edge(1,"a","b","z"),{"a","b"})==Answer::False,"withdrawn branch still predicts true");
    check(growing.revise("p1",2,true,true),"restoration rejected");growing.rebuild();
    check(growing.predict("relay",chain(1,0),{"a","b","c"})==Answer::True,"restored support did not recover dependent concept");
    ConceptMemory ambiguous(true,limits);
    check(ambiguous.teach(receipt("a0","choice",true,Scene{{{"r",{"a","b"}}},true})),"ambiguity first lesson rejected");
    check(ambiguous.teach(receipt("a1","choice",true,Scene{{{"s",{"a","b"}},{"t",{"a","b"}}},true})),"ambiguity second lesson rejected");ambiguous.rebuild();
    check(ambiguous.models().at("choice").hypotheses.size()==2,"different rule unions collapsed");
    Scene only_s{{{"s",{"x","y"}}},true};
    check(ambiguous.predict("choice",only_s,{"x","y"})==Answer::Unknown,"unresolved disjunct guessed");
    check(ambiguous.teach(receipt("contrast","choice",false,Scene{{{"t",{"a","b"}}},true})),"contrast rejected");ambiguous.rebuild();
    check(ambiguous.predict("choice",only_s,{"x","y"})==Answer::True,"contrast failed to disambiguate union");
    check(ambiguous.revise("contrast",1,false,false),"contrast withdrawal rejected");ambiguous.rebuild();
    check(ambiguous.predict("choice",only_s,{"x","y"})==Answer::Unknown,"withdrawn contrast failed to restore alternatives");
    limits.candidates=1;ConceptMemory bounded(true,limits);
    check(bounded.teach(receipt("bounded","choice",true,Scene{{{"r",{"a","b"}},{"s",{"a","b"}}},true})),"bounded lesson rejected");bounded.rebuild();
    check(bounded.models().at("choice").status=="resource_exhausted"&&bounded.models().at("choice").hypotheses.empty(),"exhausted union accepted search prefix");
    Meter meter(10000);Formula redundant{{2,{{"r",{0,1}}}},{2,{{"r",{0,1}},{"s",{0,1}}}}};
    check(normalize_formula(redundant,meter).size()==1,"contained union branch not removed");
    std::cout<<"disjunctive growth, impossible-conjunction control, reuse, alternatives, partial observations, revision and bounds passed\n";
}
