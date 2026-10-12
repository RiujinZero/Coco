#include "laminar/relational_control.h"
#include <iostream>
using namespace laminar::relational;
using namespace laminar::reasoning;
void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
int main() {
    RelationalSession s;s.enable_composed_learning(1);
    LanguageReceipt r;r.id="numeric";r.computational=true;r.text="how a compares 3 via left and right";
    r.names={"a"};r.quantities={"3"};r.relations={"left","right"};r.answer=true;
    r.scene.facts={{"left",{"a","x"}},{"left",{"a","x"}}};
    auto invalid=r;invalid.quantities={"a"};check(!s.teach_language(invalid),"non-numeric quantity accepted");
    invalid=r;invalid.names.insert("3");check(!s.teach_language(invalid),"overlapping quantity/entity accepted");
    check(s.teach_language(r),"typed numeric lesson rejected");s.rebuild();
    check(s.language_receipts().at(r.id).scene.facts.size()==2,"raw teaching duplicates discarded");
    auto q=s.computation_input("how fresh compares 27 via left and right",r.scene);
    check(q&&q->a.type==0&&q->b==Value{2,27},"numeric argument lost its type or value");
    check(!s.computation_input("how fresh compares word via left and right",r.scene),"non-number matched quantity slot");
    check(!s.computation_input("how fresh compares -27 via left and right",r.scene),"negative sign was silently discarded");
    check(!s.computation_input("how fresh compares 2.7 via left and right",r.scene),"fraction matched integer slot");
    auto copied=s;check(copied.revise_language("numeric",1,false),"revision rejected");copied.rebuild();
    check(s.language_receipts().at("numeric").answer&&!copied.language_receipts().at("numeric").answer,"revision leaked into clone");
    check(s.revise_language("numeric",1,true,false),"withdrawal rejected");s.rebuild();
    check(s.composed().receipts().empty()&&s.primitives().memory.procedures.empty(),"withdrawn journal retained compiled evidence");
    ControlledRelationalSession control;control.knowledge.enable_composed_learning(1);
    check(control.knowledge.teach_language(r),"controller lesson rejected");
    check(control.begin("numeric_feedback",r.text,r.names,{r.scene,0},{},false),"episode rejected");
    auto weights=control.policy.weights;std::fill(control.policy.weights.begin(),control.policy.weights.end(),0);
    control.policy.weights[control.policy.output_bias()+4]=1;
    check(control.choose().reply==Answer::Unknown,"early finish invented an answer");control.policy.weights=weights;
    check(control.finish(false,true,1),"numeric terminal feedback rejected");
    auto id=control.credits().back().evidence;const auto& receipt=control.knowledge.language_receipts().at(id);
    check(receipt.computational&&receipt.quantities==r.quantities&&receipt.names==r.names,"terminal feedback changed quantity into entity");
    std::cout<<"typed quantities, raw journal synchronization, withdrawal, clone isolation and controller feedback passed\n";
}
