#include "laminar/relational_control.h"
#include <iostream>
using namespace laminar::relational;
using namespace laminar::reasoning;
void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
int main() {
    PrimitiveStore store;
    PrimitiveReceipt r;r.id="r";r.cue=2;r.subject={0,0};r.answer={2,999};
    auto invalid=r;invalid.target={1,2};check(!store.teach(invalid),"invalid Boolean accepted");
    invalid=r;invalid.generated=true;check(!store.teach(invalid),"generated lesson accepted");
    invalid=r;invalid.source="model";check(!store.teach(invalid),"unverified primitive label accepted");
    check(store.teach(r)&&!store.teach(r),"duplicate primitive evidence accepted");
    check(!store.revise("r",0,{2,0})&&!store.revise("r",1,{1,2}),"invalid revision accepted");
    store.rebuild(1);check(!store.memory.available_skills[2]&&!store.acquisition()[2].acquired,"unsolved primitive marked available");
    check(store.revise("r",1,{2,0},false)&&store.dirty(),"withdrawal failed");store.rebuild(1);
    check(!store.memory.available_skills[2],"empty evidence acquired a skill");
    RelationalSession s;
    LanguageReceipt l;l.id="computed";l.text="vex a with b via left and right";l.names={"a","b"};l.relations={"left","right"};l.computational=true;
    l.scene={{{"left",{"a","x"}},{"right",{"b","x"}}},true};l.answer=true;
    ControlledRelationalSession control;
    check(control.knowledge.teach_language(l),"initial controller language rejected");
    check(control.begin("early_finish",l.text,l.names,{l.scene,0},{},false),"early query begin rejected");
    auto control_weights=control.policy.weights;
    std::fill(control.policy.weights.begin(),control.policy.weights.end(),0);control.policy.weights[control.policy.output_bias()+4]=1;
    check(control.choose().reply==Answer::Unknown,"early answer computed without reason");control.policy.weights=control_weights;
    check(control.finish(false,true,8),"early terminal feedback rejected");
    check(control.knowledge.language_receipts().at(control.credits().back().evidence).computational,"feedback before first rebuild invented an ordinary competing frame");
    auto bad=l;bad.names.insert("left");check(!s.teach_language(bad)&&s.version()==0,"overlapping mention types committed");
    bad=l;bad.relations={"left"};check(!s.teach_language(bad)&&s.version()==0,"missing relation slot committed");
    bad=l;bad.computational=false;check(!s.teach_language(bad)&&s.version()==0,"untyped relation declarations committed");
    check(s.teach_language(l),"typed language rejected");s.rebuild();
    check(s.answer(l.text,l.scene)==Answer::Unknown,"unacquired primitive answered");
    auto q=s.computation_input(l.text,l.scene);check(q&&q->facts.size()==2,"typed slots not translated");
    check(!s.computation_input("vex a with b via missing and right",l.scene),"unknown relation silently interpreted empty");
    RelationalSession grounded;grounded.enable_observed_relation_types();
    check(grounded.teach_language(l),"grounded relation fixture rejected");grounded.rebuild();
    auto renamed=l.scene;renamed.facts[0].predicate="new_left";renamed.facts[1].predicate="new_right";
    auto grounded_input=grounded.computation_input("vex a with b via new_left and new_right",renamed);
    check(grounded_input&&grounded_input->facts.size()==2,"observed binary predicates did not ground new relation names");
    check(!s.computation_input("vex a with b via new_left and new_right",renamed),"default relation semantics changed");
    check(!grounded.computation_input("vex a with b via missing and new_right",renamed),"absent unknown relation became empty");
    renamed.facts[0].args.push_back("c");
    check(!grounded.computation_input("vex a with b via new_left and new_right",renamed),"ternary predicate grounded a binary slot");
    bool late=false;try{grounded.enable_observed_relation_types();}catch(const std::logic_error&){late=true;}
    check(late,"late relation typing change accepted");
    auto partial=l.scene;partial.complete=false;check(!s.computation_input(l.text,partial),"incomplete count scope exposed");
    auto ordinary=l;ordinary.id="ordinary";ordinary.computational=false;ordinary.relations.clear();
    check(s.teach_language(ordinary),"ordinary ambiguous frame rejected");s.rebuild();
    check(!s.computation_input(l.text,l.scene),"colliding parsers silently chose a frame");
    check(s.revise_language("ordinary",1,true,false),"language withdrawal failed");s.rebuild();
    check(s.computation_input(l.text,l.scene).has_value(),"parser not restored after withdrawal");
    auto p=s.issue(l.text,l.scene,{{"motor",{"a","x"}}},true);check(p.token,"exploration issue failed");
    auto after=l.scene;after.facts.push_back({"moved",{"a","x"}});after=sorted(after);
    check(s.feedback(p.token,{after},{true},true),"computational execution evidence rejected");
    auto id="execution"+std::to_string(p.token);
    auto corrected_before=l.scene,corrected_after=after;
    corrected_before.facts.push_back({"context",{"vex","x"}});corrected_after.facts.push_back({"context",{"vex","x"}});
    check(s.revise_execution_observations(id,1,corrected_before,{corrected_after},{true},true),"computed observation correction rejected");
    const auto& receipt=s.language_receipts().at(id);
    check(receipt.computational&&receipt.names==l.names&&receipt.relations==l.relations&&receipt.scene==sorted(corrected_after),"correction corrupted typed mentions or terminal world");
    std::cout<<"primitive evidence validation, unavailable skills, typed bindings, coverage and parser ambiguity passed\n";
}
