#include "laminar/relational_session.h"
#include <iostream>
using namespace laminar::relational;
namespace concepts=laminar::concepts;
void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
concepts::Receipt concept_lesson(std::string id,std::string name,Scene world,std::vector<std::string> args,bool value) {
    concepts::Receipt r;r.id=id;r.name=name;r.scene=world;r.args=args;r.answer=value;return r;
}
LanguageReceipt language_lesson(std::string id,std::string text,Scene world,bool value) {
    LanguageReceipt r;r.id=id;r.text=text;r.scene=world;r.answer=value;r.names={"a","b","item","other","old","room","z"};return r;
}
Scene add(Scene world,std::string p,std::string a,std::string b) {world.facts.push_back({p,{a,b}});return sorted(world);}
int main() {
    RelationalSession s(7);
    Scene shared{{{"loc",{"a","room"}},{"loc",{"b","room"}}},true};
    auto first=concept_lesson("shared_yes","shared",shared,{"a","b"},true);
    check(s.teach_concept(first),"concept evidence rejected");
    Scene separate{{{"loc",{"a","room"}},{"loc",{"b","old"}}},true};
    check(s.teach_concept(concept_lesson("shared_no","shared",separate,{"a","b"},false)),"concept contrast rejected");
    check(s.teach_language(language_lesson("lang_shared_yes","do a align b",shared,true))&&s.teach_language(language_lesson("lang_shared_no","do a align b",separate,false)),"language evidence rejected");
    s.rebuild();check(s.answer("do a align b",shared)==Answer::True,"first language failed");
    Scene ready=add(add(shared,"holds","a","item"),"needs","b","item");
    std::string request="please b give item from a";
    auto goal=concept_lesson("goal_yes","goal",ready,{"a","b","item"},true);
    check(s.teach_concept(goal)&&s.teach_language(language_lesson("lang_goal_yes",request,ready,true)),"new role evidence rejected");
    for(size_t i=0;i<ready.facts.size();++i) {
        Scene negative=ready;negative.facts.erase(negative.facts.begin()+i);
        check(s.teach_concept(concept_lesson("goal_no"+std::to_string(i),"goal",negative,{"a","b","item"},false)),"goal contrast rejected");
        check(s.teach_language(language_lesson("lang_goal_no"+std::to_string(i),request,negative,false)),"language contrast rejected");
    }
    Scene before=add(add(shared,"holds","a","old"),"holds","other","item");
    TransitionReceipt hold;hold.id="hold";hold.action={"motor_x",{"a","item"}};hold.before=before;hold.after=add(before,"holds","a","item");
    auto need=hold;need.id="need";need.action.schema="motor_y";need.before=add(add(shared,"needs","a","old"),"needs","other","item");need.after=add(need.before,"needs","a","item");
    check(s.teach_transition(hold)&&s.teach_transition(need),"action observations rejected");s.rebuild();
    check(s.answer("do a align b",shared)==Answer::True&&s.answer(request,ready)==Answer::True,"earlier meanings lost");
    std::vector<Action> menu{{"motor_x",{"a","item"}},{"motor_x",{"b","item"}},{"motor_y",{"a","item"}},{"motor_y",{"b","item"}}};
    auto p=s.issue(request,shared,menu);check(p.token&&p.actions.size()==2&&!p.reused,"two-step plan not discovered");
    std::vector<Scene> actual;auto world=shared;
    for(const auto& a:p.actions){world=add(world,a.schema=="motor_x"?"holds":"needs",a.arguments[0],a.arguments[1]);actual.push_back(world);}
    check(!s.feedback(p.token,actual,{true,true},true,"untrusted"),"untrusted execution accepted");
    check(s.feedback(p.token,actual,{true,true},true),"own verified trace rejected");s.rebuild();
    check(s.procedures().size()==1,"successful procedure not acquired");
    auto reused=s.plan(request,shared,menu);check(reused.reused&&reused.actions.size()==2,"procedure not reused");
    auto remainder=s.plan(request,actual.front(),menu);
    check(remainder.reused&&remainder.actions.size()==1&&same_action(remainder.actions.front(),p.actions.back()),"reactive replanning repeated an already completed procedure prefix");
    check(s.plan(request,shared,menu,100,1).actions.empty(),"cached procedure escaped depth bound");
    auto pending=s.issue(request,shared,menu);check(pending.token,"decision missing");
    auto wrong=first;wrong.id="wrong";wrong.answer=false;check(s.teach_concept(wrong),"conflicting evidence rejected");
    const auto receipt_count=s.transition_receipts().size();
    check(!s.feedback(pending.token,actual,{true,true},true)&&s.transition_receipts().size()==receipt_count,"stale credit half-updated model");
    s.rebuild();check(s.answer(request,ready)==Answer::Unknown&&s.procedures().empty(),"dependent model survived unresolved concepts");
    check(s.revise_concept("wrong",1,true),"concept correction rejected");s.rebuild();
    check(s.answer(request,ready)==Answer::True&&s.plan(request,shared,menu).reused,"rebuild failed to restore language/procedure");
    auto bad_language=language_lesson("wrong_language",request,ready,false);
    check(s.teach_language(bad_language),"language conflict rejected");s.rebuild();
    check(s.answer(request,ready)==Answer::Unknown&&s.procedures().empty(),"language conflict did not invalidate procedure");
    check(s.revise_language("wrong_language",1,true),"language correction rejected");s.rebuild();
    auto old_after=s.transition_receipts().at("hold").after;
    check(s.revise_transition("hold",1,hold.before),"transition revision rejected");s.rebuild();
    check(s.procedures().empty()&&s.plan(request,shared,menu).actions.empty(),"contradictory effect reused stale procedure");
    check(s.revise_transition("hold",2,old_after),"effect correction rejected");s.rebuild();
    check(s.plan(request,shared,menu).reused,"effect recovery lost own procedure");
    auto copy=s;check(copy.revise_execution("execution"+std::to_string(p.token),1,false),"execution edit rejected");copy.rebuild();
    check(copy.procedures().empty()&&!s.procedures().empty(),"copy or execution revision leaked");
    auto eid="execution"+std::to_string(p.token);
    auto corrected_before=add(shared,"context","a","z");auto corrected=actual;
    for(auto& state:corrected)state=add(state,"context","a","z");
    auto unchanged=s.version();auto invalid=corrected;invalid.back().facts.push_back({"unary",{"a"}});
    check(!s.revise_execution_observations(eid,1,corrected_before,invalid,{true,true},true)&&s.version()==unchanged,"invalid execution correction partly committed");
    check(!s.revise_transition(eid+"_step0",1,corrected[0]),"derived transition edited without its dependent evidence");
    check(s.revise_execution_observations(eid,1,corrected_before,corrected,{true,true},true),"execution observation correction rejected");
    check(s.transition_receipts().at(eid+"_step1").before==corrected[0]&&s.language_receipts().at(eid).scene==corrected.back(),"execution correction left stale dependent worlds");
    s.rebuild();check(s.plan(request,shared,menu).reused,"corrected observation bundle lost valid procedure");
    const auto version=s.version();auto malformed=language_lesson("malformed",request,ready,true);malformed.generated=true;
    check(!s.teach_language(malformed)&&s.version()==version,"rejected evidence mutated version");
    Scene novel=add(shared,"unseen","a","z");
    auto predicted=s.predict(menu[0],novel);check(predicted&&std::find(predicted->facts.begin(),predicted->facts.end(),Atom{"unseen",{"a","z"}})!=predicted->facts.end(),"unknown observed predicate was lost");
    auto partial=ready;partial.complete=false;check(s.answer(request,partial)==Answer::True&&s.plan(request,partial,menu).actions.empty(),"partial-world guard failed");
    // A negative goal can be achieved by deleting a row/column even when the
    // new tuple does not match the goal. Pruning must account for the full effect.
    RelationalSession negative;
    Scene linked{{{"edge",{"a","b"}}},true},unlinked{{{"edge",{"a","z"}}},true};
    check(negative.teach_concept(concept_lesson("yes","linked",linked,{"a","b"},true))&&negative.teach_concept(concept_lesson("no","linked",unlinked,{"a","b"},false)),"negative-goal concept failed");
    check(negative.teach_language(language_lesson("neg_false","unrel a b",linked,false))&&negative.teach_language(language_lesson("neg_true","unrel a b",unlinked,true)),"negative-goal language failed");
    TransitionReceipt row;row.id="row";row.action={"row",{"a","b"}};
    row.before={{{"edge",{"a","old"}},{"edge",{"other","b"}}},true};row.after={{{"edge",{"a","b"}},{"edge",{"other","b"}}},true};
    auto column=row;column.id="column";column.action.schema="column";
    column.before={{{"edge",{"old","b"}},{"edge",{"a","other"}}},true};column.after={{{"edge",{"a","b"}},{"edge",{"a","other"}}},true};
    check(negative.teach_transition(row)&&negative.teach_transition(column),"replacement effects rejected");negative.rebuild();
    check(negative.plan("unrel a b",linked,{{"row",{"a","z"}}}).actions.size()==1,"row deletion incorrectly pruned");
    check(negative.plan("unrel a b",linked,{{"column",{"z","b"}}}).actions.size()==1,"column deletion incorrectly pruned");
    std::cout<<"joint concept/language/effect/procedure acquisition, revision, stale-credit and clone contracts passed\n";
}
