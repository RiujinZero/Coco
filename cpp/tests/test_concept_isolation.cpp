#include "laminar/relational_session.h"
#include <iostream>
using namespace laminar::relational;
void check(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
laminar::concepts::Receipt concept_row(const std::string& id,const std::string& name,const Scene& world,const std::vector<std::string>& args,bool value) {
    laminar::concepts::Receipt r;r.id=id;r.name=name;r.scene=world;r.args=args;r.answer=value;return r;
}
LanguageReceipt language_row(const std::string& id,const std::string& text,const Scene& world,bool value) {
    LanguageReceipt r;r.id=id;r.text=text;r.scene=world;r.answer=value;r.names={"a","b","c","x","z","old","other"};return r;
}
int main() {
    RelationalSession learner(7);
    Scene empty{{},true},linked{{{"edge",{"a","b"}}},true};
    check(learner.teach_concept(concept_row("old_yes","linked",linked,{"a","b"},true)),"old concept failed");
    check(learner.teach_concept(concept_row("old_no","linked",empty,{"a","b"},false)),"old contrast failed");
    check(learner.teach_language(language_row("old_language_yes","join a b",linked,true))&&learner.teach_language(language_row("old_language_no","join a b",empty,false)),"old language failed");
    TransitionReceipt effect;effect.id="effect";effect.action={"motor",{"a","b"}};
    effect.before={{{"edge",{"a","old"}},{"edge",{"other","b"}}},true};effect.after=effect.before;effect.after.facts.push_back({"edge",{"a","b"}});
    check(learner.teach_transition(effect),"effect failed");learner.rebuild();
    auto issued=learner.issue("join a b",empty,{{"motor",{"a","b"}}});
    check(issued.actions.size()==1&&learner.feedback(issued.token,{linked},{true},true),"own procedure failed");learner.rebuild();
    auto retain=[&] {
        check(learner.ready()&&learner.answer("join a b",linked)==Answer::True&&learner.answer("join a b",empty)==Answer::False,"unrelated uncertain concept disabled old language");
        check(learner.plan("join a b",empty,{{"motor",{"a","b"}}}).reused,"unrelated uncertain concept disabled old procedure");
    };
    retain();
    Scene two{{{"red",{"a","b"}},{"blue",{"a","b"}}},true};
    check(learner.teach_concept(concept_row("new_yes","mystery",two,{"a","b"},true)),"new concept failed");
    check(!learner.ready()&&learner.answer("join a b",linked)==Answer::Unknown,"stale semantics served before rebuild");
    check(learner.teach_language(language_row("new_language_yes","test a b",two,true))&&learner.teach_language(language_row("new_language_no","test a b",empty,false)),"new language failed");
    learner.rebuild();check(learner.concepts().models().at("mystery").hypotheses.size()==2,"fixture is not ambiguous");
    retain();check(learner.answer("test a b",two)==Answer::Unknown,"unresolved concept silently grounded language");
    LanguageReceipt computation=language_row("computation","measure a b via edge and blue",two,true);
    computation.computational=true;computation.relations={"edge","blue"};
    check(learner.teach_language(computation),"computational frame failed");learner.rebuild();retain();
    check(learner.computation_input(computation.text,two).has_value(),"unrelated ambiguity disabled known computation inputs");
    check(!learner.computation_input("measure a b via mystery and blue",Scene{{{"red",{"a","b"}}},true}),"uncertain derived relation supplied complete computation input");
    // A contradictory new concept must also leave independent meanings usable.
    check(learner.teach_concept(concept_row("new_conflict","mystery",two,{"a","b"},false)),"conflict rejected");learner.rebuild();
    check(learner.concepts().models().at("mystery").hypotheses.empty(),"conflict did not withhold new concept");retain();
    check(learner.revise_concept("new_conflict",1,true,false),"conflict withdrawal failed");
    Scene red{{{"red",{"a","b"}}},true};
    check(learner.teach_concept(concept_row("new_contrast","mystery",red,{"a","b"},false)),"disambiguating contrast failed");learner.rebuild();retain();
    check(learner.answer("test a b",two)==Answer::True&&learner.answer("test a b",red)==Answer::False,"new concept not acquired after contrast");
    // Grow a three-role relation and a different surface-role order in this same session.
    Scene triple{{{"edge",{"a","b"}},{"blue",{"b","c"}}},true};
    for(int missing=-1;missing<2;++missing) {
        auto world=triple;if(missing>=0)world.facts.erase(world.facts.begin()+missing);
        auto id=std::to_string(missing+1);bool value=missing<0;
        check(learner.teach_concept(concept_row("triple"+id,"chain",world,{"a","b","c"},value)),"new role structure rejected");
        check(learner.teach_language(language_row("triple_language"+id,"route c from a through b",world,value)),"new surface roles rejected");
    }
    learner.rebuild();retain();
    for(int mask=0;mask<64;++mask) {
        Scene world;std::vector<Atom> possible{{"edge",{"x","z"}},{"blue",{"z","other"}},{"edge",{"z","x"}},{"blue",{"other","z"}},{"edge",{"x","old"}},{"blue",{"old","other"}}};
        for(int bit=0;bit<6;++bit)if(mask&(1<<bit))world.facts.push_back(possible[bit]);
        bool expected=(mask&3)==3;
        check(learner.answer("route other from x through z",world)==(expected?Answer::True:Answer::False),"new roles or distractor transfer failed");
    }
    check(learner.revise_concept("new_contrast",1,false,false),"contrast withdrawal failed");learner.rebuild();retain();
    check(learner.answer("test a b",red)==Answer::Unknown,"withdrawal retained unsupported new meaning");
    check(learner.revise_concept("new_contrast",2,false,true),"contrast restoration failed");learner.rebuild();retain();
    check(learner.answer("test a b",red)==Answer::False,"contrast restoration did not recover meaning");
    // A generic computational-language frame may mention several relations.
    // Withholding one relation must not remove valid lessons for that frame.
    for(bool composed:{false,true}) {
    RelationalSession mixed;if(composed)mixed.enable_composed_learning(1);
    for(int same=0;same<2;++same) {
        laminar::reasoning::PrimitiveReceipt p;p.id="comparison"+std::to_string(same);p.cue=1;
        p.subject={0,0};p.target={0,same?0:1};p.answer={1,same};
        check(mixed.teach_primitive(p),"comparison lesson rejected");
        auto row=language_row("base"+std::to_string(same),same?"compare a a via left and right":"compare a b via left and right",empty,same);
        row.computational=true;row.relations={"left","right"};check(mixed.teach_language(row),"base computation lesson rejected");
    }
    check(mixed.teach_concept(concept_row("mixed_concept","uncertain",linked,{"a","b"},true)),"mixed concept rejected");
    auto dependent=language_row("dependent","compare a a via uncertain and right",linked,true);
    dependent.computational=true;dependent.relations={"uncertain","right"};check(mixed.teach_language(dependent),"dependent computation rejected");
    mixed.rebuild();check(mixed.answer("compare x x via left and right",empty)==Answer::True,"base computation not learned");
    check(mixed.teach_concept(concept_row("mixed_conflict","uncertain",linked,{"a","b"},false)),"mixed conflict rejected");mixed.rebuild();
    check(mixed.answer("compare x x via uncertain and right",empty)==Answer::Unknown,"dependent computation survived conflict");
    check(mixed.answer("compare x x via left and right",empty)==Answer::True&&mixed.answer("compare x z via left and right",empty)==Answer::False,"unavailable relation discarded independent lessons of the same language frame");
    if(composed)check(mixed.composed().receipts().size()==2&&!mixed.composed().receipts().count("dependent")&&mixed.composed().receipts().at("base0").question.answer==laminar::reasoning::Value{1,0},"filtered composed evidence lost its original identity or label");
    check(mixed.revise_concept("mixed_conflict",1,false,false),"mixed conflict withdrawal rejected");mixed.rebuild();
    check(mixed.answer("compare x x via uncertain and right",empty)==Answer::True,"restored relation did not recover the shared frame");
    if(composed)check(mixed.composed().receipts().count("dependent")&&mixed.composed().receipts().at("dependent").question.answer==laminar::reasoning::Value{1,1},"restoration did not restore original composed receipt");
    }
    std::cout<<"Concept isolation: ambiguity/conflict/withdrawal, retained language/actions/procedures, computation boundaries and 64 fresh role queries passed\n";
}
