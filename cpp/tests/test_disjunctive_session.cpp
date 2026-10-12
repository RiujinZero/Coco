#include "../apps/control_probe_support.h"
#include <iostream>
using namespace laminar::relational;
void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
laminar::concepts::Receipt example(const std::string& id,const std::string& head,const Scene& scene,bool answer) {
    laminar::concepts::Receipt r;r.id=id;r.name=head;r.scene=scene;r.args={"a","b"};r.answer=answer;return r;
}
LanguageReceipt language(const std::string& id,const std::string& text,const Scene& scene,bool answer) {
    LanguageReceipt r;r.id=id;r.text=text;r.scene=scene;r.names={"a","b"};r.answer=answer;return r;
}
int main(int argc,char** argv) {
    RelationalSession session(9);laminar::concepts::Limits limits;limits.body=2;limits.clauses=2;
    session.enable_disjunctive_concepts(limits);
    Scene empty{{},true},edge{{{"edge",{"a","b"}}},true};
    for(bool yes:{false,true}) {
        auto id=std::to_string(yes);auto scene=yes?edge:empty;
        check(session.teach_concept(example("old"+id,"linked",scene,yes)),"old concept rejected");
        check(session.teach_language(language("old_language"+id,"join a b",scene,yes)),"old language rejected");
    }
    TransitionReceipt effect;effect.id="edge_effect";effect.action={"edge_motor",{"a","b"}};
    effect.before={{{"edge",{"a","old"}},{"edge",{"other","b"}}},true};effect.after=effect.before;effect.after.facts.push_back({"edge",{"a","b"}});
    check(session.teach_transition(effect),"old effect rejected");session.rebuild();
    auto issued=session.issue("join a b",empty,{{"edge_motor",{"a","b"}}});
    check(issued.actions.size()==1&&session.feedback(issued.token,{edge},{true},true),"old procedure acquisition failed");session.rebuild();
    auto retained=[&]{
        check(session.answer("join a b",edge)==Answer::True&&session.answer("join a b",empty)==Answer::False,"old language forgotten");
        check(session.plan("join a b",empty,{{"edge_motor",{"a","b"}}}).reused,"old procedure forgotten");
    };
    retained();
    for(int mask=0;mask<4;++mask) {
        Scene scene;if(mask&1)scene.facts.push_back({"red",{"a","b"}});if(mask&2)scene.facts.push_back({"blue",{"a","b"}});
        auto id=std::to_string(mask);
        check(session.teach_concept(example("alternative"+id,"permitted",scene,mask!=0)),"new rule evidence rejected");
        check(session.teach_language(language("alternative_language"+id,"permit a b",scene,mask!=0)),"new surface evidence rejected");
    }
    for(const auto& predicate:{"red","blue"}) {
        TransitionReceipt t;t.id=std::string(predicate)+"_effect";t.action={std::string(predicate)+"_motor",{"a","b"}};
        t.before={{{predicate,{"a","old"}},{predicate,{"other","b"}}},true};t.after=t.before;t.after.facts.push_back({predicate,{"a","b"}});
        check(session.teach_transition(t),"new effect rejected");
    }
    session.rebuild();retained();
    const auto& model=session.concepts().models().at("permitted");
    check(model.hypotheses.size()==1&&laminar::concepts::expanded_clauses(model.hypotheses[0]).size()==2,"alternative rules not uniquely acquired");
    const std::vector<Atom> universe{{"red",{"x","z"}},{"blue",{"x","z"}},
        {"red",{"z","x"}},{"blue",{"z","x"}},{"red",{"x","other"}},{"blue",{"other","z"}},{"edge",{"x","z"}}};
    for(int mask=0;mask<128;++mask)for(bool complete:{false,true}) {
        Scene scene{{},complete};for(int bit=0;bit<7;++bit)if(mask&(1<<bit))scene.facts.push_back(universe[bit]);
        Answer expected=mask&3?Answer::True:complete?Answer::False:Answer::Unknown;
        check(session.answer("permit x z",scene)==expected,"new language failed fresh bindings/distractors/partial world");
    }
    // Either branch must remain relevant when it is the only afforded motor.
    for(const auto& predicate:{"red","blue"}) {
        auto plan=session.plan("permit a b",empty,{{std::string(predicate)+"_motor",{"a","b"}}});
        check(plan.actions.size()==1,"action pruning discarded an alternative branch");
    }
    // Contradictory evidence invalidates dependent meanings and stale plans,
    // while previously learned unrelated language and procedures survive.
    auto stale=session.issue("permit a b",empty,{{"red_motor",{"a","b"}}});
    Scene red{{{"red",{"a","b"}}},true};
    check(session.teach_concept(example("contradiction","permitted",red,false)),"contradiction rejected");
    check(!session.feedback(stale.token,{red},{true},true),"stale action plan accepted");session.rebuild();retained();
    check(session.answer("permit a b",red)==Answer::Unknown,"contradiction retained a dependent meaning");
    check(session.revise_concept("contradiction",1,false,false),"contradiction withdrawal failed");session.rebuild();retained();
    check(session.answer("permit a b",red)==Answer::True,"restoration failed");
    for(int mask=0;mask<4;++mask)check(session.revise_concept("alternative"+std::to_string(mask),1,mask!=0,false),"support withdrawal failed");
    session.rebuild();retained();check(session.answer("permit a b",red)==Answer::Unknown,"unsupported union survived withdrawal");
    for(int mask=0;mask<4;++mask)check(session.revise_concept("alternative"+std::to_string(mask),2,mask!=0,true),"support restoration failed");
    session.rebuild();retained();check(session.answer("permit a b",red)==Answer::True,"union did not recover");
    bool rejected=false;try{session.enable_disjunctive_concepts(limits);}catch(const std::logic_error&){rejected=true;}
    check(rejected,"concept class changed after teaching");
    if(argc==2){std::ofstream out(argv[1]);check(bool(out),"cannot write model snapshot");dump(session,"restored",out);}
    std::cout<<"Disjunctive session: 256 fresh complete/partial queries, both action branches, retained old language/procedure, correction, withdrawal and restoration passed\n";
}
