#include "laminar/reasoning_language.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
using World = EventGrounding::World;
void feed(LanguageReasoner& r, const std::string& s, bool self = false) { for (uint8_t c : s) r.observe(c, self); }
int main() {
    EventGrounding events;
    for (int i = 0; i < 8; ++i) {
        std::string a = "a" + std::to_string(i), b = "b" + std::to_string(i);
        World before{{"old",1,b},{a,1,"other"},{"d",0,"e"}};
        auto after = EventGrounding::update(before, 1, EventGrounding::SET_OBJECT, a, b);
        events.teach_transition(a + " took " + b, before, after, i);
    }
    ReferenceGrounding refs;
    for (int i = 0; i < 8; ++i) {
        std::string a = "a" + std::to_string(i), b = "b" + std::to_string(i);
        ReferenceGrounding::Context context{{1,3,a,b},{0,2,"place","other"}};
        World before{{a,1,b},{"old",1,"token"},{"place",0,"other"}};
        auto object_after = EventGrounding::update(before,1,3,std::string("new"),b);
        assert(refs.teach_transition("new took that item",context,before,object_after,events,i));
        auto subject_after = EventGrounding::update(before,1,3,a,std::string("token"));
        assert(refs.teach_transition("that owner took token",context,before,subject_after,events,20+i));
    }
    assert(refs.phrases.size() == 2);
    auto sig = refs.signature();
    ReferenceGrounding::Context context{{1,3,"sam","key"},{0,2,"room","hall"}};
    auto r = refs.resolve("what is item count of that owner",context);
    assert(r.resolved && r.found && r.text == "what is item count of sam ");
    assert(refs.resolve("lena took that item",context).text == "lena took key ");
    context.push_back({1,3,"sam","book"});
    r = refs.resolve("lena took that item",context);
    assert(!r.resolved && r.candidates == std::set<std::string>({"book","key"}));
    assert(refs.resolve("that owner took token",context).resolved);
    assert(!refs.resolve("that owner took token",{}).resolved);
    assert(!refs.teach_transition("new took that item",context,{},{{"new",1,"key"}},events,99,true));
    assert(refs.signature() == sig);
    LanguageReasoner reader(0); reader.grounded_events = reader.grounded_references = true;
    reader.events = events; reader.references = refs;
    feed(reader,"<|endofdialogue|>\nUSER: sam took key. sam took book. lena took token.\nASSISTANT:");
    auto facts = reader.facts;
    feed(reader,"\nUSER: lena took that item. What is item count of sam?\nASSISTANT:");
    assert(reader.status == "clarification" && !reader.prepared && reader.facts.size() == facts.size());
    assert(reader.answered==0 && reader.clarification_requests==1);
    assert(!reader.add_verified_answer("key"));
    assert(reader.pending_reference.candidates == std::set<std::string>({"book","key","token"}));
    auto pending = reader.pending_text;
    feed(reader,"\nUSER: sam.\nASSISTANT:");
    assert(reader.status == "clarification" && reader.pending_text == pending);
    assert(reader.answered==0 && reader.clarification_requests==2);
    feed(reader,"key",true); assert(reader.pending_text == pending);
    feed(reader,"\nUSER: key.\nASSISTANT:");
    assert(reader.pending_text.empty() && reader.question == "what is item count of sam ");
    bool transferred = false;
    for (const auto& f : reader.facts) if (reader.names[f.object] == "key") transferred = reader.names[f.subject] == "lena";
    assert(transferred && reader.complete && reader.references.signature() == sig);
    feed(reader,"\n<|endofdialogue|>\nUSER: lena took that item.\nASSISTANT:");
    assert(reader.status == "clarification" && reader.facts.empty() && reader.reference_context.empty());
    assert(reader.references.signature() == sig);
    // Subject ambiguity is different from object ambiguity: two actors may
    // mention the same object, while duplicate mention alone adds no candidate.
    context={{1,3,"sam","key"},{1,3,"lena","key"}};
    assert(refs.resolve("new took that item",context).resolved);
    r=refs.resolve("that owner took token",context);
    assert(!r.resolved && r.candidates==std::set<std::string>({"lena","sam"}));
    // Later factual sentences wait behind an unresolved change and are replayed
    // in order after the user chooses an entity.
    feed(reader,"\n<|endofdialogue|>\nUSER: sam took key. sam took book. lena took token.\nASSISTANT:");
    feed(reader,"\nUSER: lena took that item. sam took token. What is sam?\nASSISTANT:");
    assert(reader.deferred_facts.size()==1 && reader.status=="clarification");
    feed(reader,"\nUSER: book.\nASSISTANT:");
    assert(reader.pending_text.empty() && reader.deferred_facts.empty());
    for(const auto& f:reader.facts) {
        if(reader.names[f.object]=="book") assert(reader.names[f.subject]=="lena");
        if(reader.names[f.object]=="token") assert(reader.names[f.subject]=="sam");
    }
    // Weak evidence may leave selectors equivalent; a new context exposing their
    // disagreement must not silently pick one.
    ReferenceGrounding weak;
    for(int i=0;i<4;++i) {
        std::string a="a"+std::to_string(i), b="b"+std::to_string(i);
        World before{{a,1,b}},after{{"new",1,b}};
        assert(weak.teach_transition("new took that item",{{1,3,a,b}},before,after,events,i));
    }
    assert(weak.resolve("new took that item",{{1,3,"sam","key"}}).resolved);
    assert(!weak.resolve("new took that item",{{1,3,"sam","key"},{0,2,"room","hall"}}).resolved);
    // Current-turn evidence takes priority over the previous factual turn.
    feed(reader,"\n<|endofdialogue|>\nUSER: sam took key. lena took token.\nASSISTANT:");
    feed(reader,"\nUSER: sam took book. lena took that item. What is lena?\nASSISTANT:");
    assert(reader.pending_text.empty() && reader.complete);
    for(const auto& f:reader.facts) if(reader.names[f.object]=="book") assert(reader.names[f.subject]=="lena");
    feed(reader,"\nUSER: sam took key. What is that item?\nASSISTANT:");
    assert(reader.pending_text.empty() && reader.question=="what is key ");
    std::cout << "reference grounding contracts passed\n";
}
