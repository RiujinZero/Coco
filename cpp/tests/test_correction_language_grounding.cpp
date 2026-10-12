#include "laminar/correction_language_grounding.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
void input(LanguageReasoner& r,const std::string& s){for(uint8_t c:s)r.observe(c,false);}
LanguageReasoner fixture() {
    LanguageReasoner r(0);r.environment_dialogue=r.grounded_requests=r.grounded_events=r.uncertain_events=r.partial_observation=true;r.memory.unique_enabled=true;
    r.forms={{"who has the $0 now ",0}};r.memory.procedures={{0,{17,9}}};
    for(int i=0;i<8;++i) {
        auto a="actor"+std::to_string(i),o="item"+std::to_string(i);
        EventGrounding::World before{{o,1,"old"},{"spare",1,a}};
        assert(r.events.teach_transition(a+" has the "+o+".",before,EventGrounding::update(before,1,2,o,a),i));
    }
    input(r,"\n<|endofdialogue|>\nCONTEXT: ada has the key.\nCONTEXT: ben has the coin.\nCONTEXT: cy has the pen.\n");return r;
}
void report(LanguageReasoner& r,RevisableGrounding& j,uint64_t id,const std::string& text) {
    input(r,"\nENVIRONMENT: "+text+"\n");assert(j.append(r,id,"sensor"));
}
int main() {
    HistoryReferenceGrounding refs;
    for(int scene=0;scene<4;++scene) {
        auto r=fixture();RevisableGrounding j;assert(j.start(r));
        report(r,j,90,"ben has the key.");report(r,j,2,"cy has the key.");report(r,j,70,"ada has the key.");
        auto records=HistoryReferenceGrounding::records(j,r);
        assert(refs.teach(r,j,records,"the earliest report about key","Who has the key now?","ben",scene,0));
        // Single-match teaching intentionally leaves ordinal reference ambiguous.
        auto s=fixture();RevisableGrounding single;assert(single.start(s));report(s,single,81,"ben has the key.");
        assert(refs.teach(s,single,HistoryReferenceGrounding::records(single,s),"that report about key","Who has the key now?","ben",scene,1));
    }
    CorrectionLanguageGrounding language;
    const std::vector<std::pair<std::string,std::string>> commands={
        {"adjust the earliest report about key to say cy has the key","cy"},
        {"drop the earliest report about key","ada"},
        {"preserve the earliest report about key","ben"},
        {"use cy has the key instead of the earliest report about key","cy"}};
    for(int scene=0;scene<4;++scene) {
        auto r=fixture();RevisableGrounding j;assert(j.start(r));report(r,j,81,"ben has the key.");
        auto records=HistoryReferenceGrounding::records(j,r);const auto state=RevisableGrounding::state_stamp(r);auto count=j.changes.size();
        for(size_t i=0;i<commands.size();++i) {
            const auto& [text,answer]=commands[i];std::vector<CorrectionLanguageGrounding::QA> qa{{"Who has the key now?",answer}};
            assert(language.teach(r,j,refs,records,text,qa,scene,i));
            assert(RevisableGrounding::state_stamp(r)==state&&j.changes.size()==count&&j.history_revision==0);
            const auto signature=language.signature();assert(language.teach(r,j,refs,records,text,qa,scene,i)&&language.signature()==signature);
            assert(!language.teach(r,j,refs,records,text,qa,scene,99,true)&&language.signature()==signature);
            qa[0].second=answer=="ada"?"ben":"ada";
            assert(!language.teach(r,j,refs,records,text,qa,scene,i)&&language.status=="conflicting_feedback_id"&&language.signature()==signature);
        }
    }
    assert(language.frames.size()==4);
    for(const auto& [frame,meaning]:language.frames)assert(meaning.candidates.size()==1&&meaning.scenes.size()==4);
    auto r=fixture();RevisableGrounding j;assert(j.start(r));report(r,j,500,"cy has the coin.");report(r,j,3,"ada has the coin.");
    auto records=HistoryReferenceGrounding::records(j,r);GroundedCorrectionDialogue dialogue;
    input(r,"\nUSER: adjust the earliest report about coin to say ben has the coin\nASSISTANT:");
    assert(dialogue.begin(r,j,refs,language,records,r.request,"sensor"));
    assert(j.find(500)->text=="ben has the coin."&&j.find(3)->text=="ada has the coin.");
    assert(HistoryReferenceGrounding::consequence(r,"Who has the coin now?")==RevisableGrounding::target(r,"ada",0));
    records=HistoryReferenceGrounding::records(j,r);const auto state=RevisableGrounding::state_stamp(r);auto count=j.changes.size();
    assert(!dialogue.begin(r,j,refs,language,records,"drop the earliest report about coin","other"));
    assert(j.changes.size()==count&&RevisableGrounding::state_stamp(r)==state);
    assert(!dialogue.begin(r,j,refs,language,records,"adjust the earliest report about coin to say stranger has the coin","sensor"));
    assert(dialogue.status=="unknown_correction"&&j.changes.size()==count);
    assert(!dialogue.begin(r,j,refs,language,records,"drop the earliest report about coin","sensor",true));
    assert(j.changes.size()==count);
    // The same learned outer construction accepts a different reference frame.
    assert(!dialogue.begin(r,j,refs,language,records,"drop that report about coin","sensor"));
    assert(dialogue.correction.pending);auto token=dialogue.correction.pending->token;
    assert(!dialogue.choose(r,j,refs,language,records,token,999)&&dialogue.correction.pending);
    auto changed=language;changed.frames.begin()->second.scenes.insert(99);
    assert(!dialogue.choose(r,j,refs,changed,records,token,500)&&dialogue.status=="stale_correction_language");
    assert(!dialogue.correction.pending&&j.changes.size()==count);
    assert(!dialogue.begin(r,j,refs,language,records,"drop that report about coin","sensor"));
    token=dialogue.correction.pending->token;assert(dialogue.choose(r,j,refs,language,records,token,500));
    assert(!j.find(500)->active&&j.find(3)->active);
    // Identical outcomes do not justify choosing between keep and withdrawal.
    CorrectionLanguageGrounding weak;
    for(int scene=0;scene<4;++scene) {
        auto s=fixture();RevisableGrounding same;assert(same.start(s));report(s,same,8,"ada has the key.");
        auto rs=HistoryReferenceGrounding::records(same,s);
        assert(weak.teach(s,same,refs,rs,"consider the earliest report about key",{{"Who has the key now?","ada"}},scene,0));
    }
    records=HistoryReferenceGrounding::records(j,r);count=j.changes.size();
    assert(!dialogue.begin(r,j,refs,weak,records,"consider the earliest report about coin","sensor"));
    assert(dialogue.status=="ambiguous_correction_meaning"&&j.changes.size()==count);
    CorrectionLanguageGrounding limited;limited.word_budget=2;
    assert(!limited.teach(r,j,refs,records,commands[0].first,{{"Who has the key now?","ada"}},8,0)&&limited.status=="correction_span_budget");
    limited.word_budget=64;limited.simulation_budget=0;
    auto fresh=fixture();RevisableGrounding fresh_journal;assert(fresh_journal.start(fresh));report(fresh,fresh_journal,82,"ben has the key.");
    auto fresh_records=HistoryReferenceGrounding::records(fresh_journal,fresh);
    assert(!limited.teach(fresh,fresh_journal,refs,fresh_records,"drop the earliest report about key",{{"Who has the key now?","ada"}},8,0)&&limited.status=="correction_simulation_budget");
    assert(limited.frames.empty());
    auto invalid_records=fresh_records;invalid_records[0].id=123456;
    limited.simulation_budget=256;
    assert(!limited.teach(fresh,fresh_journal,refs,invalid_records,"drop the earliest report about key",{{"Who has the key now?","ada"}},8,0)&&limited.status=="unknown_correction_report");
    assert(limited.frames.empty());
    std::cout<<"correction construction learning, span transfer, counterfactual isolation, ambiguity and stale-language contracts passed\n";
}
