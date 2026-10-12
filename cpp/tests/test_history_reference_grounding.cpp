#include "laminar/history_reference_grounding.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
void input(LanguageReasoner& r,const std::string& s){for(uint8_t c:s)r.observe(c,false);}
LanguageReasoner fixture() {
    LanguageReasoner r(0);r.environment_dialogue=r.grounded_events=r.uncertain_events=r.partial_observation=true;r.memory.unique_enabled=true;
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
    HistoryReferenceGrounding learned;
    for(int scene=0;scene<4;++scene) {
        auto r=fixture();RevisableGrounding j;assert(j.start(r));
        report(r,j,90,"ben has the key.");report(r,j,2,"cy has the key.");report(r,j,70,"ada has the key.");
        auto records=HistoryReferenceGrounding::records(j,r);
        assert(learned.teach(r,j,records,"the earliest report about key","Who has the key now?","ben",scene,0));
        auto signature=learned.signature();
        assert(!learned.teach(r,j,records,"the earliest report about key","Who has the key now?","cy",scene,0));
        assert(learned.status=="conflicting_feedback_id"&&learned.signature()==signature);
        assert(learned.teach(r,j,records,"the earliest report about key","Who has the key now?","ben",scene,0));
        assert(learned.signature()==signature);
        assert(!learned.teach(r,j,records,"the earliest report about key","Who has the key now?","ben",scene,1,true));
    }
    auto r=fixture();RevisableGrounding j;assert(j.start(r));
    report(r,j,500,"ben has the key.");report(r,j,7,"cy has the key.");report(r,j,80,"ada has the key.");
    auto records=HistoryReferenceGrounding::records(j,r);
    auto resolved=learned.resolve(r,records,"the earliest report about key");assert(resolved.resolved&&resolved.candidates==std::set<uint64_t>{500});
    GroundedHistoryCorrection correction;const auto model=learned.signature();
    assert(correction.begin(r,j,learned,records,"the earliest report about key","cy has the key.","sensor"));
    assert(j.find(500)->text=="cy has the key."&&j.find(7)->text=="cy has the key."&&j.find(80)->text=="ada has the key.");
    assert(learned.signature()==model&&j.history_revision==1);
    // A deliberately weak reference has several retained selector meanings.
    auto weak=learned;auto& frame=weak.frames["that report about $0 "];
    frame.pattern={"that","report","about","$0"};frame.scenes={1,2,3,4};
    frame.candidates={{{0},-1,0},{{0},-1,-1}};
    records=HistoryReferenceGrounding::records(j,r);const auto state=RevisableGrounding::state_stamp(r);
    assert(!correction.begin(r,j,weak,records,"that report about key","ben has the key.","sensor"));
    assert(correction.status=="ambiguous_reference"&&correction.pending&&RevisableGrounding::state_stamp(r)==state);
    auto token=correction.pending->token;
    assert(!correction.choose(r,j,weak,records,token,999));
    assert(!correction.choose(r,j,weak,records,token,500,true));
    assert(correction.pending&&RevisableGrounding::state_stamp(r)==state);
    assert(correction.choose(r,j,weak,records,token,500));
    assert(j.find(500)->text=="ben has the key."&&weak.signature()!=model);
    auto weak_signature=weak.signature();
    records=HistoryReferenceGrounding::records(j,r);
    assert(!correction.begin(r,j,weak,records,"that report about key","cy has the key.","sensor"));
    token=correction.pending->token;report(r,j,900,"ben has the coin.");records=HistoryReferenceGrounding::records(j,r);
    assert(!correction.choose(r,j,weak,records,token,500)&&correction.status=="stale_correction");
    assert(weak.signature()==weak_signature);
    records=HistoryReferenceGrounding::records(j,r);const auto changes=j.changes.size();
    assert(!correction.begin(r,j,learned,records,"the earliest report about key","cy has the key.","other"));
    assert(j.changes.size()==changes&&correction.status=="stale_report_revision");
    assert(!correction.begin(r,j,learned,records,"the earliest report about key","cy has the key.","sensor",true,true));
    assert(j.changes.size()==changes);
    // Withdrawal preserves the record's content for later reference selection.
    assert(j.revise_report(r,500,j.find(500)->revision,"","sensor",false));
    assert(j.find(500)->text=="ben has the key."&&!j.find(500)->active);
    records=HistoryReferenceGrounding::records(j,r);assert(records.front().fields.at(0).count("key"));
    HistoryReferenceGrounding limited;limited.candidate_budget=0;
    assert(!limited.teach(r,j,records,"some report about key","Who has the key now?","ada",9,0)&&limited.status=="candidate_budget");
    limited.candidate_budget=4096;limited.evaluation_budget=0;
    assert(!limited.teach(r,j,records,"some report about key","Who has the key now?","ada",9,0)&&limited.status=="reference_search_budget");
    auto changed=r;changed.forms["unseen $0 "]=5;
    assert(!learned.teach(changed,j,records,"the earliest report about key","Who has the key now?","ada",20,0)&&learned.status=="stale_history");
    std::cout<<"historical answer grounding, ordered references, clarification, source checks and stale-context contracts passed\n";
}
