#include "laminar/revisable_grounding.h"
#include "laminar/online_grounding.h"
#include "laminar/live_observation_grounding.h"
#include <cassert>
#include <iostream>
using namespace laminar::reasoning;
void input(LanguageReasoner& r,const std::string& text){for(uint8_t c:text)r.observe(c,false);}
LanguageReasoner fixture() {
    LanguageReasoner r(0);r.environment_dialogue=r.grounded_events=r.uncertain_events=r.partial_observation=r.grounded_requests=true;
    r.world_budget=4096;r.memory.unique_enabled=true;
    r.forms={{"who has the $0 now ",0},{"where is $0 ",1},{"does $0 have the $1 ",2}};
    r.memory.procedures={{0,{17,9}},{1,{16,9}},{2,{3,11,9}}};
    for(int i=0;i<8;++i) {
        auto a="actor"+std::to_string(i),o="item"+std::to_string(i);
        EventGrounding::World before{{o,1,"old"},{"spare",1,a},{a,0,"oldroom"}};
        assert(r.events.teach_transition(a+" has the "+o+".",before,EventGrounding::update(before,1,2,o,a),i));
        assert(r.events.teach_transition(a+" moved to the room.",before,EventGrounding::update(before,0,2,a,std::string("room")),100+i));
    }
    r.requests.frames["arrange for $0 to be held by $1 "]={{"arrange","for","$0","to","be","held","by","$1"},{{1,0,1,true}},{1,2,3,4},true};
    r.actuators.frames["motor"]={3,{{1,1,2,2}},{1,2,3,4},0};
    input(r,"\n<|endofdialogue|>\nCONTEXT: ada moved to the hall.\nCONTEXT: ben moved to the kitchen.\nCONTEXT: cy moved to the garden.\nCONTEXT: ada has the key.\nCONTEXT: ben has the coin.\n");
    return r;
}
std::string ask(LanguageReasoner r,const std::string& q) {
    input(r,"\nUSER: "+q+"\nASSISTANT:");return LanguageReasoner::trim(r.reply);
}
void report(RevisableGrounding& j,LanguageReasoner& r,uint64_t id,const std::string& text) {
    input(r,"\nENVIRONMENT: "+text+"\n");assert(j.append(r,id,"sensor"));
}
int main() {
    auto r=fixture();RevisableGrounding j;assert(j.start(r));const auto episode=r.episode_revision;
    report(j,r,1,"The key was logged beside cy.");
    assert(j.feedback(r,1,0,"Who has the key now?","ada","teacher"));
    assert(ask(r,"Who has the key now?")=="ada");
    report(j,r,2,"cy moved to the kitchen.");
    report(j,r,3,"The coin was logged beside ada.");
    assert(j.feedback(r,3,0,"Who has the coin now?","ada","teacher"));
    assert(!j.consistent&&!r.complete&&j.status=="conflicting_journal_evidence");
    assert(ask(r,"Where is cy?")=="I do not know.");
    const auto n=j.changes.size();
    assert(!j.revise_feedback(r,1,0,0,"cy","teacher",true,true));
    assert(!j.revise_feedback(r,1,0,0,"cy","impostor"));
    assert(!j.revise_feedback(r,1,0,1,"cy","teacher"));
    assert(j.changes.size()==n);
    assert(j.revise_feedback(r,1,0,0,"cy","teacher")&&j.consistent);
    assert(r.episode_revision==episode&&r.history_revision==1);
    assert(ask(r,"Who has the key now?")=="cy"&&ask(r,"Who has the coin now?")=="ada");
    assert(ask(r,"Where is cy?")=="kitchen");
    auto earlier=j.at(1);assert(earlier&&ask(*earlier,"Who has the key now?")=="cy");
    assert(ask(*earlier,"Where is cy?")=="garden");
    const auto frame="the $0 was logged beside $1 ";
    assert(r.observation_language.frames.at(frame).scenes.size()==2);
    assert(j.feedback(r,3,0,"Who has the coin now?","ada","teacher"));
    assert(j.changes.size()==n+1&&r.observation_language.frames.at(frame).scenes.size()==2);
    // Withdrawal restores uncertainty without deleting the original record.
    assert(j.revise_feedback(r,1,0,1,"","teacher",false)&&j.consistent);
    assert(!j.find(1)->evidence.at(0).active&&j.find(1)->evidence.at(0).revision==2);
    assert(r.observation_language.frames.at(frame).scenes.size()==1);
    assert(j.revise_feedback(r,1,0,2,"cy","teacher")&&j.consistent);
    // A correction changes its original time slot. A later event still wins.
    report(j,r,4,"ben has the key.");
    input(r,"\nUSER: Where is cy?\nASSISTANT:");assert(LanguageReasoner::trim(r.reply)=="kitchen");
    assert(j.revise_report(r,2,0,"cy moved to the hall.","sensor")&&j.consistent);
    assert(r.reply.empty()&&!r.prepared);
    assert(ask(r,"Who has the key now?")=="ben"&&ask(r,"Where is cy?")=="hall");
    assert(j.revise_report(r,2,1,"","sensor",false)&&j.consistent);
    assert(ask(r,"Where is cy?")=="garden"&&ask(r,"Who has the key now?")=="ben");
    // Even an equivalent historical correction invalidates an issued action's
    // credit transaction; model signatures alone would miss this case.
    input(r,"\nUSER: Arrange for key to be held by cy.\n");OnlineGrounding controller(0);controller.learning=false;
    auto decision=controller.choose(r,{{"motor",{"ben","key","cy"}}});assert(decision.index==0);
    assert(j.revise_report(r,4,0,"ben has the key.","sensor"));
    assert(r.awaiting_action_observation);
    assert(!controller.feedback_transition(r,decision.token,true,true,90)&&controller.status=="stale_history_feedback");
    input(r,"\nENVIRONMENT: cy has the key.\n");assert(j.append(r,5,"sensor"));
    assert(ask(r,"Who has the key now?")=="cy");
    // Limits withhold all derived answers, then replay can recover. No partial
    // search is published as the best surviving meaning.
    j.evaluation_budget=0;assert(!j.replay(r)&&!r.complete&&j.status=="replay_evaluation_budget");
    j.evaluation_budget=1048576;assert(j.replay(r)&&j.consistent);
    auto altered=r;altered.observations.closed_rows.clear();assert(!j.replay(altered)&&j.status=="unrecorded_state_change");
    auto stale=r;stale.history_revision++;assert(!j.replay(stale)&&j.status=="stale_journal");
    input(stale,"\n<|endofdialogue|>\n");assert(!j.replay(stale));
    auto prior=r;prior.forms["foreign $0 "]=99;assert(!j.replay(prior)&&j.status=="changed_journal_prior");
    auto weak=fixture();RevisableGrounding retract;assert(retract.start(weak));
    report(retract,weak,7,"The key was logged beside cy.");
    assert(retract.feedback(weak,7,0,"Who has the key now?","cy","teacher"));
    const auto narrowed=weak.possible_worlds.size();
    assert(retract.revise_feedback(weak,7,0,0,"","teacher",false));
    assert(weak.possible_worlds.size()>narrowed&&ask(weak,"Who has the key now?")=="I do not know.");
    assert(weak.observation_language.frames.at(frame).scenes.empty());
    retract.branch_budget=0;assert(!retract.replay(weak)&&retract.status=="replay_branch_budget");
    retract.branch_budget=4096;retract.history_budget=0;
    assert(!retract.replay(weak)&&retract.status=="replay_history_budget");
    retract.history_budget=16384;assert(retract.replay(weak));
    // A current pending language lesson cannot accept feedback after revision.
    LiveEvidenceGrounding pending;assert(pending.begin(r,"The key became hidden.",500));
    input(r,"\nENVIRONMENT: The key became hidden.\n");++r.history_revision;
    assert(!pending.feedback(r,"Who has the key now?","unknown",0)&&pending.status=="stale_observation_feedback");
    auto closed=fixture();closed.partial_observation=false;LiveObservationGrounding old_pending;
    assert(old_pending.begin(closed,"The key was logged beside cy.",501));
    input(closed,"\nENVIRONMENT: The key was logged beside cy.\n");++closed.history_revision;
    assert(!old_pending.feedback(closed,"Who has the key now?","cy",0)&&old_pending.status=="stale_observation_feedback");
    std::cout<<"identified evidence revisions, history replay, support counts, budgets and stale-credit contracts passed\n";
}
