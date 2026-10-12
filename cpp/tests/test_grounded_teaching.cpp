#include "laminar/grounded_session.h"
#include <iostream>
using namespace laminar::reasoning;
void check(bool good,const std::string& message){if(!good)throw std::runtime_error(message);}
void feed(GroundedSession& s,const std::string& text){for(uint8_t c:text)s.observe(c,false);}
std::shared_ptr<GroundedSession> fixture() {
    auto r=std::make_shared<LanguageReasoner>(0);
    r->environment_dialogue=r->grounded_requests=r->grounded_events=r->uncertain_events=r->partial_observation=true;
    r->memory.unique_enabled=true;r->world_budget=512;
    r->forms={{"who has the $0 now ",0}};r->memory.procedures={{0,{17,9}}};
    for(int i=0;i<8;++i) {
        auto a="actor"+std::to_string(i),o="item"+std::to_string(i);
        EventGrounding::World before{{o,1,"old"},{"spare",1,a}};
        check(r->events.teach_transition(a+" has the "+o+".",before,EventGrounding::update(before,1,2,o,a),i),"fixture event");
    }
    auto s=std::make_shared<GroundedSession>(r);s->teachers={"teacher","other_teacher"};return s;
}
void start(GroundedSession& s) {feed(s,"\n<|endofdialogue|>\nCONTEXT: ada has the key.\nCONTEXT: ben has the coin.\nCONTEXT: cy has the pen.\n");}
std::string ask(GroundedSession& s) {feed(s,"\nUSER: Who has the key now?\nASSISTANT:");return LanguageReasoner::trim(s.reader->reply);}
int main() {
    auto s=fixture();const std::string key="the $0 was logged beside $1 ";
    for(int episode=0;episode<6;++episode) {
        start(*s);feed(*s,"ENVIRONMENT: The key was logged beside cy.\n");
        auto anchor=s->feedback_anchor(1);check(bool(anchor),"anchor missing");
        auto changes=s->journal.changes.size();auto model=s->reader->observation_language.signature();
        check(!s->teach(*anchor,0,"Who has the key now?","cy","stranger"),"unverified source accepted");
        check(!s->teach(*anchor,0,"Who has the key now?","cy","teacher",GroundedSession::ADD_FEEDBACK,0,true),"generated teacher accepted");
        check(changes==s->journal.changes.size()&&model==s->reader->observation_language.signature(),"rejected feedback changed state");
        check(s->teach(*anchor,0,"Who has the key now?","cy","teacher"),"valid teacher rejected: "+s->teaching_status);
        check(s->journal.consistent&&ask(*s)=="cy","taught answer");
        check(s->reader->observation_language.frames.at(key).scenes.size()==size_t(episode+1),"support did not cross episode boundary");
        check(s->reader->observation_language.alternatives("The key was logged beside cy.").matched==(episode>=3),"ordinary admission ignored support threshold");
        auto current=*s->feedback_anchor(1);auto accepted=s->feedback_accepted;
        check(s->teach(current,0,"Who has the key now?","cy","teacher"),"duplicate rejected");
        check(s->feedback_accepted==accepted,"duplicate inflated support");
        check(!s->teach(current,0,"Who has the key now?","ben","teacher"),"conflicting id accepted");
        // A clone can receive teaching without modifying the live history/model.
        auto clone=s->clone();auto original=s->reader->observation_language.signature();
        check(clone->teach(*clone->feedback_anchor(1),1,"Who has the coin now?","ben","teacher"),"clone teacher");
        check(s->reader->observation_language.signature()==original&&s->journal.find(1)->evidence.size()==1,"clone leaked evidence");
    }
    check(s->teach(*s->feedback_anchor(1),0,"Who has the key now?","unknown","teacher",GroundedSession::WITHDRAW_FEEDBACK),"withdraw current support");
    check(s->reader->observation_language.frames.at(key).scenes.size()==5,"withdrawal lost prior support or retained withdrawn scene");
    check(s->teach(*s->feedback_anchor(1),0,"Who has the key now?","cy","teacher",GroundedSession::REVISE_FEEDBACK,1),"restore current support");
    check(s->reader->observation_language.frames.at(key).scenes.size()==6,"restoration inflated support");
    auto old=*s->feedback_anchor(1);start(*s);feed(*s,"ENVIRONMENT: The key was logged beside cy.\n");
    check(!s->teach(old,2,"Who has the key now?","cy","teacher")&&s->teaching_status=="stale_teaching_anchor","old episode accepted");
    // Current-episode corrections can undo a premature interpretation while
    // later observations remain in their original chronological positions.
    auto revise=fixture();start(*revise);
    feed(*revise,"ENVIRONMENT: The key was logged beside cy.\n");
    check(revise->teach(*revise->feedback_anchor(1),0,"Who has the key now?","ada","teacher"),"initial mistaken answer");
    feed(*revise,"ENVIRONMENT: The coin was logged beside ada.\n");
    check(revise->teach(*revise->feedback_anchor(2),0,"Who has the coin now?","ada","teacher"),"contradiction record");
    check(!revise->journal.consistent&&ask(*revise)=="I do not know.","conflict failed to withhold");
    auto stale=*revise->feedback_anchor(1);
    check(!revise->teach(stale,0,"Who has the key now?","cy","other_teacher",GroundedSession::REVISE_FEEDBACK),"different teacher revised evidence");
    check(revise->teach(stale,0,"Who has the key now?","cy","teacher",GroundedSession::REVISE_FEEDBACK),"correction rejected");
    check(revise->journal.consistent&&ask(*revise)=="cy","correction failed to recover");
    check(!revise->teach(stale,2,"Who has the key now?","cy","teacher"),"historical token remained live");
    feed(*revise,"\nENVIRONMENT: ben has the key.\n");
    check(revise->teach(*revise->feedback_anchor(1),0,"Who has the key now?","unknown","teacher",GroundedSession::WITHDRAW_FEEDBACK,1),"withdraw feedback");
    auto after=ask(*revise);check(after=="ben","old teaching overwrote later event: "+after+" / "+revise->status+" / "+revise->journal.status);
    // A contradiction left unresolved at episode close cannot export its old
    // successful model narrowing into the next episode.
    auto bad=fixture();start(*bad);auto initial=bad->reader->observation_language.signature();
    feed(*bad,"ENVIRONMENT: The key was logged beside cy.\n");
    check(bad->teach(*bad->feedback_anchor(1),0,"Who has the key now?","ada","teacher"),"bad first");
    feed(*bad,"ENVIRONMENT: The coin was logged beside ada.\n");
    check(bad->teach(*bad->feedback_anchor(2),0,"Who has the coin now?","ada","teacher"),"bad second");
    check(!bad->journal.consistent,"expected conflict");start(*bad);
    check(bad->reader->observation_language.signature()==initial&&bad->acquired_frames.empty(),"unresolved evidence exported");
    std::cout<<"verified sources, evidence revisions, episode support, stale anchors, contradiction recovery and clone isolation passed\n";
}
