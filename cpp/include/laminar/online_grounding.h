#pragma once
// Feedback-bound interaction for an already grounded observation reader.
// Exploration is an explicit supplied policy; goal/effect meanings are learned.
#include "reasoning_language.h"
#include <optional>
#include <random>

namespace laminar::reasoning {
struct OnlineGrounding {
    using Action=GroundedActuator::Action;
    struct Decision {uint64_t token=0;int index=-1;std::string status;size_t predictions=0;};
    struct Pending {
        uint64_t token,episode,request_revision,observation_revision,history_revision;
        std::string request,event_model;
        EventGrounding event_snapshot;
        ObservationLanguage evidence_snapshot;
        Action action;
        RequestGrounding::Snapshot before;
        std::set<std::string> names;
        std::set<int> relations;
    };
    std::optional<Pending> pending;
    uint64_t next_token=1;
    bool learning=true;
    bool information_seeking=true;
    std::string status;
    std::mt19937 random;
    explicit OnlineGrounding(uint32_t seed=0):random(seed){}
    bool feedback_request(LanguageReasoner& reader,uint64_t request_revision,bool satisfied,uint64_t scene,bool self_generated=false) {
        if(!learning){status="learning_disabled";return false;}
        if(self_generated){status="self_generated_feedback";return false;}
        if(pending){status="awaiting_transition_feedback";return false;}
        if(request_revision!=reader.request_revision){status="stale_request_feedback";return false;}
        bool accepted=reader.add_verified_request_feedback(satisfied,scene);
        status=accepted?reader.requests.status:"unavailable_request_feedback";return accepted;
    }
    Decision choose(LanguageReasoner& reader,const std::vector<Action>& available,size_t budget=8192) {
        if(pending) {
            if(pending->episode==reader.episode_revision&&pending->request_revision==reader.request_revision)
                return {pending->token,-1,"awaiting_transition_feedback",0};
            pending.reset(); // a new request/episode cancels the old transaction
        }
        if(!budget)return {0,-1,"action_search_budget",0};
        auto plan=reader.choose_action(available,budget);
        Decision result{0,plan.index,plan.status,plan.predictions};
        if(!learning&&result.index<0)return result;
        if(plan.status=="already_satisfied"||plan.status=="missing_observation"||plan.status=="action_search_budget"||
           plan.status=="conflicting_request_evidence"||!reader.grounded_requests||reader.request.empty())return result;
        EventGrounding::World world;auto snapshot=reader.request_snapshot();
        if(learning&&!GroundedActuator::complete_world(snapshot,world))return {0,-1,"missing_observation",result.predictions};
        if(!snapshot.valid||snapshot.worlds.empty())return {0,-1,"missing_observation",result.predictions};
        if(available.empty())return {0,-1,"no_available_action",result.predictions};
        const int exploit_index=result.index;
        const auto hypotheses=reader.requests.meaning(reader.request,false);
        if(learning&&(result.index<0||(information_seeking&&hypotheses.alternatives.size()>1))) {
            // First acquire an effect model for unfamiliar actuator schemas.
            // Equal support ties are sampled; no request family routes this.
            size_t least=4;std::vector<int> undertrained;
            for(size_t i=0;i<available.size();++i) {
                auto frame=reader.actuators.frames.find(available[i].schema);
                if(frame!=reader.actuators.frames.end()&&frame->second.candidates.empty())continue;
                size_t support=frame==reader.actuators.frames.end()?0:frame->second.scenes.size();
                if(support<least){least=support;undertrained.clear();}
                if(support==least&&support<4)undertrained.push_back((int)i);
            }
            if(!undertrained.empty()) {
                result.index=undertrained[random()%undertrained.size()];result.status="explore_actuator";
            } else {
                const auto& meaning=hypotheses;
                size_t best=0;std::vector<int> choices;
                for(size_t i=0;i<available.size();++i) {
                    const auto& action=available[i];auto frame=reader.actuators.frames.find(action.schema);
                    if(frame==reader.actuators.frames.end()||frame->second.candidates.empty()||frame->second.arity!=action.arguments.size())continue;
                    size_t worst=meaning.alternatives.size();
                    for(const auto& rule:frame->second.candidates) {
                        if(result.predictions>=budget)return {0,-1,"action_search_budget",result.predictions};++result.predictions;
                        auto predicted=snapshot;for(auto& w:predicted.worlds)
                            w.facts=EventGrounding::update(w.facts,rule.relation,rule.effect,action.arguments[rule.subject],action.arguments[rule.object]);
                        size_t yes=0,no=0;
                        for(const auto& literal:meaning.alternatives) {
                            auto truth=RequestGrounding::truth(predicted,literal);
                            yes+=truth==RequestGrounding::TRUE_VALUE;no+=truth==RequestGrounding::FALSE_VALUE;
                        }
                        worst=std::min(worst,std::min(yes,no));
                    }
                    if(!information_seeking)worst=0;
                    if(worst>best){best=worst;choices.clear();}
                    if(worst==best)choices.push_back((int)i);
                }
                if(choices.empty())return {0,-1,"no_usable_action",result.predictions};
                if(!best&&exploit_index>=0){result.index=exploit_index;result.status="exploit";}
                else {result.index=choices[random()%choices.size()];result.status=best?"explore_meaning":"explore_uniform";}
            }
        } else result.status="exploit";
        result.token=next_token++;
        pending=Pending{result.token,reader.episode_revision,reader.request_revision,reader.observation_revision,reader.history_revision,
            reader.request,reader.events.signature(),reader.events,reader.observation_language,available.at(result.index),std::move(snapshot),
            {reader.names.begin(),reader.names.end()},{reader.relations[0],reader.relations[1]}};
        reader.awaiting_action_observation=true;
        return result;
    }
    // An independently verified observation learner may refine event meanings
    // during this transition. Its reconstructed pre-state must still match the
    // state on which the issued action was based; all old rules only narrow.
    bool rebase_observation_model(const LanguageReasoner& reader,uint64_t token,const std::string& previous_model,
                                  const RequestGrounding::Snapshot& reconstructed_before) {
        if(!pending||pending->token!=token){status="unknown_feedback_token";return false;}
        auto& p=*pending;EventGrounding::World old_world,replayed,current;
        if(p.event_model!=previous_model||p.episode!=reader.episode_revision||p.request_revision!=reader.request_revision||p.history_revision!=reader.history_revision||
           p.request!=reader.request||reader.observation_revision!=p.observation_revision+1||!reader.last_observation_environment||
           p.relations!=std::set<int>{reader.relations[0],reader.relations[1]}||
           !GroundedActuator::complete_world(p.before,old_world)||!GroundedActuator::complete_world(reconstructed_before,replayed)||
           old_world!=replayed||!GroundedActuator::complete_world(reader.request_snapshot(),current)) {
            status="invalid_observation_rebase";return false;
        }
        for(const auto& [key,frame]:p.event_snapshot.frames) {
            auto next=reader.events.frames.find(key);
            if(next==reader.events.frames.end()||frame.pattern!=next->second.pattern||next->second.candidates.empty()||
               !std::includes(frame.candidates.begin(),frame.candidates.end(),next->second.candidates.begin(),next->second.candidates.end())||
               !std::includes(next->second.scenes.begin(),next->second.scenes.end(),frame.scenes.begin(),frame.scenes.end())) {
                status="nonmonotone_observation_rebase";return false;
            }
        }
        p.event_model=reader.events.signature();p.event_snapshot=reader.events;status="observation_model_rebased";return true;
    }
    // A live evidence learner can refine its report model while the frozen
    // actuator awaits that report. All pre-action knowledge, including coverage
    // and negatives, must match; this does not admit partial effect training.
    bool rebase_evidence_model(const LanguageReasoner& reader,uint64_t token,const std::string& previous_model,
                               const RequestGrounding::Snapshot& historical) {
        if(learning||!pending||pending->token!=token){status="invalid_evidence_rebase";return false;}
        auto& p=*pending;
        if(p.evidence_snapshot.signature()!=previous_model||p.before!=historical||p.event_model!=reader.events.signature()||p.history_revision!=reader.history_revision||
           p.episode!=reader.episode_revision||p.request_revision!=reader.request_revision||p.request!=reader.request||
           reader.observation_revision!=p.observation_revision+1||!reader.last_observation_environment||
           !reader.request_snapshot().valid||p.relations!=std::set<int>{reader.relations[0],reader.relations[1]}) {
            status="invalid_evidence_rebase";return false;
        }
        for(const auto& [key,frame]:p.evidence_snapshot.frames) {
            auto next=reader.observation_language.frames.find(key);
            if(next==reader.observation_language.frames.end()||frame.pattern!=next->second.pattern||next->second.candidates.empty()||
               !std::includes(frame.candidates.begin(),frame.candidates.end(),next->second.candidates.begin(),next->second.candidates.end())||
               !std::includes(next->second.scenes.begin(),next->second.scenes.end(),frame.scenes.begin(),frame.scenes.end())) {
                status="nonmonotone_evidence_rebase";return false;
            }
        }
        p.evidence_snapshot=reader.observation_language;status="evidence_model_rebased";return true;
    }
    bool feedback_transition(LanguageReasoner& reader,uint64_t token,bool executed,bool satisfied,uint64_t scene,bool self_generated=false) {
        if(self_generated){status="self_generated_feedback";return false;}
        if(!pending||pending->token!=token){status="unknown_feedback_token";return false;}
        const auto& p=*pending;
        if(p.history_revision!=reader.history_revision){pending.reset();status="stale_history_feedback";return false;}
        if(p.episode!=reader.episode_revision||p.request_revision!=reader.request_revision||p.request!=reader.request) {
            pending.reset();status="stale_request_feedback";return false;
        }
        if(reader.observation_revision==p.observation_revision){status="awaiting_observation";return false;}
        if(reader.observation_revision!=p.observation_revision+1||!reader.last_observation_environment) {
            pending.reset();status="unmatched_observation";return false;
        }
        if(p.event_model!=reader.events.signature()||p.evidence_snapshot.signature()!=reader.observation_language.signature()||p.relations!=std::set<int>{reader.relations[0],reader.relations[1]}) {
            pending.reset();status="stale_observation_model";return false;
        }
        auto after=reader.request_snapshot();EventGrounding::World known;
        if(!learning) {
            if(!after.valid||after.worlds.empty()){pending.reset();status="missing_observation";return false;}
            EventGrounding::World before;
            if(!executed&&GroundedActuator::complete_world(p.before,before)&&GroundedActuator::complete_world(after,known)&&before!=known) {
                pending.reset();status="inconsistent_failure";return false;
            }
            pending.reset();status="observed_transition";return true;
        }
        if(!GroundedActuator::complete_world(after,known)){pending.reset();status="missing_observation";return false;}
        if(!executed&&p.before.worlds.front().facts!=known){pending.reset();status="inconsistent_failure";return false;}
        // One acknowledged transition and one satisfaction bit publish together.
        // Invalid/stale/source-mismatched feedback cannot half-update the model.
        auto motors=reader.actuators;auto requests=reader.requests;auto names=p.names;
        names.insert(reader.names.begin(),reader.names.end());
        if(!motors.teach(p.action,p.before,after,p.relations,executed,scene)) {
            status=motors.status;pending.reset();return false;
        }
        if(!requests.teach(p.request,after,names,p.relations,satisfied,scene)) {
            status=requests.status;pending.reset();return false;
        }
        status=motors.status=="conflicting_action_evidence"?motors.status:requests.status;
        reader.actuators=std::move(motors);reader.requests=std::move(requests);pending.reset();return true;
    }
};
} // namespace laminar::reasoning
