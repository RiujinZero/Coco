#pragma once
#include "laminar/relational_control.h"

namespace laminar::training {
// An explicitly declared training-side teacher. Only the controller's public
// observation enters these rules; private truth, goals and effects are excluded.
// The learner still computes plans and primitive choices using its own memory.
struct GuidanceAdvice {relational::ControlOp operation;const char* subgoal;};
inline GuidanceAdvice public_control_advice(const reasoning::Observation& o) {
    using relational::ControlOp;
    if(o[2])return {ControlOp::reason,"refresh_model"};
    if(!o[0]) {
        if(!o[12]&&(o[13]||o[21]<=o[23]))return {ControlOp::retrieve,"obtain_world"};
        if(!o[13])return {ControlOp::ask,"obtain_world"};
        return {ControlOp::finish,"report_unavailable"};
    }
    if(!o[5])return {ControlOp::reason,"evaluate_goal"};
    if(!o[4]||o[6]||o[8])return {ControlOp::finish,"report_outcome"};
    if(o[9])return {ControlOp::act,"execute_plan_step"};
    if(o[10]&&!o[20])return {ControlOp::reason,"extend_plan_search"};
    return {ControlOp::finish,"report_unavailable"};
}
struct GuidanceSchedule {
    bool configured=false;
    std::string source;
    size_t full=0,fade=0;
    unsigned seed=0;
    double rate(size_t completed)const {
        if(!configured)return 0;
        if(completed<full)return 1;
        if(completed-full>=fade)return 0;
        return double(fade-(completed-full))/double(fade);
    }
    const char* phase(size_t completed)const {
        return completed<full?"bootstrap":rate(completed)>0?"taper":"autonomous";
    }
    bool intervene(size_t completed,size_t decision)const {
        // Stable per-decision lottery, independent of policy exploration RNG.
        // Integer arithmetic is mirrored by the independent audit.
        uint64_t x=(uint64_t(seed)+1)*0x9e3779b97f4a7c15ull;
        x^=(uint64_t(completed)+1)*0xbf58476d1ce4e5b9ull;
        x^=(uint64_t(decision)+1)*0x94d049bb133111ebull;
        x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;x^=x>>31;
        return double(x>>11)*0x1.0p-53<rate(completed);
    }
};
} // namespace laminar::training
