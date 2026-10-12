#pragma once
#include "goal_control_learning_harness.h"
namespace laminar::goal_learning {
inline constexpr const char* curriculum_demonstration_source="continuing_curriculum_v1";
inline int curriculum_demonstration(const Observation& o,const std::vector<int>& menu) {
    // One explicitly supplied warm example function for every family. No task
    // identity, evaluator truth, hidden scene, plan or preferred goal is input.
    auto legal=[&](int action){return std::find(menu.begin(),menu.end(),action)!=menu.end();};
    int action=5;
    if(o[29]) {
        if(!legal(6))action=5;
        else if(o[12]&&o[17]>=o[16])action=5;
        else if(!o[0]||!o[4]||(o[13]&&o[11]&&o[6]))action=6;
        else if(o[13]&&o[10]&&o[15])action=5;
        else if((o[13]&&o[10])||!o[5])action=1;
        else if((!o[13]&&(o[6]||o[9]))||(o[13]&&o[18]&&o[19]>0))action=6;
    } else if(o[30])action=5;
    else if(!o[5])action=1;
    else if((!o[4]&&(o[6]||o[7]))||(o[4]&&o[6]))action=4;
    else if(!o[0]||o[8])action=o[12]?0:2;
    else if(o[4]&&o[9])action=3;
    else if(o[10]&&o[19]<1)action=1;
    require(legal(action),"mixed curriculum demonstration unavailable action");return action;
}
} // namespace laminar::goal_learning
