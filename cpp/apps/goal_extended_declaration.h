#pragma once
// Declared schedule and configuration of the 2,000-episode continuing subsystem run, shared by
// goal_extended_bootstrap.cpp (the original run) and subsystem_run.cpp (the same run with checkpoints).
#define LAMINAR_GOAL_MIXED_LIBRARY
#include "goal_continuing_bootstrap.cpp"

static constexpr std::array<size_t,20> extended_order={1,2,1,2,1,2,0,1,2,1,2,1,2,0,3,4,5,6,7,0};
static double assistance(size_t index,bool smoke) {
    const size_t full=smoke?20:800,warm=smoke?40:1600;
    return index<full?1.:index>=warm?0.:double(warm-1-index)/double(warm-full);
}
static std::string extended_configuration(bool smoke) {
    std::ostringstream out;
    out<<"{\"schema\":\"goal_extended_bootstrap_v1\",\"smoke\":"<<(smoke?"true":"false")
       <<",\"seed\":1,\"hidden\":1316,\"total_training\":"<<(smoke?48:2000)
       <<",\"full_guidance\":"<<(smoke?20:800)<<",\"taper_end\":"<<(smoke?40:1600)
       <<",\"family_cycle\":[1,2,1,2,1,2,0,1,2,1,2,1,2,0,3,4,5,6,7,0],"
         "\"family_counts_full_run\":[300,600,600,100,100,100,100,100],"
         "\"fit_every\":20,\"fit_steps\":\"ceil(10000*guidance_probability), balanced equally by public phase and demonstrated action, then unique states; no fits in autonomous phase\","
         "\"max_direct_fits\":800000,\"terminal_updates\":512,\"max_total_updates\":1824000,"
         "\"autonomous_epsilon\":0.1,\"prices\":\"all zero\",\"decisions\":256,\"motors\":32,"
         "\"teacher\":\"unchanged continuing_curriculum_v1; known reference scheduling failures remain logged\","
         "\"setup\":\"same raw base, reader, long and initial skills as pilot; repair/union extension after100 training missions (smoke upfront)\","
         "\"identity\":\"one continuing learner per run; fresh experimental run, not resumed full checkpoint\","
         "\"task_sampling\":\"per-family occurrence counter; same training arrival sets and stopping examples; post-extension query index offset8000; no heldout labels in training\","
         "\"evaluation\":\"same46 frozen developmental probes initially/every400/final (smoke8 initially/final); no hints/fits/learning\","
         "\"exports\":\"all decisions, computations, mission outcomes; weights initially and every20missions/final\","
         "\"limits\":\"same16derived/min2/4096preview allowance/2Mwork/depth8 and agenda actual-exhaustion observation\"}\n";
    return out.str();
}
