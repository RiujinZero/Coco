#pragma once
// Experimental learned selection of generic operations. This interface contains
// public machine state and prices only; it cannot inspect an evaluator's answer,
// hidden world, intended route, or a teacher's next-operation label.
#include "reasoning_runtime.h"
#include <random>

namespace laminar::reasoning {
enum class ComputationOp { retrieve, reason, ask, act, finish };
struct ComputationStatus {
    // 0: unread scope, 1: visible successor, 2: closed empty scope,
    // 3: retrieval returned no information. Empty and unknown are distinct.
    int scope = 0;
    bool effect_requested = false, effect_at_focus = false;
    std::array<float, 5> costs{0.01f, 0.005f, 0.05f, 0.02f, 0};
    Observation observe() const {
        if(scope < 0 || scope > 3) throw std::invalid_argument("invalid scope status");
        Observation o{};o[scope]=1;
        o[4+int(effect_requested)]=1;o[6+int(effect_at_focus)]=1;
        for(int a=0;a<5;++a) {
            if(!std::isfinite(costs[a]) || costs[a]<0) throw std::invalid_argument("invalid operation cost");
            o[8+a]=20*costs[a];
        }
        o[31]=1;return o;
    }
    std::vector<int> available() const {
        std::vector<int> a{0,2,3,4};
        if(scope==1)a.push_back(1); // a reason step requires a visible operand
        return a;
    }
};
struct ComputationDecision { Sample sample; float cost=0; };
template<class Choice> struct BasicComputationControl {
    Choice policy;
    std::mt19937 rng;
    std::vector<ComputationDecision> trace;
    size_t trace_budget=512;
    explicit BasicComputationControl(uint64_t seed):policy(5,seed),rng(seed+4099){}
    void begin(){trace.clear();}
    ComputationOp choose(const ComputationStatus& status,float epsilon=0,int disabled=-1) {
        auto obs=status.observe();auto available=status.available();
        available.erase(std::remove(available.begin(),available.end(),disabled),available.end());
        if(available.empty() || trace.size()>=trace_budget)throw std::runtime_error("computation budget exhausted");
        int a=policy.choose(obs,available);
        if(std::generate_canonical<float,24>(rng)<epsilon)a=available[rng()%available.size()];
        trace.push_back({{obs,available,a},status.costs[a]});
        return static_cast<ComputationOp>(a);
    }
    // Replay observed decisions with separate successor-value residuals. The
    // only correctness signal is the verified terminal outcome. Operation cost
    // is observed consumption, not an intermediate correctness reward.
    void credit(float outcome,bool terminal_only=false) {
        if(!std::isfinite(outcome))throw std::invalid_argument("invalid outcome");
        for(int i=int(trace.size())-1;i>=0;--i) {
            const auto& d=trace[i];float target=outcome;
            if(i+1<int(trace.size())) {
                if(terminal_only)continue;
                const auto& next=trace[i+1].sample;
                target=0.99f*policy.best_value(next.obs,next.available);
            }
            policy.value_update(d.sample.obs,d.sample.action,target-d.cost);
        }
    }
};
using ComputationControl=BasicComputationControl<LocalChoice>;
}
