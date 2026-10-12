#pragma once
// The supported continual-learning profile. Keep experiments as explicit
// ablations of this configuration, rather than silently changing this baseline.
#include "relational_control.h"

namespace laminar::training {
struct ContinualTrainingConfig {
    static constexpr const char* name="coco_continual_v1";
    std::string profile=name;
    int controller_hidden=32;
    bool episodic_return=false;
    bool greedy_trace_credit=false;
    bool disjunctive_concepts=false;
    bool freeze_control=false,shuffled_credit=false,terminal_only_credit=false;
    size_t control_ablation_after=0;
    bool repair_available=false;
    bool repair_bank=false;
    int interleave_quantum=0;
    bool observation_records=false;
    reasoning::SkillShape skill_shape;
    static ContinualTrainingConfig named(const std::string& profile) {
        ContinualTrainingConfig c;c.profile=profile;
        if(profile==name)return c;
        for(const auto& base:{std::string("controller50k"),std::string("controller50k_trace")}) {
            if(profile==base+"_frozen_after400"||profile==base+"_shuffled_after400") {
                c=named(base);c.profile=profile;c.control_ablation_after=400;
                c.freeze_control=profile==base+"_frozen_after400";
                c.shuffled_credit=profile==base+"_shuffled_after400";return c;
            }
            if(profile==base+"_frozen"||profile==base+"_shuffled"||profile==base+"_terminal") {
                c=named(base);c.profile=profile;c.freeze_control=profile.ends_with("_frozen");
                c.shuffled_credit=profile.ends_with("_shuffled");c.terminal_only_credit=profile.ends_with("_terminal");return c;
            }
        }
        if(profile=="controller50k_disjunction") {
            c.controller_hidden=1316;c.disjunctive_concepts=true;return c;
        }
        if(profile=="controller_trace"||profile=="controller50k_trace") {
            c.greedy_trace_credit=true;
            if(profile=="controller50k_trace")c.controller_hidden=1316;
            return c;
        }
        if(profile=="controller50k_records"||profile=="controller50k_repair_records"||profile=="controller50k_repair_bank_records") {
            c.controller_hidden=1316;c.observation_records=true;
            c.repair_available=profile!="controller50k_records";
            c.repair_bank=profile=="controller50k_repair_bank_records";
            if(c.repair_available)c.interleave_quantum=1000;
            return c;
        }
        if(profile=="controller50k"||profile=="both50k"||profile=="controller50k_return"||profile=="controller50k_repair"||profile=="controller50k_repair_interleaved")c.controller_hidden=1316;
        else if(profile=="controller100k"||profile=="both100k")c.controller_hidden=2632;
        if(profile=="agkern50k"||profile=="both50k")c.skill_shape={5000,16,1};
        else if(profile=="agkern100k"||profile=="both100k")c.skill_shape={10000,16,1};
        if(profile=="controller_return"||profile=="controller50k_return")c.episodic_return=true;
        if(profile=="controller50k_repair"||profile=="controller50k_repair_interleaved")c.repair_available=true;
        if(profile=="controller50k_repair_interleaved")c.interleave_quantum=1000;
        if(c.controller_hidden==32&&c.skill_shape.units==512&&!c.episodic_return)throw std::invalid_argument("unknown continual profile");
        return c;
    }
    int discovery_budget=100000;
    size_t exploration_episodes=200,updates_per_episode=128;
    size_t consolidate_every=200,consolidation_passes=64;
    size_t training_decisions=48,evaluation_decisions=128;
    float initial_exploration=0.4f,later_exploration=0.15f;
    relational::ControlBody body{true,1,32,false,false};
    bool evaluation_preserves_search=true;
    float exploration(size_t completed)const {
        return completed<exploration_episodes?initial_exploration:later_exploration;
    }
    bool control_frozen(size_t completed)const{return freeze_control&&completed>=control_ablation_after;}
    bool credit_shuffled(size_t completed)const{return shuffled_credit&&completed>=control_ablation_after;}
    relational::ControlledRelationalSession create(unsigned seed)const {
        relational::ControlledRelationalSession learner(seed,true,terminal_only_credit,true,body,false,controller_hidden,skill_shape,episodic_return,greedy_trace_credit);
        if(disjunctive_concepts){concepts::Limits limits;limits.clauses=2;learner.knowledge.enable_disjunctive_concepts(limits);}
        if(observation_records)learner.knowledge.enable_observed_relation_types();
        learner.knowledge.enable_composed_learning(discovery_budget,repair_available,interleave_quantum,repair_bank);
        if(observation_records)learner.preserve_observation_records();
        return learner;
    }
    std::string json(unsigned seed)const {
        std::ostringstream out;
        out<<"{\n  \"profile\": \""<<profile<<"\",\n  \"seed\": "<<seed
           <<",\n  \"control_learning\": \""<<(greedy_trace_credit?"empirical_greedy_trace_and_backprop_preferences":episodic_return?"empirical_episode_return_and_backprop_preferences":"empirical_td_and_backprop_preferences")<<"\","
           <<"\n  \"primitive_learning\": \"whole_policy_search_and_local_neural_rehearsal\","
           <<"\n  \"discovery_proposals_per_attempt\": "<<discovery_budget
           <<",\n  \"updates_per_episode\": "<<updates_per_episode
           <<",\n  \"consolidate_every\": "<<consolidate_every
           <<",\n  \"consolidation_passes\": "<<consolidation_passes
           <<",\n  \"initial_exploration\": "<<initial_exploration
           <<",\n  \"later_exploration\": "<<later_exploration
           <<",\n  \"exploration_episodes\": "<<exploration_episodes
           <<",\n  \"training_decisions\": "<<training_decisions
           <<",\n  \"evaluation_decisions\": "<<evaluation_decisions
           <<",\n  \"reactive\": true,\n  \"motor_batch\": "<<body.batch
           <<",\n  \"motor_attempts\": "<<body.motors
           <<",\n  \"quantitative_control_inputs\": false,"
           <<"\n  \"training_preserves_search\": false,"
           <<"\n  \"evaluation_preserves_search\": true,"
           <<"\n  \"old_skill_rehearsal\": true,"
           <<"\n  \"composed_rehearsal_passes\": 128,"
           <<"\n  \"macro_instruction_limit\": 8,\n  \"macro_state_limit\": 1000,"
           <<"\n  \"checkpoint_kind\": \"deterministic_offline_replay\"";
        if(profile!=name)out<<",\n  \"experimental_capacity\": "<<((controller_hidden!=32||skill_shape.units!=512)?"true":"false")
           <<",\n  \"controller_hidden\": "<<controller_hidden
           <<",\n  \"controller_parameters\": "<<38*controller_hidden+5
           <<",\n  \"control_preference_learning_rate\": "<<.002f*(32.f/float(controller_hidden))
           <<",\n  \"skill_units\": "<<skill_shape.units
           <<",\n  \"skill_active\": "<<skill_shape.active
           <<",\n  \"skill_channels\": "<<skill_shape.channels
           <<",\n  \"skill_trainable_parameters\": "<<10*skill_shape.units
           <<",\n  \"skill_allocated_route_weights\": "<<10*skill_shape.units*skill_shape.channels;
        if(episodic_return)out<<",\n  \"experimental_credit\": \"observed_episode_return\"";
        if(greedy_trace_credit)out<<",\n  \"experimental_credit\": \"goal_scoped_greedy_trace\"";
        if(disjunctive_concepts)out<<",\n  \"concept_local_clauses\": 2,\n  \"concept_literals_per_clause\": 3,\n  \"concept_expanded_clauses\": 64";
        if(freeze_control)out<<",\n  \"freeze_control\": true";
        if(shuffled_credit)out<<",\n  \"shuffled_credit\": true";
        if(terminal_only_credit)out<<",\n  \"terminal_only_credit\": true";
        if(control_ablation_after)out<<",\n  \"control_ablation_after\": "<<control_ablation_after;
        if(repair_available)out<<",\n  \"experimental_primitive_repair\": true";
        if(repair_bank)out<<",\n  \"experimental_repair_bank\": true";
        if(interleave_quantum)out<<",\n  \"discovery_interleave_quantum\": "<<interleave_quantum<<",\n  \"frozen_compile_choice_cache\": true,\n  \"candidate_assessment_cache_entries\": 128";
        if(observation_records)out<<",\n  \"observation_records\": \"sorted_with_multiplicity\",\n  \"relational_truth\": \"distinct_tuples\",\n  \"observed_relation_types\": true";
        out<<"\n}\n";
        return out.str();
    }
};
} // namespace laminar::training
