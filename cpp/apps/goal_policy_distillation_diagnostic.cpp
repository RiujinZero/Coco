// Reuse the exact declared warm/task routines without calling the full pilot.
// The original immutable measured source remains in its v1 evidence snapshot.
#define main goal_control_learning_pilot_unused_entry
#include "goal_control_learning_pilot.cpp"
#undef main

static std::string diagnostic_configuration(const std::string& reference) {
    std::ostringstream out;
    out<<"{\n\"schema\":\"goal_policy_distillation_diagnostic_v1\",\n\"reference_prefix\":"<<gl::quote(reference)
       <<",\n\"seed\":1,\n\"reconstruct_warm_missions\":100,\n\"verify_exact_saved_warm_weights\":true,\n"
       <<"\"verify_saved_warm_evaluation\":true,\n\"neural_evaluation_cases\":12,\n\"memory_readout_evaluation_cases\":12,\n"
       <<"\"consolidated_neural_evaluation_cases\":12,\n\"requested_consolidation_rounds\":256,\n\"additional_fit_cap\":100000,\n"
       <<"\"round_rule\":\"min(256,floor(100000/unique_active_credited_observations)); zero if no full round fits\",\n"
       <<"\"new_post_warm_environment_rewards\":0,\n\"new_post_warm_hints\":0,\n\"costs\":\"same all-zero pilot prices\",\n"
       <<"\"evaluation\":\"same12 held-out cases on independent frozen clones; explicit memory_policy true only for the memory diagnosis\",\n"
       <<"\"export\":\"all empirical values and exact32-float/menu neural/empirical readouts, assistance counts including13 rare arrival states, before/after consolidation\",\n"
       <<"\"scope\":\"diagnosis of learned values versus neural fitting; memory readout is not neural policy acquisition\"\n}\n";
    return out.str();
}
static double field_number(const std::string& row,const std::string& field) {
    auto start=row.find("\""+field+"\":");gl::require(start!=std::string::npos,"missing reference numeric field");
    return std::stod(row.substr(start+field.size()+3));
}
static bool field_bool(const std::string& row,const std::string& field) {
    auto start=row.find("\""+field+"\":");gl::require(start!=std::string::npos,"missing reference Boolean field");
    auto value=row.substr(start+field.size()+3);gl::require(value.starts_with("true")||value.starts_with("false"),"invalid reference Boolean");return value.starts_with("true");
}
static std::vector<float> saved_warm_weights(const std::string& prefix) {
    std::ifstream file(prefix+"_weights.jsonl");gl::require(bool(file),"missing original policy export");
    std::string row;std::vector<float> result;size_t matches=0;
    while(std::getline(file,row))if(row.find("\"label\":\"warm100\"")!=std::string::npos) {
        ++matches;gl::require(field_number(row,"updates")==12800&&field_number(row,"credits")==100,"unexpected saved warm checkpoint");
        auto begin=row.find("\"weights\":[");gl::require(begin!=std::string::npos,"missing saved warm weights");begin+=11;
        auto end=row.find(']',begin);gl::require(end!=std::string::npos,"unterminated saved warm weights");
        auto values=row.substr(begin,end-begin);std::replace(values.begin(),values.end(),',',' ');std::istringstream input(values);
        float value;while(input>>value){gl::require(std::isfinite(value),"nonfinite saved weight");result.push_back(value);}
    }
    gl::require(matches==1&&!result.empty(),"nonunique or empty warm checkpoint");return result;
}
struct ExpectedEvaluation {bool success=false;size_t decisions=0,motors=0,predictions=0;double utility=0;};
static std::map<std::string,ExpectedEvaluation> expected_evaluation(const std::string& prefix) {
    std::ifstream file(prefix+"_missions.jsonl");gl::require(bool(file),"missing original mission export");
    std::map<std::string,ExpectedEvaluation> result;std::string row;
    while(std::getline(file,row))if(row.find("\"arm\":\"eval_warm100\"")!=std::string::npos) {
        auto start=row.find("\"id\":\"")+6;auto end=row.find('"',start);auto id=row.substr(start,end-start);
        gl::require(result.emplace(id,ExpectedEvaluation{field_bool(row,"success"),size_t(field_number(row,"decisions")),
            size_t(field_number(row,"motors")),size_t(field_number(row,"predictions")),field_number(row,"observed_utility")}).second,"duplicate reference evaluation");
    }
    gl::require(result.size()==12,"original warm evaluation must have12 cases");return result;
}
using StateKey=std::pair<gl::Observation,std::vector<int>>;
struct StateUse {size_t visits=0,assisted=0;std::set<int> assistance;};
static std::map<StateKey,StateUse> support(const GoalControlledRelationalSession& learner) {
    std::map<StateKey,StateUse> result;
    for(const auto& receipt:learner.credits())if(receipt.active)for(const auto& row:receipt.trace) {
        auto& state=result[{row.observation,row.available}];++state.visits;
        if(!row.teacher.empty()){++state.assisted;state.assistance.insert(int(row.operation));}
    }
    return result;
}
static void export_values(const GoalControlledRelationalSession& learner,const std::string& prefix,const std::string& stage) {
    std::ofstream values(prefix+"_"+stage+"_empirical.jsonl"),readouts(prefix+"_"+stage+"_readouts.jsonl");
    gl::require(values&&readouts,"cannot export learned-value diagnostic");
    for(auto* out:{&values,&readouts})*out<<std::boolalpha<<std::setprecision(17);
    for(const auto& [observation,learned]:learner.empirical_values()) {
        values<<"{\"observation_bits\":"<<gl::quote(gl::observation_bits(observation))<<",\"observation\":";gl::numbers(values,observation);
        values<<",\"values\":";gl::numbers(values,learned);values<<"}\n";
    }
    size_t arrival_states=0,arrival_hints=0;
    for(const auto& [key,use]:support(learner)) {
        const auto& [observation,menu]=key;
        const int neural=learner.policy.choose(observation,menu);int empirical=neural;
        auto found=learner.empirical_values().find(observation);
        if(found!=learner.empirical_values().end()) {
            empirical=menu.front();for(int action:menu)if(found->second[action]>found->second[empirical])empirical=action;
        }
        bool arrival=!observation[29]&&observation[30];arrival_states+=arrival&&use.assisted;arrival_hints+=arrival?use.assisted:0;
        std::vector<float> scores;for(int action=0;action<7;++action)scores.push_back(learner.policy.action_value(observation,action));
        readouts<<"{\"observation_bits\":"<<gl::quote(gl::observation_bits(observation))<<",\"observation\":";gl::numbers(readouts,observation);
        readouts<<",\"menu\":";gl::numbers(readouts,menu);readouts<<",\"visits\":"<<use.visits<<",\"assisted\":"<<use.assisted
            <<",\"assisted_operations\":";gl::numbers(readouts,use.assistance);readouts<<",\"arrival\":"<<arrival
            <<",\"neural_operation\":"<<neural<<",\"memory_operation\":"<<empirical<<",\"memory_found\":"<<(found!=learner.empirical_values().end())
            <<",\"neural_scores\":";gl::numbers(readouts,scores);readouts<<",\"empirical_values\":";
        if(found!=learner.empirical_values().end())gl::numbers(readouts,found->second);else readouts<<"null";readouts<<"}\n";
    }
    gl::require(arrival_states==13&&arrival_hints==96,"reconstructed rare arrival guidance differs from frozen pilot");
}
static void export_procedure_inventory(const GoalControlledRelationalSession& learner,const std::string& prefix) {
    std::ofstream out(prefix+"_warm_procedures.json");gl::require(bool(out),"cannot export warm procedure inventory");
    out<<std::boolalpha<<"{\"schema\":\"cached_warm_procedure_inventory_v1\",\"model_version\":"<<learner.knowledge.version()
       <<",\"knowledge_dirty\":"<<learner.knowledge.dirty()<<",\"count\":"<<learner.knowledge.procedures().size()<<",\"procedures\":[";
    bool first=true;for(const auto& procedure:learner.knowledge.procedures()) {
        if(!first)out<<',';first=false;out<<"{\"frame\":"<<gl::quote(procedure.frame)<<",\"code\":[";
        bool first_instruction=true;for(const auto& instruction:procedure.code) {
            if(!first_instruction)out<<',';first_instruction=false;out<<"{\"schema\":"<<gl::quote(instruction.schema)<<",\"operands\":[";
            bool first_operand=true;for(const auto& operand:instruction.operands) {
                if(!first_operand)out<<',';first_operand=false;
                out<<"{\"argument\":"<<operand.argument<<",\"predicate\":"<<gl::quote(operand.predicate)
                   <<",\"source_slot\":"<<operand.source_slot<<",\"target_slot\":"<<operand.target_slot<<'}';
            }out<<"]}";
        }out<<"]}";
    }out<<"]}\n";
}
static size_t diagnostic_evaluate(const GoalControlledRelationalSession& source,const std::string& label,bool memory,
        Outputs& output,const std::map<std::string,ExpectedEvaluation>* expected=nullptr) {
    auto frozen=source;frozen.set_learning(false);const auto weights=source.policy.weights;
    const auto updates=source.updates(),credits=source.credits().size(),version=source.knowledge.version();
    size_t successes=0,index=0;
    for(unsigned order:evaluation_orders)for(size_t arrival:evaluation_arrivals) {
        SixEdgeEnvironment environment(arrival,order);auto config=options("eval_"+label,index++,false);
        config.task_order_seed=order;config.memory_policy=memory;
        auto result=gl::run_mission(frozen,environment,config,output.trace,output.missions);successes+=result.success;
        if(expected) {
            const auto& prior=expected->at(result.id);
            gl::require(result.success==prior.success&&result.decisions==prior.decisions&&result.motors==prior.motors
                &&result.predictions==prior.predictions&&std::abs(result.utility-prior.utility)<1e-12,"warm evaluation failed exact reconstruction");
        }
    }
    gl::require(frozen.policy.weights==weights&&frozen.updates()==updates&&frozen.credits().size()==credits&&frozen.knowledge.version()==version,
        "diagnostic evaluation changed frozen learned state");
    gl::require(source.policy.weights==weights&&source.updates()==updates&&source.credits().size()==credits&&source.knowledge.version()==version,
        "diagnostic evaluation changed source");
    output.evaluations<<"{\"checkpoint\":"<<gl::quote(label)<<",\"successes\":"<<successes<<",\"cases\":12,\"memory_policy\":"<<memory
        <<",\"updates\":"<<updates<<",\"credits\":"<<credits<<",\"model_version\":"<<version<<",\"hints\":0,\"epsilon\":0,\"frozen\":true}\n";output.evaluations.flush();
    std::cout<<"diagnostic evaluation "<<label<<" "<<successes<<"/12 memory="<<memory<<'\n'<<std::flush;return successes;
}
#ifndef GOAL_POLICY_DISTILLATION_ENTRY
#define GOAL_POLICY_DISTILLATION_ENTRY main
#endif
int GOAL_POLICY_DISTILLATION_ENTRY(int argc,char** argv) {
    try {
        gl::require(argc==4,"usage: goal_policy_distillation_diagnostic --declare|--run OUTPUT_PREFIX ORIGINAL_V1_PREFIX");
        const std::string mode=argv[1],prefix=argv[2],original=argv[3];
        if(mode=="--declare") {
            gl::require(!std::filesystem::exists(prefix+"_config.json"),"diagnostic declaration already exists");
            std::ofstream out(prefix+"_config.json");gl::require(bool(out),"cannot declare diagnostic");out<<diagnostic_configuration(original);return 0;
        }
        gl::require(mode=="--run","unknown diagnostic mode");std::ifstream declaration(prefix+"_config.json");
        std::string declared((std::istreambuf_iterator<char>(declaration)),std::istreambuf_iterator<char>());
        gl::require(declared==diagnostic_configuration(original),"diagnostic config missing or changed");
        for(const auto& suffix:{"_trace.jsonl","_missions.jsonl","_weights.jsonl","_evaluations.jsonl","_bootstrap.jsonl","_summary.json",
                               "_warm_empirical.jsonl","_warm_readouts.jsonl","_consolidated_empirical.jsonl","_consolidated_readouts.jsonl","_warm_procedures.json"})
            gl::require(!std::filesystem::exists(prefix+suffix),"refusing to overwrite diagnostic evidence");
        const auto expected_weights=saved_warm_weights(original);const auto expected=expected_evaluation(original);
        Outputs output(prefix);auto fixture=gt::make_fixture();GoalControlledRelationalSession warm(seed,32,true);
        warm.knowledge=RelationalSession(seed,32768,1024);fixture.lessons.apply(warm.knowledge);export_bootstrap(fixture.lessons,output.bootstrap);
        gl::weights(output.parameters,warm,"initial","reconstruction");train(warm,"warm",0,100,output);
        gl::require(warm.policy.weights==expected_weights&&warm.updates()==12800,"saved warm weights not reconstructed exactly");
        gl::weights(output.parameters,warm,"warm100_verified","reconstruction");export_values(warm,prefix,"warm");
        export_procedure_inventory(warm,prefix);
        const auto neural=diagnostic_evaluate(warm,"warm100",false,output,&expected);
        const auto memory=diagnostic_evaluate(warm,"memory_only",true,output);
        auto consolidated=warm;std::set<gl::Observation> representatives;
        for(const auto& credit:consolidated.credits())if(credit.active)for(const auto& row:credit.trace)representatives.insert(row.observation);
        gl::require(!representatives.empty(),"empty consolidation support");
        const size_t rounds=std::min<size_t>(256,100000/representatives.size());
        const auto empirical_before=consolidated.empirical_values();const auto version=consolidated.knowledge.version();const auto credit_count=consolidated.credits().size();
        const auto fitted=rounds?consolidated.consolidate(rounds):0;
        gl::require(fitted==representatives.size()*rounds&&fitted<=100000&&consolidated.updates()==12800+fitted,"consolidation exceeded declared fit bound");
        gl::require(consolidated.empirical_values()==empirical_before&&consolidated.knowledge.version()==version&&consolidated.credits().size()==credit_count,
            "consolidation changed learned values or semantic evidence");
        gl::weights(output.parameters,consolidated,"consolidated","existing_values_only");export_values(consolidated,prefix,"consolidated");
        const auto refit=diagnostic_evaluate(consolidated,"consolidated_neural",false,output);
        std::ofstream result(prefix+"_summary.json");result<<std::boolalpha;
        result<<"{\n\"schema\":\"goal_policy_distillation_diagnostic_result_v1\",\n\"complete\":true,\n\"warm_weights_exact\":true,\n"
            <<"\"warm_evaluation_exact\":true,\n\"neural_warm_successes\":"<<neural<<",\n\"memory_only_successes\":"<<memory
            <<",\n\"consolidated_neural_successes\":"<<refit<<",\n\"cases_per_evaluation\":12,\n\"representatives\":"<<representatives.size()
            <<",\n\"rounds\":"<<rounds<<",\n\"additional_fits\":"<<fitted<<",\n\"empirical_values_unchanged\":true,\n"
            <<"\"new_post_warm_rewards\":0,\n\"new_post_warm_hints\":0,\n\"memory_readout_is_diagnostic_only\":true\n}\n";
        std::cout<<"diagnostic complete reps="<<representatives.size()<<" rounds="<<rounds<<" additional_fits="<<fitted<<'\n';return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
