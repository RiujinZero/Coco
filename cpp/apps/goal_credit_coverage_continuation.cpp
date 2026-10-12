// Reuse exact warm reconstruction, task, export and frozen-evaluation helpers.
// Neither included app entry point is called by this continuation.
#define GOAL_POLICY_DISTILLATION_ENTRY goal_distillation_unused_entry
#include "goal_policy_distillation_diagnostic.cpp"
#undef GOAL_POLICY_DISTILLATION_ENTRY

static std::string continuation_configuration(const std::string& original) {
    std::ostringstream out;
    out<<"{\n\"schema\":\"goal_credit_coverage_continuation_v1\",\n\"reference_prefix\":"<<gl::quote(original)
       <<",\n\"seed\":1,\n\"hidden\":32,\n\"warm_reconstruction\":\"exact original100 missions, weights and12case evaluation verified\",\n"
       <<"\"arms\":[\"adaptive512\",\"random_credit512\"],\n\"autonomous_missions_per_arm\":200,\n\"task_indices\":[100,299],\n"
       <<"\"terminal_updates\":512,\n\"epsilon\":0.10,\n\"memory_policy\":false,\n\"post_warm_hints\":0,\n"
       <<"\"limits\":{\"roots\":2,\"derived\":0,\"decisions\":256,\"motors\":32},\n\"prices\":\"all fields0, unchanged\",\n"
       <<"\"training_arrivals\":[3,5,7],\n\"training_affordance_order_seed\":\"1000+mission_index\",\n"
       <<"\"evaluation_arrivals\":[4,6,8],\n\"evaluation_affordance_order_seeds\":[41,43,47,53],\n"
       <<"\"evaluation_scope\":\"original12cases reused for development; no untouched confirmation claim\",\n"
       <<"\"endpoint_evaluation\":\"neural before consolidation, then neural on separate bounded-consolidation clone\",\n"
       <<"\"stopping_metrics\":\"read-only postrun audit of physical satisfaction, policy-issued finish, urgent timeliness, whole missions and cached-True nonfinish/reason/retrieve loops\",\n"
       <<"\"consolidation_rounds\":\"min(256,floor(100000/unique_active_credited_observations)) at each endpoint\",\n"
       <<"\"consolidation_fit_cap_per_arm\":100000,\n\"consolidation_changes_values_or_semantics\":false,\n"
       <<"\"reuse_frozen_comparator\":\"original200mission frozen-policy control only after exact warm verification; runner task/physics remain identical\",\n"
       <<"\"rng_caveat\":\"extra replay and random-credit draws change subsequent learner RNG trajectories; public schedules and warm starting state are matched\",\n"
       <<"\"coverage_change\":\"512updates cover all <=256 current rows and then older-receipt replay; no new core credit rule\",\n"
       <<"\"scope\":\"single-seed replay-coverage trial, with readout consolidation assessed separately; no width, feature or curriculum change\"\n}\n";
    return out.str();
}
static void continue_with_coverage(GoalControlledRelationalSession& learner,const std::string& arm,Outputs& output) {
    size_t successes=0;
    for(size_t index=100;index<300;++index) {
        auto config=options(arm,index);config.terminal_updates=512;config.epsilon=.10f;
        config.random_credit=arm=="random_credit512";
        SixEdgeEnvironment environment(training_arrivals[index%training_arrivals.size()],config.task_order_seed);
        auto result=gl::run_mission(learner,environment,config,output.trace,output.missions);successes+=result.success;
        gl::require(result.hints==0&&result.cost==0&&result.updates_after-result.updates_before==512,"coverage trial deviated from declaration");
        gl::weights(output.parameters,learner,config.id,arm);
        if((index+1)%10==0)std::cout<<arm<<" missions="<<index-99<<" successes="<<successes<<" updates="<<learner.updates()<<'\n'<<std::flush;
    }
}
struct Endpoint {size_t before=0,after=0,representatives=0,rounds=0,fits=0;};
static Endpoint endpoint(const GoalControlledRelationalSession& learner,const std::string& arm,
                         const std::string& prefix,Outputs& output) {
    Endpoint result;export_values(learner,prefix,arm+"_before");
    result.before=diagnostic_evaluate(learner,arm+"_before",false,output);
    auto consolidated=learner;std::set<gl::Observation> representatives;
    for(const auto& credit:consolidated.credits())if(credit.active)for(const auto& row:credit.trace)representatives.insert(row.observation);
    gl::require(!representatives.empty(),"empty endpoint support");result.representatives=representatives.size();
    result.rounds=std::min<size_t>(256,100000/result.representatives);
    const auto before_values=consolidated.empirical_values();const auto before_version=consolidated.knowledge.version();
    const auto before_credits=consolidated.credits().size(),before_updates=consolidated.updates();
    result.fits=result.rounds?consolidated.consolidate(result.rounds):0;
    gl::require(result.fits==result.rounds*result.representatives&&result.fits<=100000&&consolidated.updates()==before_updates+result.fits,
        "endpoint consolidation exceeded declared fits");
    gl::require(consolidated.empirical_values()==before_values&&consolidated.knowledge.version()==before_version
        &&consolidated.credits().size()==before_credits,"endpoint consolidation changed values or semantic receipts");
    gl::weights(output.parameters,consolidated,arm+"_consolidated",arm);export_values(consolidated,prefix,arm+"_after");
    result.after=diagnostic_evaluate(consolidated,arm+"_after",false,output);return result;
}
static void endpoint_json(std::ostream& out,const Endpoint& value) {
    out<<"{\"neural_before\":"<<value.before<<",\"neural_after\":"<<value.after<<",\"cases\":12,\"representatives\":"<<value.representatives
       <<",\"rounds\":"<<value.rounds<<",\"additional_fits\":"<<value.fits<<",\"values_and_semantics_unchanged\":true}";
}
struct FrozenComparator {size_t missions=0,successes=0,cases=0,evaluation_successes=0,weight_checks=0;};
static FrozenComparator verify_frozen_comparator(const std::string& original,const std::vector<float>& warm_weights) {
    std::ifstream config(original+"_config.json");std::string declared((std::istreambuf_iterator<char>(config)),std::istreambuf_iterator<char>());
    gl::require(declared==configuration(),"original pilot task configuration changed");
    FrozenComparator result;std::ifstream missions(original+"_missions.jsonl");std::string row;
    gl::require(bool(missions),"missing original frozen comparator missions");
    while(std::getline(missions,row)) {
        if(row.find("\"arm\":\"frozen_policy\"")!=std::string::npos) {
            ++result.missions;result.successes+=field_bool(row,"success");
            gl::require(field_number(row,"updates_before")==12800&&field_number(row,"updates_after")==12800
                &&field_number(row,"hints")==0&&field_number(row,"cost")==0,"original frozen comparator protocol differs");
        }
        if(row.find("\"arm\":\"eval_frozen_policy300\"")!=std::string::npos) {
            ++result.cases;result.evaluation_successes+=field_bool(row,"success");
        }
    }
    std::ifstream evaluations(original+"_evaluations.jsonl");gl::require(bool(evaluations),"missing original evaluation record");size_t evaluation_rows=0;
    while(std::getline(evaluations,row))if(row.find("\"checkpoint\":\"frozen_policy300\"")!=std::string::npos) {
        ++evaluation_rows;gl::require(field_number(row,"successes")==result.evaluation_successes&&field_number(row,"cases")==result.cases,
            "original frozen comparator evaluation disagrees with missions");
    }
    std::ifstream parameters(original+"_weights.jsonl");gl::require(bool(parameters),"missing original frozen weights");
    while(std::getline(parameters,row))if(row.find("\"arm\":\"frozen_policy\"")!=std::string::npos) {
        auto start=row.find("\"weights\":[");gl::require(start!=std::string::npos,"missing original frozen vector");start+=11;
        auto stop=row.find(']',start);auto text=row.substr(start,stop-start);std::replace(text.begin(),text.end(),',',' ');
        std::istringstream values(text);std::vector<float> weights;float value;while(values>>value)weights.push_back(value);
        gl::require(weights==warm_weights,"reused frozen comparator differs from exact warm weights");++result.weight_checks;
    }
    gl::require(result.missions==200&&result.cases==12&&evaluation_rows==1&&result.weight_checks==201,"incomplete original frozen comparator");return result;
}
int main(int argc,char** argv) {
    try {
        gl::require(argc==4,"usage: goal_credit_coverage_continuation --declare|--run OUTPUT_PREFIX ORIGINAL_V1_PREFIX");
        const std::string mode=argv[1],prefix=argv[2],original=argv[3];
        if(mode=="--declare") {
            gl::require(!std::filesystem::exists(prefix+"_config.json"),"coverage declaration already exists");
            std::ofstream out(prefix+"_config.json");gl::require(bool(out),"cannot declare coverage trial");out<<continuation_configuration(original);return 0;
        }
        gl::require(mode=="--run","unknown continuation mode");std::ifstream declaration(prefix+"_config.json");
        std::string declared((std::istreambuf_iterator<char>(declaration)),std::istreambuf_iterator<char>());
        gl::require(declared==continuation_configuration(original),"coverage config missing or changed");
        for(const auto& suffix:{"_trace.jsonl","_missions.jsonl","_weights.jsonl","_evaluations.jsonl","_bootstrap.jsonl","_summary.json"})
            gl::require(!std::filesystem::exists(prefix+suffix),"refusing to overwrite coverage trial");
        for(const auto& arm:{"adaptive512","random_credit512"})for(const auto& stage:{"before","after"})for(const auto& kind:{"empirical","readouts"})
            gl::require(!std::filesystem::exists(prefix+"_"+arm+"_"+stage+"_"+kind+".jsonl"),"refusing to overwrite endpoint diagnostic");
        const auto expected_weights=saved_warm_weights(original);const auto expected=expected_evaluation(original);
        const auto reused=verify_frozen_comparator(original,expected_weights);
        Outputs output(prefix);auto fixture=gt::make_fixture();GoalControlledRelationalSession adaptive(seed,32,true);
        adaptive.knowledge=RelationalSession(seed,32768,1024);fixture.lessons.apply(adaptive.knowledge);export_bootstrap(fixture.lessons,output.bootstrap);
        gl::weights(output.parameters,adaptive,"initial","reconstruction");train(adaptive,"warm",0,100,output);
        gl::require(adaptive.policy.weights==expected_weights&&adaptive.updates()==12800,"coverage warm reconstruction mismatch");
        gl::weights(output.parameters,adaptive,"warm100_verified","shared");diagnostic_evaluate(adaptive,"warm100",false,output,&expected);
        auto random=adaptive;
        continue_with_coverage(adaptive,"adaptive512",output);auto adaptive_result=endpoint(adaptive,"adaptive512",prefix,output);
        gl::weights(output.parameters,random,"branch_start","random_credit512");
        continue_with_coverage(random,"random_credit512",output);auto random_result=endpoint(random,"random_credit512",prefix,output);
        std::ofstream summary(prefix+"_summary.json");summary<<"{\n\"schema\":\"goal_credit_coverage_result_v1\",\n\"complete\":true,\n"
            <<"\"exact_warm_reconstruction\":true,\n\"developmental_reused_suite\":true,\n\"adaptive512\":";endpoint_json(summary,adaptive_result);
        summary<<",\n\"random_credit512\":";endpoint_json(summary,random_result);
        summary<<",\n\"original_frozen_policy_comparator\":{\"sampled_successes\":"<<reused.successes<<",\"missions\":"<<reused.missions
            <<",\"neural_evaluation\":"<<reused.evaluation_successes<<",\"cases\":"<<reused.cases<<",\"verified_warm_weight_vectors\":"<<reused.weight_checks<<"},\n"
            <<"\"scope\":\"single-seed replay coverage and separate readout-fitting diagnosis\"\n}\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
