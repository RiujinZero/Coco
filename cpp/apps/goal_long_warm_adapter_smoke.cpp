// Reconstruct exactly the already measured warm100 state, extend it in place,
// then run evaluator-only reference probes. No new policy training curriculum.
#define main unused_protocol_pilot_main
#include "goal_control_learning_pilot.cpp"
#undef main
#define main unused_long_adapter_main
#include "goal_long_adapter_smoke.cpp"
#undef main

static std::vector<float> recorded_weights(const std::string& prefix) {
    std::ifstream input(prefix+"_weights.jsonl");gl::require(bool(input),"missing saved warm weights");
    std::string row;std::vector<float> found;size_t matches=0;
    while(std::getline(input,row))if(row.find("\"label\":\"warm100\"")!=std::string::npos) {
        ++matches;auto begin=row.find("\"weights\":[");gl::require(begin!=std::string::npos,"missing weights field");begin+=11;
        auto end=row.find(']',begin);auto values=row.substr(begin,end-begin);std::replace(values.begin(),values.end(),',',' ');
        std::istringstream numbers(values);float value;while(numbers>>value)found.push_back(value);
    }
    gl::require(matches==1&&!found.empty(),"nonunique warm checkpoint");return found;
}
int main(int argc,char** argv) {
    gl::require(argc==3||argc==4,"usage: goal_long_warm_adapter_smoke OUTPUT_PREFIX ORIGINAL_PILOT_PREFIX [AGENDA_SEARCH_EXHAUSTION]");
    const std::string prefix=argv[1],original=argv[2];
    const bool exhaustion_feature=argc==4&&std::string(argv[3])=="1";
    gl::require(argc==3||std::string(argv[3])=="0"||exhaustion_feature,"invalid observation variant");
    const auto expected=recorded_weights(original);Outputs reconstruction(prefix+"_reconstruction");
    GoalControlledRelationalSession warm(1,32,true);warm.knowledge=RelationalSession(1,32768,1024);
    gt::make_raw_lessons().apply(warm.knowledge);train(warm,"warm",0,100,reconstruction);
    gl::require(warm.policy.weights==expected&&warm.updates()==12800&&warm.credits().size()==100,"warm checkpoint reconstruction drift");
    const auto warm_weights=warm.policy.weights;const bool dirty_before=warm.knowledge.dirty();
    // Rebuild only the existing pending semantic receipts, before extension;
    // this neither supplies additional evidence nor changes the policy.
    if(dirty_before)warm.knowledge.rebuild();
    const auto warm_procedures=warm.knowledge.procedures().size();
    std::ofstream evidence(prefix+"_evidence.jsonl"),out(prefix+"_summary.json"),trace(prefix+"_trace.jsonl");
    for(auto* stream:{&evidence,&out,&trace}){gl::require(bool(*stream),"warm smoke output unavailable");*stream<<std::boolalpha<<std::setprecision(17);}
    const auto report=lt::extend(warm,evidence,"warm_long_extension");
    gl::require(warm.policy.weights==warm_weights&&warm.updates()==12800&&warm.credits().size()==100,"extension lost continuing state");
    const auto& model=warm.knowledge.concepts().models().at("long_route");
    out<<"{\"schema\":\"goal_long_warm_adapter_smoke_v1\",\"original_prefix\":"<<gl::quote(original)
       <<",\"agenda_search_exhaustion\":"<<exhaustion_feature
       <<",\"warm_checkpoint_exact\":true,\"warm_policy_updates\":12800,\"warm_credits\":100,\"new_policy_updates\":0"
       <<",\"warm_semantics_dirty_before_rebuild\":"<<dirty_before<<",\"warm_procedures\":"<<warm_procedures
       <<",\"extended_procedures\":"<<report.procedures_after<<",\"bootstrap_motors\":"<<report.bootstrap_motors
       <<",\"bootstrap_predictions\":"<<report.bootstrap_predictions<<",\"concept_status\":"<<gl::quote(model.status)
       <<",\"concept_hypotheses\":"<<model.hypotheses.size();
    if(model.hypotheses.size()!=1){out<<",\"stop\":\"ambiguous_actual_warm_extension\"}\n";return 0;}
    size_t long_correct=0,prior_correct=0;std::vector<std::string> nodes;
    for(int i=0;i<10;++i)nodes.push_back("withheld"+std::to_string(i));
    for(uint64_t mask=0;mask<512;++mask) {
        const auto world=gt::route_scene(nodes,mask);const auto truth=gt::exact_route(world,nodes.front(),nodes.back(),9);
        long_correct+=warm.knowledge.answer("connect withheld0 through withheld9",world)==(truth?Answer::True:Answer::False);
    }
    nodes.resize(7);
    for(uint64_t mask=0;mask<64;++mask) {
        const auto world=gt::route_scene(nodes,mask);const auto truth=gt::exact_route(world,nodes.front(),nodes.back(),6);
        prior_correct+=warm.knowledge.answer("span withheld0 toward withheld6",world)==(truth?Answer::True:Answer::False);
    }
    out<<",\"long_subset_correct\":"<<long_correct<<",\"long_subset_total\":512,\"retained_subset_correct\":"<<prior_correct<<",\"retained_subset_total\":64,\"previews\":[";
    lt::LongEnvironment env;const auto request=env.schedule().front().request;
    gp::ProposalInputs input;input.parent_text=request.text;input.world=request.visible.scene;input.actions=request.actions;
    input.names={request.names.begin(),request.names.end()};input.minimum_atoms=2;
    const auto proposals=gp::propose(warm.knowledge,input,lt::limits().preview,lt::prices().preview);
    bool first=true;size_t useful=0,irrelevant=0,invalid=0;
    for(const auto& candidate:proposals.candidates) {
        gp::Inputs in;in.parent_text=input.parent_text;in.candidate_text=candidate.text;in.world=input.world;in.actions=input.actions;in.names=input.names;
        const auto result=gp::analyze(warm.knowledge,in,lt::limits().preview,lt::prices().preview);
        useful+=result.valid&&result.delta>0;irrelevant+=result.valid&&result.delta==0;invalid+=!result.valid;
        if(!first)out<<',';first=false;out<<"{\"text\":"<<gl::quote(candidate.text)<<",\"valid\":"<<result.valid
            <<",\"status\":"<<gl::quote(result.status)<<",\"delta\":"<<result.delta<<",\"predictions\":"<<result.predictions
            <<",\"work\":"<<result.work<<",\"reused\":"<<result.plan.reused<<'}';
    }
    out<<"],\"source_candidates\":"<<proposals.enumerated_candidates<<",\"qualifying_candidates\":"<<proposals.total_candidates
       <<",\"retained_candidates\":"<<proposals.candidates.size()<<",\"proposal_work\":"<<proposals.work<<",\"proposal_complete\":"<<proposals.complete
       <<",\"proposal_truncated\":"<<proposals.truncated<<",\"valid_useful\":"<<useful<<",\"valid_irrelevant\":"<<irrelevant<<",\"invalid\":"<<invalid
       <<",\"reference_smokes\":[";first=true;
    for(bool learning:{false,true})for(size_t tick:{8,10})for(unsigned order:{0,1}) {
        auto continuing=warm;continuing.set_learning(learning);lt::LongEnvironment environment(tick,order,"withheld"+std::to_string(order));
        if(!first)out<<',';first=false;reference_smoke(continuing,environment,order,tick,out,trace,exhaustion_feature);out.flush();
    }
    out<<"]}\n";std::cout<<"Actual warm adapter feasibility recorded: "<<prefix<<"_summary.json\n";
}
