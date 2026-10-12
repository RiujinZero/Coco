// Bounded evaluator feasibility only. Preserves the existing learner in place.
#include "goal_long_environment.h"
#include <fstream>
#include <iostream>
namespace gl=laminar::goal_learning;
namespace lt=gl::long_task;
namespace gt=laminar::goal_test;
namespace gp=laminar::relational::goal_preview;
using namespace laminar::relational;

static ControlOp reference(const gl::Observation& o,bool exhaustion_feature=false) {
    if(o[29]) {
        if(!o[0]||(o[12]&&o[17]>=o[16]))return ControlOp::next_goal;
        if(!o[4]||(o[13]&&o[11]&&o[6]))return ControlOp::activate_goal;
        if(exhaustion_feature&&o[13]&&o[10]&&o[15])return ControlOp::next_goal;
        if((o[13]&&o[10])||!o[5])return ControlOp::reason;
        if((!o[13]&&(o[6]||o[9]))||(o[13]&&o[18]&&o[19]>0))return ControlOp::activate_goal;
        return ControlOp::next_goal;
    }
    if(o[30])return ControlOp::next_goal;
    if(!o[5])return ControlOp::reason;
    if(!o[4]||o[6]||o[20])return ControlOp::finish;
    if(o[9])return ControlOp::act;
    if(o[10]&&o[19]<1)return ControlOp::reason;
    return ControlOp::next_goal;
}
static void reference_smoke(GoalControlledRelationalSession& learner,lt::LongEnvironment& env,
                            unsigned seed,size_t tick,std::ostream& out,std::ostream& trace,bool exhaustion_feature=false) {
    const auto weights=learner.policy.weights;const auto updates=learner.updates();
    const auto version=learner.knowledge.version();const auto credit_count=learner.credits().size();
    const bool learning=learner.learning();auto bounds=lt::limits();bounds.agenda_search_exhaustion=exhaustion_feature;
    gl::require(learner.begin_mission("mixed_reference_"+std::to_string(seed)+"_"+std::to_string(tick),bounds,lt::prices()),"smoke mission begin");
    size_t posted=0;bool actual_interruption=false;
    auto post=[&]() {
        while(posted<env.schedule().size()&&env.schedule()[posted].clock==learner.clock()) {
            auto request=env.schedule()[posted].request;request.visible=env.read(request.world_id);
            if(posted&&!learner.active_goal().empty()) {
                const auto& active=learner.goals().at(learner.active_goal()).request;
                actual_interruption=active.effect&&!env.satisfaction(active);
            }
            gl::require(learner.offer_goal(request),"smoke request rejected");++posted;
            trace<<"{\"event\":\"request\",\"seed\":"<<seed<<",\"arrival\":"<<tick<<",\"learning\":"<<learning
                 <<",\"clock\":"<<learner.clock()<<",\"id\":"<<gl::quote(request.id)<<",\"text\":"<<gl::quote(request.text)
                 <<",\"world_id\":"<<gl::quote(request.world_id)<<",\"deadline\":"<<request.deadline<<",\"names\":";
            gl::strings(trace,request.names);trace<<",\"visible\":";gl::scene(trace,request.visible);
            trace<<",\"actions\":";gl::actions(trace,request.actions);trace<<"}\n";
        }
    };
    post();
    while(learner.clock()<bounds.decisions&&!gl::closed(learner)) {
        post();const auto observation=learner.observe();const auto menu=gl::public_menu(learner,bounds);
        const auto op=reference(observation,exhaustion_feature);const auto active=learner.active_goal(),inspected=learner.inspected_goal();
        gl::require(std::find(menu.begin(),menu.end(),int(op))!=menu.end(),"reference unavailable operation");
        // Evaluator-only forced selection is restored before any callback; no
        // reference action is passed as a training hint or policy update.
        auto saved=learner.policy.weights;std::fill(learner.policy.weights.begin(),learner.policy.weights.end(),0);
        learner.policy.weights[learner.policy.output_bias()+int(op)]=1;auto issued=learner.choose();learner.policy.weights=std::move(saved);
        gl::require(issued.operation==op,"reference choice mismatch");
        std::vector<StampedScene> observed;bool satisfaction=false,correct=false;
        if(op==ControlOp::activate_goal)gl::require(learner.activate(issued.token),"activation rejected");
        else if(op==ControlOp::act) {
            const auto& goal=learner.goals().at(active).request;std::vector<Scene> worlds;
            for(const auto& action:issued.actions){observed.push_back(env.execute(goal.world_id,action));worlds.push_back(observed.back().scene);}
            gl::require(!worlds.empty()&&learner.actions(issued.token,worlds,std::vector<bool>(worlds.size(),true),observed.back().revision),"action callback rejected");
        } else if(op==ControlOp::finish) {
            const auto& goal=learner.goals().at(active).request;satisfaction=env.satisfaction(goal);
            correct=issued.reply==(satisfaction?Answer::True:Answer::False)&&(!goal.effect||satisfaction);
            gl::require(learner.close_goal(issued.token,correct,satisfaction),"finish callback rejected");
        }
        const auto& row=learner.mission_trace().back();
        trace<<"{\"event\":\"decision\",\"seed\":"<<seed<<",\"arrival\":"<<tick<<",\"learning\":"<<learning
             <<",\"clock\":"<<row.clock<<",\"observation\":";gl::numbers(trace,observation);trace<<",\"menu\":";gl::numbers(trace,menu);
        trace<<",\"operation\":"<<int(op)<<",\"active_before\":"<<gl::quote(active)<<",\"inspected_before\":"<<gl::quote(inspected)
             <<",\"status\":"<<gl::quote(row.status)<<",\"predictions\":"<<row.predictions<<",\"primitive_steps\":"<<row.primitive_steps
             <<",\"cost\":"<<row.cost<<",\"actions\":";gl::actions(trace,row.actions);trace<<",\"observed\":[";
        for(size_t i=0;i<observed.size();++i){if(i)trace<<',';gl::scene(trace,observed[i]);}
        trace<<"],\"reply\":"<<int(row.reply)<<",\"correct\":"<<correct<<",\"satisfaction\":"<<satisfaction<<"}\n";
    }
    bool success=gl::closed(learner)&&posted==env.schedule().size();bool urgent=false;
    for(const auto& row:learner.submissions())if(row.parent.empty()) {
        success&=row.correct&&(!row.deadline||row.clock<=row.deadline);
        if(row.id=="long_urgent")urgent=row.correct&&row.clock<=row.deadline;
    }
    size_t predictions=0,work=0,proposal_work=0,global_exhaustions=0,max_allowance=0;uint64_t primitive_steps=0;double cost=0;
    for(const auto& row:learner.mission_trace()){predictions+=row.predictions;primitive_steps+=row.primitive_steps;cost+=row.cost;}
    for(const auto& row:learner.mission_computations()) {
        if(row.preview) {
            work+=row.preview->work;max_allowance=std::max(max_allowance,row.plan_allowance);
            global_exhaustions+=row.plan_allowance==4096&&row.plan.status=="search_budget";
        }
        if(row.proposals)proposal_work+=row.proposals->work;
        trace<<"{\"event\":\"paid_computation\",\"seed\":"<<seed<<",\"arrival\":"<<tick<<",\"learning\":"<<learning
             <<",\"token\":"<<row.token<<",\"candidate\":"<<gl::quote(row.candidate)<<",\"parent\":"<<gl::quote(row.parent)
             <<",\"model_before\":"<<row.model_before<<",\"model_after\":"<<row.model_after<<",\"plan_allowance\":"<<row.plan_allowance
             <<",\"global_plan_limit\":"<<bounds.preview.plan_predictions<<",\"plan_status\":"<<gl::quote(row.plan.status)
             <<",\"plan_predictions\":"<<row.plan.predictions<<",\"preview_work\":"<<(row.preview?row.preview->work:0)
             <<",\"proposal_work\":"<<(row.proposals?row.proposals->work:0)<<",\"cost\":"<<row.total_cost
             <<",\"parent_text\":"<<gl::quote(row.inputs.parent_text)<<",\"candidate_text\":"<<gl::quote(row.inputs.candidate_text)
             <<",\"names\":";gl::strings(trace,row.inputs.names);trace<<",\"world\":";gl::scene(trace,{row.inputs.world,row.inputs.observation_revision});
        trace<<",\"actions\":";gl::actions(trace,row.inputs.actions);trace<<"}\n";
    }
    gl::require(learner.finish_mission(0),"smoke final credit rejected");
    gl::require(learner.policy.weights==weights&&learner.updates()==updates,"reference changed policy");
    gl::require(learner.credits().size()==credit_count+size_t(learning),"reference credit count wrong");
    out<<"{\"seed\":"<<seed<<",\"arrival\":"<<tick<<",\"semantic_learning\":"<<learning<<",\"success\":"<<success
       <<",\"agenda_search_exhaustion\":"<<exhaustion_feature
       <<",\"urgent_on_time\":"<<urgent<<",\"actual_unfinished_interruption\":"<<actual_interruption
       <<",\"decisions\":"<<learner.clock()<<",\"motors\":"<<learner.acknowledged_motors()<<",\"predictions\":"<<predictions
       <<",\"primitive_steps\":"<<primitive_steps<<",\"preview_work\":"<<work<<",\"proposal_work\":"<<proposal_work<<",\"cost\":"<<cost
       <<",\"max_derived_allowance\":"<<max_allowance<<",\"global_cap_exhaustions\":"<<global_exhaustions
       <<",\"model_revisions\":"<<learner.knowledge.version()-version<<",\"policy_updates\":0,\"weights_unchanged\":true}";
}
int main(int argc,char** argv) {
    const std::string prefix=argc>1?argv[1]:"long_adapter",mode=argc>2?argv[2]:"semantic";
    std::ofstream evidence(prefix+"_evidence.jsonl"),trace(prefix+"_trace.jsonl"),out(prefix+"_summary.json");
    for(auto* stream:{&evidence,&trace,&out}){gl::require(bool(*stream),"output unavailable");*stream<<std::boolalpha<<std::setprecision(17);}
    GoalControlledRelationalSession learner(7,32,true);gt::make_raw_lessons().apply(learner.knowledge);
    bool late_rejected=false,frozen_rejected=false;
    try{lt::LongEnvironment late(12);}catch(const std::runtime_error&){late_rejected=true;}
    learner.set_learning(false);const auto before_rejected=learner.knowledge.version();
    try{lt::extend(learner,evidence);}catch(const std::runtime_error&){frozen_rejected=true;}
    gl::require(late_rejected&&frozen_rejected&&learner.knowledge.version()==before_rejected,"adapter boundary not enforced before mutation");
    learner.set_learning(true);
    const auto report=lt::extend(learner,evidence);const auto& model=learner.knowledge.concepts().models().at("long_route");
    out<<"{\"schema\":\"goal_long_adapter_smoke_v2\",\"policy_training\":false,\"late_arrival_rejected\":"<<late_rejected<<",\"frozen_extension_rejected\":"<<frozen_rejected<<",\"retained_library\":";gl::strings(out,report.prior_concepts);
    out<<",\"concept_status\":"<<gl::quote(model.status)<<",\"concept_work\":"<<model.work<<",\"concept_candidates\":"<<model.candidates
       <<",\"hypotheses\":"<<model.hypotheses.size()<<",\"concept_examples\":"<<report.concept_examples
       <<",\"language_examples\":"<<report.language_examples<<",\"bootstrap_motors\":"<<report.bootstrap_motors<<",\"bootstrap_predictions\":"<<report.bootstrap_predictions;
    if(model.hypotheses.size()!=1){out<<",\"stop\":\"ambiguous_mixed_library\"}\n";return 0;}
    out<<",\"local_key\":"<<gl::quote(laminar::concepts::key(model.hypotheses.front().local))
       <<",\"expanded_key\":"<<gl::quote(laminar::concepts::formula_key(laminar::concepts::expanded_clauses(model.hypotheses.front())));
    size_t long_correct=0,old_correct=0;std::vector<std::string> nodes;
    for(int i=0;i<10;++i)nodes.push_back("fresh"+std::to_string(i));
    for(uint64_t mask=0;mask<512;++mask) {
        auto world=gt::route_scene(nodes,mask);const bool truth=gt::exact_route(world,nodes.front(),nodes.back(),9);
        long_correct+=learner.knowledge.answer("connect fresh0 through fresh9",world)==(truth?Answer::True:Answer::False);
    }
    nodes.resize(7);
    for(uint64_t mask=0;mask<64;++mask) {
        auto world=gt::route_scene(nodes,mask);const bool truth=gt::exact_route(world,nodes.front(),nodes.back(),6);
        old_correct+=learner.knowledge.answer("span fresh0 toward fresh6",world)==(truth?Answer::True:Answer::False);
    }
    out<<",\"long_subset_correct\":"<<long_correct<<",\"long_subset_total\":512,\"retained_subset_correct\":"<<old_correct<<",\"retained_subset_total\":64";
    lt::LongEnvironment environment;const auto root=environment.schedule().front().request;
    size_t necessary_cases=0;
    for(int missing=0;missing<9;++missing) {
        lt::LongEnvironment check;const auto request=check.schedule().front().request;
        const Action absent{"edge_motor",{"nn"+std::to_string(missing),"nn"+std::to_string(missing+1)}};
        for(const auto& action:request.actions)if(action.arguments!=absent.arguments)check.execute(request.world_id,action);
        gl::require(!check.satisfaction(request),"distractors bypassed necessary root edge");check.execute(request.world_id,absent);
        auto renamed=request;renamed.id="arbitrary_generated_identifier";
        gl::require(check.satisfaction(renamed),"evaluator depended on generated ID");++necessary_cases;
    }
    out<<",\"necessary_edge_cases\":"<<necessary_cases<<",\"generated_id_independent\":true";
    gp::ProposalInputs input;input.parent_text=root.text;input.world=root.visible.scene;input.actions=root.actions;
    input.names={root.names.begin(),root.names.end()};input.minimum_atoms=2;
    const auto limits=lt::limits().preview;size_t useful=0,irrelevant=0,invalid=0;std::set<std::string> inventory;
    const auto proposals=gp::propose(learner.knowledge,input,limits,lt::prices().preview);
    out<<",\"source_candidates\":"<<proposals.enumerated_candidates<<",\"qualifying_candidates\":"<<proposals.total_candidates
       <<",\"retained_candidates\":"<<proposals.candidates.size()<<",\"proposal_work\":"<<proposals.work<<",\"proposal_complete\":"<<proposals.complete
       <<",\"proposal_truncated\":"<<proposals.truncated<<",\"previews\":[";
    bool first=true;
    for(const auto& candidate:proposals.candidates) {
        inventory.insert(candidate.text);gp::Inputs in;in.parent_text=root.text;in.candidate_text=candidate.text;in.world=input.world;in.actions=input.actions;in.names=input.names;
        const auto result=gp::analyze(learner.knowledge,in,limits,lt::prices().preview);
        invalid+=!result.valid;useful+=result.valid&&result.delta>0;irrelevant+=result.valid&&result.delta==0;
        if(!first)out<<',';first=false;out<<"{\"text\":"<<gl::quote(candidate.text)<<",\"valid\":"<<result.valid<<",\"status\":"<<gl::quote(result.status)
            <<",\"delta\":"<<result.delta<<",\"predictions\":"<<result.predictions<<",\"work\":"<<result.work<<'}';
    }
    size_t stable_orders=0;
    for(unsigned seed=0;seed<8;++seed){input.order_seed=seed;auto p=gp::propose(learner.knowledge,input,limits,lt::prices().preview);std::set<std::string> found;
        for(const auto& c:p.candidates)found.insert(c.text);stable_orders+=p.complete&&!p.truncated&&found==inventory;}
    out<<"],\"valid_useful\":"<<useful<<",\"valid_irrelevant\":"<<irrelevant<<",\"invalid\":"<<invalid<<",\"full_inventory_orders\":"<<stable_orders<<",\"reference_smokes\":[";
    first=true;
    if(mode=="reference")for(bool learning:{false,true})for(size_t tick:{8,10})for(unsigned seed:{0,1}) {
        GoalControlledRelationalSession continuing(seed,32,true);gt::make_raw_lessons().apply(continuing.knowledge);lt::extend(continuing,evidence);
        continuing.set_learning(learning);lt::LongEnvironment env(tick,seed,"fresh"+std::to_string(seed));
        if(!first)out<<',';first=false;reference_smoke(continuing,env,seed,tick,out,trace);out.flush();
    }
    out<<"]}\n";std::cout<<"Mixed-library adapter preflight recorded: "<<prefix<<"_summary.json\n";return 0;
}
