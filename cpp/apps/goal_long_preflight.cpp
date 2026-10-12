// Evaluator preflight only. No training labels, schedules or strategy traces
// are delivered to the controller. Failed gates remain in the output record.
#include "../tests/goal_control_fixture.h"
#include "laminar/relational_goal_preview.h"
#include "laminar/goal_control.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace gp=laminar::relational::goal_preview;
namespace gt=laminar::goal_test;
using namespace laminar::relational;

gt::RawLessons long_lessons() {
    auto lessons=gt::make_raw_lessons();
    // Keep the one-edge and three-edge learned library. A six-edge concept
    // would confound the requested three-call reuse check by offering a
    // shorter two-call expression for the same nine-edge relation.
    std::erase_if(lessons.concepts,[](const auto& r){return r.name=="parent_route";});
    std::erase_if(lessons.language,[](const auto& r){return r.text.starts_with("span ");});
    std::vector<std::string> nodes;
    for(int i=0;i<10;++i)nodes.push_back("teach"+std::to_string(i));
    auto add=[&](const Scene& world) {
        auto id="long_lesson_"+std::to_string(lessons.concepts.size());
        bool answer=gt::exact_route(world,nodes.front(),nodes.back(),9);
        laminar::concepts::Receipt row;row.id=id;row.name="long_route";row.scene=world;
        row.args={nodes.front(),nodes.back()};row.answer=answer;lessons.concepts.push_back(row);
        LanguageReceipt language;language.id=id;language.text="connect "+nodes.front()+" through "+nodes.back();
        language.names={nodes.front(),nodes.back()};language.scene=world;language.answer=answer;
        lessons.language.push_back(std::move(language));
    };
    add(gt::route_scene(nodes));
    for(int missing=0;missing<9;++missing)add(gt::route_scene(nodes,511u^(1u<<missing)));
    add(gt::route_scene({nodes.front(),"m1","m2",nodes.back()}));
    add(gt::route_scene({nodes.front(),"m1","m2","m3","m4","m5",nodes.back()}));
    return lessons;
}

struct PublicCase {
    std::string text="connect n0 through n9";
    Scene world;
    std::vector<std::string> names;
    std::vector<Action> actions;
};
PublicCase public_case(bool wide=false,bool adjacency=false) {
    PublicCase c;
    for(int i=0;i<10;++i)c.names.push_back("n"+std::to_string(i));
    for(int i=0;i<4;++i)c.names.push_back("d"+std::to_string(i));
    for(int i=0;i<9;++i)c.actions.push_back({"edge_motor",{"n"+std::to_string(i),"n"+std::to_string(i+1)}});
    for(int i=0;i<3;++i)c.actions.push_back({"edge_motor",{"d"+std::to_string(i),"d"+std::to_string(i+1)}});
    if(wide) {
        // Phase1 declared before measurement: nine root edges, a disconnected
        // three-edge chain, and28 directed bipartite additions. No distractor
        // can create any root walk or an extra nine-edge derived parent route.
        for(int i=0;i<9;++i){c.names.push_back("s"+std::to_string(i));c.names.push_back("t"+std::to_string(i));}
        for(int i=0;i<28;++i)c.actions.push_back({"edge_motor",{"s"+std::to_string(i%9),"t"+std::to_string((i/9+i)%9)}});
    }
    if(adjacency)for(const auto& action:c.actions)c.world.facts.push_back({"affordance",action.arguments});
    return c;
}
Scene physical(Scene world,const Action& action) {
    gt::require(action.schema=="edge_motor"&&action.arguments.size()==2,"unknown evaluator motor");
    world.facts.push_back({"edge",action.arguments});return sorted(std::move(world));
}
bool truth(const Scene& world){return gt::exact_route(world,"n0","n9",9);}

struct Bootstrap {
    Scene before;
    std::vector<Action> actions;
    std::vector<Scene> observed;
    size_t predictions=0;
};
Bootstrap learn_short_procedure(RelationalSession& session) {
    Bootstrap out;
    out.actions={{"edge_motor",{"a","b"}},{"edge_motor",{"b","c"}},{"edge_motor",{"c","d"}}};
    for(const auto& action:out.actions)out.before.facts.push_back({"affordance",action.arguments});
    // The ordinary learned planner chooses its actions. The external evaluator
    // supplies actual observations and one final Boolean outcome, never code.
    auto issued=session.issue("route a toward d",out.before,out.actions,false,4096,8);
    gt::require(issued.actions.size()==3,"short-route bootstrap plan failed");
    out.predictions=issued.predictions;out.actions=issued.actions;auto world=out.before;
    for(const auto& action:issued.actions){world=physical(std::move(world),action);out.observed.push_back(world);}
    gt::require(gt::exact_route(world,"a","d",3),"independent bootstrap execution failed");
    gt::require(session.feedback(issued.token,out.observed,{true,true,true},true),"raw own-action bootstrap receipt rejected");
    session.rebuild();gt::require(session.procedures().size()==1,"short-route procedure not acquired from own execution");
    return out;
}

gp::ProposalResult proposals(const RelationalSession& session,const gp::ProposalInputs& input,
                             gp::Limits limits,const gp::Prices& prices,bool macro_only) {
    auto selected=input;selected.minimum_atoms=macro_only?2:1;
    return gp::propose(session,selected,limits,prices);
}

// Evaluation-only reference. Its decision function receives only the declared
//32 public floats. Goal identities and private truth are used by callbacks,
// never by this function. Temporary output preferences are restored immediately;
// no reference action or successful schedule becomes a training receipt.
ControlOp reference_choice(const laminar::reasoning::Observation& o,int direct,int reference_variant=0) {
    if(o[29]) {
        if(!o[0])return ControlOp::next_goal;
        if(o[12]&&o[17]>=o[16])return ControlOp::next_goal;
        if(!o[4])return ControlOp::activate_goal;
        if(direct)return ControlOp::activate_goal;
        if(reference_variant&&o[13]&&o[11]&&o[6])return ControlOp::activate_goal;
        if(reference_variant&&o[13]&&o[10])return ControlOp::reason;
        if(!o[5])return ControlOp::reason;
        if(!o[13]&&(o[6]||o[9]))return ControlOp::activate_goal;
        if(o[13]&&o[18]&&o[19]>0)return ControlOp::activate_goal;
        return ControlOp::next_goal;
    }
    if(o[30])return ControlOp::next_goal;
    if(!o[5])return ControlOp::reason;
    if(!o[4]||o[6]||o[20])return ControlOp::finish;
    if(o[9])return ControlOp::act;
    if(direct==1)return ControlOp::act;
    if(o[10]&&o[19]<(direct==2?12.f/15.f:1.f))return ControlOp::reason;
    return direct?ControlOp::act:ControlOp::next_goal;
}
struct ProtocolResult {
    bool success=false,closed=false,urgent_on_time=false,unchanged=true;
    bool semantic_learning=false,arrival_active_unfinished=false,knowledge_unchanged=true;
    unsigned seed=0;size_t arrival_tick=0,semantic_revisions=0,policy_updates=0,credit_receipts=0;
    size_t decision_limit=96;double price_scale=1;
    int reference_variant=0;
    size_t global_preview_exhaustions=0,max_derived_preview_allowance=0;
    size_t motors=0,decisions=0,predictions=0,computations=0;
    double total_cost=0,discounted_cost=0,utility=0;
    std::string status;
    std::vector<GoalControlStep> trace;
    std::vector<GoalComputationReceipt> computation_receipts;
    std::vector<std::pair<size_t,PublicGoal>> public_requests;
    std::vector<GoalSubmission> submissions;
    struct Binding {std::string id,parent,text,world;size_t arrival=0,deadline=0;bool effect=false;};
    std::vector<Binding> goal_bindings;
};
ProtocolResult protocol_reference(const RelationalSession& knowledge,const PublicCase& c,unsigned seed,
                                  int direct=0,bool urgent=true,size_t arrival_tick=12,bool semantic_learning=false,
                                  size_t decision_limit=96,double price_scale=1,int reference_variant=0) {
    GoalControlledRelationalSession agent(seed,32,semantic_learning);
    // The semantic lessons are fixed, but the existing exploration RNG belongs
    // to RelationalSession. Rebuild the same knowledge with the declared seed;
    // cloning one warmed session would repeat its action randomness in every arm.
    RelationalSession seeded(seed);long_lessons().apply(seeded);learn_short_procedure(seeded);
    gt::require(laminar::concepts::formula_key(laminar::concepts::expanded_clauses(seeded.concepts().models().at("long_route").hypotheses.front()))==
        laminar::concepts::formula_key(laminar::concepts::expanded_clauses(knowledge.concepts().models().at("long_route").hypotheses.front())),
        "seed altered the fixed acquired root meaning");
    agent.knowledge=std::move(seeded);
    auto weights=agent.policy.weights;auto version=agent.knowledge.version();
    GoalControlLimits limits;limits.derived=direct?0:8;limits.proposal_min_atoms=2;
    limits.decisions=decision_limit;
    limits.preview.normalization_atoms=9;limits.preview.work=2000000;limits.preview.plan_predictions=4096;
    GoalControlPrices prices;
    for(auto& value:prices.execution.operation)value*=float(price_scale);
    prices.execution.prediction*=float(price_scale);prices.execution.motor*=float(price_scale);prices.execution.rebuild*=float(price_scale);
    prices.next_goal*=float(price_scale);prices.activate_goal*=float(price_scale);
    prices.preview.prediction*=price_scale;prices.preview.grounding*=price_scale;prices.preview.atom_check*=price_scale;
    prices.preview.answer_call*=price_scale;prices.preview.planner_call*=price_scale;
    gt::require(agent.begin_mission("reference",limits,prices),"reference mission rejected");
    PublicGoal root;root.id="root";root.world_id="public_world";root.text=c.text;
    root.names={c.names.begin(),c.names.end()};root.visible={c.world,1};root.actions=c.actions;root.effect=true;root.deadline=decision_limit;
    gt::require(agent.offer_goal(root),"reference root request rejected");
    StampedScene evaluator{c.world,1};bool posted=false;ProtocolResult result;
    result.seed=seed;result.arrival_tick=arrival_tick;result.semantic_learning=semantic_learning;
    result.decision_limit=decision_limit;result.price_scale=price_scale;
    result.reference_variant=reference_variant;
    result.public_requests.emplace_back(0,root);
    auto evaluate=[&](const PublicGoal& goal) {
        auto tokens=words(goal.text);
        size_t length=tokens.at(0)=="connect"?9:tokens.at(0)=="route"?3:1;
        return gt::exact_route(evaluator.scene,tokens.at(1),tokens.at(3),length);
    };
    try {
        while(agent.mission_active()&&agent.clock()<limits.decisions) {
            auto observation=agent.observe();auto operation=reference_choice(observation,direct,reference_variant);
            auto saved=agent.policy.weights;std::fill(agent.policy.weights.begin(),agent.policy.weights.end(),0);
            agent.policy.weights[agent.policy.output_bias()+int(operation)]=1;
            auto step=agent.choose();agent.policy.weights=std::move(saved);
            gt::require(step.operation==operation&&step.observation==observation,"reference operation not legal or feature snapshot changed");
            if(step.operation==ControlOp::activate_goal)gt::require(agent.activate(step.token),"reference activation rejected");
            else if(step.operation==ControlOp::act) {
                if(step.actions.empty())throw std::runtime_error("reference act unavailable");
                std::vector<Scene> observations;
                for(const auto& action:step.actions){evaluator.scene=physical(std::move(evaluator.scene),action);observations.push_back(evaluator.scene);++evaluator.revision;}
                gt::require(agent.actions(step.token,observations,std::vector<bool>(observations.size(),true),evaluator.revision),"reference motor callback rejected");
            } else if(step.operation==ControlOp::finish) {
                const auto& request=agent.goals().at(agent.active_goal()).request;
                bool satisfaction=evaluate(request),correct=request.effect?satisfaction:
                    step.reply==(satisfaction?Answer::True:Answer::False);
                gt::require(agent.close_goal(step.token,correct,satisfaction),"reference terminal callback rejected");
                if(agent.finish_mission(0)){result.closed=true;break;}
            }
            if(urgent&&!posted&&agent.clock()>=arrival_tick) {
                result.arrival_active_unfinished=!agent.active_goal().empty()&&agent.goals().at(agent.active_goal()).request.effect&&
                    !evaluate(agent.goals().at(agent.active_goal()).request);
                PublicGoal arrival=root;arrival.id="arrival";arrival.text="inspect n0 toward n1";
                arrival.visible=evaluator;arrival.actions.clear();arrival.effect=false;arrival.deadline=agent.clock()+16;
                gt::require(agent.offer_goal(arrival),"public urgent arrival rejected");posted=true;
                result.public_requests.emplace_back(agent.clock(),std::move(arrival));
            }
        }
        if(agent.mission_active()&&agent.clock()==limits.decisions)result.closed=agent.finish_mission(0);
        result.success=result.closed;size_t roots=0;
        for(const auto& row:agent.submissions())if(row.parent.empty()) {
            ++roots;result.success&=row.correct&&(!row.deadline||row.clock<=row.deadline);
            if(row.id=="arrival")result.urgent_on_time=row.correct&&row.clock<=row.deadline;
        }
        result.success&=roots==size_t(urgent?2:1);result.status=result.success?"success":"bounded_failure";
    }catch(const std::exception& e){result.status=e.what();}
    result.trace=agent.mission_trace();result.decisions=result.trace.size();result.motors=agent.acknowledged_motors();
    result.computations=agent.mission_computations().size();
    result.computation_receipts=agent.mission_computations();
    for(const auto& row:result.computation_receipts)if(row.preview) {
        result.max_derived_preview_allowance=std::max(result.max_derived_preview_allowance,row.limits.plan_predictions);
        result.global_preview_exhaustions+=row.limits.plan_predictions==4096&&row.plan.status=="search_budget";
    }
    result.submissions=agent.submissions();
    for(const auto& [id,e]:agent.goals())result.goal_bindings.push_back({id,e.parent,e.request.text,e.request.world_id,e.arrival,e.request.deadline,e.request.effect});
    double discount=1;
    for(const auto& step:result.trace) {
        result.total_cost+=step.cost;result.discounted_cost+=discount*step.cost;discount*=.99;result.predictions+=step.predictions;
        gt::require(reference_choice(step.observation,direct,reference_variant)==step.operation,"reference decision used hidden identity or evaluator state");
    }
    result.utility=std::pow(.99,double(result.decisions?result.decisions-1:0))*(result.success?1:-1)-result.discounted_cost;
    result.knowledge_unchanged=agent.knowledge.version()==version;result.semantic_revisions=agent.knowledge.version()-version;
    result.policy_updates=agent.updates();result.credit_receipts=agent.credits().size();
    result.unchanged=agent.policy.weights==weights&&agent.updates()==0;
    return result;
}
void strings(std::ostream& out,const std::vector<std::string>& values) {
    out<<'[';for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<std::quoted(values[i]);}out<<']';
}
void atoms(std::ostream& out,const std::vector<Atom>& values) {
    out<<'[';for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<'['<<std::quoted(values[i].predicate)<<',';strings(out,values[i].args);out<<']';}out<<']';
}
void scene(std::ostream& out,const Scene& value) {
    out<<"{\"complete\":"<<(value.complete?"true":"false")<<",\"facts\":";atoms(out,value.facts);out<<'}';
}
void actions(std::ostream& out,const std::vector<Action>& values) {
    out<<'[';for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<'['<<std::quoted(values[i].schema)<<',';strings(out,values[i].arguments);out<<']';}out<<']';
}
void computation(std::ostream& out,const GoalComputationReceipt& row) {
    out<<"{\"token\":"<<row.token<<",\"candidate\":"<<std::quoted(row.candidate)<<",\"parent\":"<<std::quoted(row.parent)
       <<",\"model_before\":"<<row.model_before<<",\"model_after\":"<<row.model_after<<",\"parent_text\":"<<std::quoted(row.inputs.parent_text)
       <<",\"candidate_text\":"<<std::quoted(row.inputs.candidate_text)<<",\"world\":";scene(out,row.inputs.world);
    out<<",\"actions\":";actions(out,row.inputs.actions);out<<",\"names\":";strings(out,row.inputs.names);
    out<<",\"revisions\":["<<row.inputs.observation_revision<<','<<row.inputs.parent_revision<<','<<row.inputs.candidate_revision<<','<<row.inputs.affordance_revision
       <<"],\"answer\":"<<int(row.answer)<<",\"plan_status\":"<<std::quoted(row.plan.status)<<",\"plan_actions\":";actions(out,row.plan.actions);
    out<<",\"plan_predictions\":"<<row.plan.predictions<<",\"preview_prediction_limit\":"<<row.limits.plan_predictions
       <<",\"preview_work_limit\":"<<row.limits.work<<",\"preview_plan_depth\":"<<row.limits.plan_depth
       <<",\"normalization_atoms\":"<<row.limits.normalization_atoms<<",\"cost_components\":["<<row.operation_cost<<','<<row.rebuild_cost<<','<<row.planning_cost<<','<<row.primitive_cost
       <<"],\"total_cost\":"<<row.total_cost;
    if(row.preview) {
        const auto& p=*row.preview;
        out<<",\"preview\":{\"valid\":"<<(p.valid?"true":"false")<<",\"status\":"<<std::quoted(p.status)
           <<",\"parent_definition\":"<<std::quoted(laminar::concepts::formula_key(p.parent.formula))
           <<",\"candidate_definition\":"<<std::quoted(laminar::concepts::formula_key(p.candidate.formula))
           <<",\"parent_args\":";strings(out,p.parent.args);out<<",\"candidate_args\":";strings(out,p.candidate.args);
        out<<",\"before_missing\":"<<p.before_missing<<",\"after_missing\":"<<p.after_missing<<",\"delta\":"<<p.delta
           <<",\"normalized_delta\":"<<p.normalized_delta<<",\"counts\":["<<p.predictions<<','<<p.groundings<<','<<p.atom_checks<<','<<p.answer_calls<<','<<p.planner_calls
           <<"],\"work\":"<<p.work<<",\"cost\":"<<p.cost<<",\"predicted\":";scene(out,p.predicted);
        out<<",\"potential\":";atoms(out,p.potential);out<<'}';
    }
    if(row.proposals) {
        const auto& p=*row.proposals;
        out<<",\"proposals\":{\"complete\":"<<(p.complete?"true":"false")<<",\"truncated\":"<<(p.truncated?"true":"false")
           <<",\"enumerated\":"<<p.enumerated_candidates<<",\"qualifying\":"<<p.total_candidates<<",\"minimum_atoms\":"<<p.inputs.minimum_atoms
           <<",\"seed\":"<<p.inputs.order_seed<<",\"capacity\":"<<p.limits.candidates<<",\"work\":"<<p.work<<",\"cost\":"<<p.cost<<",\"candidates\":[";
        for(size_t i=0;i<p.candidates.size();++i) {
            if(i)out<<',';const auto& c=p.candidates[i];out<<"{\"text\":"<<std::quoted(c.text)<<",\"predicate\":"<<std::quoted(c.predicate)
               <<",\"definition\":"<<std::quoted(laminar::concepts::formula_key(c.definition))<<",\"args\":";strings(out,c.args);
            out<<",\"witness\":";atoms(out,c.witness);out<<'}';
        }
        out<<"]}";
    }
    out<<'}';
}
void print_protocol(std::ostream& out,const ProtocolResult& result,bool full=false) {
    out<<"{\"seed\":"<<result.seed<<",\"decision_limit\":"<<result.decision_limit<<",\"price_scale\":"<<result.price_scale
       <<",\"reference_variant\":"<<result.reference_variant<<",\"global_preview_exhaustions\":"<<result.global_preview_exhaustions
       <<",\"max_derived_preview_allowance\":"<<result.max_derived_preview_allowance
       <<",\"arrival_tick\":"<<result.arrival_tick<<",\"semantic_learning\":"<<(result.semantic_learning?"true":"false")
       <<",\"arrival_active_unfinished\":"<<(result.arrival_active_unfinished?"true":"false")
       <<",\"knowledge_unchanged\":"<<(result.knowledge_unchanged?"true":"false")<<",\"semantic_revisions\":"<<result.semantic_revisions
       <<",\"policy_updates\":"<<result.policy_updates<<",\"credit_receipts\":"<<result.credit_receipts
       <<",\"success\": "<<(result.success?"true":"false")<<", \"status\": \""<<result.status
       <<"\", \"unchanged_policy\": "<<(result.unchanged?"true":"false")<<", \"urgent_on_time\": "
       <<(result.urgent_on_time?"true":"false")<<", \"decisions\": "<<result.decisions<<", \"motors\": "<<result.motors
       <<", \"predictions\": "<<result.predictions<<", \"paid_agenda_computations\": "<<result.computations
       <<", \"cost\": "<<result.total_cost<<", \"discounted_cost\": "<<result.discounted_cost<<", \"utility\": "<<result.utility;
    if(full) {
        out<<",\"public_requests\":[";
        for(size_t i=0;i<result.public_requests.size();++i) {
            if(i)out<<',';const auto& [clock,r]=result.public_requests[i];
            out<<"{\"clock\":"<<clock<<",\"id\":"<<std::quoted(r.id)<<",\"world_id\":"<<std::quoted(r.world_id)
               <<",\"text\":"<<std::quoted(r.text)<<",\"effect\":"<<(r.effect?"true":"false")<<",\"deadline\":"<<r.deadline
               <<",\"visible_revision\":"<<r.visible.revision<<",\"visible\":";scene(out,r.visible.scene);
            out<<",\"names\":";strings(out,{r.names.begin(),r.names.end()});out<<",\"actions\":";actions(out,r.actions);out<<'}';
        }
        out<<"],\"goal_bindings\":[";
        for(size_t i=0;i<result.goal_bindings.size();++i) {
            if(i)out<<',';const auto& g=result.goal_bindings[i];out<<"{\"id\":"<<std::quoted(g.id)<<",\"parent\":"<<std::quoted(g.parent)
               <<",\"text\":"<<std::quoted(g.text)<<",\"world_id\":"<<std::quoted(g.world)<<",\"arrival\":"<<g.arrival<<",\"deadline\":"<<g.deadline
               <<",\"effect\":"<<(g.effect?"true":"false")<<'}';
        }
        out<<"],\"submissions\":[";
        for(size_t i=0;i<result.submissions.size();++i) {
            if(i)out<<',';const auto& r=result.submissions[i];out<<"{\"id\":"<<std::quoted(r.id)<<",\"parent\":"<<std::quoted(r.parent)
                <<",\"clock\":"<<r.clock<<",\"deadline\":"<<r.deadline<<",\"correct\":"<<(r.correct?"true":"false")
                <<",\"satisfaction\":"<<(r.satisfaction?"true":"false")<<'}';
        }
        out<<']';
        out<<", \"trace\": [\n";
        for(size_t i=0;i<result.trace.size();++i) {
            const auto& row=result.trace[i];out<<"{\"clock\": "<<row.clock<<", \"operation\": "<<int(row.operation)
                <<", \"goal\": \""<<row.goal<<"\", \"inspected\": \""<<row.inspected<<"\", \"cost\": "<<row.cost
                <<",\"token\":"<<row.token<<",\"world_id\":"<<std::quoted(row.world_id)<<",\"reply\":"<<int(row.reply)
                <<",\"status\":"<<std::quoted(row.status)<<",\"rejected\":"<<(row.rejected?"true":"false")
                <<", \"predictions\": "<<row.predictions<<", \"features\": [";
            for(size_t n=0;n<row.observation.size();++n){if(n)out<<',';out<<row.observation[n];}
            out<<"],\"menu\":[";for(size_t n=0;n<row.available.size();++n){if(n)out<<',';out<<row.available[n];}
            out<<"],\"actions\":";actions(out,row.actions);out<<'}'<<(i+1==result.trace.size()?"":",")<<'\n';
        }
        out<<"],\"computations\":[\n";
        for(size_t i=0;i<result.computation_receipts.size();++i){computation(out,result.computation_receipts[i]);out<<(i+1==result.computation_receipts.size()?"":",")<<'\n';}
        out<<']';
    }
    out<<'}';
}

int main(int argc,char** argv) {
    const std::string mode=argc>=3?argv[2]:"narrow";
    const size_t decision_limit=argc>=4?std::stoul(argv[3]):96;
    const double price_scale=argc>=5?std::stod(argv[4]):1;
    const int reference_variant=argc>=6?std::stoi(argv[5]):0;
    gt::require(decision_limit>=1&&decision_limit<=256&&std::isfinite(price_scale)&&price_scale>=0&&price_scale<=1,
        "invalid explicit decision limit or price scale");
    gt::require(reference_variant==0||reference_variant==1,"invalid reference variant");
    const bool protocol=mode=="protocol",procedural=mode=="procedure"||protocol,macro_only=mode=="macro"||procedural,wide=mode=="wide"||macro_only;
    std::ofstream file;if(argc>=2)file.open(argv[1]);std::ostream& out=argc>=2?file:std::cout;
    out<<std::setprecision(17);
    gt::require(bool(out),"cannot open preflight artifact");
    auto lessons=long_lessons();RelationalSession session(7);lessons.apply(session);
    Bootstrap bootstrap;if(procedural)bootstrap=learn_short_procedure(session);
    const auto& model=session.concepts().models().at("long_route");
    out<<"{\n  \"schema\": \"goal_long_preflight_v1\",\n  \"training_run\": false,\n"
       <<"  \"family\": \""<<(procedural?"phase3_raw_execution_procedure_bootstrap":macro_only?"phase2_macro_only_eight_cap":wide?"phase1_forty_affordances":"phase0_twelve_affordances")<<"\",\n"
       <<"  \"own_execution_bootstrap\": "<<(procedural?"true":"false")<<",\n"
       <<"  \"bootstrap_predictions\": "<<bootstrap.predictions<<",\n  \"bootstrap_actions\": "<<bootstrap.actions.size()<<",\n"
       <<"  \"seed\": 7,\n  \"necessary_edges\": 9,\n  \"concept_status\": \""<<model.status<<"\",\n"
       <<"  \"concept_work\": "<<model.work<<",\n  \"concept_candidates\": "<<model.candidates<<",\n";
    if(model.hypotheses.size()!=1) {
        out<<"  \"representable\": false,\n  \"stop\": \"long_concept_not_uniquely_acquired\"\n}\n";return 0;
    }
    const auto& h=model.hypotheses.front();auto branches=laminar::concepts::expanded_clauses(h);
    size_t local_calls=0;for(const auto& literal:h.local.body)local_calls+=literal.predicate=="short_route";
    bool exact_shape=h.local.body.size()==3&&local_calls==3&&branches.size()==1&&branches[0].body.size()==9&&
        laminar::concepts::variables(branches[0])==10;
    // Independent syntactic equivalence check: expanded atoms must form exactly
    // one directed nine-edge chain from head0 to head1, up to variable names.
    std::map<int,int> successor;std::set<int> targets;
    for(const auto& literal:branches[0].body) {
        if(literal.predicate!="edge"||literal.vars.size()!=2||!successor.emplace(literal.vars[0],literal.vars[1]).second||
           !targets.insert(literal.vars[1]).second)exact_shape=false;
    }
    int cursor=0;std::set<int> visited;
    for(int step=0;step<9;++step) {
        if(!visited.insert(cursor).second||!successor.count(cursor)){exact_shape=false;break;}
        cursor=successor.at(cursor);
    }
    exact_shape&=cursor==1&&visited.size()==9&&!successor.count(1);
    auto c=public_case(wide,procedural);size_t semantic_cases=0,semantic_correct=0;
    if(protocol) {
        GoalControlPrices declared_prices;
        for(auto& value:declared_prices.execution.operation)value*=float(price_scale);
        declared_prices.execution.prediction*=float(price_scale);declared_prices.execution.motor*=float(price_scale);declared_prices.execution.rebuild*=float(price_scale);
        declared_prices.next_goal*=float(price_scale);declared_prices.activate_goal*=float(price_scale);
        declared_prices.preview.prediction*=price_scale;declared_prices.preview.grounding*=price_scale;declared_prices.preview.atom_check*=price_scale;
        declared_prices.preview.answer_call*=price_scale;declared_prices.preview.planner_call*=price_scale;
        out<<"  \"decision_limit\": "<<decision_limit<<",\n  \"price_scale\": "<<price_scale
           <<",\n  \"reference_variant\": "<<reference_variant
           <<",\n  \"preview_planner_cap\": 4096,\n  \"physical_planner_cap\": 32768,\n"
           <<"  \"operation_prices\": [";
        for(size_t i=0;i<declared_prices.execution.operation.size();++i){if(i)out<<',';out<<declared_prices.execution.operation[i];}
        out<<','<<declared_prices.next_goal<<','<<declared_prices.activate_goal<<"],\n  \"motor_price\": "<<declared_prices.execution.motor
           <<",\n  \"execution_prediction_price\": "<<declared_prices.execution.prediction<<",\n  \"rebuild_price\": "<<declared_prices.execution.rebuild
           <<",\n  \"preview_prices\": ["<<declared_prices.preview.prediction<<','<<declared_prices.preview.grounding<<','<<declared_prices.preview.atom_check
           <<','<<declared_prices.preview.answer_call<<','<<declared_prices.preview.planner_call<<"],\n  \"references\": [\n";
        size_t emitted=0;
        for(size_t tick:{8,12,18})for(unsigned seed=0;seed<8;++seed){if(emitted++)out<<",\n";auto result=protocol_reference(session,c,seed,0,true,tick,false,decision_limit,price_scale,reference_variant);print_protocol(out,result,true);}
        out<<"  ],\n  \"semantic_learning_references\": [\n";emitted=0;
        for(size_t tick:{8,12,18})for(unsigned seed=0;seed<8;++seed){if(emitted++)out<<",\n";auto result=protocol_reference(session,c,seed,0,true,tick,true,decision_limit,price_scale,reference_variant);print_protocol(out,result,true);}
        out<<"  ],\n  \"actual_direct_uniform\": [\n";
        for(unsigned seed=0;seed<8;++seed){auto result=protocol_reference(session,c,seed,1,true,12,false,decision_limit,price_scale,reference_variant);print_protocol(out,result,true);out<<(seed==7?"":",")<<'\n';}
        out<<"  ],\n  \"actual_direct_deliberate\": [\n";
        for(unsigned seed=0;seed<8;++seed){auto result=protocol_reference(session,c,seed,2,true,12,false,decision_limit,price_scale,reference_variant);print_protocol(out,result,true);out<<(seed==7?"":",")<<'\n';}
        out<<"  ],\n  \"zero_cost_failure_upper_bound\": "<<-std::pow(.99,double(decision_limit-1))
           <<",\n  \"immediate_failure_upper_bound\": -1,\n  \"cheap_next_goal_timeout\": "
           <<-std::pow(.99,double(decision_limit-1))-.002*price_scale*(1-std::pow(.99,double(decision_limit)))/.01<<"\n}\n";return 0;
    }
    std::vector<std::string> nodes;for(int i=0;i<10;++i)nodes.push_back("n"+std::to_string(i));
    for(uint64_t mask=0;mask<512;++mask) {
        auto world=gt::route_scene(nodes,mask);
        bool expected=truth(world);
        semantic_correct+=session.answer(c.text,world)==(expected?Answer::True:Answer::False);++semantic_cases;
    }
    Scene all=c.world;for(const auto& action:c.actions)all=physical(std::move(all),action);
    bool necessary=truth(all);
    for(int i=0;i<9;++i) {
        Scene missing=all;std::erase(missing.facts,Atom{"edge",c.actions[i].arguments});necessary&=!truth(missing);
    }
    auto parent_plan=session.plan(c.text,c.world,c.actions,4096,8);
    gp::Limits bounds;bounds.work=2000000;bounds.plan_predictions=4096;bounds.plan_depth=8;bounds.normalization_atoms=9;
    gp::Prices prices{.00001,.0000001,.0000001,.005,.005};
    gp::ProposalInputs input;input.parent_text=c.text;input.world=c.world;input.actions=c.actions;input.names=c.names;
    if(macro_only)bounds.candidates=8;
    auto inventory=proposals(session,input,bounds,prices,macro_only);
    size_t useful=0,irrelevant=0,unavailable=0;double preview_cost=0;
    std::vector<std::pair<std::string,gp::Result>> previews;
    for(const auto& candidate:inventory.candidates) {
        gp::Inputs p;p.parent_text=c.text;p.candidate_text=candidate.text;p.world=c.world;p.actions=c.actions;p.names=c.names;
        auto result=gp::analyze(session,p,bounds,prices);preview_cost+=result.cost;
        if(result.valid){useful+=result.delta>0;irrelevant+=result.delta<=0;}else ++unavailable;
        previews.emplace_back(candidate.text,std::move(result));
    }
    size_t order_cases=64,cap_missing_useful=0,cap_missing_short_useful=0,cap_missing_irrelevant=0;
    for(size_t seed=0;seed<order_cases;++seed) {
        input.order_seed=seed;auto small=bounds;small.candidates=macro_only?8:4;auto proposed=proposals(session,input,small,prices,macro_only);
        bool any_useful=false,short_useful=false,any_irrelevant=false;
        for(const auto& candidate:proposed.candidates) {
            gp::Inputs p;p.parent_text=c.text;p.candidate_text=candidate.text;p.world=c.world;p.actions=c.actions;p.names=c.names;
            auto result=gp::analyze(session,p,bounds,prices);
            any_useful|=result.valid&&result.delta>0;
            short_useful|=result.valid&&result.delta>=3;
            any_irrelevant|=result.valid&&result.delta<=0;
        }
        cap_missing_useful+=!any_useful;cap_missing_short_useful+=!short_useful;cap_missing_irrelevant+=!any_irrelevant;
    }
    // Direct ablations receive public observations and affordances only. The
    // evaluator tests final success; it does not identify useful edge actions.
    size_t sweep_success=0,random_success=0,random_decisions_success=0;
    constexpr size_t ablations=128;
    for(size_t seed=0;seed<ablations;++seed) {
        auto available=c.actions;std::mt19937 rng(unsigned(seed)+991);std::shuffle(available.begin(),available.end(),rng);
        Scene sweep=c.world;for(size_t i=0;i<std::min<size_t>(32,available.size());++i)sweep=physical(std::move(sweep),available[i]);
        sweep_success+=truth(sweep);
        Scene random=c.world;for(int i=0;i<32;++i)random=physical(std::move(random),available[rng()%available.size()]);
        random_success+=truth(random);
        // retrieve/act cycles and one final reason+finish fit96 decisions.
        random_decisions_success+=truth(random);
    }
    out<<"  \"representable\": "<<(exact_shape?"true":"false")<<",\n"
       <<"  \"local_definition\": \""<<laminar::concepts::key(h.local)<<"\",\n"
       <<"  \"expanded_definition\": \""<<laminar::concepts::key(branches[0])<<"\",\n"
       <<"  \"local_three_short_route_calls\": "<<local_calls<<",\n  \"expanded_atoms\": "<<branches[0].body.size()<<",\n"
       <<"  \"expanded_variables\": "<<laminar::concepts::variables(branches[0])<<",\n"
       <<"  \"semantic_cases\": "<<semantic_cases<<",\n  \"semantic_correct\": "<<semantic_correct<<",\n"
       <<"  \"all_nine_effects_necessary\": "<<(necessary?"true":"false")<<",\n"
       <<"  \"initial_parent_plan_status\": \""<<parent_plan.status<<"\",\n"
       <<"  \"initial_parent_predictions\": "<<parent_plan.predictions<<",\n"
       <<"  \"proposal_inventory\": "<<inventory.total_candidates<<",\n  \"proposal_complete\": "<<(inventory.complete?"true":"false")<<",\n"
       <<"  \"enumerated_candidates\": "<<inventory.enumerated_candidates<<",\n"
       <<"  \"proposal_work\": "<<inventory.work<<",\n  \"proposal_cost\": "<<inventory.cost<<",\n"
       <<"  \"valid_useful_candidates\": "<<useful<<",\n  \"valid_irrelevant_candidates\": "<<irrelevant<<",\n"
       <<"  \"invalid_candidates\": "<<unavailable<<",\n  \"all_candidate_preview_cost\": "<<preview_cost<<",\n";
    out<<"  \"candidate_previews\": [\n";
    for(size_t i=0;i<previews.size();++i) {
        const auto& [text,result]=previews[i];
        out<<"    {\"text\": \""<<text<<"\", \"status\": \""<<result.status<<"\", \"valid\": "
           <<(result.valid?"true":"false")<<", \"delta\": "<<result.delta<<", \"work\": "<<result.work
           <<", \"predictions\": "<<result.predictions<<", \"reused_procedure\": "<<(result.plan.reused?"true":"false")
           <<", \"cost\": "<<result.cost<<"}"<<(i+1==previews.size()?"":",")<<"\n";
    }
    out<<"  ],\n"
       <<"  \"candidate_cap\": "<<(macro_only?8:4)<<",\n  \"candidate_cap_order_cases\": "<<order_cases<<",\n  \"cap_missing_any_useful\": "<<cap_missing_useful<<",\n"
       <<"  \"cap_missing_three_edge_useful\": "<<cap_missing_short_useful<<",\n  \"cap_missing_irrelevant\": "<<cap_missing_irrelevant<<",\n"
       <<"  \"no_subgoal_ablations\": {\"seeds\": "<<ablations<<", \"sweep_success\": "<<sweep_success
       <<", \"sweep_motors\": "<<std::min<size_t>(32,c.actions.size())<<", \"uniform_with_replacement_success\": "<<random_success
       <<", \"uniform_motors\": 32, \"random_within_96_decisions_success\": "<<random_decisions_success<<"},\n"
       <<"  \"semantic_proposal_gate\": "<<(exact_shape&&semantic_cases==semantic_correct&&necessary?"true":"false")<<",\n"
       <<"  \"nontrivial_subgoal_gate\": false,\n  \"stop\": \""
       <<(wide?"capacity_and_actual_protocol_reference_unresolved":"public_affordance_sweep_solves_without_subgoals")<<"\"\n}\n";
}
