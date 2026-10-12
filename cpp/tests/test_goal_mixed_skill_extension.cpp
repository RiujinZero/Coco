#include "../apps/goal_mixed_skill_extension.h"
#include "../apps/goal_long_environment.h"
#include <iostream>
namespace gl=laminar::goal_learning;
namespace ms=gl::mixed_skill;
namespace gt=laminar::goal_test;
using namespace laminar::relational;
static void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
static gl::MissionResult run(GoalControlledRelationalSession& learner,ms::QueryEnvironment& env,
                            const std::string& id,std::ostream& log,std::ostream& missions,size_t updates=32) {
    gl::MissionOptions o;o.id=id;o.arm="mixed_smoke";o.demonstration=gl::curriculum_demonstration;
    o.demonstration_source="mixed_raw_bootstrap";o.hint_probability=1;o.terminal_updates=updates;
    o.limits.roots=1;o.limits.derived=0;o.limits.decisions=64;o.limits.motors=32;
    o.prices=gl::long_task::prices();return gl::run_mission(learner,env,o,log,missions);
}
int main() {
    GoalControlledRelationalSession learner(1);ms::configure_initial(learner);
    gt::make_raw_lessons().apply(learner.knowledge);
    auto& session=learner.knowledge;
    check(session.answer("span a toward g",gt::route_scene({"a","b","c","d","e","f","g"}))==Answer::True,"union configuration broke six-edge meaning");
    std::ostringstream evidence,trace,missions;evidence<<std::boolalpha<<std::setprecision(17);
    const auto initial=ms::teach_initial(learner,evidence);
    check(initial.primitives==11&&initial.language==25&&initial.active_repairs.empty(),"initial primitive/language curriculum drift");
    auto duplicate=ms::example("unused",ms::Kind::numeric_count,1,1,2,0,5000);
    auto chain=ms::example("unused",ms::Kind::follow_endpoint,0,1,1,8,5000);
    check(session.answer(duplicate.text,duplicate.scene)==Answer::False&&session.answer(chain.text,chain.scene)==Answer::False,"raw counterexample no longer exposes acquired short programs");
    ms::QueryEnvironment base(ms::Kind::equal_counts,2,2);
    check(run(learner,base,"mixed_before",trace,missions).success,"initial learned query failed owned controller callbacks");
    session.rebuild();
    const auto policy_before=learner.policy.weights;const auto credits_before=learner.credits().size();const auto updates_before=learner.updates();
    const auto extension=ms::extend(learner,evidence);
    check(extension.language==35&&extension.concepts==4&&extension.active_repairs==std::set<int>({0,2}),"bounded raw repair/union extension failed");
    check(learner.policy.weights==policy_before&&learner.credits().size()==credits_before&&learner.updates()==updates_before&&updates_before>0,
        "extension did not retain existing policy/credit lineage");
    check(session.answer(duplicate.text,duplicate.scene)==Answer::True&&session.answer(chain.text,chain.scene)==Answer::True,"raw extension did not repair computation");
    const auto combined=session.primitives().memory.skills.kernel.routes.W;
    ms::revise_follow_support(learner,evidence,1,false);
    check(session.composed().active_repairs()==std::set<int>{2}&&session.answer(duplicate.text,duplicate.scene)==Answer::True,"follow withdrawal erased count repair");
    ms::revise_follow_support(learner,evidence,2,true);
    check(session.composed().active_repairs()==std::set<int>({0,2})&&session.primitives().memory.skills.kernel.routes.W==combined,"follow restoration did not replay exact combined primitive weights");
    std::cerr<<"raw computation and union stages passed\n";
    gl::long_task::extend(learner,evidence);
    check(session.answer("connect n0 through n9",gt::route_scene({"n0","n1","n2","n3","n4","n5","n6","n7","n8","n9"}))==Answer::True,"mixed configuration broke nine-edge meaning");
    Scene empty;std::vector<Action> offered;for(int i=0;i<9;++i)offered.push_back({"edge_motor",{"n"+std::to_string(i),"n"+std::to_string(i+1)}});
    laminar::relational::goal_preview::ProposalInputs inputs;inputs.parent_text="connect n0 through n9";
    inputs.world=empty;inputs.actions=offered;for(int i=0;i<10;++i)inputs.names.push_back("n"+std::to_string(i));
    auto proposal=laminar::relational::goal_preview::propose(session,inputs,gl::long_task::limits().preview);
    check(!proposal.candidates.empty(),"mixed configuration lost generic long-goal proposals");
    size_t checked=0;
    for(auto kind:{ms::Kind::equal_counts,ms::Kind::numeric_count})for(int size:{0,1,4})for(int other:{0,1,4}) {
        auto row=ms::example("unused",kind,size,other,3,0,20000+100*int(checked));
        check(session.answer(row.text,row.scene)==Answer(ms::truth(row.text,row.scene)),"renamed retained count computation failed");++checked;
    }
    for(int depth:{0,1,4,8,16,24})for(int same:{0,1}) {
        auto row=ms::example("unused",ms::Kind::follow_endpoint,0,same,1,depth,30000+100*int(checked));
        check(session.answer(row.text,row.scene)==Answer(ms::truth(row.text,row.scene)),"renamed extended follow computation failed");++checked;
    }
    for(int mask=0;mask<4;++mask) {
        ms::QueryEnvironment env(ms::Kind::union_query,mask,0,1,0,40000+100*mask);
        check(run(learner,env,"mixed_union"+std::to_string(mask),trace,missions).success,"new union query failed actual goal-control protocol");++checked;
    }
    for(auto kind:{ms::Kind::numeric_count,ms::Kind::follow_endpoint}) {
        ms::QueryEnvironment env(kind,1,1,3,8,50000+100*int(checked));
        check(run(learner,env,"mixed_repaired"+std::to_string(checked),trace,missions).success,"repaired computation failed actual goal-control protocol");++checked;
    }
    session.rebuild();
    const auto last=ms::follow_lessons("mixed_extension").back();
    check(session.revise_language(last.id,3,last.answer,true),"failed to arrange late correction conflict");session.rebuild();
    const auto version=session.version(),prior_updates=learner.updates(),prior_credits=learner.credits().size();
    const auto prior_weights=learner.policy.weights,prior_primitive=session.primitives().memory.skills.kernel.routes.W;
    std::ostringstream failed;bool rejected=false;
    try{ms::revise_follow_support(learner,failed,3,false);}catch(const std::runtime_error&){rejected=true;}
    check(rejected&&failed.str().empty()&&session.version()==version&&learner.updates()==prior_updates&&learner.credits().size()==prior_credits
        &&learner.policy.weights==prior_weights&&session.primitives().memory.skills.kernel.routes.W==prior_primitive,"failed correction mutated continuing state or evidence");
    for(const auto& row:ms::follow_lessons("mixed_extension"))check(session.language_receipts().at(row.id).active
        &&session.language_receipts().at(row.id).version==(row.id==last.id?4u:3u),"failed later correction left an earlier partial revision");
    std::cout<<"mixed raw skill extension: "<<checked<<" probes, six/nine-edge meanings, generic proposals, nonzero policy lineage, owned query callbacks and correction replay passed\n";
}
