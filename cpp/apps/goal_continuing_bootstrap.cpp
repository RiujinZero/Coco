// One continuing controller/knowledge/reader through explicitly assisted mixed
// examples, taper, and autonomous continuation. Every intervention is logged.
#define main goal_pilot_unused_entry
#include "goal_control_learning_pilot.cpp"
#undef main
#include "goal_curriculum_demonstration.h"
#include "goal_stopping_environment.h"
#include "goal_long_environment.h"
#include "goal_reader_environment.h"
#include "goal_mixed_skill_extension.h"
#include <chrono>

namespace st=gl::stopping;
namespace lt=gl::long_task;
namespace rt=gl::reader_task;
namespace ms=gl::mixed_skill;
using Clock=std::chrono::steady_clock;
static constexpr const char* family_names[]={"stopping","six_edge_interrupt","long_derived","reader","equal_counts","numeric_count","follow_endpoint","union_query"};
struct MixedConfig {
    bool smoke=true;
    int hidden=32;
    size_t warm=20,autonomous=0,fit_cap=10000;
};
static MixedConfig mixed_config(bool smoke) {return smoke?MixedConfig{}:MixedConfig{false,1316,400,200,200000};}
static std::string mixed_configuration(const MixedConfig& c) {
    std::ostringstream out;out<<std::boolalpha;
    out<<"{\n\"schema\":\"goal_continuing_bootstrap_v1\",\n\"smoke\":"<<c.smoke
       <<",\n\"seed\":1,\n\"hidden\":"<<c.hidden<<",\n\"warm_missions\":"<<c.warm<<",\n\"autonomous_missions\":"<<c.autonomous
       <<",\n\"fit_cap\":"<<c.fit_cap<<",\n\"terminal_updates\":512,\n\"receipt_capacity\":32768,\n\"procedure_capacity\":1024,"
         "\n\"same_continuing_instance\":true,\n\"families\":[";
    for(size_t i=0;i<8;++i){if(i)out<<',';out<<gl::quote(family_names[i]);}
    out<<"],\n\"family_order\":\"index modulo8, including all failures; pilot slot7 before100 is old equal-count practice until union extension\","
         "\n\"warm_guidance\":\"smoke all1; pilot first100 p1 then (399-index)/300 through zero\","
         "\n\"direct_fitting\":\"after every10 warm missions: smoke8 rounds; pilot first100 up to8, then floor(8*(399-index)/300); cap may truncate rounds; never after warm\","
         "\n\"demonstration_source\":\"continuing_curriculum_v1\",\n\"demonstration_input\":\"only32 public floats and legal menu; one common function\","
         "\n\"autonomous_epsilon\":0.1,\n\"memory_policy\":false,\n\"prices\":\"all zero; count actual decisions, motors, symbolic predictions and primitive steps\","
         "\n\"global_limits\":{\"decisions\":256,\"motors\":32,\"agenda_search_exhaustion\":true},"
         "\n\"long_limits\":{\"derived\":16,\"proposal_min_atoms\":2,\"plan_predictions\":4096,\"preview_work\":2000000,\"depth\":8},"
         "\n\"training_arrivals\":{\"six\":[3,5,7],\"long\":[8,10]},\n\"task_order\":\"1000+index, no agenda reseed\","
         "\n\"physical_affordance_metadata\":\"all offered action argument pairs receive the same public affordance predicate; no usefulness mark\","
         "\n\"assistance_seed\":\"50000+index\",\n\"reader\":\"one archived GroundedSession trained on4 raw report answers; positive conservative projections only; ambiguity/partial negative probes remain Unknown\","
         "\n\"semantic_setup\":\"configure mixed skill support before any evidence; raw fixture once, reader, long extension and initial skills; smoke extends repair/union upfront, pilot after100missions preserving learned policy/credits\","
         "\n\"stopping_training\":\"smoke done6, partial-read-true, interrupt6; pilot cycles all14 declared training cases\","
         "\n\"skill_timing\":\"pilot before100 uses copies1, followdepth0or1 and replaces union slot with equal-count; later duplicates2or3, longer follow and union are interleaved with old skills\","
         "\n\"probes\":\"smoke8 mixed cases before and after; pilot32mixed plus all14stopping heldout cases before/warm/autonomous; each sequential frozen learner+reader clone, epsilon0, hints0, fits0\","
         "\n\"claim\":\"integration and explicit assisted bootstrap; autonomy measured separately; no four-gate closure or post-bootstrap acquisition inferred from teaching\"\n}\n";
    return out.str();
}
struct MixedOutputs:Outputs {
    std::ofstream phases,computations;
    explicit MixedOutputs(const std::string& prefix,std::ios::openmode mode=std::ios::trunc):Outputs(prefix,mode),phases(prefix+"_phases.jsonl",std::ios::out|mode),computations(prefix+"_computations.jsonl",std::ios::out|mode) {
        gl::require(bool(phases)&&bool(computations),"mixed output unavailable");
        phases<<std::boolalpha<<std::setprecision(17);computations<<std::boolalpha<<std::setprecision(17);
    }
};
// Uniform metadata repeats the public affordance adjacency already exposed in
// every action's arguments. It does not mark useful actions or supply a route.
class MixedSixEdgeEnvironment final:public gl::Environment {
    SixEdgeEnvironment physics_;
    std::vector<gl::ScheduledRequest> requests_;
    std::vector<Action> actions_;
    StampedScene enrich(StampedScene view)const {
        for(const auto& action:actions_)view.scene.facts.push_back({"affordance",action.arguments});
        view.scene=sorted(std::move(view.scene));return view;
    }
public:
    MixedSixEdgeEnvironment(size_t arrival,unsigned order):physics_(arrival,order),requests_(physics_.schedule()) {
        actions_=requests_.front().request.actions;
        for(auto& request:requests_)request.request.visible=enrich(request.request.visible);
    }
    const std::vector<gl::ScheduledRequest>& schedule()const override{return requests_;}
    StampedScene read(const std::string& world)const override{return enrich(physics_.read(world));}
    StampedScene execute(const std::string& world,const Action& action)override{return enrich(physics_.execute(world,action));}
    bool satisfaction(const PublicGoal& request)const override {
        if(request.id=="physical"||request.id=="arrival")return physics_.satisfaction(request);
        const auto tokens=words(request.text);gl::require(tokens.size()==4,"unknown derived six-edge surface");
        size_t length=tokens[0]=="inspect"?1:tokens[0]=="route"?3:tokens[0]=="span"?6:tokens[0]=="connect"?9:0;
        gl::require(length&&request.names.count(tokens[1])&&request.names.count(tokens[3]),"unbound derived six-edge query");
        return gt::exact_route(physics_.read(request.world_id).scene,tokens[1],tokens[3],length);
    }
    const char* family()const override{return "six_edge_with_uniform_public_affordance_metadata";}
};
static GoalControlLimits mixed_limits(size_t family) {
    auto value=family==2?lt::limits(true):st::limits();
    value.agenda_search_exhaustion=true;return value;
}
static std::unique_ptr<gl::Environment> mixed_environment(size_t family,size_t index,bool heldout,bool smoke,
    rt::Component& reader,std::ostream& evidence,const std::string& label) {
    const size_t cycle=index/8;const unsigned order=(heldout?10000:1000)+unsigned(index);
    if(family==0) {
        auto list=heldout?st::heldout_cases():st::training_cases();
        const size_t selected=smoke?(heldout?6:std::vector<size_t>{5,9,12}[cycle%3]):cycle%list.size();auto spec=list[selected];
        spec.order_seed=order;spec.prefix+=(heldout?"v":"t")+std::to_string(cycle);return std::make_unique<st::Environment>(spec);
    }
    if(family==1)return std::make_unique<MixedSixEdgeEnvironment>((heldout?std::vector<size_t>{4,6,8}:training_arrivals)[cycle%3],order);
    if(family==2) {
        const size_t arrival=(heldout?std::vector<size_t>{7,9,11}:std::vector<size_t>{8,10})[cycle%(heldout?3:2)];
        return std::make_unique<lt::LongEnvironment>(arrival,order,(heldout?"vlong":"tlong")+std::to_string(cycle));
    }
    if(family==3)return std::make_unique<rt::Environment>(reader,heldout?"token":"key",heldout?"dana":"cy",evidence,label);
    const int n=int(heldout?5+cycle%3:2+cycle%3),offset=int(heldout?100000+cycle*20:cycle*20);
    if(family==4)return std::make_unique<ms::QueryEnvironment>(ms::Kind::equal_counts,n,n-int(cycle%2),1,1,offset);
    const bool extended=heldout||smoke||index>=100;
    if(family==5)return std::make_unique<ms::QueryEnvironment>(ms::Kind::numeric_count,n,n-int(cycle%2),heldout?3:extended?2:1,1,offset);
    if(family==6)return std::make_unique<ms::QueryEnvironment>(ms::Kind::follow_endpoint,0,int(cycle%2),1,int(heldout?7+cycle%3:extended?4+cycle%3:cycle%2),offset);
    if(!heldout&&!smoke&&index<100)return std::make_unique<ms::QueryEnvironment>(ms::Kind::equal_counts,n,n-int(cycle%2),1,1,offset);
    return std::make_unique<ms::QueryEnvironment>(ms::Kind::union_query,int(cycle%4),0,1,1,offset);
}
static void export_computations(const GoalControlledRelationalSession& learner,const std::string& mission,std::ostream& out) {
    for(const auto& r:learner.mission_computations()) {
        out<<"{\"mission\":"<<gl::quote(mission)<<",\"token\":"<<r.token<<",\"candidate\":"<<gl::quote(r.candidate)
           <<",\"parent\":"<<gl::quote(r.parent)<<",\"model_before\":"<<r.model_before<<",\"model_after\":"<<r.model_after
           <<",\"status\":"<<gl::quote(r.status)<<",\"answer\":"<<int(r.answer)<<",\"plan_allowance\":"<<r.plan_allowance
           <<",\"primitive_steps\":"<<r.primitive_steps<<",\"cost\":"<<r.total_cost<<",\"parent_text\":"<<gl::quote(r.inputs.parent_text)
           <<",\"candidate_text\":"<<gl::quote(r.inputs.candidate_text)<<",\"world\":";gl::scene(out,{r.inputs.world,r.inputs.observation_revision});
        out<<",\"affordances\":";gl::actions(out,r.inputs.actions);out<<",\"plan\":";gl::actions(out,r.plan.actions);
        if(r.preview)out<<",\"preview\":{\"valid\":"<<r.preview->valid<<",\"status\":"<<gl::quote(r.preview->status)
            <<",\"before_missing\":"<<r.preview->before_missing<<",\"after_missing\":"<<r.preview->after_missing
            <<",\"delta\":"<<r.preview->delta<<",\"work\":"<<r.preview->work<<",\"predictions\":"<<r.preview->predictions<<'}';
        if(r.proposals) {out<<",\"proposals\":{\"complete\":"<<r.proposals->complete<<",\"truncated\":"<<r.proposals->truncated
            <<",\"enumerated\":"<<r.proposals->enumerated_candidates<<",\"total\":"<<r.proposals->total_candidates<<",\"texts\": [";
            for(size_t i=0;i<r.proposals->candidates.size();++i){if(i)out<<',';out<<gl::quote(r.proposals->candidates[i].text);}out<<"]}";}
        out<<"}\n";
    }out.flush();
}
static void ledger(std::ostream& out,const std::string& stage,const GoalControlledRelationalSession& learner,double seconds=0) {
    out<<"{\"event\":\"checkpoint\",\"stage\":"<<gl::quote(stage)<<",\"updates\":"<<learner.updates()
       <<",\"credits\":"<<learner.credits().size()<<",\"model_version\":"<<learner.knowledge.version()
       <<",\"concept_models\":"<<learner.knowledge.concepts().models().size()<<",\"language_receipts\":"<<learner.knowledge.language_receipts().size()
       <<",\"procedures\":"<<learner.knowledge.procedures().size()<<",\"seconds\":"<<seconds<<"}\n";out.flush();
}
static size_t demonstration_states(const GoalControlledRelationalSession& learner) {
    using Key=std::pair<gl::Observation,std::vector<int>>;std::map<Key,int> labels;
    for(const auto& credit:learner.credits())if(credit.active)for(const auto& row:credit.trace)if(!row.teacher.empty()) {
        gl::require(row.teacher==gl::curriculum_demonstration_source,"unexpected mixed teacher source");
        auto [it,inserted]=labels.emplace(Key{row.observation,row.available},int(row.operation));
        gl::require(inserted||it->second==int(row.operation),"cross-family example observation alias");
    }return labels.size();
}
static void fit_examples(GoalControlledRelationalSession& learner,const MixedConfig& config,size_t warm_index,size_t& fitted,MixedOutputs& out) {
    const size_t states=demonstration_states(learner),desired=config.smoke?8:warm_index<100?8:size_t(8*(399-warm_index)/300);
    const size_t rounds=states?std::min(desired,(config.fit_cap-fitted)/states):0;
    const auto before=learner.updates(),credits=learner.credits().size(),version=learner.knowledge.version();
    const auto values=learner.empirical_values();const auto start=Clock::now();
    const auto count=learner.fit_demonstrations(rounds,gl::curriculum_demonstration_source);fitted+=count;
    gl::require(count==states*rounds&&learner.updates()-before==count&&fitted<=config.fit_cap,"unexpected mixed demonstration update count");
    gl::require(learner.empirical_values()==values&&learner.credits().size()==credits&&learner.knowledge.version()==version,"demonstration changed reward/semantics");
    out.phases<<"{\"event\":\"demonstration_fit\",\"warm_index\":"<<warm_index<<",\"source\":"<<gl::quote(gl::curriculum_demonstration_source)
       <<",\"representatives\":"<<states<<",\"rounds\":"<<rounds<<",\"fits\":"<<count<<",\"cumulative_fits\":"<<fitted
       <<",\"seconds\":"<<std::chrono::duration<double>(Clock::now()-start).count()<<",\"empirical_values_unchanged\":true}\n";out.phases.flush();
}
static gl::MissionOptions mixed_options(size_t index,size_t family,const std::string& phase,bool frozen=false) {
    gl::MissionOptions options;options.id=phase+"_m"+std::to_string(index);options.arm=phase;options.index=index;
    options.terminal_updates=frozen?0:512;options.limits=mixed_limits(family);options.prices=st::prices();
    options.demonstration=gl::curriculum_demonstration;options.demonstration_source=gl::curriculum_demonstration_source;
    options.assistance_seed=50000+unsigned(index);options.task_order_seed=(frozen?10000:1000)+unsigned(index);return options;
}
static size_t mixed_evaluate(const GoalControlledRelationalSession& source,const rt::Component& reader,
    const MixedConfig& config,const std::string& label,MixedOutputs& out) {
    auto frozen=source;frozen.set_learning(false);auto reader_clone=reader.clone();
    const auto weights=source.policy.weights;const auto version=source.knowledge.version();const auto updates=source.updates();
    size_t successes=0,cases=config.smoke?8:46;std::array<size_t,8> family_success{},family_cases{};
    for(size_t index=0;index<cases;++index) {
        const auto family=index<32?index%8:0;auto options=mixed_options(index,family,"eval_"+label,true);
        std::unique_ptr<gl::Environment> environment;
        if(index<32)environment=mixed_environment(family,index,true,config.smoke,reader_clone,out.bootstrap,options.id);
        else environment=std::make_unique<st::Environment>(st::heldout_cases().at(index-32));
        auto result=gl::run_mission(frozen,*environment,options,out.trace,out.missions);successes+=result.success;family_success[family]+=result.success;
        ++family_cases[family];
        export_computations(frozen,options.id,out.computations);
    }
    gl::require(frozen.policy.weights==weights&&source.policy.weights==weights&&frozen.knowledge.version()==version&&source.knowledge.version()==version
        &&source.updates()==updates&&frozen.updates()==updates,"mixed frozen probe changed live or clone learning");
    out.evaluations<<"{\"checkpoint\":"<<gl::quote(label)<<",\"successes\":"<<successes<<",\"cases\":"<<cases
       <<",\"family_successes\":";gl::numbers(out.evaluations,family_success);out.evaluations<<",\"family_cases\":";gl::numbers(out.evaluations,family_cases);
    out.evaluations<<",\"frozen\":true,\"hints\":0,\"fits\":0}\n";out.evaluations.flush();
    reader.boundary_probes(source,out.bootstrap,label);
    std::cout<<"mixed evaluation "<<label<<' '<<successes<<'/'<<cases<<'\n'<<std::flush;return successes;
}
#ifndef LAMINAR_GOAL_MIXED_LIBRARY
int main(int argc,char** argv) {
    try {
        gl::require(argc==4,"usage: goal_continuing_bootstrap --declare|--run smoke|pilot OUTPUT_PREFIX");
        const std::string mode=argv[1],profile=argv[2],prefix=argv[3];gl::require(profile=="smoke"||profile=="pilot","unknown mixed profile");
        const auto config=mixed_config(profile=="smoke");
        if(mode=="--declare") {
            gl::require(!std::filesystem::exists(prefix+"_config.json"),"mixed declaration exists");
            std::ofstream file(prefix+"_config.json");gl::require(bool(file),"cannot declare mixed run");file<<mixed_configuration(config);return 0;
        }
        gl::require(mode=="--run","unknown mixed operation");std::ifstream file(prefix+"_config.json");
        const std::string declaration((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
        gl::require(declaration==mixed_configuration(config),"mixed declaration absent or changed");
        for(const auto* suffix:{"_trace.jsonl","_missions.jsonl","_weights.jsonl","_evaluations.jsonl","_bootstrap.jsonl","_summary.json","_phases.jsonl","_computations.jsonl"})
            gl::require(!std::filesystem::exists(prefix+suffix),"refusing to overwrite mixed evidence");
        MixedOutputs out(prefix);const auto start=Clock::now();GoalControlledRelationalSession learner(1,config.hidden,true);
        learner.knowledge=RelationalSession(1,32768,1024);ms::configure_initial(learner);
        ledger(out.phases,"configured_before_evidence",learner);
        auto fixture=gt::make_fixture();fixture.lessons.apply(learner.knowledge);export_bootstrap(fixture.lessons,out.bootstrap);
        ledger(out.phases,"raw_route_library",learner);
        rt::Component reader;reader.teach(learner,out.bootstrap);ledger(out.phases,"raw_reader",learner);
        lt::extend(learner,out.bootstrap,"mixed_long");ledger(out.phases,"long_extension",learner);
        ms::teach_initial(learner,out.bootstrap,"mixed_initial");ledger(out.phases,"skill_initial",learner);
        if(config.smoke){ms::extend(learner,out.bootstrap,"mixed_extension");ledger(out.phases,"skill_extension",learner);}
        gl::weights(out.parameters,learner,"initial","continuing");
        const size_t before=mixed_evaluate(learner,reader,config,"before",out);size_t fitted=0,warm_success=0,autonomous_success=0,warm_probe=0;
        for(size_t index=0;index<config.warm+config.autonomous;++index) {
            if(!config.smoke&&index==100) {
                const auto weights=learner.policy.weights;const auto updates=learner.updates(),credits=learner.credits().size();
                learner.knowledge.rebuild();ledger(out.phases,"before_skill_extension",learner);
                ms::extend(learner,out.bootstrap,"mixed_extension");
                gl::require(learner.policy.weights==weights&&learner.updates()==updates&&learner.credits().size()==credits&&credits==100,
                    "continuing skill extension replaced policy or credit");ledger(out.phases,"after_skill_extension",learner);
            }
            const bool warm=index<config.warm;const size_t family=index%8;const std::string phase=warm?"warm":"autonomous";
            auto options=mixed_options(index,family,phase);
            options.hint_probability=warm?(config.smoke||index<100?1.:double(399-index)/300.):0.;options.epsilon=warm?0.f:.1f;
            auto environment=mixed_environment(family,index,false,config.smoke,reader,out.bootstrap,options.id);
            const auto mission_start=Clock::now();auto result=gl::run_mission(learner,*environment,options,out.trace,out.missions);
            if(warm)warm_success+=result.success;else {autonomous_success+=result.success;gl::require(result.hints==0,"autonomous assistance");}
            export_computations(learner,options.id,out.computations);
            ledger(out.phases,options.id,learner,std::chrono::duration<double>(Clock::now()-mission_start).count());
            if(warm&&(index+1)%10==0)fit_examples(learner,config,index,fitted,out);
            gl::weights(out.parameters,learner,options.id,"continuing");
            std::cout<<options.id<<" family="<<family_names[family]<<" success="<<result.success<<" decisions="<<result.decisions<<" updates="<<learner.updates()<<'\n'<<std::flush;
            if(index+1==config.warm)warm_probe=mixed_evaluate(learner,reader,config,"warm",out);
        }
        const size_t after=config.autonomous?mixed_evaluate(learner,reader,config,"autonomous",out):0;
        std::ofstream summary(prefix+"_summary.json");summary<<std::boolalpha<<std::setprecision(17);
        summary<<"{\"schema\":\"goal_continuing_bootstrap_result_v1\",\"complete\":true,\"smoke\":"<<config.smoke
            <<",\"hidden\":"<<config.hidden<<",\"warm_missions\":"<<config.warm<<",\"warm_successes\":"<<warm_success
            <<",\"autonomous_missions\":"<<config.autonomous<<",\"autonomous_successes\":"<<autonomous_success
            <<",\"before_successes\":"<<before<<",\"warm_probe_successes\":"<<warm_probe<<",\"autonomous_probe_successes\":";
        if(config.autonomous)summary<<after;else summary<<"null";
        summary<<",\"demonstration_fits\":"<<fitted
            <<",\"updates\":"<<learner.updates()<<",\"credits\":"<<learner.credits().size()<<",\"seconds\":"<<std::chrono::duration<double>(Clock::now()-start).count()
            <<",\"claim\":\"one continuing mixed integration; guided successes are assisted bootstrap, not autonomous acquisition\"}\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif
