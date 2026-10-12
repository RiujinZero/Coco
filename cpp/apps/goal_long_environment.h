#pragma once
// Task/evaluator adapter. No policy, preferred subgoal or compiled meaning is
// delivered to the continuing learner. All extension evidence is retained.
#include "goal_control_learning_harness.h"
#include "goal_curriculum_demonstration.h"
#include "../tests/goal_control_fixture.h"

namespace laminar::goal_learning::long_task {
namespace gt=goal_test;
namespace gp=relational::goal_preview;

inline constexpr const char* demonstration_source="long_goal_curriculum_v1";
inline GoalControlLimits limits(bool exhaustion_observation=false) {
    GoalControlLimits value;value.roots=2;value.derived=16;value.decisions=256;value.motors=32;
    value.proposal_min_atoms=2;value.preview.candidates=16;value.preview.work=2000000;
    value.preview.plan_predictions=4096;value.preview.plan_depth=8;value.preview.normalization_atoms=9;
    value.agenda_search_exhaustion=exhaustion_observation;return value;
}
inline int demonstration(const Observation& o,const std::vector<int>& menu) {
    return curriculum_demonstration(o,menu);
}
inline GoalControlPrices prices() {
    GoalControlPrices value;value.execution.operation.fill(0);
    value.execution.prediction=value.execution.motor=value.execution.rebuild=0;
    value.next_goal=value.activate_goal=0;value.preview={0,0,0,0,0};return value;
}
inline Scene execute_physics(Scene world,const Action& action) {
    require(action.schema=="edge_motor"&&action.arguments.size()==2,"unsupported public motor");
    world.facts.push_back({"edge",action.arguments});return sorted(std::move(world));
}
inline gt::RawLessons extension_lessons(const std::string& prefix="long_extension") {
    gt::RawLessons lessons;std::vector<std::string> nodes;
    for(int i=0;i<10;++i)nodes.push_back("teachlong"+std::to_string(i));
    auto add=[&](Scene world) {
        const auto id=prefix+"_"+std::to_string(lessons.concepts.size());
        const bool answer=gt::exact_route(world,nodes.front(),nodes.back(),9);
        concepts::Receipt row;row.id=id;row.name="long_route";row.scene=world;
        row.args={nodes.front(),nodes.back()};row.answer=answer;lessons.concepts.push_back(row);
        LanguageReceipt language;language.id=id;language.text="connect "+nodes.front()+" through "+nodes.back();
        language.names={nodes.front(),nodes.back()};language.scene=std::move(world);language.answer=answer;
        lessons.language.push_back(std::move(language));
    };
    add(gt::route_scene(nodes));
    for(int absent=0;absent<9;++absent)add(gt::route_scene(nodes,511u^(1u<<absent)));
    add(gt::route_scene({nodes.front(),"m1","m2",nodes.back()}));
    add(gt::route_scene({nodes.front(),"m1","m2","m3","m4","m5",nodes.back()}));
    return lessons;
}
inline void export_extension(const gt::RawLessons& lessons,std::ostream& out) {
    for(const auto& r:lessons.concepts) {
        out<<"{\"event\":\"concept_teaching\",\"id\":"<<quote(r.id)<<",\"name\":"<<quote(r.name)
           <<",\"source\":"<<quote(r.source)<<",\"answer\":"<<r.answer<<",\"args\":";strings(out,r.args);
        out<<",\"scene\":";scene(out,{r.scene,0});out<<"}\n";
    }
    for(const auto& r:lessons.language) {
        out<<"{\"event\":\"language_teaching\",\"id\":"<<quote(r.id)<<",\"text\":"<<quote(r.text)
           <<",\"source\":"<<quote(r.source)<<",\"answer\":"<<r.answer<<",\"names\":";strings(out,r.names);
        out<<",\"scene\":";scene(out,{r.scene,0});out<<"}\n";
    }
}
struct ExtensionReport {
    uint64_t version_before=0,version_after=0;
    size_t concept_examples=0,language_examples=0,bootstrap_predictions=0,bootstrap_motors=0;
    size_t procedures_before=0,procedures_after=0;
    std::vector<std::string> prior_concepts;
};
inline bool same_receipt(const concepts::Receipt& a,const concepts::Receipt& b) {
    return std::tie(a.id,a.name,a.source,a.version,a.scene,a.args,a.answer,a.active,a.generated)==
           std::tie(b.id,b.name,b.source,b.version,b.scene,b.args,b.answer,b.active,b.generated);
}
inline bool same_receipt(const LanguageReceipt& a,const LanguageReceipt& b) {
    return std::tie(a.id,a.text,a.source,a.names,a.scene,a.answer,a.active,a.generated,a.version,a.computational,a.relations,a.quantities)==
           std::tie(b.id,b.text,b.source,b.names,b.scene,b.answer,b.active,b.generated,b.version,b.computational,b.relations,b.quantities);
}
inline bool same_receipt(const TransitionReceipt& a,const TransitionReceipt& b) {
    return std::tie(a.id,a.source,a.action.schema,a.action.arguments,a.before,a.after,a.executed,a.active,a.version)==
           std::tie(b.id,b.source,b.action.schema,b.action.arguments,b.before,b.after,b.executed,b.active,b.version);
}
inline bool same_receipt(const ExecutionReceipt& a,const ExecutionReceipt& b) {
    return std::tie(a.id,a.text,a.before,a.observed,a.success,a.active,a.version,a.goal_verified)==
           std::tie(b.id,b.text,b.before,b.observed,b.success,b.active,b.version,b.goal_verified)&&
           a.actions.size()==b.actions.size()&&std::equal(a.actions.begin(),a.actions.end(),b.actions.begin(),same_action);
}
inline ExtensionReport extend(GoalControlledRelationalSession& learner,std::ostream& output,
                              const std::string& prefix="long_extension") {
    require(!learner.mission_active(),"long extension requires a mission boundary");
    require(learner.learning(),"long extension requires semantic learning enabled");
    auto& session=learner.knowledge;require(!session.dirty(),"prior knowledge must be rebuilt");
    require(session.concepts().models().count("parent_route")&&session.concepts().models().count("short_route"),
        "long extension requires the retained protocol library");
    require(bool(output),"extension evidence output unavailable");
    // Successful teaching is applied to the same session. A failed extension
    // restores the complete previous state, including pending token and RNG.
    auto backup=session;std::ostringstream evidence;evidence.copyfmt(output);
    try {
    const auto weights_before=learner.policy.weights;const auto updates_before=learner.updates(),credits_before=learner.credits().size();
    const auto prior_language=session.language_receipts();const auto prior_transitions=session.transition_receipts();
    const auto prior_executions=session.execution_receipts();
    std::map<std::string,std::string> meanings;
    ExtensionReport report;report.version_before=session.version();report.procedures_before=session.procedures().size();
    for(const auto& [name,model]:session.concepts().models()) {
        require(model.hypotheses.size()==1,"prior concept not uniquely supported");
        meanings[name]=concepts::formula_key(concepts::expanded_clauses(model.hypotheses.front()));report.prior_concepts.push_back(name);
    }
    auto lessons=extension_lessons(prefix);export_extension(lessons,evidence);lessons.apply(session);
    report.concept_examples=lessons.concepts.size();report.language_examples=lessons.language.size();
    Scene before;std::vector<Action> offered{{"edge_motor",{"boot0","boot1"}},{"edge_motor",{"boot1","boot2"}},{"edge_motor",{"boot2","boot3"}}};
    for(const auto& action:offered)before.facts.push_back({"affordance",action.arguments});
    auto plan=session.issue("route boot0 toward boot3",before,offered,false,4096,8);
    require(plan.actions.size()==3,"own-action short-route bootstrap unresolved");
    std::vector<Scene> observed;auto world=before;
    for(const auto& action:plan.actions){world=execute_physics(std::move(world),action);observed.push_back(world);}
    const bool outcome=gt::exact_route(world,"boot0","boot3",3);require(outcome,"own-action bootstrap failed");
    evidence<<"{\"event\":\"own_execution_bootstrap\",\"source\":\"verified_environment\",\"text\":\"route boot0 toward boot3\",\"before\":";
    scene(evidence,{before,0});evidence<<",\"offered\":";actions(evidence,offered);evidence<<",\"chosen\":";actions(evidence,plan.actions);
    evidence<<",\"predictions\":"<<plan.predictions<<",\"reused\":"<<plan.reused<<",\"observed\":[";
    for(size_t i=0;i<observed.size();++i){if(i)evidence<<',';scene(evidence,{observed[i],i+1});}
    evidence<<"],\"outcome\":"<<outcome<<"}\n";
    require(session.feedback(plan.token,observed,{true,true,true},outcome),"own-action bootstrap receipt rejected");session.rebuild();
    report.bootstrap_predictions=plan.predictions;report.bootstrap_motors=plan.actions.size();
    report.procedures_after=session.procedures().size();report.version_after=session.version();
    for(const auto& [name,meaning]:meanings) {
        const auto& model=session.concepts().models().at(name);
        require(model.hypotheses.size()==1&&concepts::formula_key(concepts::expanded_clauses(model.hypotheses.front()))==meaning,
            "extension changed prior concept meaning");
    }
    for(const auto& [id,row]:backup.concepts().receipts())require(session.concepts().receipts().count(id)&&same_receipt(row,session.concepts().receipts().at(id)),"prior concept evidence changed");
    for(const auto& [id,row]:prior_language)require(session.language_receipts().count(id)&&same_receipt(row,session.language_receipts().at(id)),"prior language evidence changed");
    for(const auto& [id,row]:prior_transitions)require(session.transition_receipts().count(id)&&same_receipt(row,session.transition_receipts().at(id)),"prior transition evidence changed");
    for(const auto& [id,row]:prior_executions)require(session.execution_receipts().count(id)&&same_receipt(row,session.execution_receipts().at(id)),"prior own execution evidence changed");
    require(learner.policy.weights==weights_before&&learner.updates()==updates_before&&learner.credits().size()==credits_before,
        "semantic extension changed continuing policy or terminal credit");
    output<<evidence.str();output.flush();require(bool(output),"extension evidence output failed");return report;
    } catch(...) {session=std::move(backup);throw;}
}

class LongEnvironment final:public Environment {
    StampedScene world_;
    std::vector<ScheduledRequest> requests_;
    std::vector<Action> allowed_;
public:
    LongEnvironment(size_t arrival=8,unsigned order_seed=0,const std::string& prefix="n") {
        // Nine motors plus inspect/activate/finish require at least12 decisions.
        // Every required public request must arrive before any correct closure.
        require(arrival>0&&arrival<12,"long arrival must precede earliest correct root closure");
        PublicGoal root;root.id="long_root";root.world_id="long_world";root.effect=true;root.deadline=256;
        auto node=[&](const std::string& kind,int i){return prefix+kind+std::to_string(i);};
        for(int i=0;i<10;++i)root.names.insert(node("n",i));
        for(int i=0;i<4;++i)root.names.insert(node("d",i));
        for(int i=0;i<9;++i){root.names.insert(node("s",i));root.names.insert(node("t",i));}
        for(int i=0;i<9;++i)allowed_.push_back({"edge_motor",{node("n",i),node("n",i+1)}});
        for(int i=0;i<3;++i)allowed_.push_back({"edge_motor",{node("d",i),node("d",i+1)}});
        for(int i=0;i<28;++i)allowed_.push_back({"edge_motor",{node("s",i%9),node("t",(i/9+i)%9)}});
        for(const auto& action:allowed_)world_.scene.facts.push_back({"affordance",action.arguments});
        world_.scene=sorted(std::move(world_.scene));world_.revision=1;
        std::mt19937 order(order_seed);std::shuffle(allowed_.begin(),allowed_.end(),order);
        root.text="connect "+node("n",0)+" through "+node("n",9);root.visible=world_;root.actions=allowed_;
        auto urgent=root;urgent.id="long_urgent";urgent.text="inspect "+node("n",0)+" toward "+node("n",1);
        urgent.effect=false;urgent.actions.clear();urgent.deadline=arrival+16;
        requests_.push_back({0,std::move(root)});requests_.push_back({arrival,std::move(urgent)});
    }
    const std::vector<ScheduledRequest>& schedule()const override{return requests_;}
    StampedScene read(const std::string& world)const override {
        require(world=="long_world","unknown long world");return world_;
    }
    StampedScene execute(const std::string& world,const Action& action)override {
        require(world=="long_world","unknown long motor world");
        require(std::any_of(allowed_.begin(),allowed_.end(),[&](const auto& a){return a.schema==action.schema&&a.arguments==action.arguments;}),
            "nonpublic long affordance");
        world_.scene=execute_physics(std::move(world_.scene),action);++world_.revision;return world_;
    }
    bool satisfaction(const PublicGoal& goal)const override {
        const auto tokens=words(goal.text);require(tokens.size()==4,"unsupported evaluator public request");
        size_t length=0;
        if(tokens[0]=="inspect"&&tokens[2]=="toward")length=1;
        if(tokens[0]=="route"&&tokens[2]=="toward")length=3;
        if(tokens[0]=="span"&&tokens[2]=="toward")length=6;
        if(tokens[0]=="connect"&&tokens[2]=="through")length=9;
        require(length&&goal.names.count(tokens[1])&&goal.names.count(tokens[3]),"unsupported public evaluator language");
        return gt::exact_route(read(goal.world_id).scene,tokens[1],tokens[3],length);
    }
    const char* family()const override{return "retained_six_edge_library_nine_edge_extension";}
};
} // namespace laminar::goal_learning::long_task
