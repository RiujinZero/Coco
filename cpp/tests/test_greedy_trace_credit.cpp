#include "laminar/continual_training.h"
#include <iostream>
using namespace laminar::relational;
void check(bool okay,const char* why){if(!okay)throw std::runtime_error(why);}

ControlStep force(ControlledRelationalSession& learner,ControlOp op) {
    return learner.choose(0,-1,false,nullptr,ControlHint{op,"credit_contract"});
}
std::vector<ControlStep> question(ControlledRelationalSession& learner,const std::string& id) {
    check(learner.begin(id,"unknown a",{"a"},{{{},false},0},{},false),"question begin failed");
    auto read=force(learner,ControlOp::ask);
    check(learner.read(read.token,StampedScene{{{},true},1},"verified_environment"),"question read failed");
    force(learner,ControlOp::reason);force(learner,ControlOp::finish);return learner.trace();
}
using Values=std::map<laminar::reasoning::Observation,std::array<float,5>>;
// Independent forward discounted sum: stop before the first nongreedy
// subsequent action, then bootstrap once. The implementation walks backwards.
double reference_target(const std::vector<ControlStep>& trace,size_t at,double reward,const Values& values) {
    double result=-trace[at].cost,discount=.99;
    for(size_t j=at+1;j<trace.size();++j,discount*=.99) {
        auto found=values.find(trace[j].observation);std::array<float,5> row{};
        if(found!=values.end())row=found->second;
        double best=-1e9;for(int a:trace[j].available)best=std::max(best,double(row[a]));
        if(row[int(trace[j].operation)]<best)return result+discount*best;
        result-=discount*trace[j].cost;
    }
    return result+std::pow(.99,double(trace.size()-1-at))*reward;
}
void independent_targets() {
    for(bool correct:{false,true})for(bool terminal:{false,true}) {
        ControlledRelationalSession learner(5,true,terminal,true,{},false,32,{},false,true);
        for(int repetition=0;repetition<3;++repetition) {
            auto trace=question(learner,"question"+std::to_string(repetition));auto expected=learner.empirical_values();
            for(size_t i=trace.size();i-->0;) {
                if(terminal&&i+1!=trace.size())continue;
                double target=reference_target(trace,i,correct?1:-1,expected);
                if(!correct&&!terminal&&i==0)check(std::abs(target+trace[0].cost)<1e-6,
                    "nongreedy failed continuation was charged to the first choice");
                auto& value=expected[trace[i].observation][int(trace[i].operation)];value+=.2f*(float(target)-value);
            }
            check(learner.finish(correct,false,terminal?1:trace.size()),"question credit rejected");
            check(expected.size()==learner.empirical_values().size(),"trace introduced an uncredited state");
            for(const auto& [obs,row]:expected)for(int a=0;a<5;++a)
                check(std::abs(row[a]-learner.empirical_values().at(obs)[a])<1e-6,"greedy trace differs from forward discounted sum/cutoff");
            if(terminal)check(!learner.empirical_values().count(trace[0].observation),"terminal-only arm credited an earlier step");
        }
        auto weights=learner.policy.weights;auto values=learner.empirical_values();
        check(learner.revise_credit("question0",1,!correct,false)&&learner.revise_credit("question0",2,correct,false)
              &&weights==learner.policy.weights&&values==learner.empirical_values(),"greedy trace correction not exactly replayable");
    }
}
void positive_bootstrap_cutoff() {
    ControlledRelationalSession learner(9,true,false,true,{},false,32,{},false,true);
    for(int i=0;i<4;++i){auto trace=question(learner,"known"+std::to_string(i));check(learner.finish(true,false,trace.size()),"known continuation credit failed");}
    check(learner.begin("detour","unknown a",{"a"},{{{},false},0},{},false),"detour begin failed");
    for(int i=0;i<2;++i) {
        auto read=force(learner,ControlOp::ask);
        check(learner.read(read.token,StampedScene{{{},true},1},"verified_environment"),"detour read failed");
    }
    force(learner,ControlOp::reason);force(learner,ControlOp::finish);auto trace=learner.trace();auto expected=learner.empirical_values();
    check(trace.size()==4,"detour trace changed");
    for(size_t i=trace.size();i-->0;) {
        double target=reference_target(trace,i,1,expected);
        if(i==0) {
            auto row=expected.at(trace[1].observation);float best=-1e9;for(int a:trace[1].available)best=std::max(best,row[a]);
            check(best>0&&row[int(trace[1].operation)]<best,"detour is not a nongreedy choice against learned positive values");
            check(std::abs(target-(-trace[0].cost+.99*best))<1e-6,"positive bootstrap did not stop at first nongreedy action");
        }
        auto& value=expected[trace[i].observation][int(trace[i].operation)];value+=.2f*(float(target)-value);
    }
    check(learner.finish(true,false,trace.size()),"detour credit rejected");
    for(const auto& [obs,row]:expected)for(int a=0;a<5;++a)
        check(std::abs(row[a]-learner.empirical_values().at(obs)[a])<1e-6,"positive bootstrap differs from independent target");
}
Scene add(Scene world,const std::string& predicate){world.facts.push_back({predicate,{"a","b"}});return sorted(world);}
void interrupted_parent(size_t children,size_t extra_decisions) {
    const Scene initial{{{"edge",{"a","old"}},{"edge",{"other","b"}},{"marked",{"a","old"}},{"marked",{"other","b"}}},true};
    const Scene complete=add(add(initial,"edge"),"marked");const std::set<std::string> names{"a","b","old","other"};
    const std::vector<Action> actions{{"edge_motor",{"a","b"}},{"mark_motor",{"a","b"}}};
    ControlledRelationalSession learner(7,true,false,true,{true,1,32},false,32,{},false,true);
    for(int i=0;i<3;++i) {
        auto world=i==0?complete:i==1?add(initial,"edge"):add(initial,"marked");
        laminar::concepts::Receipt c;c.id="concept"+std::to_string(i);c.name="joined";c.args={"a","b"};c.scene=world;c.answer=i==0;
        check(learner.knowledge.teach_concept(c),"concept rejected");
        LanguageReceipt l;l.id="language"+std::to_string(i);l.text="join a with b";l.names=names;l.scene=world;l.answer=i==0;
        check(learner.knowledge.teach_language(l),"language rejected");
    }
    for(int i=0;i<2;++i) {
        TransitionReceipt t;t.id="effect"+std::to_string(i);t.action=actions[i];t.before=initial;t.after=add(initial,i?"marked":"edge");
        check(learner.knowledge.teach_transition(t),"effect rejected");
    }
    learner.knowledge.rebuild();
    check(learner.begin("parent","join a with b",names,{initial,1},actions,true),"parent begin failed");
    force(learner,ControlOp::reason);auto step=force(learner,ControlOp::act);
    check(step.actions.size()==1,"parent missing first action");auto world=add(initial,step.actions[0].schema=="edge_motor"?"edge":"marked");
    check(learner.actions(step.token,{world},{true},2),"parent first action rejected");
    auto first=learner.trace()[0];check(learner.suspend(),"parent did not suspend");
    for(size_t i=0;i<children;++i) {
        check(learner.begin("child"+std::to_string(i),"join a with b",names,{complete,10},{},false),"child begin failed");
        force(learner,ControlOp::reason);force(learner,ControlOp::finish);
        // Alternate signed outcomes to ensure parent credit cannot silently
        // borrow the latest intervening goal's outcome.
        check(learner.finish(i%2==0,true,32),"child credit rejected");
    }
    check(!learner.empirical_values().count(first.observation),"unfinished parent entered credit");
    world.facts.push_back({"external",{"other","a"}});world=sorted(world);
    check(learner.resume("parent",{world,11}),"parent resume failed");
    force(learner,ControlOp::reason);step=force(learner,ControlOp::act);check(step.actions.size()==1,"remaining goal not replanned");
    auto after=add(world,step.actions[0].schema=="edge_motor"?"edge":"marked");
    check(learner.actions(step.token,{after},{true},12),"parent second action rejected");
    for(size_t i=0;i<=extra_decisions;++i)force(learner,ControlOp::reason);
    check(force(learner,ControlOp::finish).reply==Answer::True,"parent goal failed");
    auto size=learner.trace().size();check(learner.finish(true,true,64),"parent delayed credit rejected");
    check(size==6+extra_decisions&&learner.credits().back().trace.size()==size,"parent trace lost or mixed with children");
    check(learner.empirical_values().at(first.observation)[int(first.operation)]>0,"delayed reward did not reach first parent choice");
    auto weights=learner.policy.weights;auto values=learner.empirical_values();
    check(learner.revise_credit("parent",1,false,false)&&learner.revise_credit("parent",2,true,true)
          &&weights==learner.policy.weights&&values==learner.empirical_values(),"interrupted correction not exactly replayable");
}
void default_parity() {
    ControlledRelationalSession original(5,true,false,true),explicit_td(5,true,false,true,{},false,32,{},false,false);
    for(int i=0;i<8;++i) {
        auto id="default"+std::to_string(i);question(original,id);question(explicit_td,id);
        check(original.finish(i%2==0,false,64)&&explicit_td.finish(i%2==0,false,64),"TD fixture failed");
    }
    check(original.policy.weights==explicit_td.policy.weights&&original.empirical_values()==explicit_td.empirical_values(),"default TD changed");
    for(const auto& profile:{"controller_trace","controller50k_trace"}) {
        auto config=laminar::training::ContinualTrainingConfig::named(profile);auto learner=config.create(1);
        check(config.greedy_trace_credit&&learner.policy.weights.size()==size_t(38*config.controller_hidden+5),"greedy trace profile mismatch");
    }
    for(int invalid=0;invalid<3;++invalid) {
        bool rejected=false;try{ControlledRelationalSession learner(5,true,false,invalid!=0,{},invalid==1,32,{},invalid==2,true);}
        catch(const std::invalid_argument&){rejected=true;}check(rejected,"incompatible credit combination accepted");
    }
}
int main() {
    independent_targets();positive_bootstrap_cutoff();default_parity();
    interrupted_parent(16,0);interrupted_parent(16,40);interrupted_parent(32,40);
    std::cout<<"Greedy trace: independent targets/cutoffs, terminal-only control, 46-decision delayed goals, 16/32 intervening outcomes, correction replay and default TD parity passed\n";
}
