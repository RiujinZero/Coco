#pragma once
// Explicit example curriculum/evaluator only. The continuing seven-action
// learner still chooses every operation, including every actual submission.
#include "goal_control_learning_harness.h"
#include "goal_curriculum_demonstration.h"
#include "../tests/goal_control_fixture.h"

namespace laminar::goal_learning::stopping {
namespace gt=goal_test;
inline constexpr const char* demonstration_source="stopping_curriculum_v1";

struct Spec {
    std::string id,prefix;
    size_t length=1;
    uint64_t mask=0;
    bool effect=false,partial=false;
    size_t arrival=0;
    unsigned order_seed=0;
    int distractors=0; //0:one unrelated chain;1:disconnected fork
};
inline std::vector<Spec> training_cases() {
    return {
        {"train_query_true1","ta",1,1,false,false,0,101,0},
        {"train_query_false1","tb",1,0,false,false,0,103,0},
        {"train_query_true3","tc",3,7,false,false,0,107,0},
        {"train_query_false3","td",3,3,false,false,0,109,0},
        {"train_done3","te",3,7,true,false,0,113,0},
        {"train_done6","tf",6,63,true,false,0,127,0},
        {"train_need1","tg",3,3,true,false,0,131,0},
        {"train_need2","th",6,60,true,false,0,137,0},
        {"train_need3","ti",3,0,true,false,0,139,0},
        {"train_read_true","tj",1,1,false,true,0,149,0},
        {"train_read_false","tk",1,0,false,true,0,151,0},
        {"train_read_effect","tl",3,3,true,true,0,157,0},
        {"train_interrupt3","tm",6,0,true,false,3,163,0},
        {"train_interrupt5","tn",6,0,true,false,5,167,0}
    };
}
inline std::vector<Spec> heldout_cases() {
    return {
        {"heldout_query_true6","va",6,63,false,false,0,1009,1},
        {"heldout_query_false6","vb",6,55,false,false,0,1013,1},
        {"heldout_query_true1","vc",1,1,false,false,0,1019,1},
        {"heldout_query_false3","vd",3,5,false,false,0,1021,1},
        {"heldout_done1","ve",1,1,true,false,0,1031,1},
        {"heldout_done6","vf",6,63,true,false,0,1033,1},
        {"heldout_need4","vg",6,33,true,false,0,1039,1},
        {"heldout_need5","vh",6,8,true,false,0,1049,1},
        {"heldout_need1_middle","vi",3,5,true,false,0,1051,1},
        {"heldout_read_true3","vj",3,7,false,true,0,1061,1},
        {"heldout_read_false6","vk",6,47,false,true,0,1063,1},
        {"heldout_read_effect6","vl",6,45,true,true,0,1069,1},
        {"heldout_interrupt4","vm",6,0,true,false,4,1087,1},
        {"heldout_interrupt6","vn",6,0,true,false,6,1091,1}
    };
}
inline std::string utterance(size_t length,const std::string& from,const std::string& to) {
    require(length==1||length==3||length==6,"unsupported stopping language length");
    return std::string(length==1?"inspect ":length==3?"route ":"span ")+from+" toward "+to;
}
inline GoalControlLimits limits() {
    GoalControlLimits value;value.roots=2;value.derived=16;value.decisions=256;value.motors=32;
    value.proposal_min_atoms=2;value.preview.candidates=16;value.preview.normalization_atoms=9;
    value.agenda_search_exhaustion=true;return value;
}
inline GoalControlPrices prices() {
    GoalControlPrices value;value.execution.operation.fill(0);value.execution.prediction=value.execution.motor=value.execution.rebuild=0;
    value.next_goal=value.activate_goal=0;value.preview={0,0,0,0,0};return value;
}
inline int demonstration(const Observation& o,const std::vector<int>& menu) {
    return curriculum_demonstration(o,menu);
}
class Environment final:public goal_learning::Environment {
    Spec spec_;
    mutable StampedScene truth_;
    mutable bool initial_disclosed_=false;
    mutable size_t read_calls_=0;
    std::vector<ScheduledRequest> schedule_;
    std::vector<Action> allowed_;
    StampedScene initial_view()const {
        auto view=truth_;
        if(spec_.partial) {
            view.scene.complete=false;
            std::erase_if(view.scene.facts,[](const auto& atom){return atom.predicate=="edge";});
        }
        return view;
    }
public:
    explicit Environment(Spec spec):spec_(std::move(spec)) {
        require(spec_.length==1||spec_.length==3||spec_.length==6,"invalid stopping length");
        require(spec_.mask<(uint64_t(1)<<spec_.length),"invalid stopping edge mask");
        require(!spec_.arrival||(spec_.effect&&!spec_.partial&&spec_.length==6&&spec_.mask==0&&spec_.arrival<9),
            "arrival must precede earliest correct completion, with no disclosure ambiguity");
        require(spec_.distractors==0||spec_.distractors==1,"invalid distractor layout");
        auto name=[&](const std::string& kind,size_t i){return spec_.prefix+kind+std::to_string(i);};
        PublicGoal root;root.id="request";root.world_id="stopping_world";root.effect=spec_.effect;root.deadline=256;
        root.text=utterance(spec_.length,name("n",0),name("n",spec_.length));
        for(size_t i=0;i<=spec_.length;++i)root.names.insert(name("n",i));
        for(size_t i=0;i<spec_.length;++i) {
            Action action{"edge_motor",{name("n",i),name("n",i+1)}};allowed_.push_back(action);
            if(spec_.mask&(uint64_t(1)<<i))truth_.scene.facts.push_back({"edge",action.arguments});
        }
        for(int i=0;i<4;++i)root.names.insert(name("d",i));
        if(!spec_.distractors) {
            allowed_.push_back({"edge_motor",{name("d",0),name("d",1)}});
            allowed_.push_back({"edge_motor",{name("d",1),name("d",2)}});
            truth_.scene.facts.push_back({"edge",{name("d",2),name("d",3)}});
        } else {
            for(int i=1;i<4;++i)allowed_.push_back({"edge_motor",{name("d",0),name("d",i)}});
            truth_.scene.facts.push_back({"edge",{name("d",3),name("d",1)}});
        }
        // Public adjacency metadata is uniform for useful and irrelevant
        // physical affordances. It contains no required-edge/usefulness bit.
        if(spec_.effect)for(const auto& action:allowed_)truth_.scene.facts.push_back({"affordance",action.arguments});
        truth_.scene=sorted(std::move(truth_.scene));truth_.revision=1;
        std::mt19937 order(spec_.order_seed);std::shuffle(allowed_.begin(),allowed_.end(),order);
        if(spec_.effect)root.actions=allowed_;root.visible=initial_view();schedule_.push_back({0,root});
        if(spec_.arrival) {
            auto urgent=root;urgent.id="urgent";urgent.text=utterance(1,name("n",0),name("n",1));
            urgent.effect=false;urgent.actions.clear();urgent.deadline=spec_.arrival+16;
            schedule_.push_back({spec_.arrival,std::move(urgent)});
        }
    }
    const Spec& spec()const{return spec_;}
    const std::vector<ScheduledRequest>& schedule()const override{return schedule_;}
    size_t read_calls()const{return read_calls_;}
    StampedScene read(const std::string& world)const override {
        require(world=="stopping_world","unknown stopping world");++read_calls_;
        if(spec_.partial&&!initial_disclosed_&&read_calls_==1)return initial_view();
        if(spec_.partial&&!initial_disclosed_){initial_disclosed_=true;++truth_.revision;}
        return truth_;
    }
    StampedScene execute(const std::string& world,const Action& action)override {
        require(world=="stopping_world"&&spec_.effect,"invalid stopping execution world");
        require(std::any_of(allowed_.begin(),allowed_.end(),[&](const auto& a){return same_action(a,action);}),"nonpublic stopping affordance");
        truth_.scene.facts.push_back({"edge",action.arguments});truth_.scene=sorted(std::move(truth_.scene));
        ++truth_.revision;initial_disclosed_=true;return truth_;
    }
    bool satisfaction(const PublicGoal& request)const override {
        require(request.world_id=="stopping_world","unknown stopping satisfaction world");
        auto tokens=words(request.text);require(tokens.size()==4&&tokens[2]=="toward","unsupported stopping utterance");
        size_t length=tokens[0]=="inspect"?1:tokens[0]=="route"?3:tokens[0]=="span"?6:0;
        require(length&&request.names.count(tokens[1])&&request.names.count(tokens[3]),"unbound stopping utterance");
        return gt::exact_route(truth_.scene,tokens[1],tokens[3],length);
    }
    const char* family()const override{return "explicit_stopping_examples_v1";}
};
} // namespace laminar::goal_learning::stopping
