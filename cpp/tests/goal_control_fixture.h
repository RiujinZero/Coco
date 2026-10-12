#pragma once
// Test-only raw lessons and evaluator physics. No next-goal decisions, learner
// features, policies, plans or compiled concept definitions are supplied here.
#include "laminar/relational_control.h"

namespace laminar::goal_test {
using relational::Action;
using relational::Answer;
using relational::LanguageReceipt;
using relational::RelationalSession;
using relational::Scene;
using relational::StampedScene;
using relational::TransitionReceipt;

inline void require(bool okay,const char* message) {
    if(!okay)throw std::runtime_error(message);
}

// Independent evaluator: exactly n directed edges, with no learner proof code.
inline bool exact_route(const Scene& scene,const std::string& from,
                        const std::string& to,size_t length) {
    std::set<std::string> frontier{from};
    for(size_t step=0;step<length;++step) {
        std::set<std::string> next;
        for(const auto& atom:scene.facts)
            if(atom.predicate=="edge"&&atom.args.size()==2&&frontier.count(atom.args[0]))
                next.insert(atom.args[1]);
        frontier=std::move(next);
    }
    return frontier.count(to)!=0;
}

inline Scene route_scene(const std::vector<std::string>& nodes,uint64_t mask=~uint64_t(0)) {
    require(nodes.size()>=2&&nodes.size()<=32,"invalid fixture route size");
    Scene scene;
    for(size_t i=0;i+1<nodes.size();++i)
        if(mask&(uint64_t(1)<<i))scene.facts.push_back({"edge",{nodes[i],nodes[i+1]}});
    return relational::sorted(std::move(scene));
}

struct RawLessons {
    std::vector<concepts::Receipt> concepts;
    std::vector<LanguageReceipt> language;
    std::vector<TransitionReceipt> transitions;

    void apply(RelationalSession& session)const {
        for(const auto& row:concepts)require(session.teach_concept(row),"fixture concept receipt rejected");
        for(const auto& row:language)require(session.teach_language(row),"fixture language receipt rejected");
        for(const auto& row:transitions)require(session.teach_transition(row),"fixture transition receipt rejected");
        session.rebuild();
    }
};

inline RawLessons make_raw_lessons() {
    RawLessons lessons;
    auto teach=[&](const std::string& name,const std::string& surface,const Scene& world,
                   const std::string& from,const std::string& to,size_t length) {
        bool answer=exact_route(world,from,to,length);
        auto suffix=std::to_string(lessons.concepts.size());
        concepts::Receipt row;
        row.id="goal_fixture_concept_"+suffix;row.name=name;
        row.args={from,to};row.scene=world;row.answer=answer;
        lessons.concepts.push_back(std::move(row));
        LanguageReceipt language;
        language.id="goal_fixture_language_"+suffix;language.text=surface;
        language.names={from,to};language.scene=world;language.answer=answer;
        for(const auto& atom:world.facts)language.names.insert(atom.args.begin(),atom.args.end());
        lessons.language.push_back(std::move(language));
    };

    // Minimal positive witnesses plus contrasts. Meanings are inferred by the
    // normal raw-evidence learner; these are final Boolean labels, not clauses.
    for(int mask=0;mask<4;++mask) {
        Scene world;
        if(mask&1)world.facts.push_back({"edge",{"a","b"}});
        if(mask&2)world.facts.push_back({"edge",{"b","a"}});
        teach("linked","inspect a toward b",world,"a","b",1);
    }
    for(uint64_t mask=0;mask<8;++mask)
        teach("short_route","route a toward d",route_scene({"a","b","c","d"},mask),"a","d",3);
    for(const auto& world:std::vector<Scene>{
        route_scene({"d","c","b","a"}),
        Scene{{{"edge",{"a","b"}},{"edge",{"c","b"}},{"edge",{"c","d"}}},true},
        route_scene({"a","b","d"})})
        teach("short_route","route a toward d",world,"a","d",3);

    // A six-edge parent is learned from whole outcomes after short_route. Its
    // small size is for protocol/progress checks, not long-horizon acquisition.
    const std::vector<std::string> parent{"a","b","c","d","e","f","g"};
    teach("parent_route","span a toward g",route_scene(parent),"a","g",6);
    for(size_t missing=0;missing<6;++missing)
        teach("parent_route","span a toward g",route_scene(parent,63u^(uint64_t(1)<<missing)),"a","g",6);
    teach("parent_route","span a toward g",route_scene({"a","b","c","g"}),"a","g",6);

    // Both row and column distractors remain, distinguishing addition from
    // replacement; a second binding checks argument orientation. These are
    // externally observed effects, not a demonstrated action sequence.
    for(int i=0;i<2;++i) {
        const std::string a=i?"x":"a",b=i?"y":"b";
        TransitionReceipt row;row.id="goal_fixture_effect_"+std::to_string(i);
        row.action={"edge_motor",{a,b}};
        row.before={{{"edge",{a,"old"}},{"edge",{"other",b}}},true};
        row.after=row.before;row.after.facts.push_back({"edge",{a,b}});
        row.after=relational::sorted(std::move(row.after));lessons.transitions.push_back(std::move(row));
    }
    return lessons;
}

// The fields below are all public task inputs. IDs express ownership only.
// This type deliberately has no desired operation, expected truth, useful-child
// bit, preferred parent, hidden world, policy, or evaluator callback.
struct PublicRequestFixture {
    std::string id,world_id,text;
    std::set<std::string> names;
    StampedScene visible;
    std::vector<Action> actions;
    bool effect=false;
    size_t deadline=96;
};

class ReferenceWorlds {
    struct Goal {std::string world,from,to;size_t length;};
    std::map<std::string,StampedScene> worlds_;
    std::map<std::string,Goal> goals_;
public:
    // Evaluator setup only. Neither this object nor Goal is a learner input.
    void add_world(const std::string& id,StampedScene world) {
        require(world.scene.complete&&worlds_.emplace(id,std::move(world)).second,"duplicate or partial evaluator world");
    }
    void add_goal(const std::string& id,const std::string& world,
                  const std::string& from,const std::string& to,size_t length) {
        require(worlds_.count(world)&&length&&goals_.emplace(id,Goal{world,from,to,length}).second,"invalid evaluator goal");
    }
    StampedScene read(const std::string& world)const{return worlds_.at(world);}
    StampedScene execute(const std::string& world,const Action& action,bool executed=true) {
        require(action.schema=="edge_motor"&&action.arguments.size()==2,"unknown evaluator action");
        auto& state=worlds_.at(world);
        if(executed)state.scene.facts.push_back({"edge",action.arguments});
        state.scene=relational::sorted(std::move(state.scene));++state.revision;return state;
    }
    bool satisfied(const std::string& id)const {
        const auto& goal=goals_.at(id);
        return exact_route(worlds_.at(goal.world).scene,goal.from,goal.to,goal.length);
    }
};

struct Fixture {
    RawLessons lessons;
    std::vector<PublicRequestFixture> requests;
    ReferenceWorlds evaluator;

    RelationalSession make_knowledge(unsigned seed=7)const {
        RelationalSession session(seed);lessons.apply(session);return session;
    }
    const PublicRequestFixture& request(const std::string& id)const {
        auto found=std::find_if(requests.begin(),requests.end(),[&](const auto& row){return row.id==id;});
        require(found!=requests.end(),"unknown public fixture request");return *found;
    }
};

inline Fixture make_fixture() {
    Fixture fixture;fixture.lessons=make_raw_lessons();
    Scene shared{{{"edge",{"spare","tail"}}},true};
    Scene separate=shared;separate.facts.push_back({"edge",{"n0","n1"}});
    fixture.evaluator.add_world("shared",{shared,1});
    fixture.evaluator.add_world("separate",{relational::sorted(separate),1});
    std::set<std::string> names{"spare","tail","d0","d1","d2","d3"};
    std::vector<Action> actions;
    for(int i=0;i<7;++i)names.insert("n"+std::to_string(i));
    for(int i=0;i<6;++i)actions.push_back({"edge_motor",{"n"+std::to_string(i),"n"+std::to_string(i+1)}});
    for(int i=0;i<3;++i)actions.push_back({"edge_motor",{"d"+std::to_string(i),"d"+std::to_string(i+1)}});
    fixture.requests.push_back({"physical","shared","span n0 toward n6",names,{shared,1},actions,true,96});
    fixture.requests.push_back({"arrival","shared","inspect n0 toward n1",names,{shared,1},{},false,16});
    fixture.requests.push_back({"separate_query","separate","inspect n0 toward n1",names,
                                fixture.evaluator.read("separate"),{},false,32});
    fixture.evaluator.add_goal("physical","shared","n0","n6",6);
    fixture.evaluator.add_goal("arrival","shared","n0","n1",1);
    fixture.evaluator.add_goal("separate_query","separate","n0","n1",1);
    return fixture;
}
} // namespace laminar::goal_test
