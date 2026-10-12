#pragma once
// Raw semantic curricula and independently scored query tasks for one continuing
// goal controller. No controller parameters or compiled programs are supplied.
#include "goal_control_learning_harness.h"

namespace laminar::goal_learning::mixed_skill {
enum class Kind {equal_counts,numeric_count,follow_endpoint,union_query};
struct Report {
    uint64_t version_before=0,version_after=0;
    size_t primitives=0,language=0,concepts=0;
    std::set<int> active_repairs;
};
inline void configure_initial(GoalControlledRelationalSession& learner) {
    require(!learner.mission_active()&&learner.learning()&&learner.knowledge.version()==0,
        "mixed skill configuration must precede every lesson");
    concepts::Limits limits;limits.body=3;limits.clauses=2;
    learner.knowledge.enable_disjunctive_concepts(limits);
    learner.knowledge.enable_composed_learning(5000,true,100,true);
}
inline reasoning::PrimitiveReceipt primitive(std::string id,int cue,reasoning::Value subject,
                                             reasoning::Value target,reasoning::Value answer) {
    reasoning::PrimitiveReceipt row;row.id=std::move(id);row.cue=cue;row.subject=subject;
    row.target=target;row.answer=answer;return row;
}
inline LanguageReceipt example(const std::string& id,Kind kind,int size,int other,
                                int copies=1,int depth=0,int offset=0) {
    require(size>=0&&size<=8&&other>=0&&other<=8&&copies>=1&&copies<=3&&depth>=0&&depth<=24&&offset>=0,
        "mixed query outside declared bounds");
    LanguageReceipt r;r.id=id;r.scene.complete=true;
    auto entity=[&](int n){return "e"+std::to_string(offset+n);};
    const auto a=entity(100),b=kind==Kind::numeric_count?std::to_string(other):
        kind==Kind::follow_endpoint?entity(other?100+depth:500):entity(200);
    r.names.insert(a);if(kind==Kind::numeric_count)r.quantities.insert(b);else r.names.insert(b);
    if(kind==Kind::union_query) {
        require(size<4,"union query mask outside declared bounds");
        r.text="permit "+a+" "+b;
        if(size&1)r.scene.facts.push_back({"red",{a,b}});
        if(size&2)r.scene.facts.push_back({"blue",{a,b}});
        r.scene.facts.push_back({"red",{b,a}});r.scene.facts.push_back({"blue",{a,entity(700)}});
        r.answer=size!=0;return r;
    }
    r.computational=true;r.relations={"left","right"};
    const std::string word=kind==Kind::equal_counts?"loma":kind==Kind::numeric_count?"navi":"telo";
    r.text=word+" "+a+" beside "+b+" across left and right";
    if(kind==Kind::follow_endpoint) {
        require(other<=1,"follow target is a Boolean fixture choice");
        for(int i=0;i<depth;++i)r.scene.facts.push_back({"left",{entity(100+i),entity(101+i)}});
        r.answer=bool(other);
    } else {
        for(int n=0;n<size;++n)for(int copy=0;copy<copies;++copy)r.scene.facts.push_back({"left",{a,entity(300+n)}});
        if(kind==Kind::equal_counts)for(int n=0;n<other;++n)for(int copy=0;copy<copies;++copy)
            r.scene.facts.push_back({"right",{b,entity(400+n)}});
        r.answer=size==other;
    }
    return r;
}
// The outcome evaluator parses public request text, then performs ordinary set
// counting or a unique-successor walk. It never reads learned meanings or IDs.
inline bool truth(const std::string& text,const Scene& world) {
    const auto tokens=words(text);
    auto pairs=[&](const std::string& relation) {
        std::set<std::pair<std::string,std::string>> result;
        for(const auto& atom:world.facts)if(atom.predicate==relation&&atom.args.size()==2)
            result.emplace(atom.args[0],atom.args[1]);
        return result;
    };
    if(tokens.size()==3&&tokens[0]=="permit")
        return pairs("red").count({tokens[1],tokens[2]})||pairs("blue").count({tokens[1],tokens[2]});
    require(tokens.size()==8&&tokens[2]=="beside"&&tokens[4]=="across"&&tokens[6]=="and","unknown mixed query surface");
    const auto left=pairs(tokens[5]),right=pairs(tokens[7]);
    auto count=[](const auto& relation,const std::string& subject) {
        size_t value=0;for(const auto& [a,b]:relation)value+=a==subject;return value;
    };
    if(tokens[0]=="loma")return count(left,tokens[1])==count(right,tokens[3]);
    if(tokens[0]=="navi") {
        auto number=quantity(tokens[3]);require(bool(number),"invalid numeric query target");
        return count(left,tokens[1])==size_t(*number);
    }
    require(tokens[0]=="telo","unknown mixed computation");
    std::string at=tokens[1];std::set<std::string> visited;
    while(visited.insert(at).second) {
        std::set<std::string> next;for(const auto& [a,b]:left)if(a==at)next.insert(b);
        if(next.empty())return at==tokens[3];if(next.size()!=1)return false;at=*next.begin();
    }
    return false;
}
inline void export_language(const LanguageReceipt& r,std::ostream& out) {
    out<<"{\"event\":\"language_teaching\",\"id\":"<<quote(r.id)<<",\"source\":"<<quote(r.source)
       <<",\"version\":"<<r.version<<",\"active\":"<<r.active<<",\"generated\":"<<r.generated
       <<",\"text\":"<<quote(r.text)<<",\"answer\":"<<r.answer<<",\"computational\":"<<r.computational<<",\"names\":";
    strings(out,r.names);out<<",\"relations\":";strings(out,r.relations);out<<",\"quantities\":";strings(out,r.quantities);
    out<<",\"scene\":";scene(out,{r.scene,0});out<<"}\n";
}
inline void export_primitive(const reasoning::PrimitiveReceipt& r,std::ostream& out) {
    auto value=[&](const auto& v){out<<'['<<v.type<<','<<v.id<<']';};
    out<<"{\"event\":\"primitive_teaching\",\"id\":"<<quote(r.id)<<",\"source\":"<<quote(r.source)
       <<",\"version\":"<<r.version<<",\"active\":"<<r.active<<",\"cue\":"<<r.cue<<",\"relation\":"<<r.relation
       <<",\"subject\":";value(r.subject);out<<",\"target\":";value(r.target);out<<",\"answer\":";value(r.answer);
    out<<",\"facts\":[";bool first=true;for(const auto& f:r.facts){if(!first)out<<',';first=false;out<<'['<<f.subject<<','<<f.relation<<','<<f.object<<']';}out<<"]}\n";
}
template<class Apply> inline Report transaction(GoalControlledRelationalSession& learner,std::ostream& output,Apply apply) {
    require(!learner.mission_active()&&learner.learning(),"mixed teaching requires a learning mission boundary");
    require(!learner.knowledge.dirty()&&bool(output),"mixed teaching requires rebuilt knowledge and an evidence sink");
    auto backup=learner.knowledge;const auto weights=learner.policy.weights;
    const auto updates=learner.updates(),credits=learner.credits().size();
    std::ostringstream evidence;evidence.copyfmt(output);Report report;report.version_before=backup.version();
    try {
        apply(learner.knowledge,evidence,report);learner.knowledge.rebuild();
        require(learner.policy.weights==weights&&learner.updates()==updates&&learner.credits().size()==credits,
            "raw mixed teaching changed continuing policy or credit");
        report.version_after=learner.knowledge.version();report.active_repairs=learner.knowledge.composed().active_repairs();
        output<<evidence.str();output.flush();require(bool(output),"mixed evidence output failed");return report;
    } catch(...) {learner.knowledge=std::move(backup);throw;}
}
inline Report teach_initial(GoalControlledRelationalSession& learner,std::ostream& output,const std::string& prefix="mixed_initial") {
    return transaction(learner,output,[&](RelationalSession& session,std::ostream& out,Report& report) {
        std::vector<reasoning::PrimitiveReceipt> rows;
        rows.push_back(primitive(prefix+"_follow_empty",0,{0,10},{},{0,10}));
        auto one=primitive(prefix+"_follow_one",0,{0,10},{},{0,11});one.facts={{10,0,11}};rows.push_back(one);
        for(int size=0;size<=2;++size){auto row=primitive(prefix+"_count"+std::to_string(size),2,{0,10},{},{2,size});
            for(int i=0;i<size;++i)row.facts.push_back({10,0,20+i});rows.push_back(row);}
        for(int type=0;type<3;++type)for(int same=0;same<2;++same)
            rows.push_back(primitive(prefix+"_equal"+std::to_string(type)+std::to_string(same),1,{type,0},{type,same?0:1},{1,same}));
        for(const auto& row:rows){require(session.teach_primitive(row),"raw mixed primitive rejected");export_primitive(row,out);++report.primitives;}
        auto teach=[&](Kind kind,int size,int other,int depth=0) {
            auto row=example(prefix+"_language"+std::to_string(report.language),kind,size,other,1,depth);
            require(row.answer==truth(row.text,row.scene),"initial raw label disagrees with independent evaluator");
            require(session.teach_language(row),"raw mixed language rejected");export_language(row,out);++report.language;
        };
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)teach(Kind::equal_counts,a,b);
        for(int a=0;a<3;++a)for(int b=0;b<4;++b)teach(Kind::numeric_count,a,b);
        for(int depth:{0,1})for(int same:{0,1})teach(Kind::follow_endpoint,0,same,depth);
    });
}
inline std::vector<LanguageReceipt> follow_lessons(const std::string& prefix) {
    std::vector<LanguageReceipt> rows;
    for(int depth:{2,3,12,20,24})for(int same:{0,1})
        rows.push_back(example(prefix+"_follow"+std::to_string(rows.size()),Kind::follow_endpoint,0,same,1,depth));
    return rows;
}
inline Report extend(GoalControlledRelationalSession& learner,std::ostream& output,const std::string& prefix="mixed_extension") {
    return transaction(learner,output,[&](RelationalSession& session,std::ostream& out,Report& report) {
        auto teach=[&](const LanguageReceipt& row){require(row.answer==truth(row.text,row.scene),"extension label disagrees with independent evaluator");
            require(session.teach_language(row),"mixed extension language rejected");export_language(row,out);++report.language;};
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)teach(example(prefix+"_count"+std::to_string(report.language),Kind::equal_counts,a,b,2));
        for(int a=0;a<3;++a)for(int b=0;b<4;++b)teach(example(prefix+"_count"+std::to_string(report.language),Kind::numeric_count,a,b,2));
        session.rebuild();
        for(const auto& row:follow_lessons(prefix))teach(row);
        session.rebuild();
        for(int mask=0;mask<4;++mask) {
            auto row=example(prefix+"_union"+std::to_string(mask),Kind::union_query,mask,0);
            concepts::Receipt concept_row;concept_row.id=row.id;concept_row.name="permitted";
            concept_row.scene=row.scene;concept_row.args={"e100","e200"};concept_row.answer=row.answer;
            require(session.teach_concept(concept_row),"mixed union evidence rejected");++report.concepts;
            out<<"{\"event\":\"concept_teaching\",\"id\":"<<quote(concept_row.id)<<",\"source\":"<<quote(concept_row.source)
               <<",\"name\":"<<quote(concept_row.name)<<",\"answer\":"<<concept_row.answer<<",\"args\":";strings(out,concept_row.args);
            out<<",\"scene\":";scene(out,{concept_row.scene,0});out<<"}\n";teach(row);
        }
    });
}
inline Report revise_follow_support(GoalControlledRelationalSession& learner,std::ostream& output,
                                     unsigned expected_version,bool active,const std::string& prefix="mixed_extension") {
    return transaction(learner,output,[&](RelationalSession& session,std::ostream& out,Report&) {
        for(const auto& row:follow_lessons(prefix)) {
            require(session.revise_language(row.id,expected_version,row.answer,active),"mixed follow support revision rejected");
            out<<"{\"event\":\"language_revision\",\"id\":"<<quote(row.id)<<",\"expected_version\":"<<expected_version
               <<",\"active\":"<<active<<",\"answer\":"<<row.answer<<"}\n";
        }
    });
}
class QueryEnvironment final:public Environment {
    StampedScene world_;
    std::vector<ScheduledRequest> schedule_;
public:
    QueryEnvironment(Kind kind,int size,int other,int copies=1,int depth=0,int offset=10000) {
        auto row=example("unused",kind,size,other,copies,depth,offset);world_={row.scene,1};
        PublicGoal goal;goal.id="mixed_query";goal.world_id="mixed_world";goal.text=row.text;
        goal.names=row.names;goal.visible=world_;goal.deadline=64;schedule_.push_back({0,std::move(goal)});
    }
    const std::vector<ScheduledRequest>& schedule()const override{return schedule_;}
    StampedScene read(const std::string& world)const override {require(world=="mixed_world","unknown mixed query world");return world_;}
    StampedScene execute(const std::string&,const Action&)override {throw std::logic_error("query environment has no public motors");}
    bool satisfaction(const PublicGoal& request)const override {require(request.world_id=="mixed_world","unknown query outcome world");return truth(request.text,world_.scene);}
    const char* family()const override{return "raw_computation_and_union_query";}
};
} // namespace laminar::goal_learning::mixed_skill
