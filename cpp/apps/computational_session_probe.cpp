// Evaluator-only worlds and terminal labels. Production code receives no
// recipes, hidden family ids, successful microinstructions or policy files.
#include "primitive_discovery_support.h"
#include "laminar/relational_control.h"
#include <filesystem>
using namespace laminar::relational;
namespace concepts=laminar::concepts;
struct Case {int family=0;std::string text,a,b,r,s;Scene world;bool gold=false;};
std::string entity(int serial,int n){return "e"+std::to_string(serial)+"x"+std::to_string(n);}
std::string phrase(int family,const std::string& a,const std::string& b,const std::string& r,const std::string& s) {
    return std::string(family==0?"vex ":family==1?"pim ":"zot ")+a+" with "+b+" via "+r+" and "+s;
}
Case make_case(std::mt19937& rng,int family,int size,int serial) {
    Case c;c.family=family;c.a=entity(serial,0);c.b=entity(serial,1);
    c.r=rng()%2?"left":"right";c.s=c.r=="left"?"right":"left";
    int n=size,m=rng()%2?n:(n+1)%13;int fresh=2;
    auto next=[&](){return entity(serial,fresh++);};
    auto add=[&](std::string r,std::string a,std::string b){c.world.facts.push_back({r,{a,b}});};
    if(family==0) {
        for(int i=0;i<n;++i)add(c.r,c.a,next());
        for(int i=0;i<m;++i)add(c.s,c.b,next());
        c.gold=n==m;
    } else if(family==1) {
        std::string x=c.a,y=c.b;
        for(int i=0;i<n;++i){auto z=next();add(c.r,x,z);x=z;}
        c.gold=rng()%2&&n>0;
        int depth=1+int(rng()%std::max(1,size));
        for(int i=0;i<depth;++i){auto z=(c.gold&&i+1==depth)?x:next();add(c.s,y,z);y=z;}
    } else {
        std::string x=c.a;int depth=1+int(rng()%4);
        for(int i=0;i<depth;++i){auto z=next();add(c.r,x,z);x=z;}
        for(int i=0;i<n;++i)add(c.s,x,next());
        for(int i=0;i<m;++i)add(c.s,c.b,next());
        c.gold=n==m;
    }
    // Duplicate input records and irrelevant relation; neither changes gold.
    if(!c.world.facts.empty())c.world.facts.push_back(c.world.facts[rng()%c.world.facts.size()]);
    add("noise",c.a,c.b);std::shuffle(c.world.facts.begin(),c.world.facts.end(),rng);
    c.text=phrase(family,c.a,c.b,c.r,c.s);return c;
}
LanguageReceipt language(const Case& c,std::string id) {
    LanguageReceipt r;r.id=std::move(id);r.text=c.text;r.scene=c.world;r.answer=c.gold;
    r.names={c.a,c.b};r.relations={c.r,c.s};r.computational=true;return r;
}
void save_weights(const RelationalSession& s,const std::string& path) {
    std::ofstream file(path);file<<std::setprecision(9);
    for(float w:s.primitives().memory.skills.kernel.routes.W)file<<w<<'\n';
}
void primitives(RelationalSession& s,int seed,int cue,const std::string& prefix) {
    std::mt19937 rng(1000+seed);
    for(int size:{0,1,2,3,5,8})for(int i=0;i<(size<=3?8:4);++i) {
        auto e=generate(rng,cue,size,size*100+i);PrimitiveReceipt r;
        r.id="skill"+std::to_string(cue)+"x"+std::to_string(size*8+i);
        r.cue=cue;r.relation=e.input.relation;r.subject=e.input.subject;r.target=e.input.target;r.facts=e.input.facts;r.answer=e.answer;
        // The older low-level stress generator permits arbitrary typed ids.
        // Live evidence enforces Boolean ids 0/1, including cross-type pairs.
        if(r.target.type==1)r.target.id%=2;
        if(cue==1)r.answer={1,r.subject==r.target};
        check(s.teach_primitive(r),"primitive lesson rejected");
    }
    s.rebuild();auto report=s.primitives().acquisition()[cue];
    std::cout<<"primitive "<<cue<<" "<<report.correct<<'/'<<report.total<<" neural="<<s.primitives().memory.available_skills[cue]<<" evaluations="<<report.evaluations<<" steps="<<report.search_steps<<std::endl;
    check(s.primitives().memory.available_skills[cue],"primitive acquisition failed");
    save_weights(s,prefix+".stage"+std::to_string(cue)+".weights.txt");
    save_policy(s.primitives().policies()[cue],prefix+".skill"+std::to_string(cue)+".policy.txt");
    PrimitiveTrace trace(prefix+".skill"+std::to_string(cue));auto policy=s.primitives().policies()[cue];
    for(const auto& [id,r]:s.primitives().receipts())if(r.cue==cue) {
        Example e;e.input={cue,r.relation,r.subject,r.target,r.facts};e.answer=r.answer;e.size=cue==2?r.answer.id:0;
        trace.add(e,run(policy,e.input,128,true),"training");
    }
}
void teach_language_set(RelationalSession& s,int seed,int family) {
    std::mt19937 rng(2000+seed+family*71);
    for(int i=0;i<48;++i)check(s.teach_language(language(make_case(rng,family,i%4,1000+family*100+i),"lang"+std::to_string(family)+"x"+std::to_string(i))),"computed language rejected");
    s.rebuild();
}
void teach_concept(RelationalSession& s) {
    concepts::Receipt r;r.id="eligible_yes";r.name="eligible";r.args={"u","v"};r.answer=true;
    r.scene={{{"edge",{"u","v"}},{"marked",{"u","v"}}},true};
    check(s.teach_concept(r),"concept positive rejected");
    auto positive=r.scene;
    for(int i=0;i<2;++i){r.id="eligible_no"+std::to_string(i);r.answer=false;r.scene=positive;r.scene.facts.erase(r.scene.facts.begin()+i);check(s.teach_concept(r),"concept negative rejected");}
    s.rebuild();check(s.concepts().models().at("eligible").hypotheses.size()==1,"concept did not resolve");
}
Case physical(int serial,int size) {
    Case c;c.a=entity(serial,0);c.b=entity(serial,1);c.r="eligible";c.s="right";
    // Marking the last raw edge is the only way to equalize distinct counts.
    for(int i=0;i<size;++i) {
        auto item=entity(serial,i+2);c.world.facts.push_back({"edge",{c.a,item}});
        if(i+1<size)c.world.facts.push_back({"marked",{c.a,item}});
        c.world.facts.push_back({"right",{c.b,item}});
    }
    c.text=phrase(0,c.a,c.b,c.r,c.s);return c;
}
ControlStep force(ControlledRelationalSession& s,ControlOp op) {
    // Protocol test only: this is not evidence for learned operation selection.
    auto weights=s.policy.weights;std::fill(s.policy.weights.begin(),s.policy.weights.end(),0);
    s.policy.weights[s.policy.output_bias()+int(op)]=1;auto result=s.choose();s.policy.weights=std::move(weights);return result;
}
void log_case(std::ofstream& out,int seed,const std::string& phase,const Case& c,Answer predicted,uint64_t work,const Action* action=nullptr) {
    out<<seed<<'\t'<<phase<<'\t'<<c.family<<'\t'<<c.a<<'\t'<<c.b<<'\t'<<c.r<<'\t'<<c.s<<'\t'<<c.gold<<'\t'<<int(predicted)<<'\t'<<work<<'\t';
    for(const auto& a:c.world.facts)out<<a.predicate<<','<<a.args[0]<<','<<a.args[1]<<';';
    out<<'\t';if(action)out<<action->schema<<','<<action->arguments[0]<<','<<action->arguments[1];out<<'\n';
}
int main(int argc,char** argv)try {
    check(argc==3,"usage: computational_session_probe seed outdir");int seed=std::stoi(argv[1]);
    std::filesystem::create_directories(argv[2]);std::string prefix=std::string(argv[2])+"/"+std::to_string(seed);
    std::ofstream out(prefix+".cases.tsv");out<<"seed\tphase\tfamily\ta\tb\tr\ts\tgold\tpredicted\twork\tfacts\taction\n";
    RelationalSession s(seed);save_weights(s,prefix+".initial.weights.txt");primitives(s,seed,1,prefix);teach_language_set(s,seed,0);
    std::mt19937 rng(9000+seed);auto missing=make_case(rng,0,4,8000);
    check(s.answer(missing.text,missing.world)==Answer::Unknown,"count language answered before skill acquisition");
    primitives(s,seed,2,prefix);
    for(const auto& [name,f]:s.computational_frames())std::cout<<"initial frame "<<name<<" learned="<<f.learned<<" states="<<f.states<<std::endl;
    std::cout<<"missing case result="<<int(s.answer(missing.text,missing.world))<<" gold="<<missing.gold<<std::endl;
    if(!s.computational_frames().begin()->second.learned) {
        auto m=s.primitives().memory;
        for(const auto& [id,l]:s.language_receipts()) {
            auto q=s.computation_input(l.text,l.scene);check(q.has_value(),"diagnostic input unavailable");
            auto value=m.execute(*q,{6,3,7,8,9});
            if(value!=Value{1,int(l.answer)}) {
                auto x=m.execute(*q,{6,9});auto y=m.execute(*q,{1,7,9});
                std::cout<<"composition training failure "<<id<<" counts="<<x.type<<','<<x.id<<'/'<<y.type<<','<<y.id<<" answer="<<value.type<<','<<value.id<<" expected="<<l.answer<<std::endl;
            }
        }
    }
    check(s.answer(missing.text,missing.world)==Answer(int(missing.gold)),"new skill did not enable existing language");
    auto before_weights=s.primitives().memory.skills.kernel.routes.W;
    primitives(s,seed,0,prefix);check(before_weights!=s.primitives().memory.skills.kernel.routes.W,"new primitive did not change neural parameters");
    teach_language_set(s,seed,1);teach_language_set(s,seed,2);
    for(const auto& [name,f]:s.computational_frames()){
        std::cout<<"frame "<<name<<" learned="<<f.learned<<" states="<<f.states<<" code=";
        for(auto a:s.primitives().memory.procedures.at(f.cue))std::cout<<a<<',';std::cout<<std::endl;
    }
    teach_concept(s);
    {std::ofstream programs(prefix+".programs.tsv");programs<<"frame\tinstructions\n";
     for(const auto& [name,f]:s.computational_frames()){programs<<name<<'\t';for(int a:s.primitives().memory.procedures.at(f.cue))programs<<a<<',';programs<<'\n';}}
    TransitionReceipt t;t.id="mark_effect";t.action={"motor",{"u","v"}};
    t.before={{{"marked",{"u","old"}},{"marked",{"other","v"}}},true};t.after=t.before;t.after.facts.push_back({"marked",{"u","v"}});
    check(s.teach_transition(t),"motor evidence rejected");s.rebuild();
    auto weights=s.primitives().memory.skills.kernel.routes.W;auto version=s.version();size_t correct=0,total=0,goals=0;
    for(int family=0;family<3;++family)for(int size:{0,1,2,3,4,8,12})for(int i=0;i<16;++i) {
        auto c=make_case(rng,family,size,10000+family*1000+size*20+i);auto before=s.primitive_steps();auto result=s.answer(c.text,c.world);
        log_case(out,seed,"heldout",c,result,s.primitive_steps()-before);++total;correct+=result==Answer(int(c.gold));
        auto partial=c.world;partial.complete=false;check(s.answer(c.text,partial)==Answer::Unknown,"partial computational world answered");
    }
    for(int size:{1,2,4,8,12})for(int i=0;i<8;++i) {
        auto c=physical(20000+size*10+i,size);std::vector<Action> menu;
        for(int j=0;j<size;++j)menu.push_back({"motor",{c.a,entity(20000+size*10+i,j+2)}});
        auto before=s.primitive_steps();auto result=s.answer(c.text,c.world);log_case(out,seed,"physical_before",c,result,s.primitive_steps()-before);
        auto p=s.plan(c.text,c.world,menu);check(p.actions.size()==1,"learned computation did not produce one-step goal plan");
        auto final=s.predict(p.actions[0],c.world);check(final.has_value(),"prediction missing");
        c.world.facts.push_back({"marked",p.actions[0].arguments});c.world=sorted(c.world);check(*final==c.world,"learned effect differs from environment");c.gold=true;
        before=s.primitive_steps();result=s.answer(c.text,c.world);log_case(out,seed,"physical_after",c,result,s.primitive_steps()-before,&p.actions[0]);
        goals+=result==Answer::True;
    }
    check(weights==s.primitives().memory.skills.kernel.routes.W&&version==s.version(),"evaluation learned");
    {PrimitiveTrace trace(prefix+".neural");auto copy=s.primitives().memory.skills;
     for(int cue=0;cue<3;++cue)for(int size:{0,1,2,3,4,8,12,24,64})for(int i=0;i<8;++i) {
        auto e=generate(rng,cue,size,50000+cue*10000+size*10+i,4);trace.add(e,run(copy,e.input,1024,true),"retained");
     }}
    // Skill ablation keeps the exact language programs. No symbolic fallback.
    auto ablated=s;auto& am=const_cast<PrimitiveStore&>(ablated.primitives()).memory;am.available_skills[2]=false;
    auto physical_case=physical(30000,1);check(ablated.answer(physical_case.text,physical_case.world)==Answer::Unknown,"count primitive not causally required");
    am.available_skills[2]=true;am.available_skills[1]=false;check(ablated.answer(physical_case.text,physical_case.world)==Answer::Unknown,"comparison primitive not required");
    check(s.answer(physical_case.text,physical_case.world)==Answer::False,"clone ablation leaked");
    // Own actions and terminal outcomes must feed the same computational frame.
    Action motor{"motor",{physical_case.a,entity(30000,2)}};auto p=s.issue(physical_case.text,physical_case.world,{motor});
    auto final=std::optional<Scene>(physical_case.world);final->facts.push_back({"marked",motor.arguments});*final=sorted(*final);
    check(p.token&&s.predict(motor,physical_case.world)==final,"own issue failed");
    check(s.feedback(p.token,{*final},{true},true),"own computational feedback rejected");s.rebuild();
    auto eid="execution"+std::to_string(p.token);
    check(s.language_receipts().at(eid).computational&&!s.procedures().empty(),"terminal evidence routed to wrong language model");
    check(s.plan(physical_case.text,physical_case.world,{motor}).reused,"computed goal procedure not reused");
    check(s.revise_execution(eid,1,false),"execution revision rejected");s.rebuild();
    check(s.answer(physical_case.text,*final)==Answer::Unknown&&s.procedures().empty(),"contradictory computational evidence survived");
    check(s.revise_execution(eid,2,true),"execution restoration rejected");s.rebuild();
    check(s.answer(physical_case.text,*final)==Answer::True,"computational evidence restoration failed");
    auto pending=s.issue(physical_case.text,physical_case.world,{motor});
    check(s.revise_primitive("skill2x0",1,s.primitives().receipts().at("skill2x0").answer),"primitive revision failed");
    check(!s.feedback(pending.token,{*final},{true},true),"stale primitive credit accepted");s.rebuild();
    check(s.primitives().memory.skills.kernel.routes.W==weights,"same primitive evidence did not deterministically rebuild");
    auto count_receipts=s.primitives().receipts();
    for(const auto& [id,r]:count_receipts)if(r.cue==2)check(s.revise_primitive(id,r.version,r.answer,false),"primitive withdrawal rejected");
    s.rebuild();check(s.answer(physical_case.text,*final)==Answer::Unknown&&s.procedures().empty(),"withdrawn count evidence left dependent procedure");
    for(const auto& [id,r]:count_receipts)if(r.cue==2)check(s.revise_primitive(id,r.version+1,r.answer,true),"primitive restore rejected");
    s.rebuild();check(s.primitives().memory.skills.kernel.routes.W==weights&&s.plan(physical_case.text,physical_case.world,{motor}).reused,"primitive restoration failed");
    // Old concept correction invalidates derived-count semantics and procedure.
    check(s.revise_concept("eligible_yes",1,false),"concept revision failed");s.rebuild();
    check(s.answer(physical_case.text,*final)==Answer::Unknown&&s.procedures().empty(),"concept dependency stayed stale");
    check(s.revise_concept("eligible_yes",2,true),"concept restoration failed");s.rebuild();
    check(s.plan(physical_case.text,physical_case.world,{motor}).reused,"concept restoration lost procedure");
    ControlledRelationalSession controlled(seed);controlled.knowledge=s;
    check(controlled.begin("computed_query",physical_case.text,{physical_case.a,physical_case.b},{*final,0},{},false),"controlled query begin failed");
    auto reasoning=force(controlled,ControlOp::reason);check(reasoning.primitive_steps>0&&reasoning.cost>0.005f,"neural inference work not metered");
    check(force(controlled,ControlOp::finish).reply==Answer::True&&controlled.finish(true,true,8),"controlled query failed");
    auto rid=controlled.credits().back().evidence;check(controlled.knowledge.language_receipts().at(rid).computational,"controller mislabeled computed language");
    auto nn=controlled.policy.weights;check(controlled.revise_credit("computed_query",1,false,false)&&controlled.revise_credit("computed_query",2,true,true)&&controlled.policy.weights==nn,"coupled computational credit replay failed");
    controlled.knowledge.rebuild();check(controlled.knowledge.answer(physical_case.text,*final)==Answer::True,"controller revision damaged semantics");
    std::cout<<"seed="<<seed<<" questions="<<correct<<'/'<<total<<" physical_goals="<<goals<<"/40 contracts=passed\n";
    check(correct==total&&goals==40,"held-out transfer failed");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
