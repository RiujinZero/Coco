#pragma once
#include "laminar/continual_training.h"
#include "continual_checkpoint.h"
#include "continual_guidance.h"
#include "control_probe_support.h"
#include "primitive_discovery_support.h"
#include <filesystem>
namespace laminar::training {
using namespace laminar::relational;
struct PrivateWorld {Scene truth;std::optional<StampedScene> memory;size_t failures=0;bool sensor_unavailable=false;};
struct Trial {std::string id,text;StampedScene initial;std::set<std::string> names;std::vector<Action> available;bool effect;int price,disabled;};
Trial read_trial(std::istream& input) {
    Trial t;int effect;input>>t.id>>effect>>t.price>>t.disabled;t.effect=bool(effect);auto world=read_world(input);
    t.initial={world.scene,1};t.names={world.names.begin(),world.names.end()};size_t n;input>>std::quoted(t.text)>>n;require(n<=128,"too many affordances");
    for(size_t i=0;i<n;++i)t.available.push_back(read_action(input));return t;
}
void weights(const std::string& path,const std::vector<float>& values) {
    std::ofstream file(path);require(bool(file),"cannot save weights");file<<std::setprecision(9);for(float v:values)file<<v<<'\n';
}
void values(const std::string& path,const ControlledRelationalSession& learner) {
    std::ofstream file(path);require(bool(file),"cannot save learned values");file<<std::setprecision(9);
    for(const auto& [obs,q]:learner.empirical_values()) {for(float v:obs)file<<v<<' ';for(float v:q)file<<v<<' ';file<<'\n';}
}
#include "composed_session_contracts.h"
#include "continual_model_audit.h"
struct Interruption {size_t after=0;Trial child;};
struct Result {bool correct=false,satisfaction=false,timeout=false;size_t decisions=0,motors=0,children_correct=0,children_total=0,interruptions=0;float cost=0;};
Result run(ControlledRelationalSession& learner,const Trial& t,const PrivateWorld& fixture,const Environment& evaluator,
           float epsilon,size_t limit,size_t updates,bool random_credit,const std::string& mode,bool computational=false,const laminar::reasoning::PreferenceTree* readout=nullptr,
           const GuidanceSchedule* guidance=nullptr,size_t completed=0,
           const std::vector<Interruption>* interruptions=nullptr,const std::map<std::string,PrivateWorld>* fixtures=nullptr,ModelAudit* model_audit=nullptr) {
    require(!interruptions||(!learner.learning()&&fixtures&&!guidance),"interleaved evaluation requires a frozen unassisted learner");
    ControlPrices prices;if(t.price){prices.operation[0]=0.07f;prices.operation[2]=0.01f;}
    require(learner.begin(t.id,t.text,t.names,t.initial,t.available,t.effect,prices),"episode rejected");
    Scene world=fixture.truth;uint64_t revision=1;Result result;auto version=learner.knowledge.version();size_t failures=fixture.failures;
    if(guidance) {
        require(learner.learning(),"guidance on a frozen learner");
        std::cout<<"G "<<t.id<<' '<<guidance->phase(completed)<<' '<<guidance->rate(completed)<<'\n';
    }
    size_t interruption=0;
    for(size_t i=0;i<limit&&!learner.done();++i) {
        std::optional<ControlHint> hint;
        if(guidance&&guidance->intervene(completed,i)) {
            auto advice=public_control_advice(learner.observe());
            if(int(advice.operation)!=t.disabled) {
                hint=ControlHint{advice.operation,guidance->source};
                std::cout<<"H "<<t.id<<' '<<i<<' '<<int(advice.operation)<<' '<<guidance->source<<' '<<advice.subgoal<<'\n';
            }
        }
        auto step=learner.choose(epsilon,t.disabled,mode=="memory"||(mode.starts_with("model")&&mode!="model_neural"),readout,hint);
        if(model_audit&&step.rebuilt)model_audit->record(learner.knowledge,t.id,i);
        std::cout<<"S "<<t.id<<' '<<i<<' '<<step.token<<' '<<int(step.operation)<<' '<<step.cost<<' '<<step.predictions<<' '<<step.rebuilt<<' '<<step.reused<<' '<<step.rejected<<' '<<int(step.reply)<<' '<<std::quoted(step.status);
        if(computational)std::cout<<' '<<step.primitive_steps;std::cout<<'\n';
        std::cout<<"O "<<t.id<<' '<<i;for(float v:step.observation)std::cout<<' '<<v;std::cout<<'\n';
        if(step.operation==ControlOp::retrieve||step.operation==ControlOp::ask) {
            auto seen=step.operation==ControlOp::retrieve?fixture.memory:std::optional<StampedScene>{{world,revision}};
            if(step.operation==ControlOp::ask&&fixture.sensor_unavailable)seen.reset();
            require(learner.read(step.token,seen,step.operation==ControlOp::retrieve?"trusted_memory":"verified_environment"),"read callback rejected");
            std::cout<<"U "<<t.id<<' '<<i<<' '<<bool(seen)<<' '<<(seen?seen->revision:0);write_world(seen?seen->scene:Scene{{},false});std::cout<<'\n';
        } else if(!step.actions.empty()) {
            std::vector<Scene> observed;std::vector<bool> executed;
            for(size_t k=0;k<step.actions.size();++k) {
                bool did_execute=failures==0;if(failures)--failures;
                if(did_execute)world=evaluator.execute(step.actions[k],world);++revision;observed.push_back(world);executed.push_back(did_execute);
                std::cout<<"A "<<t.id<<' '<<i<<' '<<k;write_action(step.actions[k]);write_world(world);std::cout<<' '<<did_execute<<'\n';++result.motors;
            }
            require(learner.actions(step.token,observed,executed,revision),"action callback rejected");
        }
        std::cout<<"N "<<t.id<<' '<<i<<' '<<learner.visible().revision;write_world(learner.visible().scene);for(float v:learner.observe())std::cout<<' '<<v;std::cout<<'\n';
        const auto& plan=learner.plan();
        std::cout<<"D "<<t.id<<' '<<i<<' '<<std::quoted(plan.status)<<' '<<plan.reused<<' '<<plan.actions.size();
        for(const auto& a:plan.actions)write_action(a);std::cout<<'\n';
        result.cost+=step.cost;++result.decisions;
        if(interruptions&&interruption<interruptions->size()&&i+1==interruptions->at(interruption).after&&!learner.done()) {
            const auto& child=interruptions->at(interruption++).child;auto remembered=learner.visible();
            require(learner.suspend(),"goal suspension rejected");std::cout<<"SUSPEND "<<t.id<<' '<<child.id<<' '<<i+1<<'\n';
            auto nested=run(learner,child,fixtures->at(child.id),evaluator,0,limit,0,false,mode,computational,readout,nullptr,0,nullptr,nullptr,model_audit);
            result.children_correct+=nested.correct;++result.children_total;++result.interruptions;
            // Context switches provide no new world information. Resume from
            // exactly the public scene/stamp available before this interruption.
            require(learner.resume(t.id,remembered),"goal resumption rejected");
            std::cout<<"RESUME "<<t.id<<' '<<child.id<<' '<<remembered.revision;write_world(remembered.scene);std::cout<<'\n';
        }
    }
    if(interruptions)for(;interruption<interruptions->size();++interruption) {
        ++result.children_total;std::cout<<"SKIP "<<t.id<<' '<<interruptions->at(interruption).child.id<<'\n';
    }
    if(!learner.done())require(learner.timeout(),"cannot close timed-out episode");
    result.satisfaction=evaluator.satisfied(t.id,world);result.timeout=learner.timed_out();
    int reply=int(learner.reply());
    bool original=evaluator.satisfied(t.id,fixture.truth);
    result.correct=!result.timeout&&(t.effect?(reply==1&&result.satisfaction):(reply==int(original)&&sorted(world)==sorted(fixture.truth)));
    require(learner.finish(result.correct,result.satisfaction,updates,random_credit),"terminal feedback rejected");
    if(learner.learning()) {
        const auto& receipt=learner.credits().back();
        std::cout<<"J "<<t.id<<' '<<std::quoted(receipt.evidence)<<' '<<receipt.reward<<' '<<receipt.execution<<'\n';
    }
    std::cout<<"F "<<t.id<<' '<<result.correct<<' '<<result.satisfaction<<' '<<reply<<' '<<result.timeout<<' '<<result.decisions<<' '<<result.motors<<' '<<result.cost<<' '<<version<<' '<<learner.knowledge.version()<<' '<<learner.updates()<<'\n';
    if(interruptions)std::cout<<"NESTED "<<t.id<<' '<<result.children_correct<<' '<<result.children_total<<' '<<result.interruptions<<'\n';
    return result;
}
inline int run_control_training(int argc,char** argv,const ReplayOptions& replay={})try {
    const auto config=ContinualTrainingConfig::named(replay.profile);
    if(argc==3&&std::string(argv[1])=="--emit-primitives") {
        int seed=std::stoi(argv[2]);
        for(int cue=0;cue<3;++cue) {
            std::mt19937 rng(1000+seed);
            for(int size:{0,1,2,3,5,8})for(int i=0;i<(size<=3?8:4);++i) {
                auto e=generate(rng,cue,size,size*100+i);auto& p=e.input;
                if(p.target.type==1)p.target.id%=2;if(cue==1)e.answer={1,p.subject==p.target};
                std::cout<<"P skill"<<cue<<'x'<<size*8+i<<' '<<cue<<' '<<p.relation<<' '<<p.subject.type<<' '<<p.subject.id<<' '<<p.target.type<<' '<<p.target.id<<' '<<e.answer.type<<' '<<e.answer.id<<' '<<p.facts.size();
                for(const auto& f:p.facts)std::cout<<' '<<f.subject<<' '<<f.relation<<' '<<f.object;std::cout<<'\n';
            }
        }
        return 0;
    }
    require(argc==7||argc==8||argc==9,"usage: relational_control_probe input environment worlds seed mode output_prefix [reactive|computational|composed] [discovery_budget]");
    std::string mode=argv[5],prefix=argv[6];require(mode=="learned_progress"||mode=="learned_tree"||mode=="scaled_tree"||mode=="scaled"||mode=="scaled_frozen"||mode=="scaled_shuffled"||mode=="scaled_terminal"||mode=="model"||mode=="model_neural"||mode=="model_frozen"||mode=="model_shuffled"||mode=="model_terminal"||mode=="model_single"||mode=="memory"||mode=="learned"||mode=="direct"||mode=="frozen"||mode=="shuffled"||mode=="terminal","invalid mode");
    bool frozen=mode.ends_with("frozen"),terminal=mode.ends_with("terminal"),shuffled=mode.ends_with("shuffled"),model=mode.starts_with("model")&&!frozen;
    if(replay.canonical){frozen=config.control_frozen(0);terminal=config.terminal_only_credit;shuffled=config.credit_shuffled(0);}
    bool reactive=argc>=8,composed=reactive&&std::string(argv[7])=="composed";
    bool computational=reactive&&(composed||std::string(argv[7])=="computational");
    require(!reactive||computational||std::string(argv[7])=="reactive","invalid body mode");
    int discovery_budget=argc==9?std::stoi(argv[8]):100000;
    require((argc!=9||composed)&&discovery_budget>=0,"invalid acquisition budget");
    bool repeated=reactive&&mode!="model_single";
    bool quantitative=mode.starts_with("scaled");
    bool tree_readout=mode.ends_with("_tree");laminar::reasoning::PreferenceTree readout;
    unsigned seed=unsigned(std::stoul(argv[4]));
    ControlledRelationalSession learner=replay.canonical?config.create(seed):ControlledRelationalSession(seed,true,terminal,mode!="direct",{repeated,config.body.batch,config.body.motors,quantitative},model);Environment environment(argv[2]);
    if(replay.search_progress)learner.knowledge.search_progress(&std::cerr);
    if(composed&&discovery_budget&&!replay.canonical)learner.knowledge.enable_composed_learning(discovery_budget);
    if(replay.canonical)require(mode=="learned_progress"&&composed&&discovery_budget==config.discovery_budget,"canonical profile drift");
    ReplayCheckpoint resume(replay,seed,config.json(seed),{argv[1],argv[2],argv[3]});
    if(replay.canonical){std::ofstream resolved(prefix+".config.json");require(bool(resolved),"cannot save resolved config");resolved<<config.json(seed);}
    std::ofstream discovery_log,attempt_log;uint64_t discovery_version=0;
    if(composed){
        discovery_log.open(prefix+".acquisition.tsv");discovery_log<<"version\tacquired\tcue\tcorrect\ttotal\tevaluations\tfirst\tsteps\n";
        attempt_log.open(prefix+".discovery_attempts.tsv");attempt_log<<"version\tcue\tavailable\tcorrect\ttotal\tevaluations\tfirst\tsteps\tstates\n";
    }
    std::unique_ptr<ModelAudit> model_audit;if(replay.audit_models)model_audit=std::make_unique<ModelAudit>(prefix);
    std::map<std::string,PrivateWorld> worlds;std::ifstream hidden(argv[3]);require(bool(hidden),"missing private worlds");std::string line;
    while(std::getline(hidden,line)) {
        std::istringstream in(line);std::string op,id;in>>op>>id;
        if(op=="W") {int present;uint64_t revision;auto truth=read_world(in);in>>present>>revision;auto memory=read_world(in);PrivateWorld world{truth.scene,{}};if(present)world.memory=StampedScene{memory.scene,revision};worlds[id]=std::move(world);}
        else if(op=="X") {int unavailable;auto& world=worlds.at(id);in>>world.failures>>unavailable;require(world.failures<=8&&(unavailable==0||unavailable==1),"invalid private interruption");world.sensor_unavailable=unavailable;}
        else require(false,"bad private world");
        std::string extra;require(bool(in)&&!(in>>extra),"bad private-world trailing input");
    }
    std::ifstream input(argv[1]);require(bool(input),"missing curriculum");std::optional<ControlledRelationalSession> probe;
    std::vector<float> checkpoint,primitive_checkpoint;uint64_t checkpoint_version=0;size_t train=0,test_correct=0,test_total=0,nested_correct=0,nested_total=0;std::string stage;
    GuidanceSchedule guidance;guidance.seed=seed;bool saved_control_freeze=false;
    std::cout<<std::setprecision(9);weights(prefix+".initial.weights",learner.policy.weights);
    if(reactive)std::cout<<"META reactive "<<repeated<<' '<<config.body.motors<<'\n';
    if(computational){std::cout<<"META computational\n";weights(prefix+".initial.primitive_weights",learner.knowledge.primitives().memory.skills.kernel.routes.W);}
    if(composed)std::cout<<"META composed "<<discovery_budget<<'\n';
    if(quantitative)std::cout<<"META quantitative\n";
    if(tree_readout)std::cout<<"META preference_tree 12 2\n";
    if(mode=="learned_progress")std::cout<<"META evaluation_search_retention\n";
    while(std::getline(input,line)) {
        resume.next_line();
        if(line.empty())continue;std::istringstream in(line);std::string op;in>>op;
        if(replay.canonical) {
            frozen=config.control_frozen(train);shuffled=config.credit_shuffled(train);
            if(frozen&&config.control_ablation_after&&!saved_control_freeze) {
                weights(prefix+".control_freeze.weights",learner.policy.weights);
                values(prefix+".control_freeze.values",learner);saved_control_freeze=true;
            }
        }
        if(op=="P"||op=="C"||op=="L"||op=="M"||op=="N"||op=="T"||op.starts_with("REVISE_"))
            require(!probe&&!learner.active(),"teaching or revision inside evaluation/active episode");
        if(op=="GUIDANCE") {
            require(replay.canonical&&!probe&&!learner.active()&&!train&&!guidance.configured,"guidance must be declared once before training");
            std::string source,extra;long long full=-1,fade=-1;in>>source>>full>>fade;
            require(bool(in)&&!(in>>extra)&&source=="public_control_v1"&&full>=0&&fade>=0&&full<=8192&&fade<=8192&&full+fade>0&&full+fade<=8192,"invalid guidance schedule");
            guidance.configured=true;guidance.source=source;guidance.full=size_t(full);guidance.fade=size_t(fade);in.clear();
            std::cout<<"GUIDANCE "<<source<<' '<<full<<' '<<fade<<' '<<seed<<'\n';
        } else if(op=="P") {
            require(computational&&!probe,"primitive evidence outside live computational profile");
            laminar::reasoning::PrimitiveReceipt r;size_t n;in>>r.id>>r.cue>>r.relation>>r.subject.type>>r.subject.id>>r.target.type>>r.target.id>>r.answer.type>>r.answer.id>>n;
            require(n<=512,"too many primitive facts");r.facts.resize(n);for(auto& f:r.facts)in>>f.subject>>f.relation>>f.object;
            auto id=r.id;require(learner.knowledge.teach_primitive(std::move(r)),"primitive receipt rejected");std::cout<<"T P "<<id<<' '<<learner.knowledge.version()<<'\n';
        } else if(op=="C") {
            concepts::Receipt r;int value;in>>r.id>>r.name>>value;auto world=read_world(in);r.args=world.names;r.scene=world.scene;r.answer=bool(value);
            auto id=r.id;require(learner.knowledge.teach_concept(std::move(r)),"concept receipt rejected");std::cout<<"T C "<<id<<' '<<learner.knowledge.version()<<'\n';
        } else if(op=="L"||op=="M"||op=="N") {
            LanguageReceipt r;int value;in>>r.id>>value;auto world=read_world(in);r.scene=world.scene;r.names={world.names.begin(),world.names.end()};r.answer=bool(value);in>>std::quoted(r.text);
            if(op=="M"||op=="N"){require(computational,"computed language outside profile");r.computational=true;size_t n;in>>n;require(n==2,"invalid relation mentions");for(size_t i=0;i<n;++i){std::string name;in>>name;r.relations.insert(name);}}
            if(op=="N"){size_t n;in>>n;require(n>0&&n<=2,"invalid quantity mentions");for(size_t i=0;i<n;++i){std::string name;in>>name;r.quantities.insert(name);r.names.erase(name);}}
            auto id=r.id;require(learner.knowledge.teach_language(std::move(r)),"language receipt rejected");std::cout<<"T "<<op<<' '<<id<<' '<<learner.knowledge.version()<<'\n';
        } else if(op=="T") {
            TransitionReceipt r;in>>r.id;r.action=read_action(in);r.before=read_world(in).scene;r.after=read_world(in).scene;
            auto id=r.id;require(learner.knowledge.teach_transition(std::move(r)),"transition receipt rejected");std::cout<<"T T "<<id<<' '<<learner.knowledge.version()<<'\n';
        } else if(op.starts_with("REVISE_")) {
            std::string id,extra;unsigned version=0;in>>id>>version;bool accepted=false;
            if(op=="REVISE_CREDIT") {
                int correct=-1,satisfaction=-1;in>>correct>>satisfaction;
                require(bool(in)&&!(in>>extra)&&(correct==0||correct==1)&&(satisfaction==0||satisfaction==1),"invalid credit revision");
                accepted=learner.revise_credit(id,version,bool(correct),bool(satisfaction));
            } else if(op=="REVISE_PRIMITIVE") {
                reasoning::Value value;int active=-1;in>>value.type>>value.id>>active;
                require(bool(in)&&!(in>>extra)&&(active==0||active==1),"invalid primitive revision");
                accepted=learner.knowledge.revise_primitive(id,version,value,bool(active));
            } else if(op=="REVISE_CONCEPT"||op=="REVISE_LANGUAGE") {
                int value=-1,active=-1;in>>value>>active;
                require(bool(in)&&!(in>>extra)&&(value==0||value==1)&&(active==0||active==1),"invalid teaching revision");
                accepted=op=="REVISE_CONCEPT"?learner.knowledge.revise_concept(id,version,bool(value),bool(active))
                    :learner.knowledge.revise_language(id,version,bool(value),bool(active));
            } else if(op=="REVISE_TRANSITION") {
                int active=-1;in>>active;auto after=read_world(in);
                require(bool(in)&&!(in>>extra)&&(active==0||active==1),"invalid transition revision");
                accepted=learner.knowledge.revise_transition(id,version,after.scene,bool(active));
            } else require(false,"unknown revision operation");
            require(accepted,"stale, unauthorized or invalid evidence revision");
            // The strict revision parsers intentionally consumed EOF above.
            in.clear();std::cout<<"REVISION "<<op<<' '<<id<<' '<<version+1<<' '<<learner.knowledge.version()<<'\n';
        } else if(op=="TRAIN") {
            require(!probe,"training inside evaluation");auto trial=read_trial(in);
            float epsilon=config.exploration(train);
            run(learner,trial,worlds.at(trial.id),environment,epsilon,config.training_decisions,frozen?0:config.updates_per_episode,shuffled,mode,computational,nullptr,guidance.configured?&guidance:nullptr,train,nullptr,nullptr,model_audit.get());++train;
            if(composed) {
                const auto& learned=learner.knowledge.composed();const auto& report=learned.report();
                if(report.evaluations&&learned.version()!=discovery_version) {
                    discovery_version=learned.version();
                    discovery_log<<discovery_version<<'\t'<<report.acquired<<'\t'<<report.cue<<'\t'<<report.correct<<'\t'<<report.total<<'\t'<<report.evaluations<<'\t'<<report.first_solution<<'\t'<<report.proposal_steps<<'\n'<<std::flush;
                    for(const auto& a:report.attempts)attempt_log<<discovery_version<<'\t'<<a.cue<<'\t'<<a.previously_available<<'\t'<<a.correct<<'\t'<<a.total<<'\t'<<a.evaluations<<'\t'<<a.first_solution<<'\t'<<a.proposal_steps<<'\t'<<a.program_states<<'\n';
                    attempt_log<<std::flush;
                    std::cerr<<"composed version="<<discovery_version<<" acquired="<<report.acquired<<" score="<<report.correct<<'/'<<report.total<<" first="<<report.first_solution<<std::endl;
                    if(report.acquired) {
                        auto acquired_prefix=prefix+".acquired"+std::to_string(discovery_version);
                        weights(acquired_prefix+".weights",learner.knowledge.primitives().memory.skills.kernel.routes.W);
                        save_policy(report.policy,acquired_prefix+".policy.txt");
                        std::ofstream evidence(acquired_prefix+".evidence.txt");
                        for(const auto& [id,r]:learned.receipts()) {
                            const auto& q=r.question.query;
                            evidence<<id<<' '<<r.version<<' '<<q.cue<<' '<<q.a.type<<' '<<q.a.id<<' '<<q.b.type<<' '<<q.b.id<<' '<<q.relation[0]<<' '<<q.relation[1]<<' '<<r.question.answer.type<<' '<<r.question.answer.id<<' '<<q.facts.size();
                            for(const auto& f:q.facts)evidence<<' '<<f.subject<<' '<<f.relation<<' '<<f.object;evidence<<'\n';
                        }
                    }
                }
            }
            if(reactive&&train%config.consolidate_every==0&&!frozen&&mode!="direct") {
                auto fitted=learner.consolidate(config.consolidation_passes);std::cout<<"CONSOLIDATE "<<train<<' '<<fitted<<' '<<learner.updates()<<'\n';
            }
        } else if(op=="CHECK") {
            in>>stage;require(!probe,"nested checkpoint");checkpoint=learner.policy.weights;checkpoint_version=learner.knowledge.version();
            weights(prefix+"."+stage+".weights",checkpoint);probe=learner;probe->set_learning(false);test_correct=test_total=nested_correct=nested_total=0;
            if(mode=="learned_progress")probe->preserve_search(config.evaluation_preserves_search);
            if(computational){primitive_checkpoint=learner.knowledge.primitives().memory.skills.kernel.routes.W;weights(prefix+"."+stage+".primitive_weights",primitive_checkpoint);}
            values(prefix+"."+stage+".values",learner);
            if(tree_readout) {
                readout.fit(learner.empirical_values());
                std::ofstream file(prefix+"."+stage+".tree");require(bool(file),"cannot save preference tree");file<<std::setprecision(17);
                for(const auto& n:readout.nodes()) {
                    file<<n.feature<<' '<<n.threshold<<' '<<n.left<<' '<<n.right;
                    for(auto count:n.counts)file<<' '<<count;file<<'\n';
                }
            }
            std::cout<<"CHECK "<<stage<<' '<<train<<' '<<checkpoint_version<<' '<<learner.updates()<<'\n';
        } else if(op=="TEST"||op=="INTERLEAVE_TEST") {
            require(bool(probe),"evaluation without checkpoint");auto trial=read_trial(in);
            std::vector<Interruption> interruptions;
            if(op=="INTERLEAVE_TEST") {
                size_t n=0;in>>n;require(n>0&&n<=8,"invalid interruption count");size_t previous=0;
                std::set<std::string> ids{trial.id};
                for(size_t i=0;i<n;++i){size_t after=0;in>>after;require(after>previous&&after<config.evaluation_decisions,"invalid interruption schedule");auto child=read_trial(in);require(ids.insert(child.id).second,"duplicate interrupted goal");interruptions.push_back({after,std::move(child)});previous=after;}
            }
            auto result=run(*probe,trial,worlds.at(trial.id),environment,0,config.evaluation_decisions,0,false,mode,computational,tree_readout?&readout:nullptr,nullptr,0,op=="INTERLEAVE_TEST"?&interruptions:nullptr,&worlds,model_audit.get());
            test_correct+=result.correct;++test_total;nested_correct+=result.children_correct;nested_total+=result.children_total;
        } else if(op=="ENDCHECK") {
            require(probe&&checkpoint==learner.policy.weights&&checkpoint_version==learner.knowledge.version()&&probe->policy.weights==checkpoint&&probe->empirical_values()==learner.empirical_values(),"evaluation mutated trained policy");
            if(computational) {
                require(learner.knowledge.primitives().memory.skills.kernel.routes.W==primitive_checkpoint,"evaluation leaked primitive changes into learner");
                if(replay.canonical)require(probe->knowledge.primitives().memory.skills.kernel.routes.W==primitive_checkpoint,"frozen evaluation acquired or changed primitive weights");
                weights(prefix+"."+stage+".probe_primitive_weights",probe->knowledge.primitives().memory.skills.kernel.routes.W);
            }
            std::cout<<"CHECK_RESULT "<<stage<<' '<<test_correct<<' '<<test_total<<'\n';
            if(nested_total)std::cout<<"CHECK_NESTED "<<stage<<' '<<nested_correct<<' '<<nested_total<<'\n';
            std::cerr<<"mode="<<mode<<" stage="<<stage<<" correct="<<test_correct<<'/'<<test_total<<" updates="<<learner.updates()<<'\n';
            dump(probe->knowledge,stage);probe.reset();
        } else require(false,"unknown control curriculum operation");
        std::string extra;require(bool(in)&&!(in>>extra),"invalid curriculum tail");
        bool safe=!probe&&!learner.active();
        resume.verify(learner,train,safe);
        if(safe&&((op=="TRAIN"&&train%config.consolidate_every==0)||op=="ENDCHECK"||resume.should_stop()))resume.save(learner,train);
        if(safe&&resume.should_stop())break;
    }
    require(!probe,"unfinished evaluation block");resume.finish(learner,train);
    if(composed&&discovery_budget&&!frozen&&(!replay.canonical||replay.lifecycle)&&!resume.stopped()) {auto validation=learner;composed_contracts(validation,prefix);}
    weights(prefix+".final.weights",learner.policy.weights);
    if(computational)weights(prefix+".final.primitive_weights",learner.knowledge.primitives().memory.skills.kernel.routes.W);
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}

} // namespace laminar::training
