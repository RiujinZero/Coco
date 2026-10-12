// Declared 2,000-episode hard-task-weighted curriculum, one persistent learner.
#include "goal_extended_declaration.h"

int main(int argc,char** argv) {
    try {
        gl::require(argc==4,"usage: goal_extended_bootstrap --declare|--run smoke|full PREFIX");
        const std::string mode=argv[1],profile=argv[2],prefix=argv[3];
        gl::require(profile=="smoke"||profile=="full","unknown extended profile");const bool smoke=profile=="smoke";
        const auto declaration=extended_configuration(smoke);
        if(mode=="--declare") {
            gl::require(!std::filesystem::exists(prefix+"_config.json"),"declaration exists");
            std::ofstream config(prefix+"_config.json");gl::require(bool(config),"cannot write declaration");config<<declaration;return 0;
        }
        gl::require(mode=="--run","unknown extended mode");std::ifstream config(prefix+"_config.json");
        gl::require(std::string((std::istreambuf_iterator<char>(config)),{})==declaration,"configuration changed or absent");
        for(const auto* suffix:{"_trace.jsonl","_missions.jsonl","_weights.jsonl","_evaluations.jsonl","_bootstrap.jsonl","_summary.json","_phases.jsonl","_computations.jsonl"})
            gl::require(!std::filesystem::exists(prefix+suffix),"evidence exists");
        MixedOutputs out(prefix);const auto start=Clock::now();GoalControlledRelationalSession learner(1,1316,true);
        learner.knowledge=RelationalSession(1,32768,1024);ms::configure_initial(learner);
        auto fixture=gt::make_fixture();fixture.lessons.apply(learner.knowledge);export_bootstrap(fixture.lessons,out.bootstrap);
        rt::Component reader;reader.teach(learner,out.bootstrap);lt::extend(learner,out.bootstrap,"mixed_long");
        ms::teach_initial(learner,out.bootstrap,"mixed_initial");if(smoke)ms::extend(learner,out.bootstrap,"mixed_extension");
        gl::weights(out.parameters,learner,"initial","continuing");
        const auto probe_config=mixed_config(smoke);mixed_evaluate(learner,reader,probe_config,"before",out);
        const size_t total=smoke?48:2000,warm=smoke?40:1600;
        std::array<size_t,8> occurrences{},successes{};size_t fits=0,warm_success=0,autonomous_success=0;
        for(size_t index=0;index<total;++index) {
            if(!smoke&&index==100) {
                const auto weights=learner.policy.weights;const auto updates=learner.updates();
                learner.knowledge.rebuild();ledger(out.phases,"before_skill_extension",learner);
                ms::extend(learner,out.bootstrap,"mixed_extension");
                gl::require(weights==learner.policy.weights&&updates==learner.updates()&&learner.credits().size()==100,"extension replaced learner state");
                ledger(out.phases,"after_skill_extension",learner);
            }
            const auto family=extended_order[index%extended_order.size()];
            size_t environment_index=8*occurrences[family]+family;++occurrences[family];
            if(family>=3&&(smoke||index>=100))environment_index+=8000;
            const bool guided=index<warm;const std::string phase=guided?"warm":"autonomous";
            auto options=mixed_options(index,family,phase);
            options.hint_probability=assistance(index,smoke);options.epsilon=guided?0.f:.1f;
            options.task_order_seed=1000+unsigned(environment_index);
            auto environment=mixed_environment(family,environment_index,false,smoke,reader,out.bootstrap,options.id);
            const auto began=Clock::now();auto result=gl::run_mission(learner,*environment,options,out.trace,out.missions);
            successes[family]+=result.success;if(guided)warm_success+=result.success;else {autonomous_success+=result.success;gl::require(result.hints==0,"autonomous guidance");}
            export_computations(learner,options.id,out.computations);
            ledger(out.phases,options.id,learner,std::chrono::duration<double>(Clock::now()-began).count());
            if(guided&&(index+1)%20==0) {
                const auto steps=size_t(std::ceil(10000*options.hint_probability));
                const auto prior=learner.updates(),version=learner.knowledge.version();const auto values=learner.empirical_values();
                const auto fitted=learner.fit_balanced_demonstrations(steps,gl::curriculum_demonstration_source);fits+=fitted;
                gl::require(fitted==steps&&learner.updates()==prior+steps&&learner.knowledge.version()==version&&learner.empirical_values()==values,
                            "balanced fitter changed semantics or unexpected update count");
                gl::require(fits<=800000,"declared fit budget exceeded");
                out.phases<<"{\"event\":\"balanced_fit\",\"after_episode\":"<<index+1<<",\"guidance\":"<<options.hint_probability
                    <<",\"fits\":"<<fitted<<",\"cumulative_fits\":"<<fits<<",\"representatives\":"<<demonstration_states(learner)<<"}\n";out.phases.flush();
            }
            if((index+1)%20==0||index+1==total)gl::weights(out.parameters,learner,options.id,"continuing");
            std::cout<<options.id<<" family="<<family_names[family]<<" success="<<result.success<<" decisions="<<result.decisions<<" updates="<<learner.updates()<<'\n'<<std::flush;
            if((!smoke&&(index+1)%400==0)||(smoke&&index+1==total))
                mixed_evaluate(learner,reader,probe_config,"after"+std::to_string(index+1),out);
        }
        std::ofstream summary(prefix+"_summary.json");summary<<std::setprecision(17);
        summary<<"{\"complete\":true,\"episodes\":"<<total<<",\"warm_successes\":"<<warm_success<<",\"autonomous_successes\":"<<autonomous_success
            <<",\"family_counts\":";gl::numbers(summary,occurrences);summary<<",\"family_successes\":";gl::numbers(summary,successes);
        summary<<",\"direct_fits\":"<<fits<<",\"updates\":"<<learner.updates()<<",\"credits\":"<<learner.credits().size()
            <<",\"seconds\":"<<std::chrono::duration<double>(Clock::now()-start).count()<<"}\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
