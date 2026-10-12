// Bounded diagnosis: saved policies, unchanged teacher, fresh matched libraries.
// This is not a reconstruction of the final mission-trained knowledge archive.
#define LAMINAR_GOAL_MIXED_LIBRARY
#include "goal_continuing_bootstrap.cpp"

static std::vector<float> parameters(const std::string& path) {
    std::ifstream input(path);gl::require(bool(input),"missing extracted parameters");
    std::vector<float> values;float value;while(input>>value)values.push_back(value);
    gl::require(values.size()==52647,"wrong policy size");return values;
}
int main(int argc,char** argv) {
    try {
        gl::require(argc==5,"usage: goal_targeted_isolation PREFIX WARM_WEIGHTS FINAL_WEIGHTS SUPPORT");
        const std::string prefix=argv[1];
        gl::require(!std::filesystem::exists(prefix+"_missions.jsonl"),"evidence already exists");
        const auto warm=parameters(argv[2]),final=parameters(argv[3]);MixedOutputs out(prefix);
        std::ifstream support(argv[4]);gl::require(bool(support),"missing demonstration support");
        std::map<std::tuple<std::string,int,int>,std::pair<size_t,size_t>> agreement;
        int group,mask,label;gl::Observation observation;
        laminar::reasoning::BackpropChoice policy(7,1,1316);
        while(support>>group>>mask>>label) {
            for(auto& value:observation)gl::require(bool(support>>value),"truncated support");
            std::vector<int> menu;for(int a=0;a<7;++a)if(mask&(1<<a))menu.push_back(a);
            for(const std::string name:{"warm","final"}) {
                policy.weights=name=="warm"?warm:final;
                auto& count=agreement[{name,group,label}];++count.second;
                count.first+=policy.choose(observation,menu)==label;
            }
        }
        for(const auto& [key,value]:agreement)
            out.phases<<"{\"event\":\"demonstration_agreement\",\"policy\":"<<gl::quote(std::get<0>(key))
                <<",\"group\":"<<std::get<1>(key)<<",\"teacher_action\":"<<std::get<2>(key)
                <<",\"matches\":"<<value.first<<",\"states\":"<<value.second<<"}\n";
        for(bool combined:{false,true}) {
            GoalControlledRelationalSession base(1,1316,true);
            base.knowledge=RelationalSession(1,32768,1024);ms::configure_initial(base);
            gt::make_raw_lessons().apply(base.knowledge);rt::Component reader;
            if(combined)reader.teach(base,out.bootstrap);
            lt::extend(base,out.bootstrap,"mixed_long");
            if(combined){ms::teach_initial(base,out.bootstrap,"mixed_initial");ms::extend(base,out.bootstrap,"mixed_extension");}
            // Exhaust all edge-subset truth cases independently of control.
            size_t correct=0;for(size_t length:{6,9}) {
                std::vector<std::string> nodes;for(size_t i=0;i<=length;++i)nodes.push_back("probe"+std::to_string(i));
                for(uint64_t mask=0;mask<(uint64_t(1)<<length);++mask) {
                    const auto scene=gt::route_scene(nodes,mask);
                    const auto text=std::string(length==6?"span probe0 toward ":"connect probe0 through ")+nodes.back();
                    const auto answer=gt::exact_route(scene,nodes.front(),nodes.back(),length)?Answer::True:Answer::False;
                    correct+=base.knowledge.answer(text,scene)==answer;
                }
            }
            gl::require(correct==576,"route semantics regressed");
            out.phases<<"{\"library\":"<<gl::quote(combined?"combined_bootstrap":"routes_only")<<",\"subset_correct\":"<<correct<<",\"subset_total\":576}\n";
            for(const std::string policy:{"reference","warm","final"})for(size_t family:{1,2})for(size_t cycle=0;cycle<4;++cycle) {
                const auto index=8*cycle+family;auto learner=base;
                learner.policy.weights=policy=="warm"?warm:final;
                const bool reference=policy=="reference";learner.set_learning(reference);
                const std::string group=std::string(combined?"combined_":"isolated_")+policy;
                auto options=mixed_options(index,family,group,!reference);
                options.task_order_seed=10000+unsigned(index);
                options.terminal_updates=0;options.hint_probability=reference?1.:0.;options.epsilon=0;
                options.id=group+"_f"+std::to_string(family)+"_c"+std::to_string(cycle);
                auto environment=mixed_environment(family,index,true,false,reader,out.bootstrap,options.id);
                auto result=gl::run_mission(learner,*environment,options,out.trace,out.missions);
                export_computations(learner,options.id,out.computations);
                std::cout<<options.id<<" success="<<result.success<<" decisions="<<result.decisions<<" motors="<<result.motors<<'\n'<<std::flush;
            }
        }
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
