#include "primitive_discovery_support.h"
int main(int argc,char** argv)try {
    check(argc==4||argc==7,"usage: primitive_checkpoint_probe seed weights output_prefix [units active channels]");
    SkillShape shape;if(argc==7)shape={std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6])};
    int seed=std::stoi(argv[1]);LocalChoice learner(10,seed,shape);std::ifstream input(argv[2]);
    for(auto& w:learner.kernel.routes.W)check(bool(input>>w)&&std::isfinite(w),"invalid primitive checkpoint");
    float extra;check(!(input>>extra)&&input.eof(),"trailing weights");auto before=learner.kernel.routes.W;
    PrimitiveTrace trace(argv[3]);std::mt19937 rng(39000+seed);
    for(int task=0;task<3;++task)for(int size:{0,1,2,3,4,8,12,24,64})for(int i=0;i<12;++i) {
        auto e=generate(rng,task,size,200000+task*10000+size*100+i,4);
        if(e.input.target.type==1)e.input.target.id%=2;if(task==1)e.answer={1,e.input.subject==e.input.target};
        trace.add(e,run(learner,e.input,1024,true),"checkpoint");
    }
    check(learner.kernel.routes.W==before,"checkpoint evaluation learned");
    std::cout<<"324 frozen primitive checkpoint cases executed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
