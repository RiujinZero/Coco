#include "interaction_probe_support.h"
#include "laminar/observation_language_learner.h"
std::string knowledge(const LanguageReasoner& reader) {
    const auto& k=reader.observations;std::string out;
    for(const auto& [a,r,b]:k.positive)out+="p,"+reader.names[a]+","+std::to_string(r)+","+reader.names[b]+";";
    for(const auto& [a,r,b]:k.negative)out+="n,"+reader.names[a]+","+std::to_string(r)+","+reader.names[b]+";";
    for(const auto& [a,r]:k.closed_rows)out+="r,"+reader.names[a]+","+std::to_string(r)+";";
    for(const auto& [r,b]:k.closed_columns)out+="c,"+std::to_string(r)+","+reader.names[b]+";";
    return out;
}
int main(int argc,char** argv) {
    assert(argc==2);int seed=std::stoi(argv[1]);auto reader=prior(seed);
    auto prior_events=reader->events.signature();auto weights=reader->memory.skills.kernel.routes.W;
    ObservationLanguageLearner learner;auto training=lessons("logs/observation_language/train.txt");
    bool learned=learner.fit(training,*reader);
    const auto root="logs/observation_language/"+std::to_string(seed);
    std::ofstream model(root+".model.txt");model<<"status\t"<<learner.status<<"\naccepted\t"<<learner.accepted<<"\nrejected\t"<<learner.rejected<<"\nchecked\t"<<learner.checked<<"\nskipped\t"<<learner.skipped<<'\n'<<learner.model.signature();
    for(const auto& [form,programs]:learner.questions){model<<"question\t"<<form<<'\t';for(const auto& p:programs)model<<Meaning::from_program(p,form.find("$1")==std::string::npos?1:2).key()<<';';model<<'\n';}model.flush();
    if(!learned){std::cerr<<"seed="<<seed<<" status="<<learner.status<<'\n';return 1;}
    assert(learner.publish(*reader));
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p);org.attach_reasoner(reader);
    std::ofstream trace(root+".trace.tsv");trace<<"mode\tcase\treply\tstatus\tcomplete\tworlds\tknowledge\n";
    auto model_signature=reader->observation_language.signature();auto grammar=reader->grounding.signature();auto programs=reader->memory.procedures;
    auto evaluate=[&](const std::string& mode) {
        std::ifstream in("logs/observation_language/test.txt");assert(in);int index=0;
        std::vector<std::string> previous;
        while(in.peek()!=EOF) {
            auto h=fields(line(in));int n=std::stoi(h[1]);std::vector<std::string> story;
            for(int j=0;j<n;++j)story.push_back(line(in));
            bool continuing=!previous.empty()&&previous.size()<=story.size()&&std::equal(previous.begin(),previous.end(),story.begin());
            std::string text=continuing?"\n":"\n<|endofdialogue|>\n";
            for(size_t j=continuing?previous.size():0;j<story.size();++j)text+=(continuing?"ENVIRONMENT: ":"CONTEXT: ")+story[j]+"\n";
            previous=std::move(story);
            const auto q=line(in);
            text+="USER: "+q+"\nASSISTANT:";stream(org,text);
            auto answer=generate(org);
            trace<<mode<<'\t'<<index++<<'\t'<<answer<<'\t'<<reader->status<<'\t'<<reader->complete<<'\t'<<reader->possible_worlds.size()<<'\t'<<knowledge(*reader)<<'\n';
        }
    };
    evaluate("grounded");
    assert(reader->events.signature()==prior_events&&reader->memory.skills.kernel.routes.W==weights&&reader->observation_language.signature()==model_signature&&reader->grounding.signature()==grammar&&reader->memory.procedures==programs);
    auto full=reader->observation_language;
    for(auto it=reader->observation_language.frames.begin();it!=reader->observation_language.frames.end();)
        if(std::any_of(it->second.candidates.begin(),it->second.candidates.end(),[](const auto& rule){return rule.kind==ObservationLanguage::FORGET_ROW||rule.kind==ObservationLanguage::FORGET_COLUMN;}))it=reader->observation_language.frames.erase(it);else ++it;
    evaluate("no_visibility_meanings");reader->observation_language=full;
    for(auto& [cue,program]:reader->memory.procedures)for(int& op:program)if(op>=16&&op<=19)op-=op>=18?6:12;
    evaluate("chain_read");reader->memory.procedures=programs;
    std::cerr<<"seed="<<seed<<" grounded observation frames="<<full.frames.size()<<" questions="<<learner.questions.size()<<'\n';
}
