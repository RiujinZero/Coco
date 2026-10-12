#include "grounding_probe_support.h"
using namespace grounding_probe;
struct Turn {
    std::string stage,text,question,gold,phrase,self_text;
    std::set<std::string> candidates; World expected; bool reset;
};
struct Case { std::string group; std::vector<Turn> turns; };
std::vector<Case> read_cases(const std::string& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("missing " + path);
    std::vector<Case> cases;
    while (in.peek()!=EOF) {
        auto h=fields(line(in)); Case c; c.group=h[0];
        for(int i=0;i<std::stoi(h[1]);++i) {
            auto f=fields(line(in)); Turn t; t.stage=f[0]; t.reset=std::stoi(f[1]); t.gold=f[2];
            t.text=line(in); t.question=line(in); t.phrase=line(in);
            std::istringstream names(line(in)); std::string name; while(names>>name) t.candidates.insert(name);
            t.self_text=line(in); std::replace(t.self_text.begin(),t.self_text.end(),'|','\n');
            t.expected=read_world(in); c.turns.push_back(t);
        }
        cases.push_back(c);
    }
    return cases;
}
void teach_references(LanguageReasoner& reader,int seed,const std::string& phase) {
    std::ifstream in("logs/reference_grounding_corpus/sensors_"+std::to_string(seed)+".txt");
    if(!in) throw std::runtime_error("missing reference observations");
    while(in.peek()!=EOF) {
        auto h=fields(line(in)); ReferenceGrounding::Context context;
        for(int i=0;i<std::stoi(h[2]);++i) {
            auto text=line(in); EventGrounding::Interpretation e; assert(reader.events.interpret(text,e)); context.push_back(e);
        }
        auto text=line(in); auto before=read_world(in),after=read_world(in);
        if(h[0]==phase) assert(reader.references.teach_transition(text,context,before,after,reader.events,std::stoull(h[1])));
    }
}
struct Score { int n=0,exact=0,state=0,binding=0,clarification=0,clarification_exact=0,answered=0; };
std::string context_signature(const ReferenceGrounding::Context& context) {
    std::string out;
    for(const auto& e:context) out+=std::to_string(e.relation)+","+std::to_string(e.effect)+","+e.subject+","+e.object+";";
    return out;
}
int main(int argc,char** argv) {
    if(argc!=3) { std::cerr<<"usage: reference_probe seed output\n"; return 2; }
    int seed=std::stoi(argv[1]); auto reader=std::make_shared<LanguageReasoner>(seed);
    reader->grounded_events=reader->shared_meanings=true;
    train_skills(*reader,seed); train_events(*reader,seed);
    Params p; p.seed=seed; p.units=128; p.vocab=256; p.train=false; p.structural=false;
    p.seq_window=65536; p.seq_hash_bits=10; p.seq_consolidate=false;
    Organism org(p); org.attach_reasoner(reader); assert(train_questions(org,*reader,seed)==18);
    teach_references(*reader,seed,"train"); auto initial=reader->references;
    teach_references(*reader,seed,"adapt"); auto adapted=reader->references;
    assert(initial.phrases.size()==4 && adapted.phrases.size()==5);
    auto cases=read_cases("logs/reference_grounding_corpus/test_"+std::to_string(seed)+".txt");
    std::ofstream trace(std::string(argv[2])+".trace.tsv");
    trace<<"mode\tcase\tgroup\tstage\tstatus\texact\tstate\tbinding\tclarification_exact\treply\n";
    std::map<std::string,Score> scores;
    for(const std::string mode : {"learned","no_references","no_roles","adapted"}) {
        reader->references=mode=="adapted"?adapted:initial; reader->grounded_references=mode!="no_references";
        if(mode=="no_roles") for(auto& [phrase,p] : reader->references.phrases) {
            std::set<ReferenceGrounding::Selector> changed;
            for(auto s:p.candidates) { s.role=2; changed.insert(s); } p.candidates=changed;
        }
        const auto signature=reader->references.signature(),event_sig=reader->events.signature(),grammar=reader->grounding.signature();
        const auto programs=reader->memory.procedures; const auto forms=reader->forms; const auto weights=reader->memory.skills.kernel.routes.W;
        std::map<int,size_t> counts; for(const auto& [cue,e]:reader->evidence) counts[cue]=e.size();
        int case_id=0;
        for(const auto& c:cases) {
            bool first=true;
            for(const auto& t:c.turns) {
                std::string input=(first||t.reset?"\n<|endofdialogue|>\n":"\n"); first=false;
                feed(org,input+"USER: "+t.text+"\nASSISTANT:");
                bool answered=reader->status=="answer"||reader->status=="grounded_answer",clarify=reader->status=="clarification";
                const auto status=reader->status;
                bool right_state=reader->complete && state(*reader)==t.expected;
                bool binding=t.gold!="_" && LanguageReasoner::trim(EventGrounding::join(EventGrounding::words(reader->question)))==t.question;
                bool clarification_exact=clarify && LanguageReasoner::trim(reader->pending_reference.phrase)==t.phrase && reader->pending_reference.candidates==t.candidates;
                const auto state_before_emission=state(*reader);
                const auto context_before_emission=context_signature(reader->reference_context);
                const auto pending_before_emission=reader->pending_text;
                auto reply=answered||clarify?generate(org):"<no proposal>";
                assert(state_before_emission==state(*reader)&&context_before_emission==context_signature(reader->reference_context)&&pending_before_emission==reader->pending_text);
                bool exact=answered && reply==t.gold;
                if(clarify) {
                    std::string wanted="Which entity do you mean by \""+t.phrase+"\"?";
                    if(!t.candidates.empty()) { wanted+=" Candidates:"; for(const auto& name:t.candidates) wanted+=" "+name; wanted+="."; }
                    clarification_exact &= reply==wanted;
                }
                auto& s=scores[mode+"/"+c.group+"/"+t.stage]; ++s.n; s.exact+=exact; s.state+=right_state;
                s.binding+=binding; s.clarification+=clarify; s.clarification_exact+=clarification_exact; s.answered+=answered;
                trace<<mode<<'\t'<<case_id<<'\t'<<c.group<<'\t'<<t.stage<<'\t'<<status<<'\t'<<exact<<'\t'<<right_state<<'\t'<<binding<<'\t'<<clarification_exact<<'\t'<<reply<<'\n';
                // A generated role marker and fabricated event must not change
                // either the state, the antecedents or a pending clarification.
                auto before=state(*reader); auto pending=reader->pending_text;
                auto context=context_signature(reader->reference_context);
                for(uint8_t byte:t.self_text) reader->observe(byte,true);
                assert(before==state(*reader)&&pending==reader->pending_text&&context==context_signature(reader->reference_context));
            }
            ++case_id;
        }
        assert(signature==reader->references.signature()&&event_sig==reader->events.signature()&&grammar==reader->grounding.signature());
        assert(programs==reader->memory.procedures&&forms==reader->forms&&weights==reader->memory.skills.kernel.routes.W);
        for(const auto& [cue,e]:reader->evidence) assert(counts.at(cue)==e.size());
        assert(reader->frames.empty()); std::cerr<<"seed="<<seed<<" "<<mode<<" complete\n";
    }
    std::ofstream out(argv[2]); out<<"{\"seed\":"<<seed<<",\"initial_phrases\":"<<initial.phrases.size()<<",\"adapted_phrases\":"<<adapted.phrases.size()<<",\"scores\":{";
    bool first=true;
    for(const auto& [key,s]:scores) {
        if(!first)out<<','; first=false;
        out<<'"'<<key<<"\":{\"n\":"<<s.n<<",\"exact\":"<<s.exact<<",\"state\":"<<s.state<<",\"binding\":"<<s.binding<<",\"clarification\":"<<s.clarification<<",\"clarification_exact\":"<<s.clarification_exact<<",\"answered\":"<<s.answered<<'}';
    }
    out<<"}}\n";
}
