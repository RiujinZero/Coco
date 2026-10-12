#include "grounding_probe_support.h"
#include "laminar/answer_grounding.h"
using namespace grounding_probe;
struct Example { std::string phase,before,event,followup,question,gold; uint64_t scene,observation; };
std::vector<Example> examples(int seed) {
    std::ifstream in("logs/answer_grounding_corpus/train_"+std::to_string(seed)+".txt");
    if(!in) throw std::runtime_error("missing answer teaching"); std::vector<Example> out;
    while(in.peek()!=EOF) {
        auto h=fields(line(in)); Example e; e.phase=h[0]; e.scene=std::stoull(h[1]);e.observation=std::stoull(h[2]);
        e.before=line(in);e.event=line(in);e.followup=line(in);e.question=line(in);e.gold=line(in);out.push_back(e);
    }
    return out;
}
std::string teach(AnswerGrounding& learner,Organism& org,LanguageReasoner& r,const Example& e,const std::string& gold,bool self=false) {
    feed(org,"\n<|endofdialogue|>\nUSER: "+e.before+" "+e.question+"\nASSISTANT:");
    auto meaning=r.question_meaning(); assert(r.prepared && meaning.type()>=0);
    Value answer;
    if(gold!="_") {
        if(std::all_of(gold.begin(),gold.end(),[](unsigned char c){return std::isdigit(c);}))answer={2,std::stoi(gold)};
        else answer={0,r.entities.at(gold)};
    }
    std::vector<EventGrounding::Interpretation> followup;
    if(!e.followup.empty()) { EventGrounding::Interpretation event;assert(r.events.interpret(e.followup,event));followup.push_back(event); }
    return learner.teach(e.event,r.names,r.query,meaning,answer,r.memory,e.scene,e.observation,self,followup);
}
struct Turn { std::string stage,gold,weak,text; };
struct Case { std::string group; World before,after; std::vector<Turn> turns; };
std::vector<Case> cases(int seed) {
    std::ifstream in("logs/answer_grounding_corpus/test_"+std::to_string(seed)+".txt");
    if(!in)throw std::runtime_error("missing answer tests");std::vector<Case> out;
    while(in.peek()!=EOF) {
        Case c;c.group=line(in);c.before=read_world(in);c.after=read_world(in);
        for(int i=0;i<3;++i) { auto h=fields(line(in));c.turns.push_back({h[0],h[1],h[2],line(in)}); }
        out.push_back(c);
    }
    return out;
}
std::string worlds_digest(const LanguageReasoner& r) {
    std::vector<std::string> encoded;
    for(const auto& world:r.possible_worlds) {
        World named;for(auto [a,rel,b]:world.facts)named.emplace(r.names.at(a),rel,r.names.at(b));
        std::string s;for(const auto& [a,rel,b]:named)s+=a+"\t"+std::to_string(rel)+"\t"+b+"\n";encoded.push_back(s);
    }
    std::sort(encoded.begin(),encoded.end());uint64_t hash=14695981039346656037ull;
    for(const auto& s:encoded) { for(unsigned char c:s){hash^=c;hash*=1099511628211ull;}hash^=30;hash*=1099511628211ull; }
    return std::to_string(hash);
}
void model_json(std::ostream& out,const EventGrounding& model,const AnswerGrounding& learner) {
    out<<'{';bool first=true;
    for(const auto& [key,frame]:learner.frames) {
        if(!first)out<<',';first=false;const auto& f=model.frames.at(key);
        out<<'"'<<key<<"\":{\"scenes\":"<<f.scenes.size()<<",\"candidates\":[";bool initial=true;
        for(const auto& rule:f.candidates) {if(!initial)out<<',';initial=false;out<<'['<<rule.relation<<','<<rule.subject<<','<<rule.object<<','<<rule.effect<<']';}
        out<<"]}";
    }
    out<<'}';
}
struct Score {int n=0,exact=0,uncertain=0,conflicted=0,state=0,contains_truth=0,answered=0;};
int main(int argc,char** argv) {
    if(argc!=3){std::cerr<<"usage: answer_grounding_probe seed output\n";return 2;}
    int seed=std::stoi(argv[1]);auto reader=std::make_shared<LanguageReasoner>(seed);
    reader->grounded_events=reader->shared_meanings=true;train_skills(*reader,seed);train_events(*reader,seed);
    Params p;p.seed=seed;p.units=128;p.vocab=256;p.train=false;p.structural=false;p.seq_window=65536;p.seq_hash_bits=10;p.seq_consolidate=false;
    Organism org(p);org.attach_reasoner(reader);assert(train_questions(org,*reader,seed)==18);
    auto training=examples(seed);AnswerGrounding full,weak;
    for(const auto& e:training) if(e.phase=="weak")assert(!teach(full,org,*reader,e,e.gold).empty());
    weak=full;
    for(const auto& e:training) {
        if(e.phase=="contrast")assert(!teach(full,org,*reader,e,e.gold).empty());
        if(e.phase=="repeat")assert(!teach(weak,org,*reader,e,e.gold).empty());
    }
    auto sig=full.signature();const auto first=*std::find_if(training.begin(),training.end(),[](const auto& e){return e.phase=="contrast";});
    assert(teach(full,org,*reader,first,"_").empty());assert(teach(full,org,*reader,first,"999",true).empty());assert(sig==full.signature());
    auto base=reader->events,weak_events=base,full_events=base;weak.publish(weak_events);full.publish(full_events);
    auto clean=full;
    const auto key=teach(full,org,*reader,first,"999");auto conflict=base;full.publish(conflict);
    assert(conflict.frames.at(key).candidates.empty());
    teach(full,org,*reader,first,first.gold);auto corrected=base;full.publish(corrected);
    assert(full.signature()==sig && corrected.signature()==full_events.signature());
    Example bad=first;bad.scene=999999;bad.observation=0;teach(full,org,*reader,bad,"999");
    auto withdrawn=base;full.publish(withdrawn);assert(withdrawn.frames.at(key).candidates.empty());
    assert(full.retract(key,bad.scene,bad.observation));full.publish(withdrawn);assert(withdrawn.signature()==full_events.signature());
    auto test=cases(seed);std::map<std::string,Score> scores;
    std::ofstream trace(std::string(argv[2])+".trace.tsv");trace<<"mode\tcase\tgroup\tstage\tstatus\treply\tworlds\tworlds_digest\tstate\tcontains_truth\n";
    reader->uncertain_events=true;
    for(const std::string mode:{"full","weak","first_candidate","conflict","corrected"}) {
        reader->events=mode=="weak"||mode=="first_candidate"?weak_events:mode=="conflict"?conflict:mode=="corrected"?corrected:full_events;
        if(mode=="first_candidate")for(auto& [name,f]:reader->events.frames)if(!f.candidates.empty())f.candidates={*f.candidates.begin()};
        const auto event_sig=reader->events.signature(),grammar=reader->grounding.signature(),evidence_sig=full.signature();
        const auto weights=reader->memory.skills.kernel.routes.W;const auto programs=reader->memory.procedures;const auto forms=reader->forms;
        int index=0;
        for(const auto& c:test) {
            bool first_turn=true;
            for(const auto& t:c.turns) {
                feed(org,(first_turn?"\n<|endofdialogue|>\n":"\n")+std::string("USER: ")+t.text+"\nASSISTANT:");first_turn=false;
                auto& score=scores[mode+"/"+c.group+"/"+t.stage];++score.n;
                bool answered=reader->status=="answer"||reader->status=="grounded_answer";
                auto status=reader->status;auto digest=worlds_digest(*reader);auto before_state=state(*reader);
                std::string reply=reader->reply.empty()?"<no proposal>":generate(org);
                assert(digest==worlds_digest(*reader)&&before_state==state(*reader));
                score.answered+=answered;score.exact+=answered&&reply==t.gold;score.uncertain+=status=="uncertain_answer";
                score.conflicted+=status=="conflicting_event_evidence";
                const auto& expected=t.stage=="before"?c.before:c.after;
                bool exact_state=reader->complete&&state(*reader)==expected;score.state+=exact_state;
                bool truth=false;
                for(const auto& w:reader->possible_worlds) {
                    World named;for(auto [a,r,b]:w.facts)named.emplace(reader->names.at(a),r,reader->names.at(b));truth|=named==expected;
                }
                score.contains_truth+=truth;
                trace<<mode<<'\t'<<index<<'\t'<<c.group<<'\t'<<t.stage<<'\t'<<status<<'\t'<<reply<<'\t'<<reader->possible_worlds.size()<<'\t'<<digest<<'\t'<<exact_state<<'\t'<<truth<<'\n';
            }
            ++index;
        }
        assert(event_sig==reader->events.signature()&&grammar==reader->grounding.signature()&&evidence_sig==full.signature());
        assert(weights==reader->memory.skills.kernel.routes.W&&programs==reader->memory.procedures&&forms==reader->forms);
        std::cerr<<"seed="<<seed<<" "<<mode<<" complete\n";
    }
    std::ofstream out(argv[2]);out<<"{\"seed\":"<<seed<<",\"weak_model\":";model_json(out,weak_events,weak);
    out<<",\"full_model\":";model_json(out,full_events,full);out<<",\"scores\":{";bool first_score=true;
    for(const auto& [key,s]:scores){if(!first_score)out<<',';first_score=false;out<<'"'<<key<<"\":{\"n\":"<<s.n<<",\"exact\":"<<s.exact<<",\"uncertain\":"<<s.uncertain<<",\"conflicted\":"<<s.conflicted<<",\"state\":"<<s.state<<",\"contains_truth\":"<<s.contains_truth<<",\"answered\":"<<s.answered<<'}';}
    out<<"}}\n";
}
