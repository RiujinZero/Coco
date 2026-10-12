#include "../apps/grounding_probe_support.h"
#include "laminar/bootstrap_grounding.h"
using namespace grounding_probe;
using Learner=BootstrapGrounding;
int main() {
    LanguageReasoner reader(0);train_skills(reader,0);
    std::vector<Learner::Episode> examples;
    for(int i=0;i<12;++i) {
        std::string a="actor"+std::to_string(i%4),b="actor"+std::to_string((i+1)%4);
        std::string x="place"+std::to_string(i%6),y="place"+std::to_string((i+1)%6);
        std::vector<std::string> story{a+" dax "+x+".",b+" dax "+x+".",y+" vop "+a+"."};
        examples.push_back({story,"zun "+a+"?",y,examples.size()});
        examples.push_back({story,"zun "+b+"?",x,examples.size()});
    }
    Learner learner;assert(learner.fit(examples,reader.memory));assert(learner.publish(reader));
    assert(reader.forms.size()==1&&reader.events.frames.size()==2);
    assert(!learner.publish(reader)); // channel meanings cannot overwrite an existing model
    reader.reset_episode();
    const std::string input="USER: nova dax alpha. pax dax alpha. beta vop nova. zun pax?\nASSISTANT:";
    for(uint8_t c:input)reader.observe(c,false);
    assert(reader.reply==" alpha\n");
    for(uint8_t c:std::string("\nUSER: zun nova?\nASSISTANT:"))reader.observe(c,false);
    assert(reader.status=="uncertain_answer"); // correlated weak evidence cannot justify beta
    for(int i=0;i<12;++i) {
        std::string a="actor"+std::to_string(i%4),b="actor"+std::to_string((i+1)%4);
        std::string x="place"+std::to_string(i%6),y="place"+std::to_string((i+1)%6);
        for(bool passive:{false,true}) {
            auto say=[&](const std::string& s,const std::string& o){return passive?o+" vop "+s+".":s+" dax "+o+".";};
            std::vector<std::string> prefix{say(a,x),say(b,x)};
            examples.push_back({prefix,"zun "+a+"?",x,examples.size()});
            prefix.push_back(say(a,y));examples.push_back({prefix,"zun "+a+"?",y,examples.size()});
            examples.push_back({prefix,"zun "+b+"?",x,examples.size()});
        }
    }
    assert(learner.fit(examples,reader.memory));
    reader.events.frames.clear();reader.forms.clear();reader.memory.procedures.clear();
    assert(learner.publish(reader));auto signature=reader.events.signature();
    for(const auto& [key,frame]:reader.events.frames)assert(frame.candidates.size()==1);
    for(uint8_t c:input)reader.observe(c,false);
    assert(reader.reply==" alpha\n");
    for(uint8_t c:std::string("\nUSER: zun nova?\nASSISTANT:"))reader.observe(c,false);
    assert(reader.reply==" beta\n");
    // Self-generated contradictory feedback cannot alter the lexical inventory
    // or constrain any candidate. Missing feedback is not supervision either.
    auto ignored=examples;ignored.push_back({{"foreign words."},"alien prompt?","poison",999,true});
    ignored.push_back({{"more foreign words."},"another prompt?","",1000,false});
    Learner safe;assert(safe.fit(ignored,reader.memory));LanguageReasoner copy(0);assert(safe.publish(copy));
    assert(copy.events.signature()==signature&&safe.lexicon==learner.lexicon&&safe.rejected==2);
    // Contradictory answers to the same visible history leave no joint model.
    auto conflict=examples;auto wrong=examples.front();wrong.id=2000;wrong.answer="place0";conflict.push_back(wrong);
    Learner contradicted;assert(!contradicted.fit(conflict,reader.memory));
    assert(contradicted.status=="conflicting_evidence"&&!contradicted.publish(copy));
    // A too-small search budget cannot manufacture evidence by truncation.
    Learner bounded;bounded.assignment_budget=1;assert(!bounded.fit(examples,reader.memory));
    assert(bounded.status=="insufficient_search_evidence"&&!bounded.publish(copy));
    // The learner explicitly declines the currently unsupported mixed-template
    // regime instead of silently binding a semantic word as an entity.
    auto mixed=examples;mixed.push_back({examples[0].observations,"different long query actor0?","place1",3000});
    Learner unsupported;assert(!unsupported.fit(mixed,reader.memory));
    assert(unsupported.status=="unsupported_question_templates");
    // A changing query word with no story referent cannot be treated as a name
    // merely because it differs at one position in an otherwise similar query.
    auto near=examples;auto changed=examples.front();changed.id=4000;changed.question="newquery actor0?";near.push_back(changed);
    Learner semantic_word;assert(!semantic_word.fit(near,reader.memory));
    assert(semantic_word.status=="unsupported_question_templates");
    std::cout<<"bootstrap grounding contracts passed\n";
}
