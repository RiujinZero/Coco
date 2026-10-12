#include "../apps/grounding_probe_support.h"
#include "laminar/incremental_grounding.h"
using namespace grounding_probe;
using Episode=BootstrapGrounding::Episode;

int main() {
    LanguageReasoner prior(0);train_skills(prior,0);
    // The inverse adapters run the existing learned micro-policy, including
    // its terminal coverage checks. They do not introduce question recipes.
    auto& memory=prior.memory;const auto weights=memory.skills.kernel.routes.W;
    FlatQuery q;q.a={0,30};q.b={0,99};q.relation[0]=0;q.relation[1]=1;
    q.facts={{10,1,30},{20,1,30},{99,0,30},{30,1,40}};
    assert((memory.execute(q,{15,9})==Value{2,2}));
    auto k=std::make_shared<FactKnowledge>();q.knowledge=k;
    assert(!memory.execute(q,{15,9}).present()&&memory.missing_evidence);
    k->closed_rows.emplace(30,1); // row coverage cannot authorize an inverse read
    assert(!memory.execute(q,{15,9}).present()&&memory.missing_evidence);
    k->closed_columns.emplace(1,30);assert((memory.execute(q,{15,9})==Value{2,2}));
    q.facts={{10,1,30},{20,1,10}};
    assert(!memory.execute(q,{13,9}).present()&&memory.missing_evidence);
    k->closed_columns.emplace(1,10);k->closed_columns.emplace(1,20);
    assert((memory.execute(q,{13,9})==Value{0,20}));
    k->forget_column(1,10);assert(!memory.execute(q,{13,9}).present()&&memory.missing_evidence);
    assert(memory.skills.kernel.routes.W==weights);
    assert(Meaning::from_program({15,9},1).type()==2);
    assert(Meaning::from_program({13,9},1).type()==0);
    auto inverse=memory;inverse.inverse_enabled=true;
    std::vector<VerifiedQuestion> counts;
    for(int i=0;i<12;++i) {
        FlatQuery example;example.cue=900;example.a={0,30};example.b={0,99};example.relation[0]=0;example.relation[1]=1;
        example.facts={{30,1,80},{30,0,81},{90,0,30}};
        for(int j=0;j<i%4;++j)example.facts.push_back({10+j,1,30});
        counts.push_back({example,{2,i%4}});
    }
    auto discovered=inverse.learn(counts);assert(discovered.found);
    assert(Meaning::from_program(discovered.program,1).key()=="15($0,)");

    // Establish the anchor from answers, then introduce opaque possession and
    // transfer frames with differently phrased questions about the same worlds.
    std::vector<Episode> old,lessons;
    auto add=[](auto& rows,const auto& story,const std::string& question,const std::string& answer){rows.push_back({story,question,answer,rows.size()});};
    for(int i=0;i<12;++i) {
        auto a="actor"+std::to_string(i%4),b="actor"+std::to_string((i+1)%4);
        auto x="room"+std::to_string(i%3),y="room"+std::to_string((i+1)%3);
        std::vector<std::string> story{a+" dax "+x+".",b+" dax "+x+"."};
        add(old,story,"zun "+a+"?",x);add(old,story,"zun "+b+"?",x);
        story.push_back(a+" dax "+y+".");add(old,story,"zun "+a+"?",y);add(old,story,"zun "+b+"?",x);
    }
    BootstrapGrounding bootstrap;assert(bootstrap.fit(old,memory));assert(bootstrap.publish(prior));
    const auto old_events=prior.events.signature(),old_grammar=prior.grounding.signature();
    const auto old_forms=prior.forms;
    const auto old_programs=prior.memory.procedures;
    for(int i=0;i<12;++i)for(bool transfer:{false,true}) {
        auto a="actor"+std::to_string(i%4),b="actor"+std::to_string((i+1)%4);
        auto x="room"+std::to_string(i%3),y="room"+std::to_string((i+1)%3);
        auto o="item"+std::to_string(i%4),p="item"+std::to_string((i+1)%4);
        std::vector<std::string> story{a+" dax "+x+".",b+" dax "+y+".",a+" kep "+o+".",a+" kep "+p+".",b+" kep "+o+"."};
        if(transfer)story.push_back(b+" vot "+o+" "+a+".");
        add(lessons,story,"holder "+o+"?",transfer?a:b);add(lessons,story,"holder "+p+"?",a);
        add(lessons,story,"quantity "+a+"?",transfer?"2":"1");add(lessons,story,"quantity "+b+"?",transfer?"0":"1");
        add(lessons,story,"zun "+a+"?",x);add(lessons,story,"zun "+b+"?",y);
    }
    IncrementalGrounding learned;assert(learned.fit(lessons,prior));
    assert(learned.rejected==0&&learned.skipped==0);
    // Fitting is observational. Even successful discovery must not mutate prior.
    assert(prior.events.signature()==old_events&&prior.grounding.signature()==old_grammar);
    assert(prior.forms==old_forms&&prior.memory.procedures==old_programs&&memory.skills.kernel.routes.W==weights);
    auto stale=prior;stale.memory.skills.kernel.routes.W[0]+=1;assert(!learned.publish(stale));
    stale=prior;stale.relations[0]=7;assert(!learned.publish(stale));
    stale=prior;stale.forms["new form "]=99;assert(!learned.publish(stale));
    auto ignored=lessons;ignored.push_back({{"unbound nonsense."},"alien?","poison",900,true});ignored.push_back({{},"alien?","",901});
    IncrementalGrounding safe;assert(safe.fit(ignored,prior)&&safe.rejected==2);
    assert(safe.questions==learned.questions&&safe.frames==learned.frames&&safe.surface.lexicon==learned.surface.lexicon);
    auto weak=lessons;weak.push_back({lessons[0].observations,"echo actor0?","actor0",950});
    IncrementalGrounding ambiguous;assert(!ambiguous.fit(weak,prior));
    assert(ambiguous.status=="ambiguous_questions"&&!ambiguous.publish(prior));
    auto conflict=lessons;auto wrong=lessons.front();wrong.answer="actor3";wrong.id=1000;conflict.push_back(wrong);
    // Use an observed but incorrect entity so this is valid counterevidence.
    conflict.back().answer="actor0";
    IncrementalGrounding contradicted;assert(!contradicted.fit(conflict,prior));
    assert(contradicted.status=="conflicting_evidence"&&!contradicted.publish(prior));
    IncrementalGrounding bounded;bounded.assignment_budget=1;assert(!bounded.fit(lessons,prior));
    assert(bounded.status=="insufficient_search_evidence"&&!bounded.publish(prior));
    assert(learned.publish(prior));assert(!learned.publish(prior));
    for(const auto& [f,c]:old_forms)assert(prior.forms.at(f)==c&&prior.memory.procedures.at(c)==old_programs.at(c));
    assert(memory.skills.kernel.routes.W==weights);
    std::string input="USER: nova dax alpha. pax dax beta. nova kep coin. pax kep book. pax vot book nova. quantity nova?\nASSISTANT:";
    for(uint8_t c:input)prior.observe(c,false);
    assert(prior.reply==" 2\n");
    auto before=state(prior);for(uint8_t c:std::string("USER: pax kep coin."))prior.observe(c,true);assert(state(prior)==before);
    std::cout<<"incremental grounding and inverse coverage contracts passed\n";
}
