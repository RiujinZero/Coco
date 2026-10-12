#pragma once
// Lifecycle assertions over the same trained continuing learner, after frozen
// evaluation. These do not contribute to the controller's benchmark scores.
inline void composed_contracts(ControlledRelationalSession& controller,const std::string& prefix) {
    auto& s=controller.knowledge;s.rebuild();
    if(!s.primitives().memory.available_skills[2])return;
    std::ofstream out(prefix+".lifecycle.tsv");out<<"stage\tversion\tcount_available\tready\tverified_answers\twithheld_answers\n";
    auto initial=s.primitives().memory.skills.kernel.routes.W;auto control=controller.policy.weights;
    const auto direct=s.composed().priors().memory.skills.kernel.routes.W;
    const bool direct_count=s.composed().priors().memory.available_skills[2];
    auto record=[&](const std::string& stage,bool verify) {
        size_t tested=0,withheld=0;
        for(const auto& [id,r]:s.language_receipts())if(r.active) {
            auto answer=s.answer(r.text,r.scene);
            if(answer==Answer::Unknown){++withheld;require(!verify,"integrated current evidence unexpectedly withheld");}
            else {require(answer==Answer(int(r.answer)),"integrated current evidence no longer fits");++tested;}
        }
        weights(prefix+".lifecycle."+stage+".weights",s.primitives().memory.skills.kernel.routes.W);
        out<<stage<<'\t'<<s.version()<<'\t'<<s.primitives().memory.available_skills[2]<<'\t'<<s.ready()<<'\t'<<tested<<'\t'<<withheld<<'\n'<<std::flush;
    };
    record("before",true);
    auto witness=s.language_receipts().find("composed1");
    if(witness==s.language_receipts().end())witness=std::find_if(s.language_receipts().begin(),s.language_receipts().end(),[](const auto& p){return p.second.active&&p.second.computational&&p.second.source=="verified_teacher";});
    require(witness!=s.language_receipts().end(),"no verified composed witness for lifecycle");
    auto raw=witness->second;auto conflict=raw;
    conflict.id="composed_conflict";conflict.answer=!raw.answer;conflict.version=1;
    require(s.teach_language(conflict)&&s.answer(raw.text,raw.scene)==Answer::Unknown,"dirty composed language served old answer");
    s.rebuild();s.rebuild();
    require(s.primitives().memory.available_skills[2]==direct_count&&s.primitives().memory.skills.kernel.routes.W==direct&&s.primitives().memory.procedures.empty()&&s.procedures().empty(),"contradiction retained derived computation");
    record("contradiction",false);
    require(s.revise_language(conflict.id,1,conflict.answer,false),"conflict withdrawal rejected");s.rebuild();
    require(s.primitives().memory.skills.kernel.routes.W==initial,"conflict restoration changed neural skills");record("restored",true);
    const auto definition=s.concepts().receipts().at("concept_eligible_0");
    require(s.revise_concept(definition.id,definition.version,!definition.answer),"old concept correction rejected");s.rebuild();
    size_t affected=0;
    for(const auto& [id,r]:s.language_receipts())if(r.active&&r.relations.count(definition.name)) {
        require(s.answer(r.text,r.scene)==Answer::Unknown,"changed concept retained dependent answers");++affected;
    }
    require(affected&&s.ready()&&s.primitives().memory.available_skills[2]&&s.primitives().memory.skills.kernel.routes.W==initial,
            "concept correction disabled independently supported primitive knowledge");record("concept_changed",false);
    require(s.revise_concept(definition.id,definition.version+1,definition.answer),"concept restoration rejected");s.rebuild();
    require(s.primitives().memory.skills.kernel.routes.W==initial,"concept restoration changed retained neural policy");record("concept_restored",true);
    // Credit revisions change both the controller and its raw language/effect
    // evidence. Restore the original receipt, then rebuild the joint model.
    auto credit=std::find_if(controller.credits().begin(),controller.credits().end(),[](const auto& r){return !r.evidence.empty();});
    require(credit!=controller.credits().end(),"no continuing credit evidence");auto old=*credit;
    require(controller.revise_credit(old.id,old.version,old.reward<=0,!old.satisfaction),"controller correction rejected");
    require(s.dirty()&&s.answer(raw.text,raw.scene)==Answer::Unknown,"controller correction left stale semantic answers");
    require(controller.revise_credit(old.id,old.version+1,old.reward>0,old.satisfaction)&&controller.policy.weights==control,"restored controller evidence changed its policy");s.rebuild();
    require(s.primitives().memory.skills.kernel.routes.W==initial,"controller evidence damaged neural primitives");record("credit_restored",true);
    auto language=s.language_receipts();auto executions=s.execution_receipts();
    for(const auto& [id,r]:language)if(r.active) {
        auto e=executions.find(id);
        bool okay=e==executions.end()?s.revise_language(id,r.version,r.answer,false,r.source):s.revise_execution(id,e->second.version,e->second.success,false);
        require(okay,"composed support withdrawal rejected");
    }
    s.rebuild();require(s.primitives().memory.available_skills[2]==direct_count&&s.primitives().memory.skills.kernel.routes.W==direct&&s.primitives().memory.procedures.empty()&&s.procedures().empty(),"derived skill survived removal of all language support");record("withdrawn",false);
    for(const auto& [id,r]:language)if(r.active) {
        auto e=executions.find(id);
        bool okay=e==executions.end()?s.revise_language(id,r.version+1,r.answer,true,r.source):s.revise_execution(id,e->second.version+1,e->second.success,true);
        require(okay,"composed support restoration rejected");
    }
    s.rebuild();require(s.primitives().memory.skills.kernel.routes.W==initial&&controller.policy.weights==control,"evidence restoration did not retain learned parameters");record("final",true);
    std::cerr<<"integrated composed lifecycle passed; final version="<<s.version()<<'\n';
}
