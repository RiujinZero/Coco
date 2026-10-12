#include "laminar/relational_goal_preview.h"
#include "goal_control_fixture.h"
#include <iostream>

namespace gp=laminar::relational::goal_preview;
using namespace laminar::relational;
using laminar::goal_test::require;

gp::Inputs inputs(const laminar::goal_test::PublicRequestFixture& request,const std::string& candidate) {
    gp::Inputs result;result.parent_text=request.text;result.candidate_text=candidate;
    result.world=request.visible.scene;result.actions=request.actions;
    result.names.assign(request.names.begin(),request.names.end());result.observation_revision=request.visible.revision;
    return result;
}

// Independent evaluator-only walk enumeration. It does not call learned proof,
// preview joins, predicates, or planners, and counts distinct missing edge facts.
std::pair<int,int> reference(const std::vector<Atom>& domain,const Scene& before,const Scene& after,
                             const std::string& from,const std::string& to,int length) {
    std::set<Atom> old(before.facts.begin(),before.facts.end()),next(after.facts.begin(),after.facts.end());
    int x=999,y=999;
    std::function<void(std::string,int,std::set<Atom>)> visit=[&](std::string position,int left,std::set<Atom> path) {
        if(!left) {
            if(position!=to)return;
            int a=0,b=0;for(const auto& edge:path){a+=!old.count(edge);b+=!next.count(edge);}
            x=std::min(x,a);y=std::min(y,b);return;
        }
        for(const auto& edge:domain)if(edge.predicate=="edge"&&edge.args.size()==2&&edge.args[0]==position) {
            auto extended=path;extended.insert(edge);visit(edge.args[1],left-1,std::move(extended));
        }
    };
    visit(from,length,{});return {x,y};
}

int main() {
    auto fixture=laminar::goal_test::make_fixture();auto session=fixture.make_knowledge();
    const auto& request=fixture.request("physical");auto useful=inputs(request,"route n0 toward n3");
    auto irrelevant=inputs(request,"route d0 toward d3");
    gp::Limits limits;limits.work=100000;limits.normalization_atoms=6;
    gp::Prices prices{.007,.003,.002,.011,.013};
    auto version=session.version();auto evidence=session.transition_receipts().size();
    auto a=gp::analyze(session,useful,limits,prices),b=gp::analyze(session,irrelevant,limits,prices);
    require(a.valid&&b.valid,"complete learned preview unavailable");
    require(a.answer==Answer::False&&b.answer==Answer::False&&a.plan.actions.size()==3&&b.plan.actions.size()==3,
            "useful/irrelevant initial satisfied or reachable flags differ");
    require(a.before_missing==6&&a.after_missing==3&&a.delta==3&&a.normalized_delta==.5,
            "useful consequence incorrect");
    require(b.before_missing==6&&b.after_missing==6&&b.delta==0,"irrelevant consequence incorrect");
    require(a.inputs.world==useful.world&&a.inputs.parent_text==useful.parent_text&&a.knowledge_version==version&&
            a.parent.formula.size()==1&&a.parent.formula.front().body.size()==6,"missing preview provenance");
    require(session.version()==version&&session.transition_receipts().size()==evidence&&session.procedures().empty(),
            "private consequence simulation taught or executed");
    require(a.predictions==a.plan.predictions+a.plan.actions.size()+2*useful.actions.size(),"prediction accounting incomplete");
    require(a.work==a.predictions+a.groundings+a.atom_checks+a.answer_calls+a.planner_calls&&
            a.answer_calls==1&&a.planner_calls==1&&a.vm_steps==0,"logical work accounting incomplete");
    double expected=a.predictions*prices.prediction+a.groundings*prices.grounding+a.atom_checks*prices.atom_check+
        a.answer_calls*prices.answer_call+a.planner_calls*prices.planner_call;
    require(std::abs(a.cost-expected)<1e-9,"fixed work prices not charged exactly");

    size_t checked=0;
    for(uint64_t mask=0;mask<64;++mask) {
        auto row=useful;row.world=laminar::goal_test::route_scene({"n0","n1","n2","n3","n4","n5","n6"},mask);
        auto result=gp::analyze(session,row,limits,prices);
        require(result.valid,"small exhaustive world preview failed");
        auto exact=reference(result.potential,row.world,result.predicted,"n0","n6",6);
        require(result.before_missing==exact.first&&result.after_missing==exact.second,"independent minimum disagreement");
        ++checked;
    }

    // Repeated ground atoms count once; two compatible routes can minimize in
    // different worlds. This unit fixture tests the supplied algorithm itself.
    {
        gp::Definition def;def.args={"a","a"};
        def.formula={laminar::concepts::Clause{2,{{"edge",{0,2}},{"edge",{2,3}},{"edge",{3,1}}}}};
        std::vector<Atom> domain{{"edge",{"a","a"}},{"edge",{"a","b"}},{"edge",{"b","c"}},{"edge",{"c","a"}}};
        Scene before{{domain[1],domain[2]},true},after{{domain[0]},true};
        gp::Work work;gp::detail::Meter meter{work,limits,prices};
        auto minimum=gp::detail::minima(def,domain,before,after,meter,{"a","b","c"});
        auto exact=reference(domain,before,after,"a","a",3);
        require(minimum==exact&&minimum==std::pair{1,0},"distinct-atom or alternative-grounding minimum wrong");
        auto reversed=gp::detail::minima(def,domain,after,before,meter,{"a","b","c"});
        require(reversed==std::pair{0,1},"a harmful consequence lost its signed delta");
        domain.push_back({"blue",{"a","a"}});before.facts.push_back(domain.back());
        def.formula.push_back({2,{{"blue",{0,1}}}});
        auto union_minimum=gp::detail::minima(def,domain,before,after,meter,{"a","b","c"});
        require(union_minimum==std::pair{0,0},"positive union minimum failed");
    }

    auto exhausted=limits;exhausted.work=a.work-1;
    auto partial_min=gp::analyze(session,useful,exhausted,prices);
    require(!partial_min.valid&&partial_min.delta==0&&partial_min.normalized_delta==0&&
            partial_min.before_missing==-1&&partial_min.status=="work_budget"&&partial_min.work==exhausted.work,
            "partial enumeration leaked a completed minimum or lost spent work");
    auto just=limits;just.work=a.work;require(gp::analyze(session,useful,just,prices).valid,"exact work boundary rejected");
    auto tiny=limits;tiny.work=1;auto tiny_result=gp::analyze(session,useful,tiny,prices);
    require(!tiny_result.valid&&tiny_result.work==1&&tiny_result.answer_calls==1,"early exhaustion lost call cost");
    auto bounded_plan=limits;bounded_plan.plan_predictions=1;
    auto no_plan=gp::analyze(session,useful,bounded_plan,prices);
    require(!no_plan.valid&&no_plan.status=="candidate_search_budget"&&no_plan.predictions==1&&
            no_plan.work==3&&no_plan.delta==0,"planner exhaustion accepted or omitted spent calls");
    auto partial=useful;partial.world.complete=false;
    require(gp::analyze(session,partial,limits,prices).status=="partial_world","partial world accepted");
    auto unavailable=useful;unavailable.parent_text="unlearned n0 n6";
    require(!gp::analyze(session,unavailable,limits,prices).valid,"unknown parent meaning accepted");
    auto absent=useful;absent.parent_text="span n0 toward nonexistent";
    require(gp::analyze(session,absent,limits,prices).status=="no_compatible_grounding","absent potential grounding accepted");
    auto bad=useful;bad.actions.push_back({"unknown_motor",{"n0","n1"}});
    require(!gp::analyze(session,bad,limits,prices).valid,"incomplete potential domain accepted");
    auto small_scale=limits;small_scale.normalization_atoms=5;
    require(gp::analyze(session,useful,small_scale,prices).status=="normalization_bound","feature scale silently clipped");

    require(gp::current(session,useful,a),"fresh preview considered stale");
    auto changed=useful;changed.observation_revision++;
    require(!gp::current(session,changed,a),"observation revision retained preview");
    changed=useful;changed.parent_revision++;
    require(!gp::current(session,changed,a),"parent revision retained preview");
    changed=useful;changed.candidate_revision++;
    require(!gp::current(session,changed,a),"candidate revision retained preview");
    changed=useful;changed.affordance_revision++;
    require(!gp::current(session,changed,a),"affordance revision retained preview");
    changed=useful;changed.actions.pop_back();
    require(!gp::current(session,changed,a),"changed public action content retained preview");
    changed=useful;changed.world.facts.push_back({"edge",{"n0","n1"}});
    require(!gp::current(session,changed,a),"changed public world content retained preview");
    changed=useful;changed.names.push_back("new_name");
    require(!gp::current(session,changed,a),"changed public referents retained preview");

    gp::ProposalInputs proposal;proposal.parent_text=useful.parent_text;proposal.world=useful.world;
    proposal.actions=useful.actions;proposal.names=useful.names;proposal.order_seed=17;
    auto proposed=gp::propose(session,proposal,limits,prices);
    require(proposed.complete&&!proposed.truncated&&proposed.status=="exact","generic proposal enumeration failed");
    auto contains=[&](const std::string& text){return std::any_of(proposed.candidates.begin(),proposed.candidates.end(),
        [&](const auto& candidate){return candidate.text==text;});};
    require(contains(useful.candidate_text)&&contains(irrelevant.candidate_text),"proposal mechanism omitted useful or irrelevant learned frame");
    for(const auto& candidate:proposed.candidates) {
        require(!candidate.definition.empty()&&!candidate.witness.empty()&&!candidate.names.empty(),"proposal derivation missing");
        require(session.answer(candidate.text,useful.world)!=Answer::Unknown,"proposed phrase is not executable learned language");
    }
    // Deadline/weight are intentionally absent from the proposal input. Changing
    // their public envelope cannot change the candidate list or ordering.
    auto envelope=request;envelope.deadline=3;
    gp::ProposalInputs same=proposal;same.world=envelope.visible.scene;
    auto repeated=gp::propose(session,same,limits,prices);
    require(repeated.candidates.size()==proposed.candidates.size(),"deadline changed proposal count");
    for(size_t i=0;i<proposed.candidates.size();++i)
        require(repeated.candidates[i].text==proposed.candidates[i].text,"deadline changed public ordering");
    auto reordered=proposal;reordered.order_seed++;
    auto shuffled=gp::propose(session,reordered,limits,prices);bool difference=false;
    require(shuffled.candidates.size()==proposed.candidates.size(),"order seed changed candidate support");
    for(size_t i=0;i<proposed.candidates.size();++i)difference|=shuffled.candidates[i].text!=proposed.candidates[i].text;
    require(difference,"independent public seed did not counterbalance ordering");
    auto capped=limits;capped.candidates=2;auto cap=gp::propose(session,proposal,capped,prices);
    require(cap.complete&&cap.truncated&&cap.total_candidates==proposed.candidates.size()&&cap.candidates.size()==2&&
            cap.status=="candidate_limit","candidate capacity silently pruned inventory");
    auto low=limits;low.work=proposed.work-1;auto unfinished=gp::propose(session,proposal,low,prices);
    require(!unfinished.complete&&unfinished.candidates.empty()&&unfinished.status=="work_budget","proposal accepted partial search prefix");
    require(gp::current(session,proposal,proposed)&&!gp::current(session,reordered,proposed),"proposal provenance ignored order revision");
    auto macros=proposal;macros.minimum_atoms=2;
    auto macro_result=gp::propose(session,macros,limits,prices);
    require(macro_result.complete&&!macro_result.truncated&&macro_result.enumerated_candidates==15&&
            macro_result.total_candidates==5&&macro_result.work==proposed.work&&macro_result.cost==proposed.cost,
            "macro granularity changed paid enumeration or lost pre-filter provenance");
    std::vector<std::string> filtered;
    for(const auto& row:proposed.candidates)if(std::all_of(row.definition.begin(),row.definition.end(),
        [](const auto& branch){return branch.body.size()>=2;}))filtered.push_back(row.text);
    for(size_t i=0;i<filtered.size();++i)require(filtered[i]==macro_result.candidates[i].text,"macro filter changed seeded relative order");
    auto macro_cap=gp::propose(session,macros,capped,prices);
    require(macro_cap.complete&&macro_cap.truncated&&macro_cap.enumerated_candidates==15&&macro_cap.total_candidates==5&&
            macro_cap.candidates.size()==2&&macro_cap.work==proposed.work,"macro cap ran before complete paid enumeration");
    require(!gp::current(session,macros,proposed)&&gp::current(session,macros,macro_result),"macro granularity omitted from cache provenance");
    for(int minimum:{0,65}) {
        auto invalid=proposal;invalid.minimum_atoms=minimum;auto rejected=gp::propose(session,invalid,limits,prices);
        require(!rejected.complete&&rejected.work==0&&rejected.status=="invalid_minimum_atoms","invalid macro minimum accepted");
    }
    auto empty_macro=proposal;empty_macro.minimum_atoms=64;auto none=gp::propose(session,empty_macro,limits,prices);
    require(none.complete&&none.candidates.empty()&&none.enumerated_candidates==15&&none.total_candidates==0&&
            none.work==proposed.work,"empty macro inventory skipped paid source enumeration");

    auto contradiction=fixture.lessons.concepts.front();contradiction.id="preview_conflict";contradiction.answer=!contradiction.answer;
    require(session.teach_concept(contradiction),"preview lifecycle contradiction rejected");
    require(!gp::current(session,useful,a)&&gp::analyze(session,useful,limits,prices).status=="knowledge_dirty",
            "dirty knowledge retained paid preview");
    session.rebuild();
    require(!gp::current(session,useful,a),"model rebuild reused old prediction");
    auto ambiguous=useful;ambiguous.candidate_text="inspect n0 toward n1";
    require(gp::analyze(session,ambiguous,limits,prices).status=="unresolved_positive_meaning","unresolved candidate definition accepted");
    require(session.revise_concept("preview_conflict",1,contradiction.answer,false),"preview lifecycle withdrawal rejected");
    session.rebuild();require(gp::analyze(session,useful,limits,prices).valid,"preview failed after knowledge recovery");
    for(bool yes:{false,true}) {
        laminar::concepts::Receipt row;row.id="ambiguous_"+std::to_string(yes);row.name="uncertain";
        row.args={"a","b"};row.answer=yes;
        if(yes)row.scene.facts={{"violet",{"a","b"}},{"orange",{"a","b"}}};
        require(session.teach_concept(row),"ambiguous example rejected");
        LanguageReceipt language;language.id=row.id;language.text="unclear a b";language.names={"a","b"};
        language.scene=row.scene;language.answer=yes;require(session.teach_language(language),"ambiguous language rejected");
    }
    session.rebuild();require(session.concepts().models().at("uncertain").hypotheses.size()>1,
        "ambiguity fixture unexpectedly resolved");
    auto uncertain=useful;uncertain.candidate_text="unclear n0 n1";
    require(gp::analyze(session,uncertain,limits,prices).status=="unresolved_positive_meaning",
            "multiple genuinely distinct concept hypotheses treated as union branches");
    std::cout<<"Goal preview: useful delta "<<a.delta<<", irrelevant delta "<<b.delta<<", "<<checked
        <<" independently enumerated worlds, "<<proposed.total_candidates<<" generic candidates; exact bounds, public order, partial/ambiguous/stale guards passed\n";
    std::cout<<"Useful work="<<a.work<<" predictions="<<a.predictions<<" groundings="<<a.groundings
        <<" atom_checks="<<a.atom_checks<<" answer_calls="<<a.answer_calls<<" planner_calls="<<a.planner_calls
        <<" cost="<<a.cost<<"; proposal_work="<<proposed.work<<"\n";
}
