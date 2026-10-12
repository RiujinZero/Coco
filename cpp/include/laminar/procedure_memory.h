#pragma once
// Induce bounded reusable procedures from several verified training outcomes.
// A flat question cue is a lookup key for LEARNED procedures, never a coded recipe.
#include "reasoning_runtime.h"
#include "fact_knowledge.h"
#include "search_progress.h"
#include <deque>
#include <map>
#include <memory>

namespace laminar::reasoning {
struct FlatQuery {
    int cue = -1;
    Value a, b;
    int relation[2] = {-1, -1};
    std::vector<Fact> facts;
    std::shared_ptr<const FactKnowledge> knowledge;
    template<class Ar> void state_io(Ar& ar){ar(cue);ar(a);ar(b);ar.array(relation,2);ar(facts);ar(knowledge);}   // knowledge: saved by value (immutable snapshot)
};
struct VerifiedQuestion { FlatQuery query; Value answer; };
using Registers = std::array<Value, 2>;
using Program = std::vector<int>;
struct SearchResult {
    Program program; size_t states = 0; bool found = false;
    Program best_program; size_t correct=0,total=0;
};

struct ProcedureMemory {
    LocalChoice skills;
    std::map<int, Program> procedures;
    bool lookup_enabled=false;
    bool inverse_enabled=false;
    bool missing_evidence=false;
    bool unique_enabled=false,nonunique_result=false;
    std::array<bool,3> available_skills{true,true,true};
    uint64_t skill_steps=0;
    template<class Ar> void state_io(Ar& ar){ar(skills);ar(procedures);ar(lookup_enabled);ar(inverse_enabled);ar(missing_evidence);ar(unique_enabled);ar(nonunique_result);ar(available_skills);ar(skill_steps);}
    explicit ProcedureMemory(uint64_t seed,SkillShape shape={}) : skills(10, seed,shape) {}
    ProcedureMemory() : ProcedureMemory(0) {}   // checkpoint restore target: every field is then loaded

    template<class Choice> bool step_with(const FlatQuery& q, Registers& regs, int a,Choice& choices) {
        // A one-hop unique read is distinct from learned transitive following.
        // Its absence/ambiguity is a semantic outcome, not a policy timeout.
        if(a>=16&&a<=19) {
            if(regs[0].type!=0)return false;
            bool inverse=a>=18;int entity=regs[0].id,r=q.relation[a%2];
            if(q.knowledge&&!(inverse?q.knowledge->closed_columns.count({r,entity}):q.knowledge->closed_rows.count({entity,r}))) {
                missing_evidence=true;return false;
            }
            std::set<int> values;
            for(const auto& f:q.facts)if(f.relation==r&&(inverse?f.object:f.subject)==entity)values.insert(inverse?f.subject:f.object);
            if(values.size()!=1){nonunique_result=true;return false;}
            regs[0]={0,*values.begin()};return true;
        }
        if(a==10||a==11) {
            if(regs[0].type!=0||regs[1].type!=0)return false;
            int r=q.relation[a%2],s=regs[0].id,o=regs[1].id;
            bool present=false;
            if(q.knowledge) {
                auto truth=q.knowledge->truth(s,r,o);
                if(truth==FactKnowledge::UNKNOWN){missing_evidence=true;return false;}
                present=truth==FactKnowledge::TRUE_VALUE;
            } else for(const auto& f:q.facts)present|=f.subject==s&&f.relation==r&&f.object==o;
            regs[0]={1,present};return true;
        }
        if (a == 0) regs[0] = q.a;
        else if (a == 1) regs[0] = q.b;
        else if (a == 2) regs[1] = regs[0];
        else if (a == 3) std::swap(regs[0], regs[1]);
        else {
            const bool inverse=a>=12&&a<=15;
            if(inverse)a-=8; // inverse follow/count reuse the learned micro-policies
            if (a < 4 || a > 8 || (a != 8 && regs[0].type != 0)) return false;
            MicroState vm;
            vm.task = a == 8 ? 1 : a >= 6 ? 2 : 0;
            if(!available_skills[vm.task])return false;
            vm.subject = regs[0]; vm.target = a == 8 ? regs[1] : Value{};
            vm.relation = q.relation[a % 2];
            auto closed=[&](int entity){return inverse?q.knowledge->closed_columns.count({vm.relation,entity}):q.knowledge->closed_rows.count({entity,vm.relation});};
            if(q.knowledge&&a>=6&&a<=7&&!closed(vm.subject.id)) {
                missing_evidence=true;return false;
            }
            std::vector<Fact> reversed;
            if(inverse)for(const auto& f:q.facts)reversed.push_back({f.object,f.relation,f.subject});
            for (int t = 0; t < 256 && !vm.done; ++t) {
                auto available=vm.available();int instruction = choices.choose(vm.observe(), available);
                if(std::find(available.begin(),available.end(),instruction)==available.end())return false;
                if(q.knowledge&&a>=4&&a<=5&&instruction==1&&
                   !closed(vm.focus.id)) {
                    missing_evidence=true;return false;
                }
                vm.step(instruction, inverse?reversed:q.facts);
                ++skill_steps;
            }
            const int expected_type = a == 8 ? 1 : a >= 6 ? 2 : 0;
            if (!vm.done || vm.output.type != expected_type) return false;
            regs[0] = vm.output;
        }
        return true;
    }
    bool step(const FlatQuery& q,Registers& regs,int a){return step_with(q,regs,a,skills);}

    // No answer, goal graph, subgoal labels or training examples are accepted by
    // execute(). New values must be computed from this query's facts and bindings.
    template<class Choice> Value execute_with(const FlatQuery& q, const Program& p,Choice& choices) {
        missing_evidence=nonunique_result=false;
        Registers regs{q.a, q.b};
        for (int a : p) {
            if (a == 9) return regs[0];
            if (!step_with(q, regs, a,choices)) return {};
        }
        return {};
    }
    Value execute(const FlatQuery& q,const Program& p){return execute_with(q,p,skills);}
    Value answer(const FlatQuery& q) {
        missing_evidence=nonunique_result=false;
        auto it = procedures.find(q.cue);
        return it == procedures.end() ? Value{} : execute(q, it->second);
    }

    template<class Choice> SearchResult learn_with(const std::vector<VerifiedQuestion>& examples,Choice& choices,
                       bool no_bindings = false, int max_instructions = 8, size_t max_states = 50000,const SearchProgress* progress=nullptr) {
        SearchResult result;
        if (examples.empty()) return result;
        result.total=examples.size();
        const int cue = examples.front().query.cue;
        for (const auto& e : examples) if (e.query.cue != cue)
            throw std::invalid_argument("procedure examples must share one goal cue");
        // New verified counterevidence must never leave an obsolete procedure
        // active merely because replacement search exhausted its budget.
        procedures.erase(cue);
        struct Candidate { std::vector<Registers> regs; Program program; };
        Candidate initial;
        for (const auto& e : examples) initial.regs.push_back({e.query.a, e.query.b});
        auto key = [](const std::vector<Registers>& regs) {
            std::vector<int> k; k.reserve(regs.size() * 4);
            for (const auto& r : regs) for (const auto& v : r) { k.push_back(v.type); k.push_back(v.id); }
            return k;
        };
        std::set<std::vector<int>> visited{key(initial.regs)};
        std::deque<Candidate> queue{initial};
        std::map<std::array<int, 6>, std::pair<bool, Registers>> transition_cache;
        while (!queue.empty()) {
            Candidate current = std::move(queue.front()); queue.pop_front();
            size_t correct = 0;
            for (size_t i = 0; i < examples.size(); ++i) correct += current.regs[i][0] == examples[i].answer;
            if(result.best_program.empty()||correct>result.correct) {
                result.correct=correct;result.best_program=current.program;result.best_program.push_back(9);
            }
            if (correct==examples.size()) {
                current.program.push_back(9); result.program = std::move(current.program);
                result.found = true; result.states = visited.size();
                procedures[examples.front().query.cue] = result.program;
                return result;
            }
            if ((int)current.program.size() >= max_instructions - 1) continue;
            for (int a = 0; a <= (unique_enabled?19:inverse_enabled?15:lookup_enabled?11:8); ++a) {
                if(a==9)continue;
                if((a==10||a==11)&&!lookup_enabled)continue;
                if(a>=12&&a<=15&&!inverse_enabled)continue;
                if (no_bindings && (a == 2 || a == 3)) continue;
                Candidate next = current;
                bool valid = true;
                for (size_t i = 0; i < examples.size(); ++i) {
                    const auto& r = current.regs[i];
                    std::array<int, 6> ck{(int)i, r[0].type, r[0].id, r[1].type, r[1].id, a};
                    auto it = transition_cache.find(ck);
                    if (it == transition_cache.end()) {
                        Registers tmp = r;
                        bool ok = step_with(examples[i].query, tmp, a,choices);
                        it = transition_cache.emplace(ck, std::make_pair(ok, tmp)).first;
                    }
                    if (!it->second.first) { valid = false; break; }
                    next.regs[i] = it->second.second;
                }
                if (!valid || !visited.insert(key(next.regs)).second) continue;
                if(progress&&visited.size()%1000==0)progress->emit("compile_budget",cue,0,visited.size(),skill_steps);
                if (visited.size() > max_states) { result.states = visited.size(); return result; }
                next.program.push_back(a); queue.push_back(std::move(next));
            }
        }
        result.states = visited.size(); return result;
    }
    SearchResult learn(const std::vector<VerifiedQuestion>& examples,bool no_bindings=false,int max_instructions=8,size_t max_states=50000,const SearchProgress* progress=nullptr) {
        return learn_with(examples,skills,no_bindings,max_instructions,max_states,progress);
    }
};
} // namespace laminar::reasoning
