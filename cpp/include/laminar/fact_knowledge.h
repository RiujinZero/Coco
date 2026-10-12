#pragma once
// Observation coverage is distinct from relational content. Missing facts are
// unknown unless an explicit negative or a complete observed scope excludes them.
#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

namespace laminar::reasoning {
struct FactKnowledge {
    using Atom=std::tuple<int,int,int>;
    using Facts=std::set<Atom>;
    enum Truth { UNKNOWN=-1, FALSE_VALUE=0, TRUE_VALUE=1 };
    Facts positive,negative;
    std::set<std::pair<int,int>> closed_rows; // subject, relation
    std::set<std::pair<int,int>> closed_columns; // relation, object
    bool operator<(const FactKnowledge& k) const {
        return std::tie(positive,negative,closed_rows,closed_columns)<std::tie(k.positive,k.negative,k.closed_rows,k.closed_columns);
    }
    Truth truth(int a,int r,int b) const {
        if(positive.count({a,r,b}))return TRUE_VALUE;
        if(negative.count({a,r,b})||closed_rows.count({a,r})||closed_columns.count({r,b}))return FALSE_VALUE;
        return UNKNOWN;
    }
    bool observe(int a,int r,int b,bool present) {
        auto known=truth(a,r,b);
        if(known!=UNKNOWN && (known==TRUE_VALUE)!=present)return false;
        (present?positive:negative).emplace(a,r,b);return true;
    }
    bool observe_row(int a,int r,const std::set<int>& objects) {
        for(const auto& [s,relation,o]:positive) if(s==a&&relation==r&&!objects.count(o))return false;
        for(int b:objects)if(truth(a,r,b)==FALSE_VALUE)return false;
        for(int b:objects)positive.emplace(a,r,b);
        closed_rows.emplace(a,r);return true;
    }
    bool observe_column(int r,int b,const std::set<int>& subjects) {
        for(const auto& [s,relation,o]:positive)if(relation==r&&o==b&&!subjects.count(s))return false;
        for(int a:subjects)if(truth(a,r,b)==FALSE_VALUE)return false;
        for(int a:subjects)positive.emplace(a,r,b);
        closed_columns.emplace(r,b);return true;
    }
    // A declared unobserved interval invalidates evidence in the affected scope.
    // It supplies no new positive or negative fact.
    void forget_row(int a,int r) {
        for(auto* facts:{&positive,&negative})for(auto it=facts->begin();it!=facts->end();)
            if(std::get<0>(*it)==a&&std::get<1>(*it)==r)it=facts->erase(it);else ++it;
        closed_rows.erase({a,r});
        for(auto it=closed_columns.begin();it!=closed_columns.end();)if(it->first==r)it=closed_columns.erase(it);else ++it;
    }
    void forget_column(int r,int b) {
        for(auto* facts:{&positive,&negative})for(auto it=facts->begin();it!=facts->end();)
            if(std::get<1>(*it)==r&&std::get<2>(*it)==b)it=facts->erase(it);else ++it;
        closed_columns.erase({r,b});
        for(auto it=closed_rows.begin();it!=closed_rows.end();)if(it->second==r)it=closed_rows.erase(it);else ++it;
    }
    void update(int a,int r,int b,int effect) {
        update_evidence(positive,negative,closed_rows,closed_columns,a,r,b,effect);
    }
    // The same evidence algebra is used for observed effects and hypothetical
    // action outcomes; prediction never writes into the observed state.
    template<class T> static void update_evidence(std::set<std::tuple<T,int,T>>& positive,
        std::set<std::tuple<T,int,T>>& negative,std::set<std::pair<T,int>>& closed_rows,
        std::set<std::pair<int,T>>& closed_columns,const T& a,int r,const T& b,int effect) {
        for(auto it=positive.begin();it!=positive.end();) {
            const auto [s,rel,o]=*it;
            bool remove=rel==r&&((effect==1&&s==a&&o==b)||(effect==2&&s==a)||(effect==3&&o==b));
            if(remove)it=positive.erase(it);else ++it;
        }
        if(effect==1)negative.emplace(a,r,b);
        else {
            for(auto it=negative.begin();it!=negative.end();) {
                const auto [s,rel,o]=*it;
                if(rel==r&&((s==a&&o==b)||(effect==2&&s==a)||(effect==3&&o==b)))it=negative.erase(it);else ++it;
            }
            positive.emplace(a,r,b);
        }
        if(effect==2)closed_rows.emplace(a,r);
        if(effect==3)closed_columns.emplace(r,b);
    }
};
} // namespace laminar::reasoning
