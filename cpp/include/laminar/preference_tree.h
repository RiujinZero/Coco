#pragma once
// Generic CART readout of experienced operation preferences. Inputs are public
// observations; targets are argmaxes of the learner's own empirical values.
// Feature and action indices carry no semantic meaning in this implementation.
#include "reasoning_runtime.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace laminar::reasoning {
class PreferenceTree {
public:
    struct Node {
        int feature=-1,left=-1,right=-1;
        double threshold=0;
        std::array<size_t,5> counts{};
    };
    static constexpr size_t max_depth=12,min_leaf=2;
private:
    struct Sample {Observation observation;int label;};
    std::vector<Node> nodes_;
public:
    template<class Ar> void state_io(Ar& ar){ar(nodes_);}
private:
    static double purity(const std::array<size_t,5>& counts,size_t n) {
        double result=0;for(auto count:counts)result+=double(count)*double(count);return result/double(n);
    }
    int build(const std::vector<Sample>& samples,std::vector<size_t> ids,size_t depth) {
        Node node;for(auto i:ids)++node.counts[samples[i].label];
        int id=int(nodes_.size());nodes_.push_back(node);
        if(depth==max_depth||ids.size()<2*min_leaf||*std::max_element(node.counts.begin(),node.counts.end())==ids.size())return id;
        double best=purity(node.counts,ids.size());int feature=-1;double threshold=0;
        for(int f=0;f<32;++f) {
            auto order=ids;
            std::sort(order.begin(),order.end(),[&](size_t a,size_t b){
                float x=samples[a].observation[f],y=samples[b].observation[f];return x<y||(x==y&&a<b);
            });
            std::array<size_t,5> left{},right=node.counts;
            for(size_t k=0;k+1<order.size();++k) {
                int label=samples[order[k]].label;++left[label];--right[label];
                size_t nl=k+1,nr=order.size()-nl;
                double lo=samples[order[k]].observation[f],hi=samples[order[k+1]].observation[f];
                if(nl<min_leaf||nr<min_leaf||lo==hi)continue;
                double score=purity(left,nl)+purity(right,nr);
                if(score>best+1e-12){best=score;feature=f;threshold=(lo+hi)/2;}
            }
        }
        if(feature<0)return id;
        std::vector<size_t> left,right;
        for(auto i:ids)(samples[i].observation[feature]<=threshold?left:right).push_back(i);
        int l=build(samples,std::move(left),depth+1),r=build(samples,std::move(right),depth+1);
        nodes_[id].feature=feature;nodes_[id].threshold=threshold;nodes_[id].left=l;nodes_[id].right=r;return id;
    }
public:
    const auto& nodes()const{return nodes_;}
    void fit(const std::map<Observation,std::array<float,5>>& values) {
        nodes_.clear();std::vector<Sample> samples;
        for(const auto& [obs,q]:values) {
            for(float v:obs)if(!std::isfinite(v))throw std::invalid_argument("nonfinite preference input");
            for(float v:q)if(!std::isfinite(v))throw std::invalid_argument("nonfinite preference value");
            int best=0;for(int a=1;a<5;++a)if(q[a]>q[best])best=a;
            samples.push_back({obs,best});
        }
        if(samples.empty())throw std::invalid_argument("no experienced preferences");
        std::vector<size_t> ids(samples.size());std::iota(ids.begin(),ids.end(),0);build(samples,std::move(ids),0);
    }
    int choose(const Observation& observation,const std::vector<int>& available)const {
        if(nodes_.empty()||available.empty())throw std::logic_error("unavailable preference readout");
        size_t node=0;
        while(nodes_[node].feature>=0) {
            const auto& n=nodes_[node];node=observation[n.feature]<=n.threshold?n.left:n.right;
        }
        int best=available.front();
        for(int a:available)if(nodes_[node].counts.at(a)>nodes_[node].counts.at(best))best=a;
        return best;
    }
};
}
