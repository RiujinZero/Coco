#pragma once
// Explicit backprop comparator for the experimental computation controller.
// A two-layer Q approximator, trained by semi-gradient TD: the bootstrapped
// target is constant for each update, with backprop through this one decision.
#include "reasoning_runtime.h"
#include <random>
namespace laminar::reasoning {
struct BackpropChoice {
    static constexpr int input=32;
    int actions,hidden;
    std::vector<float> weights;
    template<class Ar> void state_io(Ar& ar){ar(actions);ar(hidden);ar(weights);}
    static int checked_hidden(int n){if(n<1||n>8192)throw std::invalid_argument("invalid controller width");return n;}
    explicit BackpropChoice(int n,uint64_t seed,int width=32):actions(n),hidden(checked_hidden(width)),weights(hidden*input+hidden+n*hidden+n) {
        std::mt19937 rng(seed);std::uniform_real_distribution<float> init(-0.306186f,0.306186f);
        for(int i=0;i<hidden*input;++i)weights[i]=init(rng);
    }
    int h_bias()const{return hidden*input;}
    int output_weights()const{return hidden*input+hidden;}
    int output_bias()const{return output_weights()+actions*hidden;}
    std::vector<float> activate(const Observation& o)const {
        std::vector<float> h(hidden);
        for(int j=0;j<hidden;++j){float z=weights[h_bias()+j];for(int i=0;i<input;++i)z+=weights[j*input+i]*o[i];h[j]=std::tanh(z);}
        return h;
    }
    float predict(const std::vector<float>& h,int action)const {
        float v=weights[output_bias()+action];for(int j=0;j<hidden;++j)v+=weights[output_weights()+action*hidden+j]*h[j];return v;
    }
    float action_value(const Observation& o,int action)const{return predict(activate(o),action);}
    float best_value(const Observation& o,const std::vector<int>& available)const {
        auto h=activate(o);float best=-1e9f;for(int a:available)best=std::max(best,predict(h,a));return best;
    }
    int choose(const Observation& o,const std::vector<int>& available)const {
        auto h=activate(o);int best=available.front();for(int a:available)if(predict(h,a)>predict(h,best))best=a;return best;
    }
    // Derivative of 0.5*(Q(o,a)-target)^2, for numerical verification as well
    // as learning. Other outputs' weights receive zero direct gradient.
    std::vector<float> gradient(const Observation& o,int action,float target)const {
        auto h=activate(o);float error=predict(h,action)-target;std::vector<float> g(weights.size());
        g[output_bias()+action]=error;
        for(int j=0;j<hidden;++j) {
            g[output_weights()+action*hidden+j]=error*h[j];
            float dz=error*weights[output_weights()+action*hidden+j]*(1-h[j]*h[j]);
            g[h_bias()+j]=dz;for(int i=0;i<input;++i)g[j*input+i]=dz*o[i];
        }
        return g;
    }
    void value_update(const Observation& o,int action,float target) {
        auto g=gradient(o,action,target);for(size_t i=0;i<weights.size();++i)weights[i]-=0.005f*g[i];
    }
};
}
