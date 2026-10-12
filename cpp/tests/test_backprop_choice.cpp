#include "laminar/backprop_choice.h"
#include <iostream>
using namespace laminar::reasoning;
void check(bool b,const char* why){if(!b)throw std::runtime_error(why);}
int main() {
    BackpropChoice p(5,7);Observation o{};for(int i=0;i<32;++i)o[i]=float((i%7)-3)/5;
    // Nonzero output weights ensure the finite-difference check exercises
    // both layers, including the hidden nonlinear derivative and biases.
    for(size_t i=p.output_weights();i<p.weights.size();++i)p.weights[i]=float(int(i%13)-6)/19;
    auto g=p.gradient(o,2,0.7f);float worst=0;
    auto loss=[&](){double d=p.action_value(o,2)-0.7;return 0.5*d*d;};
    for(size_t i=0;i<g.size();++i) {
        float old=p.weights[i];constexpr float eps=0.001f;
        p.weights[i]=old+eps;double plus=loss();p.weights[i]=old-eps;double minus=loss();p.weights[i]=old;
        float numeric=(plus-minus)/(2*eps);worst=std::max(worst,std::abs(numeric-g[i]));
        check(std::abs(numeric-g[i])<0.0005f,"backprop gradient mismatch");
    }
    double before=loss();p.value_update(o,2,0.7f);check(loss()<before,"gradient update did not reduce loss");
    auto w=p.weights;p.choose(o,{0,2,4});p.best_value(o,{0,2,4});check(w==p.weights,"inference changed weights");
    std::cout<<"two-layer gradient, loss descent and frozen inference passed; max_error="<<worst<<'\n';
}
