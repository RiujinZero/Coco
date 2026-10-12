#include "laminar/preference_tree.h"
#include <iostream>
using namespace laminar::reasoning;
void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
int main() {
    PreferenceTree tree;std::map<Observation,std::array<float,5>> data;
    bool rejected=false;try{tree.fit(data);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"empty value table accepted");
    for(int i=0;i<24;++i) {
        Observation o{};o[27]=float(i);std::array<float,5> values{};
        values[i<8?4:i<16?2:1]=1;data[o]=values;
    }
    tree.fit(data);
    for(int i=-20;i<50;++i) {
        Observation o{};o[27]=float(i);o[9]=999; // an unused input must not acquire meaning
        check(tree.choose(o,{0,1,2,3,4})==(i<8?4:i<16?2:1),"learned threshold did not transfer");
    }
    Observation o{};o[27]=20;
    check(tree.choose(o,{0,2,3,4})==0,"disabled action selected or tie order changed");
    // Refit replaces the readout, including when earlier reward preferences change.
    for(auto& [state,q]:data){q={};q[3]=1;}
    tree.fit(data);check(tree.nodes().size()==1&&tree.choose(o,{0,1,2,3,4})==3,"old preference survived refit");
    std::cout<<"preference fitting, extrapolation, action removal and complete refit passed\n";
}
