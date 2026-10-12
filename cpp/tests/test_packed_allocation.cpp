#include "laminar/organism.h"
#include <iostream>
#include <set>
#include <stdexcept>
using namespace laminar;
void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
int main() {
    for(uint32_t units:{1u,127u,128u,129u,767u,768u,769u,1025u,1665u}) {
        Params p;p.units=units;p.vocab=32;p.train=false;p.structural=false;p.kpack=true;p.seqmem=false;p.kscreen=true;p.kscreen_eps=0;
        Organism org(p);std::set<uint32_t> seen;
        // Without observations all screen entries tie, so class zero is tried
        // first, followed by the first class with space. Uneven class wrapping
        // must not allocate a later class before that fallback occurs.
        std::vector<uint32_t> expected(units);for(uint32_t u=0;u<units;++u)expected[u]=u;
        auto key=[&](uint32_t u){return std::make_pair((u/p.motif_size)%p.k_set.size(),u);};
        std::sort(expected.begin(),expected.end(),[&](uint32_t a,uint32_t b){return key(a)<key(b);});
        for(auto wanted:expected) {
            auto u=org.alloc_unit();check(u==wanted,"allocation escaped class order or exhausted early");
            check(seen.insert(u).second,"duplicate unit allocation");
        }
        check(seen.size()==units&&org.live_units()==units,"lost available unit");
        check(org.alloc_unit()==UINT32_MAX,"full population allocated a unit");
    }
    std::cout<<"packed allocation respects absent/uneven classes, fallback order and exhaustion\n";
}
