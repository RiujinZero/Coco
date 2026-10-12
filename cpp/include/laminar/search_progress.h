#pragma once
#include <chrono>
#include <cstdint>
#include <ostream>
#include <sstream>

namespace laminar::reasoning {
// Optional observer only: no clock or counter feeds back into learning.
struct SearchProgress {
    std::ostream* out=nullptr;
    template<class Ar> void state_io(Ar&){}   // a log sink, never learning state
    void emit(const char* phase,int cue=-1,uint64_t proposals=0,uint64_t states=0,uint64_t steps=0)const {
        if(!out)return;
        const auto now=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::ostringstream row;
        row<<"search_progress\t"<<std::fixed<<now<<'\t'<<phase<<'\t'<<cue<<'\t'<<proposals<<'\t'<<states<<'\t'<<steps<<'\n';
        *out<<row.str();
        out->flush();
    }
};
}
