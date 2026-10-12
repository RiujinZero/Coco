#pragma once
#include "laminar/continual_training.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace laminar::training {
struct ReplayOptions {
    bool canonical=false,lifecycle=false,audit_models=false,search_progress=false;
    std::string checkpoint,resume,binary;
    std::string profile=ContinualTrainingConfig::name;
    uint64_t stop_after=0;
};
// Corruption/drift detection, not a cryptographic authentication scheme.
inline uint64_t fingerprint(const std::string& text,uint64_t h=14695981039346656037ull) {
    for(unsigned char c:text){h^=c;h*=1099511628211ull;}return h;
}
inline uint64_t file_fingerprint(const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot fingerprint "+path);
    uint64_t h=14695981039346656037ull;char bytes[65536];
    while(in){in.read(bytes,sizeof(bytes));h=fingerprint(std::string(bytes,size_t(in.gcount())),h);}
    if(!in.eof())throw std::runtime_error("failed reading "+path);return h;
}
class TraceDigest:public std::streambuf {
    std::streambuf* output_;
public:
    uint64_t hash=14695981039346656037ull;
    explicit TraceDigest(std::streambuf* output):output_(output){}
protected:
    int_type overflow(int_type c)override {
        if(traits_type::eq_int_type(c,traits_type::eof()))return traits_type::not_eof(c);
        char value=traits_type::to_char_type(c);hash=fingerprint(std::string(1,value),hash);
        return output_->sputc(value);
    }
    std::streamsize xsputn(const char* p,std::streamsize n)override {
        hash=fingerprint(std::string(p,size_t(n)),hash);return output_->sputn(p,n);
    }
    int sync()override{return output_->pubsync();}
};
// Offline adapter checkpoint: replay accepted public records and simulated
// environment outcomes to reconstruct *all* journals, neural weights and RNGs.
// No real actuator is called by this adapter. Search is repeated during replay.
class ReplayCheckpoint {
    ReplayOptions options_;
    std::string identity_;
    uint64_t line_=0,resume_line_=0,resume_train_=0,resume_trace_=0,resume_state_=0;
    bool verified_=true,attached_=false;
    std::streambuf* original_;
    TraceDigest trace_;
    static uint64_t state(const relational::ControlledRelationalSession& learner,size_t train) {
        std::ostringstream out;out<<std::setprecision(std::numeric_limits<float>::max_digits10);
        const auto& s=learner.knowledge;
        out<<train<<' '<<s.version()<<' '<<s.ready()<<' '<<s.dirty()<<' '<<learner.updates()<<' '<<learner.credits().size();
        for(float v:learner.policy.weights)out<<' '<<v;
        for(float v:s.primitives().memory.skills.kernel.routes.W)out<<' '<<v;
        for(const auto& [obs,q]:learner.empirical_values()){for(float v:obs)out<<' '<<v;for(float v:q)out<<' '<<v;}
        for(const auto& r:learner.credits())out<<' '<<r.id<<' '<<r.version<<' '<<r.evidence_version<<' '<<r.reward<<' '<<r.satisfaction;
        return fingerprint(out.str());
    }
public:
    ReplayCheckpoint(const ReplayOptions& options,unsigned seed,const std::string& config,
                     const std::vector<std::string>& inputs)
        :options_(options),original_(std::cout.rdbuf()),trace_(original_) {
        if(!options_.canonical)return;
        std::ostringstream identity;identity<<ContinualTrainingConfig::name<<' '<<seed<<' '<<fingerprint(config)<<' '<<file_fingerprint(options_.binary);
        for(const auto& input:inputs) {
            identity<<' '<<file_fingerprint(input);
            if(!options_.checkpoint.empty()&&std::filesystem::absolute(options_.checkpoint).lexically_normal()==std::filesystem::absolute(input).lexically_normal())
                throw std::runtime_error("checkpoint would overwrite curriculum input");
        }
        identity_=identity.str();
        if(!options_.resume.empty()) {
            std::ifstream in(options_.resume);std::string magic,identity,extra;unsigned format=0;uint64_t check=0;
            in>>magic>>format>>std::quoted(identity)>>resume_line_>>resume_train_>>resume_trace_>>resume_state_>>check;
            if(!in||magic!="COCO_REPLAY"||format!=1||!resume_line_||(in>>extra)||identity!=identity_
                ||check!=fingerprint(identity+' '+std::to_string(resume_line_)+' '+std::to_string(resume_train_)+' '+std::to_string(resume_trace_)+' '+std::to_string(resume_state_)))
                throw std::runtime_error("checkpoint is incomplete, corrupt, or belongs to different binary/config/data");
            if(options_.stop_after&&options_.stop_after<=resume_line_)throw std::runtime_error("stop_after must be beyond resume cursor");
            verified_=false;
            std::cerr<<"Replaying "<<resume_line_<<" curriculum lines to restore checkpoint; acquisition search may repeat.\n";
        }
        std::cout.rdbuf(&trace_);attached_=true;
    }
    ~ReplayCheckpoint(){if(attached_){std::cout.flush();std::cout.rdbuf(original_);}}
    void next_line(){++line_;}
    bool should_stop()const{return options_.canonical&&options_.stop_after&&line_>=options_.stop_after;}
    bool stopped()const{return should_stop();}
    void verify(const relational::ControlledRelationalSession& learner,size_t train,bool safe) {
        if(verified_)return;
        if(line_<resume_line_)return;
        if(line_!=resume_line_||!safe||train!=resume_train_||trace_.hash!=resume_trace_||state(learner,train)!=resume_state_)
            throw std::runtime_error("checkpoint replay diverged; continuation refused");
        verified_=true;std::cerr<<"Checkpoint replay verified at curriculum line "<<line_<<".\n";
    }
    void save(const relational::ControlledRelationalSession& learner,size_t train) {
        if(!options_.canonical||options_.checkpoint.empty()||!verified_)return;
        if(!line_||learner.active())throw std::logic_error("unsafe checkpoint boundary");
        auto stamp=state(learner,train);
        auto payload=identity_+' '+std::to_string(line_)+' '+std::to_string(train)+' '+std::to_string(trace_.hash)+' '+std::to_string(stamp);
        auto temporary=options_.checkpoint+".tmp";
        std::ofstream out(temporary,std::ios::trunc);
        out<<"COCO_REPLAY 1\n"<<std::quoted(identity_)<<'\n'<<line_<<' '<<train<<' '<<trace_.hash<<' '<<stamp<<' '<<fingerprint(payload)<<'\n';
        out.close();if(!out)throw std::runtime_error("cannot write checkpoint");
        std::filesystem::rename(temporary,options_.checkpoint);
    }
    void finish(const relational::ControlledRelationalSession& learner,size_t train) {
        verify(learner,train,!learner.active());
        if(!verified_)throw std::runtime_error("curriculum ended before checkpoint cursor");
        save(learner,train);
    }
};
} // namespace laminar::training
