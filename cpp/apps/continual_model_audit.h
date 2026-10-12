#pragma once
// Optional read-only evidence for independently replaying imperfect learned
// computation. No oracle or auditor data enters the learner.
struct ModelAudit {
    std::string prefix;
    std::ofstream index;
    std::map<std::vector<float>,size_t> policies;
    size_t serial=0;
    explicit ModelAudit(const std::string& path):prefix(path),index(path+".models.tsv") {
        require(bool(index),"cannot save model audit index");
        index<<"episode\tdecision\tversion\tmodel\tweights\n";
    }
    void record(const RelationalSession& session,const std::string& episode,size_t decision) {
        require(session.ready(),"cannot audit an unreconstructed model");
        const auto& w=session.primitives().memory.skills.kernel.routes.W;
        auto [p,inserted]=policies.emplace(w,policies.size());
        if(inserted)weights(prefix+".model_weights"+std::to_string(p->second),w);
        std::ofstream file(prefix+".model"+std::to_string(serial)+".txt");
        require(bool(file),"cannot save reconstructed model");
        dump(session,std::to_string(serial),file);
        for(const auto& [name,id]:session.binary_ids())file<<"SCHEMA "<<name<<'\n';
        index<<episode<<'\t'<<decision<<'\t'<<session.version()<<'\t'<<serial++<<'\t'<<p->second<<'\n'<<std::flush;
    }
};
