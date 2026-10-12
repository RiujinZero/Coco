#pragma once
#include "continual_training_runner.h"

namespace laminar::training {
inline int continual_train_cli(int argc,char** argv) {
    std::map<std::string,std::string> args;
    const std::set<std::string> allowed{"trainer","profile","file","environment","worlds","seed","output","checkpoint","resume","stop_after","lifecycle","audit_models","search_progress"};
    for(int i=1;i<argc;++i) {
        std::string token=argv[i];auto equals=token.find('=');
        if(equals==std::string::npos||equals==0||equals+1==token.size())throw std::invalid_argument("continual arguments must be key=value");
        auto key=token.substr(0,equals);
        if(!allowed.count(key)||!args.emplace(key,token.substr(equals+1)).second)throw std::invalid_argument("unknown or duplicate continual argument: "+key);
    }
    for(const char* key:{"file","environment","worlds","output"})if(!args.count(key))throw std::invalid_argument(std::string("continual trainer requires ")+key);
    auto number=[&](const std::string& key,uint64_t fallback) {
        if(!args.count(key))return fallback;const auto& text=args.at(key);
        if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("invalid "+key);
        return uint64_t(std::stoull(text));
    };
    auto seed=number("seed",1);if(seed>std::numeric_limits<unsigned>::max())throw std::invalid_argument("seed exceeds unsigned range");
    ReplayOptions options;options.canonical=true;options.binary=argv[0];
    if(args.count("profile"))options.profile=args.at("profile");
#ifndef _WIN32
    options.binary=std::filesystem::read_symlink("/proc/self/exe").string();
#endif
    options.stop_after=number("stop_after",0);
    auto lifecycle=number("lifecycle",0);if(lifecycle>1)throw std::invalid_argument("lifecycle must be 0 or 1");options.lifecycle=bool(lifecycle);
    auto audit_models=number("audit_models",0);if(audit_models>1)throw std::invalid_argument("audit_models must be 0 or 1");options.audit_models=bool(audit_models);
    auto search_progress=number("search_progress",0);if(search_progress>1)throw std::invalid_argument("search_progress must be 0 or 1");options.search_progress=bool(search_progress);
    if(args.count("resume"))options.resume=args.at("resume");
    if(args.count("checkpoint"))options.checkpoint=args.at("checkpoint");
    if(options.stop_after&&options.checkpoint.empty())throw std::invalid_argument("stop_after requires checkpoint=PATH");
    auto parent=std::filesystem::path(args.at("output")).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
    auto config=ContinualTrainingConfig::named(options.profile);
    std::vector<std::string> strings{argv[0],args.at("file"),args.at("environment"),args.at("worlds"),std::to_string(seed),"learned_progress",args.at("output"),"composed",std::to_string(config.discovery_budget)};
    std::vector<char*> wire;for(auto& value:strings)wire.push_back(value.data());
    return run_control_training(int(wire.size()),wire.data(),options);
}
} // namespace laminar::training
