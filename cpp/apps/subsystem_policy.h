#pragma once
// Loading an exported subsystem controller policy (goal harness `weights` JSONL lines) so a trained
// subsystem can act as a demonstration source. Only the public 32 floats and the legal menu enter it,
// exactly like the hand-written continuing_curriculum_v1 teacher.
#include "laminar/backprop_choice.h"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace laminar::subsystem_policy {

struct Loaded { int hidden = 0, actions = 0; std::vector<float> weights; uint64_t digest = 0; };

inline Loaded load(const std::string& path, const std::string& label) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open policy weights file: " + path);
    const std::string key = "\"label\":\"" + label + "\",";
    std::string line;
    while (std::getline(in, line)) {
        if (line.find(key) == std::string::npos) continue;
        Loaded out;
        auto field = [&](const char* name) {
            auto at = line.find(name); if (at == std::string::npos) throw std::runtime_error(std::string("policy line without ") + name);
            return std::stoi(line.substr(at + std::char_traits<char>::length(name)));
        };
        out.hidden = field("\"hidden\":"); out.actions = field("\"actions\":");
        auto w = line.find("\"weights\":["); if (w == std::string::npos) throw std::runtime_error("policy line without weights");
        const char* p = line.c_str() + w + 11;
        while (*p && *p != ']') {
            char* end = nullptr; out.weights.push_back(std::strtof(p, &end));
            if (end == p) throw std::runtime_error("bad policy weight value");
            p = end; if (*p == ',') ++p;
        }
        uint64_t h = 1469598103934665603ULL;   // FNV-1a over the exact exported text of this line
        for (unsigned char c : line) { h ^= c; h *= 1099511628211ULL; }
        out.digest = h;
        reasoning::BackpropChoice shape(out.actions, 0, out.hidden);
        if (shape.weights.size() != out.weights.size()) throw std::runtime_error("policy weights do not match hidden/actions shape");
        return out;
    }
    throw std::runtime_error("policy label not found: " + label);
}

// A frozen demonstration function over the public observation and menu.
struct Teacher {
    reasoning::BackpropChoice policy;
    explicit Teacher(const Loaded& l) : policy(l.actions, 0, l.hidden) { policy.weights = l.weights; }
    int operator()(const reasoning::Observation& o, const std::vector<int>& menu) const { return policy.choose(o, menu); }
};

} // namespace laminar::subsystem_policy
