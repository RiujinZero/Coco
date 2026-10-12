#pragma once
// Ground utterances in independently observed before/after world states.
// Entity names and relation channels are supplied by perception; no sentence
// receives annotated argument roles, a relation label, or an update-operation label.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace laminar::reasoning {
struct EventGrounding {
    using NamedFact = std::tuple<std::string, int, std::string>;
    using World = std::set<NamedFact>;
    enum Effect { ADD = 0, REMOVE = 1, SET_SUBJECT = 2, SET_OBJECT = 3 };
    struct Rule {
        int relation = -1, subject = -1, object = -1, effect = -1;
        auto key() const { return std::make_tuple(relation, subject, object, effect); }
        bool operator<(const Rule& r) const { return key() < r.key(); }
        bool operator==(const Rule& r) const { return key() == r.key(); }
    };
    struct Frame {
        std::vector<std::string> pattern;
        std::set<Rule> candidates;
        std::set<uint64_t> scenes;
        bool initialized = false;
    };
    struct Interpretation { int relation = -1, effect = -1; std::string subject, object, frame, choice;std::vector<std::string> mentions; };
    struct Alternatives { std::vector<Interpretation> values; bool matched=false, conflicted=false; };
    std::map<std::string, Frame> frames;
    static std::vector<std::string> words(const std::string& text) {
        std::vector<std::string> out; std::string word;
        for (unsigned char c : text) {
            if (std::isalnum(c)) word += (char)std::tolower(c);
            else if (!word.empty()) { out.push_back(word); word.clear(); }
        }
        if (!word.empty()) out.push_back(word); return out;
    }
    static std::string join(const std::vector<std::string>& words) {
        std::string key; for (const auto& w : words) key += w + ' '; return key;
    }
    static int slot(const std::string& word) {
        return word.size() == 2 && word[0] == '$' && word[1] >= '0' && word[1] <= '3' ? word[1] - '0' : -1;
    }
    template<class T>
    static std::set<std::tuple<T, int, T>> update(std::set<std::tuple<T, int, T>> world,
                                                int relation, int effect, const T& a, const T& b) {
        for (auto it = world.begin(); it != world.end();) {
            const auto& [s, r, o] = *it;
            bool remove = r == relation && ((effect == REMOVE && s == a && o == b)
                || (effect == SET_SUBJECT && s == a) || (effect == SET_OBJECT && o == b));
            if (remove) it = world.erase(it); else ++it;
        }
        if (effect != REMOVE) world.emplace(a, relation, b);
        return world;
    }
    bool teach_transition(const std::string& text, const World& before, const World& after,
                          uint64_t scene, bool self_generated = false) {
        if (self_generated) return false;
        std::set<std::string> entities; std::set<int> relations;
        for (const auto* world : {&before, &after}) for (const auto& [a, r, b] : *world) {
            entities.insert(a); entities.insert(b); relations.insert(r);
        }
        auto pattern = words(text); std::map<std::string, int> ids; std::vector<std::string> arguments;
        for (auto& word : pattern) if (entities.count(word)) {
            auto it = ids.find(word);
            if (it == ids.end()) { int id = (int)ids.size(); it = ids.emplace(word, id).first; arguments.push_back(word); }
            word = "$" + std::to_string(it->second);
        }
        if (arguments.size() < 2 || arguments.size() > 4) return false;
        std::set<Rule> possible;
        for (int relation : relations) for (int a = 0; a < (int)arguments.size(); ++a)
            for (int b = 0; b < (int)arguments.size(); ++b) for (int effect = 0; effect < 4; ++effect)
                if (update(before, relation, effect, arguments[a], arguments[b]) == after)
                    possible.insert({relation, a, b, effect});
        auto& frame = frames[join(pattern)]; frame.pattern = pattern;
        if (!frame.initialized) { frame.candidates = possible; frame.initialized = true; }
        else {
            std::set<Rule> intersection;
            std::set_intersection(frame.candidates.begin(), frame.candidates.end(), possible.begin(), possible.end(),
                                  std::inserter(intersection, intersection.begin()));
            frame.candidates = std::move(intersection);
        }
        frame.scenes.insert(scene);
        return true;
    }
    Alternatives alternatives(const std::string& text) const {
        const auto input = words(text);
        Alternatives result;
        std::set<std::tuple<int,int,std::string,std::string,std::string,std::string,std::vector<std::string>>> meanings;
        for (const auto& [key, frame] : frames) {
            if (frame.scenes.size() < 4 || frame.pattern.size() != input.size()) continue;
            std::vector<std::string> arguments(4); bool matched = true;
            for (size_t i = 0; i < input.size(); ++i) {
                int s = slot(frame.pattern[i]);
                if (s < 0) { if (frame.pattern[i] != input[i]) { matched = false; break; } }
                else if (arguments[s].empty()) arguments[s] = input[i];
                else if (arguments[s] != input[i]) { matched = false; break; }
            }
            if (!matched) continue;
            result.matched=true;
            std::vector<std::string> mentions;for(const auto& name:arguments)if(!name.empty())mentions.push_back(name);
            // A matching contradicted/ambiguous frame must not be bypassed by
            // another coincidentally matching surface rule.
            if (frame.candidates.empty()) result.conflicted=true;
            for(const Rule& r:frame.candidates) {
                auto choice=std::to_string(r.relation)+","+std::to_string(r.subject)+","+std::to_string(r.object)+","+std::to_string(r.effect);
                meanings.emplace(r.relation,r.effect,arguments.at(r.subject),arguments.at(r.object),key,choice,mentions);
            }
        }
        for(const auto& [r,effect,a,b,frame,choice,mentions]:meanings) result.values.push_back({r,effect,a,b,frame,choice,mentions});
        return result;
    }
    bool interpret(const std::string& text, Interpretation& result) const {
        auto found=alternatives(text);
        if(found.conflicted || found.values.empty()) return false;
        for(const auto& value:found.values) {
            const auto& first=found.values.front();
            if(std::tie(value.relation,value.effect,value.subject,value.object)!=std::tie(first.relation,first.effect,first.subject,first.object)) return false;
        }
        result=found.values.front(); return true;
    }
    size_t grounded() const {
        size_t n = 0; for (const auto& [key, f] : frames) n += f.scenes.size() >= 4 && f.candidates.size() == 1; return n;
    }
    std::string signature() const {
        std::string out;
        for (const auto& [key, f] : frames) {
            out += key + ":" + std::to_string(f.scenes.size()) + ":";
            for (const auto& r : f.candidates) out += std::to_string(r.relation) + ',' + std::to_string(r.subject) + ',' + std::to_string(r.object) + ',' + std::to_string(r.effect) + ';';
            out += '\n';
        }
        return out;
    }
};
} // namespace laminar::reasoning
