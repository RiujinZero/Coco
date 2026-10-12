#pragma once
// Learn reference phrases from verified event outcomes and recent grounded turns.
// No English reference words or grammatical gender classes are specified here.
#include "event_grounding.h"

namespace laminar::reasoning {
struct ReferenceGrounding {
    using Context = std::vector<EventGrounding::Interpretation>;
    struct Selector {
        int relation = -1, role = 0; // -1: any relation; 0: subject, 1: object, 2: either
        bool operator<(const Selector& s) const { return std::tie(relation, role) < std::tie(s.relation, s.role); }
    };
    struct Phrase { std::set<Selector> candidates; std::set<uint64_t> scenes; bool initialized = false; };
    struct Resolution {
        std::string text, phrase;
        std::set<std::string> candidates;
        bool found = false, resolved = true;
    };
    std::map<std::string, Phrase> phrases;
    static std::set<std::string> select(const Context& context, const Selector& selector) {
        std::set<std::string> out;
        for (const auto& event : context) if (selector.relation < 0 || event.relation == selector.relation) {
            if (selector.role != 1) out.insert(event.subject);
            if (selector.role != 0) out.insert(event.object);
        }
        return out;
    }
    static std::string replace(const std::vector<std::string>& words, size_t begin, size_t end, const std::string& value) {
        std::vector<std::string> result(words.begin(), words.begin() + begin); result.push_back(value);
        result.insert(result.end(), words.begin() + end, words.end()); return EventGrounding::join(result);
    }
    bool teach_transition(const std::string& text, const Context& context,
                          const EventGrounding::World& before, const EventGrounding::World& after,
                          const EventGrounding& events, uint64_t scene, bool self_generated = false) {
        if (self_generated || before == after || context.empty()) return false;
        const auto words = EventGrounding::words(text);
        std::set<std::string> names; std::set<int> relations{-1};
        for (const auto* state : {&before, &after}) for (const auto& [a, r, b] : *state) { names.insert(a); names.insert(b); }
        for (const auto& event : context) { names.insert(event.subject); names.insert(event.object); relations.insert(event.relation); }
        bool taught = false;
        for (size_t begin = 0; begin < words.size(); ++begin) for (size_t end = begin + 1; end <= words.size() && end <= begin + 4; ++end) {
            if (std::any_of(words.begin() + begin, words.begin() + end, [&](const auto& w) { return names.count(w); })) continue;
            std::set<Selector> possible;
            for (int relation : relations) for (int role = 0; role < 3; ++role) {
                Selector s{relation, role}; auto candidates = select(context, s);
                if (candidates.size() != 1) continue;
                EventGrounding::Interpretation event;
                if (!events.interpret(replace(words, begin, end, *candidates.begin()), event)) continue;
                if (EventGrounding::update(before, event.relation, event.effect, event.subject, event.object) == after) possible.insert(s);
            }
            std::string phrase = EventGrounding::join({words.begin() + begin, words.begin() + end});
            auto old = phrases.find(phrase);
            if (possible.empty() && old == phrases.end()) continue;
            auto& p = phrases[phrase];
            if (!p.initialized) { p.candidates = possible; p.initialized = true; }
            else {
                std::set<Selector> common;
                std::set_intersection(p.candidates.begin(), p.candidates.end(), possible.begin(), possible.end(), std::inserter(common, common.begin()));
                p.candidates = std::move(common);
            }
            p.scenes.insert(scene); taught = true;
        }
        return taught;
    }
    // Longest learned phrase first. Equivalent selectors may agree in one
    // context; retain their union so uncertainty is exposed when they diverge.
    Resolution resolve(const std::string& text, const Context& context) const {
        Resolution result; const auto words = EventGrounding::words(text);
        for (size_t i = 0; i < words.size();) {
            size_t end = std::min(words.size(), i + 4), match = i;
            const Phrase* learned = nullptr; std::string phrase;
            for (; end > i; --end) {
                auto key = EventGrounding::join({words.begin() + i, words.begin() + end}); auto it = phrases.find(key);
                if (it != phrases.end() && it->second.scenes.size() >= 4) { learned = &it->second; phrase = key; match = end; break; }
            }
            if (!learned) { result.text += words[i++] + ' '; continue; }
            result.found = true; std::set<std::string> candidates; bool missing = learned->candidates.empty();
            for (const auto& selector : learned->candidates) {
                auto selected = select(context, selector); missing |= selected.empty(); candidates.insert(selected.begin(), selected.end());
            }
            if (missing || candidates.size() != 1) {
                result.resolved = false; result.phrase = phrase; result.candidates = std::move(candidates); result.text = text; return result;
            }
            result.text += *candidates.begin() + ' '; i = match;
        }
        return result;
    }
    static std::string bind(const std::string& text, const std::string& phrase, const std::string& entity) {
        auto words = EventGrounding::words(text), pattern = EventGrounding::words(phrase);
        for (size_t i = 0; i + pattern.size() <= words.size(); ++i)
            if (std::equal(pattern.begin(), pattern.end(), words.begin() + i)) return replace(words, i, i + pattern.size(), entity);
        return text;
    }
    std::string signature() const {
        std::string out;
        for (const auto& [phrase, p] : phrases) {
            out += phrase + ':' + std::to_string(p.scenes.size()) + ':';
            for (const auto& s : p.candidates) out += std::to_string(s.relation) + ',' + std::to_string(s.role) + ';';
            out += '\n';
        }
        return out;
    }
};
} // namespace laminar::reasoning
