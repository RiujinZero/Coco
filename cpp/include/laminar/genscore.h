#pragma once
// Free-generation scoring shared by the harnesses: strict loop metrics, word validity, verbatim copying.
#include <string>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <cctype>

namespace laminar {

struct GenScore { double distinct4 = 0, repeated4 = 0, repeated8 = 0; int loop_len = 0, longest_repeat = 0; double word_valid = 0, unique_words = 0; size_t n = 0, words = 0; double copy32 = 0, mean_match = 0; bool looped = false; };

// Repetition is judged the strict way: any span of >= 64 bytes that recurs inside the generated text, or more than
// 30% repeated 8-grams, counts as a loop — a 250-word generation must clear both.
inline GenScore score_text(const std::string& g, const std::set<std::string>& vocab) {
    GenScore s; s.n = g.size(); if (g.size() < 16) return s;
    std::map<std::string, int> g4, g8; size_t t4 = 0, r4 = 0, t8 = 0, r8 = 0;
    for (size_t i = 0; i + 4 <= g.size(); ++i) { if (g4[g.substr(i, 4)]++) ++r4; ++t4; }
    for (size_t i = 0; i + 8 <= g.size(); ++i) { if (g8[g.substr(i, 8)]++) ++r8; ++t8; }
    s.distinct4 = (double)g4.size() / t4; s.repeated4 = (double)r4 / t4; s.repeated8 = t8 ? (double)r8 / t8 : 0.0;
    // longest internal verbatim repeat: seed on 16-grams, extend
    std::map<std::string, std::vector<size_t>> seeds;
    for (size_t i = 0; i + 16 <= g.size(); ++i) {
        auto& v = seeds[g.substr(i, 16)];
        for (size_t j : v) { size_t L = 0; while (i + L < g.size() && g[j + L] == g[i + L] && j + L < i) ++L; s.longest_repeat = std::max(s.longest_repeat, (int)L); }
        if (v.size() < 8) v.push_back(i);
    }
    std::string tail = g.size() > 256 ? g.substr(g.size() - 256) : g;
    for (int per = 1; per <= (int)tail.size() / 3; ++per) { bool ok = true; for (size_t i = per; i < tail.size() && ok; ++i) ok = tail[i] == tail[i - per]; if (ok) { s.loop_len = per; break; } }
    // words: maximal runs of letters; valid if seen in the training vocabulary
    size_t valid = 0; std::set<std::string> uniq; std::string w;
    auto flush = [&]() { if (w.size() >= 2) { ++s.words; valid += vocab.count(w) > 0; uniq.insert(w); } w.clear(); };
    for (char c : g) { if (std::isalpha((unsigned char)c)) w += (char)std::tolower((unsigned char)c); else flush(); } flush();
    s.word_valid = s.words ? (double)valid / s.words : 0.0; s.unique_words = s.words ? (double)uniq.size() / s.words : 0.0;
    s.looped = s.loop_len > 0 || s.longest_repeat >= 64 || s.repeated8 > 0.30;
    return s;
}

}  // namespace laminar
