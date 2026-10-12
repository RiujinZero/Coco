#pragma once
// Verifiable written-answer tasks from curriculum_v2 (bAbI, CLEVR, events, paraphrase, SCAN, COGS, ProPara, ...).
// Every episode is rendered exactly as in the curriculum text the bodies read during pretraining:
//     <story lines>\nUSER: <question>\n\nASSISTANT: <answer>\n<|endofdialogue|>\n
// The prompt runs through "ASSISTANT:"; the answer is what follows, up to the newline. A written answer is correct
// when it equals the reference after trimming surrounding whitespace (exact match, the verifier).
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace laminar::answers {

struct Episode {
    std::string prompt, answer, task;   // prompt ends with "ASSISTANT:"; answer without the leading space or newline
    uint8_t source = 0;
    template <class Ar> void state_io(Ar& ar) { ar(prompt); ar(answer); ar(task); ar(source); }
};

inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (unsigned char)s[a] <= ' ') ++a;
    while (b > a && (unsigned char)s[b - 1] <= ' ') --b;
    return s.substr(a, b - a);
}
inline bool correct(const std::string& written, const std::string& reference) { return trim(written) == trim(reference); }

static constexpr const char* end_marker = "<|endofdialogue|>";

// Training text: episodes separated by <|endofdialogue|>; the last "ASSISTANT:" splits prompt and answer.
inline std::vector<Episode> load_text(const std::string& path, uint8_t source, size_t max_bytes, size_t skip_bytes = 0) {
    std::ifstream f(path, std::ios::binary); if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg((std::streamoff)skip_bytes);
    std::string data(max_bytes, '\0'); f.read(data.data(), (std::streamsize)max_bytes); data.resize((size_t)f.gcount());
    std::vector<Episode> out; size_t pos = 0;
    if (skip_bytes) { const size_t first = data.find(end_marker); if (first == std::string::npos) return out; pos = first + std::string(end_marker).size(); }
    while (true) {
        const size_t end = data.find(end_marker, pos); if (end == std::string::npos) break;
        std::string ep = data.substr(pos, end - pos); pos = end + std::string(end_marker).size();
        while (!ep.empty() && (ep.front() == '\n' || ep.front() == '\r')) ep.erase(ep.begin());
        const size_t a = ep.rfind("ASSISTANT:"); if (a == std::string::npos) continue;
        Episode e; e.prompt = ep.substr(0, a + 10); e.answer = trim(ep.substr(a + 10)); e.source = source;
        if (e.answer.empty() || e.answer.find('\n') != std::string::npos) continue;   // single-line answers only
        out.push_back(std::move(e));
    }
    return out;
}

// Minimal JSON helpers for the curriculum's test files (strings, arrays of strings; no numbers needed).
inline std::string json_string(const std::string& s, size_t& i) {
    if (s[i] != '"') throw std::runtime_error("json: expected string");
    std::string out; ++i;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            if (c == 'n') out += '\n'; else if (c == 't') out += '\t'; else if (c == 'u' && i + 4 < s.size()) {
                const unsigned cp = (unsigned)std::stoul(s.substr(i + 1, 4), nullptr, 16); i += 4;
                if (cp < 0x80) out += (char)cp; else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
            } else out += c;
        } else out += s[i];
        ++i;
    }
    ++i; return out;
}
inline void skip_ws(const std::string& s, size_t& i) { while (i < s.size() && (s[i] == ' ' || s[i] == ',' || s[i] == ':')) ++i; }
inline std::string field(const std::string& line, const std::string& key) {
    const size_t k = line.find("\"" + key + "\""); if (k == std::string::npos) return "";
    size_t i = k + key.size() + 2; skip_ws(line, i); return json_string(line, i);
}

// Test items: {"task","qa":[[question, answer, _, [story...]]]}, {"command","action"}, or episode text (.txt).
inline std::vector<Episode> load_probes(const std::string& path, uint8_t source, size_t per_task, uint64_t seed) {
    std::vector<Episode> all;
    if (path.size() > 4 && path.substr(path.size() - 4) == ".txt") all = load_text(path, source, 4u << 20);
    else {
        std::ifstream f(path); if (!f) throw std::runtime_error("cannot open " + path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            Episode e; e.source = source;
            if (line.find("\"command\"") != std::string::npos) {
                e.task = "command"; e.prompt = "USER: " + field(line, "command") + "\n\nASSISTANT:"; e.answer = field(line, "action");
            } else {
                e.task = field(line, "task");
                size_t i = line.find("\"qa\""); if (i == std::string::npos) continue;
                i = line.find('[', i) + 1; i = line.find('[', i) + 1; skip_ws(line, i);
                const std::string q = json_string(line, i); skip_ws(line, i);
                const std::string a = json_string(line, i); skip_ws(line, i);
                json_string(line, i); skip_ws(line, i);
                std::string story;
                if (line[i] == '[') { ++i; skip_ws(line, i); while (i < line.size() && line[i] == '"') { story += json_string(line, i) + "\n"; skip_ws(line, i); } }
                e.prompt = story + "USER: " + q + "\n\nASSISTANT:"; e.answer = a;
            }
            if (e.answer.find('\n') == std::string::npos && !e.answer.empty()) all.push_back(std::move(e));
        }
    }
    // a fixed, task-stratified sample
    std::mt19937_64 rng(seed); std::shuffle(all.begin(), all.end(), rng);
    std::vector<Episode> out; std::vector<std::pair<std::string, size_t>> counts;
    for (auto& e : all) {
        auto it = std::find_if(counts.begin(), counts.end(), [&](auto& c) { return c.first == e.task; });
        if (it == counts.end()) { counts.push_back({e.task, 0}); it = counts.end() - 1; }
        if (it->second < per_task) { ++it->second; out.push_back(e); }
    }
    return out;
}

} // namespace laminar::answers
