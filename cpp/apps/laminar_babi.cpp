// bAbI: reasoning over supporting facts (goal-directed generation, phase 2). Train on the bAbI chat stream
// (python/babi_stream.py), then for each test story feed the story sentences and "USER: <question>\n\nASSISTANT:"
// and greedily generate the answer; exact match on the first word. Reports accuracy per task, the chance level
// (most frequent answer of the task) and the commitment trace statistics.
//
//   laminar_babi file=data/babi_en10k_train.txt test=data/babi_en10k_test.jsonl [questions=1000] [temp=0] ...params
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace laminar;

static std::string json_str(const std::string& s, size_t& i) {   // minimal JSON string reader; i at the opening quote
    std::string o; ++i;
    for (; i < s.size(); ++i) { char c = s[i]; if (c == '"') { ++i; break; } if (c == '\\' && i + 1 < s.size()) { c = s[++i]; o += c == 'n' ? '\n' : c == 't' ? '\t' : c; } else o += c; }
    return o;
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> rest;
    Params p = Params::from_args(argc, argv, &rest);
    p.vocab = 256; if (p.future_buckets == 32) p.future_buckets = 64;
    auto geti = [&](const char* k, uint64_t d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtoull(it->second.c_str(), nullptr, 10); };
    auto getf = [&](const char* k, double d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtod(it->second.c_str(), nullptr); };
    if (!rest.count("file") || !rest.count("test")) { std::fprintf(stderr, "usage: laminar_babi file=STREAM test=TEST.jsonl [questions=1000] [temp=0] [show=4] ...\n"); return 1; }
    const uint64_t maxq = geti("questions", 1000); const double temp = getf("temp", 0.0); const uint64_t show = geti("show", 4);
    const int gen_len = (int)geti("gen", 24); std::string last_reply;
    const bool gen_auto = geti("gen_auto", 0) != 0;          // reply length from the gold answer: 1.5 x its length + 32, at most 1024
    std::FILE* dump = rest.count("dump") ? std::fopen(rest["dump"].c_str(), "w") : nullptr;   // every held-out reply, for external scoring
    auto jesc = [](const std::string& x) { std::string o; for (unsigned char c : x) { if (c == '"' || c == '\\') { o += '\\'; o += (char)c; } else if (c == '\n') o += "\\n"; else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c; } return o; };
    std::vector<uint8_t> data = read_file_bytes(rest["file"], geti("max_bytes", 0));
    if (data.size() < 65536) { std::fprintf(stderr, "stream too small\n"); return 1; }

    // test stories: {"task": "qa2", "qa": [[question, answer, supporting, [story sentences...]], ...]}
    struct Q { std::string task, question, answer; std::vector<std::string> story; }; std::vector<Q> qs;
    { std::ifstream h(rest["test"]); std::string line;
      while (std::getline(h, line) && qs.size() < maxq) {
          size_t i = line.find("\"task\": \""); if (i == std::string::npos) continue; i += 8; std::string task = json_str(line, i);
          size_t k = line.find("\"qa\": [", i); if (k == std::string::npos) continue; i = k + 7;
          while (i < line.size()) {                       // each qa: [ "q", "a", "sup", [ "s", ... ] ]
              size_t b = line.find('[', i); if (b == std::string::npos || b + 1 >= line.size()) break; i = b + 1;
              if (line[i] == ']') break;
              Q q; q.task = task; i = line.find('"', i); q.question = json_str(line, i); i = line.find('"', i); q.answer = json_str(line, i); i = line.find('"', i); json_str(line, i);
              i = line.find('[', i) + 1; while (i < line.size() && line[i] != ']') { if (line[i] == '"') q.story.push_back(json_str(line, i)); else ++i; }
              i = line.find(']', i + 1) + 1;              // close this qa
              qs.push_back(q); if (qs.size() >= maxq) break;
          }
      } }
    if (qs.empty()) { std::fprintf(stderr, "no test questions parsed\n"); return 1; }

    Organism org(p);
    size_t pos = 0; BlockStats agg;
    while (pos + 1 < data.size()) { size_t n = std::min<size_t>(p.block, data.size() - 1 - pos); org.run_block(data.data() + pos, n + 1); org.end_block(); BlockStats& s = org.stats(); agg.events += s.events; agg.loss_sum += s.loss_sum; agg.seconds += s.seconds; agg.macs += s.macs; s.reset(); pos += n; }
    std::printf("trained: %zu bytes, BPB %.4f, %.0f KB/s, locks %zu, MACs/byte %.0f\n", data.size(), agg.loss_sum / agg.events / 0.6931471805599453, agg.events / agg.seconds / 1024.0, org.tree.live_locks(), (double)agg.macs / std::max<uint64_t>(agg.events, 1));
    org.set_mode(false, false);
    org.seq_trace(true);
    if (rest.count("ablate")) {   // ablate=top:K (the K motifs most specialised to question+answer) or rand:K
        const std::string a = rest["ablate"]; if (a.rfind("level", 0) == 0) { org.silence_level(std::atoi(a.substr(a.find(':') + 1).c_str())); std::printf("ablated level %s bodies\n", a.substr(a.find(':') + 1).c_str()); } else {
        const bool rnd = a.rfind("rand", 0) == 0; const int k = std::atoi(a.substr(a.find(':') + 1).c_str());
        auto ms = org.specialised_motifs(k, rnd, p.seed + 99); org.silence_motifs(ms); std::printf("ablated %zu motifs (%s)\n", ms.size(), rnd ? "random" : "most specialised"); }
    }

    Rng gr(p.seed + 11);
    auto answer = [&](const Q& q) {
        std::string prompt = "\n<|endofdialogue|>\n"; for (auto& s : q.story) prompt += s + "\n"; prompt += "USER: " + q.question + "\n\nASSISTANT:";
        std::vector<uint8_t> pb(prompt.begin(), prompt.end());
        org.free_run = false; org.run_block(pb.data(), pb.size()); org.free_run = true;
        std::string out; uint8_t cur = org.last_symbol();
        const int glen = gen_auto ? (int)std::min<size_t>(q.answer.size() * 3 / 2 + 32, 1024) : gen_len;
        for (int t = 0; t < glen; ++t) {
            uint8_t pair[2] = {cur, 0}; org.run_block(pair, 2);
            const auto& pr = org.distribution(); uint8_t next = 0;
            if (temp <= 0.0) { for (int v = 1; v < 256; ++v) if (pr[v] > pr[next]) next = (uint8_t)v; }
            else { double z = 0; std::vector<double> w(256); for (int v = 0; v < 256; ++v) { w[v] = std::pow((double)pr[v], 1.0 / temp); z += w[v]; } double u = gr.uniform() * z, c = 0; for (int v = 0; v < 256; ++v) { c += w[v]; if (u <= c) { next = (uint8_t)v; break; } } }
            org.commit_outcome(next); out += (char)next; cur = next;
            if (next == '\n' && out.size() > 1) break;
        }
        // the word after "answer:" when the reply thinks out loud, else the first word
        size_t at = out.rfind("answer:"); size_t from = at == std::string::npos ? 0 : at + 7;
        std::string w; for (size_t i = from; i < out.size(); ++i) { char c = out[i]; if (std::isalpha((unsigned char)c)) w += c; else if (!w.empty()) break; }
        last_reply = out; return w;
    };

    std::map<std::string, int> total, correct; std::map<std::string, std::map<std::string, int>> freq;
    for (auto& q : qs) freq[q.task][q.answer]++;
    std::map<std::string, uint64_t> shown;
    for (auto& q : qs) {
        std::string a = answer(q); bool ok = a == q.answer; total[q.task]++; correct[q.task] += ok;
        if (dump) std::fprintf(dump, "{\"task\": \"%s\", \"gold\": \"%s\", \"reply\": \"%s\"}\n", jesc(q.task).c_str(), jesc(q.answer).c_str(), jesc(last_reply).c_str());
        if ((ok ? shown[q.task + "+"]++ : shown[q.task + "-"]++) < show) { std::string r = last_reply.substr(0, 120); for (auto& c : r) if (c == '\n') c = '|'; std::printf("[%s] %s -> \"%s\" (want %s)%s   reply: %s   [%s]\n", q.task.c_str(), q.question.c_str(), a.c_str(), q.answer.c_str(), ok ? "" : "  X", r.c_str(), (org.seq_chain_trace() + org.attn_decision()).c_str()); }
    }
    if (dump) std::fclose(dump);
    if (p.seqmem) std::printf("%s\n", org.seq_report().c_str());
    { std::string g = org.goal_report(); if (!g.empty()) std::printf("%s\n", g.c_str()); }
    { std::string c = org.composer_report(); if (!c.empty()) std::printf("%s\n", c.c_str()); }
    for (auto& [task, n] : total) {
        int best = 0; for (auto& [a, c] : freq[task]) best = std::max(best, c);
        std::printf("BABI %s: %d/%d = %.1f%% exact (chance %.1f%% = most frequent answer)\n", task.c_str(), correct[task], n, 100.0 * correct[task] / n, 100.0 * best / n);
    }
    return 0;
}
