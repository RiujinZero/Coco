// bAbI as multiple choice (a TASK test for the memory tiers, docs/PYRAMID_MEMORY.md): train the organism online on the bAbI
// dialogues (data/babi_all_train.txt: story lines, "USER: question", "ASSISTANT: answer", "<|endofdialogue|>"), then, frozen,
// score every candidate answer of the task (the task's answer vocabulary) after each test story + question and count the
// questions whose correct answer is the most probable. Per task and overall; chance = 1 / candidates.
//
// usage: babi_probe TRAIN_BYTES [key=value organism params...]
//   env: BABI_TRAIN (default data/babi_all_train.txt), BABI_TEST (data/babi_test.tsv from the jsonl: task, question, answer,
//        candidates|..., story|...), NPER=20 (test questions per task), PASSES=1 (over the TRAIN_BYTES), TASKS (e.g. "1,2,3")
#include "laminar/organism.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace laminar;
static void feed_text(Organism& m, const std::string& s) { std::vector<uint8_t> b(s.size() + 1); b[0] = m.last_symbol(); std::memcpy(b.data() + 1, s.data(), s.size()); m.run_block(b.data(), b.size()); }
static std::vector<std::string> split(const std::string& s, char c) { std::vector<std::string> out; std::string cur; for (char ch : s) { if (ch == c) { out.push_back(cur); cur.clear(); } else cur += ch; } out.push_back(cur); return out; }
struct Item { std::string task, q, a; std::vector<std::string> cands, story; };

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: babi_probe TRAIN_BYTES [params]");
        const size_t ntr = std::stoul(argv[1]);
        const char* home = "/mnt/c/Users/user/Documents/projects/Coco/data/";
        const std::string trp = std::getenv("BABI_TRAIN") ? std::getenv("BABI_TRAIN") : std::string(home) + "babi_all_train.txt";
        const std::string tep = std::getenv("BABI_TEST") ? std::getenv("BABI_TEST") : std::string(home) + "babi_test.tsv";
        const int nper = std::getenv("NPER") ? std::atoi(std::getenv("NPER")) : 20, passes = std::getenv("PASSES") ? std::atoi(std::getenv("PASSES")) : 1;
        std::vector<int> only; if (std::getenv("TASKS")) for (auto& t : split(std::getenv("TASKS"), ',')) if (!t.empty()) only.push_back(std::stoi(t));
        std::vector<std::string> args{"babi"}; for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        std::string text; { std::ifstream f(trp, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + trp); std::stringstream ss; ss << f.rdbuf(); text = ss.str(); }
        if (text.size() > ntr) text.resize(ntr);
        model.set_mode(true, true);
        for (int pass = 0; pass < passes; ++pass)
            for (size_t off = 0; off < text.size(); off += 262144) { const size_t n = std::min<size_t>(262144, text.size() - off);
                std::vector<uint8_t> b(text.begin() + off, text.begin() + off + n); b.insert(b.begin(), model.last_symbol()); model.run_block(b.data(), b.size()); model.end_block(); }
        model.set_mode(false, false); feed_text(model, "<|endofdialogue|>\n");
        std::vector<Item> items; std::map<std::string, int> taken;
        { std::ifstream f(tep); if (!f) throw std::runtime_error("cannot read " + tep); std::string line;
          while (std::getline(f, line)) { auto c = split(line, '\t'); if (c.size() < 5) continue;
              const int tn = std::stoi(c[0].substr(2)); if (!only.empty() && std::find(only.begin(), only.end(), tn) == only.end()) continue;
              if (taken[c[0]]++ >= nper) continue;
              items.push_back(Item{c[0], c[1], c[2], split(c[3], '|'), split(c[4], '|')}); } }
        std::map<std::string, std::pair<int, int>> score;   // task -> (correct, total)
        std::map<std::string, double> chance;
        for (const Item& it : items) {
            std::string ctx; for (auto& s : it.story) ctx += s + "\n"; ctx += "USER: " + it.q + "\n\nASSISTANT: ";
            int fds[2]; if (pipe(fds) != 0) return 1; std::cout.flush(); pid_t pid = fork();
            if (pid == 0) {   // one question: the story once, then every candidate in a grandchild from that state
                close(fds[0]); feed_text(model, ctx); int best = -1; double bl = -1e300;
                for (size_t c = 0; c < it.cands.size(); ++c) { int p2[2]; if (pipe(p2) != 0) _exit(1); pid_t ch = fork();
                    if (ch == 0) { close(p2[0]); model.stats() = BlockStats{}; feed_text(model, it.cands[c] + "\n"); double v = model.stats().loss_sum; ssize_t w = write(p2[1], &v, sizeof v); (void)w; _exit(0); }
                    close(p2[1]); double v = 0; ssize_t r = read(p2[0], &v, sizeof v); (void)r; close(p2[0]); waitpid(ch, nullptr, 0);
                    if (-v > bl) { bl = -v; best = (int)c; } }
                const int ok = best >= 0 && it.cands[best] == it.a; ssize_t w = write(fds[1], &ok, sizeof ok); (void)w; _exit(0); }
            close(fds[1]); int ok = 0; ssize_t r = read(fds[0], &ok, sizeof ok); (void)r; close(fds[0]); waitpid(pid, nullptr, 0);
            auto& sc = score[it.task]; sc.first += ok; ++sc.second; chance[it.task] = 100.0 / (double)it.cands.size();
            feed_text(model, "<|endofdialogue|>\n");   // (frozen: only the stream state moves on)
        }
        int tc = 0, tt = 0; double ch = 0;
        std::cout << "bAbI multiple choice after " << text.size() * passes << " training bytes (" << passes << " pass" << (passes > 1 ? "es" : "") << "): % correct (chance)\n";
        for (int t = 1; t <= 20; ++t) { const std::string k = "qa" + std::to_string(t); if (!score.count(k)) continue; auto& sc = score[k];
            std::printf("  %-5s %5.1f  (%4.1f)\n", k.c_str(), 100.0 * sc.first / std::max(1, sc.second), chance[k]); tc += sc.first; tt += sc.second; ch += chance[k]; }
        std::printf("  all   %5.1f  (%4.1f)\n", 100.0 * tc / std::max(1, tt), ch / std::max<size_t>(1, score.size()));
        return 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
