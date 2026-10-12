// Zero-shot scoring of BabyLM minimal pairs / multiple choice with a saved model (read-only).
//
// Every candidate is scored from the SAME model state: the parent loads the checkpoint, freezes the model (no
// learning, no structure), reads a newline, and then forks one child per candidate. The child reads the candidate's
// context, then its completion, and reports the completion's summed log-probability (natural log, over bytes) and
// byte count; the operating system discards everything the child changed (sequence memory, workspace, agency
// state), so no candidate can see another. This is the byte-level analogue of the pipeline's causal backend
// (summed completion log-probability; length-normalized where the pipeline normalizes).
//
// usage: model_babylm_score DIR ITEMS.jsonl OUT.jsonl [workers=16] [limit=0] [STATE_FILE=DIR/checkpoint.state]
//   DIR: a run directory with config.json ("model_params") and checkpoint.state (organism first).
//   ITEMS: scripts/babylm_export.py output; OUT gets one line per candidate with its log-probability.
#include "laminar/organism.h"
#include "laminar/state_io.h"
#include "answer_tasks.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sys/wait.h>
#include <unistd.h>

using namespace laminar;
// Feeds bytes as a continuation of the model's own stream (the last symbol, then the bytes), through the public run_block.
static void feed_bytes(laminar::Organism& m, const unsigned char* p, size_t n) {
    std::vector<uint8_t> buf(n + 1); buf[0] = m.last_symbol(); std::memcpy(buf.data() + 1, p, n); m.run_block(buf.data(), buf.size());
}
static void feed_str(laminar::Organism& m, const std::string& s) { feed_bytes(m, (const unsigned char*)s.data(), s.size()); }
namespace an = laminar::answers;

struct Cand { std::string line, context, completion; };

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: model_babylm_score DIR ITEMS.jsonl OUT.jsonl [workers] [limit]");
        const std::string dir = argv[1]; const int workers = argc > 4 ? std::stoi(argv[4]) : 16; const size_t limit = argc > 5 ? std::stoul(argv[5]) : 0;
        const std::string state_file = argc > 6 ? std::string(argv[6]) : dir + "/checkpoint.state";
        std::ifstream cf(dir + "/config.json"); std::string config((std::istreambuf_iterator<char>(cf)), {});
        std::smatch m;
        if (!std::regex_search(config, m, std::regex("\"model_params\":\\[([^\\]]*)\\]"))) throw std::runtime_error("no model_params in config");
        std::vector<std::string> args{"score"};
        { std::string list = m[1]; std::regex item("\"([^\"]*)\""); for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) args.push_back((*it)[1]); }
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        { state::Reader r(state_file); model.load_state(r); }   // the organism is first; the rest is run state
        model.set_mode(false, false);
        auto feed = [&](const std::string& s) { feed_str(model, s); };
        feed("\n");

        std::vector<Cand> cands; { std::ifstream in(argv[2]); std::string line;
            while (std::getline(in, line)) { if (line.empty()) continue; Cand c; c.line = line; c.context = an::field(line, "context"); c.completion = an::field(line, "completion"); cands.push_back(std::move(c)); if (limit && cands.size() >= limit) break; } }
        std::ofstream out(argv[3]);
        std::cout.flush(); std::cerr.flush();
        struct Job { pid_t pid; int fd; size_t idx; };
        std::vector<Job> running; size_t next = 0, done = 0;
        auto reap = [&](Job j) {
            double v[2] = {0, 0}; ssize_t got = 0; char* p = (char*)v;
            while (got < (ssize_t)sizeof v) { ssize_t r = read(j.fd, p + got, sizeof v - got); if (r <= 0) break; got += r; }
            close(j.fd); int status = 0; waitpid(j.pid, &status, 0);
            if (got != (ssize_t)sizeof v) throw std::runtime_error("a scoring child failed");
            std::string line = cands[j.idx].line; line.pop_back();   // drop the closing brace, append the score
            out << line << ",\"logprob\":" << -v[0] << ",\"bytes\":" << (long long)v[1] << "}\n";
            ++done;
        };
        while (next < cands.size() || !running.empty()) {
            while (next < cands.size() && (int)running.size() < workers) {
                int fds[2]; if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
                pid_t pid = fork(); if (pid < 0) throw std::runtime_error("fork failed");
                if (pid == 0) {   // child: score one candidate from the parent's state, report, exit without cleanup
                    close(fds[0]);
                    feed(cands[next].context);
                    model.stats() = BlockStats{};
                    feed(cands[next].completion);
                    double v[2] = {model.stats().loss_sum, (double)cands[next].completion.size()};
                    ssize_t w = write(fds[1], v, sizeof v); (void)w; _exit(0);
                }
                close(fds[1]); running.push_back({pid, fds[0], next++});
            }
            reap(running.front()); running.erase(running.begin());
            if (done % 20000 == 0) std::cerr << "scored " << done << '/' << cands.size() << '\n';
        }
        std::cerr << "scored " << done << " candidates\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
