// Layer conformance suite: does each part of the body do the job the design doc gives it?
//
// usage: model_conformance DIR STATE_FILE TEXT_FILE [calibrate_bytes=131072] [measure_bytes=131072] [OUT.json]
//   DIR: run directory with config.json ("model_params"). TEXT_FILE: held-out text. Read-only: learning is off.
//
// Pass 1 (calibrate) records every body's mean output per layer. Then each ARM is run from that same state in a
// forked child over the same measured bytes, with one part of the bodies removed or frozen:
//   actual; silent bodies (L6 = 0); constant bodies (each body emits its own mean L6); constant L1 / L2 / L3 / L5
//   (that layer frozen at the body's own mean output); no neighbour routing; no neighbour messages into L3; no L4 return.
// For each arm: full-model bits/byte (sequence memory included), body-only bits/byte (the bodies' readout alone),
// MACs/byte, neighbour accepts/byte, L4 runs/byte.
//
// Checks (gains in bits/byte; "full" = the whole model, "body" = the bodies' own readout):
//   C1 bodies add value over the sequence memory      full(silent) - full(actual)        >= 0.05
//   C2 CONTEXT COMPUTATION adds value over the memory  full(constant) - full(actual)      >= 0.02   <- the headline gate
//   C3 each hidden layer does work (L1, L2, L3, L5)    body(constant layer) - body(actual) >= 0.01
//   C4 neighbour routing does work                     body(no routing) - body(actual)    >= 0.01
//   C5 neighbour messages inform L3                    body(no messages) - body(actual)   >= 0.005
//   C6 the L4 return does work (when L4 runs at all)   body(no return) - body(actual)     >= 0.005
//   C7 early information death happens                 fraction of admitted bodies dying before L6 >= 0.10
//   C8 no layer is pinned                              saturated fraction of each layer   <= 0.25
#include "laminar/organism.h"
#include "laminar/state_io.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>

using namespace laminar;

#include "answer_tasks.h"
constexpr int MAX_TASKS = 4;
struct ArmResult { double full = 0, body = 0, macs = 0, accepts = 0, l4 = 0; double skill[MAX_TASKS] = {0}; };
struct Cand { long item = 0; int cand = 0, label = 0; bool norm = false; std::string ctx, comp; };
static long num_field(const std::string& line, const std::string& key) {
    const size_t k = line.find("\"" + key + "\":"); if (k == std::string::npos) return -1;
    size_t i = k + key.size() + 3; while (i < line.size() && line[i] == ' ') ++i; return std::strtol(line.c_str() + i, nullptr, 10);
}
static void feed_text(Organism& m, const std::string& s) { std::vector<uint8_t> b(s.size() + 1); b[0] = m.last_symbol(); std::memcpy(b.data() + 1, s.data(), s.size()); m.run_block(b.data(), b.size()); }
// Zero-shot accuracy of the model's current state on one task (pipeline rule: highest summed log-probability wins,
// length-normalised where the item says so; ties count as the expected random pick). One forked child per candidate.
static double score_task(Organism& model, const std::vector<Cand>& cands, int workers) {
    std::vector<double> lp(cands.size(), 0.0);
    struct Job { pid_t pid; int fd; size_t idx; }; std::vector<Job> run; size_t next = 0;
    auto reap = [&](Job j) { double v[2] = {0, 0}; ssize_t got = 0; char* q = (char*)v;
        while (got < (ssize_t)sizeof v) { ssize_t r = read(j.fd, q + got, sizeof v - got); if (r <= 0) break; got += r; }
        close(j.fd); int st = 0; waitpid(j.pid, &st, 0); if (got != (ssize_t)sizeof v) throw std::runtime_error("scoring child failed");
        lp[j.idx] = cands[j.idx].norm ? -v[0] / std::max(1.0, v[1]) : -v[0]; };
    while (next < cands.size() || !run.empty()) {
        while (next < cands.size() && (int)run.size() < workers) {
            int fds[2]; if (pipe(fds) != 0) throw std::runtime_error("pipe failed"); pid_t pid = fork();
            if (pid == 0) { close(fds[0]); feed_text(model, cands[next].ctx); model.stats() = BlockStats{}; feed_text(model, cands[next].comp);
                double v[2] = {model.stats().loss_sum, (double)cands[next].comp.size()}; ssize_t w = write(fds[1], v, sizeof v); (void)w; _exit(0); }
            close(fds[1]); run.push_back({pid, fds[0], next++});
        }
        reap(run.front()); run.erase(run.begin());
    }
    double correct = 0; size_t items = 0;
    for (size_t i = 0; i < cands.size();) {
        size_t j = i; while (j < cands.size() && cands[j].item == cands[i].item) ++j;
        double best = -1e300; for (size_t k = i; k < j; ++k) best = std::max(best, lp[k]);
        int ties = 0; bool hit = false; for (size_t k = i; k < j; ++k) if (lp[k] == best) { ++ties; hit |= cands[k].cand == cands[i].label; }
        correct += hit ? 1.0 / ties : 0.0; ++items; i = j;
    }
    return items ? 100.0 * correct / items : 0.0;
}

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: model_conformance DIR STATE_FILE TEXT_FILE [calibrate_bytes] [measure_bytes] [OUT.json]");
        const std::string dir = argv[1], state_file = argv[2];
        const size_t ncal = argc > 4 ? std::stoul(argv[4]) : 131072, nmeas = argc > 5 ? std::stoul(argv[5]) : 131072;
        const std::string out_json = argc > 6 ? argv[6] : "";
        const std::string items_dir = argc > 7 ? argv[7] : ""; const long stride = argc > 8 ? std::stol(argv[8]) : 10; const int workers = argc > 9 ? std::stoi(argv[9]) : 24;
        const int skill_arms = argc > 10 ? std::stoi(argv[10]) : 99;   // skills are scored for the first N arms only (3 = actual, silent, constant)
        // skills: every stride-th item of each task (all its candidates), scored per arm
        std::vector<std::string> task_name; std::vector<std::vector<Cand>> task_cands;
        if (!items_dir.empty()) for (const char* t : {"blimp", "entity_tracking", "blimp_supplement"}) {
            std::ifstream in(items_dir + "/" + t + ".jsonl"); if (!in) continue; std::vector<Cand> cs; std::string line;
            while (std::getline(in, line)) { if (line.empty()) continue; Cand c; c.item = num_field(line, "item"); if (c.item % stride) continue;
                c.cand = (int)num_field(line, "cand"); c.label = (int)num_field(line, "label"); c.norm = line.find("\"normalize\": true") != std::string::npos;
                c.ctx = laminar::answers::field(line, "context"); c.comp = laminar::answers::field(line, "completion"); cs.push_back(std::move(c)); }
            if (!cs.empty() && (int)task_name.size() < MAX_TASKS) { task_name.push_back(t); task_cands.push_back(std::move(cs)); }
        }
        std::ifstream cf(dir + "/config.json"); std::string config((std::istreambuf_iterator<char>(cf)), {});
        std::smatch m;
        if (!std::regex_search(config, m, std::regex("\"model_params\":\\[([^\\]]*)\\]"))) throw std::runtime_error("no model_params in config");
        std::vector<std::string> args{"conformance"};
        { std::string list = m[1]; std::regex item("\"([^\"]*)\""); for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) args.push_back((*it)[1]); }
        args.push_back("train=0");   // read-only: no learning buffers (8.1 -> 0.35 GiB at 82k bodies, same outputs; forks stay cheap)
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        { state::Reader r(state_file); model.load_state(r); }
        model.set_mode(false, false);
        const uint32_t U = model.params().units; const int32_t xg = model.params().exact_gain;
        std::string text; { std::ifstream in(argv[3], std::ios::binary); text.assign((std::istreambuf_iterator<char>(in)), {}); }
        if (text.size() < ncal + nmeas + 2) throw std::runtime_error("text too short");

        // ---- calibration: per-body mean output of L1, L2, L3, L5, L6; reach / saturation statistics
        constexpr int NL = 5; const int width[NL] = {Dims::A, Dims::B, Dims::C, Dims::E, Dims::F}; const int lid[NL] = {LI_1, LI_2, LI_3, LI_5, LI_6};
        const char* lname[NL] = {"L1", "L2", "L3", "L5", "L6"};
        std::vector<std::vector<double>> sum(NL); std::vector<std::vector<uint32_t>> cnt(NL);
        for (int l = 0; l < NL; ++l) { sum[l].assign((size_t)U * width[l], 0.0); cnt[l].assign(U, 0); }
        auto layer = [](const Trace& t, int l) -> const int8_t* { return l == 0 ? t.l1 : l == 1 ? t.l2 : l == 2 ? t.l3 : l == 3 ? t.l5 : t.l6; };
        auto reached = [](const Trace& t, int l) { return l == 0 ? (t.reached & R_L1) : l <= 2 ? (t.reached & R_L23) : l == 3 ? (t.reached & R_L5) : (t.reached & R_L6); };
        uint64_t admitted = 0, reached6 = 0, outs[NL] = {0}, sat[NL] = {0};
        model.on_event = [&](uint8_t, uint8_t, const int32_t*, const float*) {
            model.for_each_trace([&](const Trace& t) {
                const uint32_t u = t.unit; ++admitted; reached6 += (t.reached & R_L6) != 0;
                const int g = model.exact_unit(u) ? xg : 1;
                for (int l = 0; l < NL; ++l) {
                    if (!reached(t, l)) continue; const int8_t* y = layer(t, l);
                    if (l == 4 && !t.emitted) continue;
                    for (int j = 0; j < width[l]; ++j) { sum[l][(size_t)u * width[l] + j] += (l == 4 ? g : 1) * y[j]; ++outs[l]; sat[l] += y[j] >= 127 || y[j] <= -127; }
                    ++cnt[l][u];
                }
            });
        };
        std::vector<uint8_t> buf; auto feed = [&](size_t off, size_t n) { buf.assign(text.begin() + off, text.begin() + off + n + 1); model.run_block(buf.data(), buf.size()); };
        feed(0, ncal);
        model.on_event = nullptr;

        // ---- arms
        enum { A_ACTUAL, A_SILENT, A_CONST, A_C1, A_C2, A_C3, A_C5, A_NOROUTE, A_NOMSG, A_NORET, A_RSILENT, A_NOCONS, A_NOWS, NARM };
        const char* arm_name[NARM] = {"actual", "silent bodies", "constant bodies", "constant L1", "constant L2", "constant L3", "constant L5",
                                      "no neighbour routing", "no neighbour messages", "no L4 return", "routed bodies silent", "no route consequence", "no workspace memory in"};
        ArmResult res[NARM];
        for (int arm = 0; arm < NARM; ++arm) {
            int fds[2]; if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
            std::cout.flush(); pid_t pid = fork();
            if (pid == 0) {
                close(fds[0]);
                double bl = 0; uint64_t bn = 0;
                model.on_event = [&](uint8_t, uint8_t target, const int32_t* ws, const float*) { bl += model.body_loss(ws, target); ++bn; };
                auto freeze = [&](int l) { return [&, l](uint32_t u, int8_t* y) { if (!cnt[l][u]) return; const int g = (l == 4 && model.exact_unit(u)) ? xg : 1;
                    for (int j = 0; j < width[l]; ++j) { long v = std::lround(sum[l][(size_t)u * width[l] + j] / cnt[l][u] / g); y[j] = (int8_t)std::max(-127L, std::min(127L, v)); } }; };
                if (arm == A_SILENT) model.l6_hook = [](uint32_t, int8_t* l6) { std::memset(l6, 0, Dims::F); };
                if (arm == A_CONST) model.l6_hook = freeze(4);
                if (arm >= A_C1 && arm <= A_C5) { const int l = arm - A_C1; auto f = freeze(l); const int li = lid[l];
                    model.layer_hook = [f, li](uint32_t u, int lay, int8_t* y) { if (lay == li) f(u, y); }; }
                if (arm == A_NOROUTE) model.diag_no_routing = true;
                if (arm == A_NOMSG) model.diag_zero_msgs = true;
                if (arm == A_NORET) model.diag_zero_return = true;
                if (arm == A_RSILENT) model.diag_routed_silent = true;
                if (arm == A_NOCONS) model.diag_no_consequence = true;
                if (arm == A_NOWS) model.diag_zero_wsin = true;
                model.stats() = BlockStats{}; feed(ncal, nmeas);
                const BlockStats& s = model.stats(); const double B = (double)std::max<uint64_t>(1, s.bytes);
                ArmResult r{s.loss_sum / B / std::log(2.0), bl / std::max<uint64_t>(1, bn) / std::log(2.0), s.macs / B, s.neighbor_accepts / B, s.l4_runs / B};
                if (!task_name.empty() && arm < skill_arms) { model.on_event = nullptr; feed_text(model, "\n"); for (size_t t = 0; t < task_name.size(); ++t) r.skill[t] = score_task(model, task_cands[t], workers); }
                ssize_t w = write(fds[1], &r, sizeof r); (void)w; _exit(0);
            }
            close(fds[1]); ssize_t got = read(fds[0], &res[arm], sizeof(ArmResult)); close(fds[0]); int st = 0; waitpid(pid, &st, 0);
            if (got != (ssize_t)sizeof(ArmResult)) throw std::runtime_error(std::string("arm failed: ") + arm_name[arm]);
        }

        // ---- report
        std::ostringstream o; o.setf(std::ios::fixed); o.precision(4);
        o << "arm                        full    body    MACs/B  accepts/B  L4/B\n";
        for (int a = 0; a < NARM; ++a) { std::string n = arm_name[a]; n.resize(26, ' ');
            o << n << " " << res[a].full << "  " << res[a].body << "  " << (int)res[a].macs << "    " << res[a].accepts << "     " << res[a].l4 << "\n"; }
        const ArmResult& A = res[A_ACTUAL];
        struct Check { std::string id, what; double value, bar; bool applicable; };
        std::vector<Check> ck;
        ck.push_back({"C1", "bodies add value over the sequence memory (full)", res[A_SILENT].full - A.full, 0.05, true});
        ck.push_back({"C2", "CONTEXT COMPUTATION adds value over the memory (full)", res[A_CONST].full - A.full, 0.02, true});
        for (int l = 0; l < 4; ++l) ck.push_back({"C3." + std::string(lname[l]), std::string(lname[l]) + " does work (body)", res[A_C1 + l].body - A.body, 0.01, true});
        ck.push_back({"C4", "neighbour routing does work (body)", res[A_NOROUTE].body - A.body, 0.01, A.accepts > 0.001});
        ck.push_back({"C5", "neighbour messages inform L3 (body)", res[A_NOMSG].body - A.body, 0.005, A.accepts > 0.001 || model.params().fm_wire});   // motif_wire: every motif body past the first gets one
        const bool rcons = res[A_NOCONS].body != A.body || res[A_NOCONS].full != A.full;
        ck.push_back({"C6", rcons ? "the L4 route consequence does work (body)" : "the L4 return does work (body)", (rcons ? res[A_NOCONS].body : res[A_NORET].body) - A.body, 0.005, A.l4 > 0.001});
        if (rcons) ck.push_back({"C9", "CIRCUITS: senders + receivers beat senders alone (full)", res[A_NOCONS].full - A.full, 0.01, true});
        ck.push_back({"C10", "TEMPORAL cooperation: bodies read what earlier bodies wrote (full)", res[A_NOWS].full - A.full, 0.01, true});
        const double died = admitted ? 1.0 - (double)reached6 / admitted : 0.0;
        ck.push_back({"C7", "early death: admitted bodies dying before L6", died, 0.10, true});
        for (int l = 0; l < NL; ++l) ck.push_back({"C8." + std::string(lname[l]), std::string(lname[l]) + " not pinned: 0.25 - saturated fraction", 0.25 - (double)sat[l] / std::max<uint64_t>(1, outs[l]), 0.0, true});
        o << "\ncheck   result  value    bar     claim\n"; int pass = 0, n = 0;
        for (auto& c : ck) { std::string id = c.id; id.resize(7, ' ');
            const char* r = !c.applicable ? "n/a " : c.value >= c.bar ? "PASS" : "FAIL"; if (c.applicable) { ++n; pass += c.value >= c.bar; }
            o << id << " " << r << "    " << (c.value >= 0 ? " " : "") << c.value << "  " << c.bar << "  " << c.what << (c.applicable ? "" : " (inactive in this model)") << "\n"; }
        if (!task_name.empty()) {   // SKILLS: the gates that matter
            o << "\nskill accuracy (%, every " << stride << "th item)\narm                       ";
            for (auto& t : task_name) { std::string h = t; h.resize(18, ' '); o << h; } o << "\n";
            for (int a = 0; a < NARM; ++a) { std::string nm = arm_name[a]; nm.resize(26, ' '); o << nm; for (size_t t = 0; t < task_name.size(); ++t) { std::ostringstream c; c.setf(std::ios::fixed); c.precision(2); c << res[a].skill[t]; std::string v = c.str(); v.resize(18, ' '); o << v; } o << "\n"; }
            o << "\n";
            for (size_t t = 0; t < task_name.size(); ++t) {
                const double s1 = A.skill[t] - res[A_SILENT].skill[t], s2 = A.skill[t] - res[A_CONST].skill[t];
                ++n; pass += s1 >= 1.0; ++n; pass += s2 >= 1.0;
                o << "S1." << task_name[t] << "  " << (s1 >= 1.0 ? "PASS" : "FAIL") << "  " << s1 << " points: bodies are needed for the skill\n";
                o << "S2." << task_name[t] << "  " << (s2 >= 1.0 ? "PASS" : "FAIL") << "  " << s2 << " points: CONTEXT COMPUTATION in the bodies is needed for the skill\n";
                for (int l = 0; l < 4; ++l) o << "   " << lname[l] << " frozen costs " << A.skill[t] - res[A_C1 + l].skill[t] << " points\n";
            }
        }
        o << "\n" << pass << " of " << n << " applicable checks pass\n";
        std::cout << o.str();
        if (!out_json.empty()) {
            std::ofstream j(out_json); j.setf(std::ios::fixed); j.precision(5);
            j << "{\"state\":\"" << state_file << "\",\"arms\":{"; for (int a = 0; a < NARM; ++a) j << (a ? "," : "") << "\"" << arm_name[a] << "\":{\"full\":" << res[a].full << ",\"body\":" << res[a].body << ",\"macs\":" << res[a].macs << ",\"accepts\":" << res[a].accepts << ",\"l4\":" << res[a].l4 << "}";
            j << "},\"checks\":{"; for (size_t i = 0; i < ck.size(); ++i) j << (i ? "," : "") << "\"" << ck[i].id << "\":{\"value\":" << ck[i].value << ",\"bar\":" << ck[i].bar << ",\"applicable\":" << (ck[i].applicable ? "true" : "false") << ",\"pass\":" << (ck[i].applicable && ck[i].value >= ck[i].bar ? "true" : "false") << "}";
            j << "},\"passed\":" << pass << ",\"applicable\":" << n << "}\n";
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
