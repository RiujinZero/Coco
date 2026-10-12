// The model writes answers (docs/MODEL_TASK_LEARNER.md, "Answer channel").
//
// One persistent model (the same bodies, workspace, sequence memory and agency kernel as the task learner) reads
// curriculum_v2 episodes in their curriculum text form and WRITES the answer after "ASSISTANT:" from its own
// next-byte distribution. An exact-match verifier checks every written answer. Sources are introduced in waves;
// each is taught (the reference answer is read with learning, teacher forcing) during the first half of its
// introduction wave, tapered to zero by the end of that wave, and afterwards must be kept up without a teacher.
//
// Credit without a teacher (all through replay; nothing is learned from a written answer directly):
//   --replay R          after every episode, re-read R stored teacher episodes with learning on (text rehearsal)
//   --attempts K        teacherless episodes are first answered K times, frozen and sampled (temperature 1)
//   --success-replay S  verified correct answers (from attempts or the live answer) are stored as episodes with
//                       the model's own answer, and S of them re-read with learning after every episode
//   --success-budget B  each stored success is re-read at most B times
//   --withhold NAME     a source that is never taught: only the verifier ever speaks about it
//   --replay-scale      rehearsal grows with the curriculum: R x (active sources / 2) teacher episodes per episode
//   --informative       a verified success is stored only when the live (greedy) answer was wrong -- something the
//                       model does not yet produce -- and its replay budget is B x (1 - fraction of attempts correct)
//   --sources a,b,...   restrict to these sources (focused runs); --wave-len N, --waves N override the schedule
// Probes: a fixed task-stratified sample of each source's TEST split, frozen, greedy, exact match; plus held-out
// text bits per byte (--text-eval FILE). Checkpoints every 1,000 episodes; an existing DIR resumes.
//
// usage: model_answer_learner smoke|full DIR [options] [key=value model overrides]
#include "laminar/model_task.h"
#include "laminar/state_io.h"
#include "answer_tasks.h"
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace laminar;
namespace fs = std::filesystem;
namespace an = laminar::answers;
using Clock = std::chrono::steady_clock;

struct AnswerSource { const char* name; const char* train; const char* test; int wave; size_t max_answer; };
static const AnswerSource sources[] = {
    {"babi_s1", "stage1/babi_s1_train.txt", "stage1/babi_s1_test.jsonl", 0, 32},
    {"babi_s2", "stage2/babi_s2_train.txt", "stage2/babi_s2_test.jsonl", 0, 32},
    {"clevr", "stage2/stage2_clevr_train.txt", "stage2/stage2_clevr_test.jsonl", 1, 32},
    {"events", "stage5/stage5_events_train.txt", "stage5/stage5_events_test.jsonl", 1, 32},
    {"paraphrase", "stage3/stage3_paraphrase_train.txt", "stage3/stage3_paraphrase_test.jsonl", 1, 32},
    {"scan", "stage3/scan_simple_train.txt", "stage3/scan_simple_test.jsonl", 2, 320},
    {"cogs", "stage3/cogs_train.txt", "stage3/cogs_test.txt", 2, 320},
    {"propara", "stage5/stage5_propara_train.txt", "stage5/stage5_propara_test.jsonl", 3, 48},
};
static constexpr size_t n_sources = sizeof(sources) / sizeof(sources[0]);

struct Profile {
    std::string name, withhold, text_eval, only;
    bool replay_scale = false, informative = false;
    std::vector<std::string> extra;
    uint32_t units = 16384, replay = 1, attempts = 0, success_replay = 0, success_budget = 8;
    size_t wave_len = 40, waves = 4, eval_every = 40, checkpoint_every = 20, per_task = 2, train_bytes = 1u << 20;
    std::vector<std::string> params() const {
        std::vector<std::string> v = {"vocab=256", "units=" + std::to_string(units), "hops=4", "hops_free=2", "hidden=modulated", "acc_bits=16",
            "agency=1", "agency_task=1", "agency_trace_life=32", "seq_window=1048576", "seq_bits=16", "seq_cbits=16", "seed=1", "agency_seed=7",
            "episodic=1", "hist=7", "agency_ops=hold,attend_episodic,attend_generic,to_memory,to_recent"};
        v.insert(v.end(), extra.begin(), extra.end()); return v;
    }
    std::string declaration() const {
        std::ostringstream o;
        o << "{\"schema\":\"model_answer_learner_v1\",\"profile\":\"" << name << "\",\"model_params\":[";
        for (size_t i = 0; i < params().size(); ++i) o << (i ? "," : "") << '"' << params()[i] << '"';
        o << "],\"sources\":[";
        for (size_t i = 0; i < n_sources; ++i) o << (i ? "," : "") << "{\"name\":\"" << sources[i].name << "\",\"train\":\"" << sources[i].train << "\",\"test\":\"" << sources[i].test << "\",\"wave\":" << sources[i].wave << "}";
        o << "],\"wave_len\":" << wave_len << ",\"waves\":" << waves << ",\"train_bytes_per_source\":" << train_bytes
          << ",\"teacher\":\"reference answer read with learning (teacher forcing); probability 1 in the first half of a source's introduction wave, linear to 0 by its end, 0 afterwards\""
          << ",\"only\":\"" << only << "\",\"replay_scale\":" << replay_scale << ",\"informative\":" << informative
          << ",\"withheld\":\"" << withhold << "\",\"replay\":" << replay << ",\"attempts\":" << attempts << ",\"success_replay\":" << success_replay << ",\"success_budget\":" << success_budget
          << ",\"verifier\":\"exact match after trimming whitespace\",\"probes\":\"" << per_task << " per task from each TEST split, fixed sample, frozen, greedy\""
          << ",\"text_eval\":\"" << text_eval << "\",\"eval_every\":" << eval_every << "}\n";
        return o.str();
    }
};

struct Stored { an::Episode e; std::string written; uint32_t replays = 0; template <class Ar> void state_io(Ar& ar) { ar(e); ar(written); ar(replays); } };
struct RunState {
    uint64_t next = 0; int64_t last_eval = -1;
    std::vector<uint64_t> cursor = std::vector<uint64_t>(n_sources, 0), log_sizes;
    std::vector<Stored> teacher, successes; uint64_t teacher_seen = 0, successes_seen = 0;
    std::string rng;
    double best_correct = -1, best_text = 1e9;   // retention controller (body market)
    uint64_t last_bodies = 0;                     // market population at the previous evaluation (parsimony)
    template <class Ar> void state_io(Ar& ar) { ar(next); ar(last_eval); ar(cursor); ar(log_sizes); ar(teacher); ar(successes); ar(teacher_seen); ar(successes_seen); ar(rng); ar(best_correct); ar(best_text); ar(last_bodies); }
};
static constexpr size_t reservoir_capacity = 4096;
static void reservoir(std::vector<Stored>& v, uint64_t& seen, Stored s, std::mt19937_64& rng) {
    ++seen; if (v.size() < reservoir_capacity) { v.push_back(std::move(s)); return; }
    const uint64_t j = std::uniform_int_distribution<uint64_t>(0, seen - 1)(rng); if (j < reservoir_capacity) v[(size_t)j] = std::move(s);
}
static const std::string END = std::string(an::end_marker) + "\n";

static void read(TaskModel& tm, const std::string& s) { tm.read_text((const uint8_t*)s.data(), s.size()); }
// Re-reads the tail of a prompt frozen (no learning), so an attempt starts from the prompt's state. Sequence memory
// keeps capturing: it predicts by following its own recorded stream, so reading without capture (Organism::replay)
// would leave its predictions stale. Every byte the answer channel reads or writes is therefore captured.
static void reprime(TaskModel& tm, Organism& m, const std::string& prompt) {
    m.set_mode(false, false);
    const std::string tail = prompt.size() > 256 ? prompt.substr(prompt.size() - 256) : prompt; read(tm, tail);
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: model_answer_learner smoke|full DIR [options]");
        Profile prof; prof.name = argv[1]; const fs::path dir = argv[2]; size_t wave_len_override = 0, waves_override = 0;
        if (prof.name == "full") { prof.units = 200000; prof.wave_len = 10000; prof.eval_every = 2500; prof.checkpoint_every = 1000; prof.per_task = 12; prof.train_bytes = 4u << 20; }
        else if (prof.name != "smoke") throw std::runtime_error("unknown profile");
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&] { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return std::string(argv[++i]); };
            if (a == "--replay") prof.replay = std::stoul(next());
            else if (a == "--attempts") prof.attempts = std::stoul(next());
            else if (a == "--success-replay") prof.success_replay = std::stoul(next());
            else if (a == "--success-budget") prof.success_budget = std::stoul(next());
            else if (a == "--withhold") prof.withhold = next();
            else if (a == "--text-eval") prof.text_eval = next();
            else if (a == "--replay-scale") prof.replay_scale = true;
            else if (a == "--informative") prof.informative = true;
            else if (a == "--sources") prof.only = next();
            else if (a == "--wave-len") wave_len_override = std::stoul(next());
            else if (a == "--waves") waves_override = std::stoul(next());
            else if (a.find('=') != std::string::npos) prof.extra.push_back(a);
            else throw std::runtime_error("unknown option " + a);
        }
        if (prof.attempts && !prof.success_replay) throw std::runtime_error("--attempts needs --success-replay");
        if (wave_len_override) { prof.wave_len = wave_len_override; prof.eval_every = std::max<size_t>(1, wave_len_override / 4); }
        if (waves_override) prof.waves = waves_override;
        std::vector<bool> enabled(n_sources, prof.only.empty());
        if (!prof.only.empty()) {
            std::stringstream ss(prof.only); std::string item;
            while (std::getline(ss, item, ',')) { bool found = false; for (size_t s = 0; s < n_sources; ++s) if (item == sources[s].name) { enabled[s] = true; found = true; } if (!found) throw std::runtime_error("unknown source " + item); }
        }
        // waves are renumbered over the enabled sources, so a focused run starts at its first source's wave
        int min_wave = 1 << 30; for (size_t s = 0; s < n_sources; ++s) if (enabled[s]) min_wave = std::min(min_wave, sources[s].wave);
        int withheld = -1; for (size_t s = 0; s < n_sources; ++s) if (prof.withhold == sources[s].name) withheld = (int)s;
        if (!prof.withhold.empty() && withheld < 0) throw std::runtime_error("unknown source to withhold");
        const std::string root = std::string(std::getenv("HOME")) + "/curriculum_v2/";

        // data: training episodes (shuffled once, fixed seed) and fixed probe samples
        std::vector<std::vector<an::Episode>> train(n_sources), probes(n_sources);
        for (size_t s = 0; s < n_sources; ++s) {
            if (!enabled[s]) continue;
            train[s] = an::load_text(root + sources[s].train, (uint8_t)s, prof.train_bytes);
            std::mt19937_64 sh(1000 + s); std::shuffle(train[s].begin(), train[s].end(), sh);
            probes[s] = an::load_probes(root + sources[s].test, (uint8_t)s, prof.per_task, 2000 + s);
            if (train[s].empty() || probes[s].empty()) throw std::runtime_error(std::string("no data for ") + sources[s].name);
        }
        std::vector<uint8_t> eval_text;
        if (!prof.text_eval.empty()) { std::ifstream f(prof.text_eval, std::ios::binary); eval_text.assign(std::istreambuf_iterator<char>(f), {}); if (eval_text.empty()) throw std::runtime_error("empty text-eval"); }

        fs::create_directories(dir);
        const bool resume = fs::exists(dir / "checkpoint.state");
        const std::string declaration = prof.declaration();
        if (!resume) { if (fs::exists(dir / "config.json")) throw std::runtime_error("declared run has no checkpoint: refusing to overwrite"); std::ofstream(dir / "config.json") << declaration; }
        else { std::ifstream f(dir / "config.json"); if (std::string((std::istreambuf_iterator<char>(f)), {}) != declaration) throw std::runtime_error("declared configuration changed"); }

        std::vector<std::string> args = prof.params(); std::vector<char*> av; std::string prog = "model"; av.push_back(prog.data()); for (auto& x : args) av.push_back(x.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        TaskModel tm(model, TaskModelConfig{}, 1);
        RunState rs; std::mt19937_64 rng(77);
        const auto mode = resume ? std::ios::app : std::ios::trunc;
        std::ofstream episodes(dir / "episodes.jsonl", mode), evals(dir / "evaluations.jsonl", mode);
        auto save = [&] {
            episodes.flush(); evals.flush();
            rs.log_sizes = {(uint64_t)fs::file_size(dir / "episodes.jsonl"), (uint64_t)fs::file_size(dir / "evaluations.jsonl")};
            { std::ostringstream o; o << rng; rs.rng = o.str(); }
            const fs::path tmp = dir / "checkpoint.state.tmp", cur = dir / "checkpoint.state", prev = dir / "checkpoint.prev.state";
            { state::Writer w(tmp.string()); model.save_state(w); tm.state_io(w); w.section("run"); rs.state_io(w); w.finish(); }
            if (fs::exists(cur)) fs::rename(cur, prev);
            fs::rename(tmp, cur);
        };
        auto load = [&](const fs::path& f) {
            state::Reader r(f.string()); model.load_state(r); tm.state_io(r); r.section("run"); rs.state_io(r); r.finish();
            std::istringstream i(rs.rng); i >> rng;
        };
        if (resume) {
            try { load(dir / "checkpoint.state"); } catch (const std::exception& e) { std::cout << "checkpoint unreadable (" << e.what() << "), using previous\n"; load(dir / "checkpoint.prev.state"); }
            episodes.close(); evals.close();
            fs::resize_file(dir / "episodes.jsonl", rs.log_sizes.at(0)); fs::resize_file(dir / "evaluations.jsonl", rs.log_sizes.at(1));
            episodes.open(dir / "episodes.jsonl", std::ios::app); evals.open(dir / "evaluations.jsonl", std::ios::app);
            std::cout << "resumed at episode " << rs.next << '\n' << std::flush;
        }
        episodes << std::boolalpha; evals << std::setprecision(6);

        const size_t total = prof.wave_len * prof.waves;
        auto evaluate = [&] {
            save();
            evals << "{\"episode\":" << rs.next << ",\"sources\":{";
            size_t all_ok = 0, all_n = 0;
            bool first_source = true;
            for (size_t s = 0; s < n_sources; ++s) {
                if (!enabled[s]) continue;
                std::vector<std::pair<std::string, std::pair<size_t, size_t>>> by_task; size_t ok = 0;
                for (const auto& p : probes[s]) {
                    model.set_mode(false, false);
                    read(tm, END + p.prompt);
                    const bool good = an::correct(tm.generate(sources[s].max_answer, false), p.answer);
                    ok += good;
                    auto it = std::find_if(by_task.begin(), by_task.end(), [&](auto& t) { return t.first == p.task; });
                    if (it == by_task.end()) { by_task.push_back({p.task, {0, 0}}); it = by_task.end() - 1; }
                    it->second.first += good; ++it->second.second;
                }
                all_ok += ok; all_n += probes[s].size();
                evals << (first_source ? "" : ",") << '"' << sources[s].name << "\":{\"correct\":" << ok << ",\"cases\":" << probes[s].size() << ",\"tasks\":{";
                for (size_t t = 0; t < by_task.size(); ++t) evals << (t ? "," : "") << '"' << by_task[t].first << "\":[" << by_task[t].second.first << ',' << by_task[t].second.second << ']';
                evals << "}}"; first_source = false;
            }
            evals << "},\"correct\":" << all_ok << ",\"cases\":" << all_n;
            double text_bpb = -1;
            { const auto mc = model.market_census(); evals << ",\"market\":{\"bar\":" << model.market_bar() << ",\"bodies\":" << mc.bodies << ",\"snapshots\":" << mc.snapshots << ",\"probation\":" << mc.probation
                << ",\"reuse\":[" << mc.reuse_p10 << ',' << mc.reuse_p50 << ',' << mc.reuse_p90 << "],\"fitness\":[" << mc.fitness_p10 << ',' << mc.fitness_p50 << ',' << mc.fitness_p90
                << "],\"pass\":" << model.market_totals().probation_pass << ",\"fail\":" << model.market_totals().probation_fail << ",\"culled_general\":" << model.market_totals().culled_general
                << ",\"culled_snapshot\":" << model.market_totals().culled_snapshot << ",\"cooldown_skips\":" << model.market_totals().cooldown_skips
                << ",\"raises\":" << model.market_totals().bar_raises << ",\"relaxes\":" << model.market_totals().bar_relaxes << ",\"parsimony\":" << model.market_totals().parsimony_raises << "}"; }
            if (!eval_text.empty()) {
                model.set_mode(false, false); model.stats() = BlockStats{};
                tm.read_text(eval_text.data(), eval_text.size());
                text_bpb = model.stats().loss_sum / std::max<uint64_t>(1, model.stats().events) / 0.6931471805599453;
                evals << ",\"text_bits_per_byte\":" << text_bpb;
                model.stats() = BlockStats{};
            }
            evals << "}\n"; evals.flush();
            std::cout << "evaluation at " << rs.next << ": " << all_ok << '/' << all_n << '\n' << std::flush;
            const uint64_t at = rs.next; load(dir / "checkpoint.state"); rs.last_eval = (int64_t)at;   // live model exactly as before
            if (model.params().market && at > 0) {   // retention: probe accuracy (2% of cases) and held-out text (0.02 bits/byte)
                const bool regressed = (rs.best_correct >= 0 && all_ok + 0.02 * all_n < rs.best_correct) || (text_bpb > 0 && text_bpb > rs.best_text + 0.02);
                const bool improved = all_ok > rs.best_correct + 0.02 * all_n || (text_bpb > 0 && text_bpb < rs.best_text - 0.02);
                const uint64_t bodies = model.market_census().bodies;
                model.market_adjust(regressed, improved, bodies > rs.last_bodies + rs.last_bodies / 20);
                rs.last_bodies = bodies;
            }
            rs.best_correct = std::max(rs.best_correct, (double)all_ok); if (text_bpb > 0) rs.best_text = std::min(rs.best_text, text_bpb);
            model.set_mode(true, true); save();
        };
        auto due = [&] { return rs.last_eval != (int64_t)rs.next && (rs.next == 0 || rs.next == total || rs.next % prof.eval_every == 0); };
        if (!resume) { model.set_mode(true, true); save(); }
        if (due()) evaluate();
        model.set_mode(true, true);

        while (rs.next < total) {
            const size_t index = rs.next, wave = index / prof.wave_len; const double f = double(index % prof.wave_len) / double(prof.wave_len);
            // active sources (introduced up to this wave); new ones drawn twice as often
            std::vector<size_t> active; std::vector<double> w;
            for (size_t s = 0; s < n_sources; ++s) if (enabled[s] && (size_t)(sources[s].wave - min_wave) <= wave) { active.push_back(s); w.push_back((size_t)(sources[s].wave - min_wave) == wave ? 2.0 : 1.0); }
            const size_t s = active[std::discrete_distribution<size_t>(w.begin(), w.end())(rng)];
            const an::Episode& ep = train[s][rs.cursor[s]++ % train[s].size()];
            double teach = 0;
            if ((int)s != withheld && (size_t)(sources[s].wave - min_wave) == wave) teach = f < 0.5 ? 1.0 : std::max(0.0, 1.0 - (f - 0.5) / 0.5);
            const bool teacher = std::uniform_real_distribution<double>(0, 1)(rng) < teach;

            model.set_mode(true, true); read(tm, ep.prompt);   // the story and question are observations
            size_t attempts_ok = 0; bool live_ok = false; std::string live;
            if (teacher) {
                read(tm, " " + ep.answer + "\n" + END);
                reservoir(rs.teacher, rs.teacher_seen, Stored{ep, ep.answer, 0}, rng);
            } else {
                std::string good;
                for (uint32_t k = 0; k < prof.attempts; ++k) {
                    reprime(tm, model, ep.prompt);
                    const std::string a = tm.generate(sources[s].max_answer, true, 1.0f);
                    if (an::correct(a, ep.answer)) { ++attempts_ok; if (good.empty()) good = an::trim(a); }
                }
                if (prof.attempts) reprime(tm, model, ep.prompt);
                live = tm.generate(sources[s].max_answer, false); live_ok = an::correct(live, ep.answer);
                if (live_ok && good.empty()) good = an::trim(live);
                model.set_mode(true, true);
                if (live.empty() || live.back() != '\n') read(tm, "\n");
                read(tm, END);
                tm.tool_outcome(live_ok ? 1.f : 0.f);   // the verdict reaches the agency kernel as a task consequence
                if (prof.success_replay && !good.empty()) {
                    // informative: only what the model does not yet produce greedily, weighted by how rarely it was found
                    const double p_hat = prof.attempts ? double(attempts_ok) / prof.attempts : 1.0;
                    const uint32_t spent = prof.informative ? prof.success_budget - (uint32_t)std::lround(prof.success_budget * (1.0 - p_hat)) : 0;
                    if (!prof.informative || (!live_ok && spent < prof.success_budget)) reservoir(rs.successes, rs.successes_seen, Stored{ep, good, spent}, rng);
                }
            }
            tm.end_mission();
            // rehearsal: stored teacher episodes and verified own successes, re-read with learning
            model.set_mode(true, true);
            const uint32_t replays = prof.replay_scale ? (uint32_t)std::max<size_t>(prof.replay, (prof.replay * active.size() + 1) / 2) : prof.replay;
            for (uint32_t r = 0; r < replays && !rs.teacher.empty(); ++r) {
                const Stored& t = rs.teacher[std::uniform_int_distribution<size_t>(0, rs.teacher.size() - 1)(rng)];
                read(tm, t.e.prompt + " " + t.written + "\n" + END);
            }
            for (uint32_t r = 0; r < prof.success_replay; ++r) {
                std::vector<size_t> open; for (size_t i = 0; i < rs.successes.size(); ++i) if (rs.successes[i].replays < prof.success_budget) open.push_back(i);
                if (open.empty()) break;
                Stored& t = rs.successes[open[std::uniform_int_distribution<size_t>(0, open.size() - 1)(rng)]]; ++t.replays;
                read(tm, t.e.prompt + " " + t.written + "\n" + END);
            }
            tm.end_mission();
            episodes << "{\"episode\":" << index << ",\"wave\":" << wave << ",\"source\":\"" << sources[s].name << "\",\"teacher\":" << teacher
                     << ",\"live_correct\":" << live_ok << ",\"attempts_correct\":" << attempts_ok << ",\"teacher_store\":" << rs.teacher.size()
                     << ",\"success_store\":" << rs.successes.size() << ",\"live_units\":" << model.live_units() << ",\"exact_locks\":" << model.exact_locks()
                     << ",\"episodic\":[" << model.episodic_totals().births << ',' << model.episodic_totals().deferred << ',' << model.episodic_totals().evicted << ',' << model.episodic_totals().evicted_unused << ',' << model.episodic_totals().promoted << ',' << model.episodic_totals().promoted_splits << ',' << model.episodic_totals().retired << "]}\n";
            ++rs.next;
            if (rs.next % prof.checkpoint_every == 0) save();
            if (due()) evaluate();
        }
        std::ofstream(dir / "final.json") << "{\"complete\":true,\"episodes\":" << total << "}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
