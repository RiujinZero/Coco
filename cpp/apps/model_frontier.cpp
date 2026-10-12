// Free generation from a saved BabyLM model: greedy, temperature sampling, and a merged-frontier search decoder (after
// "Interference Search", Bad Theory Labs 2026). Read-only: the model runs frozen (no learning, no structure) and every
// search branch runs in a forked child whose changes the operating system discards, so the decoder cannot change a weight
// and no branch can see another branch's state.
//
// usage: model_frontier DIR STATE_FILE PROMPT|@PROMPTS_FILE [key=value ...]
//   DIR         run directory with config.json ("model_params"), e.g. /home/user/babylm/runs/gpu_motif
//   STATE_FILE  a saved model, e.g. DIR/epoch_9.state
//   PROMPT      text the model reads (after a newline) before it writes; @FILE: one prompt per line (escapes \t \n \\)
//   mode=frontier|greedy|sample|bench, or a comma list: every prompt is decoded by each mode from the same state
//   len=200     bytes to write            T=0.8     temperature (sample mode, and the frontier's proposals)
//   W=4         frontier width            N=4       proposals per live state per level
//   key=3       merge key = the last `key` words of prompt + continuation, lowercased (0: the whole continuation)
//   vote=0.03   score bonus (nats/byte) per extra parent proposing the same key
//   rep=0       score penalty (nats/byte) per repeated word bigram in the continuation
//   copy=0      score penalty (nats) per byte written under a verbatim sequence-memory match >= copylen (default 32)
//               bytes, averaged over the continuation: 0 = the pure likelihood judge; 1 = a copied byte counts ~1 nat less
//   pmi=0       prompt-dependence judge: subtract pmi x the mean log-prob per byte of the same continuation replayed
//               WITHOUT the prompt (from the post-newline state); 1 = pointwise mutual information; 0 = off (as before)
//   mem=1       never re-expand a key already expanded in this generation (0: off)
//   maxword=20  bytes per proposed word   stop=len|sent (sent: also stop at a newline or a second sentence end)
//   minwords=0  stop=sent counts a stop only once the continuation has this many words (else a newline is a boundary)
//   seed=1      workers=8 (concurrent child processes)    out=FILE (one JSON line per prompt and mode)
//   vocab=FILE[,FILE...] training text for the valid-word rate     show=1 (print every output)
//   ident=1     with several prompts: also replay each output after every OTHER prompt (the prompt-identification rank)
//   profile=0   1: write the per-byte log-prob change the prompt makes to each output ("dlp") in its JSON line
//   lean=1      construct without the learning-only buffers (lean=0: the full-size model; same outputs)
//
// frontier (the method). A state is a continuation: the bytes written after the prompt. Each level:
//   1. expand: every live state is replayed in a forked child as the model's own free-running output (exactly the state
//      the model was in when it wrote those bytes); from there N proposals each write one WORD at temperature T (bytes
//      until an alphanumeric run ends, or maxword bytes), each from the same state (forked). A proposal reports its bytes
//      and their summed log-probability under the model's actual distribution (temperature 1).
//   2. merge: proposals whose key is equal are one state; it keeps the best-scoring text and pools the votes (the number
//      of distinct parents that proposed it).
//   3. memory: keys expanded earlier in this generation are never expanded again -- proposals reaching one are dropped,
//      which removes loops and revisits. When every proposal of a level is blocked, the level is relaxed (counted).
//   4. judge: score = mean log-prob per byte of the whole continuation + vote * (votes - 1) - rep * repeated bigrams
//      (- copy * fraction of its bytes written under a long verbatim match, when copy > 0)
//      (- pmi * mean log-prob per byte of the continuation replayed without the prompt, when pmi > 0).
//   5. keep the best W; stop when the best state reaches len bytes (stop=sent: or is finished by a newline / 2nd sentence).
// pmi > 0: the decode process stays in the no-prompt state (after the initial newline) and every expansion child runs two
// replays of its continuation as the model's own output: one after reading the prompt (in a forked grandchild, which
// makes the proposals exactly as above) and one without it (in the child itself, which then scores each proposal's bytes
// from there). Both log-probs accumulate word by word along a state's history, like the log-prob alone before.
// Every output is then scored from the post-prompt state: the model's mean log-prob per byte when it reads the output as
// text ("forced"), and when the output is replayed as its own emissions ("self": the decoding condition), with the
// sequence memory's verbatim-match statistics. lp_self must equal the log-prob accumulated while decoding (a check that
// replaying a continuation reproduces the decoding state exactly). Topic measures: the same replay without the prompt
// (lp0_self; gain = lp_self - lp0_self per byte, also over the second half of the output alone, "late"), which must equal
// the no-prompt log-prob accumulated while decoding when pmi > 0; reach = 1 + the last output byte whose log-prob the
// prompt changes at all; and, with several prompts (ident=1), the replay after every other prompt: rank = 1 + the number
// of other prompts under which the output is at least as probable as under its own (also for the second half alone).
#include "laminar/organism.h"
#include "laminar/state_io.h"
#include "laminar/genscore.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

using namespace laminar;

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

uint64_t splitmix(uint64_t x) { x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; return x ^ (x >> 31); }
uint64_t mix(uint64_t a, uint64_t b, uint64_t c = 0, uint64_t d = 0, uint64_t e = 0) { uint64_t x = splitmix(a); for (uint64_t v : {b, c, d, e}) x = splitmix(x ^ splitmix(v + 0x632BE59BD9B4E019ULL)); return x; }

// ---- processes: a child starts from this process's exact state, builds one message, writes it and exits ----
struct Msg {
    std::string s; size_t o = 0;
    template <class T> Msg& put(const T& v) { s.append(reinterpret_cast<const char*>(&v), sizeof v); return *this; }
    Msg& put_str(const std::string& x) { put((uint32_t)x.size()); s += x; return *this; }
    template <class T> T get() { T v; if (o + sizeof v > s.size()) throw std::runtime_error("truncated message from a child"); std::memcpy(&v, s.data() + o, sizeof v); o += sizeof v; return v; }
    std::string get_str() { const uint32_t n = get<uint32_t>(); if (o + n > s.size()) throw std::runtime_error("truncated message from a child"); std::string x = s.substr(o, n); o += n; return x; }
};
struct Child { pid_t pid = -1; int fd = -1; };
Child spawn(const std::function<Msg()>& f) {
    int fds[2]; if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
    std::cout.flush(); std::cerr.flush(); std::fflush(nullptr);
    const pid_t pid = fork(); if (pid < 0) throw std::runtime_error("fork failed");
    if (pid == 0) {   // no cleanup on exit: the parent's buffers and destructors are not ours to run
        close(fds[0]); int code = 0;
        try {
            const Msg m = f();
            for (size_t off = 0; off < m.s.size();) { const ssize_t w = write(fds[1], m.s.data() + off, m.s.size() - off); if (w < 0 && errno == EINTR) continue; if (w <= 0) { code = 2; break; } off += (size_t)w; }
        } catch (const std::exception& e) { std::fprintf(stderr, "child %d: %s\n", (int)getpid(), e.what()); code = 1; }
        _exit(code);
    }
    close(fds[1]); return {pid, fds[0]};
}
Msg collect(Child c) {
    Msg m; char buf[65536];
    for (;;) { const ssize_t r = read(c.fd, buf, sizeof buf); if (r > 0) { m.s.append(buf, (size_t)r); continue; } if (r < 0 && errno == EINTR) continue; break; }
    close(c.fd); int status = 0; while (waitpid(c.pid, &status, 0) < 0 && errno == EINTR) {}
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) throw std::runtime_error("a child process failed");
    return m;
}
long proc_kb(const char* file, const char* field) {   // a "<field>: N kB" line of /proc/self/<file>
    std::ifstream f(file); std::string line; const size_t n = std::strlen(field);
    while (std::getline(f, line)) if (line.compare(0, n, field) == 0 && line.size() > n && line[n] == ':') return std::stol(line.substr(n + 1));
    return -1;
}

// ---- the model's public interface ----
// Text the model READS: each byte is an observed outcome (teacher forcing), captured by the sequence memory.
void feed_text(Organism& m, const std::string& s) {
    if (s.empty()) return;
    std::vector<uint8_t> b(s.size() + 1); b[0] = m.last_symbol(); std::memcpy(b.data() + 1, s.data(), s.size()); m.run_block(b.data(), b.size());
}
// Text the model WRITES (m.free_run set, as in TaskModel::generate): it processes its last byte against a placeholder
// outcome and returns its distribution over the next byte; the chosen byte is committed with m.commit_outcome, which
// makes it the next input and records it in the sequence memory as the model's own emission (context, never an instance).
const std::vector<float>& next_dist(Organism& m) { uint8_t pair[2] = {m.last_symbol(), 0}; m.run_block(pair, 2); return m.distribution(); }
double lnp(const std::vector<float>& p, int v) { return std::log(std::max((double)p[v], 1e-12)); }
int choose(const std::vector<float>& p, double T, std::mt19937_64& g) {   // greedy (T <= 0) or TaskModel::generate's sampler
    int v = 0;
    if (T <= 0) { for (int k = 1; k < 256; ++k) if (p[k] > p[v]) v = k; return v; }
    double z = 0; std::array<double, 256> q{};
    for (int k = 0; k < 256; ++k) { q[k] = std::pow(std::max((double)p[k], 1e-12), 1.0 / T); z += q[k]; }
    const double u = std::uniform_real_distribution<double>(0.0, z)(g); double acc = 0; v = 255;
    for (int k = 0; k < 256; ++k) { acc += q[k]; if (u < acc) { v = k; break; } }
    return v;
}

// ---- words ----
// A word is a run of alphanumeric bytes (bytes >= 0x80 included, so UTF-8 stays whole; an apostrophe inside a word
// belongs to it), lowercased. Keys, the repetition penalty and every n-gram metric use this one tokenizer.
bool word_byte(unsigned char c, bool inside) { return std::isalnum(c) || c >= 0x80 || (c == '\'' && inside); }
std::vector<std::string> words_of(const std::string& s) {
    std::vector<std::string> w; std::string cur;
    auto flush = [&] { while (!cur.empty() && cur.back() == '\'') cur.pop_back(); if (!cur.empty()) w.push_back(cur); cur.clear(); };
    for (unsigned char c : s) { if (word_byte(c, !cur.empty())) cur += (char)std::tolower(c); else flush(); }
    flush(); return w;
}
std::string join(const std::vector<std::string>& w, size_t from, size_t n) { std::string g; for (size_t j = from; j < from + n; ++j) { g += w[j]; g += ' '; } return g; }
size_t repeated_ngrams(const std::vector<std::string>& w, size_t n) {   // occurrences of an n-gram beyond its first
    std::unordered_set<std::string> seen; size_t r = 0;
    for (size_t i = 0; i + n <= w.size(); ++i) r += !seen.insert(join(w, i, n)).second;
    return r;
}
double distinct_ngrams(const std::vector<std::string>& w, size_t n) {
    if (w.size() < n) return 0.0;
    std::unordered_set<std::string> u; for (size_t i = 0; i + n <= w.size(); ++i) u.insert(join(w, i, n));
    return (double)u.size() / (double)(w.size() - n + 1);
}
int sentence_ends(const std::string& s) {   // . ! ? not followed by an alphanumeric byte
    int n = 0; for (size_t i = 0; i < s.size(); ++i) if ((s[i] == '.' || s[i] == '!' || s[i] == '?') && (i + 1 == s.size() || !std::isalnum((unsigned char)s[i + 1]))) ++n;
    return n;
}

struct Opts {
    size_t len = 200, min_words = 0; double T = 0.8; int W = 4, N = 4, key = 3, maxword = 20, workers = 8, copy_len = 32; double vote = 0.03, rep = 0.0, copy = 0.0, pmi = 0.0;
    bool mem = true, stop_sent = false, show = true, ident = true, profile = false; uint64_t seed = 1;
};
bool finished(const Opts& o, const std::string& cont) {
    return o.stop_sent && (cont.find('\n') != std::string::npos || sentence_ends(cont) >= 2) && (!o.min_words || words_of(cont).size() >= o.min_words);
}

struct DecodeStats { uint64_t model_bytes = 0, proposals = 0, merged = 0, blocked = 0; int levels = 0, relaxed = 0, final_votes = 1; double logp = 0, logp0 = 0; };

// ---- plain decoding: greedy (T = 0) or sampling ----
std::string decode_plain(Organism& m, const Opts& o, double T, std::mt19937_64& g, DecodeStats& ds) {
    std::string out;
    while (out.size() < o.len) {
        const auto& p = next_dist(m); const int v = choose(p, T, g);
        ds.logp += lnp(p, v); m.commit_outcome((uint8_t)v); out += (char)v; ++ds.model_bytes;
        if (finished(o, out)) break;
    }
    return out;
}

// ---- frontier decoding ----
struct Cand { std::string word; double logp = 0, logp0 = 0; uint32_t copied = 0; };   // copied: bytes written under a verbatim match >= copy_len; logp0: without the prompt
Cand sample_word(Organism& m, std::mt19937_64& g, const Opts& o) {
    Cand c; bool inside = false;
    for (int i = 0; i < o.maxword; ++i) {
        const auto& p = next_dist(m); const int v = choose(p, o.T, g);
        c.copied += m.last_match_length() >= o.copy_len;
        c.logp += lnp(p, v); m.commit_outcome((uint8_t)v); c.word += (char)v;
        if (o.stop_sent && v == '\n') break;
        if (word_byte((unsigned char)v, inside)) inside = true; else if (inside) break;
    }
    return c;
}
Msg put_cands(uint64_t bytes, const std::vector<Cand>& cs) { Msg x; x.put(bytes).put((uint32_t)cs.size()); for (const Cand& c : cs) x.put_str(c.word).put(c.logp).put(c.logp0).put(c.copied); return x; }
std::vector<Cand> get_cands(Msg& x, uint64_t& bytes) { bytes += x.get<uint64_t>(); std::vector<Cand> cs(x.get<uint32_t>()); for (Cand& c : cs) { c.word = x.get_str(); c.logp = x.get<double>(); c.logp0 = x.get<double>(); c.copied = x.get<uint32_t>(); } return cs; }
// Replays text as the model's own output (free-running) from the current state; returns its summed log-prob.
double replay(Organism& m, const std::string& text) {
    m.free_run = true; double lp = 0;
    for (unsigned char c : text) { const auto& p = next_dist(m); lp += lnp(p, c); m.commit_outcome(c); }
    return lp;
}

// In a forked child: read the prompt if this process has not (unread_prompt; empty when the decode process read it),
// replay the continuation as the model's own output, then N proposals from that one state (each in a forked grandchild
// but the last, which runs in place).
Msg propose(Organism& m, const std::string& unread_prompt, const std::string& cont, const Opts& o, uint64_t seed) {
    feed_text(m, unread_prompt); m.free_run = true;
    for (unsigned char c : cont) { next_dist(m); m.commit_outcome(c); }
    uint64_t bytes = unread_prompt.size() + cont.size(); std::vector<Cand> out;
    for (int k = 0; k < o.N; ++k) {
        std::mt19937_64 g(mix(seed, (uint64_t)k));
        if (k + 1 < o.N) { Msg r = collect(spawn([&]() { const Cand c = sample_word(m, g, o); return put_cands(c.word.size(), {c}); })); auto cs = get_cands(r, bytes); out.insert(out.end(), cs.begin(), cs.end()); }
        else { out.push_back(sample_word(m, g, o)); bytes += out.back().word.size(); }
    }
    return put_cands(bytes, out);
}
// The expansion of one state. pmi = 0: the decode process has read the prompt; propose from there. pmi > 0: this child is
// in the no-prompt state (after the initial newline): the proposals are made in a forked grandchild that reads the prompt
// first, while this child replays the continuation without the prompt and then scores each distinct proposed word from
// there (a forked great-grandchild each, the last in place), which gives every proposal its no-prompt log-prob.
Msg expand(Organism& m, const std::string& prompt, const std::string& cont, const Opts& o, uint64_t seed) {
    if (o.pmi <= 0) return propose(m, "", cont, o, seed);
    const Child with_prompt = spawn([&]() { return propose(m, prompt, cont, o, seed); });
    replay(m, cont);
    uint64_t bytes = cont.size(); Msg r = collect(with_prompt); std::vector<Cand> cs = get_cands(r, bytes);
    std::vector<std::string> words; for (const Cand& c : cs) if (std::find(words.begin(), words.end(), c.word) == words.end()) words.push_back(c.word);
    std::map<std::string, double> lp0;
    for (size_t k = 0; k < words.size(); ++k) {
        if (k + 1 < words.size()) { Msg x = collect(spawn([&]() { Msg y; y.put(replay(m, words[k])); return y; })); lp0[words[k]] = x.get<double>(); }
        else lp0[words[k]] = replay(m, words[k]);
        bytes += words[k].size();
    }
    for (Cand& c : cs) c.logp0 = lp0[c.word];
    return put_cands(bytes, cs);
}

struct State { std::string cont, key; double logp = 0, logp0 = 0, base = 0, score = 0; uint64_t copied = 0; int votes = 1, proposals = 1; bool done = false; };

std::string decode_frontier(Organism& m, const std::string& prompt, const Opts& o, uint64_t seed, DecodeStats& ds) {
    auto key_for = [&](const std::string& cont) { if (o.key <= 0) return "\x01" + cont; const auto w = words_of(prompt + cont); const size_t k = (size_t)o.key; return join(w, w.size() > k ? w.size() - k : 0, std::min(k, w.size())); };
    auto base_score = [&](const State& s) {
        const double n = (double)std::max<size_t>(1, s.cont.size());
        const double lp = o.pmi > 0 ? s.logp - o.pmi * s.logp0 : s.logp;   // pmi = 1: log p(cont | prompt) - log p(cont | no prompt)
        return lp / n - (o.rep > 0 ? o.rep * (double)repeated_ngrams(words_of(s.cont), 2) : 0.0) - o.copy * (double)s.copied / n; };
    std::vector<State> live(1); live[0].key = key_for("");
    std::unordered_set<std::string> expanded;
    const size_t conc = (size_t)std::max(1, o.workers / 2);   // each expanding child runs one grandchild at a time
    for (int level = 0; level < (int)(2 * o.len + 8); ++level) {
        ds.levels = level + 1;
        // 1. expand every live, unfinished state in its own child
        std::vector<size_t> todo; for (size_t i = 0; i < live.size(); ++i) if (!live[i].done) todo.push_back(i);
        std::vector<std::vector<Cand>> props(live.size());
        std::vector<std::pair<size_t, Child>> running; size_t next = 0;
        while (next < todo.size() || !running.empty()) {
            while (next < todo.size() && running.size() < conc) {
                const size_t i = todo[next++]; const uint64_t s = mix(seed, (uint64_t)level, (uint64_t)i);
                running.push_back({i, spawn([&, i, s]() { return expand(m, prompt, live[i].cont, o, s); })});
            }
            Msg r = collect(running.front().second); props[running.front().first] = get_cands(r, ds.model_bytes); running.erase(running.begin());
        }
        for (size_t i : todo) expanded.insert(live[i].key);
        // 2-4. merge by key, drop keys already expanded, judge
        std::unordered_map<std::string, State> pool, blocked; std::unordered_map<std::string, std::set<size_t>> parents, bparents;
        auto add = [&](std::unordered_map<std::string, State>& P, std::unordered_map<std::string, std::set<size_t>>& par, State s, size_t parent) {
            auto it = P.find(s.key); par[s.key].insert(parent);
            if (it == P.end()) { P.emplace(s.key, std::move(s)); return; }
            const int pr = it->second.proposals + s.proposals;
            if (s.base > it->second.base) it->second = std::move(s);
            it->second.proposals = pr;
        };
        for (size_t i : todo) for (const Cand& c : props[i]) {
            State s; s.cont = live[i].cont + c.word; s.logp = live[i].logp + c.logp; s.logp0 = live[i].logp0 + c.logp0; s.copied = live[i].copied + c.copied; s.key = key_for(s.cont); s.base = base_score(s); s.done = finished(o, s.cont);
            ++ds.proposals;
            if (o.mem && expanded.count(s.key)) { ++ds.blocked; add(blocked, bparents, std::move(s), i); }
            else add(pool, parents, std::move(s), i);
        }
        for (size_t i = 0; i < live.size(); ++i) if (live[i].done) { State s = live[i]; s.proposals = 0; add(pool, parents, std::move(s), SIZE_MAX - i); }   // finished states wait
        if (pool.empty()) { ++ds.relaxed; pool.swap(blocked); parents.swap(bparents); }
        if (pool.empty()) break;
        std::vector<State> next_live; next_live.reserve(pool.size());
        for (auto& [k, s] : pool) {
            s.votes = std::max<int>(1, (int)parents[k].size()); ds.merged += parents[k].size() > 1 ? parents[k].size() - 1 : 0;
            s.score = s.base + o.vote * (s.votes - 1);
            next_live.push_back(std::move(s));
        }
        std::sort(next_live.begin(), next_live.end(), [](const State& a, const State& b) { return a.score != b.score ? a.score > b.score : a.cont < b.cont; });
        if (next_live.size() > (size_t)o.W) next_live.resize((size_t)o.W);
        live.swap(next_live);
        // 5. stop when the best state is long enough or finished
        if (live[0].cont.size() >= o.len || live[0].done) break;
    }
    ds.logp = live[0].logp; ds.logp0 = live[0].logp0; ds.final_votes = live[0].votes;
    return live[0].cont;
}

// ---- scoring an output from the post-prompt state ----
void put_vec(Msg& x, const std::vector<double>& v) { x.put((uint32_t)v.size()); for (double d : v) x.put(d); }
std::vector<double> get_vec(Msg& x) { std::vector<double> v(x.get<uint32_t>()); for (double& d : v) d = x.get<double>(); return v; }
// forced and self log-prob, verbatim-match statistics, the self log-prob of the bytes from `half` on, and per byte
Msg score_output(Organism& m, const std::string& out, size_t half) {
    Child forced = spawn([&]() { m.stats() = BlockStats{}; feed_text(m, out); Msg x; x.put(-m.stats().loss_sum); return x; });
    m.free_run = true; double lp = 0, late = 0; uint64_t match = 0, c32 = 0; std::vector<double> per(out.size());
    for (size_t t = 0; t < out.size(); ++t) {
        const unsigned char c = (unsigned char)out[t]; const auto& p = next_dist(m); const double l = lnp(p, c); lp += l; if (t >= half) late += l; per[t] = l;
        const int L = m.last_match_length(); match += (uint64_t)L; c32 += L >= 32; m.commit_outcome(c); }
    Msg f = collect(forced); const double lf = f.get<double>();
    Msg x; x.put(lf).put(lp).put(match).put(c32).put(late); put_vec(x, per); return x;
}
// the self log-prob of an output after reading ctx instead of its prompt (ctx empty: no prompt, straight after the initial
// newline): whole, of the bytes from `half` on, and per byte
Msg replay_after(Organism& m, const std::string& ctx, const std::string& out, size_t half) {
    feed_text(m, ctx); m.free_run = true; double lp = 0, late = 0; std::vector<double> per(out.size());
    for (size_t t = 0; t < out.size(); ++t) { const unsigned char c = (unsigned char)out[t]; const auto& p = next_dist(m); const double l = lnp(p, c); lp += l; if (t >= half) late += l; per[t] = l; m.commit_outcome(c); }
    Msg x; x.put(lp).put(late); put_vec(x, per); return x;
}
// f(k) for k < n, each in a forked child of this process, at most conc at a time; the messages in order of k
std::vector<Msg> fork_map(size_t n, size_t conc, const std::function<Msg(size_t)>& f) {
    std::vector<Msg> res(n); std::vector<std::pair<size_t, Child>> running; size_t next = 0;
    while (next < n || !running.empty()) {
        while (next < n && running.size() < std::max<size_t>(1, conc)) { const size_t k = next++; running.push_back({k, spawn([&f, k]() { return f(k); })}); }
        res[running.front().first] = collect(running.front().second); running.erase(running.begin());
    }
    return res;
}

std::string unescape(const std::string& s) {
    std::string o; for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) { const char c = s[++i]; o += c == 't' ? '\t' : c == 'n' ? '\n' : c; } else o += s[i]; }
    return o;
}
std::string json_str(const std::string& s) {   // bytes >= 0x80 as \u00XX: lossless for a byte-level model's output
    std::string o = "\""; char b[8];
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c == '\n') o += "\\n"; else if (c == '\t') o += "\\t";
        else if (c < 0x20 || c >= 0x7F) { std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}
std::string shown(const std::string& s) { std::string t = s; for (char& c : t) { if (c == '\n') c = '|'; else if (c == '\t') c = ' '; else if ((unsigned char)c < 32 || (unsigned char)c > 126) c = '~'; } return t; }

struct Row { std::string mode; double loop4 = 0, strict = 0, d2 = 0, d3 = 0, lpf = 0, lps = 0, valid = 0, copy32 = 0, sec = 0, mbytes = 0, bytes = 0, relaxed = 0, gain = 0, gain_late = 0, id1 = 0, id1_late = 0, reach = 0; int n = 0, nid = 0; };

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: model_frontier DIR STATE_FILE PROMPT|@PROMPTS_FILE [mode=frontier|greedy|sample|bench,...] [len=200] [W=4] [N=4] [T=0.8] [key=3] [vote=0.03] [rep=0] [copy=0] [pmi=0] [mem=1] [maxword=20] [stop=len|sent] [seed=1] [workers=8] [out=FILE] [vocab=FILE,...] [show=1] [ident=1]");
        const std::string dir = argv[1], state_file = argv[2], prompt_arg = argv[3];
        std::map<std::string, std::string> kv;
        for (int i = 4; i < argc; ++i) { const std::string a = argv[i]; const size_t eq = a.find('='); if (eq == std::string::npos) throw std::runtime_error("expected key=value: " + a); kv[a.substr(0, eq)] = a.substr(eq + 1); }
        auto has = [&](const char* k) { return kv.count(k) > 0; };
        Opts o;
        if (has("len")) o.len = std::stoul(kv["len"]); if (has("T")) o.T = std::stod(kv["T"]); if (has("W")) o.W = std::stoi(kv["W"]); if (has("N")) o.N = std::stoi(kv["N"]);
        if (has("key")) o.key = std::stoi(kv["key"]); if (has("vote")) o.vote = std::stod(kv["vote"]); if (has("rep")) o.rep = std::stod(kv["rep"]); if (has("mem")) o.mem = kv["mem"] != "0";
        if (has("copy")) o.copy = std::stod(kv["copy"]); if (has("copylen")) o.copy_len = std::stoi(kv["copylen"]); if (has("minwords")) o.min_words = std::stoul(kv["minwords"]);
        if (has("maxword")) o.maxword = std::stoi(kv["maxword"]); if (has("workers")) o.workers = std::stoi(kv["workers"]); if (has("seed")) o.seed = std::stoull(kv["seed"]);
        if (has("show")) o.show = kv["show"] != "0"; if (has("ident")) o.ident = kv["ident"] != "0"; if (has("pmi")) o.pmi = std::stod(kv["pmi"]);
        if (has("profile")) o.profile = kv["profile"] != "0";
        if (o.pmi < 0) throw std::runtime_error("pmi >= 0");
        if (has("stop")) { if (kv["stop"] == "sent") o.stop_sent = true; else if (kv["stop"] != "len") throw std::runtime_error("stop=len|sent"); }
        if (o.W < 1 || o.N < 1 || o.maxword < 1 || o.workers < 2) throw std::runtime_error("W, N, maxword >= 1 and workers >= 2");
        std::vector<std::string> modes; { std::string ms = has("mode") ? kv["mode"] : "frontier"; size_t s = 0; while (s <= ms.size()) { size_t c = ms.find(',', s); if (c == std::string::npos) c = ms.size(); modes.push_back(ms.substr(s, c - s)); s = c + 1; } }
        for (const auto& md : modes) if (md != "frontier" && md != "greedy" && md != "sample" && md != "bench") throw std::runtime_error("unknown mode " + md);
        std::vector<std::string> prompts;
        if (!prompt_arg.empty() && prompt_arg[0] == '@') { std::ifstream pf(prompt_arg.substr(1)); if (!pf) throw std::runtime_error("cannot read " + prompt_arg.substr(1)); std::string line; while (std::getline(pf, line)) if (!line.empty()) prompts.push_back(unescape(line)); }
        else prompts.push_back(unescape(prompt_arg));

        // the model, exactly as model_babylm_score loads it; frozen; then a newline so the prompt starts a line
        const auto t_load = Clock::now();
        std::ifstream cf(dir + "/config.json"); std::string config((std::istreambuf_iterator<char>(cf)), {});
        std::smatch mm;
        if (!std::regex_search(config, mm, std::regex("\"model_params\":\\[([^\\]]*)\\]"))) throw std::runtime_error("no model_params in " + dir + "/config.json");
        std::vector<std::string> args{"frontier"};
        { std::string list = mm[1]; std::regex item("\"([^\"]*)\""); for (auto it = std::sregex_iterator(list.begin(), list.end(), item); it != std::sregex_iterator(); ++it) args.push_back((*it)[1]); }
        // lean=1 (default): construct with train=0, so the buffers that only learning uses (adam's four floats per synapse
        // and the plasticity residual: 7.8 GiB at 82k bodies) are never allocated; load_state does not read them and
        // restores p.train itself. The frozen forward pass is unchanged (lean=0 gives bit-identical outputs); a fork of the
        // model gets 20x cheaper (8.1 -> 0.35 GiB resident).
        if (!has("lean") || kv["lean"] != "0") args.push_back("train=0");
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        { state::Reader r(state_file); model.load_state(r); }
        model.set_mode(false, false);
        feed_text(model, "\n");
        std::fprintf(stderr, "loaded %s in %.1f s, resident %.2f GB, %zu prompts, modes %s\n", state_file.c_str(), secs(t_load), proc_kb("/proc/self/status", "VmRSS") / 1048576.0, prompts.size(), has("mode") ? kv["mode"].c_str() : "frontier");

        if (modes.size() == 1 && modes[0] == "bench") {   // costs that size the search: fork, a free-running byte, a child's private memory
            const auto t0 = Clock::now(); for (int i = 0; i < 20; ++i) collect(spawn([]() { return Msg(); }));
            const double fork_ms = secs(t0) * 1000.0 / 20;
            Msg r = collect(spawn([&]() {
                feed_text(model, prompts[0]); model.free_run = true; std::mt19937_64 g(1); DecodeStats ds; Opts oo = o; oo.len = 1000;
                const auto t1 = Clock::now(); decode_plain(model, oo, 0.0, g, ds); const double us = secs(t1) * 1e6 / 1000.0;
                Msg x; x.put(us).put(proc_kb("/proc/self/smaps_rollup", "Private_Dirty")); return x; }));
            const double us = r.get<double>(); const long dirty = r.get<long>();
            std::printf("bench: fork+exit+wait %.1f ms, free-running byte %.1f us, child private dirty after prompt + 1000 bytes %.1f MB, parent resident %.2f GB\n",
                        fork_ms, us, dirty / 1024.0, proc_kb("/proc/self/status", "VmRSS") / 1048576.0);
            return 0;
        }

        std::set<std::string> vocab;   // words (>= 2 letters, lowercased) of the training text, for the valid-word rate
        if (has("vocab")) { const std::string vs = kv["vocab"]; size_t s = 0;
            while (s < vs.size()) { size_t c = vs.find(',', s); if (c == std::string::npos) c = vs.size();
                std::ifstream vf(vs.substr(s, c - s)); if (!vf) throw std::runtime_error("cannot read vocab file " + vs.substr(s, c - s));
                std::string w; char ch; while (vf.get(ch)) { if (std::isalpha((unsigned char)ch)) w += (char)std::tolower((unsigned char)ch); else { if (w.size() >= 2) vocab.insert(w); w.clear(); } }
                if (w.size() >= 2) vocab.insert(w); s = c + 1; }
            std::fprintf(stderr, "vocabulary: %zu words\n", vocab.size()); }

        std::ofstream out; if (has("out")) { out.open(kv["out"]); if (!out) throw std::runtime_error("cannot write " + kv["out"]); }
        std::vector<Row> rows(modes.size()); for (size_t k = 0; k < modes.size(); ++k) rows[k].mode = modes[k];
        for (size_t pi = 0; pi < prompts.size(); ++pi) {
            const std::string& prompt = prompts[pi];
            for (size_t mi = 0; mi < modes.size(); ++mi) {
                const std::string& md = modes[mi];
                // decode in a child that first reads the prompt (the root keeps the frozen post-newline state); the pmi judge
                // also needs the state before the prompt, so there the decode process stays in it and each expansion reads the prompt
                const bool pmi = md == "frontier" && o.pmi > 0;
                Msg d = collect(spawn([&]() {
                    if (!pmi) { feed_text(model, prompt); model.free_run = true; }
                    DecodeStats ds; const auto t0 = Clock::now(); std::string text;
                    std::mt19937_64 g(mix(o.seed, pi, 1));
                    if (md == "greedy") text = decode_plain(model, o, 0.0, g, ds);
                    else if (md == "sample") text = decode_plain(model, o, o.T, g, ds);
                    else text = decode_frontier(model, prompt, o, mix(o.seed, pi, 2), ds);
                    const double sec = secs(t0);
                    Msg x; x.put_str(text).put(sec).put(ds.logp).put(ds.logp0).put(ds.model_bytes).put(ds.proposals).put(ds.merged).put(ds.blocked).put(ds.levels).put(ds.relaxed).put(ds.final_votes); return x; }));
                const std::string text = d.get_str(); const double sec = d.get<double>(), lp_gen = d.get<double>(), lp_gen0 = d.get<double>();
                const uint64_t mbytes = d.get<uint64_t>(), props = d.get<uint64_t>(), merged = d.get<uint64_t>(), blocked = d.get<uint64_t>();
                const int levels = d.get<int>(), relaxed = d.get<int>(), votes = d.get<int>();
                // score it from the same post-prompt state; replay it without the prompt and (ident) after every other prompt
                const size_t half = text.size() / 2;
                std::vector<std::string> ctxs{""};
                if (o.ident && prompts.size() > 1) for (size_t j = 0; j < prompts.size(); ++j) if (j != pi) ctxs.push_back(prompts[j]);
                std::vector<Msg> sc = fork_map(ctxs.size() + 1, (size_t)o.workers, [&](size_t k) {
                    if (k == 0) { feed_text(model, prompt); return score_output(model, text, half); }
                    return replay_after(model, ctxs[k - 1], text, half); });
                Msg& s = sc[0];
                const double lp_forced = s.get<double>(), lp_self = s.get<double>(); const uint64_t match = s.get<uint64_t>(), copy32 = s.get<uint64_t>(); const double lp_late = s.get<double>();
                const double lp0 = sc[1].get<double>(), lp0_late = sc[1].get<double>();
                // among all prompts: 1 + the other prompts after which the output is at least as probable (a tie counts against its
                // own prompt: once the prompt no longer changes any prediction, every prompt ties and the rank is the last)
                int rank = 1, rank_late = 1;
                for (size_t k = 2; k < sc.size(); ++k) { const double a = sc[k].get<double>(), b = sc[k].get<double>(); rank += a >= lp_self; rank_late += b >= lp_late; }
                // the prompt's reach: 1 + the last output byte whose log-prob the prompt changes at all (reach) / by > 0.01 nats (reach01)
                size_t reach = 0, reach01 = 0; std::string dlp;   // profile=1: the per-byte change, in the JSON line
                { const std::vector<double> own = get_vec(sc[0]), none = get_vec(sc[1]); char b[32];
                  for (size_t t = 0; t < own.size() && t < none.size(); ++t) {
                      if (own[t] != none[t]) reach = t + 1; if (std::fabs(own[t] - none[t]) > 0.01) reach01 = t + 1;
                      if (o.profile) { std::snprintf(b, sizeof b, "%s%.5f", t ? "," : "", own[t] - none[t]); dlp += b; } } }
                if (std::fabs(lp_self - lp_gen) > 1e-6 * std::max(1.0, std::fabs(lp_gen)))
                    std::fprintf(stderr, "WARNING: replay log-prob %.9f differs from decoding log-prob %.9f (prompt %zu, %s)\n", lp_self, lp_gen, pi, md.c_str());
                if (pmi && std::fabs(lp0 - lp_gen0) > 1e-6 * std::max(1.0, std::fabs(lp_gen0)))
                    std::fprintf(stderr, "WARNING: no-prompt replay log-prob %.9f differs from the no-prompt log-prob while decoding %.9f (prompt %zu, %s)\n", lp0, lp_gen0, pi, md.c_str());
                const auto w = words_of(text); const double nb = std::max<size_t>(1, text.size()), nlate = std::max<size_t>(1, text.size() - half);
                const bool loop4 = repeated_ngrams(w, 4) > 0; const double d2 = distinct_ngrams(w, 2), d3 = distinct_ngrams(w, 3);
                const double gain = (lp_self - lp0) / nb, gain_late = (lp_late - lp0_late) / nlate;
                const GenScore gs = score_text(text, vocab);
                Row& r = rows[mi]; ++r.n; r.loop4 += loop4; r.strict += gs.looped; r.d2 += d2; r.d3 += d3; r.lpf += lp_forced / nb; r.lps += lp_self / nb;
                r.valid += gs.word_valid; r.copy32 += copy32 / nb; r.sec += sec; r.mbytes += (double)mbytes; r.bytes += (double)text.size(); r.relaxed += relaxed;
                r.gain += gain; r.gain_late += gain_late; r.reach += (double)reach; if (ctxs.size() > 1) { ++r.nid; r.id1 += rank == 1; r.id1_late += rank_late == 1; }
                if (out) { out << "{\"prompt\":" << json_str(prompt) << ",\"mode\":\"" << md << "\",\"len\":" << o.len << ",\"T\":" << o.T << ",\"W\":" << o.W << ",\"N\":" << o.N << ",\"key\":" << o.key
                             << ",\"vote\":" << o.vote << ",\"rep\":" << o.rep << ",\"copy\":" << o.copy << ",\"mem\":" << o.mem << ",\"stop\":\"" << (o.stop_sent ? "sent" : "len") << "\",\"minwords\":" << o.min_words << ",\"seed\":" << o.seed << ",\"output\":" << json_str(text)
                             << ",\"bytes\":" << text.size() << ",\"words\":" << w.size() << ",\"loop4\":" << loop4 << ",\"distinct2\":" << d2 << ",\"distinct3\":" << d3
                             << ",\"strict_loop\":" << gs.looped << ",\"valid_words\":" << gs.word_valid << ",\"lp_forced\":" << lp_forced / nb << ",\"lp_self\":" << lp_self / nb
                             << ",\"lp_gen\":" << lp_gen / nb << ",\"mean_match\":" << match / nb << ",\"copy32\":" << copy32 / nb << ",\"seconds\":" << sec << ",\"model_bytes\":" << mbytes
                             << ",\"levels\":" << levels << ",\"proposals\":" << props << ",\"merged\":" << merged << ",\"blocked\":" << blocked << ",\"relaxed\":" << relaxed << ",\"final_votes\":" << votes
                             << ",\"pmi\":" << o.pmi << ",\"lp0_self\":" << lp0 / nb << ",\"gain\":" << gain << ",\"gain_late\":" << gain_late << ",\"reach\":" << reach << ",\"reach01\":" << reach01;
                    if (pmi) out << ",\"lp_gen0\":" << lp_gen0 / nb;
                    if (ctxs.size() > 1) out << ",\"rank\":" << rank << ",\"rank_late\":" << rank_late << ",\"prompts\":" << prompts.size();
                    if (o.profile) out << ",\"dlp\":[" << dlp << "]";
                    out << "}\n" << std::flush; }
                if (o.show) std::printf("[%zu %s] %.1fs loop4 %d d2 %.2f d3 %.2f lp %.3f/%.3f valid %.2f gain %.3f/%.3f reach %zu/%zu rank %d/%d :: %s>>%s\n", pi, md.c_str(), sec, (int)loop4, d2, d3, lp_forced / nb, lp_self / nb, gs.word_valid, gain, gain_late, reach, reach01, rank, rank_late, shown(prompt).c_str(), shown(text).c_str());
                std::fflush(stdout);
            }
        }
        std::printf("\n%-9s %4s %5s %6s %6s %6s %6s %8s %8s %6s %6s %6s %6s %6s %5s %5s %8s %10s %6s\n", "mode", "n", "bytes", "loop4", "strict", "dist2", "dist3", "lp_forcd", "lp_self", "valid", "copy32", "gain", "g_late", "reach", "id1", "id1_l", "sec/smp", "events/smp", "relax");
        for (const Row& r : rows) { const double n = std::max(1, r.n);
            std::printf("%-9s %4d %5.0f %6.2f %6.2f %6.3f %6.3f %8.3f %8.3f %6.3f %6.3f %6.3f %6.3f %6.1f %5.2f %5.2f %8.2f %10.0f %6.2f\n", r.mode.c_str(), r.n, r.bytes / n, r.loop4 / n, r.strict / n, r.d2 / n, r.d3 / n, r.lpf / n, r.lps / n, r.valid / n, r.copy32 / n,
                        r.gain / n, r.gain_late / n, r.reach / n, r.nid ? r.id1 / r.nid : NAN, r.nid ? r.id1_late / r.nid : NAN, r.sec / n, r.mbytes / n, r.relaxed / n); }
        std::printf("(lp: mean natural log-prob per output byte, forced = read as text, self = replayed as the model's own output; copy32: bytes predicted under a verbatim match >= 32; "
                    "gain: lp_self minus the same replay without the prompt, per byte (g_late: over the output's second half); reach: 1 + the last output byte whose log-prob the prompt changes; "
                    "id1: the output is more probable after its own prompt than after any other prompt (id1_l: its second half); events/smp: model byte events per sample)\n");
        return 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
