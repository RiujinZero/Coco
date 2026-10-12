// The real training run: one mixed stream (prose + dialogue + reasoning episodes), one organism with only its
// general mechanisms (bodies, sequence memory with the consolidated tier, the relation store with commitment
// and selection by consequence, the agency kernel), and PROBES at checkpoints that trace which skills emerge:
//   - held-out BPB on each held-out text
//   - free generation (strict loop metrics) from held-out prose prompts
//   - stay-on-task on held-out dialogues (content-word overlap with the reference reply vs chance)
//   - exact-match QA on held-out bAbI / CLUTRR stories, with nothing but the general mechanisms
//
//   laminar_train file=data/mix_650m.txt heldout=data/wiki_valid_4m.txt chat=data/lmsys_128m.txt.holdout.jsonl
//                 qa=data/babi_all_test.jsonl,data/clutrr_chat_test.jsonl [probe_every=67108864] [units=200000] ...params
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include "laminar/genscore.h"
#include "grounded_bootstrap.h"
#include "grounded_feedback.h"
#include "continual_train_cli.h"
#include <cstdio>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>

using namespace laminar;

static const std::set<std::string> STOP = {"the","and","that","this","with","have","from","your","what","about","they","their","will","would","could","there","which","when","were","been","some","more","also","into","than","then","them","these","those","just","like","only","very","here","over","such","most","many","much","make","made","does","doing","because","should","other","after","before","while","where","being","really","thank","thanks","please","hello","help","today","user","assistant"};
static std::set<std::string> content_words(const std::string& s) { std::set<std::string> out; std::string w; auto flush = [&]() { if (w.size() >= 4 && !STOP.count(w)) out.insert(w); w.clear(); }; for (char c : s) { if (std::isalpha((unsigned char)c)) w += (char)std::tolower((unsigned char)c); else flush(); } flush(); return out; }
static double overlap(const std::set<std::string>& g, const std::set<std::string>& r) { if (r.empty()) return 0.0; size_t c = 0; for (auto& w : r) c += g.count(w); return (double)c / r.size(); }
static std::string json_str(const std::string& s, size_t& i) { std::string o; ++i; for (; i < s.size(); ++i) { char c = s[i]; if (c == '"') { ++i; break; } if (c == '\\' && i + 1 < s.size()) { c = s[++i]; o += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c; } else o += c; } return o; }
static std::vector<std::string> split(const std::string& s, char d) { std::vector<std::string> o; std::string c; for (char x : s) { if (x == d) { if (!c.empty()) o.push_back(c); c.clear(); } else c += x; } if (!c.empty()) o.push_back(c); return o; }

struct Dlg { std::string user, ref; };
struct Q { std::string task, question, answer; std::vector<std::string> story; };

static std::vector<Dlg> load_dialogues(const std::string& path, size_t n) {
    std::vector<Dlg> d; std::ifstream h(path); std::string line;
    while (std::getline(h, line) && d.size() < n) {
        auto a = line.find("\"text\": \""); if (a == std::string::npos) continue; size_t i = a + 8; std::string t = json_str(line, i);
        auto u = t.find("USER:"), as = t.find("ASSISTANT:"); if (u == std::string::npos || as == std::string::npos || as < u) continue;
        std::string user = t.substr(u + 5, as - u - 5), ref = t.substr(as + 10); auto nu = ref.find("USER:"); if (nu != std::string::npos) ref = ref.substr(0, nu);
        if (content_words(user).size() < 2) { auto u2 = t.find("USER:", as); auto a2 = t.find("ASSISTANT:", u2 == std::string::npos ? 0 : u2); if (u2 != std::string::npos && a2 != std::string::npos) { user = t.substr(u2 + 5, a2 - u2 - 5); ref = t.substr(a2 + 10); nu = ref.find("USER:"); if (nu != std::string::npos) ref = ref.substr(0, nu); } }
        if (content_words(user).size() >= 2 && content_words(ref).size() >= 3) d.push_back({user, ref});
    }
    return d;
}
static std::vector<Q> load_qa(const std::string& path, size_t per_task) {
    std::vector<Q> qs; std::map<std::string, size_t> cnt; std::ifstream h(path); std::string line;
    while (std::getline(h, line)) {
        size_t i = line.find("\"task\": \""); if (i == std::string::npos) continue; i += 8; std::string task = json_str(line, i);
        size_t k = line.find("\"qa\": [", i); if (k == std::string::npos) continue; i = k + 7;
        while (i < line.size()) { size_t b = line.find('[', i); if (b == std::string::npos || b + 1 >= line.size()) break; i = b + 1; if (line[i] == ']') break;
            Q q; q.task = task; i = line.find('"', i); q.question = json_str(line, i); i = line.find('"', i); q.answer = json_str(line, i); i = line.find('"', i); json_str(line, i);
            i = line.find('[', i) + 1; while (i < line.size() && line[i] != ']') { if (line[i] == '"') q.story.push_back(json_str(line, i)); else ++i; } i = line.find(']', i + 1) + 1;
            if (cnt[task]++ < per_task) qs.push_back(q); }
    }
    return qs;
}

int main(int argc, char** argv) try {
    // Dispatch before parsing organism-only settings: the canonical profile
    // rejects unknown settings rather than silently ignoring a byte-path knob.
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="trainer=continual")
        return laminar::training::continual_train_cli(argc,argv);
    std::map<std::string, std::string> rest;
    Params p = Params::from_args(argc, argv, &rest);
    if(rest.count("trainer"))throw std::invalid_argument("unknown trainer; use trainer=continual or omit for the byte path");
    p.vocab = 256; if (p.future_buckets == 32) p.future_buckets = 64;
    auto geti = [&](const char* k, uint64_t d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtoull(it->second.c_str(), nullptr, 10); };
    auto getf = [&](const char* k, double d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtod(it->second.c_str(), nullptr); };
    if (!rest.count("file")) { std::fprintf(stderr, "usage: laminar_train file=STREAM [heldout=a,b] [chat=holdout.jsonl] [qa=a.jsonl,b.jsonl] [probe_every=N] [max_bytes=N] [grounded_bootstrap=PATHS] [grounded_source=ID] [grounded_feedback=TSV] [grounded_teacher=ID] [grounded_archive=1] [grounded_state=1] [grounded_model=1] ...\n"); return 1; }
    const uint64_t probe_every = geti("probe_every", 67108864), max_bytes = geti("max_bytes", 0); const double temp = getf("temp", 0.7);
    const size_t ndlg = geti("dialogues", 24), per_task = geti("per_task", 10), nprompts = geti("prompts", 2), gen_len = geti("gen", 1024);
    std::vector<std::string> heldout = rest.count("heldout") ? split(rest["heldout"], ',') : std::vector<std::string>{};
    std::vector<std::vector<uint8_t>> held; for (auto& f : heldout) held.push_back(read_file_bytes(f, 0));
    std::vector<Dlg> dlgs = rest.count("chat") ? load_dialogues(rest["chat"], ndlg) : std::vector<Dlg>{};
    std::vector<std::pair<std::string, std::vector<Q>>> qasets; if (rest.count("qa")) for (auto& f : split(rest["qa"], ',')) qasets.push_back({f, load_qa(f, per_task)});
    std::printf("probes: %zu held-out texts, %zu dialogues, %zu QA sets", held.size(), dlgs.size(), qasets.size()); for (auto& [f, qs] : qasets) std::printf(" [%s: %zu]", f.c_str(), qs.size()); std::printf("\n");

    // the stream is read in 64 MB chunks so a 2 GB run does not need the whole file in memory
    std::ifstream in(rest["file"], std::ios::binary); if (!in) { std::fprintf(stderr, "cannot open stream\n"); return 1; }
    std::set<std::string> vocab; if (!held.empty()) { std::string w; for (uint8_t c : held[0]) { if (std::isalpha(c)) w += (char)std::tolower(c); else { if (w.size() >= 2) vocab.insert(w); w.clear(); } } }

    Organism org(p);
    std::shared_ptr<reasoning::GroundedSession> grounding;
    reasoning::GroundedFeedbackStream teaching;
    if(geti("grounded_archive",0)&&!rest.count("grounded_bootstrap"))throw std::runtime_error("grounded_archive requires grounded_bootstrap");
    if(rest.count("grounded_feedback")) {
        if(!rest.count("grounded_bootstrap"))throw std::runtime_error("grounded_feedback requires grounded_bootstrap");
        teaching.load(rest.at("grounded_feedback"));
    }
    auto attach_grounding=[&](const std::shared_ptr<reasoning::GroundedSession>& session) {
        org.attach_grounded_observer(session->reader,[session](uint8_t c,bool generated){session->observe(c,generated);});
    };
    if(rest.count("grounded_bootstrap")) {
        try {
            grounding=reasoning::GroundedBootstrap::load(rest.at("grounded_bootstrap"),int(p.seed));
            if(geti("grounded_archive",0))grounding->enable_archive();
            if(rest.count("grounded_source"))grounding->report_source=grounding->user_source=rest.at("grounded_source");
            if(rest.count("grounded_feedback"))grounding->teachers.insert(rest.count("grounded_teacher")?rest.at("grounded_teacher"):"verified_teacher");
            attach_grounding(grounding);
            std::printf("GROUNDED loaded reference_frames=%zu correction_frames=%zu; %s\n",grounding->references.frames.size(),grounding->corrections.frames.size(),rest.count("grounded_feedback")?"verified observation feedback enabled; other symbolic models frozen":"bootstrap meanings fixed; no verified teaching");
        } catch(const std::exception& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}
    }
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    Rng gr(p.seed + 5);

    auto generate = [&](const std::string& prompt, size_t len, double T, bool stop_at_turn) {
        // run_block observes outcomes after its leading carry symbol. Include
        // that carry so the first prompt byte (often a framing newline) arrives.
        std::vector<uint8_t> pb(1,org.last_symbol());pb.insert(pb.end(),prompt.begin(),prompt.end());
        org.free_run = false; org.run_block(pb.data(), pb.size()); org.free_run = true;
        std::string out; uint8_t cur = org.last_symbol();
        for (size_t t = 0; t < len; ++t) {
            uint8_t pair[2] = {cur, 0}; org.run_block(pair, 2);
            const auto& pr = org.distribution(); uint8_t next = 0;
            if (T <= 0.0) { for (int v = 1; v < 256; ++v) if (pr[v] > pr[next]) next = (uint8_t)v; }
            else { double z = 0; std::vector<double> w(256); for (int v = 0; v < 256; ++v) { w[v] = std::pow((double)pr[v], 1.0 / T); z += w[v]; } double u = gr.uniform() * z, c = 0; for (int v = 0; v < 256; ++v) { c += w[v]; if (u <= c) { next = (uint8_t)v; break; } } }
            org.commit_outcome(next); out += (char)next; cur = next;
            if (stop_at_turn && out.size() >= 8 && (out.find("\n\nUSER:") != std::string::npos || out.find("<|endofdialogue|>") != std::string::npos || out.find("<|endoftext|>") != std::string::npos)) break;
        }
        org.free_run = false; return out;
    };

    auto probe = [&](uint64_t trained) {
        // Evaluation gets a private reader/journal. Reattaching the training
        // session restores its in-progress episode, pending choices and sources.
        std::shared_ptr<reasoning::GroundedSession> evaluation;
        if(grounding){evaluation=grounding->clone();attach_grounding(evaluation);}
        org.set_mode(false, false); org.stats().reset();
        std::printf("=== PROBE at %.1f MB (%.0f min) ===\n", trained / 1048576.0, elapsed() / 60.0);
        for (size_t h = 0; h < held.size(); ++h) {                    // held-out BPB
            auto& d = held[h]; size_t pos = 0; BlockStats agg;
            while (pos + 1 < d.size()) { size_t n = std::min<size_t>(p.block, d.size() - 1 - pos); org.run_block(d.data() + pos, n + 1); org.end_block(); BlockStats& s = org.stats(); agg.events += s.events; agg.loss_sum += s.loss_sum; s.reset(); pos += n; }
            std::printf("HELDOUT %s: BPB %.4f\n", heldout[h].c_str(), agg.loss_sum / agg.events / 0.6931471805599453);
        }
        if (!held.empty() && nprompts) {                              // free generation from held-out prose
            GenScore tot; int loops = 0; auto& d = held[0];
            for (size_t q = 0; q < nprompts; ++q) {
                size_t start = 4096 + gr.below((uint32_t)(d.size() - 8192)); while (start < d.size() && d[start] != '\n') ++start; ++start;
                std::string prompt(d.begin() + start, d.begin() + start + 128); std::string out = generate(prompt, gen_len, temp, false);
                GenScore s = score_text(out, vocab); tot.word_valid += s.word_valid; tot.unique_words += s.unique_words; tot.repeated8 += s.repeated8; tot.words += s.words; tot.longest_repeat = std::max(tot.longest_repeat, s.longest_repeat); loops += s.looped;
                if (q == 0) { std::string shown = out.substr(0, 200); for (auto& c : shown) { if (c == '\n') c = '|'; else if ((unsigned char)c < 32 || (unsigned char)c > 126) c = '~'; } std::printf("  gen: %s\n", shown.c_str()); }
            }
            std::printf("FREE-RUN (%zu x %zu B, T %.1f): %.0f words | valid %.2f | unique %.2f | rep8 %.3f | longest repeat %d B | LOOPED %d/%zu\n", nprompts, gen_len, temp, (double)tot.words / nprompts, tot.word_valid / nprompts, tot.unique_words / nprompts, tot.repeated8 / nprompts, tot.longest_repeat, loops, nprompts);
        }
        if (!dlgs.empty()) {                                          // stay on task
            double on = 0, ch = 0, wv = 0; size_t nrep = 0;
            for (size_t i = 0; i < dlgs.size(); ++i) {
                const Dlg& d = dlgs[i]; const Dlg& o = dlgs[(i + dlgs.size() / 2) % dlgs.size()];
                std::string g = generate("\n<|endofdialogue|>\nUSER:" + d.user + "\n\nASSISTANT:", 400, temp, true), gx = generate("\n<|endofdialogue|>\nUSER:" + o.user + "\n\nASSISTANT:", 400, temp, true);
                auto cr = content_words(d.ref); on += overlap(content_words(g), cr); ch += overlap(content_words(gx), cr);
                { size_t w = 0, v = 0; std::string x; auto fl = [&]() { if (x.size() >= 2) { ++w; v += vocab.count(x) > 0; } x.clear(); }; for (char c : g) { if (std::isalpha((unsigned char)c)) x += (char)std::tolower((unsigned char)c); else fl(); } fl(); if (w) { wv += (double)v / w; ++nrep; } }
                if (i == 0) { std::string u = d.user.substr(0, 70), r = g.substr(0, 160); for (auto* s : {&u, &r}) for (auto& c : *s) if (c == '\n') c = '|'; std::printf("  chat: USER %s -> %s\n", u.c_str(), r.c_str()); }
            }
            std::printf("STAY-ON-TASK (%zu dialogues): on-topic %.3f (chance %.3f) | words-valid %.2f\n", dlgs.size(), on / dlgs.size(), ch / dlgs.size(), nrep ? wv / nrep : 0.0);
        }
        for (auto& [f, qs] : qasets) {                                // exact-match QA
            std::map<std::string, int> total, correct; std::map<std::string, std::map<std::string, int>> freq; for (auto& q : qs) freq[q.task][q.answer]++;
            int shown = 0;
            for (auto& q : qs) {
                std::string prompt = "\n<|endofdialogue|>\n"; for (auto& s : q.story) prompt += s + "\n"; prompt += "USER: " + q.question + "\n\nASSISTANT:";
                std::string out = generate(prompt, 24, 0.0, false); std::string w; for (char c : out) { if (std::isalpha((unsigned char)c)) w += c; else if (!w.empty()) break; }
                bool ok = w == q.answer; total[q.task]++; correct[q.task] += ok;
                if (shown++ < 2) std::printf("  qa: %s -> \"%s\" (want %s) [%s]\n", q.question.c_str(), w.c_str(), q.answer.c_str(), org.seq_chain_trace().c_str());
            }
            int tc = 0, tt = 0; std::string line; for (auto& [task, n] : total) { int best = 0; for (auto& [a, c] : freq[task]) best = std::max(best, c); tc += correct[task]; tt += n; line += " " + task + " " + std::to_string(correct[task]) + "/" + std::to_string(n) + "(ch " + std::to_string(best) + ")"; }
            std::printf("QA %s: %d/%d = %.1f%% exact |%s\n", f.c_str(), tc, tt, 100.0 * tc / std::max(tt, 1), line.c_str());
        }
        if (p.seqmem) std::printf("%s\n", org.seq_report().c_str());
        if (p.agency) std::printf("%s\n", org.agency_report().c_str());
        if(evaluation)std::printf("GROUNDED probe %s\n",evaluation->report().c_str());
        std::fflush(stdout);
        if(grounding)attach_grounding(grounding);
        org.stats().reset(); org.set_mode(true, true); org.free_run = false;
    };
    if (p.seqmem) org.seq_trace(true);

    const size_t CHUNK = 1 << 26; std::vector<uint8_t> buf(CHUNK + 1); uint64_t trained = 0, next_probe = probe_every; BlockStats agg; uint8_t carry = org.last_symbol(); bool have_carry = bool(grounding);
    while (true) {
        size_t off = 0; if (have_carry) { buf[0] = carry; off = 1; }
        in.read((char*)buf.data() + off, CHUNK - off); size_t got = (size_t)in.gcount() + off; if (got < 2) break;
        if (max_bytes && trained + got - off > max_bytes) got = (size_t)(max_bytes - trained) + off;
        size_t pos = 0;
        while (pos + 1 < got) { size_t n = std::min<size_t>(p.block, got - 1 - pos);
            n=std::min<uint64_t>(n,teaching.next_offset()-trained);
            org.run_block(buf.data() + pos, n + 1); org.end_block(); BlockStats& s = org.stats(); agg.events += s.events; agg.loss_sum += s.loss_sum; agg.seconds += s.seconds; s.reset(); pos += n;
            trained += n;if(grounding)teaching.deliver(trained,*grounding);
            if ((trained & ((1u << 23) - 1)) < n) { std::printf("  ... %.1f MB, %.0f KB/s, %.0f min\n", trained / 1048576.0, agg.events / std::max(agg.seconds, 1e-9) / 1024.0, elapsed() / 60.0); std::fflush(stdout); }   // progress every 8 MB
            if (trained >= next_probe) { std::printf("trained %.1f MB: BPB(window) %.4f, %.0f KB/s, locks %zu units %zu\n", trained / 1048576.0, agg.loss_sum / agg.events / 0.6931471805599453, agg.events / agg.seconds / 1024.0, org.tree.live_locks(), org.live_units()); agg = BlockStats{}; probe(trained); next_probe += probe_every; } }
        carry = buf[got - 1]; have_carry = true;
        if (max_bytes && trained >= max_bytes) break; if ((size_t)in.gcount() + off < CHUNK) break;
    }
    std::printf("trained %.1f MB total in %.0f min\n", trained / 1048576.0, elapsed() / 60.0); if (trained + probe_every != next_probe) probe(trained);   // unless the last checkpoint just ran
    if(grounding)std::printf("GROUNDED stream %s\n",grounding->report().c_str());
    if(grounding&&geti("grounded_state",0))std::printf("GROUNDED state %s\n",reasoning::RevisableGrounding::state_stamp(*grounding->reader).c_str());
    if(grounding&&rest.count("grounded_feedback"))std::printf("GROUNDED teaching accepted=%llu rejected=%llu duplicates=%llu delivered=%zu pending=%zu\n",(unsigned long long)grounding->feedback_accepted,(unsigned long long)grounding->feedback_rejected,(unsigned long long)grounding->feedback_duplicates,teaching.cursor,teaching.rows.size()-teaching.cursor);
    if(grounding&&grounding->archive)std::printf("GROUNDED archive episodes=%zu revision=%llu consistent=%d status=%s\n",grounding->archive->episodes.size(),(unsigned long long)grounding->archive->revision,int(grounding->archive->consistent),grounding->archive->status.c_str());
    if(grounding&&geti("grounded_model",0))std::printf("GROUNDED observation_model BEGIN\n%sGROUNDED observation_model END\n",grounding->reader->observation_language.signature().c_str());
    if(teaching.cursor<teaching.rows.size()&&(!max_bytes||trained<max_bytes))throw std::runtime_error("feedback offset beyond the training stream");
    return 0;
} catch(const std::exception& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}
