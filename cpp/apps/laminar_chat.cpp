// Stay-on-task test (goal-directed generation, phase 1): train on a chat stream, then for held-out dialogues feed
// "USER: <turn>\n\nASSISTANT:" and free-run the reply. Scores:
//   on-topic   content words of the reply that also occur in the user turn or the reference reply (recall of the
//              reference's content words, and overlap with the prompt's), against a CHANCE level from replies
//              generated for a different, shuffled prompt
//   drift      on-topic score of the second half of the reply minus the first half
//   loops / valid words as in laminar_generate
//
//   laminar_chat file=data/smoltalk.txt holdout=data/smoltalk.txt.holdout.jsonl [dialogues=64] [reply=512] [temp=0.7] ...
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include <cstdio>
#include <string>
#include <set>
#include <map>
#include <fstream>
#include <algorithm>

using namespace laminar;

static const std::set<std::string> STOP = {"the","and","that","this","with","have","from","your","what","about","they","their","will","would","could","there","which","when","were","been","some","more","also","into","than","then","them","these","those","just","like","only","very","here","over","such","most","many","much","make","made","does","doing","because","should","other","after","before","while","where","being","really","thank","thanks","please","hello","help","today","user","assistant"};

static std::set<std::string> content_words(const std::string& s) {
    std::set<std::string> out; std::string w;
    auto flush = [&]() { if (w.size() >= 4 && !STOP.count(w)) out.insert(w); w.clear(); };
    for (char c : s) { if (std::isalpha((unsigned char)c)) w += (char)std::tolower((unsigned char)c); else flush(); } flush();
    return out;
}
static double overlap(const std::set<std::string>& gen, const std::set<std::string>& ref) {   // fraction of ref covered
    if (ref.empty()) return 0.0; size_t c = 0; for (auto& w : ref) c += gen.count(w); return (double)c / ref.size();
}
static std::string unescape_json(const std::string& s) {   // minimal: \n \" \\ \t
    std::string o; for (size_t i = 0; i < s.size(); ++i) { if (s[i] == '\\' && i + 1 < s.size()) { char c = s[++i]; o += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c; } else o += s[i]; } return o;
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> rest;
    Params p = Params::from_args(argc, argv, &rest);
    p.vocab = 256; if (p.future_buckets == 32) p.future_buckets = 64;
    auto geti = [&](const char* k, uint64_t d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtoull(it->second.c_str(), nullptr, 10); };
    auto getf = [&](const char* k, double d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtod(it->second.c_str(), nullptr); };
    if (!rest.count("file") || !rest.count("holdout")) { std::fprintf(stderr, "usage: laminar_chat file=STREAM holdout=HOLDOUT.jsonl [dialogues=64] [reply=512] [temp=0.7] [max_bytes=N] ...\n"); return 1; }
    const uint64_t ndial = geti("dialogues", 64), reply_len = geti("reply", 512); const double temp = getf("temp", 0.7); const bool show = geti("show", 1) != 0;
    std::vector<uint8_t> data = read_file_bytes(rest["file"], geti("max_bytes", 0));
    if (data.size() < 65536) { std::fprintf(stderr, "stream too small\n"); return 1; }
    // held-out dialogues: first user turn and first assistant reply
    // turns=2: also the following exchange; the reply to the SECOND user turn is generated after the first exchange
    // was fed as context, and scored against the second reference (does the goal hold across turns?)
    const int turns = (int)geti("turns", 1);
    struct Dlg { std::string user, ref, user2, ref2; }; std::vector<Dlg> dlgs;
    { std::ifstream h(rest["holdout"]); std::string line;
      while (std::getline(h, line) && dlgs.size() < ndial) {
          auto a = line.find("\"text\": \""); if (a == std::string::npos) continue; a += 9; auto b = line.rfind("\"}"); std::string t = unescape_json(line.substr(a, b - a));
          auto u = t.find("USER:"), as = t.find("ASSISTANT:"); if (u == std::string::npos || as == std::string::npos || as < u) continue;
          std::string user = t.substr(u + 5, as - u - 5); std::string ref = t.substr(as + 10); auto nu = ref.find("USER:"); if (nu != std::string::npos) ref = ref.substr(0, nu);
          // skip a scripted opener ("Hi there" / "Hey!" ...) when a substantive turn follows
          if (content_words(user).size() < 2) { auto u2 = t.find("USER:", as); auto a2 = t.find("ASSISTANT:", u2 == std::string::npos ? 0 : u2); if (u2 != std::string::npos && a2 != std::string::npos) { user = t.substr(u2 + 5, a2 - u2 - 5); ref = t.substr(a2 + 10); nu = ref.find("USER:"); if (nu != std::string::npos) ref = ref.substr(0, nu); } }
          std::string user2, ref2;
          if (turns >= 2) { auto p1 = t.find(ref); auto u2 = p1 == std::string::npos ? p1 : t.find("USER:", p1 + ref.size()); auto a2 = u2 == std::string::npos ? u2 : t.find("ASSISTANT:", u2);
              if (u2 != std::string::npos && a2 != std::string::npos) { user2 = t.substr(u2 + 5, a2 - u2 - 5); ref2 = t.substr(a2 + 10); auto n2 = ref2.find("USER:"); if (n2 != std::string::npos) ref2 = ref2.substr(0, n2); }
              if (content_words(user2).size() < 1 || content_words(ref2).size() < 3) continue; }
          if (content_words(user).size() >= 2 && content_words(ref).size() >= 3) dlgs.push_back({user, ref, user2, ref2});
      } }
    if (dlgs.size() < 4) { std::fprintf(stderr, "not enough held-out dialogues\n"); return 1; }
    std::set<std::string> vocab; { std::string w; for (uint8_t c : data) { if (std::isalpha(c)) w += (char)std::tolower(c); else { if (w.size() >= 2) vocab.insert(w); w.clear(); } } }

    Organism org(p);
    size_t pos = 0; BlockStats agg;
    while (pos + 1 < data.size()) { size_t n = std::min<size_t>(p.block, data.size() - 1 - pos); org.run_block(data.data() + pos, n + 1); org.end_block(); BlockStats& s = org.stats(); agg.events += s.events; agg.loss_sum += s.loss_sum; agg.seconds += s.seconds; s.reset(); pos += n; }
    std::printf("trained: %zu bytes, BPB %.4f, %.0f KB/s, locks %zu\n", data.size(), agg.loss_sum / agg.events / 0.6931471805599453, agg.events / agg.seconds / 1024.0, org.tree.live_locks());
    org.set_mode(false, false);

    Rng gr(p.seed + 11);
    auto generate = [&](const Dlg& d) {
        std::string prompt = "\n<|endofdialogue|>\nUSER:" + d.user + "\n\nASSISTANT:";
        if (turns >= 2) prompt += d.ref + "\n\nUSER:" + d.user2 + "\n\nASSISTANT:";   // the first exchange is context; the goal must hold
        std::vector<uint8_t> pb(prompt.begin(), prompt.end());
        org.free_run = false; org.run_block(pb.data(), pb.size()); org.free_run = true;
        std::string out; uint8_t cur = org.last_symbol();
        for (uint64_t t = 0; t < reply_len; ++t) {
            uint8_t pair[2] = {cur, 0}; org.run_block(pair, 2);
            const auto& pr = org.distribution(); uint8_t next = 0;
            if (temp <= 0.0) { for (int v = 1; v < 256; ++v) if (pr[v] > pr[next]) next = (uint8_t)v; }
            else { double z = 0; std::vector<double> w(256); for (int v = 0; v < 256; ++v) { w[v] = std::pow((double)pr[v], 1.0 / temp); z += w[v]; } double u = gr.uniform() * z, c = 0; for (int v = 0; v < 256; ++v) { c += w[v]; if (u <= c) { next = (uint8_t)v; break; } } }
            org.commit_outcome(next); out += (char)next; cur = next;
            if (out.size() >= 8 && (out.find("\n\nUSER:") != std::string::npos || out.find("<|endofdialogue|>") != std::string::npos)) break;   // the reply ended
        }
        auto e = out.find("\n\nUSER:"); if (e != std::string::npos) out = out.substr(0, e); e = out.find("<|endofdialogue|>"); if (e != std::string::npos) out = out.substr(0, e);
        return out;
    };
    auto loops = [&](const std::string& g) { std::map<std::string, int> g8; size_t t = 0, r = 0; for (size_t i = 0; i + 8 <= g.size(); ++i) { if (g8[g.substr(i, 8)]++) ++r; ++t; } return t ? (double)r / t : 0.0; };
    auto valid = [&](const std::string& g) { size_t w = 0, v = 0; std::string x; auto fl = [&]() { if (x.size() >= 2) { ++w; v += vocab.count(x) > 0; } x.clear(); }; for (char c : g) { if (std::isalpha((unsigned char)c)) x += (char)std::tolower((unsigned char)c); else fl(); } fl(); return w ? (double)v / w : 0.0; };

    double on_ref = 0, on_prompt = 0, ch_ref = 0, ch_prompt = 0, drift = 0, rep8 = 0, wv = 0, len = 0; int shown = 0;
    for (size_t i = 0; i < dlgs.size(); ++i) {
        const Dlg& d = dlgs[i]; const Dlg& other = dlgs[(i + dlgs.size() / 2) % dlgs.size()];   // chance: a different prompt's reply scored against this reference
        std::string g = generate(d), gx = generate(other);
        auto cw = content_words(g), cu = content_words(turns >= 2 ? d.user2 : d.user), cr = content_words(turns >= 2 ? d.ref2 : d.ref), cx = content_words(gx);
        on_ref += overlap(cw, cr); on_prompt += overlap(cw, cu); ch_ref += overlap(cx, cr); ch_prompt += overlap(cx, cu);
        std::string h1 = g.substr(0, g.size() / 2), h2 = g.substr(g.size() / 2); drift += overlap(content_words(h2), cu) - overlap(content_words(h1), cu);
        rep8 += loops(g); wv += valid(g); len += g.size();
        if (show && shown++ < 4) { std::string u = (turns >= 2 ? d.user2 : d.user).substr(0, 90), r = g.substr(0, 200); for (auto* s : {&u, &r}) for (auto& c : *s) if (c == '\n') c = '|'; std::printf("[%zu] USER: %s\n     GEN : %s\n     on-topic(ref) %.2f prompt %.2f\n", i, u.c_str(), r.c_str(), overlap(cw, cr), overlap(cw, cu)); }
    }
    const double n = (double)dlgs.size();
    if (p.seqmem) std::printf("%s\n", org.seq_report().c_str());
    std::printf("STAY-ON-TASK (%zu dialogues, reply<=%llu B, temp %.2f): on-topic vs reference %.3f (chance %.3f) | vs prompt %.3f (chance %.3f) | drift(2nd-1st half) %+.3f | repeated8 %.3f | words-valid %.2f | mean reply %.0f B\n",
                dlgs.size(), (unsigned long long)reply_len, temp, on_ref / n, ch_ref / n, on_prompt / n, ch_prompt / n, drift / n, rep8 / n, wv / n, len / n);
    return 0;
}
