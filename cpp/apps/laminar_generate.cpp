// Free generation: the true test (goal). Train on a text slice with planted cue->phrase pairs, then free-run:
// the organism's own chosen byte becomes the next input, with no teacher. Scored the way ALM scores it
// (distinct 4-grams, repeated 4-grams, loops, word validity) plus planted-phrase recall from a cue.
//
//   laminar_generate file=HAYSTACK max_bytes=N [gen=2048] [prompts=8] [temp=1.0] [plants=32] ...params
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include "laminar/genscore.h"
#include <cstdio>
#include <string>
#include <set>
#include <map>
#include <algorithm>

using namespace laminar;

int main(int argc, char** argv) {
    std::map<std::string, std::string> rest;
    Params p = Params::from_args(argc, argv, &rest);
    p.vocab = 256; if (p.future_buckets == 32) p.future_buckets = 64;
    auto geti = [&](const char* k, uint64_t d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtoull(it->second.c_str(), nullptr, 10); };
    auto getf = [&](const char* k, double d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtod(it->second.c_str(), nullptr); };
    if (!rest.count("file")) { std::fprintf(stderr, "usage: laminar_generate file=PATH max_bytes=N [gen=2048] [prompts=8] [temp=1.0] [plants=32] [show=1] ...\n"); return 1; }
    const uint64_t gen_len = geti("gen", 2048), nprompts = geti("prompts", 8), nplants = geti("plants", 32), plant_reps = geti("plant_reps", 4);   // 2048 bytes ~ 350 words: clears the 250-word bar
    const double temp = getf("temp", 1.0); const bool show = geti("show", 1) != 0;
    std::vector<uint8_t> data = read_file_bytes(rest["file"], geti("max_bytes", 33554432));
    if (data.size() < 65536) { std::fprintf(stderr, "need more data\n"); return 1; }

    // plant cue->phrase pairs: "\n@KEY=VALUE\n" (6+6 lowercase letters), each repeated plant_reps times, spread over the slice
    Rng r(p.seed + 77); struct Plant { std::string key, val; }; std::vector<Plant> plants;
    for (uint64_t i = 0; i < nplants; ++i) { Plant pl; for (int k = 0; k < 6; ++k) pl.key += (char)('a' + r.below(26)); for (int k = 0; k < 6; ++k) pl.val += (char)('a' + r.below(26)); plants.push_back(pl); }
    const uint64_t slots = nplants * plant_reps, spacing = (data.size() - 65536) / (slots + 1);
    for (uint64_t s = 0; s < slots; ++s) { const Plant& pl = plants[s % nplants]; std::string t = "\n@" + pl.key + "=" + pl.val + "\n"; std::memcpy(&data[32768 + (s + 1) * spacing], t.data(), t.size()); }

    // training vocabulary for word validity
    std::set<std::string> vocab; { std::string w; for (uint8_t c : data) { if (std::isalpha(c)) w += (char)std::tolower(c); else { if (w.size() >= 2) vocab.insert(w); w.clear(); } } }

    Organism org(p);
    size_t pos = 0; BlockStats agg;
    while (pos + 1 < data.size()) { size_t n = std::min<size_t>(p.block, data.size() - 1 - pos); org.run_block(data.data() + pos, n + 1); org.end_block(); BlockStats& s = org.stats(); agg.events += s.events; agg.loss_sum += s.loss_sum; agg.seconds += s.seconds; s.reset(); pos += n; }
    std::printf("trained: %zu bytes, BPB %.4f, %.0f KB/s, locks %zu units %zu\n", data.size(), agg.loss_sum / agg.events / 0.6931471805599453, agg.events / agg.seconds / 1024.0, org.tree.live_locks(), org.live_units());
    if (p.seqmem) std::printf("%s\n", org.seq_report().c_str());
    org.set_mode(false, false);

    // free-run from prompts. Prompts and cues are still fed with real outcomes (teacher forcing) so that the
    // sequence memory sees them; only the organism's own choices go through commit_outcome. taken from the text (the organism is fed the prompt with teacher forcing, then runs free)
    Rng gr(p.seed + 5); GenScore tot; int loops = 0;
    for (uint64_t q = 0; q < nprompts; ++q) {
        size_t start = 65536 + gr.below((uint32_t)(data.size() - 65536 - 512)); while (start < data.size() && data[start] != '\n') ++start; ++start;
        std::vector<uint8_t> prompt(data.begin() + start, data.begin() + start + 128);
        org.free_run = false; org.run_block(prompt.data(), prompt.size()); org.free_run = true;
        std::string out; uint8_t cur = org.last_symbol(); double msum = 0; size_t copy32 = 0;
        for (uint64_t t = 0; t < gen_len; ++t) {
            uint8_t pair[2] = {cur, 0}; org.run_block(pair, 2);   // placeholder outcome; the chosen byte is committed below
            const auto& pr = org.distribution(); uint8_t next = 0;
            if (temp <= 0.0) { for (int v = 1; v < 256; ++v) if (pr[v] > pr[next]) next = (uint8_t)v; }
            else { double z = 0; std::vector<double> w(256); for (int v = 0; v < 256; ++v) { w[v] = std::pow((double)pr[v], 1.0 / temp); z += w[v]; } double u = gr.uniform() * z, c = 0; for (int v = 0; v < 256; ++v) { c += w[v]; if (u <= c) { next = (uint8_t)v; break; } } }
            int L = org.last_match_length(); msum += L; copy32 += L >= 32;   // how much of the past is being copied verbatim
            org.commit_outcome(next); out += (char)next; cur = next;
        }
        GenScore s = score_text(out, vocab); s.mean_match = msum / gen_len; s.copy32 = (double)copy32 / gen_len;
        tot.distinct4 += s.distinct4; tot.repeated4 += s.repeated4; tot.repeated8 += s.repeated8; tot.word_valid += s.word_valid; tot.unique_words += s.unique_words; tot.words += s.words;
        tot.mean_match += s.mean_match; tot.copy32 += s.copy32; tot.longest_repeat = std::max(tot.longest_repeat, s.longest_repeat); loops += s.looped;
        if (show) { std::string shown = out.substr(0, 240); for (auto& c : shown) { if (c == '\n') c = '|'; else if ((unsigned char)c < 32 || (unsigned char)c > 126) c = '~'; }
            std::printf("[prompt %llu] words %zu valid %.2f unique %.2f | rep8 %.3f longest-repeat %d B loop %s | match %.1f copy32 %.2f :: %s\n", (unsigned long long)q, s.words, s.word_valid, s.unique_words, s.repeated8, s.longest_repeat, s.looped ? "YES" : "no", s.mean_match, s.copy32, shown.c_str()); }
    }
    std::printf("FREE-RUN (%llu prompts x %llu bytes, temp %.2f): %.0f words/prompt | words-valid %.2f | unique-words %.2f | repeated8 %.3f | longest internal repeat %d B | LOOPED %d/%llu | mean verbatim match %.1f B | copying (match>=32) %.1f%%\n",
                (unsigned long long)nprompts, (unsigned long long)gen_len, temp, (double)tot.words / nprompts, tot.word_valid / nprompts, tot.unique_words / nprompts, tot.repeated8 / nprompts, tot.longest_repeat,
                loops, (unsigned long long)nprompts, tot.mean_match / nprompts, 100.0 * tot.copy32 / nprompts);

    // planted-phrase recall: prompt "\n@KEY=" and free-run 6 bytes greedily
    uint64_t hit_bytes = 0, exact = 0;
    for (const Plant& pl : plants) {
        std::string cue = "\n@" + pl.key + "="; std::vector<uint8_t> pb(cue.begin(), cue.end());
        org.free_run = false; org.run_block(pb.data(), pb.size()); org.free_run = true;
        std::string out; uint8_t cur = org.last_symbol();
        for (int t = 0; t < 6; ++t) { uint8_t pair[2] = {cur, 0}; org.run_block(pair, 2); const auto& pr = org.distribution(); uint8_t next = 0; for (int v = 1; v < 256; ++v) if (pr[v] > pr[next]) next = (uint8_t)v; org.commit_outcome(next); out += (char)next; cur = next; }
        int h = 0; for (int i = 0; i < 6; ++i) h += out[i] == pl.val[i]; hit_bytes += h; exact += (h == 6);
        if (show && (&pl - &plants[0]) < 4) std::printf("  cue @%s= -> \"%s\" (want %s)\n", pl.key.c_str(), out.c_str(), pl.val.c_str());
    }
    std::printf("PLANTED RECALL: %.1f%% of value bytes, %llu/%llu phrases exact (chance %.1f%%)\n", 100.0 * hit_bytes / (6.0 * nplants), (unsigned long long)exact, (unsigned long long)nplants, 100.0 / 26);
    return 0;
}
