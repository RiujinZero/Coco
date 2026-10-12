// Rolling-context acceptance test: needle-in-haystack associative recall.
//
// Haystack: a text file cycled. Needles "\n@KEY=VALUE\n" (random lowercase letters) are planted at
// uniformly spread positions within the first 2*window bytes and each is queried once at a chosen
// distance d (cycled over a set spanning window/8 .. 2*window). Recall = accuracy of the committed
// prediction on the VALUE bytes of the query occurrence. The equivalent of a "rolling context window
// of W tokens" is: recall stays high for d <= W and drops for d > W.
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include <cstdio>
#include <string>
#include <algorithm>
#include <map>

using namespace laminar;

int main(int argc, char** argv) {
    std::map<std::string, std::string> rest;
    Params p = Params::from_args(argc, argv, &rest);
    p.vocab = 256; p.record_preds = true; if (p.future_buckets == 32) p.future_buckets = 64;
    auto geti = [&](const char* k, uint64_t d) { auto it = rest.find(k); return it == rest.end() ? d : std::strtoull(it->second.c_str(), nullptr, 10); };
    if (!rest.count("file")) { std::fprintf(stderr, "usage: laminar_needle file=HAYSTACK [window=N] [needles=N] [key_len=6] [val_len=6] [warm=N] ...params\n"); return 1; }
    const uint64_t W = geti("window", 1u << 21), needles = geti("needles", 256), klen = geti("key_len", 6), vlen = geti("val_len", 6), warm = geti("warm", 0);
    std::vector<uint8_t> hay = read_file_bytes(rest["file"]);
    if (hay.size() < 1024) { std::fprintf(stderr, "haystack too small\n"); return 1; }
    struct Needle { uint64_t plant, query, dist; std::string s; };
    std::vector<Needle> ns; Rng r(p.seed + 99);
    const uint64_t dists[] = {W / 8, W / 4, W / 2, 3 * W / 4, W, 5 * W / 4, 3 * W / 2, 2 * W};
    const size_t nd = sizeof(dists) / sizeof(dists[0]);
    const uint64_t nlen = 2 + klen + 1 + vlen + 1;
    // Plants sit on a grid of `spacing` bytes and queries on the same grid shifted by spacing/2. Distances are
    // multiples of W/8; spacing = 4160 = 64*65 guarantees neither plant/query nor query/query collisions for
    // distance differences below 65*W/8 (see derivation in docs/ARCHITECTURE.md "Rolling context").
    const uint64_t spacing = 4160;
    if (spacing * needles > 2 * W) { std::fprintf(stderr, "too many needles for this window (max %llu)\n", (unsigned long long)(2 * W / spacing)); return 1; }
    for (uint64_t k = 0; k < needles; ++k) {
        Needle n; n.s = "\n@"; for (uint64_t i = 0; i < klen; ++i) n.s += (char)('a' + r.below(26)); n.s += '=';
        for (uint64_t i = 0; i < vlen; ++i) n.s += (char)('a' + r.below(26)); n.s += '\n';
        n.dist = dists[k % nd];
        n.plant = warm + k * spacing;
        n.query = n.plant + n.dist + spacing / 2;
        ns.push_back(n);
    }
    uint64_t L = 0; for (auto& n : ns) L = std::max(L, n.query + nlen + 2);
    std::vector<uint8_t> stream(L); for (uint64_t i = 0; i < L; ++i) stream[i] = hay[i % hay.size()];
    { std::vector<uint64_t> occ; for (auto& n : ns) { occ.push_back(n.plant); occ.push_back(n.query); } std::sort(occ.begin(), occ.end());
      for (size_t i = 1; i < occ.size(); ++i) if (occ[i] - occ[i - 1] < nlen) { std::fprintf(stderr, "needle occurrences overlap at %llu\n", (unsigned long long)occ[i]); return 1; } }
    for (auto& n : ns) { std::memcpy(&stream[n.plant], n.s.data(), nlen); std::memcpy(&stream[n.query], n.s.data(), nlen); }

    Organism org(p);
    if (rest.count("log")) org.open_log(rest["log"]);
    std::printf("needle test: window=%llu stream=%llu MB needles=%llu key=%llu val=%llu units=%u episodic=%d\n",
                (unsigned long long)W, (unsigned long long)(L >> 20), (unsigned long long)needles, (unsigned long long)klen, (unsigned long long)vlen, p.units, (int)p.episodic);
    size_t pos = 0; BlockStats agg; uint64_t since = 0;
    while (pos + 1 < L) {
        size_t n = std::min<size_t>(p.block, L - 1 - pos);
        org.run_block(stream.data() + pos, n + 1); org.end_block();
        BlockStats& s = org.stats();
        agg.events += s.events; agg.loss_sum += s.loss_sum; agg.correct += s.correct; agg.macs += s.macs; agg.seconds += s.seconds; agg.bytes += s.bytes;
        agg.episodic_births += s.episodic_births; agg.episodic_evictions += s.episodic_evictions; agg.exact_admissions += s.exact_admissions; agg.lock_splits += s.lock_splits; agg.imprint_ok += s.imprint_ok;
        since += s.events;
        if (since >= (1u << 19)) { since = 0;
            std::printf("  %5.1f MB | BPB %.3f | %.0f KB/s | exact locks %zu live units %zu | epi births %llu (imprint ok %.1f%%) evict %llu exact adm %.3f\n",
                        (double)pos / 1048576.0, agg.loss_sum / agg.events / 0.6931471805599453, agg.bytes / agg.seconds / 1024.0, org.exact_locks(), org.live_units(),
                        (unsigned long long)agg.episodic_births, agg.episodic_births ? 100.0 * agg.imprint_ok / agg.episodic_births : 0.0,
                        (unsigned long long)agg.episodic_evictions, (double)agg.exact_admissions / agg.events); std::fflush(stdout); }
        s.reset(); pos += n;
    }
    // preds[t] is the prediction for stream[t+1]; value bytes of an occurrence at q are q+2+klen+1 .. +vlen.
    // The event predicting value byte i is at stream index q+2+klen+i (its outcome is the value byte).
    struct Acc { double n = 0, born = 0, imprint = 0, matched = 0, correct = 0, correct_given_matched = 0; };
    auto scan = [&](uint64_t occ, const Needle& n, Acc& a, bool is_query) {
        for (uint64_t i = 0; i < vlen; ++i) {
            uint64_t t = occ + 2 + klen + i; if (t >= org.preds.size()) continue;
            uint8_t fl = org.flags[t]; bool ok = org.preds[t] == (uint8_t)n.s[2 + klen + 1 + i];
            a.n++; a.born += (fl & Organism::FL_BIRTH) != 0; a.imprint += (fl & Organism::FL_IMPRINT_OK) != 0;
            a.matched += (fl & Organism::FL_EXACT) != 0; a.correct += ok; if (is_query && (fl & Organism::FL_EXACT)) a.correct_given_matched += ok;
        } };
    std::map<uint64_t, Acc> byd; Acc plant;
    for (auto& n : ns) { scan(n.query, n, byd[n.dist], true); scan(n.plant, n, plant, false); }
    std::printf("PLANT: value bytes %.0f | episodic birth %.1f%% | imprint decoded at birth %.1f%% | first-exposure acc %.1f%%\n", plant.n, 100 * plant.born / plant.n, 100 * plant.imprint / plant.n, 100 * plant.correct / plant.n);
    std::printf("  birth by value-byte index:"); for (uint64_t i = 0; i < vlen; ++i) { double b = 0; for (auto& n : ns) { uint64_t t = n.plant + 2 + klen + i; if (t < org.flags.size()) b += (org.flags[t] & Organism::FL_BIRTH) != 0; } std::printf(" [%llu] %.0f%%", (unsigned long long)i, 100 * b / ns.size()); } std::printf("\n");
    std::printf("RECALL at query (chance ~%.1f%%):\n", 100.0 / 26);
    for (auto& [d, a] : byd) std::printf("  distance %5.2f x W (%9llu B): exact lock matched %5.1f%% | recall %5.1f%% | recall given matched %5.1f%%  (n=%.0f)\n",
                                          (double)d / W, (unsigned long long)d, 100 * a.matched / a.n, 100 * a.correct / a.n, a.matched ? 100 * a.correct_given_matched / a.matched : 0.0, a.n);
    if (p.agency) std::printf("%s\n", org.agency_report().c_str());
    std::printf("TOTAL: %llu bytes, BPB %.4f, %.0f KB/s, %.0f MACs/byte, exact locks %zu, live units %zu, episodic births %llu, evictions %llu, splits %llu\n",
                (unsigned long long)agg.events, agg.loss_sum / agg.events / 0.6931471805599453, agg.bytes / agg.seconds / 1024.0, (double)agg.macs / agg.events,
                org.exact_locks(), org.live_units(), (unsigned long long)agg.episodic_births, (unsigned long long)agg.episodic_evictions, (unsigned long long)agg.lock_splits);
    return 0;
}
