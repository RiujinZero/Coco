#pragma once
// Sequence memory in the Atomic Thought sense (Brain Simulator Thought white paper §2.4-2.5, UKS.Sequence.cs):
// the byte stream is captured as an explicit chain of steps (each step: value, NXT), everything is captured
// first and competes later, and prediction is sequence search — locate steps whose preceding elements match the
// current situation, then follow NXT to read what historically followed. Several matched sequences propose
// continuations that compete by match length and use (ALM's normalized competition). A long exact match may be
// followed as a unit (a "bubble", the shared sub-sequence of the paper) — that is serialization.
//
// Representation: a rolling ring of steps (the window) plus reverse indices: for each lag L in {4, 8, 16} a hash
// of the L bytes preceding a step maps to the most recent steps with that context. That is the paper's "reverse
// value links" made bounded. Nothing is learned in the usual sense: capture, compete (by count/recency), prune
// (ring eviction) — the capture → compete → consolidate → prune cycle.
#include <cstdint>
#include <immintrin.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <cctype>
#include "rng.h"
#include "attend.h"
#include "reasoning_language.h"
#include <memory>
#include <functional>

namespace laminar {

struct SequenceMemory {
    static constexpr int NLAG = 8; static constexpr int LAGS[NLAG] = {2, 3, 4, 6, 8, 12, 16, 24};   // the receptor ladder
    static constexpr int SLOT = 6;                      // steps remembered per context bucket
    uint32_t window = 0, hash_bits = 21;
    std::vector<uint8_t> ring;                          // step values (the stream); NXT is position + 1
    std::vector<uint8_t> self_bits;                     // provenance: 1 = the organism's own emission (context only, never an instance)
    uint64_t n = 0;                                     // steps captured
    uint64_t self_skipped = 0, self_repeats = 0;   // instances refused for provenance; predictions whose context was already the organism's own
    std::vector<uint32_t> table[NLAG];                  // [bucket][SLOT] step positions (absolute, +1; 0 = empty)
    uint32_t pre_bucket[NLAG] = {}; uint64_t pre_chash[NLAG] = {}; uint64_t pre_n = ~0ULL;   // the next predict's buckets, hashed once at capture and prefetched
    // match length between the contexts ending at pos-1 and n-1, up to 64: eight bytes at a time while neither range wraps
    inline int match_len(uint64_t pos) const {
        int L = 0;
        while (L < 64 && pos >= (uint64_t)L + 8) {
            uint64_t p1 = pos - 8 - L, p2 = n - 8 - L;          // the 8-byte words [p1, p1+8) and [p2, p2+8), highest address compared first
            size_t i1 = p1 % window, i2 = p2 % window; if (i1 + 8 > window || i2 + 8 > window) break;
            uint64_t a, b; std::memcpy(&a, &ring[i1], 8); std::memcpy(&b, &ring[i2], 8);
            if (a == b) { L += 8; continue; }
            uint64_t x = a ^ b; int eq = 0; while (eq < 8 && ((x >> (56 - 8 * eq)) & 0xFF) == 0) ++eq;   // equal bytes from the high end
            return L + eq;
        }
        while (L < 64 && pos >= (uint64_t)L + 1 && at(pos - 1 - L) == at(n - 1 - L)) ++L;
        return L;
    }
    // last prediction's support
    int match_lag = 0; float confidence = 0.f; uint32_t committed_pos = 0; int bubble_left = 0;
    uint64_t bubbles = 0, bubble_bytes = 0;
    // consolidated tier (the lifecycle's "consolidate"): per lag and context bucket, a small table of continuation
    // counts that no longer reference the text. Instances bubble up into it and can then be forgotten, so the
    // memory is bounded by table size rather than by exposure. CSLOT continuations per bucket, saturating counts.
    static constexpr int CSLOT = 4;
    // Each entry is TAGGED with a fingerprint of the context that owns it: a class count is only ever read for the
    // context that wrote it (instances are verified against the ring; without a tag, long-lag buckets aggregate
    // dozens of unrelated contexts and the tier speaks noise). A different context ages the entry and takes it
    // over only once its counts have decayed to nothing — competition for the slot, as within the entry.
    struct CEntry { uint32_t tag; uint8_t sym[CSLOT]; uint16_t cnt[CSLOT]; };
    std::vector<CEntry> ctable[NLAG]; uint32_t cbits = 0; bool consolidate = false; float class_gain = 1.f;
    uint64_t consolidated = 0, ctag_misses = 0;
    void init_consolidated(uint32_t bits) { cbits = bits; consolidate = true; for (int l = 0; l < NLAG; ++l) ctable[l].assign((size_t)1 << bits, CEntry{}); }
    inline uint64_t chash_of(uint64_t raw) const { return mix64(raw ^ 0x9E3779B97F4A7C15ULL); }   // the tier's mix of the same raw hash
    inline uint64_t chash(int li, uint64_t end_pos) const {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (int i = 1; i <= LAGS[li]; ++i) h = (h ^ at(end_pos - i)) * 0x100000001b3ULL;
        return chash_of(h ^ ((uint64_t)(li + 1) << 56));
    }
    inline uint32_t cbucket(uint64_t h) const { return (uint32_t)(h >> (64 - cbits)); }
    static inline uint32_t ctag(uint64_t h) { return (uint32_t)h | 1u; }
    void consolidate_step(uint64_t pos_end, uint8_t v) {   // context = the LAGS[l] bytes before pos_end; v followed it
        uint64_t raw[NLAG]; const bool all = pos_end >= (uint64_t)LAGS[NLAG - 1]; if (all) lag_hashes(pos_end, raw);
        for (int l = 0; l < NLAG; ++l) {
            if (pos_end < (uint64_t)LAGS[l]) continue;
            uint64_t h = all ? chash_of(raw[l]) : chash(l, pos_end); CEntry& e = ctable[l][cbucket(h)]; uint32_t tag = ctag(h);
            if (e.tag != tag) {                                 // owned by another context: age it; take it when empty
                int alive = 0; for (int s = 0; s < CSLOT; ++s) { if (e.cnt[s]) --e.cnt[s]; alive += e.cnt[s] != 0; }
                if (alive) { ++ctag_misses; continue; }
                e = CEntry{}; e.tag = tag;
            }
            int k = -1; for (int s = 0; s < CSLOT; ++s) if (e.cnt[s] && e.sym[s] == v) { k = s; break; }
            if (k < 0) { int m = 0; for (int s = 1; s < CSLOT; ++s) if (e.cnt[s] < e.cnt[m]) m = s; if (e.cnt[m] == 0) { e.sym[m] = v; e.cnt[m] = 1; } else { if (--e.cnt[m] == 0) { e.sym[m] = v; e.cnt[m] = 1; } } continue; } // compete for the slot
            if (e.cnt[k] < 65535) ++e.cnt[k]; else for (int s = 0; s < CSLOT; ++s) e.cnt[s] >>= 1;
        }
        ++consolidated;
    }
    // bubble tier: content words are thoughts ("bubbles"); each dialogue (episode) in the window keeps the set of
    // bubbles it contains — the relations between thoughts within an episode. The current dialogue's bubbles are
    // the GOAL set; a remembered instance's continuation is weighted by how many goal bubbles its own dialogue
    // shares. Attention by relation between thoughts rather than by byte adjacency: this is what lets a reply
    // stay on the topic the user raised after the local context has become generic ("\n\nASSISTANT:").
    static constexpr int BUBBLES = 48; static constexpr int CHUNK = 64; static constexpr int IDX_BITS = 18, IDX_SLOT = 8;
    std::vector<uint32_t> chunk_dialogue;               // dialogue id per 64-byte chunk of the ring
    std::vector<uint32_t> dialogue_bubbles;             // [dialogue % ndialogues][BUBBLES] word hashes (0 = empty)
    std::vector<uint64_t> dialogue_start;               // [dialogue % ndialogues] first step of the dialogue
    std::vector<uint32_t> bubble_index;                 // [bubble hash bucket][IDX_SLOT] most recent dialogues containing it (the relation store)
    std::vector<uint32_t> bubble_df;                    // [bubble hash bucket] dialogues that contain it (rarity = information)
    std::vector<float> bubble_util;                     // [bubble hash bucket] consequence: how well commitments selected through this thought held (EMA)
    float util_rate = 0.f;                              // 0 = selection by rarity only; >0 = rarity x (0.5 + utility), utility credited per commitment
    uint32_t ndialogues = 0, cur_dialogue = 1; float topic_gain = 0.f; uint32_t goal_hits = 0, goal_tests = 0;
    uint64_t topic_retrievals = 0, topic_candidates = 0; bool topic_active = false;   // retrieval by relation runs only while generating
    std::string cur_word; std::vector<uint32_t> goal;   // the current dialogue's bubbles (also its goal set)
    void init_bubbles(uint32_t max_dialogues) {
        ndialogues = max_dialogues; chunk_dialogue.assign(window / CHUNK + 1, 0); dialogue_bubbles.assign((size_t)max_dialogues * BUBBLES, 0);
        dialogue_start.assign(max_dialogues, 0); bubble_index.assign((size_t)IDX_SLOT << IDX_BITS, 0); bubble_df.assign((size_t)1 << IDX_BITS, 0); bubble_util.assign((size_t)1 << IDX_BITS, 0.5f); goal.clear(); cur_dialogue = 1;
    }
    static uint32_t word_hash(const std::string& w) { uint64_t h = 1469598103934665603ULL; for (char c : w) h = (h ^ (uint8_t)c) * 1099511628211ULL; return (uint32_t)(mix64(h) >> 32) | 1u; }
    inline uint32_t ib(uint32_t h) const { return h >> (32 - IDX_BITS); }
    void add_bubble(uint32_t dlg, uint32_t h) {
        uint32_t* b = &dialogue_bubbles[(size_t)(dlg % ndialogues) * BUBBLES];
        for (int i = 0; i < BUBBLES; ++i) { if (b[i] == h) return; if (!b[i]) { b[i] = h; goto indexed; } }
        b[(h >> 8) % BUBBLES] = h;                          // full: replace pseudo-randomly
    indexed:
        { uint32_t* ix = &bubble_index[(size_t)ib(h) * IDX_SLOT]; if (ix[0] != dlg) { for (int s = IDX_SLOT - 1; s > 0; --s) ix[s] = ix[s - 1]; ix[0] = dlg; ++bubble_df[ib(h)]; } }
    }
    void begin_dialogue() { sents.clear(); csent = Sent{}; csent.st = n + 1; ncand = 0; answer_pending = false; if (trace_on) last_trace = "(no selection)"; ++cur_dialogue; uint32_t* b = &dialogue_bubbles[(size_t)(cur_dialogue % ndialogues) * BUBBLES]; for (int i = 0; i < BUBBLES; ++i) b[i] = 0; dialogue_start[cur_dialogue % ndialogues] = n; goal.clear(); }
    float idf(uint32_t h) const { uint32_t df = bubble_df[ib(h)]; return df ? std::log((float)std::max<uint32_t>(cur_dialogue, 2) / (float)df) : 0.f; }
    // Goal-directed retrieval by relation (Agkern commitment): the goal is the user's thoughts, frozen when the
    // reply begins; the rarest of them select the most related remembered episode; the organism COMMITS to that
    // episode's reply and follows it as a single instance, re-aligning on the current context when the body
    // diverges and dropping the commitment only after sustained mismatch, when the next most related episode
    // takes over. The window supplies the episodes; the relation store (bubble index) supplies the selection.
    std::vector<uint32_t> goal_frozen; bool goal_is_frozen = false;
    uint32_t commit_dlg = 0; uint64_t commit_cur = 0, commit_st = 0, commit_en = 0; float commit_rel = 0.f; int commit_miss = 0;
    uint32_t commit_thought = 0; uint32_t commit_steps = 0, commit_faults = 0;   // consequence of the commitment: followed bytes vs re-alignments/misses
    void credit_commitment() {                          // three-factor credit onto the relation store: the selecting thought earns what its choice held
        if (!commit_thought || !commit_steps || util_rate <= 0.f) return;
        float held = (float)commit_steps / (float)(commit_steps + 4 * commit_faults);
        float& u = bubble_util[ib(commit_thought)]; u += util_rate * (held - u);
        commit_thought = 0; commit_steps = commit_faults = 0;
    }
    uint32_t tried[16]; int ntried = 0; uint64_t commits = 0, realigns = 0, commit_drops = 0;
    bool goal_extend = false;                               // bubble sequence: after a commitment ends, the reply's own new thoughts join the goal
    void topic_reset() { credit_commitment(); goal_is_frozen = false; commit_dlg = 0; ntried = 0; }
    // the reply that answered the shared thought: the assistant turn after the first occurrence of a goal word
    bool find_reply(uint32_t d, uint64_t& st, uint64_t& en, const std::vector<std::pair<float, uint32_t>>& g) const {
        st = dialogue_start[d % ndialogues]; if (n - st >= window - 64) return false;   // evicted
        en = std::min(n, st + 6144);
        uint64_t hit = 0; { std::string w; for (uint64_t q = st; q < en && !hit; ++q) { uint8_t c = at(q); if (std::isalpha(c)) w += (char)std::tolower(c); else { if (w.size() >= 4) { uint32_t h = word_hash(w); for (auto& [ww, hh] : g) if (hh == h) { hit = q; break; } } w.clear(); } } }
        if (!hit) return false;
        static const char* A = "ASSISTANT:"; uint64_t q = hit;
        for (; q + 10 <= en; ++q) { bool m = true; for (int i = 0; i < 10 && m; ++i) m = at(q + i) == (uint8_t)A[i]; if (m) break; }
        if (q + 10 > en) return false; st = q + 10; en = std::min(n, st + 2048); return en > st + 16;
    }
    bool aligned_at(uint64_t q, int L) const { if (q < (uint64_t)L || n < (uint64_t)L) return false; for (int i = 1; i <= L; ++i) if (at(q - i) != at(n - i)) return false; return true; }
    float goal_rel = 0.f; uint64_t goal_expansions = 0;      // > 0: the goal's related thoughts join episode selection at this discount
    bool select_episode() {
        if (!goal_is_frozen) { goal_frozen = goal; goal_is_frozen = true; }
        std::vector<std::pair<float, uint32_t>> g; for (uint32_t h : goal_frozen) { float w = idf(h); if (w > 1.f) g.emplace_back(w, h); }
        if (goal_rel > 0.f && !rel.empty()) {                 // relations as things, directing generation: what the goal's thoughts relate to
            size_t n0 = g.size();
            for (size_t i = 0; i < n0 && g.size() < 24; ++i) { Triple* t = rel_slots(g[i].second);
                for (int k = 0; k < RSLOT && g.size() < 24; ++k) { if (t[k].a != g[i].second || !t[k].b) continue; float w = idf(t[k].b) * t[k].u * t[k].g * goal_rel; if (w <= 0.5f) continue;
                    bool dup = false; for (auto& x : g) if (x.second == t[k].b) { dup = true; break; } if (!dup) { g.emplace_back(w, t[k].b); ++goal_expansions; } } }
        }
        if (g.empty()) return false; std::sort(g.begin(), g.end(), [](auto& a, auto& b) { return a.first > b.first; }); if (g.size() > 6) g.resize(6);
        float best = 0.f; uint32_t bd = 0, bh = 0; uint64_t bst = 0, ben = 0;
        for (auto& [w, h] : g) {
            const uint32_t* ix = &bubble_index[(size_t)ib(h) * IDX_SLOT];
            for (int s = 0; s < IDX_SLOT; ++s) {
                uint32_t d = ix[s]; if (!d || d == cur_dialogue) continue;
                bool skip = false; for (int i = 0; i < ntried; ++i) if (tried[i] == d) { skip = true; break; } if (skip) continue;
                float rel = w; for (auto& [w2, h2] : g) if (h2 != h) { const uint32_t* b = &dialogue_bubbles[(size_t)(d % ndialogues) * BUBBLES]; for (int i = 0; i < BUBBLES && b[i]; ++i) if (b[i] == h2) { rel += w2; break; } }
                if (util_rate > 0.f) rel *= 0.5f + bubble_util[ib(h)];   // consequence-weighted selection (agency)
                if (rel <= best) continue; uint64_t st, en; if (!find_reply(d, st, en, g)) continue;
                best = rel; bd = d; bh = h; bst = st; ben = en;
            }
        }
        if (!bd) return false;
        credit_commitment();
        commit_dlg = bd; commit_st = bst; commit_en = ben; commit_cur = bst; commit_rel = best; commit_miss = 0; commit_thought = bh; if (ntried < 16) tried[ntried++] = bd; ++commits;
        return true;
    }
    // ---- Relations within the episode: reasoning over supporting facts (bAbI) -------------------------------
    // Sentences of the current episode are kept as bubble sequences (their content words, with ring positions).
    // At a reply boundary ("ASSISTANT: ") the question's most specific thought (the question word that occurs in the
    // fewest earlier sentences) selects the most recent sentence containing it; that sentence's OBJECT (its last
    // content word other than the thought, else its first) is the depth-1 candidate; the object then selects the
    // next most recent sentence containing it, and so on — a chain of hops through relations between thoughts.
    // Which depth answers which question FORM (its first two words) is learned by consequence: at training time
    // the observed answer credits the depth that produced it. Generation follows the chosen candidate's bytes.
    static constexpr int NW = 16;
    struct Sent { uint64_t st = 0; uint32_t w[NW]; uint64_t wpos[NW]; uint8_t wlen[NW]; uint32_t wstem[NW]; int nw = 0; bool question = false; uint32_t form[4]; int nf = 0; uint32_t rel = 0;
                  uint32_t between[NW] = {}; bool cap[NW] = {}; };   // between[j]: relation words between thought j and j+1; cap: capitalized (a name)
    uint64_t between_acc = 0; bool cword_cap = false, cword_self = false;       // running hash of relation words since the last thought; the current word's capitalization
    std::vector<uint8_t> answer_words;                      // [stem bucket] 1 = observed as an answer: a relation word, not a node
    static bool is_function(const std::string& w) { static const char* F[] = {"the", "and", "for", "with", "from", "that", "this", "then", "there", "are", "was", "were", "has", "have", "had", "some", "all", "any", "into", "onto", "over", "out", "off"}; for (const char* x : F) if (w == x) return false; return false; }
    bool chain_on = false; std::vector<Sent> sents; Sent csent; std::string cword; uint64_t cword_start = 0;
    static constexpr int MAXD = 8; struct Cand { uint32_t h = 0; uint64_t pos = 0; int len = 0; uint32_t stem = 0; }; Cand cand[MAXD + 1]; int ncand = 0;
    std::vector<float> depth_score, answer_score; uint32_t qform = 0; int role1 = 0; bool chain_forcing = false;   // role1: the thought as subject (0) / object (1) in its first sentence
    static constexpr int END = MAXD + 1;                   // depth slot meaning "the end of the chain"
    inline size_t dkey() const { return ((size_t)(qform & 4095) * 2 + (size_t)role1) * (MAXD + 2); }   // answer_score[(form, stem) bucket]: how often such a word answered such a question
    static uint32_t stem_hash(const std::string& w) { return word_hash(w.substr(0, 3)); }
    inline uint32_t akey(uint32_t form, uint32_t stem) const { return (uint32_t)(mix64(((uint64_t)form << 32) | stem) & 65535); } bool answer_pending = false; std::string answer_word; int answer_off = 0; int chosen_depth = 0;
    uint64_t chains = 0, chain_credits = 0, chain_hits[MAXD + 2] = {}; float chain_rate = 0.1f;
    std::string last_trace; bool trace_on = false;      // legible chain of thought: thought -> candidates by depth -> chosen
    std::string word_at(uint64_t pos, int len) const { std::string w; for (int i = 0; i < len; ++i) w += (char)at(pos + i); return w; }
    std::vector<float> dir_score; Cand cand_before, cand_latest; bool have_before = false; uint32_t rel1 = 0;   // the two second hops (before S1 / latest) and S1's relation
    // ---- Paths and composition (CLUTRR): a question about TWO thoughts is answered by the composition of the
    // relations along the shortest path between them through the episode's sentences. Each hop is a class
    // (relation words, direction); unary[class] -> answer for one hop; comp[(answer-so-far, class)] -> answer for
    // each further hop, learned by consequence, so chains longer than any seen compose by associativity.
    struct Top2 { uint32_t a = 0, b = 0; uint16_t ca = 0, cb = 0; void credit(uint32_t x) { if (x == a) { if (ca < 65535) ++ca; } else if (x == b) { if (cb < 65535) ++cb; if (cb > ca) { std::swap(a, b); std::swap(ca, cb); } } else if (!a) { a = x; ca = 1; } else if (!b || cb <= 1) { b = x; cb = 1; } }
                  uint32_t best() const { return ca ? a : 0; }
                  float conf() const { return ca ? (float)ca / (float)(ca + cb + 1) : 0.f; } };
    std::vector<Top2> unary, comp; struct Form { uint64_t pos = 0; int len = 0; }; std::vector<Form> answer_form;   // [stem bucket] last observed surface form of an answer
    static constexpr int MAXP = 6; uint32_t path_cls[MAXP]; uint32_t path_cls2[MAXP]; int path_len = 0; uint32_t path_answer = 0; bool path_mode = false; uint64_t paths = 0, path_credits = 0, path_answers = 0;
    static uint32_t cls(uint32_t between, bool fwd) { return (uint32_t)(mix64(((uint64_t)between << 1) | (fwd ? 1u : 0u)) >> 32) | 1u; }
    uint32_t path_node[MAXP]; bool attr_on = false; uint64_t attr_hits = 0;
    bool attr_eq = false; struct AttrSig { uint32_t key[8] = {}; uint32_t ans[8] = {}; int n = 0; uint32_t canon = 0; }; std::vector<AttrSig> asig; uint64_t attr_merges = 0;
    void attr_observe(uint32_t attr, uint32_t key, uint32_t answer) {   // remember what this attribute answered under this key
        if (!attr_eq || !attr) return; if (asig.empty()) asig.assign(4096, AttrSig{}); AttrSig& g = asig[attr & 4095]; if (!g.canon) g.canon = attr;
        for (int i = 0; i < g.n; ++i) if (g.key[i] == key) { g.ans[i] = answer; return; }
        int i = g.n < 8 ? g.n++ : (int)(key % 8); g.key[i] = key; g.ans[i] = answer;
        // does another attribute agree on two or more shared keys and disagree on none? then they are one attribute
        for (size_t o = 0; o < asig.size(); ++o) { AttrSig& h = asig[o]; if (&h == &g || !h.n || h.canon == g.canon) continue; int agree = 0, disagree = 0;
            for (int a = 0; a < g.n; ++a) for (int b = 0; b < h.n; ++b) if (g.key[a] == h.key[b]) { if (g.ans[a] == h.ans[b]) ++agree; else ++disagree; }
            if (agree >= 2 && !disagree) { uint32_t c = std::min(g.canon, h.canon); uint32_t old_g = g.canon, old_h = h.canon; for (auto& x : asig) if (x.canon == old_g || x.canon == old_h) x.canon = c; ++attr_merges; } }
    }
    uint32_t attr_canon(uint32_t attr) const { if (!attr_eq || !attr || asig.empty()) return attr; const AttrSig& g = asig[attr & 4095]; return g.canon ? g.canon : attr; }
    uint32_t attr_of(uint32_t thought) {                     // the most informative relation-vocabulary word the store attaches to a thought (0 = none)
        if (!attr_on || rel.empty() || answer_words.empty()) return 0; Triple* t = rel_slots(thought); float best = 0.f; uint32_t a = 0;
        for (int i = 0; i < RSLOT; ++i) { if (t[i].a != thought || !t[i].b) continue; uint32_t st = stem_hash(word_at(t[i].bpos, t[i].blen)); if (!answer_words[st & 65535]) continue; float v = t[i].u * (0.1f + t[i].g); if (v > best) { best = v; a = st; } }
        return attr_canon(a);
    }
    inline uint32_t ckey(uint32_t prefix, uint32_t c, uint32_t attr) const { return (uint32_t)(mix64(((uint64_t)prefix << 32) | c) * (attr ? (uint64_t)attr * 2654435761ULL + 1 : 1) & 65535); }
    // ---- Learned key resolution. Which distinctions the operator's key must carry is not chosen here: every hop
    // keeps four candidate keys (plain; with the destination's attribute; with the source's; with both), all are
    // credited from each observed answer, and a per-form selector is credited by whether each key's existing entry
    // was right. The organism learns, per question form, which key resolves the cases — "granddaughter" vs
    // "grandson" is then a distinction it found, not one that was coded.
    static constexpr int NV = 6; std::vector<float> ksel; uint32_t path_src = 0; uint64_t ksel_credits = 0;
    inline float& kacc(int v) { if (ksel.empty()) ksel.assign((size_t)4096 * NV, 0.5f); return ksel[(size_t)(qform & 4095) * NV + v]; }
    uint32_t vkey(int v, uint32_t prefix, uint32_t c, uint32_t a_src, uint32_t a_dst, uint32_t c2 = 0) const {   // 0 if the variant's feature is unavailable
        switch (v) { case 0: return ckey(prefix, c, 0); case 1: return a_dst ? ckey(prefix, c, a_dst) : 0; case 2: return a_src ? ckey(prefix, c, a_src ^ 0x9E3779B9u) : 0;
                     case 3: return (a_src && a_dst) ? ckey(prefix, c, (uint32_t)(mix64(((uint64_t)a_src << 32) | a_dst) >> 32)) : 0;
                     case 4: return c2 ? ckey(prefix, c2, 0) : 0; default: return (c2 && a_dst) ? ckey(prefix, c2, a_dst) : 0; }
    }
    uint32_t hop_answer(int i, uint32_t prefix) {           // the operator's answer for hop i after prefix, by the best-accredited key that has an entry
        std::vector<Top2>& T = i == 0 ? unary : comp; uint32_t a_src = attr_of(i == 0 ? path_src : path_node[i - 1]), a_dst = attr_of(path_node[i]);
        float best = 0.f; uint32_t ans = 0; int bv = 0;       // the key whose entry is best accredited AND best evidenced
        for (int v = 0; v < NV; ++v) { uint32_t k = vkey(v, prefix, path_cls[i], a_src, a_dst, path_cls2[i]); if (!k) continue; const Top2& e = T[k]; if (!e.best()) continue;
            float sc = kacc(v) * e.conf(); if (sc > best) { best = sc; ans = e.best(); bv = v; } }
        if (ans && bv) ++attr_hits; return ans;
    }
    uint32_t compose_path(int upto) {                       // answer stem of the first `upto` hops, 0 if unknown
        if (upto < 1 || unary.empty()) return 0; uint32_t r = hop_answer(0, 0); if (!r) return 0;
        for (int i = 1; i < upto; ++i) { r = hop_answer(i, r); if (!r) return 0; }
        return r;
    }
    void credit_path(uint32_t answer_stem) {                // credit every candidate key of the last hop, and the selector by which keys already had it right
        if (!path_len) return; ++path_credits;
        int i = path_len - 1; uint32_t prefix = 0; if (i > 0) { prefix = compose_path(i); if (!prefix) return; }   // the prefix must already compose
        std::vector<Top2>& T = i == 0 ? unary : comp; uint32_t a_src = attr_of(i == 0 ? path_src : path_node[i - 1]), a_dst = attr_of(path_node[i]);
        { uint32_t plain = ckey(prefix, path_cls[i], 0); if (a_dst) attr_observe(a_dst, plain, answer_stem); if (a_src) attr_observe(a_src, plain ^ 0x5555u, answer_stem); }
        for (int v = 0; v < NV; ++v) { uint32_t k = vkey(v, prefix, path_cls[i], a_src, a_dst, path_cls2[i]); if (!k) continue;
            uint32_t had = T[k].best(); if (had) { float& acc = kacc(v); float w = chain_rate * T[k].conf(); acc += w * ((had == answer_stem ? 1.f : 0.f) - acc); ++ksel_credits; }   // the selector learns from evidenced entries
            T[k].credit(answer_stem); }
    }
    // shortest path from thought a to thought b through sentences (adjacent thoughts within a sentence are linked)
    // relation between name-nodes j and k of a sentence: every word between them (thoughts and relation words), in order
    // the class of a hop is the RELATION VOCABULARY inside the span — the words the organism has observed as answers
    // ("grandson", "sister"), learned from the stream — so that every paraphrase of a relation is one class;
    // a span with no such word falls back to its full content
    uint32_t span_word(const Sent& x, int j, int k) const {  // the stem of the relation word nearest the pair (0 if none): what the stream calls this relation
        int lo = std::min(j, k), hi = std::max(j, k); int best = -1; uint64_t bestd = ~0ULL;
        for (int i = std::max(0, lo - 2); i <= std::min(x.nw - 1, hi + 2); ++i) {
            if (i == j || i == k || x.cap[i] || answer_words.empty() || !answer_words[x.wstem[i] & 65535]) continue;
            uint64_t d = std::min(x.wpos[i] > x.wpos[j] ? x.wpos[i] - x.wpos[j] : x.wpos[j] - x.wpos[i], x.wpos[i] > x.wpos[k] ? x.wpos[i] - x.wpos[k] : x.wpos[k] - x.wpos[i]);
            if (d < bestd) { bestd = d; best = i; } }
        return best >= 0 ? x.wstem[best] : 0;
    }
    uint32_t span_full(const Sent& x, int j, int k) const {  // every relation-vocabulary word around the pair, in order, with attachment side (0 if none)
        int lo = std::min(j, k), hi = std::max(j, k); uint64_t h = 0x9E3779B97F4A7C15ULL; int cnt = 0;
        for (int i = std::max(0, lo - 2); i <= std::min(x.nw - 1, hi + 2); ++i) { if (i == j || i == k || x.cap[i] || answer_words.empty() || !answer_words[x.wstem[i] & 65535]) continue;
            uint64_t dj = x.wpos[i] > x.wpos[j] ? x.wpos[i] - x.wpos[j] : x.wpos[j] - x.wpos[i], dk = x.wpos[i] > x.wpos[k] ? x.wpos[i] - x.wpos[k] : x.wpos[k] - x.wpos[i];
            h = (h ^ x.wstem[i]) * 0x100000001b3ULL; h = (h ^ (dj <= dk ? 0x61u : 0x62u)) * 0x100000001b3ULL; ++cnt; }
        return cnt >= 2 ? (uint32_t)(mix64(h) >> 32) : 0;   // only informative when it says more than the nearest word
    }
    uint32_t span_rel(const Sent& x, int j, int k) const {   // j = from-name index, k = to-name index
        int lo = std::min(j, k), hi = std::max(j, k);
        // the relation word nearest the pair (between them, or within two thoughts outside)
        int best = -1; uint64_t bestd = ~0ULL;
        for (int i = std::max(0, lo - 2); i <= std::min(x.nw - 1, hi + 2); ++i) {
            if (i == j || i == k || x.cap[i] || answer_words.empty() || !answer_words[x.wstem[i] & 65535]) continue;
            uint64_t d = std::min(x.wpos[i] > x.wpos[j] ? x.wpos[i] - x.wpos[j] : x.wpos[j] - x.wpos[i], x.wpos[i] > x.wpos[k] ? x.wpos[i] - x.wpos[k] : x.wpos[k] - x.wpos[i]);
            if (d < bestd) { bestd = d; best = i; }
        }
        uint64_t h = 0x9E3779B97F4A7C15ULL;
        if (best >= 0) {
            uint64_t dj = x.wpos[best] > x.wpos[j] ? x.wpos[best] - x.wpos[j] : x.wpos[j] - x.wpos[best], dk = x.wpos[best] > x.wpos[k] ? x.wpos[best] - x.wpos[k] : x.wpos[k] - x.wpos[best];
            bool possessive = x.wpos[best] >= 3 && at(x.wpos[best] - 1) == ' ' && at(x.wpos[best] - 2) == 's' && at(x.wpos[best] - 3) == '\'';
            h = (h ^ x.wstem[best]) * 0x100000001b3ULL; h = (h ^ (possessive ? 0x51u : 0x52u)) * 0x100000001b3ULL; h = (h ^ (dj <= dk ? 0x61u : 0x62u)) * 0x100000001b3ULL;
        } else for (int i = lo; i < hi; ++i) { h = (h ^ x.between[i]) * 0x100000001b3ULL; if (i + 1 < hi) h = (h ^ x.w[i + 1]) * 0x100000001b3ULL; }
        return (uint32_t)(mix64(h) >> 32);
    }
    bool find_path(uint32_t a, uint32_t b) {
        path_len = 0; path_src = a; if (a == b) return false;
        struct Node { uint32_t w; int parent; uint32_t c; uint32_t c2; }; std::vector<Node> q; q.push_back({a, -1, 0, 0}); std::vector<uint32_t> seen = {a};
        for (size_t head = 0; head < q.size() && q.size() < 256; ++head) {
            uint32_t w = q[head].w; int depth = 0; for (int i = (int)head; i >= 0; i = q[i].parent) ++depth; if (depth > MAXP) break;
            for (int i = (int)sents.size() - 1; i >= 0; --i) { const Sent& x = sents[i]; if (x.question) continue;
                for (int j = 0; j < x.nw; ++j) { if (x.w[j] != w) continue;
                    for (int dir = 0; dir < 2; ++dir) {       // the nearest name to the right / to the left
                        int k = -1; for (int t = dir == 0 ? j + 1 : j - 1; t >= 0 && t < x.nw; t += dir == 0 ? 1 : -1) if (x.cap[t]) { k = t; break; }
                        if (k < 0) continue; uint32_t v = x.w[k]; if (std::find(seen.begin(), seen.end(), v) != seen.end()) continue;
                        uint32_t c = cls(span_rel(x, j, k), k > j); uint32_t f = span_full(x, j, k); uint32_t c2 = f ? cls(f, k > j) : 0; q.push_back({v, (int)head, c, c2}); seen.push_back(v);
                        if (v == b) { std::vector<uint32_t> rev, revn, rev2; for (int t = (int)q.size() - 1; t >= 0 && q[t].parent >= 0; t = q[t].parent) { rev.push_back(q[t].c); revn.push_back(q[t].w); rev2.push_back(q[t].c2); } if ((int)rev.size() > MAXP) return false; path_len = (int)rev.size(); for (int t = 0; t < path_len; ++t) { path_cls[t] = rev[path_len - 1 - t]; path_node[t] = revn[path_len - 1 - t]; path_cls2[t] = rev2[path_len - 1 - t]; } return true; } } } }
        }
        return false;
    }
    void init_learned_reading() { learn_reading = true; thought_u.assign(65536, 0.5f); pos_pref.assign(65536, 0.5f); }
    void init_chain() { chain_on = true; unary.assign(65536, Top2{}); comp.assign(65536, Top2{}); answer_form.assign(65536, Form{}); answer_words.assign(65536, 0); depth_score.assign((size_t)4096 * 2 * (MAXD + 2), 0.f); answer_score.assign(65536, 0.f); dir_score.assign(65536, 0.f); }
    static bool is_alpha(uint8_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
    float thought_df = 0.f;                                 // > 0: learned filter (document frequency over episodes) replaces the stop list
    struct DfEntry { uint32_t last_dlg = 0, count = 0; }; std::vector<DfEntry> word_df;
    void observe_word_df(const std::string& w) {
        if (word_df.empty()) word_df.assign(1 << 16, DfEntry{});
        DfEntry& e = word_df[word_hash(w) & 65535]; if (e.last_dlg != cur_dialogue) { e.last_dlg = cur_dialogue; ++e.count; }
    }
    bool is_thought_learned(const std::string& w) const {
        if (w.size() < 3 || word_df.empty()) return false;
        const DfEntry& e = word_df[word_hash(w) & 65535]; float df = cur_dialogue > 8 ? (float)e.count / (float)cur_dialogue : 1.f;
        return df < thought_df;
    }
    // ---- Learned reading (seq_chain=2): no stop list, no subject/object rule. Every word of >= 3 letters is a
    // candidate thought with a learned UTILITY (how often it linked a question to its answer, or was the answer);
    // a hop picks, among the words of the sentences relating the current thought, the one with the highest
    // utility x learned position preference (before/after the link, per question form) x recency; the question's
    // FORM is its words that do not occur in the story. All of it is credited from observed answers.
    bool learn_reading = false; std::vector<float> thought_u, pos_pref;   // [word bucket]; [(form, dir) bucket]
    float first_p[256]; bool first_p_valid = false; float surprise_rate = 0.05f; std::vector<float> word_surprise;   // [word bucket] EMA of the surprise a word carries
    void word_surprise_snapshot(const float* out, float total) { if (total <= 0.f) { first_p_valid = false; return; } for (int v = 0; v < 256; ++v) first_p[v] = out[v] / total; first_p_valid = true; }
    void word_surprise_update(const std::string& w, uint8_t first_byte) {   // utility <- EMA of (1 - p(first byte)); predictable words stop being thoughts
        if (!learn_reading || thought_u.empty() || !first_p_valid || w.size() < 3) return;
        if (word_surprise.empty()) word_surprise.assign(65536, 0.5f);
        float surprise = 1.f - first_p[first_byte]; float& u = word_surprise[word_hash(w) & 65535]; u += surprise_rate * (surprise - u);
    }
    float ws(uint32_t h) const { return word_surprise.empty() ? 0.5f : word_surprise[h & 65535]; }
    float chain_neg = 0.f;                                  // penalty on the words of a chain that did not answer (0 = none)
    int hop_dir[MAXD + 1] = {}; const Sent* hop_sent[MAXD + 1] = {}; int hop_j[MAXD + 1] = {}, hop_k[MAXD + 1] = {}; uint32_t chain_words[MAXD + 1] = {};
    int max_depth = 3;                                      // lever 3: the chain may walk up to this many hops (<= MAXD); the depth prior and the END slot decide
    float chain_conf = 0.f, abstain_below = 0.f; uint64_t chain_abstains = 0;   // learned abstention: the chain answers only forms it has been credited on
    // arbitration by consequence: per question form, how often the chain's pick was right vs how often the memory's own
    // prediction of the answer's first letter was right; the chain speaks only where it is the better of the two
    bool arbitrate = true; std::vector<float> form_chain_acc, form_mem_acc; int mem_first = -1; int pick_depth = 0;
    int mem_pred = -1; bool mem_ok = true, mem_seen = false;   // the memory's greedy prediction of each answer byte; right only if every byte is
    inline float& fca(uint32_t f) { if (form_chain_acc.empty()) { form_chain_acc.assign(4096, 0.f); form_mem_acc.assign(4096, 0.f); } return form_chain_acc[f & 4095]; }
    inline float& fma(uint32_t f) { fca(f); return form_mem_acc[f & 4095]; }
    int chain_pick() {                                      // the depth the chain would answer with (0 = none), its score in chain_conf
        if (!ncand) return 0; const float* sc = &depth_score[dkey()]; int cd = 1; float best = -1.f;
        for (int d = 1; d <= ncand; ++d) { float v = sc[d] + 0.1f * answer_score[akey(qform, cand[d].stem)]; if (v > best) { best = v; cd = d; } }
        { float v = sc[END] + 0.1f * answer_score[akey(qform, cand[ncand].stem)]; if (v > best) { best = v; cd = ncand; } }
        chain_conf = best; return cd;
    }
    float tu(uint32_t h) const { return thought_u.empty() ? 0.5f : thought_u[h & 65535]; }
    inline uint32_t pkey(uint32_t form, int dir, int first = 0, int last = 0) const { return (uint32_t)(mix64(((uint64_t)form << 8) | (uint64_t)(dir | (first << 1) | (last << 2))) & 65535); }
    int hop_first[MAXD + 1] = {}, hop_last[MAXD + 1] = {};
    bool is_thought(const std::string& w) const {
        if (learn_reading) { if (w.size() < 3 || w == "user" || w == "assistant") return false; uint32_t h = word_hash(w); return tu(h) > 0.15f && ws(h) > 0.15f; }   // credited by answers AND surprising in context
        return thought_df > 0.f ? is_thought_learned(w) : is_thought_listed(w);
    }
    static bool is_thought_listed(const std::string& w) { // a content word: not a function word, not a bare verb of motion/transfer
        static const char* STOP[] = {"user", "assistant", "there", "then", "back", "went", "moved", "journeyed", "travelled", "traveled", "picked", "grabbed", "took", "dropped", "discarded", "left", "down", "that", "this", "with", "from", "into", "what", "where", "which", "does", "many", "much", "either", "after", "before", "afterwards", "following", "gave", "handed", "passed", "received", "carrying", "fits", "inside", "than", "bigger", "smaller", "more", "less", "some", "most", "were", "are", "have", "has", "his", "her", "the", "and", "not", "also", "very", "how", "why", "who", "whom", "when", "been", "being", "will", "would", "could", "should", "over", "under", "above", "below", "north", "south", "east", "west", "color", "colour", "afraid"};
        static const char* STOP3[] = {"the", "and", "are", "was", "for", "his", "her", "its", "not", "but", "out", "off", "all", "any", "one", "two", "got", "has", "had", "did", "put", "who", "how", "why", "yes", "can", "may", "now", "you", "she", "him", "them", "than", "then", "too", "far", "few", "own", "way", "yet", "ago", "per", "via", "let", "get", "set", "see", "say", "they", "however", "when", "while", "since", "although", "though", "until", "unless", "because", "there", "their", "these", "those", "every", "each", "both", "other", "another"};
        if (w.size() < 3) return false; for (const char* x : STOP) if (w == x) return false; for (const char* x : STOP3) if (w == x) return false; return true;
    }
    bool is_answer_word(const std::string& w) const { return !answer_words.empty() && answer_words[stem_hash(w) & 65535]; }
    // ---- the composer's units as the chain's tokens (seq_units): a closing level-1 unit is a word for the
    // chain, a closing level-2 unit is a sentence. No alphabetic tokenizer and no punctuation rule: what counts
    // as a thought is what the organism found surprising enough to close a unit on.
    bool unit_tokens = false; bool unit_sentences = false; uint64_t units_seen = 0, unit_thoughts = 0; std::vector<std::string> unit_sample;
    void unit_close(const std::string& t, uint64_t start, bool self) {
        if (!chain_on || !unit_tokens || t.empty()) return;
        ++units_seen;
        { std::string low2; for (char c : t) low2 += (char)std::tolower((unsigned char)c);
          if (low2.find("user") != std::string::npos) csent.question = true;      // the turn markers are the stream's protocol, not a rule about words
          if (unit_sample.size() < 64) unit_sample.push_back(t); else unit_sample[units_seen % 64] = t; }
        size_t f = 0, e = t.size();
        while (f < e && !std::isalnum((unsigned char)t[f])) ++f;
        while (e > f && !std::isalnum((unsigned char)t[e - 1])) --e;
        // a piece that runs across a sentence end is split there: the end belongs to the sentence it closes
        size_t tpos = std::string::npos;
        for (size_t i = 0; i < t.size(); ++i) { const char c = t[i]; if (c == 46 || c == 63 || c == 33 || c == 10) { tpos = i; break; } }
        if (!unit_sentences && tpos != std::string::npos && tpos + 1 < t.size()) {
            unit_close(t.substr(0, tpos + 1), start, self);
            unit_close(t.substr(tpos + 1), start + tpos + 1, self);
            return;
        }
        const bool term = tpos != std::string::npos;
        if (e > f) {
            cword.clear(); for (size_t i = f; i < e; ++i) cword += (char)std::tolower((unsigned char)t[i]);
            cword_start = start + f; cword_cap = t[f] >= 65 && t[f] <= 90; cword_self = self;
            if (is_thought(cword)) ++unit_thoughts;
            chain_word_end();
        }
        if (term && !unit_sentences) chain_sentence_end();
    }
    void unit_sentence_end() { if (chain_on && unit_tokens && unit_sentences) chain_sentence_end(); }
    void chain_word_end() {
        if (thought_df > 0.f && cword.size() >= 3) observe_word_df(cword);
        if (learn_reading && !cword.empty() && !cword_self) word_surprise_update(cword, (uint8_t)at(cword_start));
        if (cword == "user") csent.question = true;
        if (!cword.empty() && !is_thought(cword) && cword.size() >= 2 && cword != "user" && cword != "assistant") between_acc = (between_acc ^ word_hash(cword)) * 0x100000001b3ULL + 0x9E37;   // a relation word
        if (csent.question && cword.size() >= 3 && !is_thought(cword) && cword != "user" && csent.nf < 4) csent.form[csent.nf++] = word_hash(cword);   // the question's form words ("where", "what", "color", "afraid")
        if (!csent.question && csent.nw >= 1 && !csent.rel && cword.size() >= 3 && !is_thought(cword)) csent.rel = word_hash(cword);   // the relation ("went", "dropped", "afraid")
        if (is_thought(cword) && csent.nw < NW) {
            if (csent.nw > 0) csent.between[csent.nw - 1] = (uint32_t)(mix64(between_acc ? between_acc : 1) >> 32); between_acc = 0;
            csent.cap[csent.nw] = cword_cap; csent.w[csent.nw] = word_hash(cword); csent.wstem[csent.nw] = stem_hash(cword); csent.wpos[csent.nw] = cword_start; csent.wlen[csent.nw] = (uint8_t)cword.size(); ++csent.nw; }
        cword.clear();
    }
    // ---- Relations as things, credited by prediction of the stream (seq_rel_gain > 0) ---------------------------
    // A triple (a, r, b): thought a, the words between, thought b, stored as sentences end (both directions). When
    // a thought recurs, its stored triples PROPOSE the next thought; the observed bytes credit the triple that was
    // right and debit the ones that were wrong — no answer turn needed, prose is self-supervised. In a reply the
    // user turn's thoughts (the frozen goal) propose too, credited by observed replies. When the stream states
    // (a, r3, c) while (a, r1, b) and (b, r2, c) are stored, that closure credits the composition r1 o r2 -> r3.
    struct Triple { uint32_t a = 0, b = 0, r = 0; float u = 0.f; uint64_t bpos = 0; uint8_t blen = 0; float g = 0.f; uint32_t c = 0; bool fwd = true; };   // g: EMA of the information it delivered when right; c: the path's class of this hop
    static constexpr int RSLOT = 8; std::vector<Triple> rel; float rel_gain = 0.f; float rel_rate = 0.1f; bool rel_closure = false;
    uint64_t rel_ingested = 0, rel_proposed = 0, rel_hits = 0, rel_closures = 0, rel_ambiguous = 0; double rel_gain_sum = 0.0;
    struct Prop { uint32_t b = 0; uint64_t bpos = 0; uint8_t blen = 0; int ti = -1; float w = 0.f; float pbase = 1.f; }; Prop props[16]; int nprops = 0;
    uint32_t last_thought = 0; std::string pword; bool in_word = false;   // the most recent completed thought; the current word so far
    void init_rel() { rel.assign((size_t)RSLOT << 16, Triple{}); }
    Triple* rel_slots(uint32_t a) { return &rel[(size_t)(a & 65535) * RSLOT]; }
    void rel_add(uint32_t a, uint32_t r, uint32_t b, uint64_t bpos, uint8_t blen, uint32_t c = 0, bool fwd = true) {
        Triple* t = rel_slots(a); int free_i = -1, weakest = 0;
        for (int i = 0; i < RSLOT; ++i) { if (t[i].a == a && t[i].b == b && t[i].r == r) { t[i].u += 0.05f * (1.f - t[i].u); t[i].bpos = bpos; t[i].blen = blen; if (c) { t[i].c = c; t[i].fwd = fwd; } return; } if (!t[i].a && free_i < 0) free_i = i; if (t[i].u < t[weakest].u) weakest = i; }
        int i = free_i >= 0 ? free_i : weakest; if (free_i < 0 && t[i].u > 0.3f) return;   // a strong relation is not evicted by a new one
        t[i] = {a, b, r, 0.3f, bpos, blen, 0.f, c, fwd}; ++rel_ingested;
    }
    void rel_ingest(const Sent& x) {                          // adjacent thoughts of a finished sentence, both directions; closure credit
        for (int j = 0; j + 1 < x.nw; ++j) { rel_add(x.w[j], x.between[j], x.w[j + 1], x.wpos[j + 1], x.wlen[j + 1]); rel_add(x.w[j + 1], x.between[j] ^ 0x80000000u, x.w[j], x.wpos[j], x.wlen[j]); }
        // name-to-name relations carry the path's own hop class (nearest relation word, possessive, attachment side)
        for (int j = 0; j < x.nw; ++j) { if (!x.cap[j]) continue; int k = -1; for (int t = j + 1; t < x.nw; ++t) if (x.cap[t]) { k = t; break; } if (k < 0) continue;
            rel_add(x.w[j], x.between[j] ^ 0x40000000u, x.w[k], x.wpos[k], x.wlen[k], span_rel(x, j, k), true); rel_add(x.w[k], x.between[j] ^ 0xC0000000u, x.w[j], x.wpos[j], x.wlen[j], span_rel(x, k, j), false); }
        // closure: any two NAME thoughts a, c of this sentence with a stored two-hop path (a, r1, b), (b, r2, c) — the
        // relation the sentence states between them (its span class) is what r1 o r2 composes to
        if (rel_closure && x.nw >= 2 && !comp.empty()) {   // credit an operator only on an UNAMBIGUOUS example: exactly one class pair over all stored paths a -> ? -> c
            for (int ia = 0; ia < x.nw; ++ia) { if (!x.cap[ia]) continue; uint32_t a = x.w[ia]; Triple* ta = rel_slots(a);
                for (int ic = 0; ic < x.nw; ++ic) { if (ic == ia || !x.cap[ic]) continue; uint32_t c = x.w[ic]; uint32_t r3 = span_word(x, ia, ic); if (!r3) continue;
                    uint32_t c_ab = 0, c_bc = 0, bnode = 0; int npairs = 0; bool ambiguous = false;
                    for (int i = 0; i < RSLOT && !ambiguous; ++i) { if (ta[i].a != a || !ta[i].b || ta[i].b == c || !ta[i].c) continue; Triple* tb = rel_slots(ta[i].b);
                        for (int k = 0; k < RSLOT; ++k) if (tb[k].a == ta[i].b && tb[k].b == c && tb[k].c) { uint32_t p1 = cls(ta[i].c, ta[i].fwd), p2 = cls(tb[k].c, tb[k].fwd);
                            if (npairs && (p1 != c_ab || p2 != c_bc)) { ambiguous = true; break; } c_ab = p1; c_bc = p2; bnode = ta[i].b; ++npairs; } }
                    if (ambiguous || npairs != 1) { if (ambiguous) ++rel_ambiguous; continue; }
                    uint32_t prefix = unary[c_ab & 65535].best(); if (!prefix) continue;   // the first hop must already have a name
                    comp[ckey(prefix, c_bc, 0)].credit(r3); ++rel_closures; } }
        }
    }
    void rel_propose() {                                      // at a word start: what the last thought's relations (and the goal's) say comes next
        nprops = 0; auto add = [&](uint32_t a, float scale) { Triple* t = rel_slots(a); for (int i = 0; i < RSLOT && nprops < 16; ++i) if (t[i].a == a && t[i].blen && n - t[i].bpos < window - 64) props[nprops++] = {t[i].b, t[i].bpos, t[i].blen, (int)(&t[i] - &rel[0]), t[i].u * t[i].g * scale}; };
        if (last_thought) add(last_thought, 1.f);
        if (topic_active && goal_is_frozen) for (uint32_t g : goal_frozen) { if (nprops >= 12) break; add(g, 0.5f); }
        if (nprops) ++rel_proposed;
    }
    void rel_credit(const std::string& w) {                   // a word finished (observed, not self): credit by INFORMATION — how far the proposal beat the base prediction
        if (!nprops) return; uint32_t h = word_hash(w); bool any = false;
        for (int i = 0; i < nprops; ++i) { Triple& t = rel[props[i].ti]; if (t.b == h) { float gain = 1.f - props[i].pbase; if (gain < 0.f) gain = 0.f; t.u += rel_rate * gain * (1.f - t.u); t.g += rel_rate * (gain - t.g); any = true; ++rel_hits; rel_gain_sum += gain;
                if (!thought_u.empty()) { float& ua = thought_u[t.a & 65535]; ua += rel_rate * gain * (1.f - ua); float& ub = thought_u[t.b & 65535]; ub += rel_rate * gain * (1.f - ub); } } }
        if (any) for (int i = 0; i < nprops; ++i) { Triple& t = rel[props[i].ti]; if (t.b != h) t.u -= rel_rate * 0.1f * t.u; }   // losers lose a little, only when someone won
        nprops = 0;
    }
    void chain_sentence_end() { chain_word_end(); if (csent.nw) { if (sents.capacity() < 65) sents.reserve(65); if (sents.size() >= 64) { sents.erase(sents.begin()); for (int d = 1; d <= MAXD; ++d) hop_sent[d] = nullptr; } sents.push_back(csent); if (rel_gain > 0.f && !csent.question) rel_ingest(sents.back()); } csent = Sent{}; csent.st = n + 1; between_acc = 0; }
    void chain_observe(uint8_t v, bool self) {
        if (rel_gain > 0.f) {
            if (is_alpha(v)) { if (!in_word) { in_word = true; pword.clear(); } pword += (char)std::tolower(v); }
            else { if (in_word) { in_word = false; if (!self) rel_credit(pword); else nprops = 0; if (is_thought(pword)) last_thought = word_hash(pword); } pword.clear(); rel_propose(); }   // proposals for the NEXT word, before its first letter
        }
        if (!unit_tokens) { if (is_alpha(v)) { if (cword.empty()) { cword_start = n; cword_cap = v >= 'A' && v <= 'Z'; cword_self = self; } cword += (char)std::tolower(v); }
        else { chain_word_end(); if (v == '.' || v == '?' || v == '!' || v == '\n') chain_sentence_end(); } }
        if (answer_pending) {                               // the reply's first word: credit the depth that produced it (observed outcomes only)
            if (!self && answer_off >= 0 && is_alpha(v)) { mem_seen = true; if (mem_pred != (int)v) mem_ok = false; }
            ++answer_off;
            if (is_alpha(v)) answer_word += (char)std::tolower(v);
            else if (!answer_word.empty()) {
                if (!self) { uint32_t as0 = stem_hash(answer_word); answer_words[as0 & 65535] = 1; answer_form[as0 & 65535] = {n - answer_word.size(), (int)answer_word.size()}; if (path_mode) credit_path(as0); }
                if (!self && arbitrate) {
                    const uint32_t as1 = stem_hash(answer_word); int pd = chain_pick();
                    if (pd) { float& ca = fca(qform); ca += chain_rate * ((cand[pd].stem == as1 ? 1.f : 0.f) - ca); }
                    if (mem_seen) { float& ma = fma(qform); ma += chain_rate * ((mem_ok ? 1.f : 0.f) - ma); }
                }
                if (!self && ncand && learn_reading) {
                    uint32_t as = stem_hash(answer_word); int hit_d = 0; for (int d = 1; d <= ncand; ++d) if (cand[d].stem == as) { hit_d = d; break; }
                    if (hit_d) {
                        for (int d = 0; d < hit_d; ++d) { float& u = thought_u[chain_words[d] & 65535]; u += chain_rate * (1.f - u); }   // the question thought and the link words
                        { float& u = thought_u[cand[hit_d].h & 65535]; u += chain_rate * (1.f - u); }                                       // the answer word
                        for (int d = 1; d <= hit_d; ++d) { float& pp = pos_pref[pkey(qform, hop_dir[d], hop_first[d], hop_last[d])]; pp += chain_rate * (1.f - pp);
                            if (hop_sent[d]) for (int k = 0; k < hop_sent[d]->nw; ++k) if (k != hop_j[d] && k != hop_k[d]) { float& pn = pos_pref[pkey(qform, k > hop_j[d], k == 0, k == hop_sent[d]->nw - 1)]; pn -= chain_rate * 0.5f * pn; } }   // the positions of the competitors lose
                        for (int d = 1; d <= hit_d; ++d) if (hop_sent[d]) for (int k = 0; k < hop_sent[d]->nw; ++k) if (k != hop_j[d] && k != hop_k[d]) { float& u = thought_u[hop_sent[d]->w[k] & 65535]; u -= chain_rate * 0.25f * u; }   // competitors in the answering sentences
                    } else if (hop_sent[1] && chain_neg > 0.f) { for (int k = 0; k < hop_sent[1]->nw; ++k) if (k != hop_j[1]) { float& u = thought_u[hop_sent[1]->w[k] & 65535]; u -= chain_rate * chain_neg * u; } }   // a chain that did not answer (off by default: on a mixed stream most turns are unanswerable by a chain, and this eroded exactly the words that answer)
                }
                if (!self && ncand) { uint32_t as = stem_hash(answer_word); float* sc = &depth_score[dkey()]; for (int d = 1; d <= ncand; ++d) { bool hit = cand[d].stem == as; sc[d] += chain_rate * ((hit ? 1.f : 0.f) - sc[d]); chain_hits[d] += hit; }
                    { bool hit = cand[ncand].stem == as; sc[END] += chain_rate * ((hit ? 1.f : 0.f) - sc[END]); chain_hits[END] += hit; }
                    float& a = answer_score[akey(qform, as)]; a += chain_rate * (1.f - a);
                    if (have_before && ncand >= 2) { float& dr = dir_score[akey(qform, rel1)]; if (cand_before.stem == as && cand_latest.stem != as) dr += chain_rate * (1.f - dr); else if (cand_latest.stem == as && cand_before.stem != as) dr += chain_rate * (-1.f - dr); }
                    ++chain_credits; }
                answer_pending = false; answer_word.clear(); ncand = 0; path_mode = false; path_len = 0;
            }
        }
        // reply boundary: the last 10 bytes are "ASSISTANT:" (the space after it is the reply's first byte, offset -1)
        if (v == ':' && n >= 10) { static const char* A = "ASSISTANT:"; bool m = true; for (int i = 0; i < 10 && m; ++i) m = at(n - 9 + i) == (uint8_t)A[i]; if (m) chain_select(); }
    }
    void chain_select() {
        ncand = 0; answer_pending = true; answer_word.clear(); answer_off = -1; chosen_depth = 0; role1 = 0; mem_first = -1; mem_pred = -1; mem_ok = true; mem_seen = false;
        const Sent* q = nullptr; for (int i = (int)sents.size() - 1; i >= 0; --i) if (sents[i].question) { q = &sents[i]; break; }
        if (!q) return;
        qform = 0; { uint64_t h = 0; for (int i = 0; i < q->nf; ++i) h = h * 1000003u + q->form[i]; qform = (uint32_t)mix64(h ? h : 1); }
        if (learn_reading) { uint64_t h = 0; for (int i = 0; i < q->nf; ++i) h = h * 1000003u + q->form[i];   // the question's non-thought words ("where", "what", "color") ...
            for (int i = 0; i < q->nw; ++i) { bool in_story = false; for (auto& x : sents) { if (x.question) continue; for (int j = 0; j < x.nw && !in_story; ++j) in_story = x.w[j] == q->w[i]; if (in_story) break; } if (!in_story) h = h * 1000003u + q->w[i]; }   // ... plus its thoughts absent from the story
            qform = (uint32_t)mix64(h ? h : 1); }
        auto occurrences = [&](uint32_t h, const Sent* excl) { int c = 0; for (auto& s : sents) { if (&s == excl || s.question) continue; for (int i = 0; i < s.nw; ++i) if (s.w[i] == h) { ++c; break; } } return c; };
        path_mode = false; path_answer = 0; path_len = 0;
        { uint32_t pa = 0, pb = 0; for (int i = 0; i < q->nw; ++i) if (q->cap[i] && occurrences(q->w[i], nullptr) > 0) { if (!pa) pa = q->w[i]; else if (!pb && q->w[i] != pa) pb = q->w[i]; }
          if (pa && pb && find_path(pa, pb)) { path_mode = true; ++paths; path_answer = compose_path(path_len); if (path_answer) ++path_answers;
              if (trace_on) { last_trace = "path(" + std::to_string(path_len) + " hops)"; for (int i = 0; i < path_len; ++i) last_trace += " c" + std::to_string(path_cls[i] % 1000); uint32_t pa1 = path_answer; last_trace += pa1 ? " => " + (answer_form[pa1 & 65535].len ? word_at(answer_form[pa1 & 65535].pos, answer_form[pa1 & 65535].len) : std::string("?")) : " => (unknown composition)"; }
              return; } }
        uint32_t t = 0; int best = 1 << 30;                   // the question's most specific thought
        for (int i = 0; i < q->nw; ++i) { int c = occurrences(q->w[i], nullptr); if (c > 0 && c < best) { best = c; t = q->w[i]; } }
        const uint32_t t0 = t;
        if (!t) { if (trace_on) last_trace = "(no thought in the story)"; return; }
        const Sent* used[MAXD + 1] = {};
        uint32_t ts = 0; { for (int i = 0; i < q->nw; ++i) if (q->w[i] == t) ts = q->wstem[i]; }
        uint32_t chain_w[MAXD + 1] = {t}; have_before = false; rel1 = 0; chain_words[0] = t; for (int d = 1; d <= MAXD; ++d) { hop_sent[d] = nullptr; chain_words[d] = 0; }
        // a sentence relating the thought: exact word first, stem ("wolf" ~ "wolves") as fallback; subject <-> object
        uint32_t arrived_rel = 0;                               // the relation of the sentence the current thought came from
        auto find = [&](uint32_t tw, uint32_t tstem, int d, uint64_t before, const Sent*& s_out, int& obj_out) {
            s_out = nullptr; obj_out = -1;
            if (learn_reading) {                                // utility x position preference x recency, over the sentences relating the thought
                float best_v = 0.f; int rank = 0;
                for (int pass = 0; pass < 2 && !s_out; ++pass)
                    for (int i = (int)sents.size() - 1; i >= 0; --i) {
                        const Sent& x = sents[i]; if (x.question || x.st >= before) continue; bool u = false; for (int k = 1; k < d; ++k) u |= used[k] == &x; if (u) continue;
                        int at_j = -1; for (int j = 0; j < x.nw; ++j) if (pass == 0 ? x.w[j] == tw : x.wstem[j] == tstem) { at_j = j; break; } if (at_j < 0) continue;
                        float rec = 1.f / (1.f + 0.5f * rank); ++rank;
                        for (int k = 0; k < x.nw; ++k) { if (k == at_j) continue; bool in_chain = false; for (int c = 0; c < d; ++c) in_chain |= x.w[k] == chain_w[c]; if (in_chain) continue;
                            float v = tu(x.w[k]) * (0.2f + pos_pref[pkey(qform, k > at_j, k == 0, k == x.nw - 1)]) * (0.2f + answer_score[akey(qform, x.wstem[k])]) * rec;
                            if (v > best_v) { best_v = v; s_out = &x; obj_out = k; hop_j[d] = at_j; hop_k[d] = k; hop_dir[d] = k > at_j; hop_first[d] = k == 0; hop_last[d] = k == x.nw - 1; hop_sent[d] = &x; } }
                    }
                return;
            }
            // pass 0/1: exact word / stem; role 0/1: the thought as subject (-> object) / as object (-> subject)
            for (int role = 0; role < 2 && !s_out; ++role)      // the thought as subject wins over the thought as object, exact or by stem
                for (int pass = 0; pass < 2 && !s_out; ++pass) {
                    const Sent* first = nullptr; int first_o = -1;   // latest admissible; an answer-like object wins over it
                    for (int i = (int)sents.size() - 1; i >= 0; --i) {
                        const Sent& x = sents[i]; if (x.question || x.st >= before) continue; bool u = false; for (int k = 1; k < d; ++k) u |= used[k] == &x; if (u) continue;
                        int at_j = -1; for (int j = 0; j < x.nw; ++j) if (pass == 0 ? x.w[j] == tw : x.wstem[j] == tstem) { at_j = j; break; } if (at_j < 0) continue;
                        if ((role == 0) != (at_j == 0)) continue;
                        int o = at_j == 0 ? x.nw - 1 : 0; if (o == at_j) continue;
                        bool in_chain = false; for (int k = 0; k < d; ++k) in_chain |= x.w[o] == chain_w[k]; if (in_chain) continue;
                        if (!first) { first = &x; first_o = o; }
                        if (role == 0 && answer_score[akey(qform, x.wstem[o])] > 0.05f) { s_out = &x; obj_out = o; break; }   // the relation that answers this form
                        if (role == 0 && arrived_rel && x.rel && x.rel != arrived_rel && !s_out) { s_out = &x; obj_out = o; }   // else: a different relation than the one arrived by (structural)
                    }
                    if (!s_out && first) { s_out = first; obj_out = first_o; }
                }
        };
        for (int d = 1; d <= max_depth; ++d) {
            const Sent* s = nullptr; int obj = -1; find(t, ts, d, ~0ULL, s, obj);
            if (d == 2 && used[1] && !learn_reading) {      // the second hop may look BEFORE the first sentence (the holder's state at a drop)
                const Sent* sb = nullptr; int ob = -1; find(t, ts, d, used[1]->st, sb, ob);
                rel1 = used[1]->rel; have_before = sb != nullptr && sb != s && s != nullptr;
                if (have_before) { cand_before = {sb->w[ob], sb->wpos[ob], sb->wlen[ob], sb->wstem[ob]}; cand_latest = {s->w[obj], s->wpos[obj], s->wlen[obj], s->wstem[obj]};
                    if (dir_score[akey(qform, rel1)] > 0.f) { s = sb; obj = ob; } }
            }
            if (!s) break;
            if (d == 1) role1 = learn_reading ? hop_dir[1] : (obj == 0 ? 1 : 0);   // the candidate is the subject <=> the thought was the object (learned mode: the hop's direction)
            arrived_rel = s->rel;
            cand[d] = {s->w[obj], s->wpos[obj], s->wlen[obj], s->wstem[obj]}; ncand = d; used[d] = s; t = s->w[obj]; ts = s->wstem[obj]; chain_w[d] = t; chain_words[d] = t;
        }
        if (ncand) ++chains;
        if (trace_on) { last_trace = "sents:"; for (auto& x : sents) { if (x.question) continue; last_trace += " {"; for (int j = 0; j < x.nw; ++j) last_trace += (j ? "," : "") + word_at(x.wpos[j], x.wlen[j]); last_trace += "}"; } last_trace += " thought="; for (int i = 0; i < q->nw; ++i) if (q->w[i] == t0) last_trace += word_at(q->wpos[i], q->wlen[i]); for (int d = 1; d <= ncand; ++d) last_trace += " d" + std::to_string(d) + "=" + word_at(cand[d].pos, cand[d].len); if (have_before && ncand >= 2) last_trace += " (alt d2=" + word_at(cand_before.pos, cand_before.len) + " dir " + std::to_string(dir_score[akey(qform, rel1)]).substr(0, 5) + ")"; if (!ncand) last_trace += " (no chain)"; }
    }
    // generation: follow the candidate at the depth this question form has learned to trust
    void rel_base(const float* out, float total) {          // at a word's first byte: what the organism predicted for each proposal's first letter before any proposal
        if (rel_gain <= 0.f || !nprops || !pword.empty()) return;
        for (int i = 0; i < nprops; ++i) props[i].pbase = total > 0.f ? out[at(props[i].bpos)] / total : 0.f;
    }
    bool rel_boost(float* out, float& total) {                // propose where UNCERTAIN: a confident continuation is never overridden, an open choice is informed
        if (rel_gain <= 0.f || !nprops) return false; bool any = false; int k = (int)pword.size();
        float pmax = 0.f; if (total > 0.f) { for (int v = 0; v < 256; ++v) pmax = std::max(pmax, out[v]); pmax /= total; }
        const float open = 1.f - pmax; if (open < 0.05f) return false;
        for (int i = 0; i < nprops; ++i) { const Prop& pr = props[i]; if (k >= pr.blen) continue; bool ok = true; for (int j = 0; j < k && ok; ++j) ok = (char)std::tolower(at(pr.bpos + j)) == pword[j]; if (!ok) continue;
            float w = rel_gain * pr.w * open * (1.f + total); out[at(pr.bpos + k)] += w; total += w; any = true; }
        return any;
    }
    bool chain_boost(float* out, float& total) {
        chain_forcing = false; if (!answer_pending) return false;
        if (path_mode) {
            if (!path_answer) return false; const Form& f = answer_form[path_answer & 65535]; if (!f.len || n - f.pos >= window - 64) return false;
            int k = answer_off; float w = topic_gain * (8.f + total); chain_forcing = true;
            if (k < 0) { out[(uint8_t)' '] += w; total += w; return true; }
            if (k < f.len) { out[at(f.pos + k)] += w; total += w; return true; }
            if (k == f.len) { out[(uint8_t)'\n'] += w; total += w; return true; }
            chain_forcing = false; return false;
        }
        if (!ncand) return false;
        if (!chosen_depth) { const float* sc = &depth_score[dkey()]; chosen_depth = 1; float best = -1.f;
            for (int d = 1; d <= ncand; ++d) { float v = sc[d] + 0.1f * answer_score[akey(qform, cand[d].stem)]; if (v > best) { best = v; chosen_depth = d; } }   // the (form, role) depth prior decides; the word's likelihood breaks ties
            { float v = sc[END] + 0.1f * answer_score[akey(qform, cand[ncand].stem)]; if (v > best) { best = v; chosen_depth = ncand; } }   // or: follow the chain to its end
            chain_conf = best;
            if (trace_on) last_trace += " chosen=d" + std::to_string(chosen_depth); }
        if (arbitrate ? (fca(qform) <= fma(qform)) : (abstain_below > 0.f && chain_conf < abstain_below)) { if (answer_off < 0) ++chain_abstains; return false; }   // this form is not the chain's to answer
        const Cand& c = cand[chosen_depth]; int k = answer_off;
        if (n - c.pos >= window - 64) return false;
        float w = topic_gain * (8.f + total);               // relative: the chain decides, the memory shades
        chain_forcing = true;                               // while the thought is being written the memory decides alone
        if (k < 0) { out[(uint8_t)' '] += w; total += w; return true; }
        if (k < c.len) { out[at(c.pos + k)] += w; total += w; return true; }
        if (k == c.len) { out[(uint8_t)'\n'] += w; total += w; return true; }
        chain_forcing = false; return false;
    }
    void topic_retrieve(float* out, float& total) {
        if (n < 8) return; ++topic_retrievals;
        if (!commit_dlg && !select_episode()) return;
        // follow the committed reply: the cursor must be aligned with what was just emitted
        int L = commit_cur == commit_st ? 10 : 4;               // the reply start aligns on "ASSISTANT:" itself
        if (!aligned_at(commit_cur, L) || commit_cur >= commit_en) {
            // re-align: the current context somewhere ahead in the committed reply (longest local match wins)
            uint64_t bq = 0; int bl = 0;
            for (int LL : {8, 6, 4}) { for (uint64_t q = commit_st + LL; q < commit_en; ++q) if (aligned_at(q, LL)) { bq = q; bl = LL; break; } if (bl) break; }
            ++commit_faults;
            if (bl) { commit_cur = bq; commit_miss = 0; ++realigns; }
            else { if (++commit_miss > 24) { commit_dlg = 0; ++commit_drops; if (goal_extend) { for (uint32_t h : goal) if (std::find(goal_frozen.begin(), goal_frozen.end(), h) == goal_frozen.end()) goal_frozen.push_back(h); } } return; }
        }
        float ww = topic_gain * commit_rel * 8.f; out[at(commit_cur)] += ww; total += ww; ++topic_candidates; ++commit_cur; ++commit_steps;
    }
    void observe_byte_for_bubbles(uint8_t v) {
        if (!ndialogues) return;
        if (std::isalpha(v)) { cur_word += (char)std::tolower(v); return; }
        if (cur_word.size() >= 4 && cur_word.size() <= 24) { uint32_t h = word_hash(cur_word); add_bubble(cur_dialogue, h); if (std::find(goal.begin(), goal.end(), h) == goal.end() && goal.size() < 96) goal.push_back(h); }
        cur_word.clear();
    }
    int goal_overlap(uint32_t dlg) const {               // how many goal bubbles this dialogue shares
        if (dlg == cur_dialogue || goal.empty()) return 0;
        const uint32_t* b = &dialogue_bubbles[(size_t)(dlg % ndialogues) * BUBBLES]; int c = 0;
        for (int i = 0; i < BUBBLES && b[i]; ++i) for (uint32_t g : goal) if (b[i] == g) { ++c; break; }
        return c;
    }
    // bubbling (UKS.Bubble.cs applied to continuations): a continuation shared by several remembered instances is
    // a class-level attribute of the context; a single instance is only that instance. A lag level is used when at
    // least min_support instances back it; after copy_max consecutive single-instance steps the memory falls back
    // to the longest consensus level, so fluency comes from what many instances agree on, not from one transcript.
    int min_support = 1, copy_max = 1 << 30, copy_run = 0, last_support = 0;
    uint64_t fallbacks = 0;
    AnswerAttention attn; bool attn_on = false;           // learned answer attention over the whole dialogue (attend.h)
    // Optional experimental interface, installed explicitly by a native caller.
    std::shared_ptr<reasoning::LanguageReasoner> reasoner;
    // A session owns automatic provenance and learned request dispatch. Its
    // callback replaces the direct reader call, so each byte is observed once.
    std::function<void(uint8_t,bool)> grounded_observer;

    // Checkpointed state of the instance window and consolidated tier. The chain,
    // relation, bubble, attention and attached-reader extensions are not covered;
    // the owner refuses to save when any of them is active.
    bool extensions_active() const { return chain_on || ndialogues || attn_on || reasoner || grounded_observer || !rel.empty(); }
    // Extensions a checkpoint cannot yet cover: answer attention, an attached reasoner/observer, the relation store.
    bool uncovered_extensions() const { return attn_on || reasoner || grounded_observer || !rel.empty(); }
    // Complete state of the dialogue (bubble) tier, the within-episode chain and the learned reading. The chain's
    // per-answer sentence pointers are saved as indices into the current episode's sentences (-2 = the sentence
    // being read, -1 = none) and restored as pointers.
    template <class Ar> void extension_state_io(Ar& ar) {
        ar.section("seq_dialogue");
        ar(chunk_dialogue); ar(dialogue_bubbles); ar(dialogue_start); ar(bubble_index); ar(bubble_df); ar(bubble_util); ar(util_rate);
        ar(ndialogues); ar(cur_dialogue); ar(topic_gain); ar(goal_hits); ar(goal_tests); ar(topic_retrievals); ar(topic_candidates);
        ar(cur_word); ar(goal); ar(goal_rel); ar(goal_expansions);
        ar.section("seq_chain");
        ar(between_acc); ar(cword_cap); ar(cword_self); ar(answer_words); ar(chain_on); ar(sents); ar(csent); ar(cword); ar(cword_start);
        ar.array(cand, MAXD + 1); ar(ncand); ar(depth_score); ar(answer_score); ar(qform); ar(role1);
        ar(answer_pending); ar(answer_word); ar(answer_off); ar(chosen_depth); ar(chains); ar(chain_credits); ar.array(chain_hits, MAXD + 2); ar(chain_rate);
        ar(last_trace); ar(trace_on); ar(dir_score); ar(cand_before); ar(cand_latest); ar(have_before); ar(rel1);
        ar(unary); ar(comp); ar(answer_form); ar.array(path_cls, MAXP); ar.array(path_cls2, MAXP); ar(path_len); ar(path_answer); ar(path_mode);
        ar(paths); ar(path_credits); ar(path_answers); ar.array(path_node, MAXP); ar(attr_on); ar(attr_hits); ar(attr_eq); ar(asig); ar(attr_merges);
        ar.section("seq_reading");
        ar(thought_df); ar(word_df); ar(learn_reading); ar(thought_u); ar(pos_pref); ar.array(first_p, 256); ar(first_p_valid); ar(surprise_rate); ar(word_surprise);
        ar(chain_neg); ar.array(hop_dir, MAXD + 1); ar.array(hop_j, MAXD + 1); ar.array(hop_k, MAXD + 1); ar.array(chain_words, MAXD + 1);
        {
            int32_t idx[MAXD + 1];
            // a pointer outside the current sentences is stale (the episode ended and its sentences were cleared); the
            // chain rebuilds these before any use, so a stale one is saved as none
            if constexpr (!Ar::loading) for (int i = 0; i <= MAXD; ++i) {
                const Sent* q = hop_sent[i];
                idx[i] = !q ? -1 : q == &csent ? -2 : (q >= sents.data() && q < sents.data() + sents.size()) ? (int32_t)(q - sents.data()) : -1;
            }
            ar.array(idx, MAXD + 1);
            if constexpr (Ar::loading) for (int i = 0; i <= MAXD; ++i) {
                if (idx[i] >= (int32_t)sents.size() || idx[i] < -2) throw std::runtime_error("state: chain sentence index out of range");
                hop_sent[i] = idx[i] == -1 ? nullptr : idx[i] == -2 ? &csent : &sents[(size_t)idx[i]];
            }
        }
        ar(max_depth); ar(chain_conf); ar(abstain_below); ar(chain_abstains); ar(arbitrate); ar(form_chain_acc); ar(form_mem_acc); ar(mem_first); ar(pick_depth);
        ar(mem_pred); ar(mem_ok); ar(mem_seen); ar.array(hop_first, MAXD + 1); ar.array(hop_last, MAXD + 1);
        ar(unit_tokens); ar(unit_sentences); ar(units_seen); ar(unit_thoughts); ar(unit_sample);
    }
    template <class Ar> void state_io(Ar& ar) {
        ar(window); ar(hash_bits); ar(ring); ar(self_bits); ar(n); ar(self_skipped); ar(self_repeats);
        for (int l = 0; l < NLAG; ++l) ar(table[l]);
        ar(pre_bucket); ar(pre_chash); ar(pre_n); ar(match_lag); ar(confidence); ar(committed_pos); ar(bubble_left); ar(bubbles); ar(bubble_bytes);
        for (int l = 0; l < NLAG; ++l) ar(ctable[l]);
        ar(cbits); ar(consolidate); ar(class_gain); ar(consolidated); ar(ctag_misses);
        ar(min_support); ar(copy_max); ar(copy_run); ar(last_support); ar(fallbacks); ar(wmask); ar(chain_forcing); ar(topic_active);
    }
    void init(uint32_t window_bytes, uint32_t bits) {
        window = window_bytes; hash_bits = bits; ring.assign(window, 0); self_bits.assign(window / 8 + 1, 0); n = 0; wmask = (window & (window - 1)) == 0 ? (uint64_t)window - 1 : 0;
        for (int l = 0; l < NLAG; ++l) table[l].assign((size_t)SLOT << hash_bits, 0);
    }
    uint64_t wmask = 0;                                    // window - 1 when the window is a power of two (the default): a mask instead of a division per byte
    inline uint8_t at(uint64_t pos) const { return ring[wmask ? (pos & wmask) : (pos % window)]; }
    // all eight lag hashes of the context ending just before end_pos in ONE backward pass (newest byte first):
    // lag L's raw hash is the running hash after L bytes, so the eight share the work instead of repeating it
    inline void lag_hashes(uint64_t end_pos, uint64_t* raw) const {
        uint64_t h = 0xcbf29ce484222325ULL; int li = 0;
        for (int i = 1; i <= LAGS[NLAG - 1] && li < NLAG; ++i) { h = (h ^ at(end_pos - i)) * 0x100000001b3ULL; if (i == LAGS[li]) raw[li++] = h ^ ((uint64_t)li << 56); }
    }
    inline uint32_t bucket_of(int li, uint64_t raw) const { return (uint32_t)(mix64(raw) >> (64 - hash_bits)); }
    inline uint32_t bucket(int li, uint64_t end_pos) const {   // hash of the LAGS[li] bytes ending just before end_pos
        uint64_t h = 0xcbf29ce484222325ULL;
        for (int i = 1; i <= LAGS[li]; ++i) h = (h ^ at(end_pos - i)) * 0x100000001b3ULL;
        return bucket_of(li, h ^ ((uint64_t)(li + 1) << 56));
    }
    inline bool is_self(uint64_t pos) const { uint64_t i = pos % window; return (self_bits[i >> 3] >> (i & 7)) & 1; }
    // capture one byte: it becomes step n; contexts ending here are indexed to point at this step.
    // Provenance discipline (Agkern): a self-generated byte is context for later matching but never an instance
    // that can be recalled, or the organism would recall its own output and loop on it.
    void capture(uint8_t v, bool self = false) {
        if (grounded_observer) grounded_observer(v, self);
        else if (reasoner) reasoner->observe(v, self);
        ring[n % window] = v;
        { uint64_t i = n % window; uint8_t m = (uint8_t)(1 << (i & 7)); if (self) self_bits[i >> 3] |= m; else self_bits[i >> 3] &= (uint8_t)~m; }
        if (consolidate && !self) consolidate_step(n, v);   // an observed outcome bubbles up into class counts; self-generated ones do not
        if (chain_on) chain_observe(v, self);
        if (attn_on) {
            if (v == ':' && answer_pending && answer_off == -1) {   // the chain has just selected: its pick (and confidence) is a proposal to the block
                attn.prop_h = 0; attn.prop_c = 0.f; attn.prop_s.clear();
                if (path_mode && path_answer) { const Form& fm = answer_form[path_answer & 65535]; if (fm.len) { attn.prop_s = word_at(fm.pos, fm.len); attn.prop_c = 1.f; } }
                else if (ncand) { int pd = chain_pick(); if (pd) { attn.prop_s = word_at(cand[pd].pos, cand[pd].len); attn.prop_c = chain_conf; } }
                if (!attn.prop_s.empty()) { std::string l; for (char c : attn.prop_s) l += (char)std::tolower((uint8_t)c); attn.prop_h = AnswerAttention::hash_tok(l); }
            } else if (v == ':') { attn.prop_h = 0; attn.prop_s.clear(); }
            attn.observe(v, self);
        }
        if (ndialogues) {
            chunk_dialogue[(n % window) / CHUNK] = cur_dialogue;
            observe_byte_for_bubbles(v);
            // a dialogue boundary: the delimiter "<|endofdialogue|>" has just completed
            if (v == '>' && n >= 16) { static const char* D = "<|endofdialogue|>"; bool ok = true; for (int i = 0; i < 17 && ok; ++i) ok = at(n - 16 + i) == (uint8_t)D[i]; if (ok) begin_dialogue(); }
        }
        if (n >= (uint64_t)LAGS[NLAG - 1]) {                     // the common case: one pass for all eight lags
            uint64_t raw[NLAG]; lag_hashes(n, raw);
            for (int l = 0; l < NLAG; ++l) { uint32_t* b = &table[l][(size_t)bucket_of(l, raw[l]) * SLOT]; for (int s = SLOT - 1; s > 0; --s) b[s] = b[s - 1]; b[0] = (uint32_t)(n + 1); }
        } else for (int l = 0; l < NLAG; ++l) {
            if (n < (uint64_t)LAGS[l]) continue;
            uint32_t* b = &table[l][(size_t)bucket(l, n) * SLOT];
            for (int s = SLOT - 1; s > 0; --s) b[s] = b[s - 1];   // most recent first
            b[0] = (uint32_t)(n + 1);
        }
        ++n;
        // the next predict's buckets: hash once here (one pass), prefetch the instance and tier lines before they are needed
        pre_n = n;
        if (n >= (uint64_t)LAGS[NLAG - 1] + 1) {
            uint64_t raw[NLAG]; lag_hashes(n, raw);
            for (int l = 0; l < NLAG; ++l) { pre_bucket[l] = bucket_of(l, raw[l]); _mm_prefetch((const char*)&table[l][(size_t)pre_bucket[l] * SLOT], _MM_HINT_T0);
                if (consolidate) { pre_chash[l] = chash_of(raw[l]); _mm_prefetch((const char*)&ctable[l][cbucket(pre_chash[l])], _MM_HINT_T0); } }
        } else for (int l = 0; l < NLAG; ++l) {
            if (n < (uint64_t)LAGS[l] + 1) { pre_bucket[l] = 0; pre_chash[l] = 0; continue; }
            pre_bucket[l] = bucket(l, n); if (consolidate) pre_chash[l] = chash(l, n);
        }
    }
    // sequence search: continuations of every remembered step whose context matches the current one
    // out[256] receives competition weights (unnormalised); returns the longest matching lag (0 = none)
    int predict(float* out, float kappa, float use_gain) {
        for (int v = 0; v < 256; ++v) out[v] = 0.f;
        int best_lag = 0; float total = 0.f; last_support = 0; int self_ctx = 0;
        const bool need_consensus = copy_run >= copy_max;      // copying an instance for too long: bubble up
        uint8_t tsym[16]; float tw[16]; int nt = 0;             // sparse scratch: at most SLOT + CSLOT entries per level
        auto add = [&](uint8_t sym, float w) { for (int i = 0; i < nt; ++i) if (tsym[i] == sym) { tw[i] += w; return; } if (nt < 16) { tsym[nt] = sym; tw[nt] = w; ++nt; } };
        const bool cached = pre_n == n;
        for (int l = NLAG - 1; l >= 0; --l) {
            if (n < (uint64_t)LAGS[l] + 1) continue;
            const uint32_t* b = &table[l][(size_t)(cached ? pre_bucket[l] : bucket(l, n)) * SLOT];
            float w_lag = (float)LAGS[l];                       // longer matches carry more weight
            nt = 0;
            float lt = 0.f; int support = 0, longest = 0; uint32_t first_pos = 0;
            for (int s = 0; s < SLOT; ++s) {
                if (!b[s]) break;
                uint64_t pos = (uint64_t)b[s] - 1;              // the step whose preceding context matched; its value is what followed
                if (pos >= n) continue;
                if (n - pos >= window - 32) continue;           // fell out of the window
                if (is_self(pos)) { ++self_skipped; if (LAGS[l] >= 8) ++self_ctx; continue; } // never recall one's own emission as an instance
                // verify the match (hash collisions) and extend it: longer real matches weigh more
                int L = match_len(pos);
                if (L < LAGS[l]) continue;
                float w = w_lag * (1.f + use_gain * (float)(L - LAGS[l])) * (s == 0 ? 1.f : 0.5f);
                add(at(pos), w); lt += w; ++support;             // NXT: what followed this context
                if (!longest) { longest = L; first_pos = (uint32_t)(pos + 1); }
            }
            // consolidated class counts for this context level: bounded, text-free, and they carry the consensus
            if (consolidate && n >= (uint64_t)LAGS[l]) {
                uint64_t h = cached ? pre_chash[l] : chash(l, n); const CEntry& e = ctable[l][cbucket(h)]; int csum = 0; if (e.tag == ctag(h)) for (int s = 0; s < CSLOT; ++s) csum += e.cnt[s];
                if (csum >= min_support) {   // the class has enough observations to speak at this level
                    // verify at least one instance (or trust the counts alone when the instance window has rolled past)
                    if (!support && csum < 2 * min_support) continue;
                    float w_lag2 = w_lag * class_gain;
                    for (int s = 0; s < CSLOT; ++s) if (e.cnt[s]) { float w = w_lag2 * (float)e.cnt[s] / (float)csum * (float)std::min(csum, 16); add(e.sym[s], w); lt += w; }
                    support = std::max(support, csum);
                    if (!longest) longest = LAGS[l];
                }
            }
            if (!support) continue;
            const int need = need_consensus ? std::max(min_support, 2) : min_support;
            if (support < need) { if (need_consensus && l > 0) continue; if (support < min_support) continue; }
            for (int i = 0; i < nt; ++i) out[tsym[i]] = tw[i]; total = lt; best_lag = longest; committed_pos = first_pos; last_support = support;
            break;                                              // the longest usable level decides
        }
        if (need_consensus && best_lag) ++fallbacks;
        if (chain_on) { rel_base(out, total); if (learn_reading && cword.empty()) word_surprise_snapshot(out, total);
            if (answer_pending && answer_off >= 0 && total > 0.f) { int b = 0; for (int v = 1; v < 256; ++v) if (out[v] > out[b]) b = v; mem_pred = b; if (answer_off == 0 && mem_first < 0) mem_first = b; } else mem_pred = -1; }
        if (ndialogues && topic_gain > 0.f) { if (!topic_active) topic_reset(); else { float before = total; chain_forcing = false; const bool ab = attn_on && attn.boost(out, total, topic_gain, chain_forcing); if (!ab && !(chain_on && chain_boost(out, total))) { topic_retrieve(out, total); if (chain_on) rel_boost(out, total); } if (total > before && !best_lag) best_lag = 4; } }   // goal-directed retrieval joins the competition
        if (reasoner) {
            // Re-evaluate ownership on every byte, including after the reply ends.
            const bool forced = topic_active && reasoner->boost(out, total);
            if (forced) { chain_forcing = true; best_lag = std::max(best_lag, 4); }
            else if (!(ndialogues && topic_gain > 0.f && topic_active)) chain_forcing = false;
        }
        // a single instance followed at long match length is copying; consensus resets the run
        if (best_lag >= 32 && last_support == 1) ++copy_run; else copy_run = 0;
        match_lag = best_lag;
        // don't repeat yourself: the current context (>= 8 bytes) already occurred in the organism's own output, so the
        // memory's grip loosens (higher semisaturation) and the bodies get to vary — provenance as an anti-loop signal
        if (self_ctx) { kappa *= (float)(1 + self_ctx); ++self_repeats; }
        confidence = total > 0.f ? total / (kappa + total) : 0.f;   // semisaturation: weak support -> weak influence
        return best_lag;
    }
};

} // namespace laminar
