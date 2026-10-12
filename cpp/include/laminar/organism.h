#pragma once
// The Laminar 200k organism: one native hot path per block (runbook §1).
//   input -> adaptive address walk -> lock verify -> admitted units -> L1 -> L2/L3
//         -> neighbour lock tests -> L5 -> L6 -> L4/exit -> local learning
#include <cstdint>
#include <cstdio>
#include <deque>
#include <vector>
#include <string>
#include <unordered_map>
#include <cctype>
#include <functional>
#include "config.h"
#include "rng.h"
#include "int4.h"
#include "address.h"
#include "telemetry.h"
#include "agency.h"
#include "sequence.h"
#include "state_io.h"

namespace laminar {

enum Source : uint8_t { SRC_SENSORY = 0, SRC_ROUTED = 1, SRC_PERSIST = 2 };   // SRC_PERSIST: a body still active from the previous byte (persist)

struct Event { uint32_t unit; uint8_t source; uint8_t hop; int16_t src_trace; }; // 8 bytes (runbook §7: <=16)

// Per traversed unit-event learning record: layer inputs (needed for local credit) and outputs.
struct Trace {
    uint32_t unit; uint8_t reached; uint8_t source; uint8_t hop; uint8_t consequential;
    int8_t x2[Dims::IN2], x3[Dims::IN3], x5[Dims::IN5], x6[Dims::IN6];
    int8_t l1[Dims::A], l2[Dims::B], l3[Dims::C], l5[Dims::E], l6[Dims::F];
    int8_t l6_pre[Dims::F]; uint8_t recirced; uint8_t emitted; int16_t src; int16_t srcs[4]; uint8_t nsrc;   // srcs: all senders (msg_merge)
    int8_t l4[Dims::R];   // route_return: a receiver's L4 output (its consequence, returned to its sender)   // src: the sender's trace index for a routed body (-1 otherwise)
    uint8_t sh[6];   // homeo: the right shift each layer used on this pass   // learned recirculation: the emission before the extra passes, for its counterfactual credit
};
enum Reached : uint8_t { R_L1 = 1, R_L23 = 2, R_L5 = 4, R_L6 = 8, R_L4 = 16 };

struct AcceptedMsg { uint32_t receiver, sender; };

class TaskModel;

// tiles (docs/PYRAMID_MEMORY.md): a tile; one stream's pyramid state; where one event's tile learning writes
struct Tile { float v[Dims::F]; float g[Dims::F]; uint64_t src = 0; uint32_t unit = UINT32_MAX; bool has_tr = false; float use = 0.f;
              uint64_t leaf[8]; uint8_t nleaf = 0; float fk[Dims::F]; uint64_t fw_t = 0; int8_t x1[Dims::IN]; Trace tr; };
    // leaf: the word tiles' close events (tile_avg credit); fk / fw_t: fast-weight key and write count; x1 / tr: a learned merge's trace
struct TileState {
    std::vector<std::deque<Tile>> q; std::vector<uint64_t> n; std::vector<Tile> wm; std::vector<uint8_t> wm_live;   // views, working memory
    float a[Dims::F] = {0};                                                      // this event's read, added to the readout's input
    std::vector<Tile*> att_slot; std::vector<float> att_k, att_al; float att_q[Dims::F] = {0}, att_r[Dims::F] = {0}, att_in[Dims::F] = {0};
    std::vector<float> fw_A; uint64_t fw_n = 0; float fw_q[Dims::F] = {0}, fw_r[Dims::F] = {0}, fw_in[Dims::F] = {0};
    float dq[Dims::F] = {0};                                                     // this event's query gradients (d loss / d readout input)
    // tile_ret: the episodic buffer (newest first), the last closed words' hashes, word frequencies (lazy decay), this event's read
    static constexpr int RET_D = 128;
    struct RetItem { float v[Dims::F]; float key[2][RET_D]; uint64_t src = 0; };   // a key per head
    std::deque<RetItem> ret; uint64_t rh[8] = {0}; uint32_t nrh = 0; std::vector<float> fq_n; std::vector<uint64_t> fq_t; uint64_t nw = 0;
    std::vector<float> ret_al; float ret_q[2][RET_D] = {{0}}, ret_r[2][Dims::F] = {{0}};   // ret_al: head-major (head x items)
};
struct TileLearn {   // weight gradients (the organism's own, or a worker's partials) and the ring (inject: that age's slot gets g x w where v is unclamped)
    float* gM = nullptr; float* gA = nullptr; float* gF = nullptr; float* gR = nullptr; float goal_w = 1.f; uint64_t event = 0;
    std::function<void(uint64_t age, const float* g, float w, const float* v)> inject;
};

class Organism {
    friend class TaskModel;   // the task interface (model_task.h) reads traces/workspace and applies task credit
    friend class MultiStream; // multi-stream trainer (multistream.h): shares this population across parallel streams
public:
    explicit Organism(const Params& p);

    // Process events s[0..n-1] whose outcomes are s[1..n]; the last symbol only serves as an outcome.
    void run_block(const uint8_t* s, size_t n);
    void set_mode(bool train, bool structural) { p.train = train; p.structural = structural && fm_base.empty(); }   // fixed motifs: never structural
    void end_block();                       // structural maintenance + telemetry snapshot
    BlockStats& stats() { return st; }
    const Params& params() const { return p; }
    uint64_t events_seen() const { return event_idx; }
    uint32_t block_index() const { return block_idx; }
    size_t live_units() const { size_t n = 0; for (auto& k : kstats) n += k.live; return n; }
    size_t free_count() const { return p.units - live_units(); }
    void open_trace(const std::string& path) { trace_fp = std::fopen(path.c_str(), "w"); }
    void open_log(const std::string& path) { log_fp = std::fopen(path.c_str(), "w"); }
    ~Organism() { if (trace_fp) std::fclose(trace_fp); if (log_fp) std::fclose(log_fp); }

    // exposed for tests / benchmarks
    AddressTree tree;
    uint8_t* weights() { return wbuf.p; }
    uint32_t alloc_unit();
    uint32_t alloc_unit(uint32_t u);        // initialise a specific free unit id
    void seed_fake_population(uint32_t n, uint32_t novel_fraction_pct = 0); // bench helper: n locks/units on random keys (runbook §11)

private:
    Params p;
    LayerDesc L[NUM_LAYERS];
    AlignedBuf wbuf;                          // units * 2816 bytes packed INT4
    std::vector<int8_t> acc;                  // INT8 plasticity residual per synapse (train only)
    std::vector<int16_t> acc16; bool wide_acc = false;   // acc_bits=16: INT16 residual (quantum x64, delta shift -6)
    // unit SoA (hot)
    std::vector<uint32_t> unit_lock, epoch, unit_k;
    std::vector<uint8_t> resist, alive;
    std::vector<uint8_t> act_ema;             // units * 4: running mean |activation| at L1, L2/3, L5, L6
    std::vector<uint64_t> assoc_value, assoc_mask;
    std::vector<int8_t> l4ret;                // units * R
    // unit SoA (cold)
    std::vector<uint32_t> assoc_noconseq, assoc_starve;
    std::vector<float> route_utility;         // per unit: EMA of bits its routed emission gained (counterfactual)
    std::vector<float> recirc_utility;        // per unit: EMA of bits its within-byte recirculation gained (counterfactual) — learned depth per step
    std::vector<float> arm_on, arm_off; std::vector<uint16_t> arm_non, arm_noff;   // recirc_learn=2: two-armed comparison of the ACTUAL loss with / without the tool
    uint64_t recirc_used = 0, recirc_explored = 0, hop_extra = 0;
    std::vector<uint32_t> free_units;
    // ---- motif packing and K screening ----
    std::vector<std::vector<uint32_t>> motif_free;   // free slots per motif (packed allocation)
    std::vector<uint32_t> fill_cursor;               // per K class: motif currently being filled
    std::vector<float> kscreen_tab;                  // [256 symbols][K classes]: EMA of consequence for bodies of that class
    struct KClassStats { uint64_t live = 0, emits = 0, consequential = 0, tests = 0, accepts = 0, births = 0; };
    std::vector<KClassStats> kstats;
    uint32_t n_kclass() const { return (uint32_t)p.k_set.size(); }
    uint32_t kclass_of(uint32_t u) const { return (u / p.motif_size) % n_kclass(); }
    uint32_t alloc_unit_in_class(uint32_t kc);
    uint32_t alloc_unit_near(uint32_t parent);
    uint32_t choose_kclass(uint8_t sym);
public:
    std::string kclass_report() const;
private:
    EvidencePool evid;
    // scheduler
    std::vector<Event> cur, nxt;
    std::vector<Trace> traces; uint32_t ntrace = 0;
    std::vector<AcceptedMsg> accepted;
    std::vector<uint32_t> executed;           // unit ids executed this event (for metabolic)
    // readout / workspace
    std::vector<float> wout;                  // [V][F] when V > F (row per symbol so the dot is contiguous)
    std::vector<float> woutT;                 // [F][V] transpose kept in sync: eight logits per 32 FMAs, no horizontal reductions
    std::vector<float> wreg;                  // reg_readout: [V][4] readout weights on the registers
    float greg_ro[4] = {0.f, 0.f, 0.f, 0.f};
    std::vector<uint32_t> fm_base;             // motifs: first unit of each fixed motif population
public:
    const std::vector<uint32_t>& motif_bases() const { return fm_base; }
    const std::vector<float>& route_utilities() const { return route_utility; }
    bool diag_zero_tiles = false;   // tiles: the view and working memory are not read (the pyramid's contribution, as an ablation)
    bool has_tiles() const { return !tst_.q.empty(); }
    const TileState& tile_state() const { return tst_; }
    std::function<void(const TileState&)> on_attend;   // diagnostics: called at the end of every tiles_attend (unset in training)
    void set_diag_zero_regs(bool v) { diag_zero_regs = v; }
    bool diag_zero_msgs_pub(bool v) { diag_zero_msgs = v; return v; }
private:
    float reg_mean[4] = {0.f, 0.f, 0.f, 0.f};   // reg_center: running mean of each register
    float reg_read(int r) const { return diag_zero_regs ? 0.f : p.reg_center ? ws_reg[r] - reg_mean[r] : ws_reg[r]; }
    uint32_t reg_pinned[4] = {0, 0, 0, 0}; int8_t reg_sign[4] = {0, 0, 0, 0}; uint64_t reg_revivals = 0;   // reg_revive  // reg_readout: d loss / d reg from the readout at this event
    std::vector<uint32_t> err_idx; std::vector<float> err_val; // sparse readout error support
    int32_t ws[Dims::F]; std::vector<float> logits, probs; float dl6[Dims::F]; int8_t q6[Dims::F];
    std::vector<float> cf_logits, cf_probs;   // counterfactual readout scratch (routed-emitter credit)
    bool cur_novel = false; uint32_t hist_target_ = 0; uint8_t cur_flags = 0;
    // stream state
    uint8_t hist[16] = {0}; uint64_t event_idx = 0; uint32_t block_idx = 0, cur_epoch = 1;
    int32_t ws_mem[Dims::F] = {0};            // persistent (decaying) workspace
    int8_t last_l4[Dims::R] = {0};            // most recent L4 return produced in this event
    float loss_ema = 0.f; bool loss_ema_init = false; uint32_t wrong_ema = 65535; // stream wrong-rate, fixed point
    std::vector<float> prior;                 // running next-symbol prior of the stream (for utility in bits)
    SequenceMemory seq; std::vector<float> seq_out, body_probs; bool seq_on = false;
    uint64_t seq_events = 0, seq_matched = 0, seq_bubble_events = 0; double seq_conf_sum = 0;
public:
    std::string seq_report() const;
    void attach_reasoner(std::shared_ptr<reasoning::LanguageReasoner> reader) {
        if (!seq_on) throw std::logic_error("reasoner requires byte sequence memory");
        seq.grounded_observer={};
        seq.reasoner = std::move(reader);
    }
    // The owner must provide the same reader used by the callback. Keeping the
    // session outside the core avoids coupling byte prediction to its learners.
    void attach_grounded_observer(std::shared_ptr<reasoning::LanguageReasoner> reader,
                                 std::function<void(uint8_t,bool)> observer) {
        attach_reasoner(std::move(reader));seq.grounded_observer=std::move(observer);
    }
    void seq_trace(bool on) { seq.trace_on = on; }
    const std::string& seq_chain_trace() const { return seq.last_trace; }
    std::string attn_decision() const { if (!seq.attn_on) return ""; char b[160]; std::snprintf(b, sizeof b, " attn=%s gate %.2f%s", seq.attn.dec_s.c_str(), seq.attn.dec_gate, seq.attn.dec_gate > 0.5f ? " SPEAKS" : ""); return b; }
private:
    void admit_children(int32_t slot, uint32_t parent_lock);
    void recycle_locks();
    Rng rng;
    KernelCtx kern;
    BlockStats st;
    std::vector<uint32_t> pending_splits;
    std::FILE* trace_fp = nullptr; std::FILE* log_fp = nullptr;
    int8_t xin[Dims::IN];
    int8_t xin_exact[Dims::IN];               // same input with the persistent-workspace cells zeroed (episodic bodies)
    int8_t xin_focus[Dims::IN];               // generic bodies' input after the agency kernel's focus gain
    // ---- agency kernel (attention directed by consequence-sensitive competition and commitment) ----
    agency::AgencyKernel agent; bool agent_on = false; std::vector<float> agent_obs;
    int agent_tendency = -1; bool agent_exact_only = false;
    float agent_last_ptarget = 0.f;
public:
    const agency::AgencyKernel& kernel() const { return agent; }
    std::string agency_report() const;
private:
    void agency_init();
    void agency_observe(uint64_t key, uint32_t ncand, bool novel, bool exact_matched);
    void agency_apply_focus();
    void agency_consequence(float p_target, bool correct, float loss);
public:
    // Task-level agency channel (p.agency_task). The task interface writes the public task state into
    // task_obs; a verified task outcome is posted with post_task_outcome and delivered to the kernel as the
    // consequence of the next processed event. Between outcomes the task channel reports nothing.
    float task_obs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool freeze_weights = false;   // control arm: body synapses never change (readout, routing utilities and agency still adapt)
    void post_task_outcome(float value) { task_value = value; task_reported = true; }
    // Cumulative accounting of snapshot (episodic) bodies: births, deferred births (surprises that have not yet recurred),
    // evictions, and evictions of bodies that were never hit again after birth (pure one-shots).
    struct EpisodicTotals { uint64_t births = 0, deferred = 0, evicted = 0, evicted_unused = 0, promoted = 0, retired = 0, promoted_splits = 0; };
    const EpisodicTotals& episodic_totals() const { return ep_totals; }
    // Body market (Params::market): cumulative accounting, the current admission bar, the retention controller, and a
    // census of the live non-backbone population (reuse and fitness quantiles).
    struct MarketTotals { uint64_t probation_pass = 0, probation_fail = 0, culled_general = 0, culled_snapshot = 0, cooldown_skips = 0, bar_raises = 0, bar_relaxes = 0, parsimony_raises = 0, gate_accepts = 0, gate_rejects = 0, contest_free = 0, contest_wins = 0, contest_losses = 0, throttled = 0; };
    const MarketTotals& market_totals() const { return mk_totals; }
    float market_bar() const { return mk_bar; }
    // Retention controller: regressed -> raise the bar; improved -> relax it; flat while the population grew -> raise it
    // (parsimony: growth must pay for itself in retention); flat without growth -> hold.
    void market_adjust(bool regressed, bool improved = true, bool grew = false);
    struct MarketCensus { uint32_t bodies = 0, snapshots = 0, probation = 0; float reuse_p10 = 0, reuse_p50 = 0, reuse_p90 = 0, fitness_p10 = 0, fitness_p50 = 0, fitness_p90 = 0; };
    MarketCensus market_census() const;
    // Counterfactual anti-regression gate (supplied by the caller, who owns the data): given a batch of lock ids the market
    // intends to release, return true when releasing them does not regress retention (e.g. loss on a retention sample
    // measured in forked children with and without the batch). A rejected batch releases only its unproven probationers.
    // Not state: set it again after constructing or loading a model.
    std::function<bool(const std::vector<uint32_t>&)> market_gate;
    // Releases a batch of market bodies (used by the gate's counterfactual child, and by the market itself).
    void market_cull(const std::vector<uint32_t>& ids);
private:
    float task_value = 0.f; bool task_reported = false;
    EpisodicTotals ep_totals;
    std::vector<uint32_t> recur_tag; std::vector<uint8_t> recur_cnt;   // pending surprises (episodic_recur > 1)
    bool recur_ready(uint64_t key);
    // Promotion bookkeeping, per unit of a snapshot body: its general parent (lock id, that lock's birth block to detect id
    // reuse, the parent's mask when promotion started), and whether it is promoted / due for retirement.
    struct SnapLink { uint32_t parent = UINT32_MAX, parent_born = 0; uint64_t parent_mask = 0; uint8_t promoted = 0, retire = 0; };
    std::vector<SnapLink> snap;
    MarketTotals mk_totals; float mk_bar = -1.f;           // -1: not yet initialised from Params
    std::vector<uint32_t> mk_last_struct, mk_born_hits;     // per lock id: block of its birth or last structural act; hits at birth
    std::vector<uint8_t> mk_state;                          // per lock id: 0 backbone / not in market, 1 on probation, 2 admitted
    std::vector<float> mk_util;                             // per lock id: shared utility (each event's gain divided among its admitting bodies)
    void market_new_lock(uint32_t id, bool enters_market);
    bool market_cooling(uint32_t id) const;
    float market_fitness(const Lock& lk, uint32_t id) const;
    void market_step();
    void learn_backprop();
    std::vector<float> ad_sh, ad_m, ad_v, ad_g; std::vector<uint32_t> ad_uses, ad_t, ad_last;   // adam: shadow, moments, gradient buffer; per body uses / steps
    void adam_grad(uint32_t u, uint32_t nib0, const int8_t* x, int in, const float* g, int out, float scale);
    void adam_touch(uint32_t u);
    void adam_step(uint32_t u, float n_uses);
    std::vector<float> ro_g, ro_m, ro_v, rg_g, rg_m, rg_v, hd_g, hd_m, hd_v, sg_g, sg_m, sg_v; uint64_t ro_t = 0, ro_n = 0; bool ro_pending = false;   // ro_adam: readout / register readout / head; ro_pending: tiles defer the step past tiles_learn
    void ro_adam_step();
    uint64_t sr_state = 0x9E3779B97F4A7C15ULL;               // stoch_round: xorshift state
    uint64_t sr_next() { sr_state ^= sr_state << 13; sr_state ^= sr_state >> 7; sr_state ^= sr_state << 17; return sr_state; }
    float sr_uniform() { return (float)(sr_next() >> 40) * (1.f / 16777216.f); }
    int32_t scale_sr(int32_t v, int32_t shift) {   // v / 2^shift: truncation toward zero, or (stoch_round) unbiased stochastic rounding
        if (!p.stoch_round) return v >= 0 ? (v >> shift) : -((-v) >> shift);
        const int32_t fl = v >> shift, frac = v - (fl << shift);   // arithmetic shift = floor
        return fl + ((int32_t)(sr_next() & ((1u << shift) - 1)) < frac ? 1 : 0);
    }
    std::vector<float> bp_msg;
    std::vector<int32_t> pc_d2, pc_d6, pc_dmsg;              // pc_settle: settle-step depth of each trace's L2 / L6 sites, and of its receivers' message error
    std::vector<int32_t> mslot; std::vector<std::array<int16_t, 4>> msrc; std::vector<uint8_t> mcount, mskip;   // msg_merge scratch
    const int16_t* cur_msrc = nullptr; uint8_t cur_mn = 0;                                                     // senders of the event being run
    void route_consequence();
    uint32_t shared_base = UINT32_MAX;
    float tc_greg[4] = {0, 0, 0, 0};
    float hw_g[4][Dims::F] = {}, hw_v[4][Dims::F] = {}, hb_g[4] = {0, 0, 0, 0}; bool head_init = false;   // ws_regs=2: the shared write head
    struct PStep { Trace tr; int8_t x1[Dims::IN]; };               // persist: one byte of a persistent body's life
    struct PChain { std::vector<PStep> steps; uint32_t born = 0; };
    std::unordered_map<uint32_t, PChain> pchain;                    // persist: unit -> its recent steps (oldest first)
    std::vector<uint32_t> persist_next;                             // persist: bodies carried into the next byte
    void persist_select();                                          // end of byte: compute L4 for emitters, choose who persists
    void persist_bptt(uint32_t t, const float* gret);               // credit through a persistent body's own recurrence                       // ws_regs: this event's readers' d loss / d register
    float body_resp = 1.f;
    struct TcEvent { std::vector<Trace> tr; std::vector<uint8_t> exact; int8_t x1[Dims::IN]; int8_t x1e[Dims::IN]; float tg[Dims::F]; float goal_w;
                     float reg_prev[4]; float G[4]; float tg_reg[4]; float ha[Dims::F]; float hg[4]; float hv[4]; float hu[4]; float hs = 1.f / 128.f; };   // ws_regs=2: head input, gates, values, pre-tanh   // ws_regs: registers before this event's writes, its total gates, d loss / d its writes
    std::vector<TcEvent> tc_ring; uint32_t tc_head = 0, tc_fill = 0;   // tc_window: the last writing events and their accumulated future error
    void temporal_credit(const float* gws);                // gws: this event's d loss / d ws_mem (all readers)
    void temporal_backprop(const TcEvent& ev);
    // tiles (the pyramid): the shared weights (mixings, attention, fast-weight projections) with gradients and Adam state; the
    // per-stream state is a TileState (this organism's own: tst_; the multi-stream trainer keeps one per stream)
    std::vector<uint32_t> tile_base;
    std::vector<float> tile_M, tile_Mg, tile_Mm, tile_Mv, tile_A, tile_Ag, tile_Am, tile_Av, fw_W, fw_Wg, fw_Wm, fw_Wv;
    std::vector<float> ret_W, ret_Wg, ret_Wm, ret_Wv;   // tile_ret: the read's output map (F x F)
    std::vector<float> rh_W1, rh_W1g, rh_W1m, rh_W1v, rh_b1, rh_b1g, rh_b1m, rh_b1v, rh_W2, rh_W2g, rh_W2m, rh_W2v;   // ro_hidden
    std::vector<float> wsp5, wsp5_g, wsp5_m, wsp5_v, dl5m; std::vector<int32_t> wsm5;   // ws_split5: the motifs' L5 channels
    void ro_hidden_input(const float* a, float* u) const;   // the hidden readout's input: the shared input, then every motif's channel
    void muon_step(uint32_t u, float n_uses);
    std::vector<float> comp_w;   // compete: each trace's emission weight this event
    std::vector<float> dis_w, dis_g, dis_m, dis_v;   // disinh: per motif [v (4 registers) | b], their gradients / moments
    float dis_release(uint32_t u, const int8_t* x1) const { const float* v = &dis_w[(size_t)motif_of(u) * 5]; float s = v[4];   // d in (0, 1)
        for (int r = 0; r < 4; ++r) s += v[r] * (float)x1[124 + r] * (1.f / 127.f); return 1.f / (1.f + std::exp(-s)); }
    std::vector<float> mix_th, mix_g, mix_m, mix_v, mix_pi;   // motif_mix: per-body gate logits over the earlier positions (units x 8), their
                                                            // gradients / moments; mix_pi: each trace's mixing weights this event (x 8)
    std::vector<float> att_W, att_Wg, att_Wm, att_Wv;   // motif_attn: [Wq 8 positions x 8 x A | Wk 8 x F | bias 8 x 8], shared by all bodies
    static constexpr size_t ATT_WK = 8 * 8 * Dims::A, ATT_B = ATT_WK + 8 * Dims::F, ATT_N = ATT_B + 64;
    void att_query(int m, const int8_t* l1, float* q) const { const float* W = &att_W[(size_t)m * 8 * Dims::A];
        for (int d = 0; d < 8; ++d) { float s = 0.f; for (int i = 0; i < Dims::A; ++i) s += W[d * Dims::A + i] * (float)l1[i]; q[d] = s * (1.f / 127.f); } }
    void att_key(const int8_t* l6, float* k) const { const float* W = &att_W[ATT_WK];
        for (int d = 0; d < 8; ++d) { float s = 0.f; for (int j = 0; j < Dims::F; ++j) s += W[d * Dims::F + j] * (float)l6[j]; k[d] = s * (1.f / 127.f); } }
    std::vector<float> wtl, wtl_g, wtl_m, wtl_v; float dl6t[Dims::F] = {0};   // tile_split: the tiles' own readout channel, its error
    std::vector<float> wsp, wsp_g, wsp_m, wsp_v, dl6m; std::vector<int32_t> wsm;   // ws_split: per-motif readout weights (V x M x F), the motifs'
    bool ws_writer(uint32_t u) const { return !p.ws_emit_max || motif_of(u) < (int)p.ws_emit_max; }   // ws_emit_max
    int motif_of(uint32_t u) const { int m = 0; for (size_t i = 1; i < fm_base.size(); ++i) if (u >= fm_base[i]) m = (int)i; return m; }   // own emissions, their errors
    uint32_t ret_heads() const { return p.tile_ret_heads >= 2 ? 2u : 1u; }
    bool ret_pos(uint32_t h) const { return ret_heads() == 1 ? p.tile_ret_pos != 0 : h == 0; }   // two heads: 0 positional, 1 a bag
    void ret_bag(const TileState& ts, float* out, uint64_t first, bool positional) const;   // tile_ret: the code of the last tile_ret_c closed words (first: a word
                                                                              // ending at this byte, not yet closed, goes first), unit length
    static uint64_t word_hash(const uint8_t* hist) { uint64_t h = 0xCBF29CE484222325ULL; for (int i = 1; i < 16 && std::isalnum(hist[i]); ++i) h = (h ^ (uint64_t)std::tolower(hist[i])) * 0x100000001B3ULL; return h; }
    TileState tst_;
    void tile_push(TileState& ts, uint32_t l, Tile&& t, uint64_t event); void tile_offer(TileState& ts, Tile&& t);
    void tile_merge(uint32_t l, const Tile& older, const Tile& newer, Tile& out, uint64_t event); void tile_backprop(Tile& t);
    void fw_write(TileState& ts, Tile& t) const;
public:
    // the pyramid's per-event steps on one stream's state (Organism on tst_, the multi-stream trainer on each stream's)
    void tiles_read(TileState& ts) const;                                     // view + working memory, linearly, into ts.a (event start)
    void tiles_attend(TileState& ts, const int32_t* ws, const uint8_t* hist = nullptr) const;   // hist: tile_ret's query sees a word ending at this byte                // attention / fast-weight reads cued by this event (before the readout)
    void tiles_learn(TileState& ts, const float* dl6, TileLearn& tl) const;   // the reads' gradients (dl6: d loss / d readout input); query grads -> ts.dq
    void tiles_close(TileState& ts, const int32_t* ws, const uint8_t* hist, uint64_t event);   // a word ended: its tile, parents, working memory
    const std::vector<float>& tile_weights_M() const { return tile_M; }
    const std::vector<float>& motif_mix_logits() const { return mix_th; }   // motif_mix gate logits (units x 8; empty when off)
    const std::vector<float>& motif_mix_weights() const { return mix_pi; }  // this event's mixing weights per trace (x 8; zero: not mixed)
    const std::vector<float>& motif_attn_weights() const { return att_W; }  // motif_attn: [Wq | Wk | bias] (empty when off)
private:
    void mtp_learn(uint8_t target);                        // mtp: the extra heads' loss on this event's target, errors into earlier ring slots
    std::vector<float> mtp_W, mtp_g, mtp_m, mtp_v, mtp_p;   // mtp: head weights [h][v][F] (Adam with the readout; not saved), scratch
public:
    double mtp_loss[8] = {0}; uint64_t mtp_n[8] = {0};    // mtp: summed loss (nats) and count per head since construction
private:
    static constexpr uint32_t EMB_ROWS = 65536, EMB_D = 16;
    std::vector<float> emb;                                // in_words: learned word embeddings (EMB_ROWS x EMB_D)
    uint64_t word_cur = 0; uint32_t word_ids[3] = {0, 0, 0}; // in_words: running hash of the word in progress; the last 3 completed words                                 // seq_linear: the bodies' responsibility for the target this byte                     // shared_types: first shared unit id                              // route_return: receivers' L4 -> senders, senders recompute (see config)
    std::vector<int32_t> rc_acc; std::vector<uint16_t> rc_n; std::vector<float> rr_gret;                             // bp_route: per-trace gradient at the body's L2 from its receivers                                 // hidden=backprop: exact within-body gradient (comparator for the local rules)
public:
    uint64_t bp_outputs = 0, bp_saturated = 0;             // hidden=backprop: hidden outputs seen / at the clamp (zero derivative)
private:
    bool market_make_room(float bid);                      // contest birth: true when a unit is available for a newborn with this bid
    float market_reuse(const Lock& lk, uint32_t id) const; // hits per block since birth
    std::vector<std::pair<float, uint32_t>> mk_queue; size_t mk_qpos = 0;
    std::vector<uint8_t> ushift; std::vector<float> hsat, hmag;   // homeo: per unit x layer right shift; homeo=2 running saturation / magnitude
    float ws_reg[4] = {0.f, 0.f, 0.f, 0.f}, ws_reg_prev[4] = {0.f, 0.f, 0.f, 0.f}, ws_regG[4] = {0.f, 0.f, 0.f, 0.f};   // ws_regs
    float reg_gate(int8_t y) const { return 1.f / (1.f + std::exp(-((float)y / 32.f + p.reg_bias))); }
    float ws_scale = 0.f; float ws_bscale[4] = {0.f, 0.f, 0.f, 0.f};   // ws_bands: per-band normaliser
    int32_t ws_decay_of(int j) const { static const int32_t D[4] = {128, 224, 248, 254}; return p.ws_bands ? D[j >> 3] : p.ws_decay; }
    static constexpr int SEQ_GATE_F = 6; float seq_gw[SEQ_GATE_F] = {-0.4f, 0.f, 0.f, 0.f, 0.f, 0.f};   // seq_gate weights (bias starts at a ~ 0.4)                                   // in_struct: running mean |ws_mem| (the workspace input's normaliser)                            // homeo: per unit x layer right shift
    uint64_t mk_tokens = UINT64_MAX, mk_pass_mark = 0;     // birth throttle: births left this batch; passes at the last batch   // contest: admitted bodies, weakest first (rebuilt per batch)
    void market_release(uint32_t id);
    std::vector<uint32_t> promo_hold;   // per lock id: > 0 while its evidence is held open for promotion (not released as unconflicted)
    void promote_snapshots(uint64_t key, uint32_t target, const uint32_t* cand, uint32_t ncand);
    void retire_promoted();

    // ---- hot path ----
    void process_event(uint8_t sym, uint8_t target);
    void run_pass(uint64_t key, bool novel, int sign, const std::vector<Event>* replay);
    void run_body(const Event& ev, bool novel, int sign, uint32_t sched_pos);
    void perturb(int8_t* v, int n, uint32_t unit, int layer, int sign);
    int32_t threshold(int32_t base, uint32_t unit, bool novel, int stage, int32_t level);
    uint64_t make_key() const;
    void make_input(uint64_t key, uint8_t sym);
    uint64_t route_key(const int8_t* l2) const;
    void readout(const int32_t* w, float* out_logits, float* out_probs) const;
    // ---- learning ----
    void learn_l6(uint8_t target);
    void learn_hidden(int32_t dq, const std::vector<Trace>& trA, uint32_t nA);
    void learn_modulated(int32_t M);
    // The same local rules applied to stored traces (task credit): an L6 output error in workspace space,
    // and the three-factor modulated rule with an externally supplied modulator.
    void apply_l6_error(const Trace* tr, uint32_t n, bool novel);
    void modulated_on(const Trace* tr, uint32_t n, const int8_t* x1, int32_t M, bool novel);
    void apply_acc(uint32_t unit, uint32_t nib, int32_t delta);
    void apply_acc_block(uint32_t unit, uint32_t nib0, const int32_t* delta, int n);
    void apply_acc16(uint32_t unit, uint32_t nib, int32_t delta);
    void apply_acc_block16(uint32_t unit, uint32_t nib0, const int32_t* delta, int n);
    std::vector<int32_t> delta_buf = std::vector<int32_t>(Dims::P1);
    void metabolic(bool consequential, bool novel);
    void assoc_update(bool novel, bool consequential);
    // ---- structural ----
    void birth_unit_weights(uint32_t unit);
    void free_unit(uint32_t unit);
    void commit_splits();
    void episodic_birth(uint64_t key, uint32_t parent_lock);
    void episodic_evict();
public:
    std::vector<uint8_t> preds;               // per-event predictions when record_preds
    std::function<void(uint8_t sym, uint8_t target, const int32_t* ws, const float* probs)> on_event;
    // ---- goal-credited plasticity and motif specialisation (reply-turn protocol: "USER:" ... "ASSISTANT:" answer)
    int phase = 0, ans_state = 0; float goal_w = 1.f; double ans_loss = 0.0; int ans_n = 0; float ans_ema = -1.f;
    std::vector<float> elig; std::vector<uint64_t> elig_t; std::vector<uint32_t> touched; std::vector<uint64_t> touch_stamp; uint64_t reply_id = 1;
    std::vector<uint32_t> m_emit_q, m_emit_a, m_emit_o; uint64_t bytes_q = 0, bytes_a = 0, bytes_o = 0, goal_credits = 0; double goal_R_sum = 0.0;
    std::vector<uint8_t> silenced;            // per motif: emission suppressed (ablation)
    // ---- composer: units above bytes, each level with its own bodies ----
    static constexpr int MAXLV = 4;
    struct Level {
        AddressTree tree; std::vector<uint32_t> pool;           // this level's own address space and unit pool (reserved motifs)
        std::vector<Trace> tr; int32_t ws[Dims::F] = {0}, mem[Dims::F] = {0}; float acc[Dims::F] = {0}; uint32_t acc_n = 0;
        uint64_t cur = 0; uint32_t len = 0; uint64_t ids[3] = {0, 0, 0}; float hema = -1.f; std::string cur_str;
        struct Next { uint64_t id[4] = {0, 0, 0, 0}; uint16_t n[4] = {0, 0, 0, 0}; }; std::vector<Next> next;   // [bucket of id] -> next ids (this level's predictor)
        uint64_t events = 0, units_len = 0, births = 0; std::vector<std::string> sample;
        std::unordered_map<uint64_t, uint32_t> pair_n;          // how often unit a was followed by unit b
        std::unordered_map<uint64_t, uint32_t> merge;           // (a, b) -> the unit that stands for them: the learned vocabulary
        std::vector<uint32_t> ch_a, ch_b, standalone; std::vector<uint16_t> ulen;   // the merge tree: a unit's two parts, how often each part occurs on its own, and its length
        struct Pend { uint32_t id; uint64_t start; uint32_t len; std::string str; };
        std::vector<Pend> pend;                                 // the merge window: merges apply by rank, not by arrival
        uint32_t next_id = 256, since_round = 0, rounds = 0;
        uint64_t from_lex = 0, from_surprise = 0;
    };
    Level lv[MAXLV + 1]; uint64_t n_bytes_seen = 0; int nlv = 0; int cur_level = 0; std::vector<uint8_t> unit_level; int silenced_level = -1;
    std::vector<float> lvl_w;                 // [level][V][F] each level's own readout weights
    std::vector<float> lvl_logits;            // this byte's level contribution to the logits (constant within the byte)
    float prev_s = 0.f;
    void composer_init();
    void composer_step(uint8_t sym, float pmax);
    void feed_unit(int l, uint32_t id, uint64_t start, uint32_t len, const std::string& text);
    void emit_pieces(Level& L, uint32_t id, const std::string& str, uint32_t off, uint64_t start, int depth);
    void close_unit(int l, uint64_t id, uint32_t raw_id, const std::string& str, uint64_t start, uint32_t len);
    void level_event(int l, uint64_t id);
    void level_learn(Level& L);
public:
    std::string composer_report() const;
    void silence_level(int l) { silenced_level = l; }
private:
    void goal_track(uint8_t sym, uint8_t target);
    void goal_credit(float R);
public:
    std::string goal_report() const;
    std::vector<uint32_t> specialised_motifs(int k, bool random, uint64_t seed) const;   // most question/answer-specialised motifs (or k random live ones)
    void silence_motifs(const std::vector<uint32_t>& m) { silenced.assign(p.units / p.motif_size + 1, 0); for (uint32_t x : m) if (x < silenced.size()) silenced[x] = 1; }
    // diagnostics (valid inside on_event): every body executed this byte, its readout without the sequence memory, its kind
    std::function<void(uint32_t unit, int8_t* l6)> l6_hook;
    std::function<void(uint32_t unit, int layer, int8_t* y)> layer_hook;
    bool diag_no_routing = false, diag_zero_msgs = false, diag_zero_return = false, diag_routed_silent = false, diag_no_consequence = false, diag_zero_wsin = false, diag_zero_regs = false;   // diagnostics: no neighbour routing / L3 sees no neighbour message / no L4 return   // diagnostics: may rewrite a hidden layer's output (LI_1/2/3/5)   // diagnostics: may rewrite a body's L6 emission before it enters the workspace
    template<class F> void for_each_trace(F f) const { for (uint32_t t = 0; t < ntrace; ++t) f(traces[t]); }
    float body_loss(const int32_t* w, uint8_t target) const { std::vector<float> lg(p.vocab), pr(p.vocab); readout(w, lg.data(), pr.data()); return -std::log(std::max(pr[target], 1e-9f)); }
    // diagnostics: nibbles of layer li that differ from the unit's birth weights (meaningful for bodies born fresh, not copies)
    uint32_t weights_changed(uint32_t u, int li) const;
    bool unit_live(uint32_t u) const { return alive[u] != 0; }
    int32_t* ws_memory() { return ws_mem; }
    float& ws_register(int r) { return ws_reg[r]; }
    float diag_reg_target = 0.f; bool diag_reg_has_target = false;   // diagnostics: a perfect error for register 0 at this byte (writer unit test)   // diagnostics: oracle register writer   // diagnostics: the persistent workspace (e.g. an oracle writer)
    uint32_t unit_hits(uint32_t u) const { return unit_lock[u] != UINT32_MAX ? tree.locks[unit_lock[u]].hits : 0; }
    bool unit_copied(uint32_t u) const { return unit_lock[u] != UINT32_MAX && tree.locks[unit_lock[u]].mask != SENSORY_MASK_; }
    static constexpr uint64_t SENSORY_MASK_ = (1ULL << Dims::SENSORY_BITS) - 1;
    bool exact_unit(uint32_t u) const { return unit_lock[u] != UINT32_MAX && tree.locks[unit_lock[u]].mask == ~0ULL; }
    template<class F> void for_each_emitter(F f) const { for (uint32_t t = 0; t < ntrace; ++t) if (traces[t].emitted) f(traces[t].unit, traces[t].l6); }   // valid inside on_event: the bodies that wrote to the workspace this byte   // per-byte observer (hybrid heads); unset = no cost
    // free-running generation: process an event with a placeholder outcome, read the distribution, then commit
    // the chosen byte as what actually followed (it becomes the next event's current symbol)
    const std::vector<float>& distribution() const { return probs; }
    bool free_run = false;                    // when set, process_event does not capture the placeholder outcome
    bool replay = false;                      // re-encoding stored situations: bytes are processed but never captured into sequence memory (transient, never set at a checkpoint)
    void commit_outcome(uint8_t sym) { hist[0] = sym; if (seq_on) seq.capture(sym, /*self=*/true); }
    int last_match_length() const { return seq_on ? seq.match_lag : 0; }   // verbatim match behind the last prediction (copy metric)
    uint8_t last_symbol() const { return hist[0]; }
    std::vector<uint8_t> flags;               // per-event: 1 = exact lock admitted, 2 = episodic birth, 4 = imprint decoded at birth
    enum { FL_EXACT = 1, FL_BIRTH = 2, FL_IMPRINT_OK = 4 };
    size_t exact_locks() const { size_t n = 0; for (auto& l : tree.locks) n += l.alive && l.mask == ~0ULL; return n; }
private:
    void neighbor_iter(uint32_t u, uint32_t k, uint32_t* out, uint32_t& n) const;
public:
    // Complete learning state of the task profile: bodies, plasticity residuals, locks and address tree,
    // routing and recirculation utilities, readout, workspace, stream state, sequence memory (instance
    // window and consolidated tier), agency kernel, goal credit and RNG. Throws for configurations whose
    // extensions are not covered (composer levels, chain/relation/bubble memory, answer attention, reader).
    void save_state(state::Writer& w);
    void load_state(state::Reader& r);
    uint64_t state_fingerprint();
    template <class Ar> void state_io(Ar& ar);
};

} // namespace laminar
