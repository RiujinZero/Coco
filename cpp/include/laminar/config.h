#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cstdlib>

namespace laminar {

// ---------------------------------------------------------------------------
// Fixed unit geometry (design doc §2, tensor-friendly rounding agreed with user).
//
//   L1 : IN(128)            -> A(16)          2048
//   L2 : A + R2  (32)       -> B(24)           768
//   L3 : A + B   (40)       -> C(16)           640   (B = neighbour L2 message)
//   L5 : B + C   (40)       -> E(16)           640
//   L6 : E + R6  (24)       -> F(32)           768
//   L4 : F(32)              -> R2 + R6 (24)    768   (return to L2 and to L6)
//                                             ----
//                                             5632 INT4 synapses = 2816 bytes = 44 cache lines
//
// Weights are stored [out][in] row-major so a dot product walks contiguous nibbles.
// Layers are laid out in execution order (L1, L2, L3, L5, L6, L4) so an admitted unit
// that dies early touches only the leading cache lines of its 2816-byte block.
// ---------------------------------------------------------------------------
struct Dims {
    static constexpr int IN = 128, A = 16, B = 24, C = 16, E = 16, F = 32, R2 = 16, R6 = 8;
    static constexpr int R = R2 + R6;                 // 24 : L4 output
    static constexpr int IN2 = A + R2;                // 32
    static constexpr int IN3 = A + B;                 // 40
    static constexpr int IN5 = B + C;                 // 40
    static constexpr int IN6 = E + R6;                // 24
    static constexpr int IN4 = F;                     // 32
    static constexpr int P1 = IN * A, P2 = IN2 * B, P3 = IN3 * C, P5 = IN5 * E, P6 = IN6 * F, P4 = IN4 * R;
    static constexpr int NIB_L1 = 0;
    static constexpr int NIB_L2 = NIB_L1 + P1;
    static constexpr int NIB_L3 = NIB_L2 + P2;
    static constexpr int NIB_L5 = NIB_L3 + P3;
    static constexpr int NIB_L6 = NIB_L5 + P5;
    static constexpr int NIB_L4 = NIB_L6 + P6;
    static constexpr int UNIT_NIBBLES = NIB_L4 + P4;          // 5632
    static constexpr int UNIT_BYTES = UNIT_NIBBLES / 2;       // 2816
    static_assert(UNIT_BYTES % 64 == 0, "unit must be cache-line sized");
    static constexpr int KEY_BITS = 64;
    static constexpr int SENSORY_BITS = 8;                    // bits 0..7 = current symbol
    static constexpr int TUMBLER_BITS = KEY_BITS - SENSORY_BITS; // 56 candidate context tumblers
};

enum LayerId { LI_1 = 0, LI_2 = 1, LI_3 = 2, LI_5 = 3, LI_6 = 4, LI_4 = 5, NUM_LAYERS = 6 };

struct LayerDesc {
    uint32_t nib_off; uint16_t in; uint16_t out; uint8_t shift;
};

// shift = right shift applied to the INT32 dot product before clamping to INT8.
inline const LayerDesc& layer_desc(int li, const uint8_t* shifts) {
    static LayerDesc d[NUM_LAYERS];
    d[LI_1] = {Dims::NIB_L1, Dims::IN, Dims::A, shifts[0]};
    d[LI_2] = {Dims::NIB_L2, Dims::IN2, Dims::B, shifts[1]};
    d[LI_3] = {Dims::NIB_L3, Dims::IN3, Dims::C, shifts[2]};
    d[LI_5] = {Dims::NIB_L5, Dims::IN5, Dims::E, shifts[3]};
    d[LI_6] = {Dims::NIB_L6, Dims::IN6, Dims::F, shifts[4]};
    d[LI_4] = {Dims::NIB_L4, Dims::IN4, Dims::R, shifts[5]};
    return d[li];
}

enum class Kernel { Scalar, Avx2, Expand };
enum class HiddenLearning { None, Perturb, Modulated, Backprop };

struct Params {
    // population / topology
    uint32_t units = 200000;          // physical unit capacity
    uint32_t motif_size = 128;
    std::vector<uint32_t> k_set = {2, 4, 8, 16, 32, 64}; // motif m uses k_set[m % size]
    uint32_t max_hops = 2;            // sensory admission = hop 0; routed neighbour events add hops. Hops beyond hops_free are LEARNED:
                                      // taken only by bodies whose routed emissions have paid. hops=4 hops_free=2: -0.16% BPB TinyStories,
                                      // -0.30% mixed stream, same speed (kept out of the defaults so the NumPy parity config is unchanged)
    bool kpack = true;                // pack route bodies motif by motif; split children next to their parent
    bool kscreen = true;              // choose a new route's K class by measured benefit for this kind of input
    float kscreen_eps = 0.10f;        // exploration rate of the K screen
    float kscreen_rate = 0.02f;       // EMA rate of the per-(symbol, K class) consequence table
    uint32_t vocab = 32;              // symbols (32 synthetic, 256 raw bytes)
    uint64_t seed = 1;

    // activations / death
    // right-shift per layer (L1,L2,L3,L5,L6,L4), sized so a fresh unit's activations sit ~25-45 of 127
    uint8_t shifts[NUM_LAYERS] = {1, 2, 2, 3, 3, 3};
    int32_t death_l1 = 6, death_l23 = 6, death_l5 = 6, death_l6 = 6; // mean |act| thresholds (of 127)
    int32_t death_rel = 0;            // relative death: also die below (unit's own activity EMA * death_rel) >> 6 at that stage; 0 = off
    int32_t recirc_thresh = 40;       // mean |l6| needed to run L4
    uint32_t seq_max_depth = 3;       // lever 3 (steps per answer): hops the within-episode chain may walk (<= 8); a learned END slot lets it follow to the end
    bool seq_arbitrate = false;       // OFF: training-stream accuracy is inflated for the memory by the stream's own redundancy. When on: the chain answers a question form only where its credited accuracy beats the memory's own (replaces the fixed bar)
    uint32_t seq_attn = 0;            // learned answer attention (attend.h): 0 off; 1 backprop throughout; 2 backprop warm start then forward gradients; 3 warm start then frozen; 4 forward gradients from scratch
    uint32_t attn_warm = 20000;       // answer turns in the warm start (modes 2, 3)
    uint32_t attn_epochs = 2;         // replays of the warm-start buffer at the switch
    // goal-credited plasticity: on the bytes of a reply's first token (the answer) every structural and learning
    // signal is scaled by goal_gain; bodies that fired while the question was read keep an eligibility trace
    // (decay goal_decay per byte) and are credited by how the answer went relative to usual (goal_elig > 0)
    float goal_gain = 1.f, goal_elig = 0.f, goal_decay = 0.99f;
    // composer: learned units above bytes. levels = how many levels above bytes (0 off). A level-1 unit closes when
    // the organism's surprise about the next byte exceeds its running mean x lvl_margin (or at lvl_max bytes); level
    // l+1 closes over level-l units by the same rule on its own next-unit counts. Each level has its own motifs
    // (lvl_motifs each, reserved from the byte pool) whose bodies see only that level's unit identities.
    uint32_t lvl_warm = 0;            // bytes to let the vocabulary settle before the memory side switches to units
    uint32_t lvl_minc = 5;            // a proposed unit joins the lexicon once it has recurred this often; segmentation then prefers the longest known unit (stable identities)
    uint32_t seq_units = 0;           // the composer feeds the memory side: 1 = level-1 units are the chain words, 2 = level-2 units are its sentences too
    uint32_t levels = 0, lvl_motifs = 48, lvl_max = 24, lvl_feed = 0; float lvl_margin = 1.5f;   // lvl_feed: level outputs also enter the byte bodies' persistent workspace
    uint32_t attn_holdout = 5;        // one dialogue in this many trains only trust (gate, source mixture); 0 = no split
    float attn_lr = 0.03f, attn_fg_lr = 0.003f;   // 0.1 is best on bAbI alone; on a mixed stream 0.1 tracks bursts instead of learning (scripts/attn_mixed.sh)
    float seq_abstain = 0.3f;         // the chain answers a question form only when its credited accuracy there is at least this (0 = always answer)
    uint32_t recirc_iters = 0;
    uint32_t recirc_learn = 0;        // 0 off; 1 counterfactual utility (biased: the path a unit trains on looks better); 2 two-armed bandit on actual loss        // learned depth per step: a unit recirculates (up to recirc_iters) only where its counterfactual gain has paid
    float recirc_bits = 0.02f;        // minimum EMA gain (bits) for a unit to keep recirculating (the price of the extra passes)
    float recirc_explore = 0.03f;
    float recirc_init = 0.0f;         // starting utility: 0 = a unit must EARN recirculation by exploration (its weights train on the ordinary path)     // probability of recirculating anyway, so an untried unit can earn the right
    uint32_t hops_free = 99;          // (99 = all hops free, the reference behaviour; training runs use hops=4 hops_free=2)          // learned fan-out depth: hops beyond this are taken only by bodies whose route utility exceeds hop_gate_bits
    float hop_gate_bits = 0.05f;        // lever 1 (depth per step): extra passes of L4 -> L2/L3 -> L5 -> L6 within the same byte (0 = L4 feeds the next byte only)
    int32_t resist_min = 24, resist_max = 255, resist_init = 64;     // 64 == 1.0
    int32_t resist_up = 2, resist_up_dead = 1, resist_down = 3;
    uint32_t novel_hits = 64;         // lock with fewer hits is "novel"
    uint32_t surprise_thresh = 16384; // lock is "surprising" when err_ema exceeds the stream's wrong-rate EMA by this margin (of 65535)
    int32_t novelty_num = 1, novelty_den = 2; // threshold multiplier for novel events

    // addressing
    uint32_t leaf_max = 4;            // split leaf when it holds more locks than this
    uint32_t assoc_extra_bits = 1;    // assoc lock init mask bits = log2(K) + this
    uint32_t assoc_tighten_after = 16;// accepted messages without consequence before mask grows
    uint32_t assoc_loosen_after = 64; // novel events without any accepted message before mask shrinks
    // utility-gated routed contribution (ALM: a route contributes only while its utility EMA exceeds a bits threshold)
    bool route_gate = true;
    float route_gate_bits = 0.05f;    // minimum EMA of counterfactual gain (bits) for a routed body's emission to enter the workspace
    float route_utility_decay = 0.99f;
    float route_utility_init = 0.10f; // newborn bodies start above the gate so they get evaluated

    // learning
    bool train = true;
    bool structural = true;           // lock births / splits / deaths
    HiddenLearning hidden = HiddenLearning::None;
    int32_t lr_num = 1, lr_shift = 9; // accumulator delta = (x * q * lr_num) >> lr_shift
    int32_t quantum = 64, half_quantum = 32; // INT8 accumulator quantum for one INT4 level
    int32_t acc_bits = 8;             // 8: INT8 residual (deltas (x*q*lr) >> lr_shift; products below 2^lr_shift are lost);
                                      // 16: INT16 residual, quantum x64 and shift -6 (same dynamics for large deltas, 64x finer for small ones)
    float err_gain = 8.0f;            // L6 error scaling before INT8 quantisation
    float lr_out = 0.005f;            // shared readout learning rate (vocab > F only); 0.02 made BPB drift up on TinyStories
    float err_sparse = 0.01f;         // readout error entries below this are ignored (vocab > F only)
    int32_t lr_anneal = 3;            // extra lr_shift applied when the event is familiar (not novel/surprising)
    int32_t perturb_delta = 8;        // node perturbation magnitude (INT8 units)
    int32_t perturb_gain = 64;        // reward difference scaling
    int32_t hidden_shift = 0;         // extra lr_shift for hidden-layer (perturbation) updates
    bool perturb_clean = false;       // commit prediction / L6 credit from an unperturbed third pass
    float bp_gain = 1.f;              // hidden=backprop: the backpropagated error is quantised like the L6 error (err_gain x 127) times this
    int32_t homeo = 0;                // 1 = per-event adjustment; 2 = SLOW: running saturation/magnitude averages per body x layer, shift moves only when an average crosses its bar
                                      // homeostatic gain: each body keeps its OWN right shift per layer, raised when >1/8 of the layer's
                                      // outputs sit at the +-127 clamp and lowered when mean |y| falls below homeo_lo (adapts while
                                      // training; frozen for evaluation). Fixed shifts leave the layers either saturated (the ws memory
                                      // inputs arrive at +-127) or vanishing (without them), so a body computes a constant code.
    int32_t homeo_lo = 24;
    float homeo_sat = 0.10f;
    bool homeo_out = true;            // homeo also governs L6 (the emission the shared readout reads); 0 = hidden layers only          // homeo=2: raise the shift when the running saturated fraction exceeds this
    bool in_struct = false;           // structured body input instead of key bits + a random symbol code: 16 features (bits and
                                      // character classes) for the current byte and each of the 4 before it, 16 word/sentence position
                                      // features, and the persistent workspace normalised by its running scale. All at +-in_scale, so no
                                      // group drowns the others. (The lock KEYS, i.e. routing, are unchanged.)
    int32_t in_scale = 32;
    uint32_t persist = 0;             // option B (2026-09-24), the design's L4 "recurrent return": up to N emitting bodies whose L4 activity (mean |l4|)
                                      // is >= persist_thresh stay ACTIVE into the next byte, fed their own L4 return + the new input, and keep
                                      // emitting until their activity falls or persist_age bytes pass. Credit: truncated backprop through each
                                      // body's own recurrence (persist_bptt steps). 0 = off.
    int32_t persist_thresh = 40; uint32_t persist_age = 64, persist_bptt = 16;
    int32_t ws_regs = 0;              // 1: per-body gates (L6 dims); 2: ONE SHARED WRITE HEAD reading the workspace (see organism.cpp)             // option A (2026-09-24): 4 gated workspace REGISTERS. An emitting body's L6 dims 24-27 propose values and
                                      // 28-31 write gates (sigmoid(l6/32 - 2), mostly closed); reg = (1 - G) reg + sum(g v), G = min(1, sum g):
                                      // a write overwrites instead of being diluted. Read in input cells 124-127. Temporal credit (tc_window)
                                      // follows the gates: a write's effect on a later read = its gate x every later (1 - G).
    float reg_bias = -2.f;
    float head_lr = 0.01f;
    bool adam = false;                // hidden=backprop: bodies learn by ADAM on a float shadow per synapse (INT4 weight = round(shadow)); every
                                      // backprop path adds its exact gradient to a per-synapse buffer and a body takes one bias-corrected step
                                      // after every adam_k uses (seq_probe harness: the accumulator SGD never learns agreement, Adam does)
    float adam_lr = 0.003f;           // step size in INT4 grid units
    uint32_t adam_k = 32;             // uses of a body per Adam step (a per-body minibatch)
    float adam_wd = 0.f;              // AdamW: decoupled weight decay on the body shadows (per step: sh -= lr * wd * sh)
    float ro_adam_wd = 0.f;           // AdamW on the readout / register readout / head weights (biases and the sequence gate: none)
    bool adam_l6 = true;              // adam: 0 = Adam for the hidden layers only; L6 keeps the original fast per-event quantised rule
    bool adam_warm = false;           // a young body steps after 1, 2, 4, 8, 16 uses (the batch doubles up to adam_k), so new bodies learn at once
    bool ro_adam = false;             // the shared float readout (wout, register readout, write head) by minibatch Adam instead of per-event SGD
                                      // (harness: with an SGD readout neither Adam form learns agreement by 1.2M sentences; with Adam 2/2 seeds)
    float ro_adam_lr = 0.002f;
    uint32_t ro_adam_k = 512;         // events per readout step
    float reg_ro_init = 0.f;          // reg_readout: uniform +-init for the register readout weights (0 = zeros: no credit reaches the head
                                      // or the writers until chance correlations build up - the failing seeds never start)
    uint32_t reg_center = 0;          // ws_regs: readers (readout + body inputs) see reg - running mean (time constant reg_center events), so
                                      // a pinned register reads 0: registers can no longer serve as free BIASES (the readout and bodies have none)
    uint32_t reg_revive = 0;          // ws_regs=2: a register pinned at the SAME sign (|reg| > 99) for this many consecutive events is dead
                                      // (saturated tanh: no gradient) and is reborn: fresh random head row, zeroed Adam moments, value 0
    bool head_norm = false;           // ws_regs=2: the write head reads the MEAN emitting L6 (ws / n_emitters / 128, as the harness's head reads
                                      // one body's L6) instead of the clamped SUM, which saturates a random head (seed-dependent dead registers)
    bool reg_readout = false;         // ws_regs: the byte readout ALSO reads the registers directly (logits += Wr . reg / 100), so a register
                                      // carries content to the prediction without needing learned hidden weights on the reading side
    float reg_scale = 50.f;           // ws_regs: register input cell = reg x reg_scale / 100 (reg in +-100). With a small in_scale this makes the
                                      // registers the strongest inputs (a bounded INT4 weight on them can then dominate the other cells)            // ws_regs=2: SGD rate of the shared write head (4 x (32 gate + 32 value) weights + 4 gate biases)
    bool reg_ste = false;             // ws_regs: straight-through gate credit (the gate logit gets (v - reg_prev) x error, without g(1-g))            // ws_regs: gate bias (sigmoid(l6/32 + reg_bias)); more negative = gates start closed
    bool ws_bands = false;            // multi-timescale workspace memory: dims 0-7 / 8-15 / 16-23 / 24-31 decay at 128 / 224 / 248 / 254 of 256
                                      // (half-lives ~1 / 5 / 27 / 177 bytes), each band normalised on its own scale; temporal credit discounts
                                      // each band by its own decay. A slow band can carry what one body wrote past many bytes of other writes.
    float tc_gain = 1.f;              // gain on temporal (delayed) credit updates
    uint32_t tc_window = 0;           // temporal credit (hidden=backprop + in_struct): bodies that WROTE the persistent workspace learn what the
                                      // bodies READING it up to tc_window bytes later needed (their error at the workspace input cells,
                                      // discounted by ws_decay), applied when the writing event leaves the window. 0 = off.
    bool in_words = false;            // in_struct + learned embeddings of the previous 3 words (Bengio 2003 neural n-gram): a 65,536 x 16
                                      // table shared by all bodies, trained by hidden=backprop at L1's input. Layout: bytes t..t-2 (48),
                                      // position (16), words -1..-3 (48), workspace memory (16).
    float emb_lr = 0.0005f;
    bool seq_gate = false;            // learned blend weight between the bodies and the sequence memory (see organism.cpp); replaces seq_alpha x confidence
    float seq_gate_lr = 0.01f;
    int32_t body_err = 0;             // 1: bodies learn their own error; 2: ... weighted by their responsibility in the LINEAR blend (the
                                      // exact mixture gradient: bodies concentrate on what the memory misses; TRIME-style training with memory)
    bool seq_linear = false;          // linear mixture p = a * memory + (1 - a) * bodies (kNN-LM / cache-LM practice) instead of the geometric blend
    uint32_t shared_always = 0;       // DeepSeekMoE shared experts: the first N shared_types bodies run on EVERY byte and emit into the workspace            // learn_l6 from the bodies' own readout error instead of the blended (sequence memory) error
    bool bp_ste = false;
    float bp_leak = 0.f;              // hidden=backprop: gradient slope through a SATURATED (+-127) output during learning (forward clamp unchanged).
                                      // 0 = exact (dead units stay dead); 0.1 stabilised long-range learning in the reference (log #24)
    bool bp_route = false;
    uint32_t pc_settle = 0;           // hidden=backprop as PC-ALM settling (predictive coding, fixed-prediction form): credit moves one layer per
                                      // settle step from its source (a body's output error, a receiver's return, a later reader), each error unit
                                      // using only its own layer's weights and the error one layer above; sites more than pc_settle steps away get
                                      // none. 0 = settle to completion (identical to exact backprop; organism depth <= 8).
    bool stoch_round = false;         // unbiased stochastic rounding of the L6 error and of every synaptic delta (instead of truncating sub-step updates to zero)
    uint32_t shared_types = 0;        // shared cell-type bodies (0 = off): the last N units belong to no lock. Every lock-admitted body routes to
                                      // one of them (fixed wiring by unit id) and, with route_return, gets its computation back through L4.
                                      // A shared body serves many contexts, so it sees far more data than any specialist (fruitfly: one motif
                                      // shared everywhere is what learned XOR-1024). Needs hops >= 1.
    bool nbr_route = true;            // K-neighbour routing in the motif (the original circuit wiring); 0 = shared types only
    bool route_ungated = false;       // no usefulness-based selection of links (coop protocol, 2026-09-24): no assoc tightening/loosening by a
                                      // receiver's own consequence, every hop allowed (no route-utility hop gate), no route-gate emission cut.
                                      // A conjunctive circuit's parts are worthless alone until the others work; usefulness gating cuts them first.
    bool msg_merge = false;           // full-width delivery: a receiver routed by several senders in one hop runs ONCE on the SUM of their
                                      // messages (not only the first); its route consequence returns to every sender, and the message
                                      // gradient reaches every sender.
    bool route_return = false;        // circuits per the design doc's L4 ("recurrent return / route consequence"): a routed body does
                                      // not write to the workspace; its L6 result goes through ITS L4 back to the sender that routed to
                                      // it. The sender sums its receivers' returns into its L2/L6 return inputs, recomputes L2 -> L5 -> L6
                                      // and alone emits. With hidden=backprop the gradient follows the same path (sender -> receivers' L4
                                      // -> their L6/L5/L3/L1), so L4 learns the route consequence.            // hidden=backprop: also pass the gradient across neighbour messages (receiver L3 -> sender L2): trainable circuits              // hidden=backprop: straight-through at saturated (+-127) outputs instead of the exact zero derivative
    float mod_gain = 32.f;            // hidden=modulated: M = clamp(mod_gain * (bits gained over running loss)) in INT8

    // structural split rule (design doc §6, discrete prototype form)
    uint32_t future_buckets = 32;     // y bucket count for conditional futures
    uint32_t evidence_slots = 4096;   // pool of conflicted locks collecting evidence
    uint32_t conflict_thresh = 22000; // err_ema (of 65535) above which a lock is "conflicted"
    uint32_t split_min_evidence = 64; // min samples on each branch
    float split_g_per_df = 3.0f;      // G statistic threshold = this * (buckets-1)
    uint32_t lock_death_blocks = 0;   // 0 = never; else free locks unhit for this many blocks
    // ALM-style context lifecycle: nursery -> confirmations -> admit by information gain -> utility -> recycle
    bool nursery = true;              // admit child locks on (lag, previous symbol) from the conflicted lock's nursery
    uint32_t admit_min = 2;           // confirmations before a child may be admitted (32 MB sweeps: 2/0.02 -> 2.67 BPB, 4/0.05 -> 2.82, 8/0.10 -> 2.99)
    float admit_bits = 0.02f;         // minimum information gain (bits) of the child over its parent
    uint32_t admit_max_per_lock = 8;  // children admitted per conflicted lock per block
    bool specific_wins = true;        // among generic candidates only the most specific address emits
    float utility_decay = 0.99f;      // lock utility EMA
    float recycle_utility = -0.05f;   // generic locks below this utility may be recycled when the pool is short
    uint32_t recycle_min_age = 8;     // blocks

    // runtime
    uint32_t block = 4096;
    Kernel kernel = Kernel::Avx2;
    uint32_t max_active = 8192;       // per-event trace capacity (learning records)
    bool stage_timers = true;
    uint32_t hist_bytes = 4;          // key context window: bits 32..63 hash the last hist_bytes symbols (max 15)
    // working memory (all off by default; see docs/ARCHITECTURE.md "Working memory")
    int32_t ws_decay = 128;           // persistent workspace: ws_mem = (ws_mem * ws_decay) >> 8 + ws ; 0 = off
    bool ws_key = false;              // key bits 48..63 = sign pattern of ws_mem[0..15]
    uint32_t ws_emit_max = 0;         // with ws_split: only motif populations 0..ws_emit_max-1 write the shared workspace; the others feed only
                                      // their own readout channel (and learn only its error); 0 = every population writes it
    bool fm_salt = false;             // fixed motifs: salt each population's context hash with its index (populations of the same context
                                      // length then choose independent bodies; off = the original hashes)
    std::vector<uint32_t> fm_h, fm_n;  // motifs=h:n,h:n,... FIXED MOTIFS: populations created once, no births / splits / deaths / address
                                      // tree; each byte runs one body per motif (h=1: the current byte, n=256; h>1: a hash of the last h
                                      // bytes mod n); only the weights inside the layers learn
    float ff_inh = 0.f;               // cortical motif 3, FEEDFORWARD INHIBITION: the drive that excites a layer also recruits a shunting inhibitor:
                                      // every unit's drive / (1 + ff_inh x mean|input| / 127)
    float fb_inh = 0.f;               // cortical motif 4, FEEDBACK INHIBITION: the layer's own response recruits it: drive / (1 + fb_inh x mean|output| / 127)
    float disinh = 0.f;               // cortical motif 6, DISINHIBITION (VIP -| SST -| E): an inhibition of every body's output layer (L6 drive /
                                      // (1 + disinh x (1 - d) x mean|L6 input| / 127)) that a context signal releases: d = sigmoid(v_m . registers
                                      // / 127 + b_m), learned per motif m from the 4 register input cells (top-down state); d starts at 0.5
    float compete = 0.f;              // cortical motif 5, CROSS-ASSEMBLY COMPETITION: the byte's bodies compete for the workspace - emissions weighted by a
                                      // softmax of compete x activity (x the number of emitters, so equal activity changes nothing)
    bool motif_mix = false;           // G / motif 8 (selective fan-out / fan-in): with motif_stream, each body learns WHICH earlier bodies it reads -
                                      // its message is a softmax-weighted mix (per-body gate logits, zero at start = the plain mean)
    uint32_t iter_hops = 0;           // G, AN ITERATED PROCESSOR (docs/UNBOUNDED_CIRCUITS.md): the motif population of context 0 (last in
                                      // motifs=, e.g. 0:256 - its body is chosen by the byte) is not run once per byte but for iter_hops
                                      // HOPS after the other motifs, each hop on the stream of every earlier body of the byte (its own
                                      // earlier hops included; motif_stream=1), with a per-hop cue in its return cells. One learned step,
                                      // repeated: depth costs compute, not bodies
    bool iter_rand = false;           // training draws each byte's hops uniformly from 0..iter_hops (depth-agnostic, as randdepth in the harness)
    uint32_t iter_eval = UINT32_MAX;  // hops when not training (default iter_hops; more than trained = the extrapolation test; <= 16)
    uint32_t iter_eval_hops() const { return iter_eval == UINT32_MAX ? iter_hops : iter_eval; }
    float mix_res = 0.f;              // motif_mix / motif_attn as a RESIDUAL on the plain stream: a body reads mix_res x the mean of the
                                      // earlier bodies' L6 + (1 - mix_res) x the learned mix (0: the mix alone)
    bool motif_attn = false;          // G, CONTENT-ROUTED fan-in / fan-out: with motif_stream, a body's message is an attention-weighted mix of the
                                      // earlier bodies' L6: score = query(its own L1, per position) . key(the sender's L6) / sqrt(8) + a position bias;
                                      // the projections are shared by all bodies (every event trains them), so which senders a body reads (fan-in)
                                      // and which receivers read a sender (fan-out, through its key) depend on what they computed. With motif_mix
                                      // the per-body gate logits add to the scores
    uint32_t ro_hidden = 0;           // readout width: H > 0 adds a hidden layer to the readout (relu, H units) over the shared input and, with
                                      // ws_split, every motif's channel; its output weights start at zero
    bool ws_split5 = false;           // readout width: each motif's L5 (16) gets its own readout channel too (as ws_split does for L6)
    bool muon = false;                // MUON for the bodies instead of Adam: Nesterov momentum per layer matrix, orthogonalised by 5 Newton-Schulz
    float muon_lr = 0.02f, muon_mom = 0.95f;   // iterations, applied to the INT4 shadows (x sqrt(max(1, rows/cols)))
    bool deep_in = false;             // cortical motif 1, PARALLEL FEEDFORWARD ROUTES (sensory input reaches deep layers directly): a body whose
                                      // L4 return is unused (no recirculation / persistence) gets the current byte's 16 input features in its L2
                                      // return cells and the byte's 8 bit features in its L6 return cells - L2 and L6 see the byte first-hand
    bool tile_split = false;          // tiles: the pyramid's reads also get their OWN readout channel (V x 32, zero at start) instead of only
                                      // sharing the summed workspace's 32 inputs (the wide-readout lesson applied to the tiles)
    bool ws_split = false;            // motifs: a WIDER readout - besides the shared readout of the summed workspace, each motif's own emission
                                      // (its body's L6) is read through its own weights (zero at start): the readout sees M x 32 inputs, not 32
    bool fm_stream = false;           // motif_stream (goal item G, docs/UNBOUNDED_CIRCUITS.md): with motif_wire, body m's L3 message is the MEAN of
                                      // the L6 of this byte's earlier motif bodies (x 0.5, clamped), and its gradient returns to all of their
                                      // L6 - a normalised residual stream that composes (motif_wire's L2 message does not); needs bp_route
    bool fm_wire = false;             // motifs: each active body's L2 feeds the NEXT motif's body's L3 (neighbour message; bp_route passes
                                      // the receiver's gradient back to the sender's L2) - a multi-body circuit inside every byte
    bool ws_mem_in = true;            // 0: the fixed-decay persistent workspace is NOT fed to the bodies (cells 96..123 zero); registers still are
    float ws_in_gain = 1.f;           // gain on the fixed-decay workspace input cells (ws_mem_in=1): 1 = scaled to +-in_scale like the byte cells
    uint32_t apical = 0;              // 1: APICAL register gating (cortical motif 7, dendritic integration): each L5 unit's output is multiplied by
                                      // 1 + a/256 (0..2x), a = (sum_r U[j][r] reg_input_r) >> apical_shift; U = 64 INT4 weights per body kept in
                                      // the L4 block (L4 unused: needs recirculation / persist / route_return off), zero at birth (starts as off)
    uint32_t apical_shift = 4;
    uint32_t tiles = 0;               // TILE PYRAMID (user, 2026-09-25: vector tiling for information zooming), levels 0 = off. Level-0 tiles are
                                      // WORDS (closed at a word boundary: the workspace summary then); a level-l tile closes at every 2nd level-(l-1)
                                      // tile and is the MERGE of those two by a merge body (a fixed population per level, chosen by the children's
                                      // sign bits). The view is the last tile_k tiles of every level, read into the readout's input through learned
                                      // 32x32 mixings (zero at start). Holding is structural: nothing learns to keep a gate shut.
    uint32_t tile_k = 4, tile_bodies = 256;
    uint32_t tile_wm = 0;             // WORKING MEMORY (docs/PYRAMID_MEMORY.md tier 2): slots that keep a tile leaving the view if its readers
                                      // leaned on it (its USE: accumulated read-gradient size) more than on the weakest slot's item
    bool tile_ctx = false;            // tiles carry CONTEXT: a word tile (and every query of the attention / fast-weight reads) is built from the
                                      // full context vector ws/128 + what the tiles were read into it, not the bodies' own output alone - so "key"
                                      // is stored as "key after red box has the", and the question "in the red box is the" can cue it
    float tile_id = 0.f;              // > 0: a word tile also carries the word's IDENTITY - a fixed +-tile_id code hashed from its bytes (what a
                                      // tokenizer's embedding gives a transformer); the workspace summary alone holds only what predicts the next byte
    uint32_t tile_fw = 0;             // 1: FAST-WEIGHT memory (synaptic, nature-like; docs/PYRAMID_MEMORY.md): each closing word tile writes a decaying
                                      // Hebbian outer product A <- l A + v k^T (k = unit(Wk v)); every event reads r = A (Wq ws/128) into the readout
                                      // through Wo. Linear attention: constant cost per byte however much was stored; capacity by interference.
    float tile_fw_decay = 0.95f;      // l: per word
    uint32_t tile_att = 0;            // 1: a CONTENT-ADDRESSED read of the view + working memory: query = Wq (this event's workspace / 128), keys =
                                      // Wk tile, softmax over the live slots, read = Wo sum(alpha tile) into the readout's input. Bounded: the slot
                                      // count is fixed (not a KV cache). Picks the binding the current context asks for.
    uint32_t tile_ret = 0;            // N > 0: an EPISODIC BUFFER of the last N word tiles, read by FIXED context keys (docs/PYRAMID_MEMORY.md,
                                      // "retrieval with fixed context keys"): a word's key is the frequency-weighted bag of hashed codes of the
                                      // tile_ret_c words before it; each event's query is the same bag over the last tile_ret_c closed words;
                                      // softmax(tile_ret_beta x cosine) over the buffer, read through a learned Wo. Nothing learns to address.
    uint32_t tile_ret_c = 4; float tile_ret_beta = 10.f;
    uint32_t tile_ret_heads = 1;      // 2: two heads, each its own softmax - positional (the item that followed this context) and an order-free
                                      // bag (items near these words, whatever the phrasing); one output map reads both
    uint32_t tile_ret_skip = 0;       // the newest items that cannot be retrieved: the query's own words (their keys ARE the query - a bag head
                                      // would retrieve the question itself); set to tile_ret_c
    float tile_ret_recency = 0.f;     // > 0: newer items win among equal matches (logit - recency x rank / N): a binding made again
                                      // later replaces the older one (proactive interference in a large buffer; "where is Mary NOW")
    uint32_t tile_ret_pos = 1;        // 1: a word's code depends on its POSITION in the key (1st, 2nd, ... word back): the item that FOLLOWED a
                                      // context like the current one matches (an induction match), not every word near it; 0: an order-free bag
    bool tile_avg = false;            // tiles: a parent is the MEAN of its two children (no merge body): keeps any linear signal, and credit
                                      // reaches every word tile underneath (1/leaves each) - the bisection baseline for the learned merges
    float tile_wm_decay = 0.95f;      // a slot's use decays by this per word, so an item that stops mattering is replaced   // tiles kept per level; merge bodies per level >= 1 (after the motifs: units must hold them)
    uint32_t reg_clock = 0;           // 1: the write head writes only at WORD BOUNDARIES (the byte just read is not a letter or digit); inside a
                                      // word its gate is shut, so a stored value faces one overwrite decision per word, not per byte (the
                                      // timescale a subword tokenizer gives a transformer); registers are still read at every byte
    uint32_t mtp = 0;                 // multi-token prediction (goal item E): heads h = 1..mtp read an earlier event's workspace (its ring slot's head
                                      // input) and predict the byte h events later; the error reaches that event's writers through the ring
    float mtp_w = 1.f;                // weight of the extra heads' loss (relative to the next-byte readout)
    bool ws_input = true;             // L1 input cells 96..127 = ws_mem[0..31] scaled to INT8 (replaces half the symbol code)
    bool l4_key = false;              // key bits 32..47 = sign pattern of the last L4 return produced this event
    // episodic memory: exact-key locks born on surprise, imprinted one-shot, evicted least-recently-hit
    bool episodic = false;
    float episodic_margin = 0.0f;     // episodic birth when the prediction was wrong AND loss > loss_ema + margin (nats)
    uint32_t episodic_recur = 1;      // class over instance: a snapshot body is born only when the SAME exact context has surprised the
                                      // model this many times (1 = every surprise, the original rule). One-off contexts never get a body; a
                                      // 2^20-slot tag table remembers pending surprises (a colliding context simply restarts the count).
    // Body market (off by default; market=1): every non-backbone body -- nursery child, split child, snapshot -- competes for
    // capacity on ONE fitness, fitness = utility x log(1 + reuse rate), where utility is the body's bits gained over the
    // stream's prior when it admits (the existing lock utility; replayed material counts too) and reuse is hits per block
    // since birth. A new body is on probation for market_probation blocks: at the end it must have been reused at least
    // market_min_hits times and hold utility >= the current admission bar, or it is released. When capacity is short the
    // least fit bodies are released, whatever their kind (replacing snapshot LRU eviction and utility recycling). A body
    // that just split or admitted children cannot do so again for market_cooldown blocks (learn before cloning), and
    // nursery admission requires the bar as its information gain. The bar starts at market_bar and is moved by the
    // caller's retention check (market_adjust): raised when retention regressed, relaxed when it held.
    // Parsimony (equal capability with fewer bodies wins): market fitness uses a SHARED utility -- each event's gain in
    // bits is divided among the bodies that admitted it -- so a body that delivers a gain alone keeps all of it and a
    // body that only fires beside capable ones keeps a fraction; and the controller also raises the bar when retention
    // stayed flat while the population grew.
    bool market = false;
    uint32_t market_probation = 16, market_min_hits = 8, market_cooldown = 8;
    bool market_share = false;        // true: fitness from SHARED credit (gain split among co-firing bodies); false: plain utility.
                                      // Shared credit punished bodies inside busy, useful structures (16-way splits) -- off by default.
    uint32_t market_gate_blocks = 8;  // capacity culls are batched every this many blocks, each batch passing the counterfactual gate
    uint32_t market_reserve_ppm = 10000; // free units the market keeps (fraction of capacity, ppm; 1%). The snapshot-eviction reserve
                                         // (2 blocks of births) was sized for 200k bodies and would keep 82% of a 20k model empty.
    float market_bar = 0.05f, market_bar_min = 0.02f, market_bar_max = 1.0f;
    bool market_contest = false;      // birth-side market: no reserve refill and no batch culls. A birth takes a free unit while more
                                      // than a small backbone reserve is free; otherwise it must OUTBID the weakest admitted body
                                      // (its fitness below the newborn's bid) and displace exactly that one, or it is not born.
                                      // Bids: nursery child = its gain bits x log(1 + parent reuse); split child = parent utility x
                                      // log(1 + parent reuse / 2); snapshot = 0 (may only displace a body with negative fitness).
                                      // Probationers are never displaced: they leave only by failing probation.
    bool market_protect = false;      // contest: an admitted GENERAL body is displaceable only once its utility is below
                                      // recycle_utility (the market-off recycling rule); fitness ranking still governs snapshots.
    float market_birth_ratio = 0.f;   // contest birth throttle (0 = off): market births per batch are limited to this many per
                                      // probation pass in the previous batch (floor units/1000), so a low pass rate means fewer births.
    uint32_t episodic_promote = 0;    // promotion (0 = off): once a snapshot has been reused this many times after birth, its general
                                      // parent's split evidence is opened and kept open, so the parent records every context it sees and
                                      // the evidence-driven split can carve out a GENERAL child for the recurring family of contexts.
                                      // When the parent has split, the snapshot is retired and its unit freed.
    int32_t imprint_iters = 8;        // perceptron refinements of the imprint direction until it decodes
    bool exact_only = false;          // an exact-key lock silences generic candidates (else all emit into the workspace)
    int32_t exact_gain = 1;           // workspace weight of an episodic body's emission (a specific address carries more evidence)
    uint32_t unit_reserve_ppm = 5000; // evict episodic locks when free units fall below this fraction (ppm)
    bool record_preds = false;        // keep per-event predictions (needle test)
    uint64_t readout_freeze = 4194304; // stop updating the shared readout after this many events (0 = never). A readout that keeps
                                       // moving is a moving target for the INT4 bodies' L6 codes: on long streams BPB rose 3.2 -> 4.9
                                       // over 128 MB with it live, and stayed flat at 3.0 with it frozen (scripts/drift_ablation.sh).
    // sequence memory (Atomic Thought sequences): rolling chain of steps with reverse context indices
    bool seqmem = true;               // on for 256-symbol streams (ignored for the synthetic 32-symbol task)
    uint32_t seq_window = 4194304;    // steps kept (bytes): the episode store (instances, relations between thoughts). Bounded memory:
                                      // with the consolidated tier a 256 KB-1 MB window already beats a 32 MB one (BPB 1.75 vs 1.94;
                                      // 128 MB stream: 1.63 vs 1.92 with a 128 MB window). 4 MB keeps a few thousand dialogues for commitment.
    uint32_t seq_hash_bits = 20;
    float seq_kappa = 8.f;            // semisaturation of the sequence support (normalized competition)
    float seq_alpha = 0.8f;           // max weight of the sequence continuation in the blend (scaled by confidence)
    float seq_use_gain = 0.25f;       // extra weight per matched byte beyond the lag
    int32_t seq_bubble_min = 1 << 30; // a match at least this long may be followed as a bubble (hard commitment; off by default —
                                      // measured as a takeover that wrecks BPB and generation; serialization already emerges from
                                      // composition chaining along the same remembered sequence)
    int32_t seq_bubble_len = 32;      // bytes a bubble may be followed before re-evaluation
    float seq_bubble_support = 0.02f; // the bodies' probability of the bubble's next byte must stay above this
    int32_t seq_min_support = 2;      // instances that must back a lag level before it is used (bubbling: class over instance).
                                      // Strict 250-word test at T=0.7: support 1 -> 0/8 loops but 74% verbatim copying; support 2 -> 0/8 loops,
                                      // 11.7% copying, 98% valid words, planted recall 18/32; support 3 -> 5.9% copying, 16/32.
    int32_t seq_copy_max = 1 << 30;   // consecutive single-instance long-match steps allowed before falling back to consensus
    bool seq_consolidate = true;      // consolidated class-count tier (bounded by seq_cbits, text-free, TAGGED per context); instances then only need a short window
    uint32_t seq_cbits = 20;          // 2^20 buckets x 8 lags x 16 B = 128 MB
    float seq_class_gain = 0.5f;      // 1.0 recalls more planted phrases (23-25/32 vs 17/32) but consensus from counts is deterministic
                                      // enough to loop once in 8 prompts (68 B repeat); 0.5 stayed at 0/8 in every trial
    float seq_topic_gain = 0.f;       // goal-directed commitment to a related episode while generating (0 = off; 4-16 in laminar_chat)
    float seq_util_rate = 0.f;        // agency in selection: EMA rate of a thought's utility (how well commitments chosen through it held); 0 = rarity only
    float seq_thought_df = 0.f;       // > 0: a thought is a word seen in fewer than this fraction of episodes (learned filter; 0 = stop list)
    float seq_goal_rel = 0.f;         // goal expansion by learned relations in episode selection (discount on related thoughts; 0 = off)
    bool seq_attr_eq = false;         // attribute equivalence by consequence: attribute words that answer alike under the same keys become one attribute
    bool seq_attr = false;            // endpoint attributes (the most informative relation word attached to a thought) in the composition key
    bool seq_rel_closure = false;     // closure credit for composition (measured negative on CLUTRR prose; kept as an experiment)
    float seq_rel_gain = 0.f;         // relations as things: triples credited by prediction of the stream; weight of their proposals in generation (0 = off)
    float seq_chain_neg = 0.f;        // learned reading: penalty on a non-answering chain's words (0.1 eroded names/locations on the mixed stream; 0 = off)
    int32_t seq_chain_mode = 1;       // 1 = hand-written reading (stop list, subject/object); 2 = LEARNED reading (word utility, position preference, credited from answers)
    bool seq_chain = false;           // relations within the episode: question thought -> sentence -> object -> next sentence (hops), depth learned per question form
    bool seq_goal_extend = false;     // bubble sequence: when a commitment ends, the thoughts emitted so far extend the goal for the next one
    uint32_t seq_dialogues = 65536;   // dialogues remembered in the bubble tier
    // agency kernel (port of Agkern) directing attention: 0 off, 1 on, 2 control (fixed focus bump, no operations)
    int32_t agency = 0;
    // the source's pool and commitment were tuned with four tendencies; keep K=4 unless re-tuning the pool
    std::string agency_ops = "hold,to_recent,to_context,to_memory";   // with episodic=1 use hold,attend_episodic,attend_generic,to_memory
    uint32_t agency_units = 512, agency_k = 16;
    float agency_focus_width = 24.f, agency_focus_floor = 0.25f, agency_epistemic = 0.3f, agency_route_lr = 0.1f, agency_arousal = 0.3f;
    float agency_cost = 0.0f;         // baseline persistence cost l (source default 0.10 sized for 0/1 outcomes; here contrasts are in p[target], so a focus persists unless it demonstrably hurts)
    int32_t agency_goal = 1;          // goal channel 0 = 0: p[target] (probability mass) | 1: 1 - loss/ln(V) (bounded log-loss, aligned with BPB)
    uint64_t agency_seed = 0;
    // Task-level agency (model task learner): adds a third goal channel whose consequences are public task
    // outcomes (tool results, goal closures, verified mission results) and eight task-state observation slots.
    int32_t agency_task = 0; float agency_task_demand = 1.0f; float agency_trace_life = 8.f;

    static Params from_args(int argc, char** argv, std::map<std::string, std::string>* rest = nullptr);
};

inline Params Params::from_args(int argc, char** argv, std::map<std::string, std::string>* rest) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (eq == std::string::npos) { if (rest) (*rest)[a] = "1"; continue; }
        std::string k = a.substr(0, eq), v = a.substr(eq + 1);
        auto u = [&]{ return (uint32_t)std::strtoul(v.c_str(), nullptr, 10); };
        auto f = [&]{ return std::strtof(v.c_str(), nullptr); };
        auto i32 = [&]{ return (int32_t)std::strtol(v.c_str(), nullptr, 10); };
        if      (k == "units") p.units = u();
        else if (k == "motif") p.motif_size = u();
        else if (k == "kset") { p.k_set.clear(); size_t s = 0; while (s < v.size()) { auto c = v.find(',', s); if (c == std::string::npos) c = v.size(); p.k_set.push_back((uint32_t)std::strtoul(v.substr(s, c - s).c_str(), nullptr, 10)); s = c + 1; } }
        else if (k == "hops") p.max_hops = u();
        else if (k == "kpack") p.kpack = v != "0";
        else if (k == "route_gate") p.route_gate = v != "0";
        else if (k == "route_gate_bits") p.route_gate_bits = f();
        else if (k == "kscreen") p.kscreen = v != "0";
        else if (k == "kscreen_eps") p.kscreen_eps = f();
        else if (k == "kscreen_rate") p.kscreen_rate = f();
        else if (k == "vocab") p.vocab = u();
        else if (k == "seed") p.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "death") { p.death_l1 = p.death_l23 = p.death_l5 = p.death_l6 = i32(); }
        else if (k == "death_l1") p.death_l1 = i32();
        else if (k == "death_l23") p.death_l23 = i32();
        else if (k == "death_l5") p.death_l5 = i32();
        else if (k == "death_l6") p.death_l6 = i32();
        else if (k == "recirc") p.recirc_thresh = i32();
        else if (k == "recirc_iters") p.recirc_iters = u();
        else if (k == "recirc_learn") p.recirc_learn = u();
        else if (k == "recirc_bits") p.recirc_bits = f();
        else if (k == "recirc_explore") p.recirc_explore = f();
        else if (k == "recirc_init") p.recirc_init = f();
        else if (k == "hops_free") p.hops_free = u();
        else if (k == "hop_gate_bits") p.hop_gate_bits = f();
        else if (k == "seq_max_depth") p.seq_max_depth = u();
        else if (k == "seq_abstain") p.seq_abstain = f();
        else if (k == "seq_arbitrate") p.seq_arbitrate = v != "0";
        else if (k == "seq_attn") p.seq_attn = u();
        else if (k == "attn_warm") p.attn_warm = u();
        else if (k == "attn_epochs") p.attn_epochs = u();
        else if (k == "attn_holdout") p.attn_holdout = u();
        else if (k == "goal_gain") p.goal_gain = f();
        else if (k == "goal_elig") p.goal_elig = f();
        else if (k == "goal_decay") p.goal_decay = f();
        else if (k == "levels") p.levels = u();
        else if (k == "lvl_motifs") p.lvl_motifs = u();
        else if (k == "lvl_max") p.lvl_max = u();
        else if (k == "lvl_margin") p.lvl_margin = f();
        else if (k == "lvl_feed") p.lvl_feed = u();
        else if (k == "seq_units") p.seq_units = u();
        else if (k == "lvl_minc") p.lvl_minc = u();
        else if (k == "lvl_warm") p.lvl_warm = u();
        else if (k == "attn_lr") p.attn_lr = f();
        else if (k == "attn_fg_lr") p.attn_fg_lr = f();
        else if (k == "death_rel") p.death_rel = i32();
        else if (k == "novel_hits") p.novel_hits = u();
        else if (k == "surprise") p.surprise_thresh = u();
        else if (k == "resist_up") p.resist_up = i32();
        else if (k == "resist_up_dead") p.resist_up_dead = i32();
        else if (k == "resist_down") p.resist_down = i32();
        else if (k == "leaf_max") p.leaf_max = u();
        else if (k == "train") p.train = v != "0";
        else if (k == "structural") p.structural = v != "0";
        else if (k == "hidden") p.hidden = (v == "perturb") ? HiddenLearning::Perturb : (v == "modulated") ? HiddenLearning::Modulated : (v == "backprop") ? HiddenLearning::Backprop : HiddenLearning::None;
        else if (k == "bp_gain") p.bp_gain = f();
        else if (k == "bp_ste") p.bp_ste = v != "0";
        else if (k == "bp_route") p.bp_route = v != "0";
        else if (k == "bp_leak") p.bp_leak = f();
        else if (k == "pc_settle") p.pc_settle = u();
        else if (k == "stoch_round") p.stoch_round = v != "0";
        else if (k == "route_return") p.route_return = v != "0";
        else if (k == "route_ungated") p.route_ungated = v != "0";
        else if (k == "msg_merge") p.msg_merge = v != "0";
        else if (k == "shared_types") p.shared_types = u();
        else if (k == "nbr_route") p.nbr_route = v != "0";
        else if (k == "body_err") p.body_err = i32();
        else if (k == "seq_linear") p.seq_linear = v != "0";
        else if (k == "shared_always") p.shared_always = u();
        else if (k == "seq_gate") p.seq_gate = v != "0";
        else if (k == "seq_gate_lr") p.seq_gate_lr = f();
        else if (k == "homeo") p.homeo = i32();
        else if (k == "homeo_sat") p.homeo_sat = f();
        else if (k == "homeo_out") p.homeo_out = v != "0";
        else if (k == "homeo_lo") p.homeo_lo = i32();
        else if (k == "in_struct") p.in_struct = v != "0";
        else if (k == "in_scale") p.in_scale = i32();
        else if (k == "in_words") p.in_words = v != "0";
        else if (k == "tc_window") p.tc_window = u();
        else if (k == "tc_gain") p.tc_gain = f();
        else if (k == "ws_bands") p.ws_bands = v != "0";
        else if (k == "ws_regs") p.ws_regs = std::stoi(v);
        else if (k == "head_lr") p.head_lr = f();
        else if (k == "reg_scale") p.reg_scale = f();
        else if (k == "reg_readout") p.reg_readout = v != "0";
        else if (k == "head_norm") p.head_norm = v != "0";
        else if (k == "reg_revive") p.reg_revive = (uint32_t)std::stoul(v);
        else if (k == "reg_center") p.reg_center = (uint32_t)std::stoul(v);
        else if (k == "ws_mem_in") p.ws_mem_in = v != "0";
        else if (k == "motif_wire") p.fm_wire = v != "0";
        else if (k == "ws_split") p.ws_split = v != "0";
        else if (k == "tile_split") p.tile_split = v != "0";
        else if (k == "deep_in") p.deep_in = v != "0";
        else if (k == "ro_hidden") p.ro_hidden = u();
        else if (k == "ff_inh") p.ff_inh = f();
        else if (k == "disinh") p.disinh = f();
        else if (k == "fb_inh") p.fb_inh = f();
        else if (k == "compete") p.compete = f();
        else if (k == "motif_mix") p.motif_mix = v != "0";
        else if (k == "motif_attn") p.motif_attn = v != "0";
        else if (k == "mix_res") p.mix_res = f();
        else if (k == "iter_hops") p.iter_hops = (uint32_t)std::stoul(v);
        else if (k == "iter_rand") p.iter_rand = v != "0";
        else if (k == "iter_eval") p.iter_eval = (uint32_t)std::stoul(v);
        else if (k == "ws_split5") p.ws_split5 = v != "0";
        else if (k == "muon") p.muon = v != "0";
        else if (k == "muon_lr") p.muon_lr = f();
        else if (k == "muon_mom") p.muon_mom = f();
        else if (k == "motif_stream") { p.fm_stream = v != "0"; if (p.fm_stream) p.fm_wire = true; }
        else if (k == "apical") p.apical = u();
        else if (k == "ws_in_gain") p.ws_in_gain = f();
        else if (k == "apical_shift") p.apical_shift = u();
        else if (k == "mtp") p.mtp = u();
        else if (k == "reg_clock") p.reg_clock = u();
        else if (k == "tiles") p.tiles = u();
        else if (k == "tile_k") p.tile_k = u();
        else if (k == "tile_bodies") p.tile_bodies = u();
        else if (k == "tile_wm") p.tile_wm = u();
        else if (k == "tile_avg") p.tile_avg = v != "0";
        else if (k == "tile_att") p.tile_att = u();
        else if (k == "tile_fw") p.tile_fw = u();
        else if (k == "tile_ret") p.tile_ret = u();
        else if (k == "tile_ret_c") p.tile_ret_c = u();
        else if (k == "tile_ret_beta") p.tile_ret_beta = f();
        else if (k == "tile_ret_pos") p.tile_ret_pos = u();
        else if (k == "tile_ret_heads") p.tile_ret_heads = u();
        else if (k == "tile_ret_recency") p.tile_ret_recency = f();
        else if (k == "tile_ret_skip") p.tile_ret_skip = u();
        else if (k == "tile_id") p.tile_id = f();
        else if (k == "tile_ctx") p.tile_ctx = v != "0";
        else if (k == "tile_fw_decay") p.tile_fw_decay = f();
        else if (k == "tile_wm_decay") p.tile_wm_decay = f();
        else if (k == "mtp_w") p.mtp_w = f();
        else if (k == "fm_salt") p.fm_salt = v != "0";
        else if (k == "ws_emit_max") p.ws_emit_max = u();
        else if (k == "motifs") { p.fm_h.clear(); p.fm_n.clear(); size_t a = 0;
            while (a < v.size()) { size_t c = v.find(',', a); if (c == std::string::npos) c = v.size(); const std::string it = v.substr(a, c - a); const size_t q = it.find(':');
                if (q == std::string::npos) throw std::runtime_error("motifs: expected h:n,h:n"); p.fm_h.push_back((uint32_t)std::stoul(it.substr(0, q))); p.fm_n.push_back((uint32_t)std::stoul(it.substr(q + 1))); a = c + 1; } }
        else if (k == "reg_ro_init") p.reg_ro_init = f();
        else if (k == "adam") p.adam = v != "0";
        else if (k == "adam_lr") p.adam_lr = f();
        else if (k == "adam_k") p.adam_k = (uint32_t)std::stoul(v);
        else if (k == "adam_warm") p.adam_warm = v != "0";
        else if (k == "adam_l6") p.adam_l6 = v != "0";
        else if (k == "adam_wd") p.adam_wd = f();
        else if (k == "ro_adam_wd") p.ro_adam_wd = f();
        else if (k == "ro_adam") p.ro_adam = v != "0";
        else if (k == "ro_adam_lr") p.ro_adam_lr = f();
        else if (k == "ro_adam_k") p.ro_adam_k = (uint32_t)std::stoul(v);
        else if (k == "persist") p.persist = u();
        else if (k == "persist_thresh") p.persist_thresh = i32();
        else if (k == "persist_age") p.persist_age = u();
        else if (k == "persist_bptt") p.persist_bptt = u();
        else if (k == "reg_bias") p.reg_bias = f();
        else if (k == "reg_ste") p.reg_ste = v != "0";
        else if (k == "emb_lr") p.emb_lr = f();
        else if (k == "mod_gain") p.mod_gain = f();
        else if (k == "lr_shift") p.lr_shift = i32();
        else if (k == "lr_num") p.lr_num = i32();
        else if (k == "err_gain") p.err_gain = f();
        else if (k == "lr_out") p.lr_out = f();
        else if (k == "err_sparse") p.err_sparse = f();
        else if (k == "lr_anneal") p.lr_anneal = i32();
        else if (k == "perturb_delta") p.perturb_delta = i32();
        else if (k == "perturb_gain") p.perturb_gain = i32();
        else if (k == "hidden_shift") p.hidden_shift = i32();
        else if (k == "perturb_clean") p.perturb_clean = v != "0";
        else if (k == "hist") p.hist_bytes = std::min<uint32_t>(15, u());
        else if (k == "ws_decay") p.ws_decay = i32();
        else if (k == "ws_key") p.ws_key = v != "0";
        else if (k == "ws_input") p.ws_input = v != "0";
        else if (k == "l4_key") p.l4_key = v != "0";
        else if (k == "episodic") p.episodic = v != "0";
        else if (k == "episodic_margin") p.episodic_margin = f();
        else if (k == "episodic_recur") p.episodic_recur = std::max<uint32_t>(1, u());
        else if (k == "episodic_promote") p.episodic_promote = u();
        else if (k == "market") p.market = v != "0";
        else if (k == "market_probation") p.market_probation = u();
        else if (k == "market_min_hits") p.market_min_hits = u();
        else if (k == "market_cooldown") p.market_cooldown = u();
        else if (k == "market_reserve_ppm") p.market_reserve_ppm = u();
        else if (k == "market_share") p.market_share = v != "0";
        else if (k == "market_gate_blocks") p.market_gate_blocks = std::max<uint32_t>(1, u());
        else if (k == "market_bar") p.market_bar = f();
        else if (k == "market_bar_min") p.market_bar_min = f();
        else if (k == "market_bar_max") p.market_bar_max = f();
        else if (k == "market_contest") p.market_contest = v != "0";
        else if (k == "market_protect") p.market_protect = v != "0";
        else if (k == "market_birth_ratio") p.market_birth_ratio = f();
        else if (k == "imprint_iters") p.imprint_iters = i32();
        else if (k == "exact_only") p.exact_only = v != "0";
        else if (k == "exact_gain") p.exact_gain = i32();
        else if (k == "reserve_ppm") p.unit_reserve_ppm = u();
        else if (k == "readout_freeze") p.readout_freeze = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "seqmem") p.seqmem = v != "0";
        else if (k == "seq_window") p.seq_window = u();
        else if (k == "seq_bits") p.seq_hash_bits = u();
        else if (k == "seq_kappa") p.seq_kappa = f();
        else if (k == "seq_alpha") p.seq_alpha = f();
        else if (k == "seq_use_gain") p.seq_use_gain = f();
        else if (k == "seq_bubble_min") p.seq_bubble_min = i32();
        else if (k == "seq_bubble_len") p.seq_bubble_len = i32();
        else if (k == "seq_bubble_support") p.seq_bubble_support = f();
        else if (k == "seq_min_support") p.seq_min_support = i32();
        else if (k == "seq_copy_max") p.seq_copy_max = i32();
        else if (k == "seq_consolidate") p.seq_consolidate = v != "0";
        else if (k == "seq_cbits") p.seq_cbits = u();
        else if (k == "seq_goal_extend") p.seq_goal_extend = v != "0";
        else if (k == "seq_chain") { p.seq_chain = v != "0"; p.seq_chain_mode = v == "2" ? 2 : 1; }
        else if (k == "acc_bits") p.acc_bits = (int32_t)u();
        else if (k == "seq_chain_neg") p.seq_chain_neg = f();
        else if (k == "seq_rel_gain") p.seq_rel_gain = f();
        else if (k == "seq_rel_closure") p.seq_rel_closure = v != "0";
        else if (k == "seq_attr") p.seq_attr = v != "0";
        else if (k == "seq_attr_eq") p.seq_attr_eq = v != "0";
        else if (k == "seq_goal_rel") p.seq_goal_rel = f();
        else if (k == "seq_thought_df") p.seq_thought_df = f();
        else if (k == "seq_util_rate") p.seq_util_rate = f();
        else if (k == "seq_class_gain") p.seq_class_gain = f();
        else if (k == "seq_topic_gain") p.seq_topic_gain = f();
        else if (k == "seq_dialogues") p.seq_dialogues = u();
        else if (k == "agency") p.agency = i32();
        else if (k == "agency_ops") p.agency_ops = v;
        else if (k == "agency_units") p.agency_units = u();
        else if (k == "agency_k") p.agency_k = u();
        else if (k == "agency_width") p.agency_focus_width = f();
        else if (k == "agency_floor") p.agency_focus_floor = f();
        else if (k == "agency_epistemic") p.agency_epistemic = f();
        else if (k == "agency_lr") p.agency_route_lr = f();
        else if (k == "agency_arousal") p.agency_arousal = f();
        else if (k == "agency_cost") p.agency_cost = f();
        else if (k == "agency_goal") p.agency_goal = i32();
        else if (k == "agency_seed") p.agency_seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "agency_task") p.agency_task = i32();
        else if (k == "agency_task_demand") p.agency_task_demand = f();
        else if (k == "agency_trace_life") p.agency_trace_life = f();
        else if (k == "buckets") p.future_buckets = u();
        else if (k == "evidence_slots") p.evidence_slots = u();
        else if (k == "conflict") p.conflict_thresh = u();
        else if (k == "split_min") p.split_min_evidence = u();
        else if (k == "split_g") p.split_g_per_df = f();
        else if (k == "lock_death") p.lock_death_blocks = u();
        else if (k == "nursery") p.nursery = v != "0";
        else if (k == "admit_min") p.admit_min = u();
        else if (k == "admit_bits") p.admit_bits = f();
        else if (k == "admit_max") p.admit_max_per_lock = u();
        else if (k == "specific_wins") p.specific_wins = v != "0";
        else if (k == "recycle_utility") p.recycle_utility = f();
        else if (k == "block") p.block = u();
        else if (k == "kernel") p.kernel = (v == "scalar") ? Kernel::Scalar : (v == "expand") ? Kernel::Expand : Kernel::Avx2;
        else if (k == "max_active") p.max_active = u();
        else if (k == "timers") p.stage_timers = v != "0";
        else if (k == "shifts") { for (int j = 0; j < NUM_LAYERS && (size_t)(2 * j) < v.size(); ++j) p.shifts[j] = (uint8_t)(v[2 * j] - '0'); }
        else if (rest) (*rest)[k] = v;
        else throw std::runtime_error("unknown option " + k);
    }
    return p;
}

} // namespace laminar
