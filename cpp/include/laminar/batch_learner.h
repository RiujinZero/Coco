#pragma once
// Batched body learner for the multi-stream trainer (multistream.h, docs/GPU_PORT_LOG.md).
//
// One call to Learner::step() is the learning of one lockstep step of S streams, in the order Organism::process_event
// learns one byte (hidden=backprop adam=1 in_struct=1 tc_window>0 ws_regs=2, no routing, no L4):
//   1. direct credit: every emitting body's exact backward pass from its readout error (learn_backprop), its gradient
//      g_j x_i added to the body's Adam buffer, and the error at its workspace input cells summed per stream (gws);
//   2. Adam steps of the bodies that became due (adam_touch / adam_step), the new INT4 weights copied to the host mirror;
//   3. temporal credit per stream (temporal_credit): gws and the register error spread over the tc_window ring,
//      write-head gradients accumulated;
//   4. the writers of each stream's oldest ring event take their delayed backward pass (temporal_backprop);
//   5. the current event is stored in each stream's ring slot.
// The host owns everything structural (which bodies exist, their INT4 weights after births and copies, use counters,
// ring bookkeeping); the learner owns the float optimiser state. RefLearner is the CPU reference, written to be
// bit-identical to organism.cpp at one stream (-ffp-contract=off); the CUDA learner is the port.
#include <cstdint>
#include <memory>
#include <string>
#include "config.h"

namespace laminar {

// A body's learning record (the fields of Trace that learn_backprop / temporal_backprop read).
constexpr int MS_MAXB = 256;       // bodies per stream per byte in the multi-stream trainer (fixed motifs: one per population)
constexpr int MS_MAXM = 256;       // ws_split readout channels (motif populations)

struct LTrace {
    uint32_t unit; uint16_t stream; uint8_t exact; uint8_t slot;   // slot: ring slot (temporal jobs)
    float scale, gain;                                             // gradient scale (goal_w, or goal_w x tc_gain) and emission gain
    int8_t x2[Dims::IN2], x3[Dims::IN3], x5[Dims::IN5], x6[Dims::IN6];
    int8_t l1[Dims::A], l2[Dims::B], l3[Dims::C], l5[Dims::E], l6[Dims::F];
    uint8_t sh[NUM_LAYERS];
    uint8_t flags;    // LF_OUT: the body emitted (its output takes the readout error; a writer for temporal credit). Without it a
                      // direct job only carries its receivers' message gradient into its L2 (bp_route: every body past L2 learns)
    uint8_t sender;   // bp_route: a receiver's sender is the direct job this many places later in the same step (0: none). The
                      // receiver's L3 read the sender's L2, so d loss / d (sender L2) = W3[:, A..] g3 goes back there
};
enum : uint8_t { LF_OUT = 1, LF_L6 = 2, LF_STRM = 4, LF_NOWS = 8 };   // LF_NOWS: ws_emit_max - its channel's error only;   // LF_L6: its L6 is in the byte's stream (motif_stream: it takes the later
                                                      // bodies' stream gradient); LF_STRM: its L3 message is the stream (sender = its position)
static_assert(sizeof(LTrace) == 264, "LTrace layout is shared with the CUDA learner");

// Per stream, per step.
struct LStream {
    float dl6[Dims::F];              // d loss / d ws (the readout error through wout), before the emission gain
    float greg_ro[4];                // d loss / d register from the register readout
    float dl6m[MS_MAXM][Dims::F];    // ws_split: each motif's own readout error (a direct job's motif is its slot)
    float goal_w, gscale;            // goal weight; workspace input scale in_scale / ws_scale
    float reg_now[4];                // registers at learning time (the stored slot's reg_prev)
    // the write head's values for ring slot h_slot (the previous step's event), written after that event's learning
    float h_reg_prev[4], h_G[4], h_g[4], h_v[4], h_u[4], h_a[Dims::F];
    uint16_t head, fill;             // ring position of this event and events already in the ring (before this step)
    uint16_t h_slot, h_valid;
    uint32_t active;                 // 0: the stream has ended (nothing to learn, ring untouched)
    int8_t x1[Dims::IN], x1e[Dims::IN];   // the event's L1 input (generic bodies) and its exact-key input
};
static_assert(sizeof(LStream) % 4 == 0, "LStream is copied as a block");

struct AdamDue { uint32_t unit, t; float n_uses; };   // one Adam step: bias-correction count and the uses it averages

struct LearnConfig {
    uint32_t units = 0, streams = 1, tc_window = 64;
    float decay = 0.5f;              // ws_decay / 256
    float bp_leak = 0.1f;
    float adam_lr = 0.003f, adam_wd = 0.f;
    float reg_scale = 50.f;
    float tc_gain = 1.f;
    uint32_t split = 0;              // ws_split: direct jobs' output error is (dl6 + dl6m[job.slot]) x gain
    uint32_t muon = 0;               // muon: the bodies' step is Organism::muon_step (momentum muon_mom, lr muon_lr) instead of Adam
    float muon_lr = 0.02f, muon_mom = 0.95f;
};

// A ring injection: this step, the stream's slot of the event `age` events back gets g added to its accumulated future error
// (tiles: a word tile's gradient to the writers of the event that closed it). Applied in order, before the ring's own credit.
struct LInject { uint32_t stream; uint32_t age; float g[Dims::F]; };

struct StepIn {
    const LStream* streams = nullptr;          // [cfg.streams]
    const LTrace* direct = nullptr; uint32_t ndirect = 0;      // per stream in the order Organism visits them
    const AdamDue* due = nullptr; uint32_t ndue = 0;
    const LTrace* temporal = nullptr; uint32_t ntemporal = 0;  // writers of each full stream's oldest slot
    const uint32_t* dirty = nullptr; uint32_t ndirty = 0;      // units whose host weights changed (births, imprints)
    const uint32_t* reset = nullptr; uint32_t nreset = 0;      // units given fresh optimiser state (birth_unit_weights)
    const LInject* inj = nullptr; uint32_t ninj = 0;          // ring injections, grouped by stream in stream order
};

class Learner {
public:
    virtual ~Learner() = default;
    virtual std::string name() const = 0;
    virtual void step(const StepIn& in) = 0;                   // host weights of stepped units are current on return
    // Asynchronous form: submit() may return before the work is done (it copies what it needs from `in`); wait()
    // completes it and writes the stepped bodies' weights to the host. Default: synchronous.
    virtual void submit(const StepIn& in) { step(in); }
    virtual void wait() {}
    virtual void upload_all() = 0;                             // after structural maintenance: the host weights are the truth
    virtual void upload_units(const uint32_t* ids, uint32_t n) = 0;   // these units' host weights are the truth
    virtual void set_head(const float* hw_g, const float* hw_v, const float* hb) = 0;   // [4][F], [4][F], [4]
    virtual void take_head_grads(float* hd) = 0;               // add the accumulated head gradients (8F + 4) into hd, zero them
    // diagnostics / validation
    virtual void read_unit_state(uint32_t u, float* sh, float* m, float* v, float* g) = 0;   // 5632 each (null = skip)
    virtual void read_ring_tg(uint32_t stream, uint32_t slot, float* tg) = 0;
    virtual double seconds_in_step() const { return 0.0; }
    virtual std::string report() const { return {}; }
};

enum class LearnerKind { Reference, Gpu, Dual, None };   // None: no learning (scoring)

std::unique_ptr<Learner> make_ref_learner(const LearnConfig& c, uint8_t* host_weights);
std::unique_ptr<Learner> make_gpu_learner(const LearnConfig& c, uint8_t* host_weights);   // throws when built without CUDA
// validation: the reference drives the host weights; the GPU learner runs the same steps on its own copy and is compared
// after every step (stepped bodies' weights and shadows, ring errors, head gradients), its drifted bodies re-synced
std::unique_ptr<Learner> make_dual_learner(const LearnConfig& c, uint8_t* host_weights);
std::unique_ptr<Learner> make_learner(LearnerKind k, const LearnConfig& c, uint8_t* host_weights);

} // namespace laminar
