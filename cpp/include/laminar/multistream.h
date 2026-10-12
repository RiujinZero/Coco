#pragma once
// Multi-stream lockstep trainer: S byte streams (documents) read in parallel by ONE organism (docs/GPU_PORT_LOG.md).
//
// Shared by all streams (held in an Organism used as the container, so checkpoints and scoring keep working): the body
// population and its INT4 weights, locks and address tree, structural evidence, unit statistics, the float readout,
// the register readout, the write head and the sequence-memory gate. Per stream: history, workspace memory,
// registers, the sequence memory, the agency kernel, running loss/prior, goal tracking and the temporal-credit ring.
//
// A step advances every active stream by one byte, in phases that reproduce Organism::process_event:
//   P1 (threads)  key, input, address lookup, agency deliberation
//   S1 (serial)   births for streams without candidates, lock hits / novelty, the event's body list
//   P2 (threads)  body forwards (the CPU INT4 kernels: bit-exact by construction)
//   S2 (serial)   death thresholds and activity averages, emission gating
//   P3 (threads)  workspace, readout, sequence-memory blend, loss, counterfactual route gains, agency consequence,
//                 the readout's own gradients (learn_l6) into per-thread partials
//   S3 (serial)   partials reduced, readout Adam step when due; job lists; Learner::step (bodies: GPU or reference)
//   S4 (serial)   metabolic, assoc, episodic births, lock evidence and utility, route utilities
//   P4 (threads)  workspace memory decay, write head, sequence capture, history
// Serial phases walk the streams in index order, so S=1 is Organism's order exactly. Differences at S>1 are those of a
// minibatch: all streams of a step see the weights of the step's start, and a body's Adam batch collects its uses
// across streams (a body used by more than adam_k streams in one step takes one step over all of them).
//
// Supported configuration: the BabyLM body-learning path (vocab=256, in_struct, hidden=backprop adam=1 ro_adam=1,
// tc_window>0, ws_regs=2 reg_readout=1, hops=0, no recirculation / persist / routing / homeo / market / composer).
// The constructor throws on anything else.
#include "organism.h"
#include "batch_learner.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace laminar {

// Fork-join pool: run(n, f) calls f(i, worker) for i in [0, n) on all workers (the caller is worker 0).
class WorkPool {
public:
    explicit WorkPool(unsigned threads);
    ~WorkPool();
    template <class F> void run(uint32_t n, F&& f) { fn_ = [&](uint32_t i, unsigned w) { f(i, w); }; dispatch(n); }
    unsigned size() const { return (unsigned)workers_.size() + 1; }
private:
    void dispatch(uint32_t n);
    void work(unsigned w);
    std::vector<std::thread> workers_;
    std::function<void(uint32_t, unsigned)> fn_;
    std::atomic<uint32_t> next_{0}, n_{0}, done_{0}; std::atomic<uint64_t> gen_{0}; std::atomic<bool> stop_{false};
    std::mutex mu_; std::condition_variable cv_;
};

class MultiStream {
public:
    // model_params: key=value strings as for Organism (the same defaults model_babylm prepends apply).
    MultiStream(const Params& p, uint32_t streams, LearnerKind learner, unsigned threads);
    ~MultiStream();

    // Streams read their own byte sequences. set_stream_data keeps a pointer (the caller owns the bytes); a stream's
    // first event predicts data[0] from its current last symbol, like feed_bytes does for a single stream.
    void set_stream_data(uint32_t s, const uint8_t* data, size_t n);
    void set_stream_weights(uint32_t s, const float* w);   // per-byte learning weights aligned with the data (nullptr: all 1); caller owns them
    bool any_active() const;
    void step();                                   // every active stream advances one byte
    uint64_t events() const { return events_; }
    uint32_t streams() const { return S_; }
    void end_block();                              // structural maintenance (Organism::end_block) + weight re-upload
    void set_train(bool train, bool structural);
    // overlap=true: the learner's work for step t runs while the CPU does step t+1 (a body's Adam step reaches the
    // host forward one step late). Off by default; S=1 exactness needs it off.
    void set_overlap(bool on) { overlap_ = on; }
    void sync() { if (learner_) learner_->wait(); }

    // Statistics since the last reset
    struct Stats { double loss_sum = 0; uint64_t events = 0, correct = 0, steps = 0, adam_steps = 0, adam_uses = 0, direct_jobs = 0, temporal_jobs = 0, births = 0, episodic_births = 0;
                   double s3_reduce = 0, s3_ro = 0, s3_build = 0, s3_ring = 0, sec_block = 0, sec_upload = 0, sec_p1 = 0, sec_s1 = 0, sec_p2 = 0, sec_s2 = 0, sec_p3 = 0, sec_s3 = 0, sec_learn = 0, sec_s4 = 0, sec_p4 = 0; };
    Stats& stats() { return st_; }
    std::string learner_name() const { return learner_->name(); }
    Learner& learner() { return *learner_; }
    Organism& organism() { return org_; }
    // Per-event observer (validation): stream, symbol, target, loss, prediction.
    std::function<void(uint32_t, uint8_t, uint8_t, float, uint32_t)> on_event;
    // Diagnostic arms, as Organism's: rewrite a body's L6 after its threshold (must be thread-safe: phases run in parallel
    // when scoring), and hide the workspace memory and registers from the bodies' input.
    std::function<void(uint32_t unit, int8_t* l6)> l6_hook; bool diag_zero_wsin = false;
    bool diag_zero_msgs = false;   // Organism::diag_zero_msgs: motif_wire's L2 -> L3 messages cut (forward and gradient)
    // Write stream s's state into the container organism (for a checkpoint that model_babylm_score can read).
    void export_stream(uint32_t s);
    // after loading a saved state into organism() (e.g. circuits pre-trained on curated data): the learner takes its weights
    // and write head; the streams and the optimiser state start fresh
    void reload_from_container();

    // Batched scoring (model_babylm_score without fork). Load a checkpoint into organism(), freeze it and read whatever
    // prefix the protocol wants through Organism itself (model_babylm_score: one newline); then score(): every item is
    // read by a stream that starts from a copy of that frozen stream state. What evaluation changes in shared state (lock
    // hits and error rates, unit activity averages, route utilities) goes to the stream's private overlay, so no item
    // sees another: each result equals the forked scorer's. Returns per item the completion's summed -log p (nats).
    struct ScoreItem { std::string context, completion; };
    std::vector<double> score(const std::vector<ScoreItem>& items);

private:
    struct Stream;
    Params p_; uint32_t S_; Organism org_;
    std::unique_ptr<Learner> learner_;
    std::vector<std::unique_ptr<Stream>> st_s_;
    WorkPool pool_;
    uint64_t events_ = 0, ro_events_ = 0, ro_next_ = 0;
    Stats st_;
    // host optimiser bookkeeping (Organism's ad_uses / ad_t / ad_last)
    std::vector<uint32_t> ad_uses_, ad_t_, ad_last_;
    // per-step batches
    std::vector<LStream> lstreams_; std::vector<LTrace> direct_, temporal_; std::vector<AdamDue> due_; std::vector<LInject> inj_;
    std::vector<uint32_t> dirty_, reset_;
    std::vector<uint32_t> step_uses_, touched_units_;
    // readout gradient partials per worker
    struct Partial { std::vector<float> ro, rg; float sg[6]; bool sg_any; std::vector<float> tm, ta, tf, tr, wsp, wtl; };   // tm / ta / tf / tr: tiles' gradients; wsp: ws_split
    std::vector<Partial> part_;
    uint32_t in_scale_; bool reg_readout_; bool overlap_ = false;

    void check_config() const;
    void p1_address(Stream& s);
    void s1_admit(Stream& s);
    void stream_msg(const Stream& s, uint32_t k, int8_t* out, bool all) const;   // motif_stream: body k's L3 message
    void p2_forward(Stream& s);
    void s2_thresholds(Stream& s);
    void p3_readout(Stream& s, Partial& part);
    void s3_learn();
    void s4_structure(Stream& s);
    void s4_learned(Stream& s);
    void p4_advance(Stream& s);
    void readout(const Stream& s, const int32_t* w, float* logits, float* probs) const;
    void episodic_birth(Stream& s, uint32_t parent_lock);
    // evaluation overlays (eval_): reads and writes of shared state that evaluation changes
    bool eval_ = false;
    uint32_t lk_hits(const Stream& s, uint32_t id) const;
    uint32_t lk_err(const Stream& s, uint32_t id) const;
    int32_t threshold(Stream& s, int32_t base, uint32_t unit, bool novel, int stage, int32_t level);
    float& route_util(Stream& s, uint32_t unit);
    void import_stream(Stream& s);
    bool imprint(Stream& s, uint32_t unit);
    std::vector<std::pair<Stream*, uint32_t>> births_;
    void agency_init(Stream& s);
};

} // namespace laminar
