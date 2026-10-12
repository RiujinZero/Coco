// Multi-stream lockstep trainer (multistream.h). Every phase is organism.cpp's process_event for the supported
// configuration, with the stream's own state in place of the organism's stream members; the comments name the
// Organism code each block follows. Keep this file in step with organism.cpp: the S=1 check (apps/ms_check.cpp,
// mode exact) fails on any divergence.
#include "laminar/multistream.h"
#include <algorithm>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace laminar {

using Clock = std::chrono::steady_clock;
static inline double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
static constexpr uint64_t SENSORY_MASK = (1ULL << Dims::SENSORY_BITS) - 1;
static constexpr int SEQ_GATE_F = 6;
static constexpr int AG_OBS = 64;

// ---- WorkPool ----------------------------------------------------------------------------------------------------
WorkPool::WorkPool(unsigned threads) { for (unsigned w = 1; w < std::max(1u, threads); ++w) workers_.emplace_back([this, w] { work(w); }); }
WorkPool::~WorkPool() {
    { std::lock_guard<std::mutex> l(mu_); stop_ = true; gen_.fetch_add(1); } cv_.notify_all();
    for (auto& t : workers_) t.join();
}
void WorkPool::dispatch(uint32_t n) {
    if (workers_.empty() || n <= 1) { for (uint32_t i = 0; i < n; ++i) fn_(i, 0); return; }
    next_.store(0); n_.store(n); done_.store(0);
    { std::lock_guard<std::mutex> l(mu_); gen_.fetch_add(1, std::memory_order_release); } cv_.notify_all();
    for (;;) { const uint32_t i = next_.fetch_add(1); if (i >= n) break; fn_(i, 0); }
    while (done_.load(std::memory_order_acquire) < workers_.size()) {
#if defined(__AVX2__)
        _mm_pause();
#endif
    }
}
void WorkPool::work(unsigned w) {
    uint64_t seen = 0;
    for (;;) {
        // spin (the next phase usually follows within a GPU round trip) and only then sleep: waking a sleeping thread
        // costs tens of microseconds per phase
        const auto t0 = std::chrono::steady_clock::now(); uint32_t spins = 0;
        while (gen_.load(std::memory_order_acquire) == seen) {
#if defined(__AVX2__)
            _mm_pause();
#endif
            if ((++spins & 255) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(5)) {
                std::unique_lock<std::mutex> l(mu_); cv_.wait(l, [&] { return gen_.load() != seen; }); }
        }
        seen = gen_.load(std::memory_order_acquire);
        if (stop_) return;
        const uint32_t n = n_.load();
        for (;;) { const uint32_t i = next_.fetch_add(1); if (i >= n) break; fn_(i, w); }
        done_.fetch_add(1, std::memory_order_release);
    }
}

// ---- helpers copied from organism.cpp (file-static there) -----------------------------------------------------------
static inline float dot_n(const float* a, const float* b, size_t n) { float s = 0.f; for (size_t i = 0; i < n; ++i) s += a[i] * b[i]; return s; }   // ws_split (Organism's)
static inline float dot_f(const float* a, const float* b) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s = _mm256_setzero_ps();
    for (int k = 0; k < Dims::F; k += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), s);
    __m128 lo = _mm256_castps256_ps128(s), hi = _mm256_extractf128_ps(s, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo)); lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
#else
    float s = 0.f; for (int k = 0; k < Dims::F; ++k) s += a[k] * b[k]; return s;
#endif
}
#if defined(__AVX2__) && defined(__FMA__)
static inline __m256 log8(__m256 x) {
    __m256i xi = _mm256_castps_si256(x);
    __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
    __m256 m = _mm256_or_ps(_mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFF))), _mm256_set1_ps(1.0f));
    __m256 big = _mm256_cmp_ps(m, _mm256_set1_ps(1.41421356f), _CMP_GT_OQ);
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), big);
    __m256 ef = _mm256_add_ps(_mm256_cvtepi32_ps(e), _mm256_and_ps(big, _mm256_set1_ps(1.0f)));
    __m256 f = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    __m256 pl = _mm256_set1_ps(1.0f / 7.0f);
    pl = _mm256_fmadd_ps(pl, f, _mm256_set1_ps(-1.0f / 6.0f)); pl = _mm256_fmadd_ps(pl, f, _mm256_set1_ps(1.0f / 5.0f));
    pl = _mm256_fmadd_ps(pl, f, _mm256_set1_ps(-1.0f / 4.0f)); pl = _mm256_fmadd_ps(pl, f, _mm256_set1_ps(1.0f / 3.0f));
    pl = _mm256_fmadd_ps(pl, f, _mm256_set1_ps(-0.5f)); pl = _mm256_fmadd_ps(pl, f, _mm256_set1_ps(1.0f)); pl = _mm256_mul_ps(pl, f);
    return _mm256_fmadd_ps(ef, _mm256_set1_ps(0.69314718f), pl);
}
static inline __m256 exp8(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.f));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145751953125f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428606765330187e-06f), r);
    __m256 pl = _mm256_set1_ps(1.f / 720.f);
    pl = _mm256_fmadd_ps(pl, r, _mm256_set1_ps(1.f / 120.f));
    pl = _mm256_fmadd_ps(pl, r, _mm256_set1_ps(1.f / 24.f));
    pl = _mm256_fmadd_ps(pl, r, _mm256_set1_ps(1.f / 6.f));
    pl = _mm256_fmadd_ps(pl, r, _mm256_set1_ps(0.5f));
    pl = _mm256_fmadd_ps(pl, r, _mm256_set1_ps(1.f));
    pl = _mm256_fmadd_ps(pl, r, _mm256_set1_ps(1.f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(pl, _mm256_castsi256_ps(e));
}
#endif
static inline void byte_features(uint8_t c, int8_t S, int8_t* out) {
    for (int b = 0; b < 8; ++b) out[b] = ((c >> b) & 1) ? S : (int8_t)-S;
    const bool alpha = std::isalpha(c) != 0, upper = std::isupper(c) != 0, digit = std::isdigit(c) != 0, space = c == ' ' || c == '\t';
    const bool punct = std::ispunct(c) != 0 && c != '\'' && c != '"'; const int lower = std::tolower(c);
    const bool vowel = alpha && (lower == 'a' || lower == 'e' || lower == 'i' || lower == 'o' || lower == 'u');
    const bool f[8] = {alpha, upper, digit, space, punct, vowel, c == '\n' || c == '\r', c == '\'' || c == '"'};
    for (int k = 0; k < 8; ++k) out[8 + k] = f[k] ? S : (int8_t)-S;
}
static inline int32_t level(const int8_t* v, int n) { int32_t s = 0; for (int i = 0; i < n; ++i) s += v[i] < 0 ? -v[i] : v[i]; return s / n; }

// ---- per-stream state ----------------------------------------------------------------------------------------------
struct MultiStream::Stream {
    uint32_t id = 0;
    const uint8_t* data = nullptr; size_t len = 0, pos = 0; bool active = false;
    const float* wts = nullptr;      // per-byte learning weights (set_stream_weights), multiplied into goal_w
    uint64_t n_events_global = 0;   // Organism's event_idx for this event (the readout freeze counts all streams' events)
    // Organism's stream members
    uint8_t hist[16] = {0}; uint64_t n_events = 0;
    int32_t ws_mem[Dims::F] = {0}; int32_t ws[Dims::F] = {0};
    float loss_ema = 0.f; bool loss_ema_init = false; uint32_t wrong_ema = 65535;
    std::vector<float> prior, logits, probs, cf_logits, cf_probs;
    SequenceMemory seq; std::vector<float> seq_out, body_probs; bool seq_on = false;
    uint64_t seq_events = 0, seq_matched = 0, seq_bubble_events = 0; double seq_conf_sum = 0;
    int8_t xin[Dims::IN] = {0}, xin_exact[Dims::IN] = {0}, xin_focus[Dims::IN] = {0};
    agency::AgencyKernel agent; bool agent_on = false; std::vector<float> agent_obs; int agent_tendency = -1; bool agent_exact_only = false; float agent_last_ptarget = 0.f;
    float ws_scale = 0.f, ws_reg[4] = {0, 0, 0, 0}, ws_reg_prev[4] = {0, 0, 0, 0}, ws_regG[4] = {0, 0, 0, 0};
    int phase = 0, ans_state = 0; float goal_w = 1.f; double ans_loss = 0.0; int ans_n = 0; float ans_ema = -1.f;
    float body_resp = 1.f;
    // temporal-credit ring bookkeeping (the float ring lives in the learner)
    uint16_t tc_head = 0, tc_fill = 0; std::vector<std::vector<LTrace>> ring_tr; std::vector<float> ring_goal_w;
    float h_reg_prev[4] = {0}, h_G[4] = {0}, h_g[4] = {0}, h_v[4] = {0}, h_u[4] = {0}, h_a[Dims::F] = {0}; bool h_valid = false; uint16_t h_slot = 0;
    // this event
    uint8_t sym = 0, target = 0; uint64_t key = 0; uint32_t cand[16]; uint32_t ncand = 0; uint8_t cur_flags = 0; bool novel = false;
    uint32_t nbody = 0; uint32_t body[MS_MAXB]; uint8_t body_exact[MS_MAXB]; int32_t lv[MS_MAXB][4];
    Trace tr[MS_MAXB];
    int8_t l6e[MS_MAXB][Dims::F]; bool hooked[MS_MAXB] = {};   // l6_hook: the emitted L6 when a diagnostic hook rewrote it
    float dl6[Dims::F] = {0}, greg_ro[4] = {0};
    float dl6t[Dims::F] = {0};   // tile_split: the tiles' channel error
    int32_t wsm[MS_MAXM][Dims::F] = {{0}}; float dl6m[MS_MAXM][Dims::F] = {{0}};   // ws_split: each motif's emission (body k is motif k) and own error
    uint32_t pred = 0; float loss = 0.f; bool correct = false, consequential = false;
    float route_gain[MS_MAXB]; uint8_t route_has[MS_MAXB];
    std::vector<uint32_t> err_idx; std::vector<float> err_val;
    std::vector<LTrace> jobs;   // this event's direct jobs (Organism's visiting order)
    std::vector<LTrace> tjobs;  // delayed jobs: the writers of the event leaving the ring
    TileState ts;               // tiles: this stream's pyramid (Organism's tst_)
    std::vector<LInject> inj;   // tiles: this event's ring injections (word tiles' credit), in Organism's order
    // evaluation: private copies of the shared state this stream changed, and the item being scored
    struct LockOv { uint32_t hits, err_ema; }; struct UnitOv { uint8_t act[4]; float ru; bool act_set = false, ru_set = false; };
    std::unordered_map<uint32_t, LockOv> lov; std::unordered_map<uint32_t, UnitOv> uov;
    std::string buf; size_t ctx_len = 0; double item_loss = 0.0; int64_t item = -1;
    uint32_t wrong = 0; float gain_bits = 0.f;
};

// ---- construction ----------------------------------------------------------------------------------------------------
static Params container_params(Params p) { p.train = false; return p; }   // no INT8/INT16 residuals or Adam state on the host

MultiStream::MultiStream(const Params& p, uint32_t streams, LearnerKind kind, unsigned threads)
    : p_(p), S_(streams), org_(container_params(p)), pool_(threads) {
    if (!p_.fm_h.empty()) { p_.structural = false; p_.episodic = false; p_.agency = 0; p_.market = false; }   // fixed motifs (as Organism)
    check_config();
    org_.set_mode(p_.train, p_.structural); org_.p.train = p_.train;
    in_scale_ = (uint32_t)std::min(127, std::max(1, p_.in_scale)); reg_readout_ = p_.reg_readout;
    // Shared learning state that Organism allocates lazily; allocating it up front is equivalent (zero gradients step
    // nothing, a zero register readout adds nothing).
    const uint32_t V = p_.vocab;
    org_.wreg.assign((size_t)V * 4, 0.f);
    org_.ro_g.assign((size_t)V * Dims::F, 0.f); org_.ro_m = org_.ro_g; org_.ro_v = org_.ro_g;
    org_.rg_g.assign((size_t)V * 4, 0.f); org_.rg_m = org_.rg_g; org_.rg_v = org_.rg_g;
    org_.hd_g.assign(4 * Dims::F * 2 + 4, 0.f); org_.hd_m = org_.hd_g; org_.hd_v = org_.hd_g;
    { Rng hr(mix64(p_.seed ^ 0x4EADULL));   // the write head, as Organism initialises it on first use
      for (int r = 0; r < 4; ++r) { org_.hb_g[r] = p_.reg_bias; for (int j = 0; j < Dims::F; ++j) { org_.hw_g[r][j] = (float)(hr.uniform() * 2 - 1) * 0.1f; org_.hw_v[r][j] = (float)(hr.uniform() * 2 - 1) * 0.1f; } }
      org_.head_init = true; }
    ad_uses_.assign(p_.units, 0); ad_t_.assign(p_.units, 0); ad_last_.assign(p_.units, 0); step_uses_.assign(p_.units, 0);
    LearnConfig lc; lc.units = p_.units; lc.streams = S_; lc.tc_window = p_.tc_window; lc.decay = (float)p_.ws_decay / 256.f; lc.bp_leak = p_.bp_leak;
    lc.adam_lr = p_.adam_lr; lc.adam_wd = p_.adam_wd; lc.reg_scale = p_.reg_scale; lc.tc_gain = p_.tc_gain; lc.split = p_.ws_split ? (uint32_t)p_.fm_h.size() : 0;
    lc.muon = p_.muon; lc.muon_lr = p_.muon_lr; lc.muon_mom = p_.muon_mom;
    learner_ = make_learner(kind, lc, org_.wbuf.p);
    if (learner_) { learner_->upload_all(); learner_->set_head(&org_.hw_g[0][0], &org_.hw_v[0][0], org_.hb_g); }
    for (uint32_t s = 0; s < S_; ++s) {
        auto st = std::make_unique<Stream>(); st->id = s; Stream& x = *st;
        x.prior.assign(V, 1.f / V); x.logits.assign(V, 0.f); x.probs.assign(V, 0.f); x.cf_logits.assign(V, 0.f); x.cf_probs.assign(V, 0.f);
        if (p_.seqmem && V == 256) {   // Organism::Organism, the sequence memory of the supported configuration
            x.seq.init(p_.seq_window, p_.seq_hash_bits); x.seq.min_support = p_.seq_min_support; x.seq.copy_max = p_.seq_copy_max; x.seq_out.assign(256, 0.f); x.body_probs.assign(256, 0.f); x.seq_on = true;
            if (p_.seq_consolidate) { x.seq.init_consolidated(p_.seq_cbits); x.seq.class_gain = p_.seq_class_gain; }
            x.seq.goal_extend = p_.seq_goal_extend; x.seq.util_rate = p_.seq_util_rate;
        }
        if (p_.agency) agency_init(x);
        x.ring_tr.assign(p_.tc_window, {}); x.ring_goal_w.assign(p_.tc_window, 1.f);
        x.ts = org_.tst_;
        st_s_.push_back(std::move(st));
    }
    part_.resize(pool_.size());
    for (auto& pt : part_) { pt.ro.assign((size_t)V * Dims::F, 0.f); pt.rg.assign((size_t)V * 4, 0.f); std::fill(pt.sg, pt.sg + 6, 0.f); pt.sg_any = false;
        pt.tm.assign(org_.tile_Mg.size(), 0.f); pt.ta.assign(org_.tile_Ag.size(), 0.f); pt.tf.assign(org_.fw_Wg.size(), 0.f); pt.tr.assign(org_.ret_Wg.size(), 0.f); pt.wsp.assign(org_.wsp_g.size(), 0.f); pt.wtl.assign(org_.wtl_g.size(), 0.f); }
    lstreams_.resize(S_);
    ro_next_ = std::max(1u, p_.ro_adam_k);
}
MultiStream::~MultiStream() = default;

void MultiStream::check_config() const {
    auto need = [](bool ok, const char* what) { if (!ok) throw std::runtime_error(std::string("multistream: unsupported configuration: ") + what); };
    need(p_.vocab == 256, "vocab must be 256"); need(p_.in_struct && !p_.in_words, "in_struct=1 in_words=0"); need(p_.ws_input && !p_.ws_bands, "ws_input=1 ws_bands=0");
    need(p_.hidden == HiddenLearning::Backprop && p_.adam && p_.ro_adam, "hidden=backprop adam=1 ro_adam=1");
    need(p_.tc_window > 0 && p_.tc_window < 65536 && p_.ws_regs == 2 && p_.reg_readout, "tc_window>0 ws_regs=2 reg_readout=1");
    need(p_.max_hops == 0, "hops=0"); need(p_.recirc_thresh > 127 && p_.recirc_iters == 0 && !p_.recirc_learn, "recirculation off (recirc>127)");
    need(!p_.persist && !p_.route_return && !p_.shared_types && !p_.shared_always, "persist / route_return / shared types off");
    need(!p_.homeo && !p_.market && !p_.levels && !p_.episodic_promote && p_.goal_elig == 0.f && !p_.agency_task, "homeo / market / composer / promotion / goal credit / task agency off");
    need(!p_.seq_chain && p_.seq_topic_gain == 0.f && !p_.seq_attn && !p_.seq_units && p_.seq_bubble_min >= (1 << 30), "sequence-memory extensions off");
    need(!p_.l4_key && !p_.ws_key && !p_.pc_settle && !p_.bp_ste && p_.ws_decay > 0, "l4_key / ws_key / pc_settle / bp_ste off, ws_decay>0");
    need(!p_.msg_merge, "msg_merge off");
    need(!p_.apical && !p_.mtp, "apical / mtp (organism only so far)");
    need(!p_.fm_stream || (p_.bp_route && p_.fm_wire && !p_.fm_h.empty()), "motif_stream needs motifs and bp_route=1");
    need(!p_.tiles || p_.tile_avg, "tiles need tile_avg=1 (learned merge bodies are organism-only)");
    need(!p_.ws_split || (!p_.fm_h.empty() && p_.fm_h.size() <= (size_t)MS_MAXM), "ws_split needs 1..256 motifs");
    need(p_.fm_h.size() <= (size_t)MS_MAXB, "at most 256 fixed-motif populations (bodies per byte)");
    need(!p_.ws_emit_max || (p_.ws_split && !p_.fm_stream && !p_.bp_route), "ws_emit_max needs ws_split (and no motif_stream / bp_route)");
    need(!p_.ro_hidden && !p_.ws_split5, "ro_hidden / ws_split5 (organism only so far)");
    need(p_.disinh == 0.f, "disinh (organism only so far)");
    need(!p_.iter_hops, "iter_hops (organism only so far)");
    need(p_.compete == 0.f && !p_.motif_mix && !p_.motif_attn, "compete / motif_mix / motif_attn (organism only so far)");
    need(p_.seqmem, "seqmem=1");
    need(!p_.reg_center && !p_.reg_revive && !p_.head_norm && p_.reg_ro_init == 0.f && p_.adam_l6, "reg_center / reg_revive / head_norm / reg_ro_init / adam_l6=0 - not yet");
}

void MultiStream::agency_init(Stream& s) {   // Organism::agency_init
    agency::KernelConfig c;
    c.obs_dim = AG_OBS; c.n_goals = 2; c.consequence_dim = 4; c.n_units = (int)p_.agency_units; c.k_active = (int)p_.agency_k;
    c.trace_lifetime = p_.agency_trace_life;
    c.focus_dim = Dims::IN; c.focus_width = p_.agency_focus_width; c.focus_floor = p_.agency_focus_floor; c.focus_active = true;
    c.route_lr = p_.agency_route_lr; c.arousal = p_.agency_arousal; c.baseline_cost = p_.agency_cost; c.epistemic_channel = 1; c.seed = p_.agency_seed ^ p_.seed;
    std::vector<std::string> names; size_t q = 0; const std::string& L = p_.agency_ops;
    while (q <= L.size()) { size_t e = L.find(',', q); if (e == std::string::npos) e = L.size(); if (e > q) names.push_back(L.substr(q, e - q)); q = e + 1; }
    if (p_.agency == 2) names = {"hold"};
    c.n_tendencies = (int)names.size();
    s.agent.init(c);
    for (auto& n : names) {
        agency::FocusOp op;
        if (n == "hold") op = agency::op_hold;
        else if (n == "toward_error") op = agency::op_toward_error;
        else if (n == "sweep") op = agency::op_sweep;
        else if (n == "to_recent") op = agency::op_move_to(20.f);
        else if (n == "to_context") op = agency::op_move_to(48.f);
        else if (n == "to_symbol") op = agency::op_move_to(80.f);
        else if (n == "to_memory") op = agency::op_move_to(112.f);
        else if (n == "attend_episodic") op = agency::op_attend(1);
        else if (n == "attend_generic") op = agency::op_attend(0);
        else throw std::runtime_error("unknown agency op " + n);
        s.agent.ops.push_back(op); s.agent.op_names.push_back(n);
    }
    s.agent.set_goal(0, 1.0f); s.agent.set_goal(1, p_.agency_epistemic);
    s.agent_obs.assign(AG_OBS, 0.f); s.agent_on = true;
    std::memcpy(s.xin_focus, s.xin, Dims::IN);
}

void MultiStream::set_stream_data(uint32_t s, const uint8_t* data, size_t n) { Stream& x = *st_s_[s]; x.data = data; x.len = n; x.pos = 0; x.active = n > 0; x.wts = nullptr; }
void MultiStream::set_stream_weights(uint32_t s, const float* w) { st_s_[s]->wts = w; }
bool MultiStream::any_active() const { for (auto& s : st_s_) if (s->active) return true; return false; }
void MultiStream::set_train(bool train, bool structural) { structural = structural && org_.fm_base.empty(); p_.train = train; p_.structural = structural; org_.set_mode(train, structural); }

// ---- readout (Organism::readout with the stream's registers) ---------------------------------------------------------
void MultiStream::readout(const Stream& s, const int32_t* w, float* out_logits, float* out_probs) const {
    const uint32_t V = p_.vocab; const float* woutT = org_.woutT.data(); const float* wout = org_.wout.data();
    alignas(32) float a[Dims::F]; for (int k = 0; k < Dims::F; ++k) a[k] = (float)w[k] * (1.0f / 128.0f);
    if (!s.ts.q.empty()) for (int k = 0; k < Dims::F; ++k) a[k] += s.ts.a[k];   // tiles (Organism::readout)
#if defined(__AVX2__) && defined(__FMA__)
    {
        uint32_t v = 0;
        for (; v + 32 <= V; v += 32) {
            __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps(), c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
            for (int k = 0; k < Dims::F; ++k) { const float* r = &woutT[(size_t)k * V + v]; __m256 b = _mm256_set1_ps(a[k]);
                c0 = _mm256_fmadd_ps(b, _mm256_loadu_ps(r), c0); c1 = _mm256_fmadd_ps(b, _mm256_loadu_ps(r + 8), c1);
                c2 = _mm256_fmadd_ps(b, _mm256_loadu_ps(r + 16), c2); c3 = _mm256_fmadd_ps(b, _mm256_loadu_ps(r + 24), c3); }
            _mm256_storeu_ps(out_logits + v, c0); _mm256_storeu_ps(out_logits + v + 8, c1); _mm256_storeu_ps(out_logits + v + 16, c2); _mm256_storeu_ps(out_logits + v + 24, c3);
        }
        for (; v < V; ++v) out_logits[v] = dot_f(a, &wout[(size_t)v * Dims::F]);
    }
#else
    for (uint32_t v = 0; v < V; ++v) out_logits[v] = dot_f(a, &wout[(size_t)v * Dims::F]);
#endif
    if (!org_.wsp.empty()) { const size_t M = p_.fm_h.size(), MF = M * Dims::F; float b[MS_MAXM * Dims::F]; for (size_t i = 0; i < MF; ++i) b[i] = (float)s.wsm[i / Dims::F][i % Dims::F] * (1.0f / 128.0f);   // ws_split
        for (uint32_t v2 = 0; v2 < V; ++v2) out_logits[v2] += dot_n(b, &org_.wsp[(size_t)v2 * MF], MF); }
    if (!org_.wtl.empty()) for (uint32_t v2 = 0; v2 < V; ++v2) out_logits[v2] += dot_n(s.ts.a, &org_.wtl[(size_t)v2 * Dims::F], Dims::F);   // tile_split
    if (reg_readout_ && !org_.wreg.empty()) for (uint32_t v2 = 0; v2 < V; ++v2) { const float* r = &org_.wreg[(size_t)v2 * 4]; out_logits[v2] += (r[0] * s.ws_reg[0] + r[1] * s.ws_reg[1] + r[2] * s.ws_reg[2] + r[3] * s.ws_reg[3]) * 0.01f; }
    float mx = out_logits[0]; for (uint32_t v = 1; v < V; ++v) mx = std::max(mx, out_logits[v]);
    float z = 0.f; uint32_t v = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 zs = _mm256_setzero_ps(), mxs = _mm256_set1_ps(mx);
    for (; v + 8 <= V; v += 8) { __m256 e = exp8(_mm256_sub_ps(_mm256_loadu_ps(out_logits + v), mxs)); _mm256_storeu_ps(out_probs + v, e); zs = _mm256_add_ps(zs, e); }
    alignas(32) float zt[8]; _mm256_store_ps(zt, zs); for (int i = 0; i < 8; ++i) z += zt[i];
#endif
    for (; v < V; ++v) { out_probs[v] = std::exp(out_logits[v] - mx); z += out_probs[v]; }
    float inv = 1.f / z; for (v = 0; v < V; ++v) out_probs[v] *= inv;
}

// ---- P1: key, input, address, agency (process_event up to the births) -------------------------------------------------
void MultiStream::p1_address(Stream& s) {
    const Params& p = p_;
    s.sym = s.hist[0]; s.target = s.data[s.pos];
    { // Organism::goal_track (goal_gain / goal_elig are 1 / 0 in the supported configuration; the phase is kept for fidelity)
        const uint8_t sym = s.sym, target = s.target;
        auto ends_with = [&](const char* str, int L) { if (s.n_events < (uint64_t)L) return false; if (sym != (uint8_t)str[L - 1]) return false; for (int i = 1; i < L; ++i) if (s.hist[i] != (uint8_t)str[L - 1 - i]) return false; return true; };
        if (ends_with("USER:", 5)) { s.phase = 1; s.ans_state = 0; }
        else if (ends_with("ASSISTANT:", 10)) { s.phase = 2; s.ans_state = 1; s.ans_loss = 0.0; s.ans_n = 0; }
        s.goal_w = 1.f;
        if (s.ans_state) {
            const bool an = std::isalnum(target);
            if (s.ans_state == 1) { if (an) s.ans_state = 2; else if (target != ' ') s.ans_state = 3; }
            else if (s.ans_state == 2 && !an) s.ans_state = 3;
            else if (s.ans_state == 3) s.ans_state = 0;
            if (s.ans_state == 2 || s.ans_state == 3) s.goal_w = p.goal_gain;
            if (s.ans_state == 0 || (s.ans_state == 3 && !std::isalnum(target))) {
                if (s.ans_state == 0 && s.ans_n) { const float mean = (float)(s.ans_loss / s.ans_n); if (s.ans_ema < 0.f) s.ans_ema = mean; s.ans_ema += 0.02f * (mean - s.ans_ema); s.ans_n = 0; s.phase = 0; }
            }
        }
        if (s.wts) s.goal_w *= s.wts[s.pos];   // per-byte learning weight (set_stream_weights)
    }
    { // Organism::make_key
        const uint8_t* hist = s.hist;
        uint64_t k = hist[0] | ((uint64_t)hist[1] << 8) | ((uint64_t)hist[2] << 16) | ((uint64_t)hist[3] << 24);
        uint64_t h = 0x5A5A00000000ULL;
        for (uint32_t i = 1; i <= p.hist_bytes && i < 16; ++i) h = (h * 0x100000001B3ULL) ^ hist[i];
        s.key = k | (mix64(h) & 0xFFFFFFFF00000000ULL);
    }
    { // Organism::make_input (in_struct, ws_input, no word embeddings, no bands, registers)
        const uint8_t sym = s.sym; const uint8_t* hist = s.hist; int8_t* xin = s.xin;
        const int8_t S = (int8_t)std::min(127, std::max(1, p.in_scale));
        for (int i = 0; i < 5; ++i) byte_features(i == 0 ? sym : hist[i], S, xin + 16 * i);
        int pos = 0; while (pos < 15 && std::isalpha(pos == 0 ? sym : hist[pos])) ++pos;
        int j = pos; while (j < 15 && !std::isalpha(hist[j])) ++j; int prev = 0; while (j + prev < 15 && std::isalpha(hist[j + prev])) ++prev;
        int sent = 0; while (sent < 15) { const uint8_t c = sent == 0 ? sym : hist[sent]; if (c == '.' || c == '!' || c == '?' || c == '\n') break; ++sent; }
        int8_t* w = xin + 80;
        for (int k = 0; k < 6; ++k) w[k] = pos > k ? S : (int8_t)-S;
        for (int k = 0; k < 5; ++k) w[6 + k] = prev > k + 1 ? S : (int8_t)-S;
        for (int k = 0; k < 5; ++k) w[11 + k] = sent > (1 << k) ? S : (int8_t)-S;
        std::memcpy(s.xin_exact, xin, Dims::IN);
        float m = 0.f; for (int j2 = 0; j2 < Dims::F; ++j2) m += std::abs((float)s.ws_mem[j2]); m /= Dims::F;
        s.ws_scale = s.ws_scale == 0.f ? std::max(m, 1.f) : s.ws_scale + 0.001f * (m - s.ws_scale);
        const float g = (float)S / std::max(s.ws_scale, 1.f);
        if (p.ws_mem_in) for (int j2 = 0; j2 < Dims::F; ++j2) { const float v = (float)s.ws_mem[j2] * g * p.ws_in_gain; xin[96 + j2] = (int8_t)std::max(-127.f, std::min(127.f, v)); }
        else std::memset(xin + 96, 0, Dims::F);   // ws_mem_in=0 (Organism::make_input)
        std::memset(s.xin_exact + 96, 0, Dims::F);
        for (int r = 0; r < 4; ++r) xin[124 + r] = (int8_t)std::max(-127.f, std::min(127.f, s.ws_reg[r] * p.reg_scale / 100.f));
        if (diag_zero_wsin) std::memset(xin + 96, 0, Dims::F);   // make_input: bodies cannot read the workspace memory (nor the registers)
    }
    const auto& fmb = org_.fm_base;
    if (!fmb.empty()) {   // fixed motifs (Organism::process_event): one body per motif, a fixed function of the recent bytes
        s.ncand = 0; s.cur_flags = 0; s.nbody = 0;
        for (size_t m = 0; m < fmb.size(); ++m) { const uint32_t h = p.fm_h[m]; uint32_t off;
            if (h <= 1) off = s.sym; else { uint64_t x = 1469598103934665603ULL; for (uint32_t i = 0; i < h && i < 16; ++i) x = (x ^ (i == 0 ? s.sym : s.hist[i])) * 1099511628211ULL;
                if (p.fm_salt) x ^= 0x9E3779B97F4A7C15ULL * (uint64_t)(m + 1);   // fm_salt (Organism::process_event)
                off = (uint32_t)(mix64(x) % p.fm_n[m]); }
            s.body_exact[s.nbody] = false; s.body[s.nbody++] = fmb[m] + off; }
        return;
    }
    const AddressTree& tree = org_.tree;
    uint32_t depth; int32_t leaf = tree.walk(s.key, depth);
    s.ncand = tree.match(leaf, s.key, s.cand, 16);
    if (p.episodic && p.exact_only && s.ncand > 1) { uint32_t m = 0; for (uint32_t i = 0; i < s.ncand; ++i) if (tree.locks[s.cand[i]].mask == ~0ULL) s.cand[m++] = s.cand[i]; if (m) s.ncand = m; }
    s.cur_flags = 0;
    for (uint32_t i = 0; i < s.ncand; ++i) if (tree.locks[s.cand[i]].mask == ~0ULL) { s.cur_flags |= Organism::FL_EXACT; break; }
    if (p.specific_wins && s.ncand > 1) {
        uint32_t m = 0; int best = -1; for (uint32_t i = 0; i < s.ncand; ++i) { const Lock& lk = tree.locks[s.cand[i]]; if (lk.mask == ~0ULL) continue; int pc = std::popcount(lk.mask); if (pc > best) best = pc; }
        for (uint32_t i = 0; i < s.ncand; ++i) { const Lock& lk = tree.locks[s.cand[i]]; if (lk.mask == ~0ULL || std::popcount(lk.mask) == best) s.cand[m++] = s.cand[i]; }
        s.ncand = m;
    }
    if (s.agent_on) {   // Organism::agency_observe / agency_apply_focus
        bool nov = false; for (uint32_t i = 0; i < s.ncand; ++i) { const Lock& lk = tree.locks[s.cand[i]]; if (lk_hits(s, s.cand[i]) < p.novel_hits || lk_err(s, s.cand[i]) > s.wrong_ema + p.surprise_thresh) nov = true; (void)lk; }
        float* o = s.agent_obs.data(); const uint64_t key = s.key;
        for (int j = 0; j < Dims::F; ++j) o[j] = std::tanh((float)s.ws_mem[j] / 256.f);
        for (int b = 0; b < 8; ++b) { o[32 + b] = ((key >> b) & 1) ? 1.f : -1.f; o[40 + b] = ((key >> (8 + b)) & 1) ? 1.f : -1.f; }
        o[48] = s.loss_ema / 3.f; o[49] = (float)s.wrong_ema / 65535.f; o[50] = nov ? 1.f : 0.f; o[51] = (s.cur_flags & Organism::FL_EXACT) ? 1.f : 0.f;
        o[52] = (float)s.ncand / 4.f; o[53] = s.agent_last_ptarget; o[54] = s.agent.learning_progress * 10.f; o[55] = s.agent.signals.sensory_error / 8.f;
        for (int i = 56; i < AG_OBS; ++i) o[i] = 0.f;
        s.agent.perceive(o, p.train);
        s.agent_tendency = s.agent.deliberate(true, true);
        s.agent_exact_only = s.agent.focus.stream == 1;
        const int32_t* gq = s.agent.focus.gain_q8.data();
        for (int i = 0; i < Dims::IN; ++i) { int32_t v = (int32_t)s.xin[i] * gq[i]; s.xin_focus[i] = (int8_t)((v >= 0 ? v + 128 : v - 128) >> 8); }
        if (p.episodic && s.agent_exact_only && s.ncand > 1) { uint32_t m = 0; for (uint32_t i = 0; i < s.ncand; ++i) if (tree.locks[s.cand[i]].mask == ~0ULL) s.cand[m++] = s.cand[i]; if (m) s.ncand = m; }
    }
}

// ---- S1: births without candidates, lock hits (process_event's address stage, remainder) ------------------------------------
void MultiStream::s1_admit(Stream& s) {
    if (!org_.fm_base.empty()) { s.novel = false; return; }   // fixed motifs: bodies already chosen in P1
    Organism& o = org_; const Params& p = p_;
    if (s.ncand == 0) {
        if (S_ > 1) { uint32_t depth; int32_t leaf = o.tree.walk(s.key, depth); s.ncand = o.tree.match(leaf, s.key, s.cand, 16); }   // an earlier stream may have just created it
        if (s.ncand == 0 && p.structural) {
            std::memcpy(o.hist, s.hist, sizeof o.hist);   // alloc_unit's K screen reads the current symbol
            uint32_t u = o.alloc_unit();
            if (u != UINT32_MAX) {
                o.birth_unit_weights(u); dirty_.push_back(u); reset_.push_back(u); ad_uses_[u] = 0; ad_t_[u] = 0; ad_last_[u] = 0;
                uint32_t id = o.tree.new_lock(s.key, SENSORY_MASK, u, o.block_idx); o.unit_lock[u] = id; s.cand[s.ncand++] = id; ++st_.births;
            }
        }
    }
    s.nbody = 0; s.novel = false;
    for (uint32_t i = 0; i < s.ncand; ++i) {
        Lock& lk = o.tree.locks[s.cand[i]];
        if (lk_hits(s, s.cand[i]) < p.novel_hits || lk_err(s, s.cand[i]) > s.wrong_ema + p.surprise_thresh) s.novel = true;
        if (eval_) { auto it = s.lov.try_emplace(s.cand[i], Stream::LockOv{lk.hits, lk.err_ema}).first; ++it->second.hits; }
        else { ++lk.hits; lk.last_hit_block = o.block_idx; }
        const uint32_t u = lk.unit; bool dup = false; for (uint32_t k = 0; k < s.nbody; ++k) dup |= s.body[k] == u;   // run_pass's epoch dedup
        if (dup) continue;
        s.body_exact[s.nbody] = p.episodic && o.unit_lock[u] != UINT32_MAX && o.tree.locks[o.unit_lock[u]].mask == ~0ULL;
        s.body[s.nbody++] = u;
    }
}

// motif_stream: body k's L3 message, the mean of the earlier bodies' L6 (Organism::run_body); all = every earlier body counted as
// having reached L6 (P2's guess, before the thresholds)
void MultiStream::stream_msg(const Stream& s, uint32_t k, int8_t* out, bool all) const {
    const float sc = 0.5f / (float)k;
    for (int i = 0; i < Dims::B; ++i) { int32_t sum = 0; for (uint32_t t = 0; t < k; ++t) if (all || (s.tr[t].reached & R_L6)) sum += s.tr[t].l6[i];
        out[i] = clamp8((int32_t)std::lround((float)sum * sc)); }
}

// cortical motifs 3 / 4 (Organism::run_body's runL, the same arithmetic): shunting (divisive) inhibition of a layer's drive by
// its mean |input| (ff_inh) and its mean |output| (fb_inh); kern.run when both are off. The backward treats the factor as a
// constant gain, as the organism's does, so the learners need nothing new.
static inline void run_layer(KernelCtx& kern, const Params& p, const uint8_t* W, const LayerDesc& d, int li, const int8_t* x, int8_t* y) {
    if ((p.ff_inh <= 0.f && p.fb_inh <= 0.f) || li == LI_4 || p.homeo) { kern.run(W, d, x, y); return; }
    int32_t z[64]; int64_t ix = 0; for (int i = 0; i < d.in; ++i) ix += x[i] < 0 ? -x[i] : x[i];
    for (int j = 0; j < d.out; ++j) { int32_t s = 0; const uint32_t off = d.nib_off + (uint32_t)(j * d.in); for (int i = 0; i < d.in; ++i) s += (int32_t)x[i] * nib_get(W, off + (uint32_t)i); z[j] = s; }
    float f = 1.f; if (p.ff_inh > 0.f) f /= 1.f + p.ff_inh * (float)ix / (float)d.in / 127.f;
    if (p.fb_inh > 0.f) { int64_t oy = 0; for (int j = 0; j < d.out; ++j) { const int32_t v = clamp8((int32_t)std::lround(z[j] * f) >> d.shift); oy += v < 0 ? -v : v; }
        f /= 1.f + p.fb_inh * (float)oy / (float)d.out / 127.f; }
    for (int j = 0; j < d.out; ++j) y[j] = clamp8((int32_t)std::lround(z[j] * f) >> d.shift);
}

// ---- P2: body forwards (run_body: every layer; the thresholds are applied in S2) -------------------------------------------------
void MultiStream::p2_forward(Stream& s) {
    thread_local KernelCtx kern; kern.kind = p_.kernel;
    const Organism& o = org_; static const int8_t zero_ret[Dims::R] = {0};
    const bool wire = p_.fm_wire && !o.fm_base.empty();   // motif_wire: body k (motif k) is routed from body k-1 (process_event)
    for (uint32_t k = 0; k < s.nbody; ++k) {
        const uint32_t u = s.body[k]; const bool exact = s.body_exact[k];
        const uint8_t* W = o.wbuf.p + (size_t)u * Dims::UNIT_BYTES;
        Trace& tr = s.tr[k]; const bool routed = wire && k > 0;
        tr.unit = u; tr.reached = 0; tr.consequential = 0; tr.recirced = 0; tr.emitted = 0;
        if (routed) { tr.src = (int16_t)(k - 1); tr.nsrc = 1; tr.srcs[0] = tr.src; tr.source = SRC_ROUTED; tr.hop = 1; }
        else { tr.src = -1; tr.nsrc = 0; tr.source = SRC_SENSORY; tr.hop = 0; }
        for (int li = 0; li < NUM_LAYERS; ++li) tr.sh[li] = o.L[li].shift;
        const int8_t* ret = exact ? zero_ret : &o.l4ret[(size_t)u * Dims::R];
        const int8_t* x1 = exact ? s.xin_exact : (s.agent_on ? s.xin_focus : s.xin);
        int8_t deep_ret[Dims::R];
        if (p_.deep_in && !exact) { std::memcpy(deep_ret, x1, Dims::R2); std::memcpy(deep_ret + Dims::R2, x1, Dims::R6); ret = deep_ret; }   // deep_in (run_body)
        run_layer(kern, p_, W, o.L[LI_1], LI_1, x1, tr.l1);
        s.lv[k][0] = level(tr.l1, Dims::A);
        std::memcpy(tr.x2, tr.l1, Dims::A); std::memcpy(tr.x2 + Dims::A, ret, Dims::R2);
        std::memcpy(tr.x3, tr.l1, Dims::A);
        if (routed && !diag_zero_msgs && p_.fm_stream) stream_msg(s, k, tr.x3 + Dims::A, true);   // motif_stream: the mean of the earlier L6 (S2 corrects)
        else if (routed && !diag_zero_msgs) std::memcpy(tr.x3 + Dims::A, s.tr[k - 1].l2, Dims::B);   // the sender's L2 (S2 zeroes it if the sender stops at L1)
        else std::memset(tr.x3 + Dims::A, 0, Dims::B);
        run_layer(kern, p_, W, o.L[LI_2], LI_2, tr.x2, tr.l2); run_layer(kern, p_, W, o.L[LI_3], LI_3, tr.x3, tr.l3);
        s.lv[k][1] = (level(tr.l2, Dims::B) * Dims::B + level(tr.l3, Dims::C) * Dims::C) / (Dims::B + Dims::C);
        std::memcpy(tr.x5, tr.l2, Dims::B); std::memcpy(tr.x5 + Dims::B, tr.l3, Dims::C);
        run_layer(kern, p_, W, o.L[LI_5], LI_5, tr.x5, tr.l5);
        s.lv[k][2] = level(tr.l5, Dims::E);
        std::memcpy(tr.x6, tr.l5, Dims::E); std::memcpy(tr.x6 + Dims::E, ret + Dims::R2, Dims::R6);
        run_layer(kern, p_, W, o.L[LI_6], LI_6, tr.x6, tr.l6);
        s.lv[k][3] = level(tr.l6, Dims::F);
    }
}

// ---- S2: death thresholds (threshold() updates the unit's activity averages), emission gating --------------------------------
void MultiStream::s2_thresholds(Stream& s) {
    Organism& o = org_; const Params& p = p_;
    for (uint32_t k = 0; k < s.nbody; ++k) {
        Trace& tr = s.tr[k]; const uint32_t u = tr.unit; const bool exact = s.body_exact[k];
        int8_t sm[Dims::B]; const bool restream = p.fm_stream && tr.source == SRC_ROUTED && !diag_zero_msgs && (stream_msg(s, k, sm, false), std::memcmp(sm, tr.x3 + Dims::A, Dims::B) != 0);
        if (restream || (!p.fm_stream && tr.source == SRC_ROUTED && !diag_zero_msgs && !(s.tr[tr.src].reached & R_L23))) {   // the sender stopped at L1 (motif_wire:
            // run_body sends no message) / an earlier body did not reach L6 (motif_stream: the mean has fewer terms)
            thread_local KernelCtx kern; kern.kind = p.kernel; static const int8_t zero_ret[Dims::R] = {0};
            const uint8_t* W = o.wbuf.p + (size_t)u * Dims::UNIT_BYTES; const int8_t* ret = exact ? zero_ret : &o.l4ret[(size_t)u * Dims::R];
            int8_t deep_ret[Dims::R];
            if (p.deep_in && !exact) { const int8_t* x1 = s.agent_on ? s.xin_focus : s.xin; std::memcpy(deep_ret, x1, Dims::R2); std::memcpy(deep_ret + Dims::R2, x1, Dims::R6); ret = deep_ret; }
            if (restream) std::memcpy(tr.x3 + Dims::A, sm, Dims::B); else std::memset(tr.x3 + Dims::A, 0, Dims::B); run_layer(kern, p, W, o.L[LI_3], LI_3, tr.x3, tr.l3);
            s.lv[k][1] = (level(tr.l2, Dims::B) * Dims::B + level(tr.l3, Dims::C) * Dims::C) / (Dims::B + Dims::C);
            std::memcpy(tr.x5 + Dims::B, tr.l3, Dims::C); run_layer(kern, p, W, o.L[LI_5], LI_5, tr.x5, tr.l5); s.lv[k][2] = level(tr.l5, Dims::E);
            std::memcpy(tr.x6, tr.l5, Dims::E); std::memcpy(tr.x6 + Dims::E, ret + Dims::R2, Dims::R6); run_layer(kern, p, W, o.L[LI_6], LI_6, tr.x6, tr.l6); s.lv[k][3] = level(tr.l6, Dims::F);
        }
        tr.reached |= R_L1; if (s.lv[k][0] < threshold(s, p.death_l1, u, s.novel, 0, s.lv[k][0])) continue;
        tr.reached |= R_L23; if (s.lv[k][1] < threshold(s, p.death_l23, u, s.novel, 1, s.lv[k][1])) continue;
        tr.reached |= R_L5; if (s.lv[k][2] < threshold(s, p.death_l5, u, s.novel, 2, s.lv[k][2])) continue;
        tr.reached |= R_L6; if (s.lv[k][3] < threshold(s, p.death_l6, u, s.novel, 3, s.lv[k][3])) continue;
        s.hooked[k] = false;
        if (l6_hook) { std::memcpy(s.l6e[k], tr.l6, Dims::F); l6_hook(u, s.l6e[k]); s.hooked[k] = true; }   // run_body: the emission is hooked, the trace keeps the body's own L6
        const bool gated_out = (p.route_gate && !p.route_ungated && route_util(s, u) < p.route_gate_bits) && (tr.source == SRC_ROUTED || (exact && !(s.agent_on && s.agent_exact_only) && !p.exact_only));
        tr.consequential = gated_out ? 2 : 0;
        tr.emitted = !gated_out;
    }
}

// ---- P3: readout, sequence blend, loss, route counterfactuals, agency consequence, learn_l6's readout gradients ----------------
void MultiStream::p3_readout(Stream& s, Partial& part) {
    const Params& p = p_; const Organism& o = org_; const uint32_t V = p.vocab; const uint8_t target = s.target;
    std::memset(s.ws, 0, sizeof s.ws);
    const bool split = !o.wsp.empty(); if (split) std::memset(s.wsm, 0, sizeof s.wsm);
    for (uint32_t k = 0; k < s.nbody; ++k) { const Trace& tr = s.tr[k]; if (!tr.emitted) continue; const int32_t gain = s.body_exact[k] ? p.exact_gain : 1; const int8_t* l6 = s.hooked[k] ? s.l6e[k] : tr.l6; if (!p.ws_emit_max || k < p.ws_emit_max) for (int j = 0; j < Dims::F; ++j) s.ws[j] += gain * l6[j];
        if (split) for (int j = 0; j < Dims::F; ++j) s.wsm[k][j] += gain * l6[j]; }   // ws_split: body k is motif k
    float* probs = s.probs.data();
    if (!s.ts.q.empty()) { o.tiles_read(s.ts); o.tiles_attend(s.ts, s.ws, s.hist); }   // Organism: tiles_read at the event's start (tiles do not change
    readout(s, s.ws, s.logits.data(), probs);                                    // within it), tiles_attend after the bodies, before the readout
    float gf[SEQ_GATE_F] = {0}; bool sg_used = false; float sg_g = 0.f;
    if (s.seq_on) {   // process_event's composition with the sequence memory
        SequenceMemory& seq = s.seq;
        s.body_probs = s.probs; ++s.seq_events;
        seq.topic_active = false;
        int L = seq.predict(s.seq_out.data(), p.seq_kappa, p.seq_use_gain);
        if (seq.chain_forcing) seq.bubble_left = 0;
        bool bubbling = false; s.body_resp = 1.f;
        const float* body_probs = s.body_probs.data(); const float* seq_out = s.seq_out.data();
        if (seq.bubble_left > 0) {
            uint8_t nx = seq.at(seq.committed_pos); --seq.bubble_left; ++seq.committed_pos;
            if (body_probs[nx] >= p.seq_bubble_support) { bubbling = true; ++s.seq_bubble_events; ++seq.bubble_bytes;
                for (int v = 0; v < 256; ++v) probs[v] = 0.02f * body_probs[v]; probs[nx] += 0.98f; }
            else seq.bubble_left = 0;
        }
        if (!bubbling && L) {
            ++s.seq_matched; s.seq_conf_sum += seq.confidence;
            float tot = 0.f; for (int v = 0; v < 256; ++v) tot += seq_out[v];
            float a = seq.chain_forcing ? 1.f : p.seq_alpha * seq.confidence;
            const bool gated = p.seq_gate && !seq.chain_forcing;
            if (gated) {
                int sv = 0; for (int v = 1; v < 256; ++v) if (seq_out[v] > seq_out[sv]) sv = v;
                float bt = 0.f; for (int v = 0; v < 256; ++v) bt = std::max(bt, body_probs[v]);
                gf[0] = 1.f; gf[1] = seq.confidence; gf[2] = std::min(L, 32) / 32.f; gf[3] = seq_out[sv] / std::max(tot, 1e-9f); gf[4] = bt; gf[5] = body_probs[sv];
                float z0 = 0.f; for (int k = 0; k < SEQ_GATE_F; ++k) z0 += o.seq_gw[k] * gf[k];
                a = 1.f / (1.f + std::exp(-z0));
            }
            float z = 0.f;
#if defined(__AVX2__) && defined(__FMA__)
            {
                const __m256 eps = _mm256_set1_ps(0.01f * tot / 256.f), inv = _mm256_set1_ps(1.f / (tot * 1.01f)), floor9 = _mm256_set1_ps(1e-9f);
                const __m256 wa = _mm256_set1_ps(a), wb = _mm256_set1_ps(1.f - a); __m256 zs = _mm256_setzero_ps();
                for (int v = 0; v < 256; v += 8) {
                    __m256 lb = log8(_mm256_max_ps(_mm256_loadu_ps(&body_probs[v]), floor9));
                    __m256 ls = log8(_mm256_mul_ps(_mm256_add_ps(_mm256_loadu_ps(&seq_out[v]), eps), inv));
                    __m256 e = exp8(_mm256_fmadd_ps(wb, lb, _mm256_mul_ps(wa, ls)));
                    _mm256_storeu_ps(&probs[v], e); zs = _mm256_add_ps(zs, e);
                }
                alignas(32) float zt[8]; _mm256_store_ps(zt, zs); for (int i = 0; i < 8; ++i) z += zt[i];
            }
#else
            for (int v = 0; v < 256; ++v) { float ps = (seq_out[v] + 0.01f * tot / 256.f) / (tot * (1.f + 0.01f)); probs[v] = std::pow(std::max(body_probs[v], 1e-9f), 1.f - a) * std::pow(ps, a); z += probs[v]; }
#endif
            { const float iz = 1.f / z; for (int v = 0; v < 256; ++v) probs[v] *= iz; }
            if (p.seq_linear) {
                const float eps = 0.01f * tot / 256.f, inv = 1.f / (tot * 1.01f);
                for (int v = 0; v < 256; ++v) probs[v] = a * (seq_out[v] + eps) * inv + (1.f - a) * body_probs[v];
                const float pst = (seq_out[target] + eps) * inv, pt = std::max(probs[target], 1e-9f);
                s.body_resp = (1.f - a) * body_probs[target] / pt;
                if (gated && p.train) { sg_g = -(pst - body_probs[target]) / pt * a * (1.f - a); sg_used = true; }
            } else if (gated && p.train) {
                const float eps = 0.01f * tot / 256.f, inv = 1.f / (tot * 1.01f); float E = 0.f, dt = 0.f;
                for (int v = 0; v < 256; ++v) { const float d = std::log((seq_out[v] + eps) * inv) - std::log(std::max(body_probs[v], 1e-9f)); E += probs[v] * d; if (v == target) dt = d; }
                sg_g = -(dt - E) * a * (1.f - a); sg_used = true;
            }
            if (L >= p.seq_bubble_min) { seq.bubble_left = p.seq_bubble_len; ++seq.bubbles; }
        }
    }
    if (sg_used) { for (int k = 0; k < SEQ_GATE_F; ++k) part.sg[k] += sg_g * gf[k]; part.sg_any = true; }
    uint32_t pred = 0; for (uint32_t v = 1; v < V; ++v) if (probs[v] > probs[pred]) pred = v;
    s.pred = pred; s.loss = -std::log(std::max(probs[target], 1e-9f)); s.correct = pred == target;
    if (s.ans_state == 2 || (s.ans_state == 1 && s.goal_w > 1.f)) { s.ans_loss += s.loss; ++s.ans_n; }
    if (!s.loss_ema_init) { s.loss_ema = s.loss; s.loss_ema_init = true; } else s.loss_ema += 0.01f * (s.loss - s.loss_ema);
    s.consequential = s.loss <= s.loss_ema;
    for (uint32_t t = 0; t < s.nbody; ++t) { Trace& tr = s.tr[t]; s.route_has[t] = 0;   // counterfactual credit of episodic bodies (applied in S4)
        const bool is_exact = o.unit_lock[tr.unit] != UINT32_MAX && o.tree.locks[o.unit_lock[tr.unit]].mask == ~0ULL;
        if ((tr.source != SRC_ROUTED && !is_exact) || !(tr.reached & R_L6)) { tr.consequential = s.consequential; continue; }
        const bool silent = tr.consequential == 2;
        int32_t wsx[Dims::F]; for (int j = 0; j < Dims::F; ++j) wsx[j] = silent ? s.ws[j] + tr.l6[j] : s.ws[j] - tr.l6[j];
        readout(s, wsx, s.cf_logits.data(), s.cf_probs.data());
        const float loss_other = -std::log(std::max(s.cf_probs[target], 1e-9f));
        const float gain_bits = (silent ? (s.loss - loss_other) : (loss_other - s.loss)) / 0.6931471805599453f;
        s.route_gain[t] = gain_bits; s.route_has[t] = 1;
        tr.consequential = gain_bits > 0.f ? 1 : 0; }
    if (s.agent_on && p.train) {   // Organism::agency_consequence
        const float p_target = probs[target];
        const float goal0 = p.agency_goal == 1 ? std::max(0.f, 1.f - s.loss / std::log((float)p.vocab)) : p_target;
        s.agent_last_ptarget = p_target;
        float obs[4] = {goal0, 0.f, 1.f, s.correct ? 1.f : 0.f};
        s.agent.consequence(obs, true);
    }
    {   // process_event's stream-level evidence (the lock updates stay serial, in S4)
        s.wrong = s.correct ? 0u : 65535u;
        s.wrong_ema += ((int32_t)s.wrong - (int32_t)s.wrong_ema) / 256;
        s.gain_bits = std::log2(std::max(probs[target], 1e-9f) / std::max(s.prior[target], 1e-9f));
        if (p.train) for (uint32_t v = 0; v < V; ++v) s.prior[v] += 0.0005f * ((v == target ? 1.f : 0.f) - s.prior[v]);
    }
    s.jobs.clear(); s.tjobs.clear();
    if (!p.train) return;
    // Organism::learn_l6: the bodies' own error (body_err), its readout gradients, the register readout's
    const std::vector<float>& bprobs = (p.body_err && s.seq_on && s.body_probs.size() == s.probs.size()) ? s.body_probs : s.probs;
    const float resp = (p.body_err == 2 && p.seq_linear && s.seq_on) ? s.body_resp : 1.f;
    s.err_idx.clear(); s.err_val.clear();
    for (uint32_t v = 0; v < V; ++v) { float e = (bprobs[v] - (v == target ? 1.f : 0.f)) * resp; if (e > p.err_sparse || e < -p.err_sparse) { s.err_idx.push_back(v); s.err_val.push_back(e); } }
    for (int k = 0; k < Dims::F; ++k) s.dl6[k] = 0.f;
    for (size_t q = 0; q < s.err_idx.size(); ++q) { const float* row = &o.wout[(size_t)s.err_idx[q] * Dims::F]; for (int k = 0; k < Dims::F; ++k) s.dl6[k] += s.err_val[q] * row[k]; }
    if (!o.wsp.empty()) { const size_t MF = p.fm_h.size() * Dims::F; float* dm = &s.dl6m[0][0]; std::fill(dm, dm + MF, 0.f);   // ws_split (Organism::learn_l6)
        for (size_t q = 0; q < s.err_idx.size(); ++q) { const float* row = &o.wsp[(size_t)s.err_idx[q] * MF]; float* gr = &part.wsp[(size_t)s.err_idx[q] * MF]; const float g = s.err_val[q] * std::min(s.goal_w, 4.f);
            for (size_t i = 0; i < MF; ++i) { dm[i] += s.err_val[q] * row[i]; gr[i] += g * (float)s.wsm[i / Dims::F][i % Dims::F] * (1.0f / 128.0f); } } }
    if (!o.wtl.empty()) { std::fill(s.dl6t, s.dl6t + Dims::F, 0.f);   // tile_split (Organism::learn_l6)
        for (size_t q = 0; q < s.err_idx.size(); ++q) { const float* row = &o.wtl[(size_t)s.err_idx[q] * Dims::F]; float* gr = &part.wtl[(size_t)s.err_idx[q] * Dims::F]; const float g = s.err_val[q] * std::min(s.goal_w, 4.f);
            for (int k = 0; k < Dims::F; ++k) { s.dl6t[k] += s.err_val[q] * row[k]; gr[k] += g * s.ts.a[k]; } } }
    s.inj.clear();
    if (!s.ts.q.empty()) {   // Organism::process_event: tiles_learn after learn_l6 (dl6), before learn_backprop; at one stream straight into the
        TileLearn tl;        // organism's gradients (Organism's summation order), else into this worker's partials
        tl.gM = S_ == 1 ? org_.tile_Mg.data() : part.tm.data(); tl.gA = org_.tile_Ag.empty() ? nullptr : S_ == 1 ? org_.tile_Ag.data() : part.ta.data();
        tl.gF = org_.fw_Wg.empty() ? nullptr : S_ == 1 ? org_.fw_Wg.data() : part.tf.data(); tl.goal_w = s.goal_w; tl.event = s.n_events;
        tl.gR = org_.ret_Wg.empty() ? nullptr : S_ == 1 ? org_.ret_Wg.data() : part.tr.data();
        const uint32_t Wn = p.tc_window, fill = s.tc_fill; const uint32_t sid = s.id; std::vector<LInject>& inj = s.inj;
        tl.inject = [Wn, fill, sid, &inj](uint64_t age, const float* g, float w, const float* v) { if (!Wn || age < 1 || age > fill || age >= Wn) return;
            LInject x; x.stream = sid; x.age = (uint32_t)age; for (int j = 0; j < Dims::F; ++j) x.g[j] = (v[j] > -4.f && v[j] < 4.f) ? g[j] * w : 0.f; inj.push_back(x); };
        if (!o.wtl.empty()) { float dd[Dims::F]; for (int k = 0; k < Dims::F; ++k) dd[k] = s.dl6[k] + s.dl6t[k]; o.tiles_learn(s.ts, dd, tl); }   // tile_split
        else o.tiles_learn(s.ts, s.dl6, tl);
        for (int k = 0; k < Dims::F; ++k) s.dl6[k] += s.ts.dq[k]; }
    for (int r = 0; r < 4; ++r) s.greg_ro[r] = 0.f;
    const float gmin = std::min(s.goal_w, 4.f);
    for (size_t q = 0; q < s.err_idx.size(); ++q) { const float* row = &o.wreg[(size_t)s.err_idx[q] * 4];
        for (int r = 0; r < 4; ++r) { s.greg_ro[r] += s.err_val[q] * row[r] * 0.01f; part.rg[(size_t)s.err_idx[q] * 4 + r] += s.err_val[q] * gmin * s.ws_reg[r] * 0.01f; } }
    if (!p.readout_freeze || s.n_events_global < p.readout_freeze) {
        float a[Dims::F]; for (int k = 0; k < Dims::F; ++k) a[k] = (float)s.ws[k] * (1.0f / 128.0f);
        for (size_t q = 0; q < s.err_idx.size(); ++q) { float* gr = &part.ro[(size_t)s.err_idx[q] * Dims::F]; const float g = s.err_val[q] * gmin; for (int k = 0; k < Dims::F; ++k) gr[k] += g * a[k]; }
    }
    // learn_backprop's direct jobs, visited last to first: the emitting bodies, and with bp_route every body past L2 (its
    // receivers' message gradient reaches its L2; learn_backprop counts the body as used even when that gradient is zero)
    int32_t job_of[MS_MAXB];
    for (uint32_t k = s.nbody; k-- > 0;) { const Trace& tr = s.tr[k]; job_of[k] = -1;
        const bool out = (tr.reached & R_L6) && tr.emitted, msg = p.bp_route && (tr.reached & R_L23);
        if (!out && !msg) continue;
        LTrace j; j.unit = tr.unit; j.stream = (uint16_t)s.id; j.exact = s.body_exact[k]; j.slot = 0; j.scale = s.goal_w; j.gain = out ? (s.body_exact[k] ? (float)p.exact_gain : 1.f) : 0.f;
        std::memcpy(j.x2, tr.x2, sizeof j.x2); std::memcpy(j.x3, tr.x3, sizeof j.x3); std::memcpy(j.x5, tr.x5, sizeof j.x5); std::memcpy(j.x6, tr.x6, sizeof j.x6);
        std::memcpy(j.l1, tr.l1, sizeof j.l1); std::memcpy(j.l2, tr.l2, sizeof j.l2); std::memcpy(j.l3, tr.l3, sizeof j.l3); std::memcpy(j.l5, tr.l5, sizeof j.l5); std::memcpy(j.l6, tr.l6, sizeof j.l6);
        std::memcpy(j.sh, tr.sh, sizeof j.sh); j.flags = out ? LF_OUT : 0; j.sender = 0;
        if (out && p.ws_emit_max && k >= p.ws_emit_max) j.flags |= LF_NOWS;   // ws_emit_max: its channel's error only, no temporal credit
        if (!o.wsp.empty()) j.slot = (uint8_t)k;   // ws_split: its motif
        if (p.fm_stream) { if (tr.reached & R_L6) j.flags |= LF_L6; if (tr.src >= 0 && !diag_zero_msgs) { j.flags |= LF_STRM; j.sender = (uint8_t)k; } }   // position k in the chain
        job_of[k] = (int32_t)s.jobs.size(); s.jobs.push_back(j); }
    if (p.bp_route && !p.fm_stream && !diag_zero_msgs)   // a receiver passes W3[:, A..] g3 to its sender (visited after it: senders precede receivers)
        for (uint32_t k = 0; k < s.nbody; ++k) { const int16_t sd = s.tr[k].src; if (job_of[k] >= 0 && sd >= 0 && (uint32_t)sd < k && job_of[sd] >= 0) s.jobs[job_of[k]].sender = (uint8_t)(job_of[sd] - job_of[k]); }
    LStream& ls = lstreams_[s.id];   // this stream's learner inputs
    std::memcpy(ls.dl6, s.dl6, sizeof ls.dl6); std::memcpy(ls.greg_ro, s.greg_ro, sizeof ls.greg_ro); if (!o.wsp.empty()) std::memcpy(ls.dl6m, s.dl6m, sizeof ls.dl6m);
    ls.goal_w = s.goal_w; ls.gscale = p_.ws_mem_in ? (float)in_scale_ / std::max(s.ws_scale, 1.f) * p_.ws_in_gain : 0.f;   // ws_mem_in=0: nobody reads the workspace
    std::memcpy(ls.reg_now, s.ws_reg, sizeof ls.reg_now);
    std::memcpy(ls.h_reg_prev, s.h_reg_prev, sizeof ls.h_reg_prev); std::memcpy(ls.h_G, s.h_G, sizeof ls.h_G); std::memcpy(ls.h_g, s.h_g, sizeof ls.h_g);
    std::memcpy(ls.h_v, s.h_v, sizeof ls.h_v); std::memcpy(ls.h_u, s.h_u, sizeof ls.h_u); std::memcpy(ls.h_a, s.h_a, sizeof ls.h_a);
    ls.h_valid = s.h_valid; ls.h_slot = s.h_slot; s.h_valid = false;
    ls.head = s.tc_head; ls.fill = s.tc_fill;
    std::memcpy(ls.x1, s.agent_on ? s.xin_focus : s.xin, Dims::IN); std::memcpy(ls.x1e, s.xin_exact, Dims::IN);
    if (s.tc_fill == p.tc_window) {   // the oldest event leaves the window: its writers' delayed pass (alive is read-only in P3)
        const uint16_t slot = s.tc_head; const float ascale = s.ring_goal_w[slot] * p.tc_gain;
        for (const LTrace& w : s.ring_tr[slot]) { if (!o.alive[w.unit]) continue; LTrace j = w; j.slot = (uint8_t)slot; j.scale = ascale; s.tjobs.push_back(j); }
    }
}

// ---- S3: readout Adam, body learning through the Learner -------------------------------------------------------------------
void MultiStream::s3_learn() {
    Organism& o = org_; const Params& p = p_; const uint32_t V = p.vocab;
    auto tq = Clock::now();
    uint32_t active = 0; for (auto& sp : st_s_) active += sp->active;
    const bool ro_due = ro_events_ + active >= ro_next_;
    // per-worker readout gradients into the shared buffers (rows split over the workers). With several streams they are
    // summed only when the readout steps (at one stream every event, which keeps Organism's summation order).
    const uint32_t nw = (uint32_t)part_.size(), rows_per = (V + nw - 1) / nw;
    if (S_ == 1 || ro_due) pool_.run(nw, [&](uint32_t w, unsigned) {
        const uint32_t v0 = w * rows_per, v1 = std::min(V, v0 + rows_per);
        for (Partial& pt : part_) {
            for (size_t i = (size_t)v0 * Dims::F; i < (size_t)v1 * Dims::F; ++i) { o.ro_g[i] += pt.ro[i]; pt.ro[i] = 0.f; }
            if (!pt.wsp.empty()) { const size_t MF = pt.wsp.size() / V; for (size_t i = (size_t)v0 * MF; i < (size_t)v1 * MF; ++i) { o.wsp_g[i] += pt.wsp[i]; pt.wsp[i] = 0.f; } }   // ws_split
            if (!pt.wtl.empty()) for (size_t i = (size_t)v0 * Dims::F; i < (size_t)v1 * Dims::F; ++i) { o.wtl_g[i] += pt.wtl[i]; pt.wtl[i] = 0.f; }   // tile_split
            for (size_t i = (size_t)v0 * 4; i < (size_t)v1 * 4; ++i) { o.rg_g[i] += pt.rg[i]; pt.rg[i] = 0.f; }
        }
    });
    if (S_ > 1 && ro_due) for (Partial& pt : part_) {   // tiles' gradients (at one stream they went straight into the organism's)
        for (size_t i = 0; i < pt.tm.size(); ++i) { o.tile_Mg[i] += pt.tm[i]; pt.tm[i] = 0.f; }
        for (size_t i = 0; i < pt.ta.size(); ++i) { o.tile_Ag[i] += pt.ta[i]; pt.ta[i] = 0.f; }
        for (size_t i = 0; i < pt.tf.size(); ++i) { o.fw_Wg[i] += pt.tf[i]; pt.tf[i] = 0.f; }
        for (size_t i = 0; i < pt.tr.size(); ++i) { o.ret_Wg[i] += pt.tr[i]; pt.tr[i] = 0.f; } }
    if (S_ == 1 || ro_due) for (Partial& pt : part_) { if (pt.sg_any) { if (o.sg_g.empty()) { o.sg_g.assign(SEQ_GATE_F, 0.f); o.sg_m = o.sg_g; o.sg_v = o.sg_g; } for (int k = 0; k < SEQ_GATE_F; ++k) o.sg_g[k] += pt.sg[k]; }
        std::fill(pt.sg, pt.sg + 6, 0.f); pt.sg_any = false; }
    st_.s3_reduce += since(tq); tq = Clock::now();
    learner_->wait();   // the previous step's learning (overlap): its weights and head gradients are needed from here on
    ro_events_ += active;
    if (ro_due) {   // one minibatch Adam step of the readout, the register readout, the head and the gate
        learner_->take_head_grads(o.hd_g.data());
        o.ro_adam_step();
        learner_->set_head(&o.hw_g[0][0], &o.hw_v[0][0], o.hb_g);
        const uint64_t k = std::max(1u, p.ro_adam_k); while (ro_next_ <= ro_events_) ro_next_ += k;
    }
    st_.s3_ro += since(tq); tq = Clock::now();
    // jobs
    direct_.clear(); temporal_.clear(); due_.clear(); touched_units_.clear(); inj_.clear();
    for (uint32_t si = 0; si < S_; ++si) {   // the learner inputs were prepared per stream in P3
        Stream& s = *st_s_[si]; lstreams_[si].active = s.active;
        if (!s.active) continue;
        direct_.insert(direct_.end(), s.jobs.begin(), s.jobs.end()); temporal_.insert(temporal_.end(), s.tjobs.begin(), s.tjobs.end());
        inj_.insert(inj_.end(), s.inj.begin(), s.inj.end());
        for (const LTrace& j : s.jobs) if (step_uses_[j.unit]++ == 0) touched_units_.push_back(j.unit);
    }
    // Adam: uses counted per body; a body that crossed a step boundary this step takes one step over all its uses since the last
    const uint32_t k = std::max(1u, p.adam_k);
    for (uint32_t u : touched_units_) {
        const uint32_t c = step_uses_[u]; step_uses_[u] = 0;
        const uint32_t n0 = ad_uses_[u], n = n0 + c; ad_uses_[u] = n; st_.adam_uses += c;
        bool due = false;
        if (p.adam_warm) { const uint32_t last = ad_last_[u]; for (uint32_t x = n0 + 1; x <= n && !due; ++x) due = x - last >= std::min(k, std::max(1u, last)); }
        else due = n / k > n0 / k;
        if (due) { due_.push_back(AdamDue{u, ++ad_t_[u], (float)(n - ad_last_[u])}); ad_last_[u] = n; ++st_.adam_steps; }
    }
    StepIn in; in.streams = lstreams_.data(); in.direct = direct_.data(); in.ndirect = (uint32_t)direct_.size();
    in.due = due_.data(); in.ndue = (uint32_t)due_.size(); in.temporal = temporal_.data(); in.ntemporal = (uint32_t)temporal_.size();
    in.dirty = dirty_.data(); in.ndirty = (uint32_t)dirty_.size(); in.reset = reset_.data(); in.nreset = (uint32_t)reset_.size();
    in.inj = inj_.data(); in.ninj = (uint32_t)inj_.size();
    st_.s3_build += since(tq);
    const auto t0 = Clock::now();
    if (overlap_) learner_->submit(in); else learner_->step(in);
    st_.sec_learn += since(t0);
    st_.direct_jobs += direct_.size(); st_.temporal_jobs += temporal_.size();
    dirty_.clear(); reset_.clear();
}

// ---- S4: metabolic, assoc, episodic births, lock evidence and utility, route utilities ------------------------------------------
void MultiStream::s4_structure(Stream& s) {
    Organism& o = org_; const Params& p = p_;
    for (uint32_t t = 0; t < s.nbody; ++t) if (s.route_has[t]) { float& ru = route_util(s, s.tr[t].unit); ru = p.route_utility_decay * ru + (1.f - p.route_utility_decay) * s.route_gain[t]; }
    if (p.train) s4_learned(s);
    const uint32_t wrong = s.wrong; const float gain_bits = s.gain_bits;   // process_event's lock-local evidence (stream part in P3)
    for (uint32_t i = 0; i < s.ncand; ++i) {
        Lock& lk = o.tree.locks[s.cand[i]];
        if (eval_) { auto it = s.lov.try_emplace(s.cand[i], Stream::LockOv{lk.hits, lk.err_ema}).first; it->second.err_ema += ((int32_t)wrong - (int32_t)it->second.err_ema) / 32; continue; }   // utility: not read in evaluation
        lk.err_ema += ((int32_t)wrong - (int32_t)lk.err_ema) / 32;
        lk.utility = p.utility_decay * lk.utility + (1.f - p.utility_decay) * gain_bits * s.goal_w;
        if (p.structural && p.train) {
            if (lk.evidence < 0 && lk.err_ema > p.conflict_thresh) lk.evidence = o.evid.acquire(s.cand[i], lk.mask);
            if (lk.evidence >= 0) o.evid.record(lk.evidence, s.key, s.target);
        }
    }
}
// process_event's learning-time structure: metabolic pressure, assoc locks, episodic births.
void MultiStream::s4_learned(Stream& s) {
    Organism& o = org_; const Params& p = p_; const uint32_t nk = o.n_kclass();
    for (uint32_t t = 0; t < s.nbody; ++t) {   // Organism::metabolic
        const Trace& tr = s.tr[t]; int32_t r = o.resist[tr.unit];
        const bool cons = tr.consequential != 0;
        if (tr.reached & R_L6) { uint32_t kc = o.kclass_of(tr.unit); auto& ks = o.kstats[kc]; ++ks.emits; ks.consequential += cons;
            float& e = o.kscreen_tab[(size_t)s.hist[0] * nk + kc]; e += p.kscreen_rate * ((cons ? 1.f : 0.f) - e); }
        if (tr.reached & R_L6) r += cons ? -(int32_t)std::lround(p.resist_down * s.goal_w) : p.resist_up;
        else r += s.novel ? -p.resist_down : p.resist_up_dead;
        o.resist[tr.unit] = (uint8_t)std::max(p.resist_min, std::min(p.resist_max, r));
    }
    if (!p.route_ungated && s.novel) for (uint32_t t = 0; t < s.nbody; ++t) {   // Organism::assoc_update (no routed messages)
        const uint32_t u = s.body[t];
        if (++o.assoc_starve[u] >= p.assoc_loosen_after && o.assoc_mask[u]) { o.assoc_starve[u] = 0; int b = o.rng.below(64); uint64_t m = o.assoc_mask[u]; while (!((m >> b) & 1)) b = (b + 1) & 63; o.assoc_mask[u] &= ~(1ULL << b); }
    }
    if (p.episodic && p.structural && !s.correct && s.loss > s.loss_ema + (s.goal_w > 1.f ? 0.f : p.episodic_margin) && s.ncand && !(s.cur_flags & Organism::FL_EXACT)) {
        if (o.recur_ready(s.key)) episodic_birth(s, s.cand[0]); else ++o.ep_totals.deferred;
    }
}

// Organism::episodic_birth with the stream's input, error, target and registers.
void MultiStream::episodic_birth(Stream& s, uint32_t parent_lock) {
    Organism& o = org_; const Params& p = p_;
    std::memcpy(o.hist, s.hist, sizeof o.hist);
    uint32_t u = o.alloc_unit(); if (u == UINT32_MAX) return;
    uint32_t pu = o.tree.locks[parent_lock].unit;
    std::memcpy(o.wbuf.p + (size_t)u * Dims::UNIT_BYTES, o.wbuf.p + (size_t)pu * Dims::UNIT_BYTES, Dims::UNIT_BYTES);
    std::memcpy(&o.act_ema[(size_t)u * 4], &o.act_ema[(size_t)pu * 4], 4);
    uint32_t id = o.tree.new_lock(s.key, ~0ULL, u, o.block_idx); o.unit_lock[u] = id; o.tree.locks[id].hits = p.novel_hits;
    ++o.st.episodic_births; ++o.ep_totals.births; s.cur_flags |= Organism::FL_BIRTH; ++st_.episodic_births;
    dirty_.push_back(u); births_.emplace_back(&s, u);   // the imprint writes only this unit: it runs after S4, in parallel
}
// episodic_birth's one-shot imprint of the newborn's L6 (reads the stream's input, error, target and registers, which
// nothing changes before P4, and the shared readout)
bool MultiStream::imprint(Stream& s, uint32_t u) {
    const Organism& o = org_; const Params& p = p_;
    thread_local KernelCtx kern; kern.kind = p.kernel;
    uint8_t* w = o.wbuf.p + (size_t)u * Dims::UNIT_BYTES;
    int8_t l1[Dims::A], l2[Dims::B], l3[Dims::C], l5[Dims::E], l6[Dims::F], x2[Dims::IN2], x3[Dims::IN3], x5[Dims::IN5], x6[Dims::IN6];
    kern.run(w, o.L[LI_1], s.xin_exact, l1);
    std::memcpy(x2, l1, Dims::A); std::memset(x2 + Dims::A, 0, Dims::R2); std::memcpy(x3, l1, Dims::A); std::memset(x3 + Dims::A, 0, Dims::B);
    kern.run(w, o.L[LI_2], x2, l2); kern.run(w, o.L[LI_3], x3, l3);
    std::memcpy(x5, l2, Dims::B); std::memcpy(x5 + Dims::B, l3, Dims::C); kern.run(w, o.L[LI_5], x5, l5);
    std::memcpy(x6, l5, Dims::E); std::memset(x6 + Dims::E, 0, Dims::R6);
    float g[Dims::F]; for (int j = 0; j < Dims::F; ++j) g[j] = -s.dl6[j];
    std::vector<float> lg(p.vocab), pr(p.vocab); int32_t wtmp[Dims::F]; const uint32_t V = p.vocab, t = s.target;
    for (int it = 0; it <= p.imprint_iters; ++it) {
        float mx = 0.f; for (int j = 0; j < Dims::F; ++j) mx = std::max(mx, std::fabs(g[j]));
        for (int j = 0; j < Dims::F; ++j) {
            int gj = mx > 0.f ? (int)std::lround(g[j] / mx * 7.f) : 0;
            for (int i = 0; i < Dims::IN6; ++i) { int si = x6[i] > 0 ? 1 : x6[i] < 0 ? -1 : 0; nib_set(w, Dims::NIB_L6 + j * Dims::IN6 + i, (si && gj) ? gj * si : 0); }
        }
        kern.run(w, o.L[LI_6], x6, l6); for (int j = 0; j < Dims::F; ++j) wtmp[j] = l6[j];
        readout(s, wtmp, lg.data(), pr.data());
        uint32_t am = 0; for (uint32_t v = 1; v < V; ++v) if (pr[v] > pr[am]) am = v;
        if (am == t) { s.cur_flags |= Organism::FL_IMPRINT_OK; return true; }
        const float* rt = &o.wout[(size_t)t * Dims::F]; const float* ra = &o.wout[(size_t)am * Dims::F]; for (int j = 0; j < Dims::F; ++j) g[j] += (rt[j] - ra[j]) * 4.f;
    }
    return false;
}

// ---- P4: workspace memory, write head, sequence capture, history ----------------------------------------------------------------
void MultiStream::p4_advance(Stream& s) {
    const Params& p = p_; const Organism& o = org_;
    if (p.train) {   // temporal_credit's store of this event's writers and the ring advance
        const uint16_t slot = s.tc_head; auto& ring = s.ring_tr[slot]; ring.clear();
        for (size_t q = s.jobs.size(); q-- > 0;) if ((s.jobs[q].flags & LF_OUT) && !(s.jobs[q].flags & LF_NOWS)) { ring.push_back(s.jobs[q]); ring.back().sender = 0; }   // the emitting bodies in trace order (jobs are last to first)
        s.ring_goal_w[slot] = s.goal_w;
        if (s.tc_fill < p.tc_window) ++s.tc_fill;
        s.tc_head = (uint16_t)((s.tc_head + 1) % p.tc_window);
    }
    for (int j = 0; j < Dims::F; ++j) s.ws_mem[j] = ((s.ws_mem[j] * p.ws_decay) >> 8) + s.ws[j];
    {   // ws_regs == 2: the shared write head
        float a[Dims::F]; for (int j = 0; j < Dims::F; ++j) a[j] = std::max(-4.f, std::min(4.f, (float)s.ws[j] / 128.f));
        const bool e = p.tc_window && s.tc_fill;
        const bool clk = !p.reg_clock || !std::isalnum(s.hist[0]);   // reg_clock=1 (Organism): writes only at word boundaries
        for (int r = 0; r < 4; ++r) { float zg = o.hb_g[r], zv = 0.f; for (int j = 0; j < Dims::F; ++j) { zg += o.hw_g[r][j] * a[j]; zv += o.hw_v[r][j] * a[j]; }
            const float g = clk ? 1.f / (1.f + std::exp(-zg)) : 0.f, v = 100.f * std::tanh(zv);
            s.ws_reg_prev[r] = s.ws_reg[r]; s.ws_regG[r] = g; s.ws_reg[r] = (1.f - g) * s.ws_reg[r] + g * v;
            if (e) { s.h_reg_prev[r] = s.ws_reg_prev[r]; s.h_G[r] = g; s.h_g[r] = g; s.h_v[r] = v; s.h_u[r] = zv; } }
        if (e) { std::memcpy(s.h_a, a, sizeof a); s.h_valid = true; s.h_slot = (uint16_t)((s.tc_head + p.tc_window - 1) % p.tc_window); }
    }
    if (!s.ts.q.empty()) org_.tiles_close(s.ts, s.ws, s.hist, s.n_events);   // Organism: after the write head, before the history shift
    if (s.seq_on) s.seq.capture(s.target);
    for (int i = 15; i > 0; --i) s.hist[i] = s.hist[i - 1];
    s.hist[0] = s.target;
    ++s.n_events; ++s.pos; if (s.pos >= s.len) s.active = false;
}

// ---- the step ------------------------------------------------------------------------------------------------------------------
void MultiStream::step() {
    std::vector<Stream*> act; act.reserve(S_);
    for (auto& sp : st_s_) if (sp->active) act.push_back(sp.get());
    if (act.empty()) return;
    const uint32_t n = (uint32_t)act.size();
    for (uint32_t i = 0; i < n; ++i) act[i]->n_events_global = events_ + i;
    auto t = Clock::now();
    pool_.run(n, [&](uint32_t i, unsigned) { p1_address(*act[i]); });                 st_.sec_p1 += since(t); t = Clock::now();
    if (eval_) pool_.run(n, [&](uint32_t i, unsigned) { s1_admit(*act[i]); }); else for (Stream* s : act) s1_admit(*s);                                                 st_.sec_s1 += since(t); t = Clock::now();
    pool_.run(n, [&](uint32_t i, unsigned) { p2_forward(*act[i]); });                 st_.sec_p2 += since(t); t = Clock::now();
    if (eval_) pool_.run(n, [&](uint32_t i, unsigned) { s2_thresholds(*act[i]); }); else for (Stream* s : act) s2_thresholds(*s);                                            st_.sec_s2 += since(t); t = Clock::now();
    pool_.run(n, [&](uint32_t i, unsigned w) { p3_readout(*act[i], part_[w]); });     st_.sec_p3 += since(t); t = Clock::now();
    for (Stream* s : act) { if (eval_ && s->pos >= s->ctx_len) s->item_loss += s->loss; st_.loss_sum += s->loss; st_.correct += s->correct; if (on_event) on_event(s->id, s->sym, s->target, s->loss, s->pred); }
    if (p_.train) s3_learn();                                                           st_.sec_s3 += since(t); t = Clock::now();
    births_.clear();
    if (eval_) pool_.run(n, [&](uint32_t i, unsigned) { s4_structure(*act[i]); }); else for (Stream* s : act) s4_structure(*s);
    if (!births_.empty()) { std::atomic<uint32_t> ok{0}; pool_.run((uint32_t)births_.size(), [&](uint32_t i, unsigned) { if (imprint(*births_[i].first, births_[i].second)) ++ok; }); org_.st.imprint_ok += ok.load(); }
    st_.sec_s4 += since(t); t = Clock::now();
    pool_.run(n, [&](uint32_t i, unsigned) { p4_advance(*act[i]); });                 st_.sec_p4 += since(t);
    events_ += n; st_.events += n; ++st_.steps;
    org_.event_idx = events_;
}

void MultiStream::end_block() {
    std::memcpy(org_.hist, st_s_[0]->hist, sizeof org_.hist);
    if (learner_) learner_->wait();
    auto t = Clock::now();
    org_.end_block(); org_.stats() = BlockStats{};
    st_.sec_block += since(t); t = Clock::now();
    if (learner_) learner_->upload_all();
    st_.sec_upload += since(t);
}

void MultiStream::reload_from_container() {
    if (!learner_) return;
    learner_->wait(); learner_->upload_all(); learner_->set_head(&org_.hw_g[0][0], &org_.hw_v[0][0], org_.hb_g);
}
void MultiStream::export_stream(uint32_t si) {
    if (learner_) learner_->wait();
    Organism& o = org_; Stream& s = *st_s_[si];
    std::memcpy(o.hist, s.hist, sizeof o.hist); std::memcpy(o.ws_mem, s.ws_mem, sizeof o.ws_mem); std::memcpy(o.ws, s.ws, sizeof o.ws);
    o.loss_ema = s.loss_ema; o.loss_ema_init = s.loss_ema_init; o.wrong_ema = s.wrong_ema; o.prior = s.prior;
    std::memcpy(o.xin, s.xin, Dims::IN); std::memcpy(o.xin_exact, s.xin_exact, Dims::IN); std::memcpy(o.xin_focus, s.xin_focus, Dims::IN);
    if (s.seq_on) { o.seq = s.seq; o.seq_out = s.seq_out; o.body_probs = s.body_probs; o.seq_events = s.seq_events; o.seq_matched = s.seq_matched; o.seq_conf_sum = s.seq_conf_sum; }
    if (s.agent_on) { o.agent = s.agent; o.agent_obs = s.agent_obs; o.agent_tendency = s.agent_tendency; o.agent_exact_only = s.agent_exact_only; o.agent_last_ptarget = s.agent_last_ptarget; }
    o.ws_scale = s.ws_scale; std::memcpy(o.ws_reg, s.ws_reg, sizeof o.ws_reg);
    o.phase = s.phase; o.ans_state = s.ans_state; o.goal_w = s.goal_w; o.ans_loss = s.ans_loss; o.ans_n = s.ans_n; o.ans_ema = s.ans_ema;
    o.event_idx = events_; o.probs = s.probs; o.logits = s.logits;
    o.tst_ = s.ts; o.tst_.att_slot.clear();   // tiles
}

} // namespace laminar

namespace laminar {
// ---- evaluation overlays ----------------------------------------------------------------------------------------------------
uint32_t MultiStream::lk_hits(const Stream& s, uint32_t id) const {
    if (eval_) { auto it = s.lov.find(id); if (it != s.lov.end()) return it->second.hits; } return org_.tree.locks[id].hits; }
uint32_t MultiStream::lk_err(const Stream& s, uint32_t id) const {
    if (eval_) { auto it = s.lov.find(id); if (it != s.lov.end()) return it->second.err_ema; } return org_.tree.locks[id].err_ema; }
float& MultiStream::route_util(Stream& s, uint32_t u) {
    if (!eval_) return org_.route_utility[u];
    auto& ov = s.uov[u]; if (!ov.ru_set) { ov.ru = org_.route_utility[u]; ov.ru_set = true; } return ov.ru; }
// Organism::threshold, with the activity average read and written through the overlay in evaluation
int32_t MultiStream::threshold(Stream& s, int32_t base, uint32_t unit, bool novel, int stage, int32_t lvl) {
    Organism& o = org_; const Params& p = p_;
    uint8_t* ema;
    if (eval_) { auto& ov = s.uov[unit]; if (!ov.act_set) { std::memcpy(ov.act, &o.act_ema[(size_t)unit * 4], 4); ov.act_set = true; } ema = &ov.act[stage]; }
    else ema = &o.act_ema[(size_t)unit * 4 + stage];
    int32_t t = base;
    if (p.death_rel) { int32_t rel = ((int32_t)*ema * p.death_rel) >> 6; if (rel > t) t = rel; }
    t = (t * (int32_t)o.resist[unit]) / 64;
    if (novel) t = (t * p.novelty_num) / p.novelty_den;
    *ema = (uint8_t)((int32_t)*ema + (((lvl > 127 ? 127 : lvl) - (int32_t)*ema) >> 4));
    return t;
}
// the container organism's stream state into a stream (the inverse of export_stream)
void MultiStream::import_stream(Stream& s) {
    const Organism& o = org_;
    std::memcpy(s.hist, o.hist, sizeof s.hist); std::memcpy(s.ws_mem, o.ws_mem, sizeof s.ws_mem); std::memcpy(s.ws, o.ws, sizeof s.ws);
    s.loss_ema = o.loss_ema; s.loss_ema_init = o.loss_ema_init; s.wrong_ema = o.wrong_ema; s.prior = o.prior;
    std::memcpy(s.xin, o.xin, Dims::IN); std::memcpy(s.xin_exact, o.xin_exact, Dims::IN); std::memcpy(s.xin_focus, o.xin_focus, Dims::IN);
    s.seq_on = o.seq_on; if (o.seq_on) { s.seq = o.seq; s.seq_out = o.seq_out; s.body_probs = o.body_probs; s.seq_events = o.seq_events; s.seq_matched = o.seq_matched; s.seq_conf_sum = o.seq_conf_sum; }
    s.agent_on = o.agent_on; if (o.agent_on) { s.agent = o.agent; s.agent_obs = o.agent_obs; s.agent_tendency = o.agent_tendency; s.agent_exact_only = o.agent_exact_only; s.agent_last_ptarget = o.agent_last_ptarget; }
    s.ws_scale = o.ws_scale; std::memcpy(s.ws_reg, o.ws_reg, sizeof s.ws_reg);
    s.phase = o.phase; s.ans_state = o.ans_state; s.goal_w = o.goal_w; s.ans_loss = o.ans_loss; s.ans_n = o.ans_n; s.ans_ema = o.ans_ema;
    s.n_events = o.event_idx; s.probs = o.probs; s.logits = o.logits;
    s.h_valid = false; s.lov.clear(); s.uov.clear();
    s.ts = o.tst_; s.ts.att_slot.clear();   // tiles
}
std::vector<double> MultiStream::score(const std::vector<ScoreItem>& items) {
    if (p_.train || p_.structural) throw std::runtime_error("score: freeze the model first (set_train(false, false))");
    eval_ = true;
    auto base = std::make_unique<Stream>(); import_stream(*base);
    std::vector<double> out(items.size(), 0.0); size_t next = 0;
    auto load = [&](Stream& s, size_t i) {   // a fresh copy of the frozen stream state reads item i
        const uint32_t id = s.id; s = *base; s.id = id;
        s.buf = items[i].context + items[i].completion; s.ctx_len = items[i].context.size(); s.item_loss = 0.0; s.item = (int64_t)i;
        s.data = (const uint8_t*)s.buf.data(); s.len = s.buf.size(); s.pos = 0; s.active = s.len > 0;
    };
    for (;;) {
        std::vector<std::pair<Stream*, size_t>> refill;
        for (auto& sp : st_s_) { Stream& s = *sp; if (s.active) continue;
            if (s.item >= 0) { out[(size_t)s.item] = s.item_loss; s.item = -1; }
            while (next < items.size() && items[next].context.empty() && items[next].completion.empty()) ++next;
            if (next < items.size()) refill.emplace_back(&s, next++); }
        if (!refill.empty()) pool_.run((uint32_t)refill.size(), [&](uint32_t i, unsigned) { load(*refill[i].first, refill[i].second); });
        if (!any_active()) break;
        step();
    }
    eval_ = false;
    return out;
}
} // namespace laminar
