#include <cctype>
#include "laminar/organism.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <bit>

namespace laminar {

static constexpr uint64_t SENSORY_MASK = (1ULL << Dims::SENSORY_BITS) - 1;
static constexpr int32_t IN_SCALE = 4;      // input activations are +-4

Organism::Organism(const Params& p_) : p(p_), rng(p_.seed) {
    for (int i = 0; i < NUM_LAYERS; ++i) L[i] = layer_desc(i, p.shifts);
    kern.kind = p.kernel;
    wbuf.alloc((size_t)p.units * Dims::UNIT_BYTES);
    wide_acc = p.acc_bits == 16;
    if (wide_acc) { p.quantum <<= 6; p.half_quantum <<= 6; p.lr_shift -= 6; }   // same dynamics for large deltas, 64x finer for small ones
    if (p.train) { if (wide_acc) acc16.assign((size_t)p.units * Dims::UNIT_NIBBLES, 0); else acc.assign((size_t)p.units * Dims::UNIT_NIBBLES, 0); }
    if (p.train && p.adam) { const size_t n = (size_t)p.units * Dims::UNIT_NIBBLES; ad_sh.assign(n, 0.f); ad_m.assign(n, 0.f); ad_v.assign(n, 0.f); ad_g.assign(n, 0.f); ad_uses.assign(p.units, 0); ad_t.assign(p.units, 0); ad_last.assign(p.units, 0); }
    unit_lock.assign(p.units, UINT32_MAX); epoch.assign(p.units, 0); unit_k.assign(p.units, 0);
    if (p.homeo == 2) { hsat.assign((size_t)p.units * NUM_LAYERS, 0.f); hmag.assign((size_t)p.units * NUM_LAYERS, (float)p.homeo_lo); }
    if (p.homeo) { ushift.resize((size_t)p.units * NUM_LAYERS); for (uint32_t u = 0; u < p.units; ++u) std::memcpy(&ushift[(size_t)u * NUM_LAYERS], p.shifts, NUM_LAYERS); }
    resist.assign(p.units, (uint8_t)p.resist_init); alive.assign(p.units, 0); act_ema.assign((size_t)p.units * 4, 0);
    assoc_value.assign(p.units, 0); assoc_mask.assign(p.units, 0);
    l4ret.assign((size_t)p.units * Dims::R, 0);
    assoc_noconseq.assign(p.units, 0); assoc_starve.assign(p.units, 0); route_utility.assign(p.units, p.route_utility_init); recirc_utility.assign(p.units, p.recirc_init); if (p.recirc_learn == 2) { arm_on.assign(p.units, 0.f); arm_off.assign(p.units, 0.f); arm_non.assign(p.units, 0); arm_noff.assign(p.units, 0); }
    // free list in shuffled order so route bodies scatter across the 560 MB store like a real population
    free_units.resize(p.units); for (uint32_t i = 0; i < p.units; ++i) free_units[i] = i;
    for (uint32_t i = p.units - 1; i > 0; --i) { uint32_t j = rng.below(i + 1); std::swap(free_units[i], free_units[j]); }
    std::reverse(free_units.begin(), free_units.end());
    if (p.kpack) {   // packed allocation: per-motif free stacks (lowest id on top), one fill cursor per K class
        const uint32_t nm = (p.units + p.motif_size - 1) / p.motif_size;
        motif_free.assign(nm, {});
        for (uint32_t m = 0; m < nm; ++m) { uint32_t lo = m * p.motif_size, hi = std::min(p.units, lo + p.motif_size); for (uint32_t u = hi; u > lo; --u) motif_free[m].push_back(u - 1); }
        fill_cursor.assign(n_kclass(), 0); for (uint32_t c = 0; c < n_kclass(); ++c) fill_cursor[c] = c;
        free_units.clear();
    }
    kscreen_tab.assign((size_t)256 * n_kclass(), 0.5f); kstats.assign(n_kclass(), {});
    tree.init(p.leaf_max);
    composer_init();
    evid.init(p.evidence_slots, p.future_buckets);
    cur.reserve(p.max_active); nxt.reserve(p.max_active);
    traces.resize(p.max_active);
    if (p.in_words) { emb.resize((size_t)EMB_ROWS * EMB_D); Rng er(mix64(p.seed ^ 0xE3B0ULL)); for (auto& e : emb) e = (float)(er.uniform() * 2.0 - 1.0); }
    if (p.shared_types) {   // reserve the shared cell-type bodies: alive from birth, owned by no lock
        const uint32_t S = std::min(p.shared_types, p.units / 2); shared_base = p.units - S;
        for (uint32_t u = shared_base; u < p.units; ++u) {
            if (p.kpack) { auto& f = motif_free[u / p.motif_size]; f.erase(std::remove(f.begin(), f.end(), u), f.end()); }
            else free_units.erase(std::remove(free_units.begin(), free_units.end(), u), free_units.end());
            alloc_unit(u); birth_unit_weights(u);
        }
    }
    if (p.mtp && (p.ws_regs != 2 || !p.tc_window || !p.ro_adam || p.mtp > 8)) throw std::runtime_error("mtp: needs ws_regs=2 (its ring slots keep the head input), tc_window>0, ro_adam=1, mtp<=8");
    if (p.apical && (p.persist || p.route_return || p.recirc_iters || p.recirc_learn || p.recirc_thresh <= 127 || !p.ws_input || !p.in_struct || p.in_words))
        throw std::runtime_error("apical: its weights live in the L4 block (needs recirculation, persist and route_return off) and it reads the register input cells (ws_input=1 in_struct=1 in_words=0)");
    if (!p.fm_h.empty()) {   // fixed motifs: every body exists from the start and never dies; no structure learning
        p.structural = false; p.episodic = false; p.market = false; p.agency = 0; uint32_t base = 0;   // agency focuses the address tree's bodies: off
        for (size_t m = 0; m < p.fm_h.size(); ++m) { const uint32_t n = p.fm_h[m] == 1 ? 256u : p.fm_n[m]; fm_base.push_back(base);
            if (base + n > p.units) throw std::runtime_error("motifs: populations exceed units");
            for (uint32_t u = base; u < base + n; ++u) {
                if (p.kpack) { auto& f = motif_free[u / p.motif_size]; f.erase(std::remove(f.begin(), f.end(), u), f.end()); }
                else free_units.erase(std::remove(free_units.begin(), free_units.end(), u), free_units.end());
                alloc_unit(u); birth_unit_weights(u); }
            base += n; }
        if (p.tiles) {   // the pyramid's merge populations, one per level above words
            if (p.tile_k < 2) throw std::runtime_error("tiles: tile_k must be >= 2");
            for (uint32_t l = 1; l < p.tiles; ++l) { tile_base.push_back(base);
                if (base + p.tile_bodies > p.units) throw std::runtime_error("tiles: merge populations exceed units (need " + std::to_string(base + (p.tiles - l) * p.tile_bodies) + ")");
                for (uint32_t u = base; u < base + p.tile_bodies; ++u) {
                    if (p.kpack) { auto& f = motif_free[u / p.motif_size]; f.erase(std::remove(f.begin(), f.end(), u), f.end()); }
                    else free_units.erase(std::remove(free_units.begin(), free_units.end(), u), free_units.end());
                    alloc_unit(u); birth_unit_weights(u); }
                base += p.tile_bodies; }
            tst_.q.assign(p.tiles, {}); tst_.n.assign(p.tiles, 0); tst_.wm.assign(p.tile_wm, Tile{}); tst_.wm_live.assign(p.tile_wm, 0);
            tile_M.assign(((size_t)p.tiles * p.tile_k + p.tile_wm) * Dims::F * Dims::F, 0.f); tile_Mg = tile_M; tile_Mm = tile_M; tile_Mv = tile_M;
            if (p.tile_fw) { const size_t FF = (size_t)Dims::F * Dims::F; tst_.fw_A.assign(FF, 0.f); fw_W.assign(3 * FF, 0.f); Rng fr(mix64(p.seed ^ 0xFA57ULL));   // Wq, Wk small random; Wo 0
                for (size_t i = 0; i < 2 * FF; ++i) fw_W[i] = (float)(fr.uniform() * 2 - 1) * 0.1f; fw_Wg.assign(fw_W.size(), 0.f); fw_Wm = fw_Wg; fw_Wv = fw_Wg; }
            if (p.tile_att) { const size_t FF = (size_t)Dims::F * Dims::F; tile_A.assign(3 * FF, 0.f); Rng ar(mix64(p.seed ^ 0xA77E57ULL));   // Wq, Wk small random; Wo 0
                for (size_t i = 0; i < 2 * FF; ++i) tile_A[i] = (float)(ar.uniform() * 2 - 1) * 0.1f; tile_Ag.assign(tile_A.size(), 0.f); tile_Am = tile_Ag; tile_Av = tile_Ag; }
            if (p.tile_ret) { ret_W.assign((size_t)Dims::F * Dims::F * ret_heads(), 0.f); ret_Wg = ret_W; ret_Wm = ret_W; ret_Wv = ret_W;   // Wo 0: the read starts silent
                tst_.fq_n.assign(4096, 0.f); tst_.fq_t.assign(4096, 0); } }
    }
    if (p.tiles && fm_base.empty()) throw std::runtime_error("tiles: needs fixed motifs (motifs=...)");
    if (p.tile_split && p.tiles) { wtl.assign((size_t)p.vocab * Dims::F, 0.f); wtl_g = wtl; wtl_m = wtl; wtl_v = wtl; }
    if (p.motif_mix) { if (!p.fm_stream) throw std::runtime_error("motif_mix needs motif_stream=1"); mix_th.assign((size_t)p.units * 8, 0.f); mix_g = mix_th; mix_m = mix_th; mix_v = mix_th; }
    { const bool has0 = std::find(p.fm_h.begin(), p.fm_h.end(), 0u) != p.fm_h.end();
      if (has0 != (p.iter_hops > 0) || (p.iter_hops && (p.fm_h.back() != 0 || !p.fm_stream || p.iter_hops > 16 || p.iter_eval_hops() > 16)))
          throw std::runtime_error("iter_hops: needs motif_stream=1 and the iterated population (context 0, e.g. 0:256) last in motifs=; at most 16 hops");
      if (p.iter_hops && (p.motif_mix || p.motif_attn) && fm_base.size() - 1 + std::max(p.iter_hops, p.iter_eval_hops()) > 8)
          throw std::runtime_error("iter_hops: motif_mix / motif_attn read at most 8 positions"); }
    if (p.motif_attn) { if (!p.fm_stream) throw std::runtime_error("motif_attn needs motif_stream=1"); att_W.assign(ATT_N, 0.f); Rng ar2(mix64(p.seed ^ 0xA77E57ULL));
        for (size_t i = 0; i < ATT_B; ++i) att_W[i] = ((float)ar2.below(20001) / 10000.f - 1.f) * 0.5f;   // random projections, zero position bias
        att_Wg.assign(ATT_N, 0.f); att_Wm = att_Wg; att_Wv = att_Wg; }
    if (p.disinh > 0.f) { if (fm_base.empty() || fm_base.size() > 8 || !p.ws_input || !p.in_struct || p.in_words) throw std::runtime_error("disinh: needs 1..8 fixed motifs and the register input cells (ws_input=1 in_struct=1 in_words=0)");
        dis_w.assign(fm_base.size() * 5, 0.f); dis_g = dis_w; dis_m = dis_w; dis_v = dis_w; }
    if (p.ws_split5) { if (fm_base.empty()) throw std::runtime_error("ws_split5: needs fixed motifs"); const size_t M = fm_base.size();
        wsm5.assign(M * Dims::E, 0); wsp5.assign((size_t)p.vocab * M * Dims::E, 0.f); wsp5_g = wsp5; wsp5_m = wsp5; wsp5_v = wsp5; dl5m.assign(M * Dims::E, 0.f); }
    if (p.ro_hidden) { const size_t H = p.ro_hidden, D = Dims::F + (p.ws_split ? fm_base.size() * Dims::F : 0); Rng hr(mix64(p.seed ^ 0x81DDE7ULL));
        rh_W1.assign(H * D, 0.f); for (auto& x : rh_W1) x = (float)(hr.uniform() * 2 - 1) * (1.f / std::sqrt((float)D)); rh_W1g.assign(H * D, 0.f); rh_W1m = rh_W1g; rh_W1v = rh_W1g;
        rh_b1.assign(H, 0.f); rh_b1g = rh_b1; rh_b1m = rh_b1; rh_b1v = rh_b1; rh_W2.assign((size_t)p.vocab * H, 0.f); rh_W2g = rh_W2; rh_W2m = rh_W2; rh_W2v = rh_W2; }
    if (p.ws_split) { if (fm_base.empty()) throw std::runtime_error("ws_split: needs fixed motifs (motifs=...)");
        const size_t M = fm_base.size(); wsp.assign((size_t)p.vocab * M * Dims::F, 0.f); wsp_g = wsp; wsp_m = wsp; wsp_v = wsp; wsm.assign(M * Dims::F, 0); dl6m.assign(M * Dims::F, 0.f); }
    logits.assign(p.vocab, 0.f); probs.assign(p.vocab, 0.f); cf_logits.assign(p.vocab, 0.f); cf_probs.assign(p.vocab, 0.f);
    prior.assign(p.vocab, 1.f / p.vocab);
    if (p.seqmem && p.vocab == 256) { seq.init(p.seq_window, p.seq_hash_bits); seq.min_support = p.seq_min_support; seq.copy_max = p.seq_copy_max; seq_out.assign(256, 0.f); body_probs.assign(256, 0.f); seq_on = true;
        if (p.seq_consolidate) { seq.init_consolidated(p.seq_cbits); seq.class_gain = p.seq_class_gain; }
        seq.goal_extend = p.seq_goal_extend; seq.util_rate = p.seq_util_rate; if (p.seq_chain) { seq.init_chain(); seq.max_depth = (int)std::min<uint32_t>(p.seq_max_depth, 8); seq.abstain_below = p.seq_abstain; seq.arbitrate = p.seq_arbitrate; seq.thought_df = p.seq_thought_df; if (p.seq_chain_mode == 2) { seq.init_learned_reading(); seq.chain_neg = p.seq_chain_neg; } if (p.seq_rel_gain > 0.f) { seq.init_rel(); seq.rel_gain = p.seq_rel_gain; seq.rel_closure = p.seq_rel_closure; seq.goal_rel = p.seq_goal_rel; seq.attr_on = p.seq_attr; seq.attr_eq = p.seq_attr_eq; } }
        if (p.seq_topic_gain > 0.f) { seq.init_bubbles(p.seq_dialogues); seq.topic_gain = p.seq_topic_gain; }
        if (p.seq_attn) { seq.attn.init((int)p.seq_attn, ws_mem, p.seed); seq.attn.warm = p.attn_warm; seq.attn.epochs = (int)p.attn_epochs; seq.attn.lr = p.attn_lr; seq.attn.fg_lr = p.attn_fg_lr; seq.attn.holdout = (int)p.attn_holdout; seq.attn_on = true; }
        if (p.seq_units && p.levels) { seq.unit_tokens = p.lvl_warm == 0; seq.unit_sentences = p.seq_units >= 2; } }
    if (p.vocab > (uint32_t)Dims::F) {
        wout.resize((size_t)p.vocab * Dims::F);
        for (auto& w : wout) w = (float)(rng.uniform() * 2.0 - 1.0) * 0.05f;
        woutT.resize(wout.size()); for (uint32_t v = 0; v < p.vocab; ++v) for (int k = 0; k < Dims::F; ++k) woutT[(size_t)k * p.vocab + v] = wout[(size_t)v * Dims::F + k];
        err_idx.reserve(p.vocab); err_val.reserve(p.vocab);
    }
    if (p.agency) agency_init();
}

// ---------------------------------------------------------------------------------------------
// Agency kernel wiring. The kernel observes the organism's condition, its tendencies are focus operations
// over the organism's own L1 input (where the Gaussian gain sits: recent history, hashed context, symbol
// code, persistent workspace) and over which evidence stream is attended (episodic exact locks vs generic),
// and its consequence is the organism's committed prediction outcome. Commitment makes a focus setting
// persist across bytes; DAN-gated credit makes it consequence-sensitive.
static constexpr int AG_OBS = 64;
void Organism::agency_init() {
    agency::KernelConfig c;
    c.obs_dim = AG_OBS; c.n_goals = p.agency_task ? 3 : 2; c.consequence_dim = p.agency_task ? 5 : 4; c.n_units = (int)p.agency_units; c.k_active = (int)p.agency_k;
    c.trace_lifetime = p.agency_trace_life;
    c.focus_dim = Dims::IN; c.focus_width = p.agency_focus_width; c.focus_floor = p.agency_focus_floor; c.focus_active = true;
    c.route_lr = p.agency_route_lr; c.arousal = p.agency_arousal; c.baseline_cost = p.agency_cost; c.epistemic_channel = 1; c.seed = p.agency_seed ^ p.seed;
    // tendencies from the op list
    std::vector<std::string> names; size_t s = 0; const std::string& L = p.agency_ops;
    while (s <= L.size()) { size_t e = L.find(',', s); if (e == std::string::npos) e = L.size(); if (e > s) names.push_back(L.substr(s, e - s)); s = e + 1; }
    if (p.agency == 2) names = {"hold"};                                  // control: fixed bump, nothing can move it
    c.n_tendencies = (int)names.size();
    agent.init(c);
    for (auto& n : names) {
        agency::FocusOp op;
        if (n == "hold") op = agency::op_hold;
        else if (n == "toward_error") op = agency::op_toward_error;
        else if (n == "sweep") op = agency::op_sweep;
        else if (n == "to_recent") op = agency::op_move_to(20.f);          // key bits 8..31: raw last three symbols
        else if (n == "to_context") op = agency::op_move_to(48.f);         // key bits 32..63: hashed context
        else if (n == "to_symbol") op = agency::op_move_to(80.f);          // symbol code
        else if (n == "to_memory") op = agency::op_move_to(112.f);         // persistent workspace cells
        else if (n == "attend_episodic") op = agency::op_attend(1);
        else if (n == "attend_generic") op = agency::op_attend(0);
        else throw std::runtime_error("unknown agency op " + n);
        agent.ops.push_back(op); agent.op_names.push_back(n);
    }
    agent.set_goal(0, 1.0f);                   // predict the next byte
    agent.set_goal(1, p.agency_epistemic);     // learn
    if (p.agency_task) agent.set_goal(2, p.agency_task_demand);   // complete the public task
    agent_obs.assign(AG_OBS, 0.f); agent_on = true;
    std::memcpy(xin_focus, xin, Dims::IN);
}
void Organism::agency_observe(uint64_t key, uint32_t ncand, bool novel, bool exact_matched) {
    // condition = persistent workspace (32) + current symbol bits (8) + previous symbol bits (8) + signals (16)
    for (int j = 0; j < Dims::F; ++j) agent_obs[j] = std::tanh((float)ws_mem[j] / 256.f);
    for (int b = 0; b < 8; ++b) { agent_obs[32 + b] = ((key >> b) & 1) ? 1.f : -1.f; agent_obs[40 + b] = ((key >> (8 + b)) & 1) ? 1.f : -1.f; }
    agent_obs[48] = loss_ema / 3.f; agent_obs[49] = (float)wrong_ema / 65535.f; agent_obs[50] = novel ? 1.f : 0.f; agent_obs[51] = exact_matched ? 1.f : 0.f;
    agent_obs[52] = (float)ncand / 4.f; agent_obs[53] = agent_last_ptarget; agent_obs[54] = agent.learning_progress * 10.f; agent_obs[55] = agent.signals.sensory_error / 8.f;
    for (int i = 56; i < AG_OBS; ++i) agent_obs[i] = p.agency_task ? task_obs[i - 56] : 0.f;   // public goal state and tool/mission outcomes
    agent.perceive(agent_obs.data(), p.train);
    agent_tendency = agent.deliberate(/*consequence_arrived=*/true, /*action_due=*/true);
    agent_exact_only = agent.focus.stream == 1;
}
void Organism::agency_apply_focus() {
    const int32_t* g = agent.focus.gain_q8.data();
    for (int i = 0; i < Dims::IN; ++i) { int32_t v = (int32_t)xin[i] * g[i]; xin_focus[i] = (int8_t)((v >= 0 ? v + 128 : v - 128) >> 8); }
}
void Organism::agency_consequence(float p_target, bool correct, float loss) {
    const float goal0 = p.agency_goal == 1 ? std::max(0.f, 1.f - loss / std::log((float)p.vocab)) : p_target;
    agent_last_ptarget = p_target;
    if (p.agency_task) {   // same layout as the byte channels, with the task goal inserted as goal 2
        float obs[5] = {goal0, 0.f, task_value, 1.f, correct ? 1.f : 0.f};
        const uint8_t reported[5] = {1, 1, (uint8_t)task_reported, 1, 1};
        agent.consequence(obs, /*verified=*/true, reported); task_reported = false; return;
    }
    float obs[4] = {goal0, 0.f /* epistemic: written by the kernel */, 1.f /* effort */, correct ? 1.f : 0.f};
    agent.consequence(obs, /*verified=*/true);
}
std::string Organism::seq_report() const {
    if (!seq_on) return "";
    char buf[3072];
    std::snprintf(buf, sizeof buf, "seqmem: matched %.1f%% of events, mean confidence %.2f, consensus fallbacks %llu, self-instances skipped %llu, topic retrievals %llu (%.1f candidates each)",
                  seq_events ? 100.0 * seq_matched / seq_events : 0.0, seq_matched ? seq_conf_sum / seq_matched : 0.0, (unsigned long long)seq.fallbacks, (unsigned long long)seq.self_skipped,
                  (unsigned long long)seq.topic_retrievals, seq.topic_retrievals ? (double)seq.topic_candidates / seq.topic_retrievals : 0.0);
    if (seq.paths) { char b4[160]; std::snprintf(b4, sizeof b4, "; paths %llu (composed %llu), path credits %llu", (unsigned long long)seq.paths, (unsigned long long)seq.path_answers, (unsigned long long)seq.path_credits); std::strncat(buf, b4, sizeof buf - std::strlen(buf) - 1); }
    if (seq.rel_ingested) { char b5[200]; std::snprintf(b5, sizeof b5, "; relations %llu ingested, %llu proposals, %llu hits (mean gain %.2f), %llu closures (%llu ambiguous skipped), %llu goal expansions, %llu attributed lookups, %llu key-selector credits, %llu attribute merges", (unsigned long long)seq.rel_ingested, (unsigned long long)seq.rel_proposed, (unsigned long long)seq.rel_hits, seq.rel_hits ? seq.rel_gain_sum / seq.rel_hits : 0.0, (unsigned long long)seq.rel_closures, (unsigned long long)seq.rel_ambiguous, (unsigned long long)seq.goal_expansions, (unsigned long long)seq.attr_hits, (unsigned long long)seq.ksel_credits, (unsigned long long)seq.attr_merges); std::strncat(buf, b5, sizeof buf - std::strlen(buf) - 1); }
    if (seq.chains) { char b3[240]; std::snprintf(b3, sizeof b3, "; chains %llu, credited %llu, hits by depth %llu/%llu/%llu, at END %llu, abstained %llu", (unsigned long long)seq.chains, (unsigned long long)seq.chain_credits, (unsigned long long)seq.chain_hits[1], (unsigned long long)seq.chain_hits[2], (unsigned long long)seq.chain_hits[3], (unsigned long long)seq.chain_hits[SequenceMemory::END], (unsigned long long)seq.chain_abstains); std::strncat(buf, b3, sizeof buf - std::strlen(buf) - 1); }
    if (seq.attn_on) { std::string a = "; " + seq.attn.report(); std::strncat(buf, a.c_str(), sizeof buf - std::strlen(buf) - 1); }
    if (seq.commits) { char b2[160]; std::snprintf(b2, sizeof b2, "; commitments %llu, re-alignments %llu, drops %llu", (unsigned long long)seq.commits, (unsigned long long)seq.realigns, (unsigned long long)seq.commit_drops); std::strncat(buf, b2, sizeof buf - std::strlen(buf) - 1); }
    return buf;
}
std::string Organism::kclass_report() const {
    std::string s = "kclass:"; char buf[160];
    { uint64_t gated = 0, live = 0; for (uint32_t u = 0; u < p.units; ++u) if (alive[u]) { ++live; gated += route_utility[u] < p.route_gate_bits; }
      std::snprintf(buf, sizeof buf, " route-gated %llu/%llu", (unsigned long long)gated, (unsigned long long)live); s += buf; }
    if (p.recirc_learn || p.hops_free < p.max_hops) { uint64_t paid = 0, live = 0; for (uint32_t u = 0; u < p.units; ++u) if (alive[u]) { ++live;
          paid += p.recirc_learn == 2 ? (arm_non[u] >= 4 && arm_noff[u] >= 4 && arm_on[u] < arm_off[u] - p.recirc_bits * 0.6931f) : (p.recirc_learn == 1 && recirc_utility[u] > p.recirc_bits); }
      std::snprintf(buf, sizeof buf, " | learned compute: recirculating units %llu/%llu, recirc used %llu explored %llu, extra hops %llu", (unsigned long long)paid, (unsigned long long)live, (unsigned long long)recirc_used, (unsigned long long)recirc_explored, (unsigned long long)hop_extra); s += buf; }
    for (uint32_t c = 0; c < n_kclass(); ++c) { const KClassStats& k = kstats[c];
        std::snprintf(buf, sizeof buf, " K%u[live %llu births %llu emit %llu conseq %.2f tests %llu acc %llu]", p.k_set[c], (unsigned long long)k.live, (unsigned long long)k.births,
                      (unsigned long long)k.emits, k.emits ? (double)k.consequential / k.emits : 0.0, (unsigned long long)k.tests, (unsigned long long)k.accepts); s += buf; }
    return s;
}
std::string Organism::agency_report() const {
    if (!agent_on) return "";
    char buf[512]; std::string s = "agency:";
    for (size_t t = 0; t < agent.op_names.size(); ++t) { std::snprintf(buf, sizeof buf, " %s=%.1f%%", agent.op_names[t].c_str(), agent.actions ? 100.0 * agent.emitted_count[t] / agent.actions : 0.0); s += buf; }
    std::snprintf(buf, sizeof buf, " | commits %llu mean-age %.1f | releases", (unsigned long long)agent.commitment.commits, agent.mean_commit_age()); s += buf;
    for (int r = 0; r < 6; ++r) if (agent.commitment.releases[r]) { std::snprintf(buf, sizeof buf, " %s=%llu", agency::Commitment::reason_name(r), (unsigned long long)agent.commitment.releases[r]); s += buf; }
    std::snprintf(buf, sizeof buf, " | focus centre %.0f stream %d | demand %.2f/%.2f | dan %llu", agent.focus.centre, agent.focus.stream, agent.signals.demand[0], agent.signals.demand[1], (unsigned long long)agent.dan.broadcasts); s += buf;
    return s;
}

// ---- readout helpers: contiguous 32-float dot and a vectorised exp for the 256-way softmax ----
static inline float dot_n(const float* a, const float* b, size_t n) { float s = 0.f; for (size_t i = 0; i < n; ++i) s += a[i] * b[i]; return s; }   // ws_split
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
// exp(x) for x <= 0, ~1e-7 relative error: range reduction by ln2, degree-6 polynomial, exponent bit insertion.
// natural log of 8 positive floats: exponent + a 7-term series on the mantissa folded into [sqrt(1/2), sqrt(2))
// (relative error ~1e-4, ample for a probability blend)
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

// ---------------------------------------------------------------------------------------------
// keys and inputs
uint64_t Organism::make_key() const {
    // bits 0..7 current symbol (sensory), 8..31 raw last three symbols, 32..63 hashed context of the last
    // hist_bytes symbols. With l4_key / ws_key, internal state replaces bits 32..47 / 48..63 so that
    // locks can split on internal state (design doc §6 "conditional future internal states").
    uint64_t k = hist[0] | ((uint64_t)hist[1] << 8) | ((uint64_t)hist[2] << 16) | ((uint64_t)hist[3] << 24);
    uint64_t h = 0x5A5A00000000ULL;
    for (uint32_t i = 1; i <= p.hist_bytes && i < 16; ++i) h = (h * 0x100000001B3ULL) ^ hist[i];
    uint64_t ctx = mix64(h) & 0xFFFFFFFF00000000ULL;
    if (p.l4_key) { uint64_t s = 0; for (int j = 0; j < 16; ++j) if (last_l4[j] > 0) s |= 1ULL << j; ctx = (ctx & 0xFFFF000000000000ULL) | (s << 32); }
    if (p.ws_key) { uint64_t s = 0; for (int j = 0; j < 16; ++j) if (ws_mem[j] > 0) s |= 1ULL << j; ctx = (ctx & 0x0000FFFF00000000ULL) | (s << 48); }
    return k | ctx;
}
// in_struct: 16 features of one byte, each +-S: its 8 bits (ASCII bits carry case and ranges) and 8 character classes
static inline void byte_features(uint8_t c, int8_t S, int8_t* out) {
    for (int b = 0; b < 8; ++b) out[b] = ((c >> b) & 1) ? S : (int8_t)-S;
    const bool alpha = std::isalpha(c) != 0, upper = std::isupper(c) != 0, digit = std::isdigit(c) != 0, space = c == ' ' || c == '\t';
    const bool punct = std::ispunct(c) != 0 && c != '\'' && c != '"'; const int lower = std::tolower(c);
    const bool vowel = alpha && (lower == 'a' || lower == 'e' || lower == 'i' || lower == 'o' || lower == 'u');
    const bool f[8] = {alpha, upper, digit, space, punct, vowel, c == '\n' || c == '\r', c == '\'' || c == '"'};
    for (int k = 0; k < 8; ++k) out[8 + k] = f[k] ? S : (int8_t)-S;
}
void Organism::make_input(uint64_t key, uint8_t sym) {
    if (p.in_struct) {
        const int8_t S = (int8_t)std::min(127, std::max(1, p.in_scale));
        if (p.in_words) {   // track words: a completed word (letters/apostrophes, case-folded) shifts into the last-3 list
            const bool wc = std::isalpha(sym) || sym == '\'';
            if (wc) word_cur = (word_cur ^ (uint64_t)std::tolower(sym)) * 0x100000001B3ULL + 1;
            else if (word_cur) { word_ids[2] = word_ids[1]; word_ids[1] = word_ids[0]; word_ids[0] = (uint32_t)(mix64(word_cur) % EMB_ROWS); word_cur = 0; }
        }
        for (int i = 0; i < (p.in_words ? 3 : 5); ++i) byte_features(i == 0 ? sym : hist[i], S, xin + 16 * i);   // the current byte, then t-1..
        // word / sentence position: letters since the last non-letter, the previous word's length, bytes since a sentence end
        int pos = 0; while (pos < 15 && std::isalpha(pos == 0 ? sym : hist[pos])) ++pos;
        int j = pos; while (j < 15 && !std::isalpha(hist[j])) ++j; int prev = 0; while (j + prev < 15 && std::isalpha(hist[j + prev])) ++prev;
        int sent = 0; while (sent < 15) { const uint8_t c = sent == 0 ? sym : hist[sent]; if (c == '.' || c == '!' || c == '?' || c == '\n') break; ++sent; }
        int8_t* w = xin + (p.in_words ? 48 : 80);
        for (int k = 0; k < 6; ++k) w[k] = pos > k ? S : (int8_t)-S;          // thermometer codes: ordered, so nearby counts share features
        for (int k = 0; k < 5; ++k) w[6 + k] = prev > k + 1 ? S : (int8_t)-S;
        for (int k = 0; k < 5; ++k) w[11 + k] = sent > (1 << k) ? S : (int8_t)-S;
        if (p.in_words) for (int q = 0; q < 3; ++q) { const float* e = &emb[(size_t)word_ids[q] * EMB_D];
            for (uint32_t k = 0; k < EMB_D; ++k) { const float v = e[k] * S; xin[64 + 16 * q + k] = (int8_t)std::max(-127.f, std::min(127.f, std::round(v))); } }
        std::memcpy(xin_exact, xin, Dims::IN);
        if (p.ws_input) {   // the persistent workspace, scaled to +-S by its running mean magnitude
            float m = 0.f; for (int j2 = 0; j2 < Dims::F; ++j2) m += std::abs((float)ws_mem[j2]); m /= Dims::F;
            ws_scale = ws_scale == 0.f ? std::max(m, 1.f) : ws_scale + 0.001f * (m - ws_scale);
            const float g = (float)S / std::max(ws_scale, 1.f);
            if (p.in_words) {   // 16 cells: the 32 workspace dims folded in pairs
                for (int j2 = 0; j2 < 16; ++j2) { const float v = 0.5f * (float)(ws_mem[j2] + ws_mem[j2 + 16]) * g; xin[112 + j2] = (int8_t)std::max(-127.f, std::min(127.f, v)); }
                std::memset(xin_exact + 112, 0, 16);
            } else {
                if (p.ws_bands) for (int b2 = 0; b2 < 4; ++b2) {   // each band on its own scale
                    float mb = 0.f; for (int j2 = 8 * b2; j2 < 8 * b2 + 8; ++j2) mb += std::abs((float)ws_mem[j2]); mb /= 8.f;
                    ws_bscale[b2] = ws_bscale[b2] == 0.f ? std::max(mb, 1.f) : ws_bscale[b2] + 0.001f * (mb - ws_bscale[b2]);
                    const float gb = (float)S / std::max(ws_bscale[b2], 1.f);
                    for (int j2 = 8 * b2; j2 < 8 * b2 + 8; ++j2) { const float v = (float)ws_mem[j2] * gb; xin[96 + j2] = (int8_t)std::max(-127.f, std::min(127.f, v)); } }
                else if (p.ws_mem_in) for (int j2 = 0; j2 < Dims::F; ++j2) { const float v = (float)ws_mem[j2] * g * p.ws_in_gain; xin[96 + j2] = (int8_t)std::max(-127.f, std::min(127.f, v)); }
                else std::memset(xin + 96, 0, Dims::F);   // ws_mem_in=0: no fixed-decay workspace in the input (persistence must be built: registers, L4)
                std::memset(xin_exact + 96, 0, Dims::F);
                if (p.ws_regs) for (int r = 0; r < 4; ++r) xin[124 + r] = (int8_t)std::max(-127.f, std::min(127.f, reg_read(r) * p.reg_scale / 100.f));
            }
        } else if (p.in_words) { std::memset(xin + 112, 0, 16); std::memset(xin_exact + 112, 0, 16); }
        else { std::memset(xin + 96, 0, Dims::F); std::memset(xin_exact + 96, 0, Dims::F); }
        if (diag_zero_wsin) std::memset(xin + (p.in_words ? 112 : 96), 0, p.in_words ? 16 : Dims::F);   // diagnostics: bodies cannot read the workspace memory
        (void)key; return;
    }
    for (int b = 0; b < 64; ++b) xin[b] = (int8_t)(((key >> b) & 1) ? IN_SCALE : -IN_SCALE);
    uint64_t code = mix64(0xC0DE0000ULL + sym);
    for (int b = 0; b < 64; ++b) xin[64 + b] = (int8_t)(((code >> b) & 1) ? IN_SCALE : -IN_SCALE);
    std::memcpy(xin_exact, xin, Dims::IN);
    if (p.ws_input) { for (int j = 0; j < Dims::F; ++j) xin[96 + j] = clamp8(ws_mem[j] >> 3); std::memset(xin_exact + 96, 0, Dims::F); }   // persistent workspace as input
}
uint64_t Organism::route_key(const int8_t* l2) const {
    uint64_t k = 0;
    for (int j = 0; j < Dims::B; ++j) { if (l2[j] > 0) k |= 1ULL << j; if (l2[j] >= 32 || l2[j] <= -32) k |= 1ULL << (24 + j); }
    return k | (mix64(k) & 0xFFFF000000000000ULL);
}
// Death threshold for one stage: absolute floor, or relative to the unit's own running activity at that
// stage (design doc §3: information death; §7: metabolic resistance scales it). Updates the running mean.
int32_t Organism::threshold(int32_t base, uint32_t unit, bool novel, int stage, int32_t level) {
    uint8_t& ema = act_ema[(size_t)unit * 4 + stage];
    int32_t t = base;
    if (p.death_rel) { int32_t rel = ((int32_t)ema * p.death_rel) >> 6; if (rel > t) t = rel; }
    t = (t * (int32_t)resist[unit]) / 64;
    if (novel) t = (t * p.novelty_num) / p.novelty_den;
    ema = (uint8_t)((int32_t)ema + (((level > 127 ? 127 : level) - (int32_t)ema) >> 4));
    return t;
}
static inline int32_t level(const int8_t* v, int n) { int32_t s = 0; for (int i = 0; i < n; ++i) s += v[i] < 0 ? -v[i] : v[i]; return s / n; }
// apical: the gate factors (x256, 0..512) of a body's L5 units from the register input cells its L1 reads (x1[124..127])
static inline void apical_factors(const uint8_t* W, const int8_t* x1, uint32_t shift, int32_t* fac) {
    for (int j = 0; j < Dims::E; ++j) { int32_t a = 0; for (int r = 0; r < 4; ++r) a += nib_get(W, Dims::NIB_L4 + j * 4 + r) * (int32_t)x1[124 + r];
        fac[j] = std::max(0, std::min(512, 256 + (a >> shift))); }
}
// ... and its backward: g5 = d loss / d (gated L5) in, d loss / d (the layer's own L5) out; ga = d loss / d a (the gate's input)
static inline void apical_backward(const uint8_t* W, const int8_t* x1, uint32_t shift, const int8_t* l5, const int8_t* l5g, float leak, float* g5, float* ga) {
    int32_t fac[Dims::E]; apical_factors(W, x1, shift, fac);
    for (int j = 0; j < Dims::E; ++j) { ga[j] = 0.f; if (g5[j] == 0.f) continue;
        const float gg = g5[j] * ((l5g[j] >= 127 || l5g[j] <= -127) ? leak : 1.f);
        ga[j] = gg * (float)l5[j] / 256.f / (float)(1 << shift) * ((fac[j] > 0 && fac[j] < 512) ? 1.f : leak);
        g5[j] = gg * (float)fac[j] / 256.f; }
}

void Organism::perturb(int8_t* v, int n, uint32_t unit, int layer, int sign) {
    if (sign == 0) return;
    uint64_t bits = mix64(event_idx * 0x9E3779B97F4A7C15ULL ^ ((uint64_t)unit << 20) ^ ((uint64_t)layer << 56));
    for (int j = 0; j < n; ++j) { int s = ((bits >> j) & 1) ? 1 : -1; v[j] = clamp8(v[j] + sign * s * p.perturb_delta); }
}

// ---------------------------------------------------------------------------------------------
// units / motifs
// ---- packed allocation and the K screen --------------------------------------------------------
// A route body is placed motif by motif so that a motif's neighbourhood is dense with related routes:
// a split child goes into its parent's motif while there is room, a fresh birth goes to the motif
// currently being filled in the K class the screen chooses for this kind of input.
uint32_t Organism::alloc_unit_in_class(uint32_t kc) {
    const uint32_t nm = (uint32_t)motif_free.size(), nk = n_kclass();
    // Small populations may not contain every class. For uneven motif counts,
    // wrapping modulo nm would also move a cursor into a different K class.
    if (kc >= nk || kc >= nm) return UINT32_MAX;
    const uint32_t count = (nm - 1 - kc) / nk + 1;
    for (uint32_t tries = 0; tries < count; ++tries) {
        uint32_t m = fill_cursor[kc];
        if (!motif_free[m].empty()) { uint32_t u = motif_free[m].back(); motif_free[m].pop_back(); return u; }
        fill_cursor[kc] = m + nk < nm ? m + nk : kc;
    }
    return UINT32_MAX;
}
uint32_t Organism::alloc_unit_near(uint32_t parent) {
    if (!p.kpack) return alloc_unit();
    uint32_t m = parent / p.motif_size;
    if (!motif_free[m].empty()) { uint32_t u = motif_free[m].back(); motif_free[m].pop_back(); return alloc_unit(u); }
    uint32_t u = alloc_unit_in_class(kclass_of(parent)); if (u == UINT32_MAX) return alloc_unit();
    return alloc_unit(u);
}
uint32_t Organism::choose_kclass(uint8_t sym) {
    const uint32_t nk = n_kclass();
    if (!p.kscreen) return rng.below(nk);
    if (rng.uniform() < p.kscreen_eps) return rng.below(nk);
    const float* row = &kscreen_tab[(size_t)sym * nk]; uint32_t best = 0;
    for (uint32_t c = 1; c < nk; ++c) if (row[c] > row[best]) best = c;
    return best;
}
uint32_t Organism::alloc_unit() {
    if (p.kpack) {
        uint32_t kc = choose_kclass(hist[0]);
        uint32_t u = alloc_unit_in_class(kc);
        if (u == UINT32_MAX) for (uint32_t c = 0; c < n_kclass() && u == UINT32_MAX; ++c) u = alloc_unit_in_class(c);
        if (u == UINT32_MAX) return UINT32_MAX;
        return alloc_unit(u);
    }
    if (free_units.empty()) return UINT32_MAX;
    uint32_t u = free_units.back(); free_units.pop_back();
    return alloc_unit(u);
}
uint32_t Organism::alloc_unit(uint32_t u) {
    ++kstats[kclass_of(u)].live; ++kstats[kclass_of(u)].births;
    alive[u] = 1; resist[u] = (uint8_t)p.resist_init; epoch[u] = 0; std::memset(&act_ema[(size_t)u * 4], 0, 4);
    if (p.homeo) std::memcpy(&ushift[(size_t)u * NUM_LAYERS], p.shifts, NUM_LAYERS);
    if (p.homeo == 2) for (int l = 0; l < NUM_LAYERS; ++l) { hsat[(size_t)u * NUM_LAYERS + l] = 0.f; hmag[(size_t)u * NUM_LAYERS + l] = (float)p.homeo_lo; }
    std::memset(&l4ret[(size_t)u * Dims::R], 0, Dims::R);
    uint32_t motif = u / p.motif_size; uint32_t k = p.k_set[motif % p.k_set.size()];
    if (k >= p.motif_size) k = p.motif_size - 1;
    if (k > 254) k = 254;
    unit_k[u] = k;
    Rng r(mix64(p.seed * 0x1234567ULL ^ (0xA55A0000ULL + u)));
    assoc_value[u] = r.next();
    uint32_t mbits = p.assoc_extra_bits; for (uint32_t kk = k; kk > 1; kk >>= 1) ++mbits;
    uint64_t m = 0; while (mbits) { uint64_t b = 1ULL << r.below(64); if (!(m & b)) { m |= b; --mbits; } }
    assoc_mask[u] = m; assoc_noconseq[u] = 0; assoc_starve[u] = 0; route_utility[u] = p.route_utility_init;
    return u;
}
void Organism::birth_unit_weights(uint32_t u) {
    Rng r(mix64(p.seed * 0x9876543ULL ^ (0xB00B0000ULL + u)));
    uint8_t* w = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
    for (uint32_t nib = 0; nib < (uint32_t)Dims::UNIT_NIBBLES; ++nib) nib_set(w, nib, (int)r.below(5) - 2);
    if (p.apical) for (uint32_t nib = Dims::NIB_L4; nib < (uint32_t)Dims::UNIT_NIBBLES; ++nib) nib_set(w, nib, 0);   // apical gate starts open (x1)
    if (!acc.empty()) std::memset(&acc[(size_t)u * Dims::UNIT_NIBBLES], 0, Dims::UNIT_NIBBLES);
    if (!acc16.empty()) std::memset(&acc16[(size_t)u * Dims::UNIT_NIBBLES], 0, Dims::UNIT_NIBBLES * sizeof(int16_t));
    if (!ad_sh.empty()) { const size_t b = (size_t)u * Dims::UNIT_NIBBLES;   // a new life: fresh optimiser state (the shadow resyncs at the next step)
        std::fill(ad_m.begin() + b, ad_m.begin() + b + Dims::UNIT_NIBBLES, 0.f); std::fill(ad_v.begin() + b, ad_v.begin() + b + Dims::UNIT_NIBBLES, 0.f);
        std::fill(ad_g.begin() + b, ad_g.begin() + b + Dims::UNIT_NIBBLES, 0.f); ad_uses[u] = 0; ad_t[u] = 0; ad_last[u] = 0; }
}
// adam: exact gradient g_j x_i into the body's buffer; a step after every adam_k uses of the body.
void Organism::adam_grad(uint32_t u, uint32_t nib0, const int8_t* x, int in, const float* g, int out, float scale) {
    float* G = &ad_g[(size_t)u * Dims::UNIT_NIBBLES + nib0];
    for (int j = 0; j < out; ++j) { if (g[j] == 0.f) continue; const float gj = g[j] * scale; float* r = G + (size_t)j * in; for (int i = 0; i < in; ++i) r[i] += gj * (float)x[i]; }
}
void Organism::adam_touch(uint32_t u) {
    const uint32_t n = ++ad_uses[u], k = std::max(1u, p.adam_k);
    const bool due = p.adam_warm ? (n - ad_last[u] >= std::min(k, std::max(1u, ad_last[u]))) : n % k == 0;   // warm: steps at 1, 2, 4, 8, ... then every k
    if (due) { adam_step(u, (float)(n - ad_last[u])); ad_last[u] = n; }
}
void Organism::adam_step(uint32_t u, float n_uses) {
    if (freeze_weights) return;
    if (p.muon) { muon_step(u, n_uses); return; }
    const uint32_t t = ++ad_t[u]; const float c1 = 1.f - std::pow(0.9f, (float)t), c2 = 1.f - std::pow(0.999f, (float)t), lr = p.adam_lr, inv_k = 1.f / std::max(1.f, n_uses);
    uint8_t* w = wbuf.p + (size_t)u * Dims::UNIT_BYTES; const size_t b = (size_t)u * Dims::UNIT_NIBBLES;
    float* sh = &ad_sh[b]; float* m = &ad_m[b]; float* v = &ad_v[b]; float* G = &ad_g[b];
    for (int n = 0; n < Dims::NIB_L4 + Dims::P4; ++n) {
        const int cw = nib_get(w, n); if ((int)std::lround(sh[n]) != cw) sh[n] = (float)cw;   // written elsewhere (birth, imprint, copy): resync
        const float gr = G[n] * inv_k; G[n] = 0.f;
        m[n] = 0.9f * m[n] + 0.1f * gr; v[n] = 0.999f * v[n] + 0.001f * gr * gr;
        if (v[n] == 0.f) continue;
        sh[n] = std::max(-8.49f, std::min(7.49f, sh[n] - lr * ((m[n] / c1) / (std::sqrt(v[n] / c2) + 1e-8f) + p.adam_wd * sh[n])));
        const int nw = (int)std::lround(sh[n]); if (nw != cw) { nib_set(w, n, nw); ++st.level_changes; }
    }
}
// Muon (the user's roadmap after AdamW): per layer matrix W (out x in), Nesterov momentum U = G + mu M (M <- mu M + G), orthogonalised
// by 5 Newton-Schulz iterations (Keller Jordan's quintic: X <- aX + (bA + cA^2)X, A = XX^T, on the short side), applied to the float
// shadow scaled by sqrt(max(1, out/in)); the INT4 weight is the rounded shadow (as with Adam).
void Organism::muon_step(uint32_t u, float n_uses) {
    ++ad_t[u]; const float inv_k = 1.f / std::max(1.f, n_uses), lr = p.muon_lr, mu = p.muon_mom;
    uint8_t* w = wbuf.p + (size_t)u * Dims::UNIT_BYTES; const size_t b = (size_t)u * Dims::UNIT_NIBBLES;
    float* sh = &ad_sh[b]; float* m = &ad_m[b]; float* G = &ad_g[b];
    for (int n = 0; n < Dims::NIB_L4 + Dims::P4; ++n) { const int cw = nib_get(w, n); if ((int)std::lround(sh[n]) != cw) sh[n] = (float)cw; }   // resync
    thread_local std::vector<float> U, X, A, B, T;
    for (int li : {LI_1, LI_2, LI_3, LI_5, LI_6, LI_4}) {
        const int rows = L[li].out, cols = L[li].in; const uint32_t off = L[li].nib_off;
        U.assign((size_t)rows * cols, 0.f); bool any = false;
        for (int i = 0; i < rows * cols; ++i) { const float g = G[off + i] * inv_k; G[off + i] = 0.f; m[off + i] = mu * m[off + i] + g; U[i] = g + mu * m[off + i]; any |= U[i] != 0.f; }
        if (!any) continue;
        const bool tr = rows > cols; const int r = tr ? cols : rows, c = tr ? rows : cols;
        X.assign((size_t)r * c, 0.f); double n2 = 0.0;
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) { const float x = U[(size_t)i * cols + j]; if (tr) X[(size_t)j * c + i] = x; else X[(size_t)i * c + j] = x; n2 += (double)x * x; }
        const float in = 1.f / ((float)std::sqrt(n2) + 1e-7f); for (auto& x : X) x *= in;
        A.assign((size_t)r * r, 0.f); B.assign((size_t)r * r, 0.f); T.assign((size_t)r * c, 0.f);
        for (int it = 0; it < 5; ++it) {
            for (int i = 0; i < r; ++i) for (int k = 0; k < r; ++k) { float s2 = 0.f; for (int j = 0; j < c; ++j) s2 += X[(size_t)i * c + j] * X[(size_t)k * c + j]; A[(size_t)i * r + k] = s2; }
            for (int i = 0; i < r; ++i) for (int k = 0; k < r; ++k) { float s2 = 0.f; for (int j = 0; j < r; ++j) s2 += A[(size_t)i * r + j] * A[(size_t)j * r + k]; B[(size_t)i * r + k] = -4.7750f * A[(size_t)i * r + k] + 2.0315f * s2; }
            for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) { float s2 = 0.f; for (int k = 0; k < r; ++k) s2 += B[(size_t)i * r + k] * X[(size_t)k * c + j]; T[(size_t)i * c + j] = 3.4445f * X[(size_t)i * c + j] + s2; }
            X.swap(T);
        }
        const float scale = std::sqrt(std::max(1.f, (float)rows / (float)cols));
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) { const float o = tr ? X[(size_t)j * c + i] : X[(size_t)i * c + j]; const uint32_t n = off + (uint32_t)(i * cols + j);
            sh[n] = std::max(-8.49f, std::min(7.49f, sh[n] - lr * (scale * o + p.adam_wd * sh[n])));
            const int cw = nib_get(w, n), nw = (int)std::lround(sh[n]); if (nw != cw) { nib_set(w, n, nw); ++st.level_changes; } }
    }
}
void Organism::ro_hidden_input(const float* a, float* u) const {   // ro_hidden: [shared input (incl. tiles) | every motif's channel / 128]
    for (int k = 0; k < Dims::F; ++k) u[k] = a[k];
    for (size_t i = 0; i < wsm.size() && p.ws_split; ++i) u[Dims::F + i] = (float)wsm[i] * (1.0f / 128.0f);
}
uint32_t Organism::weights_changed(uint32_t u, int li) const {
    Rng r(mix64(p.seed * 0x9876543ULL ^ (0xB00B0000ULL + u)));
    const uint8_t* w = wbuf.p + (size_t)u * Dims::UNIT_BYTES; const LayerDesc& d = L[li]; uint32_t n = 0;
    for (uint32_t nib = 0; nib < (uint32_t)Dims::UNIT_NIBBLES; ++nib) { const int b = (int)r.below(5) - 2;
        if (nib >= d.nib_off && nib < d.nib_off + (uint32_t)d.in * d.out) n += nib_get(w, nib) != b; }
    return n;
}
void Organism::free_unit(uint32_t u) {
    alive[u] = 0; unit_lock[u] = UINT32_MAX; --kstats[kclass_of(u)].live;
    if (p.kpack) motif_free[u / p.motif_size].push_back(u); else free_units.push_back(u);
}

void Organism::neighbor_iter(uint32_t u, uint32_t k, uint32_t* out, uint32_t& n) const {
    uint32_t base = (u / p.motif_size) * p.motif_size, m = p.motif_size, off = u - base; n = 0;
    if (base + m > p.units) m = p.units - base;                       // last motif may be partial
    if (k / 2 >= m) k = (m - 1) * 2;
    for (uint32_t d = 1; d <= k / 2; ++d) { out[n++] = base + (off + m - d) % m; out[n++] = base + (off + d) % m; }
}

// ---------------------------------------------------------------------------------------------
// body execution with early exits (runbook §4)
void Organism::run_body(const Event& ev, bool novel, int sign, uint32_t sched_pos) {
    const uint32_t u = ev.unit;
    const uint8_t* W = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
    Trace* tr = nullptr;
    if (ntrace < p.max_active) { tr = &traces[ntrace++]; tr->unit = u; tr->reached = 0; tr->src = ev.source == SRC_ROUTED ? ev.src_trace : (int16_t)-1; tr->nsrc = 0; if (tr->src >= 0) { if (cur_mn > 1 && cur_msrc) { tr->nsrc = cur_mn; for (int q = 0; q < cur_mn; ++q) tr->srcs[q] = cur_msrc[q]; } else { tr->nsrc = 1; tr->srcs[0] = tr->src; } } tr->source = ev.source; tr->hop = ev.hop; tr->consequential = 0; tr->recirced = 0; tr->emitted = 0;
        if (!mix_pi.empty()) { const size_t ti = (size_t)(tr - traces.data()); if (mix_pi.size() >= (ti + 1) * 8) std::fill(mix_pi.begin() + ti * 8, mix_pi.begin() + ti * 8 + 8, 0.f); } }   // not mixed (yet) this event
    else ++st.trace_overflow;
    executed.push_back(u);
    int8_t l1[Dims::A], l2[Dims::B], l3[Dims::C], l5[Dims::E], l6[Dims::F], l4[Dims::R];
    int8_t x2[Dims::IN2], x3[Dims::IN3], x5[Dims::IN5], x6[Dims::IN6];
    float dis_d = 0.f;   // disinh: this body's release (set before L6 runs)
    auto runL = [&](int li, const int8_t* x, int8_t* y) {   // one layer; with homeo, at this body's own adaptive gain
        if ((p.ff_inh > 0.f || p.fb_inh > 0.f || (p.disinh > 0.f && li == LI_6)) && li != LI_4 && !p.homeo) {   // cortical motifs 3 / 4 / 6: shunting (divisive) inhibition of the drive
            const LayerDesc& d = L[li]; int32_t z[64]; int64_t ix = 0; for (int i = 0; i < d.in; ++i) ix += x[i] < 0 ? -x[i] : x[i];
            for (int j = 0; j < d.out; ++j) { int32_t s = 0; const uint32_t off = d.nib_off + (uint32_t)(j * d.in); for (int i = 0; i < d.in; ++i) s += (int32_t)x[i] * nib_get(W, off + (uint32_t)i); z[j] = s; }
            float f = 1.f; if (p.ff_inh > 0.f) f /= 1.f + p.ff_inh * (float)ix / (float)d.in / 127.f;
            if (p.disinh > 0.f && li == LI_6) f /= 1.f + p.disinh * (1.f - dis_d) * (float)ix / (float)d.in / 127.f;   // motif 6: released by context
            if (p.fb_inh > 0.f) { int64_t oy = 0; for (int j = 0; j < d.out; ++j) { const int32_t v = clamp8((int32_t)std::lround(z[j] * f) >> d.shift); oy += v < 0 ? -v : v; }
                f /= 1.f + p.fb_inh * (float)oy / (float)d.out / 127.f; }
            for (int j = 0; j < d.out; ++j) y[j] = clamp8((int32_t)std::lround(z[j] * f) >> d.shift);
            if (tr) tr->sh[li] = L[li].shift; return; }
        if (!p.homeo || (li == LI_6 && !p.homeo_out)) { kern.run(W, L[li], x, y); if (tr) tr->sh[li] = L[li].shift; return; }
        uint8_t& s = ushift[(size_t)u * NUM_LAYERS + li];
        LayerDesc d = L[li]; d.shift = s; kern.run(W, d, x, y);
        if (tr) tr->sh[li] = s;
        if (!p.train) return;
        int sat = 0, m = 0; for (int j = 0; j < d.out; ++j) { const int a = y[j] < 0 ? -y[j] : y[j]; m += a; sat += a >= 127; }
        if (p.homeo == 1) { if (sat * 8 > d.out && s < 15) ++s; else if (m < p.homeo_lo * d.out && s > 0) --s; return; }
        float& se = hsat[(size_t)u * NUM_LAYERS + li]; float& me = hmag[(size_t)u * NUM_LAYERS + li];
        se += ((float)sat / d.out - se) * (1.f / 32.f); me += ((float)m / d.out - me) * (1.f / 32.f);
        if (se > p.homeo_sat && s < 15) { ++s; se = 0.f; me *= 0.5f; }                  // halve the gain; the averages restart at the new scale
        else if (me < (float)p.homeo_lo * 0.5f && s > 0) { --s; me *= 2.f; se = 0.f; }
    };
    // an episodic (exact-key) body must see key-determined input only: no persistent workspace, no return state
    const bool exact = p.episodic && unit_lock[u] != UINT32_MAX && tree.locks[unit_lock[u]].mask == ~0ULL;
    static const int8_t zero_ret[Dims::R] = {0};
    const int8_t* ret = (exact || diag_zero_return) ? zero_ret : &l4ret[(size_t)u * Dims::R];
    const int8_t* x1 = exact ? xin_exact : (agent_on ? xin_focus : xin);   // the agency kernel's focus gain shapes generic bodies' input
    int8_t deep_ret[Dims::R];
    if (p.deep_in && !exact) { std::memcpy(deep_ret, x1, Dims::R2); std::memcpy(deep_ret + Dims::R2, x1, Dims::R6); ret = deep_ret; }   // motif 1: the byte, first-hand
    int8_t iter_ret[Dims::R];
    if (ev.hop >= 2 && p.iter_hops) { const uint64_t hh = ev.hop - 2u;   // iter_hops: the per-hop cue (a fixed +-64 pattern per hop)
        for (int i = 0; i < Dims::R; ++i) iter_ret[i] = (mix64(0x17E7ULL * (hh + 1) + (uint64_t)i) & 1) ? 64 : -64; ret = iter_ret; }

    { StageTimer T(&st, ST_L1, p.stage_timers);
      runL(LI_1, x1, l1); st.macs += Dims::P1; st.weight_bytes += Dims::P1 / 2; ++st.l1_admissions;
      perturb(l1, Dims::A, u, LI_1, sign);
      if (layer_hook) layer_hook(u, LI_1, l1);
      if (tr) { tr->reached |= R_L1; std::memcpy(tr->l1, l1, Dims::A); }
      { int32_t lv = level(l1, Dims::A); if (lv < threshold(p.death_l1, u, novel, 0, lv)) return; } }
    ++st.l1_survive;

    { StageTimer T(&st, ST_L23, p.stage_timers);
      std::memcpy(x2, l1, Dims::A); std::memcpy(x2 + Dims::A, ret, Dims::R2);
      std::memcpy(x3, l1, Dims::A);
      if (p.fm_stream && ev.source == SRC_ROUTED && ev.src_trace >= 0 && !diag_zero_msgs) {   // motif_stream: the mean of the earlier bodies' L6
          const int m = ev.src_trace + 1; const float sc = 0.5f / (float)m;
          if (!mix_th.empty() || !att_W.empty()) {   // motif_mix / motif_attn: which earlier bodies this body reads, learned
              float th[8] = {0}; if (!mix_th.empty()) for (int q = 0; q < m && q < 8; ++q) th[q] = mix_th[(size_t)u * 8 + q];
              if (!att_W.empty() && m < 8) { float qv[8], kv[8]; att_query(m, l1, qv);   // content: this body's query against every sender's key
                  for (int q = 0; q < m && q < 8; ++q) { th[q] += att_W[ATT_B + (size_t)m * 8 + q]; if (!(traces[q].reached & R_L6)) continue;
                      att_key(traces[q].l6, kv); float s = 0.f; for (int d = 0; d < 8; ++d) s += qv[d] * kv[d]; th[q] += s * 0.35355339f; } }
              float pi[8], mx = -1e30f, Z = 0.f; for (int q = 0; q < m && q < 8; ++q) mx = std::max(mx, th[q]);
              for (int q = 0; q < m && q < 8; ++q) { pi[q] = std::exp(th[q] - mx); Z += pi[q]; } for (int q = 0; q < m && q < 8; ++q) pi[q] /= Z;
              if (tr) { const size_t ti = (size_t)(tr - traces.data()); if (mix_pi.size() < (ti + 1) * 8) mix_pi.resize((ti + 1) * 8, 0.f); for (int q = 0; q < 8; ++q) mix_pi[ti * 8 + q] = q < m ? pi[q] : 0.f; }
              const float mres = p.mix_res, mu0 = mres / (float)std::min(m, 8); for (int q = 0; q < m && q < 8; ++q) pi[q] = mu0 + (1.f - mres) * pi[q];   // mix_res
              if (tr && mres > 0.f) { const size_t ti = (size_t)(tr - traces.data()); for (int q = 0; q < m && q < 8; ++q) mix_pi[ti * 8 + q] = pi[q]; }   // (the backward's weights)
              for (int i = 0; i < Dims::B; ++i) { float sum = 0.f; for (int t = 0; t < m && t < 8; ++t) if (traces[t].reached & R_L6) sum += pi[t] * (float)traces[t].l6[i];
                  x3[Dims::A + i] = clamp8((int32_t)std::lround(sum * 0.5f)); }
          } else
          for (int i = 0; i < Dims::B; ++i) { int32_t sum = 0; for (int t = 0; t < m; ++t) if (traces[t].reached & R_L6) sum += traces[t].l6[i];
              x3[Dims::A + i] = clamp8((int32_t)std::lround((float)sum * sc)); }
      } else if (ev.source == SRC_ROUTED && ev.src_trace >= 0 && !diag_zero_msgs) {
          if (cur_mn > 1 && cur_msrc) { for (int i = 0; i < Dims::B; ++i) { int32_t s = 0; for (int q = 0; q < cur_mn; ++q) s += traces[cur_msrc[q]].l2[i]; x3[Dims::A + i] = clamp8(s); } }   // summed, full width
          else if (traces[ev.src_trace].reached & R_L23) std::memcpy(x3 + Dims::A, traces[ev.src_trace].l2, Dims::B);
          else std::memset(x3 + Dims::A, 0, Dims::B);   // motif_wire: the sender stopped at L1 this byte (its trace's L2 is stale)
      } else std::memset(x3 + Dims::A, 0, Dims::B);
      runL(LI_2, x2, l2); runL(LI_3, x3, l3);
      st.macs += Dims::P2 + Dims::P3; st.weight_bytes += (Dims::P2 + Dims::P3) / 2;
      perturb(l2, Dims::B, u, LI_2, sign); perturb(l3, Dims::C, u, LI_3, sign);
      if (layer_hook) { layer_hook(u, LI_2, l2); layer_hook(u, LI_3, l3); }
      if (tr) { tr->reached |= R_L23; std::memcpy(tr->x2, x2, Dims::IN2); std::memcpy(tr->x3, x3, Dims::IN3); std::memcpy(tr->l2, l2, Dims::B); std::memcpy(tr->l3, l3, Dims::C); } }

    // cheap K-neighbour lock tests: only accepted neighbours become events (design doc §4, runbook §6)
    if (!diag_no_routing && ev.hop < p.max_hops && sched_pos == UINT32_MAX && (p.route_ungated || ev.hop < p.hops_free || route_utility[u] > p.hop_gate_bits)) {
        if (ev.hop >= p.hops_free) ++hop_extra;
        StageTimer T(&st, ST_NEIGHBOR, p.stage_timers);
        uint64_t rk = route_key(l2); uint32_t nb[256]; uint32_t nn;
        if (p.nbr_route) neighbor_iter(u, unit_k[u], nb, nn); else nn = 0;
        if (p.shared_types && p.shared_types > p.shared_always && ev.source != SRC_ROUTED && u < shared_base && nxt.size() < p.max_active) {   // a specialist calls its shared cell type
            const uint32_t pool = p.units - shared_base > p.shared_always ? p.units - shared_base - p.shared_always : 0;
            const uint32_t v = pool ? shared_base + p.shared_always + (uint32_t)(mix64(0x5EA7ULL ^ u) % pool) : UINT32_MAX;
            nxt.push_back(Event{v, SRC_ROUTED, (uint8_t)(ev.hop + 1), (int16_t)(tr ? (int)(tr - traces.data()) : -1)}); ++st.neighbor_accepts;
        }
        KClassStats& ks = kstats[kclass_of(u)];
        for (uint32_t i = 0; i < nn; ++i) {
            uint32_t v = nb[i]; if (!alive[v]) continue;
            ++st.neighbor_tests; ++ks.tests;
            if (((rk ^ assoc_value[v]) & assoc_mask[v]) == 0 && nxt.size() < p.max_active) {
                ++st.neighbor_accepts; ++ks.accepts;
                nxt.push_back(Event{v, SRC_ROUTED, (uint8_t)(ev.hop + 1), (int16_t)(tr ? (int)(tr - traces.data()) : -1)});
                accepted.push_back(AcceptedMsg{v, u});
            }
        }
    }
    { int32_t lv = (level(l2, Dims::B) * Dims::B + level(l3, Dims::C) * Dims::C) / (Dims::B + Dims::C);
      if (lv < threshold(p.death_l23, u, novel, 1, lv)) return; }
    ++st.l23_survive;

    { StageTimer T(&st, ST_L5, p.stage_timers);
      std::memcpy(x5, l2, Dims::B); std::memcpy(x5 + Dims::B, l3, Dims::C);
      runL(LI_5, x5, l5); st.macs += Dims::P5; st.weight_bytes += Dims::P5 / 2;
      perturb(l5, Dims::E, u, LI_5, sign);
      if (layer_hook) layer_hook(u, LI_5, l5);
      if (tr) { tr->reached |= R_L5; std::memcpy(tr->x5, x5, Dims::IN5); std::memcpy(tr->l5, l5, Dims::E); }
      if (p.apical) { int32_t fac[Dims::E]; apical_factors(W, x1, p.apical_shift, fac);   // the dendritic gate (the trace keeps the layer's own L5;
          for (int j = 0; j < Dims::E; ++j) l5[j] = clamp8((l5[j] * fac[j] + 128) >> 8); }  // x6 below holds the gated one)
      { int32_t lv = level(l5, Dims::E); if (lv < threshold(p.death_l5, u, novel, 2, lv)) return; } }
    ++st.l5_survive;

    { StageTimer T(&st, ST_L6, p.stage_timers);
      std::memcpy(x6, l5, Dims::E); std::memcpy(x6 + Dims::E, ret + Dims::R2, Dims::R6);
      if (p.disinh > 0.f) dis_d = dis_release(u, x1);
      runL(LI_6, x6, l6); st.macs += Dims::P6; st.weight_bytes += Dims::P6 / 2;
      if (tr) { tr->reached |= R_L6; std::memcpy(tr->x6, x6, Dims::IN6); std::memcpy(tr->l6, l6, Dims::F); }
      int32_t lv = level(l6, Dims::F);
      if (lv < threshold(p.death_l6, u, novel, 3, lv)) return;
      if (l6_hook) l6_hook(u, l6);
      ++st.l6_survive;
      // emit to the shared workspace. A routed body's emission is gated by its utility: below the bits threshold it
      // is still computed and scored counterfactually (so it can earn its way back) but does not enter the workspace.
      // Episodic bodies are gated the same way (item 3): silent while their utility is below the threshold, unless
      // the agency kernel is attending the episodic stream, in which case they always speak.
      const bool gated_out = ((diag_routed_silent || p.route_return) && ev.source == SRC_ROUTED) || (p.route_gate && !p.route_ungated && route_utility[u] < p.route_gate_bits)
                             && (ev.source == SRC_ROUTED || (exact && !(agent_on && agent_exact_only) && !p.exact_only));
      if (tr) tr->consequential = gated_out ? 2 : 0;   // 2 = "silent this event"
      const int32_t gain = exact ? p.exact_gain : 1;
      const bool silent_m = (!silenced.empty() && silenced[u / p.motif_size]) || (silenced_level >= 0 && !unit_level.empty() && unit_level[u] == silenced_level);
      if (!gated_out && !silent_m && ws_writer(u)) for (int j = 0; j < Dims::F; ++j) ws[j] += gain * l6[j];
      if (!gated_out && !silent_m && !wsm.empty()) { int32_t* wm = &wsm[(size_t)motif_of(u) * Dims::F]; for (int j = 0; j < Dims::F; ++j) wm[j] += gain * l6[j]; }   // ws_split
      if (!gated_out && !silent_m && !wsm5.empty()) { int32_t* wm = &wsm5[(size_t)motif_of(u) * Dims::E]; for (int j = 0; j < Dims::E; ++j) wm[j] += gain * l5[j]; }   // ws_split5
      if (tr) tr->emitted = !gated_out && !silent_m;
      if (exact || lv < p.recirc_thresh) return;

    { StageTimer T4(&st, ST_L4, p.stage_timers);                             // L4 only when recirculation is active
      runL(LI_4, l6, l4); st.macs += Dims::P4; st.weight_bytes += Dims::P4 / 2; ++st.l4_runs;
      if (tr) tr->reached |= R_L4; }
    // lever 1: depth per step. The body re-reads its own return within this byte: L2/L3 see the fresh L4 output,
    // L5 and L6 recompute, and the new emission REPLACES the previous one in the workspace. Stops when L6 falls
    // below the recirculation threshold. Learning sees the last pass (the traces are overwritten).
    uint32_t iters = p.recirc_iters;
    if (p.recirc_learn == 2 && iters) {                     // two-armed: use the tool where its ACTUAL loss beats not using it; sample the other arm now and then
        const uint16_t nOn = arm_non[u], nOff = arm_noff[u];
        bool use = nOn >= 4 && nOff >= 4 && arm_on[u] < arm_off[u] - p.recirc_bits * 0.6931f;   // margin in nats
        if (p.train && (nOn < 4 || nOff < 4 || rng.uniform() < p.recirc_explore)) use = nOn <= nOff ? true : false;   // balance the arms while either is unmeasured; explore otherwise
        if (!use) iters = 0; else { ++recirc_used; }
        if (tr) tr->recirced = use ? 2 : 3;                 // 2 = arm on, 3 = arm off (for the outcome credit)
    }
    if (p.recirc_learn == 1 && iters) {                     // learned: recirculate where it has paid, explore rarely
        const bool paid = recirc_utility[u] > p.recirc_bits, explore = !paid && p.train && rng.uniform() < p.recirc_explore;
        if (!paid && !explore) iters = 0; else { if (explore) ++recirc_explored; else ++recirc_used; if (tr) { std::memcpy(tr->l6_pre, l6, Dims::F); tr->recirced = 1; } }
    }
    for (uint32_t it = 0; it < iters; ++it) {
      std::memcpy(x2 + Dims::A, l4, Dims::R2);
      runL(LI_2, x2, l2); runL(LI_3, x3, l3);
      std::memcpy(x5, l2, Dims::B); std::memcpy(x5 + Dims::B, l3, Dims::C); runL(LI_5, x5, l5);
      std::memcpy(x6, l5, Dims::E); std::memcpy(x6 + Dims::E, l4 + Dims::R2, Dims::R6);
      int8_t l6n[Dims::F]; runL(LI_6, x6, l6n);
      st.macs += Dims::P2 + Dims::P3 + Dims::P5 + Dims::P6; st.weight_bytes += (Dims::P2 + Dims::P3 + Dims::P5 + Dims::P6) / 2; ++st.recirc_passes;
      if (!gated_out) for (int j = 0; j < Dims::F; ++j) ws[j] += gain * ((int32_t)l6n[j] - (int32_t)l6[j]);
      std::memcpy(l6, l6n, Dims::F);
      if (tr) { std::memcpy(tr->x2, x2, Dims::IN2); std::memcpy(tr->l2, l2, Dims::B); std::memcpy(tr->l3, l3, Dims::C); std::memcpy(tr->x5, x5, Dims::IN5); std::memcpy(tr->l5, l5, Dims::E); std::memcpy(tr->x6, x6, Dims::IN6); std::memcpy(tr->l6, l6, Dims::F); }
      if (level(l6, Dims::F) < p.recirc_thresh) break;
      runL(LI_4, l6, l4); st.macs += Dims::P4; st.weight_bytes += Dims::P4 / 2;
    }
    std::memcpy(&l4ret[(size_t)u * Dims::R], l4, Dims::R); std::memcpy(last_l4, l4, Dims::R);
    }
}

// One forward pass over the event's schedule. replay==nullptr builds the schedule dynamically (pass A);
// otherwise the recorded schedule is re-executed (pass B of paired perturbation, runbook §8).
void Organism::run_pass(uint64_t key, bool novel, int sign, const std::vector<Event>* replay) {
    (void)key;
    std::memset(ws, 0, sizeof ws); ntrace = 0; ++cur_epoch; if (!wsm.empty()) std::fill(wsm.begin(), wsm.end(), 0); if (!wsm5.empty()) std::fill(wsm5.begin(), wsm5.end(), 0);
    if (replay) {
        for (uint32_t i = 0; i < replay->size(); ++i) run_body((*replay)[i], novel, sign, i);
        if (p.route_return) route_consequence();
        return;
    }
    for (uint32_t hop = 0; hop <= p.max_hops && !cur.empty(); ++hop) {
        nxt.clear();
        if (p.msg_merge) {   // one run per routed receiver per hop, on the sum of all its senders' messages
            if (mslot.size() != p.units) mslot.assign(p.units, -1);
            msrc.assign(cur.size(), {}); mcount.assign(cur.size(), 0); mskip.assign(cur.size(), 0);
            for (size_t i = 0; i < cur.size(); ++i) { const Event& ev = cur[i]; if (ev.source != SRC_ROUTED || ev.src_trace < 0) continue;
                int32_t& s = mslot[ev.unit];
                if (s >= 0 && (size_t)s < i && cur[s].unit == ev.unit && mcount[s]) { if (mcount[s] < 4) msrc[s][mcount[s]++] = ev.src_trace; mskip[i] = 1; }
                else { s = (int32_t)i; msrc[i][0] = ev.src_trace; mcount[i] = 1; } }
        }
        for (size_t i = 0; i < cur.size(); ++i) {
            const Event& ev = cur[i];
            if (p.msg_merge && mskip[i]) continue;
            if (epoch[ev.unit] == cur_epoch && ev.hop < 2) { ++st.dedup_skips; continue; }   // epoch dedup (runbook §6; iter_hops' hops re-run their body)
            epoch[ev.unit] = cur_epoch;
            if (p.msg_merge) { cur_msrc = msrc[i].data(); cur_mn = mcount[i]; }
            run_body(ev, novel, sign, UINT32_MAX);
            cur_msrc = nullptr; cur_mn = 0;
        }
        std::swap(cur, nxt);
    }
    if (p.route_return) route_consequence();
}
// Receivers follow their senders in trace order, so walking the traces last to first finalises every receiver (including
// its own receivers' consequences) before its sender integrates it.
void Organism::route_consequence() {
    if (!ntrace) return;
    rc_acc.assign((size_t)ntrace * Dims::R, 0); rc_n.assign(ntrace, 0);
    auto run_at = [&](const uint8_t* W, const Trace& tr, int li, const int8_t* x, int8_t* y) { LayerDesc d = L[li]; d.shift = tr.sh[li]; kern.run(W, d, x, y); };
    for (uint32_t t = ntrace; t-- > 0;) {
        Trace& tr = traces[t]; const uint32_t u = tr.unit; const uint8_t* W = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
        const bool exact = unit_lock[u] != UINT32_MAX && tree.locks[unit_lock[u]].mask == ~0ULL;
        if (rc_n[t] && !exact && (tr.reached & R_L5) && !diag_no_consequence) {   // integrate the receivers' consequences
            int8_t ret[Dims::R]; for (int i = 0; i < Dims::R; ++i) { const int32_t v = rc_acc[(size_t)t * Dims::R + i]; ret[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v); }
            std::memcpy(tr.x2 + Dims::A, ret, Dims::R2); run_at(W, tr, LI_2, tr.x2, tr.l2);
            std::memcpy(tr.x5, tr.l2, Dims::B); std::memcpy(tr.x5 + Dims::B, tr.l3, Dims::C); run_at(W, tr, LI_5, tr.x5, tr.l5);
            std::memcpy(tr.x6, tr.l5, Dims::E); std::memcpy(tr.x6 + Dims::E, ret + Dims::R2, Dims::R6);
            int8_t l6n[Dims::F]; run_at(W, tr, LI_6, tr.x6, l6n);
            if (tr.emitted) for (int j = 0; j < Dims::F; ++j) ws[j] += (int32_t)l6n[j] - (int32_t)tr.l6[j];
            std::memcpy(tr.l6, l6n, Dims::F);
            st.macs += Dims::P2 + Dims::P5 + Dims::P6;
        }
        if (tr.src >= 0 && (uint32_t)tr.src < t && (tr.reached & R_L6)) {   // a receiver: its consequence returns through its L4
            tr.sh[LI_4] = L[LI_4].shift; run_at(W, tr, LI_4, tr.l6, tr.l4); tr.reached |= R_L4; st.macs += Dims::P4; ++st.l4_runs;
            for (int q = 0; q < std::max<int>(1, tr.nsrc); ++q) { const int16_t s = tr.nsrc ? tr.srcs[q] : tr.src; if (s < 0 || (uint32_t)s >= t) continue;
                for (int i = 0; i < Dims::R; ++i) rc_acc[(size_t)s * Dims::R + i] += tr.l4[i];
                ++rc_n[s]; }
        }
    }
}

void Organism::readout(const int32_t* w, float* out_logits, float* out_probs) const {
    const uint32_t V = p.vocab;
    if (V <= (uint32_t)Dims::F) {
        // designated L6 output cells; exact expf keeps this path bit-comparable with the NumPy reference
        for (uint32_t v = 0; v < V; ++v) out_logits[v] = (float)w[v] * (1.0f / 32.0f);
        float mx = out_logits[0]; for (uint32_t v = 1; v < V; ++v) mx = std::max(mx, out_logits[v]);
        float z = 0.f; for (uint32_t v = 0; v < V; ++v) { out_probs[v] = std::exp(out_logits[v] - mx); z += out_probs[v]; }
        for (uint32_t v = 0; v < V; ++v) out_probs[v] /= z;
        return;
    }
    alignas(32) float a[Dims::F]; for (int k = 0; k < Dims::F; ++k) a[k] = (float)w[k] * (1.0f / 128.0f);
    if (!tst_.q.empty()) for (int k = 0; k < Dims::F; ++k) a[k] += tst_.a[k];   // tiles: the pyramid's reads, mixed into the readout's input
#if defined(__AVX2__) && defined(__FMA__)
    {   // transposed weights: 32 logits per pass, four independent accumulators so the FMA chains overlap;
        // each cell k contributes one broadcast x 128 contiguous bytes of its transposed row
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
    if (!wsp.empty()) { const size_t MF = wsm.size(); float b[64 * Dims::F]; for (size_t i = 0; i < MF && i < 64 * Dims::F; ++i) b[i] = (float)wsm[i] * (1.0f / 128.0f);   // ws_split
        for (uint32_t v2 = 0; v2 < V; ++v2) out_logits[v2] += dot_n(b, &wsp[(size_t)v2 * MF], MF); }
    if (!wtl.empty()) for (uint32_t v2 = 0; v2 < V; ++v2) out_logits[v2] += dot_n(tst_.a, &wtl[(size_t)v2 * Dims::F], Dims::F);   // tile_split
    if (!wsp5.empty()) { const size_t MF = wsm5.size(); float b5[8 * Dims::E]; for (size_t i = 0; i < MF; ++i) b5[i] = (float)wsm5[i] * (1.0f / 128.0f);   // ws_split5
        for (uint32_t v2 = 0; v2 < V; ++v2) out_logits[v2] += dot_n(b5, &wsp5[(size_t)v2 * MF], MF); }
    if (!rh_W2.empty()) { const size_t H = p.ro_hidden, D = rh_W1.size() / H; float u[Dims::F * 9], h[1024];   // ro_hidden
        ro_hidden_input(a, u); for (size_t k = 0; k < H; ++k) h[k] = std::max(0.f, rh_b1[k] + dot_n(u, &rh_W1[k * D], D));
        for (uint32_t v2 = 0; v2 < V; ++v2) out_logits[v2] += dot_n(h, &rh_W2[(size_t)v2 * H], H); }
    if (nlv) for (uint32_t v2 = 0; v2 < V; ++v2) out_logits[v2] += lvl_logits[v2];
    if (p.reg_readout && p.ws_regs && !wreg.empty()) for (uint32_t v2 = 0; v2 < V; ++v2) { const float* r = &wreg[(size_t)v2 * 4]; out_logits[v2] += (r[0] * reg_read(0) + r[1] * reg_read(1) + r[2] * reg_read(2) + r[3] * reg_read(3)) * 0.01f; }
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

// ---------------------------------------------------------------------------------------------
void Organism::process_event(uint8_t sym, uint8_t target) {
    hist_target_ = target; goal_track(sym, target);
    uint64_t key;
    { StageTimer T(&st, ST_INPUT, p.stage_timers); key = make_key(); make_input(key, sym); }
    if (!tst_.q.empty()) tiles_read(tst_);   // tiles do not change within an event
    uint32_t cand[16]; uint32_t ncand; uint32_t depth; bool novel = false;
    { StageTimer T(&st, ST_ADDRESS, p.stage_timers);
      if (!fm_base.empty()) {   // fixed motifs: the bodies for this byte are a fixed function of the recent bytes
          ncand = 0; depth = 0; cur_flags = 0; cur.clear(); accepted.clear(); executed.clear();
          for (size_t m = 0; m < fm_base.size(); ++m) { const uint32_t h = p.fm_h[m]; uint32_t off;
              if (h == 0) {   // iter_hops: the byte's processor body, hop after hop on the stream (hop field 2 + hop: no dedup, the cue)
                  const uint32_t hops = p.train ? (p.iter_rand ? rng.below(p.iter_hops + 1) : p.iter_hops) : p.iter_eval_hops();
                  for (uint32_t hh = 0; hh < hops; ++hh) cur.push_back(Event{fm_base[m] + (uint32_t)sym % p.fm_n[m], SRC_ROUTED, (uint8_t)(2 + hh), (int16_t)((int)cur.size() - 1)});
                  continue; }
              if (h <= 1) off = sym; else { uint64_t x = 1469598103934665603ULL; for (uint32_t i = 0; i < h && i < 16; ++i) x = (x ^ (i == 0 ? sym : hist[i])) * 1099511628211ULL;
                  if (p.fm_salt) x ^= 0x9E3779B97F4A7C15ULL * (uint64_t)(m + 1);   // fm_salt: this population's own hash
                  off = (uint32_t)(mix64(x) % p.fm_n[m]); }
              const bool wired = p.fm_wire && m > 0;   // trace m-1 is the previous motif's body (one trace per event, in order)
              cur.push_back(Event{fm_base[m] + off, wired ? SRC_ROUTED : SRC_SENSORY, (uint8_t)(wired ? 1 : 0), (int16_t)(wired ? (int)m - 1 : -1)}); }
          cur_novel = false;
      } else {
      int32_t leaf = tree.walk(key, depth); st.addr_decisions += depth;
      ncand = tree.match(leaf, key, cand, 16); st.candidate_locks += ncand;
      if (p.episodic && p.exact_only && ncand > 1) {   // optional fixed rule: most specific address wins (the agency kernel can decide this per condition instead)
          uint32_t m = 0; for (uint32_t i = 0; i < ncand; ++i) if (tree.locks[cand[i]].mask == ~0ULL) cand[m++] = cand[i];
          if (m) ncand = m;
      }
      cur_flags = 0;
      for (uint32_t i = 0; i < ncand; ++i) if (tree.locks[cand[i]].mask == ~0ULL) { ++st.exact_admissions; cur_flags |= FL_EXACT; break; }
      if (p.specific_wins && ncand > 1) {       // the most specific generic address wins (a nursery child over its parent)
          uint32_t m = 0; int best = -1; for (uint32_t i = 0; i < ncand; ++i) { const Lock& lk = tree.locks[cand[i]]; if (lk.mask == ~0ULL) continue; int pc = std::popcount(lk.mask); if (pc > best) best = pc; }
          for (uint32_t i = 0; i < ncand; ++i) { const Lock& lk = tree.locks[cand[i]]; if (lk.mask == ~0ULL || std::popcount(lk.mask) == best) cand[m++] = cand[i]; }
          ncand = m;
      }
      if (agent_on) {   // the kernel deliberates on this condition and its committed focus shapes what the bodies see
          bool nov = false; for (uint32_t i = 0; i < ncand; ++i) { const Lock& lk = tree.locks[cand[i]]; if (lk.hits < p.novel_hits || lk.err_ema > wrong_ema + p.surprise_thresh) nov = true; }
          agency_observe(key, ncand, nov, (cur_flags & FL_EXACT) != 0); agency_apply_focus();
          if (p.episodic && agent_exact_only && ncand > 1) { uint32_t m = 0; for (uint32_t i = 0; i < ncand; ++i) if (tree.locks[cand[i]].mask == ~0ULL) cand[m++] = cand[i]; if (m) ncand = m; }
      }
      if (ncand == 0) {
          ++st.no_candidate;
          if (p.structural) { uint32_t u = alloc_unit(); if (u != UINT32_MAX) { birth_unit_weights(u); uint32_t id = tree.new_lock(key, SENSORY_MASK, u, block_idx); unit_lock[u] = id; cand[ncand++] = id; ++st.lock_births; if (p.market) market_new_lock(id, false); } }
      }
      cur.clear(); accepted.clear(); executed.clear();
      for (uint32_t i = 0; i < ncand; ++i) {
          Lock& lk = tree.locks[cand[i]];
          // novelty (unfamiliar lock) and surprise (lock whose predictions keep failing) both buy computation
          if (lk.hits < p.novel_hits || lk.err_ema > wrong_ema + p.surprise_thresh) novel = true;
          ++lk.hits; lk.last_hit_block = block_idx;
          cur.push_back(Event{lk.unit, SRC_SENSORY, 0, -1});
      }
      if (p.shared_always && shared_base != UINT32_MAX)   // DeepSeekMoE shared experts: always active, emit into the workspace
          for (uint32_t i = 0; i < p.shared_always && shared_base + i < p.units; ++i) cur.push_back(Event{shared_base + i, SRC_SENSORY, 0, -1});
      if (p.persist) for (uint32_t u2 : persist_next) cur.push_back(Event{u2, SRC_PERSIST, 0, -1});   // still active from the last byte
      if (novel) ++st.novel_events; cur_novel = novel; } }

    const bool perturb_mode = p.train && p.hidden == HiddenLearning::Perturb;
    run_pass(key, novel, perturb_mode ? +1 : 0, nullptr);
    if (nlv) {   // the composed units, fed back as a whole: each level through its own readout weights
        std::fill(lvl_logits.begin(), lvl_logits.end(), 0.f);
        for (int l = 1; l <= nlv; ++l) { if (silenced_level == l) continue; float a[Dims::F]; bool any = false; for (int k = 0; k < Dims::F; ++k) { a[k] = lv[l].ws[k] * (1.f / 128.f); any |= a[k] != 0.f; } if (!any) continue;
            const float* W = &lvl_w[(size_t)(l - 1) * p.vocab * Dims::F]; for (uint32_t v = 0; v < p.vocab; ++v) { const float* r = W + (size_t)v * Dims::F; float s2 = 0.f; for (int k = 0; k < Dims::F; ++k) s2 += r[k] * a[k]; lvl_logits[v] += s2; } } }
    st.queue_sizes.push_back((uint32_t)executed.size());

    float lossA = 0.f, lossB = 0.f; int32_t wsA[Dims::F];
    std::vector<Trace> trA; uint32_t nA = 0; std::vector<Event> sched;
    if (perturb_mode) {
        std::memcpy(wsA, ws, sizeof ws); readout(wsA, logits.data(), probs.data()); lossA = -std::log(std::max(probs[target], 1e-9f));
        nA = ntrace; trA.assign(traces.begin(), traces.begin() + nA);
        sched.reserve(nA); for (uint32_t i = 0; i < nA; ++i) sched.push_back(Event{trA[i].unit, trA[i].source, trA[i].hop, -1});
        // routed events need their sender's trace index: senders precede receivers in schedule order
        for (uint32_t i = 0; i < nA; ++i) if (trA[i].source == SRC_ROUTED) for (auto& am : accepted) if (am.receiver == trA[i].unit) { for (uint32_t j = 0; j < i; ++j) if (trA[j].unit == am.sender) { sched[i].src_trace = (int16_t)j; break; } break; }
        std::vector<uint32_t> execA = executed; executed.clear();
        run_pass(key, novel, -1, &sched);
        readout(ws, logits.data(), probs.data()); lossB = -std::log(std::max(probs[target], 1e-9f));
        if (p.perturb_clean) { executed.clear(); run_pass(key, novel, 0, &sched); }   // clean pass: the committed prediction
        else for (int j = 0; j < Dims::F; ++j) ws[j] = (wsA[j] + ws[j]) / 2;        // else ~unperturbed by averaging
        executed = execA;
    }
    if (p.compete > 0.f) {   // cortical motif 5: the byte's bodies compete for the workspace (soft winner-take-all by activity)
        comp_w.assign(ntrace, 1.f); float a[64]; uint32_t idx[64]; uint32_t ne = 0; float mx = -1e30f;
        for (uint32_t t = 0; t < ntrace && ne < 64; ++t) { const Trace& tr = traces[t]; if (!tr.emitted || !(tr.reached & R_L6)) continue;
            float s = 0.f; for (int j = 0; j < Dims::F; ++j) s += std::fabs((float)tr.l6[j]); a[ne] = p.compete * s / (Dims::F * 127.f); mx = std::max(mx, a[ne]); idx[ne++] = t; }
        if (ne > 1) { float Z = 0.f; for (uint32_t k = 0; k < ne; ++k) { a[k] = std::exp(a[k] - mx); Z += a[k]; }
            std::memset(ws, 0, sizeof ws); if (!wsm.empty()) std::fill(wsm.begin(), wsm.end(), 0);
            for (uint32_t k = 0; k < ne; ++k) { const uint32_t t = idx[k]; const float w = (float)ne * a[k] / Z; comp_w[t] = w; const Trace& tr = traces[t];
                const bool ex = unit_lock[tr.unit] != UINT32_MAX && tree.locks[unit_lock[tr.unit]].mask == ~0ULL; const float gain = ex ? (float)p.exact_gain : 1.f;
                int32_t* wm = wsm.empty() ? nullptr : &wsm[(size_t)motif_of(tr.unit) * Dims::F];
                for (int j = 0; j < Dims::F; ++j) { const int32_t e = (int32_t)std::lround(w * gain * (float)tr.l6[j]); ws[j] += e; if (wm) wm[j] += e; } } } }
    uint32_t pred; float loss; bool correct, consequential;
    { StageTimer T(&st, ST_READOUT, p.stage_timers);
      if (!tst_.q.empty()) tiles_attend(tst_, ws, hist);   // tiles: the cued reads need this event's workspace
      readout(ws, logits.data(), probs.data());
      if (seq_on) {   // compose the bodies' distribution with the sequence memory's continuations
          body_probs = probs; ++seq_events;
          seq.topic_active = free_run;
          int L = seq.predict(seq_out.data(), p.seq_kappa, p.seq_use_gain);
          if (seq.chain_forcing) seq.bubble_left = 0;   // a committed answer (chain or learned attention) overrides replaying a remembered continuation
          bool bubbling = false; body_resp = 1.f;
          if (seq.bubble_left > 0) {   // committed to a remembered sequence: follow NXT while the bodies still support it
              uint8_t nxt = seq.at(seq.committed_pos); --seq.bubble_left; ++seq.committed_pos;
              if (body_probs[nxt] >= p.seq_bubble_support) { bubbling = true; ++seq_bubble_events; ++seq.bubble_bytes;
                  for (int v = 0; v < 256; ++v) probs[v] = 0.02f * body_probs[v]; probs[nxt] += 0.98f; }
              else seq.bubble_left = 0;
          }
          if (!bubbling && L) {
              ++seq_matched; seq_conf_sum += seq.confidence;
              float tot = 0.f; for (int v = 0; v < 256; ++v) tot += seq_out[v];
              float a = seq.chain_forcing ? 1.f : p.seq_alpha * seq.confidence;   // relevance-weighted composition (geometric blend); a seeded thought is the memory's alone
              // seq_gate: the blend weight is LEARNED per byte, a = sigmoid(w . f), from the memory's confidence, match length and
              // top probability, the bodies' top probability, and the bodies' probability for the memory's pick
              float gf[SEQ_GATE_F]; const bool gated = p.seq_gate && !seq.chain_forcing;
              if (gated) {
                  int sv = 0; for (int v = 1; v < 256; ++v) if (seq_out[v] > seq_out[sv]) sv = v;
                  float bt = 0.f; for (int v = 0; v < 256; ++v) bt = std::max(bt, body_probs[v]);
                  gf[0] = 1.f; gf[1] = seq.confidence; gf[2] = std::min(L, 32) / 32.f; gf[3] = seq_out[sv] / std::max(tot, 1e-9f); gf[4] = bt; gf[5] = body_probs[sv];
                  float z0 = 0.f; for (int k = 0; k < SEQ_GATE_F; ++k) z0 += seq_gw[k] * gf[k];
                  a = 1.f / (1.f + std::exp(-z0));
              }
              float z = 0.f;
#if defined(__AVX2__) && defined(__FMA__)
              {   // one vector pass: exp((1-a) log body + a log ps) — replaces 512 scalar pow() per byte
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
              if (p.seq_linear) {   // linear mixture: each source keeps its own sharpness; the gate says whom to trust
                  const float eps = 0.01f * tot / 256.f, inv = 1.f / (tot * 1.01f);
                  for (int v = 0; v < 256; ++v) probs[v] = a * (seq_out[v] + eps) * inv + (1.f - a) * body_probs[v];
                  const float pst = (seq_out[target] + eps) * inv, pt = std::max(probs[target], 1e-9f);
                  body_resp = (1.f - a) * body_probs[target] / pt;
                  if (gated && p.train) { const float g = -(pst - body_probs[target]) / pt * a * (1.f - a);
                      if (p.ro_adam) { if (sg_g.empty()) { sg_g.assign(SEQ_GATE_F, 0.f); sg_m = sg_g; sg_v = sg_g; } for (int k = 0; k < SEQ_GATE_F; ++k) sg_g[k] += g * gf[k]; }
                      else for (int k = 0; k < SEQ_GATE_F; ++k) seq_gw[k] -= p.seq_gate_lr * g * gf[k]; }
              } else if (gated && p.train) {   // exact gradient of -log p(target) w.r.t. the gate logit: -(d_t - E_p[d]) a (1 - a), d = log ps - log body
                  const float eps = 0.01f * tot / 256.f, inv = 1.f / (tot * 1.01f); float E = 0.f, dt = 0.f;
                  for (int v = 0; v < 256; ++v) { const float d = std::log((seq_out[v] + eps) * inv) - std::log(std::max(body_probs[v], 1e-9f)); E += probs[v] * d; if (v == target) dt = d; }
                  const float g = -(dt - E) * a * (1.f - a);
                  if (p.ro_adam) { if (sg_g.empty()) { sg_g.assign(SEQ_GATE_F, 0.f); sg_m = sg_g; sg_v = sg_g; } for (int k = 0; k < SEQ_GATE_F; ++k) sg_g[k] += g * gf[k]; }
                  else for (int k = 0; k < SEQ_GATE_F; ++k) seq_gw[k] -= p.seq_gate_lr * g * gf[k];
              }
              if (L >= p.seq_bubble_min) { seq.bubble_left = p.seq_bubble_len; ++seq.bubbles; }   // commit to the sequence as a unit
          }
      }
      pred = 0; for (uint32_t v = 1; v < p.vocab; ++v) if (probs[v] > probs[pred]) pred = v;
      loss = -std::log(std::max(probs[target], 1e-9f));
      if (on_event) on_event(sym, target, ws, probs.data());
      if (ans_state == 2 || (ans_state == 1 && goal_w > 1.f)) { ans_loss += loss; ++ans_n; }
      st.loss_sum += loss; ++st.events; ++st.bytes;
      correct = pred == target; st.correct += correct;
      if (!loss_ema_init) { loss_ema = loss; loss_ema_init = true; } else loss_ema += 0.01f * (loss - loss_ema);
      consequential = loss <= loss_ema;                                     // predictive consequence, committed before outcome
      // A routed (neighbour-admitted) body is credited only if its own emission lowered this loss: read the
      // workspace without it. Predictive consequence preserves routes; noise does not (design doc §7).
      if (p.recirc_learn == 2 && p.train) for (uint32_t t = 0; t < ntrace; ++t) { const Trace& tr = traces[t];   // the actual loss, filed under the arm this unit used
          if (tr.recirced != 2 && tr.recirced != 3) continue; const uint32_t uu = tr.unit;
          if (tr.recirced == 2) { arm_on[uu] += (arm_non[uu] < 64 ? 1.f / (arm_non[uu] + 1) : 1.f / 64) * (loss - arm_on[uu]); if (arm_non[uu] < 65535) ++arm_non[uu]; }
          else { arm_off[uu] += (arm_noff[uu] < 64 ? 1.f / (arm_noff[uu] + 1) : 1.f / 64) * (loss - arm_off[uu]); if (arm_noff[uu] < 65535) ++arm_noff[uu]; } }
      if (p.recirc_learn == 1 && p.train) for (uint32_t t = 0; t < ntrace; ++t) { Trace& tr = traces[t];   // what did the extra passes buy?
          if (!tr.recirced || !tr.emitted) continue;
          int32_t wsx[Dims::F]; for (int j = 0; j < Dims::F; ++j) wsx[j] = ws[j] - tr.l6[j] + tr.l6_pre[j];   // the workspace with the pre-recirculation emission
          readout(wsx, cf_logits.data(), cf_probs.data());
          const float loss_pre = -std::log(std::max(cf_probs[target], 1e-9f));
          const float gain_bits = (loss_pre - loss) / 0.6931471805599453f;
          recirc_utility[tr.unit] = p.route_utility_decay * recirc_utility[tr.unit] + (1.f - p.route_utility_decay) * gain_bits; }
      for (uint32_t t = 0; t < ntrace; ++t) { Trace& tr = traces[t];
          const bool is_exact = unit_lock[tr.unit] != UINT32_MAX && tree.locks[unit_lock[tr.unit]].mask == ~0ULL;
          if ((tr.source != SRC_ROUTED && !is_exact) || !(tr.reached & R_L6)) { tr.consequential = consequential; continue; }
          const bool silent = tr.consequential == 2;
          int32_t wsx[Dims::F]; for (int j = 0; j < Dims::F; ++j) wsx[j] = silent ? ws[j] + tr.l6[j] : ws[j] - tr.l6[j];   // the workspace with it toggled
          readout(wsx, cf_logits.data(), cf_probs.data());
          const float loss_other = -std::log(std::max(cf_probs[target], 1e-9f));
          const float gain_bits = (silent ? (loss - loss_other) : (loss_other - loss)) / 0.6931471805599453f;   // bits it gained (or would have)
          route_utility[tr.unit] = p.route_utility_decay * route_utility[tr.unit] + (1.f - p.route_utility_decay) * gain_bits;
          tr.consequential = gain_bits > 0.f ? 1 : 0; } }
    {
      if (trace_fp) {
          std::fprintf(trace_fp, "{\"t\":%llu,\"key\":%llu,\"cand\":%u,\"novel\":%d,\"exec\":[", (unsigned long long)event_idx, (unsigned long long)key, ncand, (int)novel);
          for (size_t i = 0; i < executed.size(); ++i) std::fprintf(trace_fp, "%s%u", i ? "," : "", executed[i]);
          std::fprintf(trace_fp, "],\"reached\":[");
          for (uint32_t i = 0; i < ntrace; ++i) std::fprintf(trace_fp, "%s%u", i ? "," : "", traces[i].reached);
          std::fprintf(trace_fp, "],\"pred\":%u,\"target\":%u,\"loss\":%.5f}\n", pred, target, loss);
      }

      if (p.record_preds) preds.push_back((uint8_t)pred);
      if (agent_on && p.train) agency_consequence(probs[target], correct, loss);   // the verified outcome of this act
      if (p.train) {
          StageTimer T2(&st, ST_LEARN, p.stage_timers);
          if (perturb_mode) {
              int32_t dq = (int32_t)std::lround((lossA - lossB) * (float)p.perturb_gain); dq = std::max(-127, std::min(127, dq));
              learn_hidden(dq, trA, nA);
          }
          learn_l6(target);
          if (p.mtp) mtp_learn(target);   // before learn_backprop's temporal credit: it writes into earlier ring slots
          if (!tst_.q.empty()) {   // the reads' gradients; word tiles' credit into the ring; then the queries' gradient joins the bodies' own
              TileLearn tl; tl.gM = tile_Mg.data(); tl.gA = tile_Ag.empty() ? nullptr : tile_Ag.data(); tl.gF = fw_Wg.empty() ? nullptr : fw_Wg.data();
              tl.gR = ret_Wg.empty() ? nullptr : ret_Wg.data();
              tl.goal_w = goal_w; tl.event = event_idx;
              tl.inject = [this](uint64_t age, const float* g, float w, const float* v) { const uint32_t Wn = p.tc_window; if (!Wn || age < 1 || age > tc_fill || age >= Wn) return;
                  TcEvent& e = tc_ring[(tc_head + Wn - age) % Wn]; for (int j = 0; j < Dims::F; ++j) if (v[j] > -4.f && v[j] < 4.f) e.tg[j] += g[j] * w; };
              if (!wtl.empty()) { float dd[Dims::F]; for (int j = 0; j < Dims::F; ++j) dd[j] = dl6[j] + dl6t[j]; tiles_learn(tst_, dd, tl); }   // tile_split
              else tiles_learn(tst_, dl6, tl);
              for (int j = 0; j < Dims::F; ++j) dl6[j] += tst_.dq[j];
              if (ro_pending) { ro_pending = false; ro_adam_step(); } }   // the readout's step, after the reads' gradients (as the trainer)
          if (p.hidden == HiddenLearning::Backprop) learn_backprop();
          if (p.hidden == HiddenLearning::Modulated) {
              int32_t M = (int32_t)std::lround((loss_ema - loss) / 0.6931471805599453f * p.mod_gain); M = std::max(-127, std::min(127, M));
              learn_modulated(M);
          }
          metabolic(consequential, novel);
          assoc_update(novel, consequential);
          if (p.episodic && p.structural && !correct && loss > loss_ema + (goal_w > 1.f ? 0.f : p.episodic_margin) && ncand && !(cur_flags & FL_EXACT)) {
              if (recur_ready(key)) episodic_birth(key, cand[0]); else ++ep_totals.deferred;
          }
      }
      if (p.record_preds) flags.push_back(cur_flags);
      // lock-local structural evidence (design doc §6): belongs to the ambiguous lock, never the input family
      StageTimer T3(&st, ST_EVIDENCE, p.stage_timers);
      const uint32_t wrong = correct ? 0u : 65535u;
      wrong_ema += ((int32_t)wrong - (int32_t)wrong_ema) / 256;
      // utility in bits: what this prediction gained over the stream's own prior (ALM route utility)
      const float gain_bits = std::log2(std::max(probs[target], 1e-9f) / std::max(prior[target], 1e-9f));
      if (p.train) for (uint32_t v = 0; v < p.vocab; ++v) prior[v] += 0.0005f * ((v == target ? 1.f : 0.f) - prior[v]);
      if (p.market) {   // parsimony: the event's gain is shared among the bodies that admitted it
          if (mk_util.size() < tree.locks.size()) mk_util.resize(tree.locks.size(), 0.f);
          const float share = gain_bits * goal_w / (float)std::max<uint32_t>(1, ncand);
          for (uint32_t i = 0; i < ncand; ++i) { float& u = mk_util[cand[i]]; u = p.utility_decay * u + (1.f - p.utility_decay) * share; }
      }
      for (uint32_t i = 0; i < ncand; ++i) {
          Lock& lk = tree.locks[cand[i]];
          lk.err_ema += ((int32_t)wrong - (int32_t)lk.err_ema) / 32;
          lk.utility = p.utility_decay * lk.utility + (1.f - p.utility_decay) * gain_bits * goal_w;
          if (p.structural && p.train) {
              if (lk.evidence < 0 && lk.err_ema > p.conflict_thresh) lk.evidence = evid.acquire(cand[i], lk.mask);
              if (lk.evidence >= 0) evid.record(lk.evidence, key, target);
          }
      }
      if (p.episodic_promote && p.structural && p.train) promote_snapshots(key, target, cand, ncand); }

    if (p.ws_decay) for (int j = 0; j < Dims::F; ++j) ws_mem[j] = ((ws_mem[j] * ws_decay_of(j)) >> 8) + ws[j];   // leaky persistent workspace (ws_bands: per-band decay)
    if (p.persist) persist_select();
    if (p.ws_regs == 2) {   // ONE shared write head reads the workspace (the bodies' collective output): reg = (1 - g) reg + g v
        if (!head_init) { Rng hr(mix64(p.seed ^ 0x4EADULL)); for (int r = 0; r < 4; ++r) { hb_g[r] = p.reg_bias; for (int j = 0; j < Dims::F; ++j) { hw_g[r][j] = (float)(hr.uniform() * 2 - 1) * 0.1f; hw_v[r][j] = (float)(hr.uniform() * 2 - 1) * 0.1f; } } head_init = true; }
        float hs = 1.f / 128.f;
        if (p.head_norm) { uint32_t ne = 0; for (uint32_t t = 0; t < ntrace; ++t) ne += traces[t].emitted && (traces[t].reached & R_L6); hs /= (float)std::max(1u, ne); }
        float a[Dims::F]; for (int j = 0; j < Dims::F; ++j) a[j] = std::max(-4.f, std::min(4.f, (float)ws[j] * hs));
        TcEvent* e = (p.tc_window && tc_fill) ? &tc_ring[(tc_head + p.tc_window - 1) % p.tc_window] : nullptr;
        const bool clk = !p.reg_clock || !std::isalnum(hist[0]);   // reg_clock=1: writes only at word boundaries (gate shut inside words)
        for (int r = 0; r < 4; ++r) { float zg = hb_g[r], zv = 0.f; for (int j = 0; j < Dims::F; ++j) { zg += hw_g[r][j] * a[j]; zv += hw_v[r][j] * a[j]; }
            const float g = clk ? 1.f / (1.f + std::exp(-zg)) : 0.f, v = 100.f * std::tanh(zv);
            ws_reg_prev[r] = ws_reg[r]; ws_regG[r] = g; ws_reg[r] = (1.f - g) * ws_reg[r] + g * v;
            if (p.reg_center && p.train) reg_mean[r] += (ws_reg[r] - reg_mean[r]) / (float)p.reg_center;
            if (p.reg_revive && p.train) {   // dead register: pinned at one sign -> rebirth of its head row
                const int8_t sg = ws_reg[r] > 99.f ? 1 : ws_reg[r] < -99.f ? -1 : 0;
                if (sg != 0 && sg == reg_sign[r]) ++reg_pinned[r]; else { reg_pinned[r] = 0; reg_sign[r] = sg; }
                if (reg_pinned[r] >= p.reg_revive) {
                    Rng rr(mix64(p.seed ^ event_idx ^ (0xBEEF00ULL + (uint64_t)r))); hb_g[r] = p.reg_bias;
                    for (int j = 0; j < Dims::F; ++j) { hw_g[r][j] = (float)(rr.uniform() * 2 - 1) * 0.1f; hw_v[r][j] = (float)(rr.uniform() * 2 - 1) * 0.1f;
                        if (!hd_m.empty()) { hd_m[(size_t)r * Dims::F + j] = hd_v[(size_t)r * Dims::F + j] = 0.f; hd_m[(size_t)(4 + r) * Dims::F + j] = hd_v[(size_t)(4 + r) * Dims::F + j] = 0.f; } }
                    if (!hd_m.empty()) hd_m[8 * Dims::F + r] = hd_v[8 * Dims::F + r] = 0.f;
                    ws_reg[r] = 0.f; reg_pinned[r] = 0; reg_sign[r] = 0; ++reg_revivals;
                    std::fprintf(stderr, "reg_revive: register %d reborn at event %llu (total %llu)\n", r, (unsigned long long)event_idx, (unsigned long long)reg_revivals);
                } }
            if (e) { e->reg_prev[r] = ws_reg_prev[r]; e->G[r] = g; e->hg[r] = g; e->hv[r] = v; e->hu[r] = zv; } }
        if (e) { std::memcpy(e->ha, a, sizeof a); e->hs = hs; }
        if (e && diag_reg_has_target) { e->tg_reg[0] += (ws_reg[0] - diag_reg_target) / 100.f; diag_reg_has_target = false; }
    } else if (p.ws_regs) {   // gated registers: each emitting body proposes (value, gate) per register; a strong write overwrites
        float gs[4] = {0, 0, 0, 0}, gv[4] = {0, 0, 0, 0};
        for (uint32_t t = 0; t < ntrace; ++t) { const Trace& tr = traces[t]; if (!tr.emitted || !(tr.reached & R_L6)) continue;
            for (int r = 0; r < 4; ++r) { const float g = reg_gate(tr.l6[28 + r]); gs[r] += g; gv[r] += g * (float)tr.l6[24 + r]; } }
        for (int r = 0; r < 4; ++r) { ws_reg_prev[r] = ws_reg[r]; const float G = std::min(1.f, gs[r]); ws_regG[r] = G;
            ws_reg[r] = gs[r] > 1.f ? gv[r] / gs[r] : (1.f - G) * ws_reg[r] + gv[r]; }
        if (p.tc_window && tc_fill) { TcEvent& e = tc_ring[(tc_head + p.tc_window - 1) % p.tc_window]; for (int r = 0; r < 4; ++r) { e.reg_prev[r] = ws_reg_prev[r]; e.G[r] = ws_regG[r]; }
            if (diag_reg_has_target) { e.tg_reg[0] += (ws_reg[0] - diag_reg_target) / 100.f; diag_reg_has_target = false; } }
    }
    if (p.lvl_feed && nlv && p.ws_decay) for (int l = 1; l <= nlv; ++l) for (int j = 0; j < Dims::F; ++j) ws_mem[j] += (lv[l].ws[j] * (256 - p.ws_decay)) >> 8;   // the levels as context for the byte bodies
    if (seq_on && !free_run && !replay) seq.capture(target);   // capture everything: the outcome becomes a step in the sequence memory
    if (!tst_.q.empty()) tiles_close(tst_, ws, hist, event_idx);   // hist[0] is still the byte just read
    for (int i = 15; i > 0; --i) hist[i] = hist[i - 1];
    hist[0] = target;   // the outcome becomes the next event's current symbol
    ++event_idx;
    if (nlv) composer_step(sym, probs[pred]);
}

// ---------------------------------------------------------------------------------------------
// learning: strictly local, only traversed synapses (runbook §8)
// Scaled delta = trunc(v / 2^shift) toward zero. An arithmetic shift would floor, turning every small
// positive product into -1 and every small negative one into 0: a systematic one-way drift that
// silently wrecked the hidden-layer learner.
static inline int32_t scale_trunc(int32_t v, int32_t shift) { return v >= 0 ? (v >> shift) : -((-v) >> shift); }
static inline int16_t clamp16(int32_t v) { return (int16_t)(v > 32767 ? 32767 : v < -32767 ? -32767 : v); }
void Organism::apply_acc16(uint32_t u, uint32_t nib, int32_t delta) {
    if (freeze_weights) return;   // frozen-bodies control: no synapse of any body changes
    if (delta == 0) return;
    int16_t& a = acc16[(size_t)u * Dims::UNIT_NIBBLES + nib];
    int32_t v = (int32_t)a + delta; ++st.acc_updates;
    uint8_t* w = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
    if (v >= p.half_quantum) { int cw = nib_get(w, nib); if (cw < 7) { nib_set(w, nib, cw + 1); v -= p.quantum; ++st.level_changes; } else v = p.half_quantum - 1; }
    else if (v <= -p.half_quantum) { int cw = nib_get(w, nib); if (cw > -8) { nib_set(w, nib, cw - 1); v += p.quantum; ++st.level_changes; } else v = -(p.half_quantum - 1); }
    a = clamp16(v);
}
// 16-lane vector path for the INT16 residual: accumulate-only lanes in bulk, crossing lanes to apply_acc16.
void Organism::apply_acc_block16(uint32_t u, uint32_t nib0, const int32_t* delta, int n) {
    if (freeze_weights) return;   // frozen-bodies control: no synapse of any body changes
    int i = 0;
#if defined(__AVX2__)
    int16_t* a = &acc16[(size_t)u * Dims::UNIT_NIBBLES + nib0];
    const __m256i zero = _mm256_setzero_si256(), ones = _mm256_set1_epi16(-1);
    const __m256i hm1 = _mm256_set1_epi16((int16_t)(p.half_quantum - 1)), nhp1 = _mm256_set1_epi16((int16_t)(1 - p.half_quantum));
    for (; i + 16 <= n; i += 16) {
        __m256i d0 = _mm256_loadu_si256((const __m256i*)(delta + i)), d1 = _mm256_loadu_si256((const __m256i*)(delta + i + 8));
        __m256i d16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(d0, d1), 0xD8);
        __m256i old = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i v = _mm256_adds_epi16(old, d16);
        __m256i nz = _mm256_xor_si256(_mm256_cmpeq_epi16(d16, zero), ones);
        __m256i cross = _mm256_and_si256(nz, _mm256_or_si256(_mm256_cmpgt_epi16(v, hm1), _mm256_cmpgt_epi16(nhp1, v)));
        uint32_t cm = (uint32_t)_mm256_movemask_epi8(cross), nzm = (uint32_t)_mm256_movemask_epi8(nz);   // 2 bits per lane
        _mm256_storeu_si256((__m256i*)(a + i), _mm256_blendv_epi8(v, old, cross));
        st.acc_updates += (uint64_t)std::popcount(nzm & ~cm) / 2;
        cm &= 0x55555555u;
        while (cm) { int l = std::countr_zero(cm) / 2; cm &= cm - 1; apply_acc16(u, nib0 + i + l, delta[i + l]); }
    }
#endif
    for (; i < n; ++i) apply_acc16(u, nib0 + i, delta[i]);
}
void Organism::apply_acc(uint32_t u, uint32_t nib, int32_t delta) {
    if (freeze_weights) return;   // frozen-bodies control: no synapse of any body changes
    if (wide_acc) { apply_acc16(u, nib, delta); return; }
    if (delta == 0) return;
    int8_t& a = acc[(size_t)u * Dims::UNIT_NIBBLES + nib];
    int32_t v = (int32_t)a + delta; ++st.acc_updates;
    uint8_t* w = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
    if (v >= p.half_quantum) { int cw = nib_get(w, nib); if (cw < 7) { nib_set(w, nib, cw + 1); v -= p.quantum; ++st.level_changes; } else v = p.half_quantum - 1; }
    else if (v <= -p.half_quantum) { int cw = nib_get(w, nib); if (cw > -8) { nib_set(w, nib, cw - 1); v += p.quantum; ++st.level_changes; } else v = -(p.half_quantum - 1); }
    a = clamp8(v);
}
// Same rule as apply_acc over a contiguous block of synapses. Level crossings are rare, so the vector path
// handles the common "accumulate only" lanes 32 at a time and hands crossing lanes to apply_acc.
void Organism::apply_acc_block(uint32_t u, uint32_t nib0, const int32_t* delta, int n) {
    if (freeze_weights) return;
    if (wide_acc) { apply_acc_block16(u, nib0, delta, n); return; }
    int i = 0;
#if defined(__AVX2__)
    int8_t* a = &acc[(size_t)u * Dims::UNIT_NIBBLES + nib0];
    const uint8_t* wb = wbuf.p + (size_t)u * Dims::UNIT_BYTES + (nib0 >> 1);   // nib0 is even for every layer
    const __m256i zero = _mm256_setzero_si256(), ones = _mm256_set1_epi8(-1);
    const __m256i hm1 = _mm256_set1_epi8((int8_t)(p.half_quantum - 1)), nhp1 = _mm256_set1_epi8((int8_t)(1 - p.half_quantum));
    const __m256i six = _mm256_set1_epi8(6), m7 = _mm256_set1_epi8(-7);
    const __m128i m15 = _mm_set1_epi8(0x0F), m8 = _mm_set1_epi8(8);
    for (; i + 32 <= n; i += 32) {
        // pack int32 deltas to int8 (saturating); a delta that does not fit is a crossing and goes scalar anyway
        __m256i d0 = _mm256_loadu_si256((const __m256i*)(delta + i)), d1 = _mm256_loadu_si256((const __m256i*)(delta + i + 8));
        __m256i d2 = _mm256_loadu_si256((const __m256i*)(delta + i + 16)), d3 = _mm256_loadu_si256((const __m256i*)(delta + i + 24));
        __m256i d8 = _mm256_packs_epi16(_mm256_packs_epi32(d0, d1), _mm256_packs_epi32(d2, d3));
        d8 = _mm256_permutevar8x32_epi32(d8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        __m256i old = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i v = _mm256_adds_epi8(old, d8);
        __m256i nz = _mm256_xor_si256(_mm256_cmpeq_epi8(d8, zero), ones);
        __m256i up = _mm256_and_si256(nz, _mm256_cmpgt_epi8(v, hm1)), dn = _mm256_and_si256(nz, _mm256_cmpgt_epi8(nhp1, v));
        // current INT4 weights of these 32 synapses, in nibble order (low nibble first)
        __m128i b = _mm_loadu_si128((const __m128i*)(wb + (i >> 1)));
        __m128i lo = _mm_sub_epi8(_mm_xor_si128(_mm_and_si128(b, m15), m8), m8);
        __m128i hi = _mm_sub_epi8(_mm_xor_si128(_mm_and_si128(_mm_srli_epi16(b, 4), m15), m8), m8);
        __m256i w = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
        __m256i sat_up = _mm256_and_si256(up, _mm256_cmpgt_epi8(w, six));      // w == 7  : cannot rise, clamp residual
        __m256i sat_dn = _mm256_and_si256(dn, _mm256_cmpgt_epi8(m7, w));       // w == -8 : cannot fall, clamp residual
        __m256i change = _mm256_andnot_si256(_mm256_or_si256(sat_up, sat_dn), _mm256_or_si256(up, dn)); // real level changes
        v = _mm256_blendv_epi8(v, hm1, sat_up);
        v = _mm256_blendv_epi8(v, nhp1, sat_dn);
        uint32_t chm = (uint32_t)_mm256_movemask_epi8(change), nzm = (uint32_t)_mm256_movemask_epi8(nz);
        _mm256_storeu_si256((__m256i*)(a + i), _mm256_blendv_epi8(v, old, change));
        st.acc_updates += (uint64_t)std::popcount(nzm & ~chm);
        while (chm) { int l = std::countr_zero(chm); chm &= chm - 1; apply_acc(u, nib0 + i + l, delta[i + l]); }
    }
#endif
    for (; i < n; ++i) apply_acc(u, nib0 + i, delta[i]);
}
void Organism::learn_l6(uint8_t target) {
    const uint32_t V = p.vocab;
    // body_err: the bodies and their readout learn from their OWN prediction error, not the error left after the blend with the
    // sequence memory (which trains them only as a small residual on top of it)
    const std::vector<float>& probs = (p.body_err && seq_on && body_probs.size() == this->probs.size()) ? body_probs : this->probs;
    const float resp = (p.body_err == 2 && p.seq_linear && seq_on) ? body_resp : 1.f;   // exact mixture gradient scale
    if (V <= (uint32_t)Dims::F) { for (int j = 0; j < Dims::F; ++j) dl6[j] = (uint32_t)j < V ? probs[j] - (j == target ? 1.f : 0.f) : 0.f; }
    else {
        // sparse error support: the softmax error is concentrated on the target and a few confident wrong symbols
        err_idx.clear(); err_val.clear();
        for (uint32_t v = 0; v < V; ++v) { float e = (probs[v] - (v == target ? 1.f : 0.f)) * resp; if (e > p.err_sparse || e < -p.err_sparse) { err_idx.push_back(v); err_val.push_back(e); } }
        for (int k = 0; k < Dims::F; ++k) dl6[k] = 0.f;
        for (size_t s = 0; s < err_idx.size(); ++s) { const float* row = &wout[(size_t)err_idx[s] * Dims::F]; for (int k = 0; k < Dims::F; ++k) dl6[k] += err_val[s] * row[k]; }
        if (!wsp.empty()) { const size_t MF = wsm.size(); std::fill(dl6m.begin(), dl6m.end(), 0.f);   // ws_split: each motif's own error, and its weights' gradient
            for (size_t s = 0; s < err_idx.size(); ++s) { const float* row = &wsp[(size_t)err_idx[s] * MF]; float* gr = &wsp_g[(size_t)err_idx[s] * MF]; const float g = err_val[s] * std::min(goal_w, 4.f);
                for (size_t i = 0; i < MF; ++i) { dl6m[i] += err_val[s] * row[i]; if (p.train) gr[i] += g * (float)wsm[i] * (1.0f / 128.0f); } } }
        if (!wtl.empty()) { std::fill(dl6t, dl6t + Dims::F, 0.f);   // tile_split: the tiles' own channel
            for (size_t s = 0; s < err_idx.size(); ++s) { const float* row = &wtl[(size_t)err_idx[s] * Dims::F]; float* gr = &wtl_g[(size_t)err_idx[s] * Dims::F]; const float g = err_val[s] * std::min(goal_w, 4.f);
                for (int k = 0; k < Dims::F; ++k) { dl6t[k] += err_val[s] * row[k]; if (p.train) gr[k] += g * tst_.a[k]; } } }
        if (!wsp5.empty()) { const size_t MF = wsm5.size(); std::fill(dl5m.begin(), dl5m.end(), 0.f);   // ws_split5
            for (size_t s = 0; s < err_idx.size(); ++s) { const float* row = &wsp5[(size_t)err_idx[s] * MF]; float* gr = &wsp5_g[(size_t)err_idx[s] * MF]; const float g = err_val[s] * std::min(goal_w, 4.f);
                for (size_t i = 0; i < MF; ++i) { dl5m[i] += err_val[s] * row[i]; if (p.train) gr[i] += g * (float)wsm5[i] * (1.0f / 128.0f); } } }
        if (!rh_W2.empty()) {   // ro_hidden: the hidden layer's gradients; its input's error joins the shared channel (dl6) and the motifs' (dl6m)
            const size_t H = p.ro_hidden, D = rh_W1.size() / H; float a[Dims::F], u[Dims::F * 9], h[1024], dh[1024] = {0};
            for (int k = 0; k < Dims::F; ++k) a[k] = (float)ws[k] * (1.0f / 128.0f) + (tst_.q.empty() ? 0.f : tst_.a[k]);
            ro_hidden_input(a, u); for (size_t k = 0; k < H; ++k) h[k] = std::max(0.f, rh_b1[k] + dot_n(u, &rh_W1[k * D], D));
            const float gm = std::min(goal_w, 4.f);
            for (size_t s = 0; s < err_idx.size(); ++s) { const float* row = &rh_W2[(size_t)err_idx[s] * H]; float* gr = &rh_W2g[(size_t)err_idx[s] * H];
                for (size_t k = 0; k < H; ++k) { dh[k] += err_val[s] * row[k]; if (p.train) gr[k] += err_val[s] * gm * h[k]; } }
            float du[Dims::F * 9] = {0};
            for (size_t k = 0; k < H; ++k) { if (h[k] <= 0.f || dh[k] == 0.f) continue; if (p.train) { rh_b1g[k] += dh[k] * gm; float* gw = &rh_W1g[k * D]; for (size_t i = 0; i < D; ++i) gw[i] += dh[k] * gm * u[i]; }
                const float* w1 = &rh_W1[k * D]; for (size_t i = 0; i < D; ++i) du[i] += dh[k] * w1[i]; }
            for (int k = 0; k < Dims::F; ++k) dl6[k] += du[k];
            for (size_t i = Dims::F; i < D; ++i) dl6m[i - Dims::F] += du[i]; }
        if (p.reg_readout && p.ws_regs) { if (wreg.empty()) { wreg.assign((size_t)V * 4, 0.f);
                if (p.reg_ro_init > 0.f) { Rng rr(mix64(p.seed ^ 0x5E6ULL)); for (auto& w : wreg) w = (float)(rr.uniform() * 2 - 1) * p.reg_ro_init; } }   // the register read by the readout: its error and its weights
            for (int r = 0; r < 4; ++r) greg_ro[r] = 0.f;
            for (size_t s = 0; s < err_idx.size(); ++s) { float* row = &wreg[(size_t)err_idx[s] * 4]; const float g = p.lr_out * err_val[s] * std::min(goal_w, 4.f);
                for (int r = 0; r < 4; ++r) { greg_ro[r] += err_val[s] * row[r] * 0.01f;
                    if (p.train && p.ro_adam) { if (rg_g.empty()) { rg_g.assign((size_t)V * 4, 0.f); rg_m = rg_g; rg_v = rg_g; } rg_g[(size_t)err_idx[s] * 4 + r] += err_val[s] * std::min(goal_w, 4.f) * reg_read(r) * 0.01f; }
                    else if (p.train) row[r] -= g * reg_read(r) * 0.01f; } } }
        if (!p.readout_freeze || event_idx < p.readout_freeze) {
            float a[Dims::F]; for (int k = 0; k < Dims::F; ++k) a[k] = (float)ws[k] * (1.0f / 128.0f);
            if (p.ro_adam) { if (ro_g.empty()) { ro_g.assign((size_t)V * Dims::F, 0.f); ro_m = ro_g; ro_v = ro_g; }
                for (size_t s = 0; s < err_idx.size(); ++s) { float* gr = &ro_g[(size_t)err_idx[s] * Dims::F]; const float g = err_val[s] * std::min(goal_w, 4.f); for (int k = 0; k < Dims::F; ++k) gr[k] += g * a[k]; } }
            else for (size_t s = 0; s < err_idx.size(); ++s) { float* row = &wout[(size_t)err_idx[s] * Dims::F]; float g = p.lr_out * err_val[s] * std::min(goal_w, 4.f); for (int k = 0; k < Dims::F; ++k) { row[k] -= g * a[k]; woutT[(size_t)k * V + err_idx[s]] = row[k]; } }
        }
    }
    for (int l = 1; l <= nlv; ++l) { if (V <= (uint32_t)Dims::F) break; float* W = &lvl_w[(size_t)(l - 1) * V * Dims::F]; float a[Dims::F]; bool any = false; for (int k = 0; k < Dims::F; ++k) { a[k] = lv[l].ws[k] * (1.f / 128.f); any |= a[k] != 0.f; }
        float dl[Dims::F] = {0}; for (size_t s = 0; s < err_idx.size(); ++s) { float* row = W + (size_t)err_idx[s] * Dims::F; for (int k = 0; k < Dims::F; ++k) dl[k] += err_val[s] * row[k]; if (any && p.train) { const float g = p.lr_out * err_val[s]; for (int k = 0; k < Dims::F; ++k) row[k] -= g * a[k]; } }
        if (!lv[l].tr.empty()) { for (int j = 0; j < Dims::F; ++j) lv[l].acc[j] += dl[j]; ++lv[l].acc_n; } }
    for (int j = 0; j < Dims::F; ++j) { float g = dl6[j] * p.err_gain * goal_w * 127.f; g = std::max(-127.f, std::min(127.f, g));
        if (p.stoch_round) { const float fl = std::floor(g); q6[j] = (int8_t)((int)fl + (sr_uniform() < g - fl ? 1 : 0)); }
        else q6[j] = (int8_t)(int)(g + (g >= 0.f ? 0.5f : -0.5f)); } // lround semantics
    if (!(p.adam && p.adam_l6 && p.hidden == HiddenLearning::Backprop && !ad_g.empty())) apply_l6_error(traces.data(), ntrace, cur_novel);
    if (p.ro_adam && ++ro_n % std::max(1u, p.ro_adam_k) == 0) { if (tst_.q.empty()) ro_adam_step(); else ro_pending = true; }   // tiles: after tiles_learn
}
void Organism::ro_adam_step() {   // one minibatch Adam step of the shared float readout, the register readout and the write head
    const uint64_t t = ++ro_t; const float c1 = 1.f - std::pow(0.9f, (float)t), c2 = 1.f - std::pow(0.999f, (float)t), lr = p.ro_adam_lr, ik = 1.f / (float)std::max(1u, p.ro_adam_k);
    auto step = [&](float* w, float* g, float* m, float* v, size_t n, float wd) { for (size_t i = 0; i < n; ++i) { const float gr = g[i] * ik; g[i] = 0.f;
        m[i] = 0.9f * m[i] + 0.1f * gr; v[i] = 0.999f * v[i] + 0.001f * gr * gr; if (v[i] > 0.f) w[i] -= lr * ((m[i] / c1) / (std::sqrt(v[i] / c2) + 1e-8f) + wd * w[i]); } };
    const float wd = p.ro_adam_wd;
    const uint32_t V = p.vocab;
    if (!ro_g.empty()) { step(wout.data(), ro_g.data(), ro_m.data(), ro_v.data(), ro_g.size(), wd); for (uint32_t v = 0; v < V; ++v) for (int k = 0; k < Dims::F; ++k) woutT[(size_t)k * V + v] = wout[(size_t)v * Dims::F + k]; }
    if (!rg_g.empty() && !wreg.empty()) step(wreg.data(), rg_g.data(), rg_m.data(), rg_v.data(), rg_g.size(), wd);
    if (!hd_g.empty()) { step(&hw_g[0][0], hd_g.data(), hd_m.data(), hd_v.data(), 4 * Dims::F, wd); step(&hw_v[0][0], hd_g.data() + 4 * Dims::F, hd_m.data() + 4 * Dims::F, hd_v.data() + 4 * Dims::F, 4 * Dims::F, wd);
        step(hb_g, hd_g.data() + 8 * Dims::F, hd_m.data() + 8 * Dims::F, hd_v.data() + 8 * Dims::F, 4, 0.f); }
    if (!sg_g.empty()) step(seq_gw, sg_g.data(), sg_m.data(), sg_v.data(), sg_g.size(), 0.f);   // the sequence-memory gate
    if (!mtp_g.empty()) step(mtp_W.data(), mtp_g.data(), mtp_m.data(), mtp_v.data(), mtp_g.size(), wd);   // mtp heads
    if (!tile_Mg.empty()) step(tile_M.data(), tile_Mg.data(), tile_Mm.data(), tile_Mv.data(), tile_Mg.size(), wd);   // tiles: read mixings
    if (!tile_Ag.empty()) step(tile_A.data(), tile_Ag.data(), tile_Am.data(), tile_Av.data(), tile_Ag.size(), wd);   // tiles: attention read
    if (!fw_Wg.empty()) step(fw_W.data(), fw_Wg.data(), fw_Wm.data(), fw_Wv.data(), fw_Wg.size(), wd);   // tiles: fast-weight projections
    if (!ret_Wg.empty()) step(ret_W.data(), ret_Wg.data(), ret_Wm.data(), ret_Wv.data(), ret_Wg.size(), wd);   // tiles: the episodic read's output map
    if (!wsp_g.empty()) step(wsp.data(), wsp_g.data(), wsp_m.data(), wsp_v.data(), wsp_g.size(), wd);   // ws_split: the per-motif readout
    if (!wtl_g.empty()) step(wtl.data(), wtl_g.data(), wtl_m.data(), wtl_v.data(), wtl_g.size(), wd);   // tile_split: the tiles' channel
    if (!wsp5_g.empty()) step(wsp5.data(), wsp5_g.data(), wsp5_m.data(), wsp5_v.data(), wsp5_g.size(), wd);   // ws_split5
    if (!mix_g.empty()) step(mix_th.data(), mix_g.data(), mix_m.data(), mix_v.data(), mix_g.size(), 0.f);   // motif_mix: the gates
    if (!dis_g.empty()) step(dis_w.data(), dis_g.data(), dis_m.data(), dis_v.data(), dis_g.size(), 0.f);   // disinh: the releases
    if (!att_Wg.empty()) step(att_W.data(), att_Wg.data(), att_Wm.data(), att_Wv.data(), att_Wg.size(), 0.f);   // motif_attn: the projections
    if (!rh_W2g.empty()) { step(rh_W1.data(), rh_W1g.data(), rh_W1m.data(), rh_W1v.data(), rh_W1g.size(), wd); step(rh_b1.data(), rh_b1g.data(), rh_b1m.data(), rh_b1v.data(), rh_b1g.size(), 0.f);
        step(rh_W2.data(), rh_W2g.data(), rh_W2m.data(), rh_W2v.data(), rh_W2g.size(), wd); }   // ro_hidden
}
// ---- tiles: the pyramid (docs/PYRAMID_MEMORY.md), on one stream's TileState ------------------------------------------------------
void Organism::tiles_read(TileState& ts) const {   // ts.a = sum over the view and working memory of M(slot) v
    std::fill(ts.a, ts.a + Dims::F, 0.f); const uint32_t K = p.tile_k; if (diag_zero_tiles) return;
    for (uint32_t l = 0; l < ts.q.size(); ++l) for (uint32_t k = 0; k < ts.q[l].size(); ++k) {
        const float* M = &tile_M[((size_t)l * K + k) * Dims::F * Dims::F]; const float* v = ts.q[l][k].v;
        for (int i = 0; i < Dims::F; ++i) { float s = 0.f; const float* r = M + (size_t)i * Dims::F; for (int j = 0; j < Dims::F; ++j) s += r[j] * v[j]; ts.a[i] += s; } }
    for (uint32_t s = 0; s < ts.wm.size(); ++s) { if (!ts.wm_live[s]) continue;   // working memory
        const float* M = &tile_M[((size_t)ts.q.size() * K + s) * Dims::F * Dims::F]; const float* v = ts.wm[s].v;
        for (int i = 0; i < Dims::F; ++i) { float a = 0.f; const float* r = M + (size_t)i * Dims::F; for (int j = 0; j < Dims::F; ++j) a += r[j] * v[j]; ts.a[i] += a; } }
}
// tile_att: query from this event's context, softmax over the live slots (view + working memory); tile_fw: r = A q. Both read into ts.a.
void Organism::tiles_attend(TileState& ts, const int32_t* w, const uint8_t* hist) const {
    const int F = Dims::F;
    ts.att_slot.clear(); std::fill(ts.att_r, ts.att_r + F, 0.f); std::fill(ts.fw_r, ts.fw_r + F, 0.f); ts.ret_al.clear();
    if (diag_zero_tiles) return;
    if (!tile_A.empty()) {
        const float* Wq = tile_A.data(); const float* Wk = Wq + F * F; const float* Wo = Wk + F * F;
        for (auto& q : ts.q) for (auto& t : q) ts.att_slot.push_back(&t); for (size_t s = 0; s < ts.wm.size(); ++s) if (ts.wm_live[s]) ts.att_slot.push_back(&ts.wm[s]);
        const size_t n = ts.att_slot.size();
        if (n) {
            for (int i = 0; i < F; ++i) ts.att_in[i] = (float)w[i] / 128.f + (p.tile_ctx ? ts.a[i] : 0.f);   // tile_ctx: the query sees the read context
            for (int i = 0; i < F; ++i) { float s = 0.f; for (int j = 0; j < F; ++j) s += Wq[i * F + j] * ts.att_in[j]; ts.att_q[i] = s; }
            ts.att_k.assign(n * F, 0.f); ts.att_al.assign(n, 0.f); const float isd = 1.f / std::sqrt((float)F); float mx = -1e30f;
            for (size_t s = 0; s < n; ++s) { float* k = &ts.att_k[s * F]; const float* v = ts.att_slot[s]->v; float sc = 0.f;
                for (int i = 0; i < F; ++i) { float a = 0.f; for (int j = 0; j < F; ++j) a += Wk[i * F + j] * v[j]; k[i] = a; sc += ts.att_q[i] * a; }
                ts.att_al[s] = sc * isd; mx = std::max(mx, ts.att_al[s]); }
            float Z = 0.f; for (size_t s = 0; s < n; ++s) { ts.att_al[s] = std::exp(ts.att_al[s] - mx); Z += ts.att_al[s]; }
            for (size_t s = 0; s < n; ++s) { ts.att_al[s] /= Z; const float* v = ts.att_slot[s]->v; for (int j = 0; j < F; ++j) ts.att_r[j] += ts.att_al[s] * v[j]; }
            for (int i = 0; i < F; ++i) { float a = 0.f; for (int j = 0; j < F; ++j) a += Wo[i * F + j] * ts.att_r[j]; ts.a[i] += a; } }
    }
    if (!ts.fw_A.empty()) {
        const float* Wq = fw_W.data(); const float* Wo = Wq + 2 * F * F;
        for (int i = 0; i < F; ++i) ts.fw_in[i] = (float)w[i] / 128.f + (p.tile_ctx ? ts.a[i] : 0.f);   // tile_ctx: the cue sees the read context
        for (int i = 0; i < F; ++i) { float a = 0.f; for (int j = 0; j < F; ++j) a += Wq[i * F + j] * ts.fw_in[j]; ts.fw_q[i] = a; }
        for (int i = 0; i < F; ++i) { float a = 0.f; for (int j = 0; j < F; ++j) a += ts.fw_A[i * F + j] * ts.fw_q[j]; ts.fw_r[i] = a; }
        for (int i = 0; i < F; ++i) { float a = 0.f; for (int j = 0; j < F; ++j) a += Wo[i * F + j] * ts.fw_r[j]; ts.a[i] += a; }
    }
    if (!ret_W.empty() && !ts.ret.empty()) {   // tile_ret: fixed-key retrieval over the episodic buffer, one softmax per head
        const uint64_t first = (hist && std::isalnum(hist[1]) && !std::isalnum(hist[0])) ? word_hash(hist) : 0;
        const uint32_t H = ret_heads(); const size_t n = ts.ret.size(); ts.ret_al.assign(H * n, 0.f);
        const float rec = p.tile_ret_recency / (float)std::max<uint32_t>(1, p.tile_ret);   // items are newest first
        for (uint32_t h = 0; h < H; ++h) { float* al = &ts.ret_al[h * n]; float* q = ts.ret_q[h]; float* r = ts.ret_r[h];
            ret_bag(ts, q, first, ret_pos(h)); float mx = -1e30f;
            for (size_t j = 0; j < n; ++j) { float c = 0.f; const float* k = ts.ret[j].key[h]; for (int d = 0; d < TileState::RET_D; ++d) c += q[d] * k[d];
                al[j] = p.tile_ret_beta * c; if (rec != 0.f) al[j] -= rec * (float)j; if (j < p.tile_ret_skip) al[j] = -1e30f; mx = std::max(mx, al[j]); }
            float Z = 0.f; for (size_t j = 0; j < n; ++j) { al[j] = mx > -1e29f ? std::exp(al[j] - mx) : 0.f; Z += al[j]; } if (Z <= 0.f) Z = 1.f;
            std::fill(r, r + F, 0.f); for (size_t j = 0; j < n; ++j) { al[j] /= Z; for (int i = 0; i < F; ++i) r[i] += al[j] * ts.ret[j].v[i]; } }
        const int HF = (int)H * F;
        for (int i = 0; i < F; ++i) { float a = 0.f; for (int j = 0; j < HF; ++j) a += ret_W[i * HF + j] * ts.ret_r[j / F][j % F]; ts.a[i] += a; } }
    if (on_attend) on_attend(ts);
}
// tile_ret: a word's hashed identity as a +-1 code of TileState::RET_D; the bag of the last tile_ret_c closed words, each weighted by
// 1 / sqrt(1 + its running frequency) (frequent words - "the" - say little about which binding is meant), unit length
void Organism::ret_bag(const TileState& ts, float* out, uint64_t first, bool positional) const {
    std::fill(out, out + TileState::RET_D, 0.f); const uint32_t n = ts.nrh + (first ? 1 : 0), C = std::min<uint32_t>(std::min<uint32_t>(p.tile_ret_c, 8), n);
    for (uint32_t i = 0; i < C; ++i) { const uint64_t h = first ? (i == 0 ? first : ts.rh[i - 1]) : ts.rh[i]; const size_t b = h % ts.fq_n.size();
        const float f = ts.fq_n[b] * std::pow(0.9999f, (float)(ts.nw - ts.fq_t[b])), w = 1.f / std::sqrt(1.f + f);
        const uint64_t salt = positional ? (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ULL : 0ULL;   // positional: a code per position
        for (int d = 0; d < TileState::RET_D; d += 64) { const uint64_t c = mix64(h ^ salt ^ (0xC0DE0000ULL + (uint64_t)d)); for (int k = 0; k < 64; ++k) out[d + k] += ((c >> k) & 1) ? w : -w; } }
    float n2 = 0.f; for (int d = 0; d < TileState::RET_D; ++d) n2 += out[d] * out[d];
    if (n2 > 0.f) { const float in = 1.f / std::sqrt(n2); for (int d = 0; d < TileState::RET_D; ++d) out[d] *= in; }
}
// dl6 = d loss / d (the readout's input). The mixings', attention's and fast weights' gradients; each tile's gradient: a word (or
// averaged) tile's to the writers of the events that closed its words, through the ring; a merged tile's accumulates for its merge
// body's delayed pass; its size counts as the tile's use. The queries' gradient goes to ts.dq (the caller adds it to dl6 afterwards).
void Organism::tiles_learn(TileState& ts, const float* dl6, TileLearn& tl) const {
    const int F = Dims::F; const uint32_t K = p.tile_k; const float gw = std::min(tl.goal_w, 4.f);
    std::fill(ts.dq, ts.dq + F, 0.f);
    auto leafy = [](const Tile& t) { return !t.has_tr && t.unit >= UINT32_MAX - 1; };   // a word tile, or an averaged one
    auto credit = [&](Tile& t, const float* gt) {   // this tile's gradient: its use; its words' writers through the ring, or its accumulator
        float u = 0.f; for (int j = 0; j < F; ++j) u += std::fabs(gt[j]); t.use += u;
        if (leafy(t)) { const float w = 1.f / (float)std::max<uint8_t>(1, t.nleaf);
            for (uint8_t i = 0; i < std::max<uint8_t>(1, t.nleaf); ++i) tl.inject(tl.event - (t.nleaf ? t.leaf[i] : t.src), gt, w, t.v); }
        else for (int j = 0; j < F; ++j) t.g[j] += gt[j];
    };
    auto mixing = [&](Tile& t, size_t slot) {
        const size_t o = slot * F * F; const float* M = &tile_M[o]; float* G = tl.gM + o; float gt[Dims::F] = {0};
        for (int i = 0; i < F; ++i) { const float d = dl6[i]; if (d == 0.f) continue; const float* r = M + (size_t)i * F; float* gr = G + (size_t)i * F;
            for (int j = 0; j < F; ++j) { gr[j] += d * gw * t.v[j]; gt[j] += r[j] * d; } }
        credit(t, gt);
    };
    for (uint32_t l = 0; l < ts.q.size(); ++l) for (uint32_t k = 0; k < ts.q[l].size(); ++k) mixing(ts.q[l][k], (size_t)l * K + k);
    for (uint32_t s = 0; s < ts.wm.size(); ++s) if (ts.wm_live[s]) mixing(ts.wm[s], (size_t)ts.q.size() * K + s);
    if (tl.gA && !ts.att_slot.empty()) {   // attention: through the read, the weights and the keys
        const size_t n = ts.att_slot.size(); const float* Wq = tile_A.data(); const float* Wk = Wq + F * F; const float* Wo = Wk + F * F;
        float* Gq = tl.gA; float* Gk = Gq + F * F; float* Go = Gk + F * F; const float isd = 1.f / std::sqrt((float)F);
        float dr[Dims::F] = {0}; for (int i = 0; i < F; ++i) { const float d = dl6[i]; if (d == 0.f) continue; for (int j = 0; j < F; ++j) { Go[i * F + j] += d * gw * ts.att_r[j]; dr[j] += Wo[i * F + j] * d; } }
        std::vector<float> dal(n, 0.f); float dot = 0.f;
        for (size_t s = 0; s < n; ++s) { const float* v = ts.att_slot[s]->v; float a = 0.f; for (int j = 0; j < F; ++j) a += dr[j] * v[j]; dal[s] = a; dot += ts.att_al[s] * a; }
        float dq[Dims::F] = {0};
        for (size_t s = 0; s < n; ++s) { Tile& t = *ts.att_slot[s]; const float ds = ts.att_al[s] * (dal[s] - dot) * isd; const float* k = &ts.att_k[s * F];
            float dk[Dims::F]; for (int i = 0; i < F; ++i) { dq[i] += ds * k[i]; dk[i] = ds * ts.att_q[i]; }
            float dv[Dims::F]; for (int j = 0; j < F; ++j) dv[j] = ts.att_al[s] * dr[j];
            for (int i = 0; i < F; ++i) { if (dk[i] == 0.f) continue; for (int j = 0; j < F; ++j) { Gk[i * F + j] += dk[i] * gw * t.v[j]; dv[j] += Wk[i * F + j] * dk[i]; } }
            credit(t, dv); }
        for (int i = 0; i < F; ++i) { if (dq[i] == 0.f) continue; for (int j = 0; j < F; ++j) Gq[i * F + j] += dq[i] * gw * ts.att_in[j]; }
        for (int j = 0; j < F; ++j) { float a = 0.f; for (int i = 0; i < F; ++i) a += Wq[i * F + j] * dq[i]; ts.dq[j] += a; }   // the bodies learn to ask
    }
    if (tl.gF && !ts.fw_A.empty()) {   // fast weights: Wo, Wq from every read; Wk and the stored values from the words still in the level-0 view
        const float* Wq = fw_W.data(); const float* Wk = Wq + F * F; const float* Wo = Wk + F * F; float* Gq = tl.gF; float* Gk = Gq + F * F; float* Go = Gk + F * F;
        float dr[Dims::F] = {0}; for (int i = 0; i < F; ++i) { const float d = dl6[i]; if (d == 0.f) continue; for (int j = 0; j < F; ++j) { Go[i * F + j] += d * gw * ts.fw_r[j]; dr[j] += Wo[i * F + j] * d; } }
        float dq[Dims::F] = {0}; for (int j = 0; j < F; ++j) { float a = 0.f; for (int i = 0; i < F; ++i) a += ts.fw_A[i * F + j] * dr[i]; dq[j] = a; }
        for (int i = 0; i < F; ++i) { if (dq[i] == 0.f) continue; for (int j = 0; j < F; ++j) Gq[i * F + j] += dq[i] * gw * ts.fw_in[j]; }
        if (!ts.q.empty()) for (Tile& t : ts.q[0]) { if (!t.fw_t) continue; const float fac = std::pow(p.tile_fw_decay, (float)(ts.fw_n - t.fw_t));
            float kq = 0.f, vd = 0.f; for (int j = 0; j < F; ++j) { kq += t.fk[j] * ts.fw_q[j]; vd += t.v[j] * dr[j]; }
            float dk[Dims::F], dv[Dims::F]; for (int j = 0; j < F; ++j) { dk[j] = fac * vd * ts.fw_q[j]; dv[j] = fac * kq * dr[j]; }
            float kd = 0.f; for (int j = 0; j < F; ++j) kd += t.fk[j] * dk[j];   // through the unit-length key (norm treated as fixed)
            for (int i = 0; i < F; ++i) { const float dki = dk[i] - t.fk[i] * kd; if (dki == 0.f) continue; for (int j = 0; j < F; ++j) { Gk[i * F + j] += dki * gw * t.v[j]; dv[j] += Wk[i * F + j] * dki; } }
            credit(t, dv); }
        for (int j = 0; j < F; ++j) { float a = 0.f; for (int i = 0; i < F; ++i) a += Wq[i * F + j] * dq[i]; ts.dq[j] += a; }   // the bodies learn to cue
    }
    if (tl.gR && !ts.ret_al.empty()) {   // tile_ret: the output map from every read; each item its share (to the writers of its word, through the ring)
        const uint32_t H = ret_heads(); const int HF = (int)H * F; const size_t n = ts.ret_al.size() / H;
        const float* Wo = ret_W.data(); float* Go = tl.gR; float dr[2 * Dims::F] = {0};
        for (int i = 0; i < F; ++i) { const float d = dl6[i]; if (d == 0.f) continue; for (int j = 0; j < HF; ++j) { Go[i * HF + j] += d * gw * ts.ret_r[j / F][j % F]; dr[j] += Wo[i * HF + j] * d; } }
        for (size_t j = 0; j < n && j < ts.ret.size(); ++j) { float amax = 0.f; for (uint32_t h = 0; h < H; ++h) amax = std::max(amax, ts.ret_al[h * n + j]); if (amax < 1e-3f) continue;
            float dv[Dims::F]; for (int i = 0; i < F; ++i) { float sdv = 0.f; for (uint32_t h = 0; h < H; ++h) sdv += ts.ret_al[h * n + j] * dr[h * F + i]; dv[i] = sdv; }
            tl.inject(tl.event - ts.ret[j].src, dv, 1.f, ts.ret[j].v); }
    }
}
void Organism::fw_write(TileState& ts, Tile& t) const {   // tile_fw: a closing word writes value x unit key into the decaying matrix
    const int F = Dims::F; const float* Wk = fw_W.data() + F * F; float n2 = 0.f;
    for (int i = 0; i < F; ++i) { float a = 0.f; for (int j = 0; j < F; ++j) a += Wk[i * F + j] * t.v[j]; t.fk[i] = a; n2 += a * a; }
    const float in = 1.f / std::sqrt(std::max(n2, 1e-12f)); for (int i = 0; i < F; ++i) t.fk[i] *= in;
    const float l = p.tile_fw_decay; for (int i = 0; i < F; ++i) for (int j = 0; j < F; ++j) ts.fw_A[i * F + j] = l * ts.fw_A[i * F + j] + t.v[i] * t.fk[j];
    t.fw_t = ++ts.fw_n;
}
void Organism::tiles_close(TileState& ts, const int32_t* ws, const uint8_t* hist, uint64_t event) {   // the byte just read ends a word
    if (!(std::isalnum(hist[1]) && !std::isalnum(hist[0]))) return;
    for (size_t s = 0; s < ts.wm.size(); ++s) ts.wm[s].use *= p.tile_wm_decay;   // working memory: use decays per word
    Tile t; for (int j = 0; j < Dims::F; ++j) { t.v[j] = std::max(-4.f, std::min(4.f, (float)ws[j] / 128.f + (p.tile_ctx ? ts.a[j] : 0.f))); t.g[j] = 0.f; }
    t.src = event; t.leaf[0] = event; t.nleaf = 1;
    const uint64_t h = word_hash(hist);   // the word's bytes
    if (p.tile_id > 0.f) {   // the word's identity: its bytes (hist[1] back to the previous non-alphanumeric byte) hashed to a +-tile_id code
        const uint64_t c = mix64(h ^ 0x1D1DULL); for (int j = 0; j < Dims::F; ++j) t.v[j] = std::max(-4.f, std::min(4.f, t.v[j] + (((c >> j) & 1) ? p.tile_id : -p.tile_id))); }
    if (!ts.fw_A.empty()) fw_write(ts, t);
    if (p.tile_ret) {   // into the episodic buffer, keyed by the words BEFORE it; then it joins the recent words and the frequencies
        TileState::RetItem it; std::copy(t.v, t.v + Dims::F, it.v); for (uint32_t h = 0; h < ret_heads(); ++h) ret_bag(ts, it.key[h], 0, ret_pos(h)); it.src = event;
        ts.ret.push_front(it); if (ts.ret.size() > p.tile_ret) ts.ret.pop_back();
        for (int i = 7; i > 0; --i) ts.rh[i] = ts.rh[i - 1]; ts.rh[0] = h; ts.nrh = std::min<uint32_t>(ts.nrh + 1, 8);
        const size_t b = h % ts.fq_n.size(); ts.fq_n[b] = ts.fq_n[b] * std::pow(0.9999f, (float)(ts.nw - ts.fq_t[b])) + 1.f; ts.fq_t[b] = ts.nw; ++ts.nw; }
    tile_push(ts, 0, std::move(t), event);
}
void Organism::tile_push(TileState& ts, uint32_t l, Tile&& t, uint64_t event) {   // into level l's view; the oldest is offered to working memory; every 2nd closes a parent
    auto& q = ts.q[l]; q.push_front(std::move(t)); ++ts.n[l];
    if (q.size() > p.tile_k) { tile_offer(ts, std::move(q.back())); q.pop_back(); }
    if (l + 1 < ts.q.size() && ts.n[l] % 2 == 0) { Tile par; tile_merge(l + 1, q[1], q[0], par, event); tile_push(ts, l + 1, std::move(par), event); }
}
// a tile leaving the view: into working memory if its readers leaned on it more than on the weakest slot's item (use decays per word);
// whatever is finally dropped takes its merge body's delayed pass
void Organism::tile_offer(TileState& ts, Tile&& t) {
    int weakest = -1; float wu = 0.f;
    for (size_t s = 0; s < ts.wm.size(); ++s) { if (!ts.wm_live[s]) { weakest = (int)s; wu = -1.f; break; } if (weakest < 0 || ts.wm[s].use < wu) { weakest = (int)s; wu = ts.wm[s].use; } }
    if (weakest >= 0 && t.use > wu) {
        if (ts.wm_live[weakest] && ts.wm[weakest].has_tr && p.train) tile_backprop(ts.wm[weakest]);
        ts.wm[weakest] = std::move(t); ts.wm_live[weakest] = 1; return; }
    if (t.has_tr && p.train) tile_backprop(t);
}
void Organism::tile_merge(uint32_t l, const Tile& a, const Tile& b, Tile& out, uint64_t event) {   // a level-l parent of [older, newer]
    if (p.tile_avg) {   // the mean of the two children; its leaves are theirs (credit goes to every word underneath)
        for (int j = 0; j < Dims::F; ++j) { out.v[j] = 0.5f * (a.v[j] + b.v[j]); out.g[j] = 0.f; } out.src = event; out.unit = UINT32_MAX - 1; out.has_tr = false; out.nleaf = 0;
        for (const Tile* c : {&a, &b}) for (uint8_t i = 0; i < c->nleaf && out.nleaf < 8; ++i) out.leaf[out.nleaf++] = c->leaf[i];
        return; }
    uint64_t bits = 0; for (int j = 0; j < 16; ++j) { bits |= (uint64_t)(a.v[j] > 0.f) << j; bits |= (uint64_t)(b.v[j] > 0.f) << (16 + j); }
    const uint32_t u = tile_base[l - 1] + (uint32_t)(mix64(bits ^ ((uint64_t)l << 40)) % p.tile_bodies);   // same roles -> same merge body
    const uint8_t* W = wbuf.p + (size_t)u * Dims::UNIT_BYTES; Trace& tr = out.tr;
    std::memset(out.x1, 0, Dims::IN);
    for (int j = 0; j < Dims::F; ++j) { out.x1[j] = clamp8((int32_t)std::lround(a.v[j] * 32.f)); out.x1[Dims::F + j] = clamp8((int32_t)std::lround(b.v[j] * 32.f)); }
    tr.unit = u; tr.reached = R_L1 | R_L23 | R_L5 | R_L6; tr.emitted = 0; for (int li = 0; li < NUM_LAYERS; ++li) tr.sh[li] = L[li].shift;
    kern.run(W, L[LI_1], out.x1, tr.l1);
    std::memcpy(tr.x2, tr.l1, Dims::A); std::memset(tr.x2 + Dims::A, 0, Dims::R2); kern.run(W, L[LI_2], tr.x2, tr.l2);
    std::memcpy(tr.x3, tr.l1, Dims::A); std::memset(tr.x3 + Dims::A, 0, Dims::B); kern.run(W, L[LI_3], tr.x3, tr.l3);
    std::memcpy(tr.x5, tr.l2, Dims::B); std::memcpy(tr.x5 + Dims::B, tr.l3, Dims::C); kern.run(W, L[LI_5], tr.x5, tr.l5);
    std::memcpy(tr.x6, tr.l5, Dims::E); std::memset(tr.x6 + Dims::E, 0, Dims::R6); kern.run(W, L[LI_6], tr.x6, tr.l6);
    for (int j = 0; j < Dims::F; ++j) { out.v[j] = (float)tr.l6[j] / 32.f; out.g[j] = 0.f; } out.src = event; out.unit = u; out.has_tr = true;
}
void Organism::tile_backprop(Tile& t) {   // a merged tile leaves the view: its merge body learns what its readers needed (children not credited)
    bool any = false; for (int j = 0; j < Dims::F; ++j) any |= t.g[j] != 0.f; if (!any || !p.adam || ad_g.empty()) return;
    const Trace& tr = t.tr; const uint32_t u = t.unit; const uint8_t* W = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
    auto live = [&](int8_t y) { return (y >= 127 || y <= -127) ? p.bp_leak : 1.f; };
    float g6[Dims::F], g5[Dims::E] = {0}, g2[Dims::B] = {0}, g3[Dims::C] = {0}, g1[Dims::A] = {0};
    for (int j = 0; j < Dims::F; ++j) g6[j] = t.g[j] / 32.f * live(tr.l6[j]) / (float)(1 << tr.sh[LI_6]);
    for (int j = 0; j < Dims::F; ++j) if (g6[j] != 0.f) for (int i = 0; i < Dims::E; ++i) g5[i] += (float)nib_get(W, Dims::NIB_L6 + j * Dims::IN6 + i) * g6[j];
    for (int i = 0; i < Dims::E; ++i) g5[i] *= live(tr.l5[i]) / (float)(1 << tr.sh[LI_5]);
    for (int j = 0; j < Dims::E; ++j) if (g5[j] != 0.f) for (int i = 0; i < Dims::IN5; ++i) { const float gx = (float)nib_get(W, Dims::NIB_L5 + j * Dims::IN5 + i) * g5[j]; if (i < Dims::B) g2[i] += gx; else g3[i - Dims::B] += gx; }
    for (int i = 0; i < Dims::C; ++i) g3[i] *= live(tr.l3[i]) / (float)(1 << tr.sh[LI_3]);
    for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + i) * g3[j];
    for (int i = 0; i < Dims::B; ++i) g2[i] *= live(tr.l2[i]) / (float)(1 << tr.sh[LI_2]);
    for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L2 + j * Dims::IN2 + i) * g2[j];
    for (int i = 0; i < Dims::A; ++i) g1[i] *= live(tr.l1[i]) / (float)(1 << tr.sh[LI_1]);
    if (p.adam_l6) adam_grad(u, Dims::NIB_L6, tr.x6, Dims::IN6, g6, Dims::F, 1.f);
    adam_grad(u, Dims::NIB_L5, tr.x5, Dims::IN5, g5, Dims::E, 1.f); adam_grad(u, Dims::NIB_L2, tr.x2, Dims::IN2, g2, Dims::B, 1.f);
    adam_grad(u, Dims::NIB_L3, tr.x3, Dims::IN3, g3, Dims::C, 1.f); adam_grad(u, Dims::NIB_L1, t.x1, Dims::IN, g1, Dims::A, 1.f);
    adam_touch(u);
}
// mtp (goal item E): head h reads event t-h's workspace (its ring slot's head input a = clamp(ws/128)) and predicts THIS event's
// target, i.e. the byte h events after that event's own. The error at a goes to event t-h's writers with the rest of their
// future error (their delayed pass when the slot leaves the window), in the next-byte readout's convention (d loss / d a).
void Organism::mtp_learn(uint8_t target) {
    const uint32_t K = std::min<uint32_t>(p.mtp, 8), V = p.vocab, Wn = p.tc_window;
    if (mtp_W.empty()) { mtp_W.assign((size_t)K * V * Dims::F, 0.f); mtp_g = mtp_W; mtp_m = mtp_W; mtp_v = mtp_W; mtp_p.assign(V, 0.f); }
    for (uint32_t h = 1; h <= K && h <= tc_fill && h < Wn; ++h) {
        TcEvent& e = tc_ring[(tc_head + Wn - h) % Wn];
        const float* W = &mtp_W[(size_t)(h - 1) * V * Dims::F]; float* G = &mtp_g[(size_t)(h - 1) * V * Dims::F]; float* pr = mtp_p.data();
        float mx = -1e30f; for (uint32_t v = 0; v < V; ++v) { const float* row = W + (size_t)v * Dims::F; float z = 0.f; for (int j = 0; j < Dims::F; ++j) z += row[j] * e.ha[j]; pr[v] = z; mx = std::max(mx, z); }
        float Z = 0.f; for (uint32_t v = 0; v < V; ++v) { pr[v] = std::exp(pr[v] - mx); Z += pr[v]; }
        const float iz = 1.f / Z; mtp_loss[h - 1] += -std::log(std::max(pr[target] * iz, 1e-9f)); ++mtp_n[h - 1];
        float gha[Dims::F] = {0};
        for (uint32_t v = 0; v < V; ++v) { const float err = (pr[v] * iz - (v == target ? 1.f : 0.f)) * p.mtp_w;
            const float* row = W + (size_t)v * Dims::F; float* grow = G + (size_t)v * Dims::F;
            for (int j = 0; j < Dims::F; ++j) { gha[j] += err * row[j]; grow[j] += err * e.ha[j]; } }
        for (int j = 0; j < Dims::F; ++j) if (e.ha[j] > -4.f && e.ha[j] < 4.f) e.tg[j] += gha[j];
    }
}
// The L6 local rule for an error already quantised into q6, over any set of traces.
void Organism::apply_l6_error(const Trace* trs, uint32_t n, bool novel) {
    const int32_t shift = p.lr_shift + (novel ? 0 : p.lr_anneal);
    for (uint32_t t = 0; t < n; ++t) {
        const Trace& tr = trs[t]; if (!(tr.reached & R_L6)) continue;
        if (p.route_return && tr.source == SRC_ROUTED) continue;
        int32_t* d = delta_buf.data();
        if (p.stoch_round) {   // unbiased: sub-step products survive in expectation instead of truncating to zero
            for (int j = 0; j < Dims::F; ++j) for (int i = 0; i < Dims::IN6; ++i) d[j * Dims::IN6 + i] = scale_sr(-((int32_t)tr.x6[i] * q6[j] * p.lr_num), shift);
            apply_acc_block(tr.unit, Dims::NIB_L6, d, Dims::P6); continue;
        }
#if defined(__AVX2__)
        static_assert(Dims::IN6 % 8 == 0, "vector delta path assumes IN6 multiple of 8");
        const __m128i sh = _mm_cvtsi32_si128(shift);
        __m256i x[Dims::IN6 / 8];
        for (int i = 0; i < Dims::IN6; i += 8) x[i / 8] = _mm256_mullo_epi32(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(tr.x6 + i))), _mm256_set1_epi32(p.lr_num));
        for (int j = 0; j < Dims::F; ++j) {
            __m256i q = _mm256_set1_epi32(-q6[j]);
            for (int i = 0; i < Dims::IN6; i += 8) {
                __m256i v = _mm256_mullo_epi32(x[i / 8], q);                                   // -(x*q*lr_num)
                __m256i mag = _mm256_srl_epi32(_mm256_abs_epi32(v), sh);                       // |v| >> shift
                _mm256_storeu_si256((__m256i*)(d + j * Dims::IN6 + i), _mm256_sign_epi32(mag, v)); // restore sign (trunc toward zero)
            }
        }
#else
        for (int j = 0; j < Dims::F; ++j) for (int i = 0; i < Dims::IN6; ++i) d[j * Dims::IN6 + i] = scale_trunc(-((int32_t)tr.x6[i] * q6[j] * p.lr_num), shift);
#endif
        apply_acc_block(tr.unit, Dims::NIB_L6, d, Dims::P6);
    }
}
void Organism::learn_hidden(int32_t dq, const std::vector<Trace>& trA, uint32_t nA) {
    if (dq == 0) return;
    const int32_t shift = p.lr_shift + p.hidden_shift + (cur_novel ? 0 : p.lr_anneal);
    auto upd = [&](uint32_t unit, int layer, uint32_t nib0, const int8_t* x, int in, int out) {
        uint64_t bits = mix64(event_idx * 0x9E3779B97F4A7C15ULL ^ ((uint64_t)unit << 20) ^ ((uint64_t)layer << 56));
        for (int j = 0; j < out; ++j) { int s = ((bits >> j) & 1) ? 1 : -1;
            for (int i = 0; i < in; ++i) apply_acc(unit, nib0 + j * in + i, scale_trunc(-(dq * s * (int32_t)x[i] * p.lr_num), shift)); }
    };
    for (uint32_t t = 0; t < nA; ++t) {
        const Trace& tr = trA[t];
        if (tr.reached & R_L1) upd(tr.unit, LI_1, Dims::NIB_L1, xin, Dims::IN, Dims::A);
        if (tr.reached & R_L23) { upd(tr.unit, LI_2, Dims::NIB_L2, tr.x2, Dims::IN2, Dims::B); upd(tr.unit, LI_3, Dims::NIB_L3, tr.x3, Dims::IN3, Dims::C); }
        if (tr.reached & R_L5) upd(tr.unit, LI_5, Dims::NIB_L5, tr.x5, Dims::IN5, Dims::E);
    }
}
// hidden=modulated (item 4, ALM's three-factor rule): Delta w = eta * M * e with e = pre * post the local
// eligibility of the synapse and M one diffuse scalar modulator, the bits this event gained over the running
// loss. No error is propagated between layers; a synapse strengthens the input/output conjunction that was
// active when the organism did better than it usually does, and weakens it when it did worse.
void Organism::learn_modulated(int32_t M) { modulated_on(traces.data(), ntrace, xin, M, cur_novel); }
void Organism::modulated_on(const Trace* trs, uint32_t n, const int8_t* x1, int32_t M, bool novel) {
    if (M == 0) return;
    const int32_t shift = p.lr_shift + p.hidden_shift + (novel ? 0 : p.lr_anneal);
    int32_t* d = delta_buf.data();
    auto upd = [&](uint32_t unit, uint32_t nib0, const int8_t* x, const int8_t* y, int in, int out) {
        for (int j = 0; j < out; ++j) { int32_t post = y[j] / 16;                     // post-synaptic activity, coarse
            for (int i = 0; i < in; ++i) d[j * in + i] = scale_trunc(M * post * (int32_t)x[i] * p.lr_num, shift); }
        apply_acc_block(unit, nib0, d, in * out);
    };
    for (uint32_t t = 0; t < n; ++t) {
        const Trace& tr = trs[t];
        if (tr.reached & R_L1) upd(tr.unit, Dims::NIB_L1, x1, tr.l1, Dims::IN, Dims::A);
        if (tr.reached & R_L23) { upd(tr.unit, Dims::NIB_L2, tr.x2, tr.l2, Dims::IN2, Dims::B); upd(tr.unit, Dims::NIB_L3, tr.x3, tr.l3, Dims::IN3, Dims::C); }
        if (tr.reached & R_L5) upd(tr.unit, Dims::NIB_L5, tr.x5, tr.l5, Dims::IN5, Dims::E);
    }
}
// hidden=backprop: the backprop comparator. The exact gradient of the loss w.r.t. every hidden synapse of each body
// that emitted into the workspace, through that body's own layers (L6 -> L5 -> L2/L3 -> L1, the clamp8 derivative,
// the fixed right shifts), fed through the same quantised accumulators as the local rules: only the learning SIGNAL
// differs. Routing, the neighbour messages (x3's sender half) and the return state are treated as inputs, so what this
// measures is whether full-depth credit inside a body beats the three-factor rule on the same architecture.
// L6 keeps its existing rule (already the exact gradient w.r.t. W6); dl6 comes from learn_l6.
void Organism::learn_backprop() {
    const int32_t shift = p.lr_shift + p.hidden_shift + (cur_novel ? 0 : p.lr_anneal);
    const float qs = p.err_gain * goal_w * 127.f * p.bp_gain;
    int32_t* d = delta_buf.data();
    auto live = [&](int8_t y) { const bool sat = y >= 127 || y <= -127; ++bp_outputs; bp_saturated += sat; return sat ? (p.bp_ste ? 1.f : p.bp_leak) : 1.f; };
    auto quant = [&](float g, uint32_t u, int layer, int j) {   // stochastic rounding keeps small hidden errors unbiased
        float v = g * qs; v = std::max(-127.f, std::min(127.f, v));
        const float fl = std::floor(v); const uint64_t h = mix64(event_idx * 0x9E3779B97F4A7C15ULL ^ ((uint64_t)u << 20) ^ ((uint64_t)layer << 56) ^ (uint64_t)j);
        return (int32_t)fl + ((float)(h >> 40) * (1.f / 16777216.f) < v - fl ? 1 : 0);
    };
    const bool adam = p.adam && !ad_g.empty();
    auto upd = [&](uint32_t u, int layer, uint32_t nib0, const int8_t* x, int in, const float* g, int out) {
        if (adam && (layer != LI_6 || p.adam_l6)) { adam_grad(u, nib0, x, in, g, out, goal_w); return; }
        bool any = false;
        for (int j = 0; j < out; ++j) { const int32_t q = quant(g[j], u, layer, j); any |= q != 0;
            for (int i = 0; i < in; ++i) d[j * in + i] = scale_sr(-((int32_t)x[i] * q * p.lr_num), shift); }
        if (any) apply_acc_block(u, nib0, d, in * out);
    };
    // bp_route: circuits. A routed body's L3 reads its SENDER's L2 (x3 = [l1, sender l2]); the gradient at that input is
    // passed back to the sender's L2, so a sender learns what its receivers need. Receivers follow their senders in the
    // schedule, so traces are processed last to first and each sender collects its receivers' message gradients first.
    // route_return: senders come first (forward order) because a receiver's error is what its sender needs from its return.
    const bool rr = p.route_return && rc_n.size() == ntrace;
    if (p.bp_route) bp_msg.assign((size_t)ntrace * Dims::B, 0.f);
    if (rr) rr_gret.assign((size_t)ntrace * Dims::R, 0.f);
    constexpr int32_t PC_INF = 1 << 20; const int32_t K = p.pc_settle ? (int32_t)p.pc_settle : PC_INF;
    pc_d2.assign(ntrace, PC_INF); pc_d6.assign(ntrace, PC_INF); pc_dmsg.assign(ntrace, PC_INF);
    auto cut = [&](int32_t dd) { return dd > K ? 0.f : 1.f; };
    float gword[48] = {0}; bool gw_any = false;   // in_words: d loss / d (word input cells 64..111), summed over bodies
    float gws_in[Dims::F] = {0};                  // tc_window: d loss / d (workspace input cells), summed over reading bodies
    float strm[Dims::B] = {0};                    // motif_stream: d loss / d (the stream's first B cells of every earlier body's L6), from the later bodies
    float strmq[8][Dims::B] = {{0}};              // motif_mix: the same per sender position (each receiver weights its senders)
    for (uint32_t k = 0; k < ntrace; ++k) {
        const uint32_t t = rr ? k : ntrace - 1 - k;
        const Trace& tr = traces[t];
        const bool out = (tr.reached & R_L6) && tr.emitted;
        const bool msg = p.bp_route && (tr.reached & R_L23);   // may carry gradient from receivers even if it never emitted
        const bool recv = rr && tr.src >= 0 && (uint32_t)tr.src < t && (tr.reached & R_L4);
        if (!out && !msg && !recv) continue;
        const uint32_t u = tr.unit; const uint8_t* W = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
        const bool exact = unit_lock[u] != UINT32_MAX && tree.locks[unit_lock[u]].mask == ~0ULL;
        const float gain = exact ? (float)p.exact_gain : 1.f;
        float g6[Dims::F], g5[Dims::E] = {0}, g2[Dims::B] = {0}, g3[Dims::C] = {0}, g1[Dims::A] = {0};
        float gout[Dims::F], gl4[Dims::R] = {0};
        if (out && !dl6m.empty()) { const float* dm = &dl6m[(size_t)motif_of(u) * Dims::F]; const float ws1 = ws_writer(u) ? 1.f : 0.f; for (int j = 0; j < Dims::F; ++j) gout[j] = (ws1 * dl6[j] + dm[j]) * gain; }   // ws_split
        else for (int j = 0; j < Dims::F; ++j) gout[j] = out ? dl6[j] * gain : 0.f;
        if (!comp_w.empty() && t < comp_w.size() && out) for (int j = 0; j < Dims::F; ++j) gout[j] *= comp_w[t];   // compete: its weighted share
        if (p.fm_stream && (tr.reached & R_L6)) { if ((!mix_th.empty() || !att_W.empty()) && t < 8) { for (int j = 0; j < Dims::B; ++j) gout[j] += strmq[t][j]; }
            else for (int j = 0; j < Dims::B; ++j) gout[j] += strm[j]; }   // what the later bodies needed from this L6
        int32_t d4 = PC_INF;   // settle depth of this body's return site (a receiver: one step beyond its senders' return inputs)
        if (recv) for (int q2 = 0; q2 < std::max<int>(1, tr.nsrc); ++q2) { const int16_t s2 = tr.nsrc ? tr.srcs[q2] : tr.src; if (s2 >= 0 && (uint32_t)s2 < t) d4 = std::min(d4, std::min(pc_d2[s2], pc_d6[s2]) + 1); }
        const bool in_stream = p.fm_stream && (tr.reached & R_L6);   // motif_stream: a silent body's L6 still feeds the stream (its error arrives)
        const int32_t d6 = std::min((out || in_stream) ? 1 : PC_INF, d4 + 1), d5 = d6 + 1, d3 = d5 + 1, d2 = std::min(d5 + 1, pc_dmsg[t]), d1 = std::min(d2, d3) + 1;
        pc_d2[t] = d2; pc_d6[t] = d6;
        if (recv) {   // the error at this receiver's return (from its sender) -> through its L4 -> its L6 output
            for (int q = 0; q < Dims::R; ++q) { float gsum = 0.f; for (int k2 = 0; k2 < std::max<int>(1, tr.nsrc); ++k2) { const int16_t s = tr.nsrc ? tr.srcs[k2] : tr.src; if (s >= 0 && (uint32_t)s < t) gsum += rr_gret[(size_t)s * Dims::R + q]; }
                gl4[q] = gsum * live(tr.l4[q]) / (float)(1 << tr.sh[LI_4]) * cut(d4); }
            for (int q = 0; q < Dims::R; ++q) if (gl4[q] != 0.f) for (int j = 0; j < Dims::F; ++j) gout[j] += (float)nib_get(W, Dims::NIB_L4 + q * Dims::IN4 + j) * gl4[q];
        }
        if (p.disinh > 0.f && p.train && !dis_w.empty() && (tr.reached & R_L6)) {   // motif 6: d loss / d release = sum_j gout_j l6_j k a6 f6
            const int8_t* x1d = exact ? xin_exact : (agent_on ? xin_focus : xin); const float dd = dis_release(u, x1d);
            int64_t ix = 0; for (int i = 0; i < Dims::IN6; ++i) ix += tr.x6[i] < 0 ? -tr.x6[i] : tr.x6[i]; const float a6 = (float)ix / (float)Dims::IN6 / 127.f;
            float f6 = 1.f / (1.f + p.disinh * (1.f - dd) * a6); if (p.ff_inh > 0.f) f6 /= 1.f + p.ff_inh * a6;
            float gd = 0.f; for (int j = 0; j < Dims::F; ++j) gd += gout[j] * (float)tr.l6[j]; gd *= p.disinh * a6 * f6 * dd * (1.f - dd) * goal_w;
            float* g = &dis_g[(size_t)motif_of(u) * 5]; for (int r = 0; r < 4; ++r) g[r] += gd * (float)x1d[124 + r] * (1.f / 127.f); g[4] += gd; }
        for (int j = 0; j < Dims::F; ++j) g6[j] = gout[j] != 0.f ? gout[j] * live(tr.l6[j]) / (float)(1 << tr.sh[LI_6]) * cut(d6) : 0.f;
        for (int j = 0; j < Dims::F; ++j) if (g6[j] != 0.f) for (int i = 0; i < Dims::E; ++i) g5[i] += (float)nib_get(W, Dims::NIB_L6 + j * Dims::IN6 + i) * g6[j];
        if (out && !dl5m.empty()) { const float* d5 = &dl5m[(size_t)motif_of(u) * Dims::E]; for (int i = 0; i < Dims::E; ++i) g5[i] += d5[i] * gain; }   // ws_split5
        float ga[Dims::E] = {0}; const int8_t* x1t = exact ? xin_exact : (agent_on ? xin_focus : xin);
        if (p.apical) apical_backward(W, x1t, p.apical_shift, tr.l5, tr.x6, p.bp_ste ? 1.f : p.bp_leak, g5, ga);   // through the dendritic gate
        for (int i = 0; i < Dims::E; ++i) g5[i] *= live(tr.l5[i]) / (float)(1 << tr.sh[LI_5]) * cut(d5);
        for (int j = 0; j < Dims::E; ++j) if (g5[j] != 0.f) for (int i = 0; i < Dims::IN5; ++i) {
            const float gx = (float)nib_get(W, Dims::NIB_L5 + j * Dims::IN5 + i) * g5[j];
            if (i < Dims::B) g2[i] += gx; else g3[i - Dims::B] += gx; }
        if (p.bp_route) for (int i = 0; i < Dims::B; ++i) g2[i] += bp_msg[(size_t)t * Dims::B + i];   // what this body's receivers need from its L2
        for (int i = 0; i < Dims::C; ++i) g3[i] *= live(tr.l3[i]) / (float)(1 << tr.sh[LI_3]) * cut(d3);
        if (p.bp_route && tr.src >= 0) for (int q2 = 0; q2 < std::max<int>(1, tr.nsrc); ++q2) { const int16_t s2 = tr.nsrc ? tr.srcs[q2] : tr.src; if (s2 >= 0 && (uint32_t)s2 < t) pc_dmsg[s2] = std::min(pc_dmsg[s2], d3 + 1); }
        for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + i) * g3[j];
        if (p.fm_stream && (!mix_th.empty() || !att_W.empty())) { if (tr.src >= 0 && !diag_zero_msgs && t < 8) {   // motif_mix / attn: each sender its weight's share; the gates' gradient
            const int m = tr.src + 1; const float* pi = &mix_pi[(size_t)t * 8]; float gm[Dims::B];
            for (int i = 0; i < Dims::B; ++i) { gm[i] = 0.f; if (tr.x3[Dims::A + i] >= 127 || tr.x3[Dims::A + i] <= -127) continue;
                for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) gm[i] += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + Dims::A + i) * g3[j]; }
            float dq[8] = {0}, dbar = 0.f;
            for (int q = 0; q < m && q < 8; ++q) { if (!(traces[q].reached & R_L6)) continue; for (int i = 0; i < Dims::B; ++i) { strmq[q][i] += gm[i] * 0.5f * pi[q]; dq[q] += gm[i] * 0.5f * (float)traces[q].l6[i]; } }
            float sm[8] = {0}; { const float mres = p.mix_res, mu0 = mres / (float)std::min(m, 8);   // the softmax part of the weights (mix_res)
                for (int q = 0; q < m && q < 8; ++q) sm[q] = mres > 0.f ? (pi[q] - mu0) / (1.f - mres) : pi[q]; }
            for (int q = 0; q < m && q < 8; ++q) dbar += sm[q] * dq[q];
            if (p.train && !mix_th.empty()) for (int q = 0; q < m && q < 8; ++q) mix_g[(size_t)u * 8 + q] += (1.f - p.mix_res) * sm[q] * (dq[q] - dbar) * goal_w;
            if (p.train && !att_W.empty() && m < 8) {   // motif_attn: score = q.k / sqrt(8) + bias -> the projections' and the bias' gradients
                float qv[8], kv[8], gq[8] = {0}; att_query(m, tr.l1, qv);
                for (int q = 0; q < m && q < 8; ++q) { const float ds = (1.f - p.mix_res) * sm[q] * (dq[q] - dbar) * goal_w; att_Wg[ATT_B + (size_t)m * 8 + q] += ds;
                    if (ds == 0.f || !(traces[q].reached & R_L6)) continue; att_key(traces[q].l6, kv);
                    for (int d = 0; d < 8; ++d) { gq[d] += ds * kv[d] * 0.35355339f; const float gk = ds * qv[d] * 0.35355339f * (1.f / 127.f);
                        for (int j = 0; j < Dims::F; ++j) att_Wg[ATT_WK + (size_t)d * Dims::F + j] += gk * (float)traces[q].l6[j]; } }
                for (int d = 0; d < 8; ++d) { const float g = gq[d] * (1.f / 127.f); float* Wg = &att_Wg[(size_t)m * 8 * Dims::A + (size_t)d * Dims::A];
                    for (int i = 0; i < Dims::A; ++i) Wg[i] += g * (float)tr.l1[i]; } } } }
        else if (p.fm_stream) { if (tr.src >= 0 && !diag_zero_msgs) { const float sc = 0.5f / (float)(tr.src + 1);   // to every earlier body's L6 (the mean's share)
            for (int i = 0; i < Dims::B; ++i) { if (tr.x3[Dims::A + i] >= 127 || tr.x3[Dims::A + i] <= -127) continue; float a = 0.f;
                for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) a += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + Dims::A + i) * g3[j];
                strm[i] += a * sc; } } }
        else if (p.bp_route && tr.src >= 0 && !diag_zero_msgs)   // pass the message gradient back to every sender (summed delivery: each gets all of it)
            for (int q = 0; q < std::max<int>(1, tr.nsrc); ++q) { const int16_t s = tr.nsrc ? tr.srcs[q] : tr.src; if (s < 0 || (uint32_t)s >= t) continue;
                for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) for (int i = 0; i < Dims::B; ++i) bp_msg[(size_t)s * Dims::B + i] += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + Dims::A + i) * g3[j]; }
        for (int i = 0; i < Dims::B; ++i) g2[i] *= live(tr.l2[i]) / (float)(1 << tr.sh[LI_2]) * cut(d2);
        for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L2 + j * Dims::IN2 + i) * g2[j];
        for (int i = 0; i < Dims::A; ++i) g1[i] *= live(tr.l1[i]) / (float)(1 << tr.sh[LI_1]) * cut(d1);
        if (p.tc_window && p.in_struct && !p.in_words && !exact) for (int j = 0; j < Dims::A; ++j) if (g1[j] != 0.f) for (int i = 0; i < Dims::F; ++i) gws_in[i] += (float)nib_get(W, Dims::NIB_L1 + j * Dims::IN + 96 + i) * g1[j];
        if (p.apical && p.tc_window && !exact) for (int j = 0; j < Dims::E; ++j) if (ga[j] != 0.f) for (int r = 0; r < 4; ++r) gws_in[28 + r] += (float)nib_get(W, Dims::NIB_L4 + j * 4 + r) * ga[j];   // the gate reads the registers
        if (p.in_words) for (int j = 0; j < Dims::A; ++j) if (g1[j] != 0.f) { gw_any = true; for (int i = 0; i < 48; ++i) gword[i] += (float)nib_get(W, Dims::NIB_L1 + j * Dims::IN + 64 + i) * g1[j]; }
        if (p.persist && tr.source == SRC_PERSIST) {   // the error at this body's return inputs goes back through its own L4 into its previous byte
            float gr[Dims::R] = {0};
            for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) for (int i = 0; i < Dims::R2; ++i) gr[i] += (float)nib_get(W, Dims::NIB_L2 + j * Dims::IN2 + Dims::A + i) * g2[j];
            for (int j = 0; j < Dims::F; ++j) if (g6[j] != 0.f) for (int i = 0; i < Dims::R6; ++i) gr[Dims::R2 + i] += (float)nib_get(W, Dims::NIB_L6 + j * Dims::IN6 + Dims::E + i) * g6[j];
            persist_bptt(t, gr);
        }
        if (rr && rc_n[t]) {   // the error at this sender's return inputs (x2[A..], x6[E..]) goes to its receivers' L4
            float* gr = &rr_gret[(size_t)t * Dims::R];
            for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) for (int i = 0; i < Dims::R2; ++i) gr[i] += (float)nib_get(W, Dims::NIB_L2 + j * Dims::IN2 + Dims::A + i) * g2[j];
            for (int j = 0; j < Dims::F; ++j) if (g6[j] != 0.f) for (int i = 0; i < Dims::R6; ++i) gr[Dims::R2 + i] += (float)nib_get(W, Dims::NIB_L6 + j * Dims::IN6 + Dims::E + i) * g6[j];
        }
        // weight updates (all gradients read before any write)
        if (adam) { if (p.adam_l6 || (recv && !out)) upd(u, LI_6, Dims::NIB_L6, tr.x6, Dims::IN6, g6, Dims::F); if (recv) upd(u, LI_4, Dims::NIB_L4, tr.l6, Dims::IN4, gl4, Dims::R); }
        else if (recv) { upd(u, LI_4, Dims::NIB_L4, tr.l6, Dims::IN4, gl4, Dims::R); if (!out) upd(u, LI_6, Dims::NIB_L6, tr.x6, Dims::IN6, g6, Dims::F); }
        upd(u, LI_5, Dims::NIB_L5, tr.x5, Dims::IN5, g5, Dims::E);
        upd(u, LI_2, Dims::NIB_L2, tr.x2, Dims::IN2, g2, Dims::B);
        upd(u, LI_3, Dims::NIB_L3, tr.x3, Dims::IN3, g3, Dims::C);
        upd(u, LI_1, Dims::NIB_L1, exact ? xin_exact : (agent_on ? xin_focus : xin), Dims::IN, g1, Dims::A);
        if (p.apical) upd(u, LI_4, Dims::NIB_L4, x1t + 124, 4, ga, Dims::E);   // the gate's weights U[j][r]
        if (adam) adam_touch(u);
    }
    if (p.in_words && gw_any && p.train) {   // x = E * S, so dL/dE = S * dL/dx; the three word rows move by SGD
        const float S = (float)std::min(127, std::max(1, p.in_scale));
        for (int q = 0; q < 3; ++q) { float* e = &emb[(size_t)word_ids[q] * EMB_D]; for (uint32_t k = 0; k < EMB_D; ++k) e[k] -= p.emb_lr * S * gword[16 * q + k]; }
    }
    if (p.tc_window && p.in_struct && !p.in_words && p.train) {   // x = ws_mem * (S / ws_scale): error at ws_mem itself
        const float g = (float)std::min(127, std::max(1, p.in_scale)) / std::max(ws_scale, 1.f) * p.ws_in_gain;
        float greg[4] = {0, 0, 0, 0};
        if (p.ws_regs) for (int r = 0; r < 4; ++r) { greg[r] = gws_in[28 + r] * p.reg_scale / 100.f + (p.reg_readout ? greg_ro[r] : 0.f); gws_in[28 + r] = 0.f; }
        if (!p.ws_mem_in) std::fill(gws_in, gws_in + Dims::F, 0.f);   // the workspace is not read by anyone: no credit through it (registers kept above)
        for (int i = 0; i < Dims::F; ++i) gws_in[i] *= p.ws_bands ? (float)std::min(127, std::max(1, p.in_scale)) / std::max(ws_bscale[i >> 3], 1.f) : g;
        std::memcpy(tc_greg, greg, sizeof greg);
        temporal_credit(gws_in);
    }
    if (rr && p.bp_route) for (uint32_t t = 0; t < ntrace; ++t) {   // deferred: what receivers needed from this sender's L2 (-> its L2 and L1)
        const Trace& tr = traces[t]; if (!(tr.reached & R_L23)) continue;
        float g2[Dims::B], g1[Dims::A] = {0}; bool any = false;
        for (int i = 0; i < Dims::B; ++i) { g2[i] = bp_msg[(size_t)t * Dims::B + i] * live(tr.l2[i]) / (float)(1 << tr.sh[LI_2]) * cut(pc_dmsg[t]); any |= g2[i] != 0.f; }
        if (!any) continue;
        const uint32_t u = tr.unit; const uint8_t* W = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
        const bool exact = unit_lock[u] != UINT32_MAX && tree.locks[unit_lock[u]].mask == ~0ULL;
        for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L2 + j * Dims::IN2 + i) * g2[j];
        for (int i = 0; i < Dims::A; ++i) g1[i] *= live(tr.l1[i]) / (float)(1 << tr.sh[LI_1]) * cut(pc_dmsg[t] + 1);
        upd(u, LI_2, Dims::NIB_L2, tr.x2, Dims::IN2, g2, Dims::B);
        upd(u, LI_1, Dims::NIB_L1, exact ? xin_exact : (agent_on ? xin_focus : xin), Dims::IN, g1, Dims::A);
    }
}
// tc_window: ws_mem(t) = sum_{k>=1} d^(k-1) ws(t-k), d = ws_decay/256, and ws(t-k) = sum of the writers' gain * L6. So the error
// this event's readers have at ws_mem reaches the writers of event t-k scaled by d^(k-1). Accumulate it in the ring; when an
// event leaves the window its writers take one backward pass with it (weights may have moved a little since: truncated,
// delayed credit, one extra pass per writing body).
void Organism::temporal_credit(const float* gws) {
    const uint32_t Wn = p.tc_window; const float d = (float)p.ws_decay / 256.f;
    if (tc_ring.size() != Wn) { tc_ring.assign(Wn, {}); tc_head = 0; tc_fill = 0; }
    float s[Dims::F]; for (int i = 0; i < Dims::F; ++i) s[i] = 1.f;   // d^(k-1) per dimension (ws_bands: each band's own decay)
    float sr[4] = {1.f, 1.f, 1.f, 1.f};                                  // ws_regs: survival of event t-k's write to the read at t
    for (uint32_t k = 1; k <= tc_fill; ++k) { TcEvent& e = tc_ring[(tc_head + Wn - k) % Wn]; for (int i = 0; i < Dims::F; ++i) { e.tg[i] += s[i] * gws[i]; s[i] *= p.ws_bands ? (float)ws_decay_of(i) / 256.f : d; }
        if (p.ws_regs == 2) {   // d loss / d (write of event e) = survival x error; head and writers learn from it now
            for (int r = 0; r < 4; ++r) { const float dlt = sr[r] * tc_greg[r]; if (dlt == 0.f) { sr[r] *= 1.f - e.G[r]; continue; }
                const float dg = dlt * (e.hv[r] - e.reg_prev[r]) * e.hg[r] * (1.f - e.hg[r]), dv = dlt * e.hg[r] * 100.f * (1.f - std::tanh(e.hu[r]) * std::tanh(e.hu[r]));
                for (int j = 0; j < Dims::F; ++j) { e.tg[j] += (hw_g[r][j] * dg + hw_v[r][j] * dv) * e.hs;   // to the writers, through ws
                    if (p.ro_adam) { if (hd_g.empty()) { hd_g.assign(4 * Dims::F * 2 + 4, 0.f); hd_m = hd_g; hd_v = hd_g; } hd_g[(size_t)r * Dims::F + j] += dg * e.ha[j]; hd_g[(size_t)(4 + r) * Dims::F + j] += dv * e.ha[j]; }
                    else { hw_g[r][j] -= p.head_lr * dg * e.ha[j]; hw_v[r][j] -= p.head_lr * dv * e.ha[j]; } }
                if (p.ro_adam && !hd_g.empty()) hd_g[8 * Dims::F + r] += dg; else hb_g[r] -= p.head_lr * dg; sr[r] *= 1.f - e.G[r]; } }
        else if (p.ws_regs) for (int r = 0; r < 4; ++r) { e.tg_reg[r] += sr[r] * tc_greg[r]; sr[r] *= 1.f - e.G[r]; } }
    TcEvent& slot = tc_ring[tc_head];
    if (tc_fill == Wn) temporal_backprop(slot); else ++tc_fill;
    slot.tr.clear(); slot.exact.clear();
    for (uint32_t t = 0; t < ntrace; ++t) { const Trace& tr = traces[t]; if (!tr.emitted || !(tr.reached & R_L6) || !ws_writer(tr.unit)) continue;
        slot.tr.push_back(tr); slot.exact.push_back(unit_lock[tr.unit] != UINT32_MAX && tree.locks[unit_lock[tr.unit]].mask == ~0ULL); }
    std::memcpy(slot.x1, agent_on ? xin_focus : xin, Dims::IN); std::memcpy(slot.x1e, xin_exact, Dims::IN);
    std::fill(slot.tg, slot.tg + Dims::F, 0.f); slot.goal_w = goal_w;
    for (int r = 0; r < 4; ++r) { slot.tg_reg[r] = 0.f; slot.reg_prev[r] = ws_reg[r]; slot.G[r] = 0.f; }   // this event's writes happen after learning: filled below
    tc_head = (tc_head + 1) % Wn;
}
void Organism::temporal_backprop(const TcEvent& ev) {
    bool any = false; for (int i = 0; i < Dims::F; ++i) any |= ev.tg[i] != 0.f; for (int r = 0; r < 4; ++r) any |= p.ws_regs && ev.tg_reg[r] != 0.f; if (!any) return;
    const int32_t shift = p.lr_shift + p.hidden_shift;   // no familiarity annealing: delayed credit is rare and must not be divided by 8
    const float qs = p.err_gain * ev.goal_w * 127.f * p.bp_gain * p.tc_gain;
    int32_t* d = delta_buf.data();
    auto live = [&](int8_t y) { return (y >= 127 || y <= -127) ? p.bp_leak : 1.f; };
    const bool adam = p.adam && !ad_g.empty(); const float ascale = ev.goal_w * p.tc_gain;
    auto upd = [&](uint32_t u, int layer, uint32_t nib0, const int8_t* x, int in, const float* g, int out) {
        if (adam && (layer != LI_6 || p.adam_l6)) { adam_grad(u, nib0, x, in, g, out, ascale); return; }
        bool nz = false;
        for (int j = 0; j < out; ++j) { float v = std::max(-127.f, std::min(127.f, g[j] * qs)); const float fl = std::floor(v);
            const uint64_t h = mix64(event_idx * 0x9E3779B97F4A7C15ULL ^ ((uint64_t)u << 20) ^ ((uint64_t)layer << 56) ^ (uint64_t)j ^ 0x7C0ULL);
            const int32_t q = (int32_t)fl + ((float)(h >> 40) * (1.f / 16777216.f) < v - fl ? 1 : 0); nz |= q != 0;
            for (int i = 0; i < in; ++i) d[j * in + i] = scale_sr(-((int32_t)x[i] * q * p.lr_num), shift); }
        if (nz) apply_acc_block(u, nib0, d, in * out);
    };
    for (size_t b = 0; b < ev.tr.size(); ++b) {
        const Trace& tr = ev.tr[b]; const uint32_t u = tr.unit; if (!alive[u]) continue;
        const uint8_t* W = wbuf.p + (size_t)u * Dims::UNIT_BYTES; const float gain = ev.exact[b] ? (float)p.exact_gain : 1.f;
        float g6[Dims::F], g5[Dims::E] = {0}, g2[Dims::B] = {0}, g3[Dims::C] = {0}, g1[Dims::A] = {0};
        const int32_t K = p.pc_settle ? (int32_t)p.pc_settle : (1 << 20); if (K < 5) continue;   // pc_settle: writers are 5..8 steps from their readers' error
        const float c5 = K >= 6 ? 1.f : 0.f, c23 = K >= 7 ? 1.f : 0.f, c1 = K >= 8 ? 1.f : 0.f;
        float go[Dims::F]; for (int j = 0; j < Dims::F; ++j) go[j] = ev.tg[j] * gain;
        if (p.ws_regs == 1) for (int r = 0; r < 4; ++r) if (ev.tg_reg[r] != 0.f) {   // reg = (1-G) reg_prev + sum g v: d/dv = g, d/dg = v - reg_prev
            const float g = reg_gate(tr.l6[28 + r]); go[24 + r] += ev.tg_reg[r] * g;
            if (ev.G[r] < 1.f) go[28 + r] += ev.tg_reg[r] * ((float)tr.l6[24 + r] - ev.reg_prev[r]) * (p.reg_ste ? 0.25f : g * (1.f - g)) / 32.f; }
        for (int j = 0; j < Dims::F; ++j) g6[j] = go[j] * live(tr.l6[j]) / (float)(1 << tr.sh[LI_6]);
        if (std::getenv("COCO_DEBUG_TC")) { static double s_reg = 0, s_ws = 0, s_dl6 = 0, s_greg = 0, s_q = 0; static uint64_t nb = 0, nq = 0; ++nb;
            for (int r = 0; r < 4; ++r) { s_reg += std::abs(go[24 + r]) + std::abs(go[28 + r]); s_greg += std::abs(ev.tg_reg[r]); }
            for (int j = 0; j < 24; ++j) s_ws += std::abs(go[j]); for (int j = 0; j < Dims::F; ++j) { s_dl6 += std::abs(dl6[j]); const float v = std::abs(g6[j] * qs); s_q += v; nq += v >= 1.f; }
            if (nb % 50000 == 0) std::fprintf(stderr, "tc-debug writers %llu: mean |reg-dim error| %.3g  |tg_reg| %.3g  |ws-dim error| %.3g  vs |dl6| now %.3g ; mean |q| %.3g, q>=1 on %.2f%%\n",
                (unsigned long long)nb, s_reg / nb / 8, s_greg / nb / 4, s_ws / nb / 24, s_dl6 / nb / 32, s_q / nb / 32, 100.0 * nq / nb / 32); }
        for (int j = 0; j < Dims::F; ++j) if (g6[j] != 0.f) for (int i = 0; i < Dims::E; ++i) g5[i] += (float)nib_get(W, Dims::NIB_L6 + j * Dims::IN6 + i) * g6[j];
        float ga[Dims::E] = {0}; const int8_t* x1t = ev.exact[b] ? ev.x1e : ev.x1;
        if (p.apical) apical_backward(W, x1t, p.apical_shift, tr.l5, tr.x6, p.bp_leak, g5, ga);   // through the dendritic gate
        for (int i = 0; i < Dims::E; ++i) g5[i] *= live(tr.l5[i]) / (float)(1 << tr.sh[LI_5]) * c5;
        for (int j = 0; j < Dims::E; ++j) if (g5[j] != 0.f) for (int i = 0; i < Dims::IN5; ++i) { const float gx = (float)nib_get(W, Dims::NIB_L5 + j * Dims::IN5 + i) * g5[j]; if (i < Dims::B) g2[i] += gx; else g3[i - Dims::B] += gx; }
        for (int i = 0; i < Dims::C; ++i) g3[i] *= live(tr.l3[i]) / (float)(1 << tr.sh[LI_3]) * c23;
        for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + i) * g3[j];
        for (int i = 0; i < Dims::B; ++i) g2[i] *= live(tr.l2[i]) / (float)(1 << tr.sh[LI_2]) * c23;
        for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L2 + j * Dims::IN2 + i) * g2[j];
        for (int i = 0; i < Dims::A; ++i) g1[i] *= live(tr.l1[i]) / (float)(1 << tr.sh[LI_1]) * c1;
        upd(u, LI_6, Dims::NIB_L6, tr.x6, Dims::IN6, g6, Dims::F);
        upd(u, LI_5, Dims::NIB_L5, tr.x5, Dims::IN5, g5, Dims::E);
        upd(u, LI_2, Dims::NIB_L2, tr.x2, Dims::IN2, g2, Dims::B);
        upd(u, LI_3, Dims::NIB_L3, tr.x3, Dims::IN3, g3, Dims::C);
        upd(u, LI_1, Dims::NIB_L1, ev.exact[b] ? ev.x1e : ev.x1, Dims::IN, g1, Dims::A);
        if (p.apical) upd(u, LI_4, Dims::NIB_L4, x1t + 124, 4, ga, Dims::E);
    }
}
// persist: at the end of each byte every emitting (non-exact) body computes its L4 return; the most active (mean |l4| >= persist_thresh,
// at most `persist` of them) stay active into the next byte with that return as their L2/L6 return input. A body's chain of
// steps is kept (up to persist_bptt) for credit through its own recurrence.
void Organism::persist_select() {
    std::vector<std::pair<int32_t, uint32_t>> cand;
    for (uint32_t t = 0; t < ntrace; ++t) {
        Trace& tr = traces[t]; if (!tr.emitted || !(tr.reached & R_L6)) continue;
        const uint32_t u = tr.unit; if (unit_lock[u] != UINT32_MAX && tree.locks[unit_lock[u]].mask == ~0ULL) continue;
        const uint8_t* W = wbuf.p + (size_t)u * Dims::UNIT_BYTES; LayerDesc d = L[LI_4]; kern.run(W, d, tr.l6, tr.l4); tr.sh[LI_4] = d.shift;
        int32_t lv = 0; for (int i = 0; i < Dims::R; ++i) lv += tr.l4[i] < 0 ? -tr.l4[i] : tr.l4[i]; lv /= Dims::R;
        PChain& ch = pchain[u];
        if (tr.source != SRC_PERSIST) { ch.steps.clear(); ch.born = block_idx * 0 + (uint32_t)(event_idx & 0xFFFFFFFF); }   // a fresh life
        PStep ps; ps.tr = tr; std::memcpy(ps.x1, agent_on ? xin_focus : xin, Dims::IN); ch.steps.push_back(ps);
        if (ch.steps.size() > p.persist_bptt) ch.steps.erase(ch.steps.begin());
        const uint32_t age = (uint32_t)(event_idx & 0xFFFFFFFF) - ch.born;
        if (lv >= p.persist_thresh && age < p.persist_age) cand.emplace_back(lv, u);
    }
    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
    persist_next.clear(); for (size_t i = 0; i < cand.size() && i < p.persist; ++i) persist_next.push_back(cand[i].second);
    for (uint32_t t = 0; t < ntrace; ++t) { const Trace& tr = traces[t]; if (!tr.emitted || !(tr.reached & R_L6)) continue;   // only a persisting body carries a return
        const bool keep = std::find(persist_next.begin(), persist_next.end(), tr.unit) != persist_next.end();
        if (keep) std::memcpy(&l4ret[(size_t)tr.unit * Dims::R], tr.l4, Dims::R); else std::memset(&l4ret[(size_t)tr.unit * Dims::R], 0, Dims::R); }
    if (pchain.size() > 4 * (size_t)p.persist + 64) {   // forget chains of bodies that are no longer active
        std::vector<uint32_t> drop; for (auto& kv : pchain) if (std::find(persist_next.begin(), persist_next.end(), kv.first) == persist_next.end()) drop.push_back(kv.first);
        for (uint32_t u : drop) pchain.erase(u);
    }
}
// gret: d loss / d (this byte's return input of trace t's body) = d loss / d (its L4 output one byte earlier). Walk back through the
// body's stored steps: L4 -> L6 -> L5 -> L2/L3 -> L1 of each earlier byte, and the earlier byte's own return input onward.
void Organism::persist_bptt(uint32_t t, const float* gret) {
    const uint32_t u = traces[t].unit; auto it = pchain.find(u); if (it == pchain.end() || it->second.steps.empty()) return;
    auto& steps = it->second.steps;
    const int32_t shift = p.lr_shift + p.hidden_shift; const float qs = p.err_gain * goal_w * 127.f * p.bp_gain * p.tc_gain;
    int32_t* d = delta_buf.data();
    auto live = [&](int8_t y) { return (y >= 127 || y <= -127) ? p.bp_leak : 1.f; };
    const bool adam = p.adam && !ad_g.empty();
    auto upd = [&](int layer, uint32_t nib0, const int8_t* x, int in, const float* g, int out) {
        if (adam && (layer != LI_6 || p.adam_l6)) { adam_grad(u, nib0, x, in, g, out, goal_w * p.tc_gain); return; }
        bool nz = false;
        for (int j = 0; j < out; ++j) { float v = std::max(-127.f, std::min(127.f, g[j] * qs)); const float fl = std::floor(v);
            const int32_t q = (int32_t)fl + (sr_uniform() < v - fl ? 1 : 0); nz |= q != 0;
            for (int i = 0; i < in; ++i) d[j * in + i] = scale_sr(-((int32_t)x[i] * q * p.lr_num), shift); }
        if (nz) apply_acc_block(u, nib0, d, in * out);
    };
    const uint8_t* W = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
    float gr[Dims::R]; std::memcpy(gr, gret, sizeof gr);
    for (int k = (int)steps.size() - 1; k >= 0; --k) {   // the most recent stored step is the previous byte
        const Trace& tr = steps[k].tr;
        float g4[Dims::R], g6[Dims::F] = {0}, g5[Dims::E] = {0}, g2[Dims::B] = {0}, g3[Dims::C] = {0}, g1[Dims::A] = {0}; bool any = false;
        for (int q = 0; q < Dims::R; ++q) { g4[q] = gr[q] * live(tr.l4[q]) / (float)(1 << tr.sh[LI_4]); any |= g4[q] != 0.f; }
        if (!any) break;
        for (int q = 0; q < Dims::R; ++q) if (g4[q] != 0.f) for (int j = 0; j < Dims::F; ++j) g6[j] += (float)nib_get(W, Dims::NIB_L4 + q * Dims::IN4 + j) * g4[q];
        for (int j = 0; j < Dims::F; ++j) g6[j] *= live(tr.l6[j]) / (float)(1 << tr.sh[LI_6]);
        for (int j = 0; j < Dims::F; ++j) if (g6[j] != 0.f) for (int i = 0; i < Dims::E; ++i) g5[i] += (float)nib_get(W, Dims::NIB_L6 + j * Dims::IN6 + i) * g6[j];
        for (int i = 0; i < Dims::E; ++i) g5[i] *= live(tr.l5[i]) / (float)(1 << tr.sh[LI_5]);
        for (int j = 0; j < Dims::E; ++j) if (g5[j] != 0.f) for (int i = 0; i < Dims::IN5; ++i) { const float gx = (float)nib_get(W, Dims::NIB_L5 + j * Dims::IN5 + i) * g5[j]; if (i < Dims::B) g2[i] += gx; else g3[i - Dims::B] += gx; }
        for (int i = 0; i < Dims::C; ++i) g3[i] *= live(tr.l3[i]) / (float)(1 << tr.sh[LI_3]);
        for (int j = 0; j < Dims::C; ++j) if (g3[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L3 + j * Dims::IN3 + i) * g3[j];
        for (int i = 0; i < Dims::B; ++i) g2[i] *= live(tr.l2[i]) / (float)(1 << tr.sh[LI_2]);
        for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) for (int i = 0; i < Dims::A; ++i) g1[i] += (float)nib_get(W, Dims::NIB_L2 + j * Dims::IN2 + i) * g2[j];
        for (int i = 0; i < Dims::A; ++i) g1[i] *= live(tr.l1[i]) / (float)(1 << tr.sh[LI_1]);
        // this step's own return input came from the step before it (if it was a persistent continuation)
        std::fill(gr, gr + Dims::R, 0.f);
        if (tr.source == SRC_PERSIST) {
            for (int j = 0; j < Dims::B; ++j) if (g2[j] != 0.f) for (int i = 0; i < Dims::R2; ++i) gr[i] += (float)nib_get(W, Dims::NIB_L2 + j * Dims::IN2 + Dims::A + i) * g2[j];
            for (int j = 0; j < Dims::F; ++j) if (g6[j] != 0.f) for (int i = 0; i < Dims::R6; ++i) gr[Dims::R2 + i] += (float)nib_get(W, Dims::NIB_L6 + j * Dims::IN6 + Dims::E + i) * g6[j];
        }
        upd(LI_4, Dims::NIB_L4, tr.l6, Dims::IN4, g4, Dims::R);
        upd(LI_6, Dims::NIB_L6, tr.x6, Dims::IN6, g6, Dims::F);
        upd(LI_5, Dims::NIB_L5, tr.x5, Dims::IN5, g5, Dims::E);
        upd(LI_2, Dims::NIB_L2, tr.x2, Dims::IN2, g2, Dims::B);
        upd(LI_3, Dims::NIB_L3, tr.x3, Dims::IN3, g3, Dims::C);
        upd(LI_1, Dims::NIB_L1, steps[k].x1, Dims::IN, g1, Dims::A);
        if (tr.source != SRC_PERSIST) break;   // the start of this body's life
    }
}
// metabolic pressure: consequence lowers resistance, useless compute raises it (design doc §7)
void Organism::metabolic(bool consequential, bool novel) {
    const uint32_t nk = n_kclass();
    for (uint32_t t = 0; t < ntrace; ++t) {
        const Trace& tr = traces[t]; int32_t r = resist[tr.unit];
        const bool cons = tr.consequential != 0;   // per-body: global for sensory admissions, counterfactual for routed ones
        if (tr.reached & R_L6) {   // the K screen learns which class pays for this kind of input
            uint32_t kc = kclass_of(tr.unit); KClassStats& ks = kstats[kc]; ++ks.emits; ks.consequential += cons;
            float& e = kscreen_tab[(size_t)hist[0] * nk + kc]; e += p.kscreen_rate * ((cons ? 1.f : 0.f) - e);
        }
        if (tr.emitted && p.goal_elig > 0.f) {                          // eligibility: this body took part in the current reply's reading
            float& e = elig[tr.unit]; e = e * std::pow(p.goal_decay, (float)(event_idx - elig_t[tr.unit])) + 1.f; elig_t[tr.unit] = event_idx;
            if (touch_stamp[tr.unit] != reply_id) { touch_stamp[tr.unit] = reply_id; if (touched.size() < 65536) touched.push_back(tr.unit); } }
        if (tr.emitted) { const uint32_t m = tr.unit / p.motif_size; if (m < m_emit_o.size()) { if (phase == 1) ++m_emit_q[m]; else if (phase == 2) ++m_emit_a[m]; else ++m_emit_o[m]; } }
        if (tr.reached & R_L6) r += cons ? -(int32_t)std::lround(p.resist_down * goal_w) : p.resist_up;
        else r += novel ? -p.resist_down : p.resist_up_dead;      // dying on a novel/surprising event: buy computation
        resist[tr.unit] = (uint8_t)std::max(p.resist_min, std::min(p.resist_max, r));
    }
}
// ---- goal-credited plasticity ----------------------------------------------------------------------------------
// The dialogue protocol marks turns ("USER:" ... "ASSISTANT:"); the reply's first token is the outcome the goal is
// judged by. phase: 0 other, 1 reading the question, 2 the answer. On answer bytes goal_w = goal_gain scales the
// organism's own signals (L6 error, resistance, lock utility, birth margin). When the answer ends, R = how much better
// than usual it was predicted (nats per byte vs a running mean) credits the eligible bodies: resistance down (easier
// to fire) and lock utility up when R > 0, the reverse when R < 0.
void Organism::goal_track(uint8_t sym, uint8_t target) {
    if (m_emit_o.empty()) { const size_t nm = p.units / p.motif_size + 1; m_emit_q.assign(nm, 0); m_emit_a.assign(nm, 0); m_emit_o.assign(nm, 0);
        if (p.goal_elig > 0.f) { elig.assign(p.units, 0.f); elig_t.assign(p.units, 0); touch_stamp.assign(p.units, 0); } }
    auto ends_with = [&](const char* s, int L) { if (event_idx < (uint64_t)L) return false; if (sym != (uint8_t)s[L - 1]) return false; for (int i = 1; i < L; ++i) if (hist[i] != (uint8_t)s[L - 1 - i]) return false; return true; };
    if (ends_with("USER:", 5)) { phase = 1; ans_state = 0; }
    else if (ends_with("ASSISTANT:", 10)) { phase = 2; ans_state = 1; ans_loss = 0.0; ans_n = 0; }
    goal_w = 1.f;
    if (ans_state) {
        const bool an = std::isalnum(target);
        if (ans_state == 1) { if (an) ans_state = 2; else if (target != ' ') ans_state = 3; }
        else if (ans_state == 2 && !an) ans_state = 3;   // the terminator byte still belongs to the answer
        else if (ans_state == 3) ans_state = 0;
        if (ans_state == 2 || ans_state == 3) goal_w = p.goal_gain;
        if (ans_state == 0 || (ans_state == 3 && !std::isalnum(target))) {
            if (ans_state == 0 && ans_n) {                  // the answer is complete: judge it, credit the readers
                const float mean = (float)(ans_loss / ans_n); if (ans_ema < 0.f) ans_ema = mean;
                const float R = ans_ema - mean; ans_ema += 0.02f * (mean - ans_ema);
                if (p.train && p.goal_elig > 0.f) goal_credit(R);
                ans_n = 0; phase = 0;
            }
        }
    }
    if (phase == 1) ++bytes_q; else if (phase == 2) ++bytes_a; else ++bytes_o;
}
void Organism::goal_credit(float R) {
    ++goal_credits; goal_R_sum += R;
    for (uint32_t u : touched) { if (!alive[u]) continue; const float e = elig[u] * std::pow(p.goal_decay, (float)(event_idx - elig_t[u])); if (e < 0.01f) continue;
        const float c = p.goal_elig * R * e;
        resist[u] = (uint8_t)std::max(p.resist_min, std::min(p.resist_max, (int32_t)resist[u] - (int32_t)std::lround(c * p.resist_down)));
        if (unit_lock[u] != UINT32_MAX) tree.locks[unit_lock[u]].utility += 0.1f * c; }
    touched.clear(); ++reply_id;
}
std::string Organism::goal_report() const {
    if (m_emit_o.empty()) return "";
    uint64_t q = 0, a = 0, o = 0; for (size_t m = 0; m < m_emit_o.size(); ++m) { q += m_emit_q[m]; a += m_emit_a[m]; o += m_emit_o[m]; }
    const double base = (double)(bytes_q + bytes_a) / std::max<uint64_t>(1, bytes_q + bytes_a + bytes_o);
    int spec = 0; for (size_t m = 0; m < m_emit_o.size(); ++m) { const double tot = m_emit_q[m] + m_emit_a[m] + m_emit_o[m]; if (tot >= 200 && (m_emit_q[m] + m_emit_a[m]) / tot > 3 * base) ++spec; }
    char b[400]; std::snprintf(b, sizeof b, "goal: credits %llu (mean R %.3f nats/byte), emissions question %llu answer %llu other %llu; question+answer share of bytes %.3f; motifs >3x specialised to question+answer: %d",
        (unsigned long long)goal_credits, goal_credits ? goal_R_sum / goal_credits : 0.0, (unsigned long long)q, (unsigned long long)a, (unsigned long long)o, base, spec);
    return b;
}
std::vector<uint32_t> Organism::specialised_motifs(int k, bool random, uint64_t seed) const {
    std::vector<std::pair<double, uint32_t>> c; const double base = (double)(bytes_q + bytes_a) / std::max<uint64_t>(1, bytes_q + bytes_a + bytes_o);
    for (size_t m = 0; m < m_emit_o.size(); ++m) { const double tot = m_emit_q[m] + m_emit_a[m] + m_emit_o[m]; if (tot < 200) continue; c.emplace_back(random ? (double)(mix64(seed + m) >> 11) : (m_emit_q[m] + m_emit_a[m]) / tot / std::max(base, 1e-9), (uint32_t)m); }
    std::sort(c.begin(), c.end(), [](auto& x, auto& y) { return x.first > y.first; });
    std::vector<uint32_t> r; for (int i = 0; i < k && i < (int)c.size(); ++i) r.push_back(c[i].second); return r;
}
// ---- composer ---------------------------------------------------------------------------------------------------
// Units above bytes, found by the organism's own surprise. Level 1 accumulates bytes; a unit closes when the
// organism's surprise about the NEXT byte (-log of its most likely continuation) rises above its running mean x
// lvl_margin: where the continuation stops being predictable, a new unit begins (words, mostly, without being told
// what a word is). Level l+1 does the same over level-l units, with each level's own next-unit counts as its
// predictor. A closing unit is an EVENT at its level: its identity and its predecessors' form the key into the level's
// own address tree and the input of the level's own bodies (reserved motifs: they never see a byte); their L6 output
// is that level's workspace, added to every byte's workspace until the level's next unit (the unit fed back as a
// whole, and through the persistent workspace into the byte bodies' input). Credit: the byte readout's error over the
// bytes a level output served is averaged and applied to the bodies that produced it, by the same L6 rule.
void Organism::composer_init() {
    nlv = (int)std::min<uint32_t>(p.levels, MAXLV); if (!nlv || !p.kpack) { nlv = 0; return; }
    unit_level.assign(p.units, 0); lvl_w.assign((size_t)nlv * p.vocab * Dims::F, 0.f); lvl_logits.assign(p.vocab, 0.f);
    const uint32_t nm = (uint32_t)motif_free.size();
    for (int l = 1; l <= nlv; ++l) {
        Level& L = lv[l]; L.tree.init(p.leaf_max); L.next.assign(1 << 16, Level::Next{});
        L.ch_a.assign(256, 0); L.ch_b.assign(256, 0); L.standalone.assign(256, 0); L.ulen.assign(256, 1);
        for (uint32_t k = 0; k < p.lvl_motifs; ++k) { const uint32_t m = nm - 1 - (uint32_t)(l - 1) * p.lvl_motifs - k; if (m >= nm || (int)m < 0) break;
            for (uint32_t u : motif_free[m]) { L.pool.push_back(u); unit_level[u] = (uint8_t)l; } motif_free[m].clear(); }
    }
}
void Organism::composer_step(uint8_t sym, float pmax) {
    if (p.lvl_warm && p.seq_units && seq_on && !seq.unit_tokens && event_idx >= p.lvl_warm) seq.unit_tokens = true;   // the vocabulary has settled: the memory side switches to units
    (void)pmax; feed_unit(1, sym, n_bytes_seen++, 1, std::string(1, (char)sym));
}
// One unit arriving at level l. Units start as single bytes; a pair that keeps recurring becomes a unit of its own
// (lvl_minc times), and merges are applied greedily as the stream arrives, so the same text always yields the same
// units: identity is stable, which is what the memory side needs. Merges of merges give the levels their hierarchy.
void Organism::feed_unit(int l, uint32_t id, uint64_t start, uint32_t len, const std::string& text) {
    if (l > nlv) return;
    Level& L = lv[l];
    L.pend.push_back(Level::Pend{id, start, len, text});
    // apply the learned merges by RANK: the earliest-learned (most frequent) applicable pair in the window wins, and
    // merging repeats until none applies. Arrival order no longer decides where a unit breaks.
    for (;;) {
        size_t best_i = SIZE_MAX; uint32_t best_rank = 0xFFFFFFFFu;
        for (size_t i = 0; i + 1 < L.pend.size(); ++i) {
            auto m = L.merge.find(((uint64_t)L.pend[i].id << 32) | L.pend[i + 1].id);
            if (m != L.merge.end() && m->second < best_rank) { best_rank = m->second; best_i = i; }
        }
        if (best_i == SIZE_MAX) break;
        Level::Pend& x = L.pend[best_i]; const Level::Pend& y = L.pend[best_i + 1];
        x.id = best_rank; x.len += y.len; if (x.str.size() < 64) x.str += y.str;
        L.pend.erase(L.pend.begin() + best_i + 1);
    }
    while (L.pend.size() > 24) {                      // out of the window: this unit is settled
        const Level::Pend f = L.pend.front(); L.pend.erase(L.pend.begin());
        if (L.pend.size() >= 1) { const uint64_t key = ((uint64_t)f.id << 32) | L.pend.front().id; ++L.pair_n[key]; }
        if (++L.since_round >= 65536) {               // a round of merges: the pairs that recurred most become units
            L.since_round = 0;
            std::vector<std::pair<uint32_t, uint64_t>> top;
            for (auto& kv : L.pair_n) if (kv.second >= p.lvl_minc && !L.merge.count(kv.first)) top.emplace_back(kv.second, kv.first);
            std::sort(top.begin(), top.end(), [](auto& a2, auto& b2) { return a2.first > b2.first; });
            const size_t take = std::min<size_t>(top.size(), 128);
            for (size_t i = 0; i < take && L.next_id < 60000 && L.merge.size() < 40000; ++i) {
                const uint32_t ca = (uint32_t)(top[i].second >> 32), cb = (uint32_t)top[i].second;
                L.ch_a.push_back(ca); L.ch_b.push_back(cb); L.standalone.push_back(0);
                L.ulen.push_back((uint16_t)std::min<uint32_t>(65535, (uint32_t)L.ulen[ca] + L.ulen[cb]));
                L.merge[top[i].second] = L.next_id++; }
            for (auto& kv : L.pair_n) kv.second = (kv.second + 1) / 2;
            ++L.rounds;
        }
        close_unit(l, ((uint64_t)f.id << 1) | 1ULL, f.id, f.str, f.start, f.len);
    }
}
// A unit has settled: the memory side gets its parts, the level runs its own bodies on it, and it becomes one
// unit of the level above.
void Organism::close_unit(int l, uint64_t id, uint32_t raw_id, const std::string& str_in, uint64_t start, uint32_t len) {
    const std::string str = str_in;
    Level& L = lv[l];
    if (seq_on && seq.unit_tokens) {   // the memory side thinks in composed units, not in hand-cut words
        if (l == 1) { const uint64_t st = seq.n > (uint64_t)len + 1 ? seq.n - 1 - len : 0; emit_pieces(L, raw_id, str, 0, st, 0); }
        else if (l == 2) seq.unit_sentence_end(); }
    if (raw_id < L.standalone.size() && L.standalone[raw_id] < 4000000000u) ++L.standalone[raw_id];
    L.units_len += len; if (L.sample.size() < 400) L.sample.push_back(str); else L.sample[L.events % 400] = str;
    level_event(l, id);
    if (l >= nlv) return;
    feed_unit(l + 1, raw_id, start, len, str);
}
// A unit splits into its two parts only when both parts occur on their own often enough; splitting then recurses.
// The rule depends on the merge tree and the counts alone, so the same text always decomposes the same way: the
// question's [Bill in the ] yields [Bill ] and [in the ], and the story's [Bill ] stays [Bill ] — they match again.
void Organism::emit_pieces(Level& L, uint32_t id, const std::string& str, uint32_t off, uint64_t start, int depth) {
    const uint32_t whole = id < L.standalone.size() ? L.standalone[id] : 0;
    auto stands = [&](uint32_t x) { return x >= 256 && x < L.standalone.size() && L.standalone[x] >= p.lvl_minc && L.standalone[x] >= 2 * whole; };
    if (depth < 8 && id >= 256 && id < L.ch_a.size()) {
        const uint32_t a = L.ch_a[id], b = L.ch_b[id];
        if (stands(a) && stands(b)) {
            emit_pieces(L, a, str, off, start, depth + 1);
            emit_pieces(L, b, str, off + L.ulen[a], start, depth + 1);
            return;
        }
    }
    const uint32_t ln = id < L.ulen.size() ? L.ulen[id] : 1;
    if (off + ln <= str.size()) seq.unit_close(str.substr(off, ln), start + off, free_run);
    else if (off < str.size()) seq.unit_close(str.substr(off), start + off, free_run);
}
void Organism::level_learn(Level& L) {                  // the averaged byte-readout error over the bytes this output served
    float g[Dims::F]; for (int j = 0; j < Dims::F; ++j) g[j] = L.acc[j] / (float)L.acc_n;
    int8_t q[Dims::F]; for (int j = 0; j < Dims::F; ++j) { float v = g[j] * p.err_gain * 127.f; v = std::max(-127.f, std::min(127.f, v)); q[j] = (int8_t)(int)(v + (v >= 0.f ? 0.5f : -0.5f)); }
    const int32_t shift = p.lr_shift; int32_t* d = delta_buf.data();
    for (const Trace& tr : L.tr) { if (!(tr.reached & R_L6) || !alive[tr.unit]) continue;
        for (int j = 0; j < Dims::F; ++j) for (int i = 0; i < Dims::IN6; ++i) d[j * Dims::IN6 + i] = scale_trunc(-((int32_t)tr.x6[i] * q[j] * p.lr_num), shift);
        apply_acc_block(tr.unit, Dims::NIB_L6, d, Dims::P6); }
}
void Organism::level_event(int l, uint64_t id) {
    Level& L = lv[l]; ++L.events;
    if (p.train && !L.tr.empty() && L.acc_n) level_learn(L);
    std::memset(L.acc, 0, sizeof L.acc); L.acc_n = 0;
    L.ids[2] = L.ids[1]; L.ids[1] = L.ids[0]; L.ids[0] = id;
    // key: this unit's identity (low 12 bits: the birth mask), its predecessor, and a hash of the two before
    const uint64_t key = (id & 0xFFFULL) | ((L.ids[1] & 0xFFFFFULL) << 12) | (mix64(L.ids[1] * 31 + L.ids[2]) & 0xFFFFFFFF00000000ULL);
    for (int b = 0; b < 64; ++b) xin[b] = (int8_t)(((key >> b) & 1) ? IN_SCALE : -IN_SCALE);
    const uint64_t code = mix64(id ^ 0xC0DE5EEDULL); for (int b = 0; b < 32; ++b) xin[64 + b] = (int8_t)(((code >> b) & 1) ? IN_SCALE : -IN_SCALE);
    for (int j = 0; j < Dims::F; ++j) xin[96 + j] = clamp8(L.mem[j] >> 3);   // the level's own persistent workspace: context at this level only
    std::memcpy(xin_exact, xin, Dims::IN); if (agent_on) std::memcpy(xin_focus, xin, Dims::IN);
    uint32_t depth; int32_t leaf = L.tree.walk(key, depth); uint32_t cand[16]; uint32_t ncand = L.tree.match(leaf, key, cand, 16);
    if (!ncand && p.structural && p.train && !L.pool.empty()) {   // a new unit identity at this level: a body of its own
        const uint32_t u = L.pool.back(); L.pool.pop_back(); alloc_unit(u); birth_unit_weights(u); unit_lock[u] = UINT32_MAX;
        cand[ncand++] = L.tree.new_lock(key, 0xFFFULL, u, block_idx); ++L.births; }
    cur.clear(); accepted.clear(); executed.clear(); bool novel = false;
    for (uint32_t i = 0; i < ncand; ++i) { Lock& lk = L.tree.locks[cand[i]]; if (lk.hits < p.novel_hits) novel = true; ++lk.hits; lk.last_hit_block = block_idx; cur.push_back(Event{lk.unit, SRC_SENSORY, 0, -1}); }
    cur_level = l; run_pass(key, novel, 0, nullptr); cur_level = 0;
    std::memcpy(L.ws, ws, sizeof ws);
    for (int j = 0; j < Dims::F; ++j) L.mem[j] = ((L.mem[j] * (p.ws_decay ? p.ws_decay : 192)) >> 8) + ws[j];
    L.tr.assign(traces.begin(), traces.begin() + ntrace);
}
std::string Organism::composer_report() const {
    if (!nlv) return "";
    std::string r = "composer:";
    for (int l = 1; l <= nlv; ++l) { const Level& L = lv[l]; char b[200];
        std::snprintf(b, sizeof b, " L%d %llu units (mean %.1f %s, %zu merges learned), %llu bodies, %zu locks;", l, (unsigned long long)L.events, L.events ? (double)L.units_len / L.events : 0.0, l == 1 ? "bytes" : "lower units", L.merge.size(), (unsigned long long)L.births, L.tree.live_locks()); r += b; }
    if (seq_on && seq.unit_tokens) { r += " thoughts:"; size_t n0 = seq.unit_sample.size() > 20 ? seq.unit_sample.size() - 20 : 0; for (size_t i = n0; i < seq.unit_sample.size(); ++i) { std::string x = seq.unit_sample[i]; for (auto& c : x) if (c == 10) c = 47; r += " [" + x + "]"; } }
    for (int l = 1; l <= nlv; ++l) { r += "\n  L" + std::to_string(l) + " sample:"; const Level& L = lv[l]; size_t n0 = L.sample.size() > 24 ? L.sample.size() - 24 : 0;
        for (size_t i = n0; i < L.sample.size(); ++i) { std::string s = L.sample[i]; for (auto& c : s) if (c == '\n') c = '/'; r += " [" + s + "]"; } }
    return r;
}
void Organism::assoc_update(bool novel, bool consequential) {
    if (p.route_ungated) return;   // links are not rewired by a receiver's own consequence
    for (const AcceptedMsg& am : accepted) {
        uint32_t v = am.receiver;
        bool helped = false; for (uint32_t t = 0; t < ntrace; ++t) if (traces[t].unit == v && (traces[t].reached & R_L6)) { helped = traces[t].consequential != 0; break; }
        if (helped) { assoc_noconseq[v] = 0; continue; }
        if (++assoc_noconseq[v] >= p.assoc_tighten_after) {           // raise resistance: one more tumbler on the assoc lock
            assoc_noconseq[v] = 0; uint64_t free_bits = ~assoc_mask[v]; if (!free_bits) continue;
            uint64_t pick = 0; for (int tries = 0; tries < 8; ++tries) { pick = 1ULL << rng.below(64); if (free_bits & pick) break; pick = 0; }
            if (pick) { assoc_mask[v] |= pick; assoc_value[v] ^= pick; }   // flip value there so this useless sender is rejected
        }
    }
    if (!novel) return;
    for (uint32_t u : executed) {
        bool got = false; for (auto& am : accepted) if (am.receiver == u) { got = true; break; }
        if (got) { assoc_starve[u] = 0; continue; }
        if (++assoc_starve[u] >= p.assoc_loosen_after && assoc_mask[u]) { assoc_starve[u] = 0; int b = rng.below(64); uint64_t m = assoc_mask[u]; while (!((m >> b) & 1)) b = (b + 1) & 63; assoc_mask[u] &= ~(1ULL << b); }
    }
}

// ---------------------------------------------------------------------------------------------
// block driver and structural maintenance (runbook §9: splits commit between blocks)
void Organism::run_block(const uint8_t* s, size_t n) {
    auto t0 = std::chrono::steady_clock::now();
    if (n < 2) return;
    if (event_idx == 0) hist[0] = s[0];
    for (size_t i = 0; i + 1 < n; ++i) process_event(s[i], s[i + 1]);
    st.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
// ALM-style admission: children of a conflicted lock on (lag receptor, previous symbol), admitted after a few
// confirmations when they gain bits over the parent. The child copies the parent's body and lives beside it.
void Organism::admit_children(int32_t s, uint32_t parent) {
    std::vector<std::pair<uint32_t, float>> cands; evid.nursery_candidates(s, p.admit_min, p.market ? std::max(p.admit_bits, mk_bar < 0.f ? p.market_bar : mk_bar) : p.admit_bits, cands);
    if (cands.empty()) return;
    std::sort(cands.begin(), cands.end(), [](auto& a, auto& b) { return a.second > b.second; });
    const int lag = evid.slots[s].lag; const uint64_t lagbits = 0xFFULL << (8 * lag);
    uint32_t admitted = 0;
    for (auto& c : cands) {
        if (admitted >= p.admit_max_per_lock) break;
        const Lock& pl = tree.locks[parent];
        uint64_t value = (pl.value & ~lagbits) | ((uint64_t)c.first << (8 * lag)), mask = pl.mask | lagbits;
        if (p.market && p.market_contest && !market_make_room(c.second * std::log1p(market_reuse(pl, parent)))) break;
        uint32_t u = alloc_unit_near(pl.unit); if (u == UINT32_MAX) break;
        std::memcpy(wbuf.p + (size_t)u * Dims::UNIT_BYTES, wbuf.p + (size_t)pl.unit * Dims::UNIT_BYTES, Dims::UNIT_BYTES);
        if (!acc.empty()) std::memcpy(&acc[(size_t)u * Dims::UNIT_NIBBLES], &acc[(size_t)pl.unit * Dims::UNIT_NIBBLES], Dims::UNIT_NIBBLES);
        if (!acc16.empty()) std::memcpy(&acc16[(size_t)u * Dims::UNIT_NIBBLES], &acc16[(size_t)pl.unit * Dims::UNIT_NIBBLES], Dims::UNIT_NIBBLES * sizeof(int16_t));
        resist[u] = resist[pl.unit];
        std::memcpy(&act_ema[(size_t)u * 4], &act_ema[(size_t)pl.unit * 4], 4);
        uint32_t nid = tree.new_lock(value, mask, u, block_idx); unit_lock[u] = nid; tree.locks[nid].hits = p.novel_hits; tree.locks[nid].utility = c.second;
        ++admitted; ++st.nursery_admissions;
        if (p.market) { market_new_lock(nid, true); if (parent < mk_last_struct.size()) mk_last_struct[parent] = block_idx; }
    }
    // the parent keeps collecting for the remaining symbols; clear the admitted rows
    for (uint32_t i = 0; i < admitted; ++i) { uint32_t sy = cands[i].first; std::fill(evid.slots[s].nursery.begin() + (size_t)sy * evid.buckets, evid.slots[s].nursery.begin() + (size_t)(sy + 1) * evid.buckets, 0); evid.slots[s].nursery_n[sy] = 0; }
}
// When the pool runs short, generic locks whose utility has gone negative are recycled (ALM: low-utility slots
// are recyclable once past a minimum age).
void Organism::recycle_locks() {
    uint64_t reserve = std::max<uint64_t>((uint64_t)p.units * p.unit_reserve_ppm / 1000000ULL, 2ULL * p.block);
    if (free_count() >= reserve) return;
    std::vector<std::pair<float, uint32_t>> c;
    for (uint32_t id = 0; id < tree.locks.size(); ++id) { const Lock& lk = tree.locks[id]; if (lk.alive && lk.mask != ~0ULL && lk.mask != SENSORY_MASK && lk.utility < p.recycle_utility && block_idx - lk.born_block >= p.recycle_min_age) c.emplace_back(lk.utility, id); }
    size_t need = std::min(c.size(), (size_t)(2 * reserve) - free_count());
    std::nth_element(c.begin(), c.begin() + need, c.end());
    for (size_t i = 0; i < need; ++i) { Lock& lk = tree.locks[c[i].second]; if (lk.evidence >= 0) { evid.release(lk.evidence); lk.evidence = -1; } free_unit(lk.unit); tree.kill_lock(c[i].second); ++st.lock_recycled; }
}
void Organism::commit_splits() {
    const float thr = p.split_g_per_df * (float)(p.future_buckets - 1);
    for (uint32_t s = 0; s < evid.slots.size(); ++s) {
        uint32_t id = evid.slots[s].lock; if (id == UINT32_MAX) continue;
        if (!tree.locks[id].alive) { evid.release((int32_t)s); continue; }
        if (p.market && market_cooling(id)) { ++mk_totals.cooldown_skips; continue; }   // learn before cloning
        if (p.nursery) admit_children((int32_t)s, id);
        Lock& lk = tree.locks[id];
        float g; int b = evid.best_split((int32_t)s, p.split_min_evidence, g);
        if (b >= 0 && g > thr) {
            if (p.market && p.market_contest && !market_make_room(lk.utility * std::log1p(market_reuse(lk, id) / 2.f))) continue;
            uint32_t u = alloc_unit_near(lk.unit);      // a split child lives next to its parent
            if (u == UINT32_MAX) { lk.evidence = -1; evid.release((int32_t)s); continue; }
            uint64_t bit = 1ULL << (b + Dims::SENSORY_BITS);
            // child B: same route body copied, bit=1; parent keeps bit=0. Both keep their leaves (bit was unmasked).
            std::memcpy(wbuf.p + (size_t)u * Dims::UNIT_BYTES, wbuf.p + (size_t)lk.unit * Dims::UNIT_BYTES, Dims::UNIT_BYTES);
            if (!acc.empty()) std::memcpy(&acc[(size_t)u * Dims::UNIT_NIBBLES], &acc[(size_t)lk.unit * Dims::UNIT_NIBBLES], Dims::UNIT_NIBBLES);
            if (!acc16.empty()) std::memcpy(&acc16[(size_t)u * Dims::UNIT_NIBBLES], &acc16[(size_t)lk.unit * Dims::UNIT_NIBBLES], Dims::UNIT_NIBBLES * sizeof(int16_t));
            resist[u] = resist[lk.unit]; std::memcpy(&act_ema[(size_t)u * 4], &act_ema[(size_t)lk.unit * 4], 4);
            lk.mask |= bit; lk.value &= ~bit; lk.err_ema = 0; lk.evidence = -1;
            uint32_t nid = tree.new_lock(lk.value | bit, lk.mask, u, block_idx);
            unit_lock[u] = nid;
            if (p.market) { market_new_lock(nid, true); mk_last_struct[id] = block_idx; }
            evid.release((int32_t)s); ++st.lock_splits;
            if (id < promo_hold.size()) promo_hold[id] = 0;
        } else if (lk.err_ema < p.conflict_thresh / 2 && !(id < promo_hold.size() && promo_hold[id])) { lk.evidence = -1; evid.release((int32_t)s); }
    }
}
// ---------------------------------------------------------------------------------------------
// episodic memory: an exact-key lock born on a surprising event, its body a copy of the admitting
// route with the outcome imprinted one-shot into L6. The unit pool is the rolling window: when it
// runs low, the least-recently-hit episodic locks are evicted.
void Organism::episodic_birth(uint64_t key, uint32_t parent_lock) {
    if (p.market && p.market_contest && !market_make_room(0.f)) return;
    uint32_t u = alloc_unit(); if (u == UINT32_MAX) return;
    uint32_t pu = tree.locks[parent_lock].unit;
    std::memcpy(wbuf.p + (size_t)u * Dims::UNIT_BYTES, wbuf.p + (size_t)pu * Dims::UNIT_BYTES, Dims::UNIT_BYTES);
    if (!acc.empty()) std::memset(&acc[(size_t)u * Dims::UNIT_NIBBLES], 0, Dims::UNIT_NIBBLES);
    if (!acc16.empty()) std::memset(&acc16[(size_t)u * Dims::UNIT_NIBBLES], 0, Dims::UNIT_NIBBLES * sizeof(int16_t));
    std::memcpy(&act_ema[(size_t)u * 4], &act_ema[(size_t)pu * 4], 4);
    uint32_t id = tree.new_lock(key, ~0ULL, u, block_idx); unit_lock[u] = id; tree.locks[id].hits = p.novel_hits; // born familiar
    if (p.market) market_new_lock(id, true);
    if (p.episodic_promote) { if (snap.size() < p.units) snap.resize(p.units); snap[u] = SnapLink{parent_lock, tree.locks[parent_lock].born_block, 0, 0, 0}; }
    ++st.episodic_births; ++ep_totals.births; cur_flags |= FL_BIRTH;
    // imprint: forward the new body on the key-determined input (no workspace, no return state), then write
    // L6 row j = -sign(dl6_j) * sign(x6_i) * 7 so that the body's L6 output is the maximal INT8 pattern
    // aligned against the prediction error, i.e. the readout's direction toward the outcome. One-shot, INT4.
    uint8_t* w = wbuf.p + (size_t)u * Dims::UNIT_BYTES;
    int8_t l1[Dims::A], l2[Dims::B], l3[Dims::C], l5[Dims::E], l6[Dims::F], x2[Dims::IN2], x3[Dims::IN3], x5[Dims::IN5], x6[Dims::IN6];
    kern.run(w, L[LI_1], xin_exact, l1);
    std::memcpy(x2, l1, Dims::A); std::memset(x2 + Dims::A, 0, Dims::R2); std::memcpy(x3, l1, Dims::A); std::memset(x3 + Dims::A, 0, Dims::B);
    kern.run(w, L[LI_2], x2, l2); kern.run(w, L[LI_3], x3, l3);
    std::memcpy(x5, l2, Dims::B); std::memcpy(x5 + Dims::B, l3, Dims::C); kern.run(w, L[LI_5], x5, l5);
    std::memcpy(x6, l5, Dims::E); std::memset(x6 + Dims::E, 0, Dims::R6);
    st.macs += Dims::P1 + Dims::P2 + Dims::P3 + Dims::P5;
    // graded: row j magnitude follows the imprint direction g (initially -dl6, the readout's matched filter for the
    // outcome). If the pattern does not decode, refine g perceptron-style against the winning wrong symbol.
    float g[Dims::F]; for (int j = 0; j < Dims::F; ++j) g[j] = -dl6[j];
    std::vector<float> lg(p.vocab), pr(p.vocab); int32_t wtmp[Dims::F]; const uint32_t V = p.vocab, t = hist_target_;
    for (int it = 0; it <= p.imprint_iters; ++it) {
        float mx = 0.f; for (int j = 0; j < Dims::F; ++j) mx = std::max(mx, std::fabs(g[j]));
        for (int j = 0; j < Dims::F; ++j) {
            int gj = mx > 0.f ? (int)std::lround(g[j] / mx * 7.f) : 0;
            for (int i = 0; i < Dims::IN6; ++i) { int si = x6[i] > 0 ? 1 : x6[i] < 0 ? -1 : 0; nib_set(w, Dims::NIB_L6 + j * Dims::IN6 + i, (si && gj) ? gj * si : 0); }
        }
        kern.run(w, L[LI_6], x6, l6); for (int j = 0; j < Dims::F; ++j) wtmp[j] = l6[j];
        readout(wtmp, lg.data(), pr.data());
        uint32_t am = 0; for (uint32_t v = 1; v < V; ++v) if (pr[v] > pr[am]) am = v;
        if (am == t) { ++st.imprint_ok; cur_flags |= FL_IMPRINT_OK; break; }
        if (V <= (uint32_t)Dims::F) { for (uint32_t j = 0; j < (uint32_t)Dims::F; ++j) g[j] += (j == t ? 1.f : 0.f) - (j == am ? 1.f : 0.f); }
        else { const float* rt = &wout[(size_t)t * Dims::F]; const float* ra = &wout[(size_t)am * Dims::F]; for (int j = 0; j < Dims::F; ++j) g[j] += (rt[j] - ra[j]) * 4.f; }
    }
    st.level_changes += Dims::P6;
}
// Class over instance for snapshot bodies: the same exact context must surprise episodic_recur times.
bool Organism::recur_ready(uint64_t key) {
    if (p.episodic_recur <= 1) return true;
    if (recur_tag.empty()) { recur_tag.assign(1u << 20, 0); recur_cnt.assign(1u << 20, 0); }
    const uint64_t h = mix64(key ^ 0x5EC0DDULL); const size_t i = (size_t)(h & ((1u << 20) - 1)); const uint32_t tag = (uint32_t)(h >> 32) | 1u;
    if (recur_tag[i] != tag) { recur_tag[i] = tag; recur_cnt[i] = 1; return false; }
    if (++recur_cnt[i] < p.episodic_recur) return false;
    recur_tag[i] = 0; recur_cnt[i] = 0; return true;
}
// Promotion: a snapshot reused episodic_promote times marks a recurring context its general parent mispredicts. The
// parent's split evidence is opened (and held open), so it records every context it admits -- the contrast a split needs;
// the snapshot's own recurring context is recorded too. When the parent splits, a general child covers a family of contexts
// and the snapshot is retired at the next block boundary.
void Organism::promote_snapshots(uint64_t key, uint32_t target, const uint32_t* cand, uint32_t ncand) {
    if (promo_hold.size() < tree.locks.size()) promo_hold.resize(tree.locks.size(), 0);
    for (uint32_t i = 0; i < ncand; ++i) {
        const Lock& lk = tree.locks[cand[i]];
        if (lk.mask != ~0ULL || lk.hits < p.novel_hits + p.episodic_promote || lk.unit >= snap.size()) continue;
        SnapLink& sl = snap[lk.unit]; if (sl.parent == UINT32_MAX || sl.retire) continue;
        Lock& pl = tree.locks[sl.parent];
        if (!pl.alive || pl.born_block != sl.parent_born || pl.mask == ~0ULL) { if (sl.promoted && sl.parent < promo_hold.size() && promo_hold[sl.parent]) --promo_hold[sl.parent]; sl.parent = UINT32_MAX; continue; }
        if (!sl.promoted) { sl.promoted = 1; sl.parent_mask = pl.mask; ++promo_hold[sl.parent]; ++ep_totals.promoted; }
        else if (pl.mask != sl.parent_mask) {   // the parent has split since promotion: a general child now covers this family
            sl.retire = 1; if (promo_hold[sl.parent]) --promo_hold[sl.parent]; ++ep_totals.promoted_splits; continue;
        }
        if (pl.evidence < 0) pl.evidence = evid.acquire(sl.parent, pl.mask);
        if (pl.evidence >= 0) evid.record(pl.evidence, key, target);
    }
}
void Organism::retire_promoted() {
    for (uint32_t id = 0; id < tree.locks.size(); ++id) {
        Lock& lk = tree.locks[id];
        if (!lk.alive || lk.mask != ~0ULL || lk.unit >= snap.size() || !snap[lk.unit].retire) continue;
        snap[lk.unit] = SnapLink{};
        if (lk.evidence >= 0) { evid.release(lk.evidence); lk.evidence = -1; }
        free_unit(lk.unit); tree.kill_lock(id); ++ep_totals.retired;
    }
}
// ---- body market (Params::market) -------------------------------------------------------------------------
void Organism::market_new_lock(uint32_t id, bool enters_market) {
    if (mk_last_struct.size() < tree.locks.size()) { mk_last_struct.resize(tree.locks.size(), 0); mk_born_hits.resize(tree.locks.size(), 0); mk_state.resize(tree.locks.size(), 0); }
    mk_last_struct[id] = block_idx; mk_born_hits[id] = tree.locks[id].hits; mk_state[id] = enters_market ? 1 : 0;
    if (mk_util.size() < tree.locks.size()) mk_util.resize(tree.locks.size(), 0.f);
    mk_util[id] = 0.f;
}
bool Organism::market_cooling(uint32_t id) const { return id < mk_last_struct.size() && block_idx - mk_last_struct[id] < p.market_cooldown; }
float Organism::market_reuse(const Lock& lk, uint32_t id) const {
    const uint32_t age = block_idx - lk.born_block + 1, base = id < mk_born_hits.size() ? mk_born_hits[id] : 0;
    return (float)(lk.hits > base ? lk.hits - base : 0) / (float)age;
}
float Organism::market_fitness(const Lock& lk, uint32_t id) const {
    const float u = p.market_share && id < mk_util.size() ? mk_util[id] : lk.utility;
    return u * std::log1p(market_reuse(lk, id));
}
bool Organism::market_make_room(float bid) {
    const uint64_t backbone = std::max<uint64_t>(16, p.units / 1000);   // kept for sensory backbone births, which never contest
    const bool full = p.market_birth_ratio > 0.f && free_count() < p.units / 10;   // the throttle binds only once 90% full
    if (full && mk_tokens == 0) { ++mk_totals.throttled; return false; }
    auto spend = [&] { if (full && mk_tokens != UINT64_MAX) --mk_tokens; };   // only a birth spends
    if (free_count() > backbone) { ++mk_totals.contest_free; spend(); return true; }
    if (mk_qpos >= mk_queue.size()) { ++mk_totals.contest_losses; return false; }
    while (mk_qpos < mk_queue.size()) {
        const auto [fit, id] = mk_queue[mk_qpos];
        if (id >= tree.locks.size() || !tree.locks[id].alive || id >= mk_state.size() || mk_state[id] != 2) { ++mk_qpos; continue; }
        if (!(fit < bid)) break;                         // the weakest admitted body is at least as fit as the newcomer
        ++mk_qpos; if (tree.locks[id].mask == ~0ULL) ++mk_totals.culled_snapshot; else ++mk_totals.culled_general;
        market_release(id); ++mk_totals.contest_wins; spend(); return true;
    }
    ++mk_totals.contest_losses; return false;
}
void Organism::market_release(uint32_t id) {
    Lock& lk = tree.locks[id];
    if (lk.evidence >= 0) { evid.release(lk.evidence); lk.evidence = -1; }
    if (lk.unit < snap.size()) snap[lk.unit] = SnapLink{};
    if (id < mk_state.size()) mk_state[id] = 0;
    free_unit(lk.unit); tree.kill_lock(id);
}
void Organism::market_adjust(bool regressed, bool improved, bool grew) {
    if (mk_bar < 0.f) mk_bar = p.market_bar;
    if (regressed) { mk_bar = std::min(p.market_bar_max, mk_bar * 1.5f); ++mk_totals.bar_raises; }
    else if (improved) { mk_bar = std::max(p.market_bar_min, mk_bar * 0.9f); ++mk_totals.bar_relaxes; }
    else if (grew) { mk_bar = std::min(p.market_bar_max, mk_bar * 1.25f); ++mk_totals.parsimony_raises; }
}
void Organism::market_step() {
    if (mk_bar < 0.f) mk_bar = p.market_bar;
    if (mk_state.size() < tree.locks.size()) { mk_last_struct.resize(tree.locks.size(), 0); mk_born_hits.resize(tree.locks.size(), 0); mk_state.resize(tree.locks.size(), 0); }
    // probation: a new body must have been reused and hold utility at the bar, or it is released
    for (uint32_t id = 0; id < tree.locks.size(); ++id) {
        const Lock& lk = tree.locks[id];
        if (!lk.alive || mk_state[id] != 1 || block_idx - lk.born_block < p.market_probation) continue;
        const uint32_t reused = lk.hits > mk_born_hits[id] ? lk.hits - mk_born_hits[id] : 0;
        const float u = p.market_share && id < mk_util.size() ? mk_util[id] : lk.utility;
        if (reused >= p.market_min_hits && u >= mk_bar) { mk_state[id] = 2; ++mk_totals.probation_pass; }
        else { ++mk_totals.probation_fail; market_release(id); }
    }
    // capacity: release the least fit market bodies of any kind -- on probation or admitted -- until twice the reserve is
    // free. Probationers compete on the same fitness: an unreused newcomer (fitness ~0) goes before any proven body with
    // positive fitness; bodies with negative fitness go first. (Protecting probationers let a flood of new snapshots evict
    // every body that had passed probation.)
    // batched every market_gate_blocks blocks: free up to 2 x (reserve + the births expected until the next batch)
    if (block_idx % p.market_gate_blocks != 0 && !(p.market_contest && mk_queue.empty())) return;
    if (p.market_contest) {   // no refill: rank admitted bodies, weakest first, for the births of the next batch to outbid
        mk_queue.clear(); mk_qpos = 0;
        if (p.market_birth_ratio > 0.f) {
            const uint64_t passes = mk_totals.probation_pass - mk_pass_mark; mk_pass_mark = mk_totals.probation_pass;
            mk_tokens = std::max<uint64_t>(std::max<uint64_t>(16, p.units / 1000), (uint64_t)(p.market_birth_ratio * (float)passes));
        }
        for (uint32_t id = 0; id < tree.locks.size(); ++id) {
            const Lock& lk = tree.locks[id]; if (!lk.alive || mk_state[id] != 2) continue;
            if (p.market_protect && lk.mask != ~0ULL && !(lk.utility < p.recycle_utility)) continue;   // proven general body: keep
            mk_queue.emplace_back(market_fitness(lk, id), id);
        }
        std::sort(mk_queue.begin(), mk_queue.end());
        return;
    }
    uint64_t reserve = std::max<uint64_t>(16, (uint64_t)p.units * p.market_reserve_ppm / 1000000ULL) * p.market_gate_blocks;
    if (free_count() >= reserve) return;
    std::vector<std::pair<float, uint32_t>> c;
    for (uint32_t id = 0; id < tree.locks.size(); ++id) { const Lock& lk = tree.locks[id]; if (lk.alive && mk_state[id] != 0) c.emplace_back(market_fitness(lk, id), id); }
    size_t need = std::min(c.size(), (size_t)(2 * reserve) - free_count());
    std::nth_element(c.begin(), c.begin() + need, c.end());
    std::vector<uint32_t> batch; batch.reserve(need); for (size_t i = 0; i < need; ++i) batch.push_back(c[i].second);
    if (market_gate && !batch.empty()) {
        if (market_gate(batch)) ++mk_totals.gate_accepts;
        else {   // the batch as a whole regressed retention: keep proven bodies, release only unproven probationers
            ++mk_totals.gate_rejects;
            std::vector<uint32_t> kept; for (uint32_t id : batch) if (mk_state[id] == 1) kept.push_back(id);
            batch.swap(kept);
        }
    }
    market_cull(batch);
}
void Organism::market_cull(const std::vector<uint32_t>& ids) {
    for (uint32_t id : ids) {
        if (id >= tree.locks.size() || !tree.locks[id].alive || id >= mk_state.size() || mk_state[id] == 0) continue;
        if (tree.locks[id].mask == ~0ULL) ++mk_totals.culled_snapshot; else ++mk_totals.culled_general;
        market_release(id);
    }
}
Organism::MarketCensus Organism::market_census() const {
    MarketCensus m; std::vector<float> reuse, fit;
    for (uint32_t id = 0; id < tree.locks.size() && id < mk_state.size(); ++id) {
        const Lock& lk = tree.locks[id]; if (!lk.alive || mk_state[id] == 0) continue;
        ++m.bodies; m.snapshots += lk.mask == ~0ULL; m.probation += mk_state[id] == 1;
        const uint32_t age = block_idx - lk.born_block + 1, base = mk_born_hits[id];
        reuse.push_back((float)(lk.hits > base ? lk.hits - base : 0) / (float)age); fit.push_back(market_fitness(lk, id));
    }
    auto q = [](std::vector<float>& v, float f) { if (v.empty()) return 0.f; size_t k = (size_t)(f * (v.size() - 1)); std::nth_element(v.begin(), v.begin() + k, v.end()); return v[k]; };
    m.reuse_p10 = q(reuse, 0.1f); m.reuse_p50 = q(reuse, 0.5f); m.reuse_p90 = q(reuse, 0.9f);
    m.fitness_p10 = q(fit, 0.1f); m.fitness_p50 = q(fit, 0.5f); m.fitness_p90 = q(fit, 0.9f);
    return m;
}
void Organism::episodic_evict() {
    // keep at least a block's worth of births free: eviction runs between blocks, births happen inside them
    uint64_t reserve = std::max<uint64_t>((uint64_t)p.units * p.unit_reserve_ppm / 1000000ULL, 2ULL * p.block); if (reserve < 16) reserve = 16;
    if (free_count() >= reserve) return;
    std::vector<std::pair<uint32_t, uint32_t>> cands;   // (last_hit_block, lock id) over episodic locks
    for (uint32_t id = 0; id < tree.locks.size(); ++id) { const Lock& lk = tree.locks[id]; if (lk.alive && lk.mask == ~0ULL) cands.emplace_back(lk.last_hit_block, id); }
    size_t need = (size_t)(2 * reserve) - free_count(); if (need > cands.size()) need = cands.size();
    std::nth_element(cands.begin(), cands.begin() + need, cands.end());
    for (size_t i = 0; i < need; ++i) { Lock& lk = tree.locks[cands[i].second]; if (lk.evidence >= 0) { evid.release(lk.evidence); lk.evidence = -1; } ++ep_totals.evicted; if (lk.hits <= p.novel_hits) ++ep_totals.evicted_unused;
        if (lk.unit < snap.size()) { SnapLink& sl = snap[lk.unit]; if (sl.promoted && !sl.retire && sl.parent < promo_hold.size() && promo_hold[sl.parent]) --promo_hold[sl.parent]; sl = SnapLink{}; }
        free_unit(lk.unit); tree.kill_lock(cands[i].second); ++st.episodic_evictions; }
}
void Organism::end_block() {
    { StageTimer T(&st, ST_STRUCT, p.stage_timers);
      if (p.episodic && p.structural && p.episodic_promote) retire_promoted();
      if (p.market && p.structural && p.train) market_step();
      else if (p.episodic && p.structural) episodic_evict();
      if (p.structural && p.train) { commit_splits(); if (p.nursery && !p.market) recycle_locks(); }
      if (p.structural) { st.leaf_splits += tree.maintain(); for (int l = 1; l <= nlv; ++l) lv[l].tree.maintain(); }
      if (p.structural && p.lock_death_blocks) {
          for (uint32_t id = 0; id < tree.locks.size(); ++id) { Lock& lk = tree.locks[id];
              if (lk.alive && block_idx - lk.last_hit_block > p.lock_death_blocks) { if (lk.evidence >= 0) evid.release(lk.evidence); free_unit(lk.unit); tree.kill_lock(id); ++st.lock_deaths; } }
      } }
    st.live_locks = tree.live_locks(); st.live_units = live_units(); st.tree_nodes = tree.num_nodes();
    tree.depth_stats(st.depth_mean, st.depth_p95, st.leaf_occ);
    if (log_fp) { std::fprintf(log_fp, "%s\n", st.json(block_idx).c_str()); std::fflush(log_fp); }
    ++block_idx;
}

void Organism::seed_fake_population(uint32_t n, uint32_t novel_fraction_pct) {
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t u = alloc_unit(); if (u == UINT32_MAX) break; birth_unit_weights(u);
        uint64_t key = rng.next();
        uint32_t id = tree.new_lock(key, SENSORY_MASK | (rng.next() & 0xFFFFFF00ULL), u, 0); unit_lock[u] = id;
        tree.locks[id].hits = (i % 100 < novel_fraction_pct) ? 0 : p.novel_hits * 4 + 1;
    }
    for (int i = 0; i < 40; ++i) if (!tree.maintain()) break;
}


// ---- checkpoint -----------------------------------------------------------------------------------------------
// The model is constructed from the same Params first (layer descriptors, sizes, kernel choice); these are the
// members that learning, structure and the stream change afterwards. Per-event scratch (queues, traces of the
// last event, counterfactual buffers) is not state: every event rebuilds it before use.
template <class Ar> void Organism::state_io(Ar& ar) {
    if (nlv || seq.uncovered_extensions() || lvl_w.size())
        throw std::runtime_error("model checkpoint does not cover composer levels, answer attention, attached reasoners or the relation store");
    ar.section("params");
    { uint32_t units = p.units, vocab = p.vocab, motif = p.motif_size; int32_t agency = p.agency, task = p.agency_task; uint8_t seqon = seq_on, wide = wide_acc, train = p.train;
      ar(units); ar(vocab); ar(motif); ar(agency); ar(task); ar(seqon); ar(wide); ar(train);
      if (units != p.units || vocab != p.vocab || motif != p.motif_size || agency != p.agency || task != p.agency_task || seqon != (uint8_t)seq_on || wide != (uint8_t)wide_acc)
          throw std::runtime_error("checkpoint was written by a model with different structural parameters");
      p.train = train; }
    ar.section("weights");
    if (wbuf.n != (size_t)p.units * Dims::UNIT_BYTES) throw std::runtime_error("weight buffer size mismatch");
    ar.array(wbuf.p, wbuf.n);
    ar(acc); ar(acc16);
    ar.section("units");
    ar(unit_lock); ar(epoch); ar(unit_k); ar(resist); ar(alive); ar(act_ema); ar(assoc_value); ar(assoc_mask); ar(l4ret);
    ar(assoc_noconseq); ar(assoc_starve); ar(route_utility); ar(recirc_utility); ar(arm_on); ar(arm_off); ar(arm_non); ar(arm_noff);
    ar(recirc_used); ar(recirc_explored); ar(hop_extra);
    ar(free_units); ar(motif_free); ar(fill_cursor); ar(kscreen_tab);
    { uint64_t n = kstats.size(); ar(n); kstats.resize(n); ar.array(kstats.data(), kstats.size()); }
    ar.section("address");
    ar(tree); ar(evid); ar(pending_splits);
    ar.section("readout");
    ar(wout); ar(woutT); ar.array(ws, Dims::F); ar(logits); ar(probs); ar.array(dl6, Dims::F); ar.array(q6, Dims::F);
    ar(cur_novel); ar(hist_target_); ar(cur_flags);
    ar.section("stream");
    ar.array(hist, 16); ar(event_idx); ar(block_idx); ar(cur_epoch); ar.array(ws_mem, Dims::F); ar.array(last_l4, Dims::R);
    ar(loss_ema); ar(loss_ema_init); ar(wrong_ema); ar(prior);
    ar.array(xin, Dims::IN); ar.array(xin_exact, Dims::IN); ar.array(xin_focus, Dims::IN);
    ar(free_run); ar(prev_s);
    ar(ep_totals); ar(recur_tag); ar(recur_cnt); ar(snap); ar(promo_hold);
    ar(mk_totals); ar(mk_bar); ar(mk_last_struct); ar(mk_born_hits); ar(mk_state); ar(mk_util);
    ar.section("sequence");
    ar(seq);
    { uint8_t ext = seq.chain_on || seq.ndialogues; ar(ext); if (Ar::loading && ext != (uint8_t)(seq.chain_on || seq.ndialogues)) throw std::runtime_error("checkpoint sequence-memory extensions differ from the configuration"); if (ext) seq.extension_state_io(ar); }
    ar(seq_out); ar(body_probs); ar(seq_events); ar(seq_matched); ar(seq_bubble_events); ar(seq_conf_sum);
    ar.section("rng");
    ar(rng);
    ar.section("agency");
    ar(agent_on); ar(agent_obs); ar(agent_tendency); ar(agent_exact_only); ar(agent_last_ptarget);
    ar.array(task_obs, 8); ar(task_value); ar(task_reported);
    if (agent_on) {
        std::vector<std::string> names = agent.op_names; ar(agent);
        if (agent.op_names != names || agent.ops.size() != names.size()) throw std::runtime_error("agency operations differ from checkpoint");
    }
    ar.section("goal_credit");
    ar(phase); ar(ans_state); ar(goal_w); ar(ans_loss); ar(ans_n); ar(ans_ema); ar(elig); ar(elig_t); ar(touched); ar(touch_stamp); ar(reply_id);
    ar(m_emit_q); ar(m_emit_a); ar(m_emit_o); ar(bytes_q); ar(bytes_a); ar(bytes_o); ar(goal_credits); ar(goal_R_sum); ar(silenced);
    if (p.homeo) { ar.section("homeo"); ar(ushift); if (p.homeo == 2) { ar(hsat); ar(hmag); } }
    if (p.in_struct) { ar.section("in_struct"); ar(ws_scale); if (p.ws_bands) ar.array(ws_bscale, 4); if (p.ws_regs) ar.array(ws_reg, 4);
        if (p.ws_regs == 2) { for (int r = 0; r < 4; ++r) { ar.array(hw_g[r], Dims::F); ar.array(hw_v[r], Dims::F); } ar.array(hb_g, 4); ar(head_init); }
        if (p.ws_regs && p.reg_readout) ar(wreg);
        if (p.ws_regs && p.reg_center) ar.array(reg_mean, 4);
        if (p.in_words) { ar(emb); ar(word_cur); ar.array(word_ids, 3); } }
    if (p.seq_gate) { ar.section("seq_gate"); ar.array(seq_gw, SEQ_GATE_F); }
    if (p.ws_split) { ar.section("ws_split"); ar(wsp); }
    if (p.tile_split && p.tiles) { ar.section("tile_split"); ar(wtl); }
    if (p.ws_split5) { ar.section("ws_split5"); ar(wsp5); }
    if (p.ro_hidden) { ar.section("ro_hidden"); ar(rh_W1); ar(rh_b1); ar(rh_W2); }
    if (p.motif_mix) { ar.section("motif_mix"); ar(mix_th); }
    if (p.disinh > 0.f) { ar.section("disinh"); ar(dis_w); }
    if (p.motif_attn) { ar.section("motif_attn"); ar(att_W); }
    if (p.tiles) { ar.section("tiles"); ar(tile_M); ar(tst_.n);   // the mixings and the views (not the training accumulators / traces)
        for (uint32_t l = 0; l < p.tiles; ++l) { uint32_t n = (uint32_t)tst_.q[l].size(); ar(n); if (Ar::loading) tst_.q[l].assign(n, Tile{});
            for (auto& t : tst_.q[l]) { ar.array(t.v, Dims::F); ar(t.src); ar(t.unit); if (p.tile_wm) ar(t.use); if (Ar::loading) { std::fill(t.g, t.g + Dims::F, 0.f); t.has_tr = false; } } }
        if (p.tile_att) ar(tile_A);
        if (p.tile_fw) { ar(tst_.fw_A); ar(fw_W); ar(tst_.fw_n); }
        if (p.tile_ret) { ar(ret_W); ar(tst_.nw); ar(tst_.fq_n); ar(tst_.fq_t); ar.array(tst_.rh, 8); ar(tst_.nrh);
            uint32_t n = (uint32_t)tst_.ret.size(); ar(n); if (Ar::loading) tst_.ret.assign(n, TileState::RetItem{});
            for (auto& it : tst_.ret) { ar.array(it.v, Dims::F); ar.array(it.key[0], TileState::RET_D); if (ret_heads() == 2) ar.array(it.key[1], TileState::RET_D); ar(it.src); } }
        if (p.tile_wm) { ar(tst_.wm_live); for (auto& t : tst_.wm) { ar.array(t.v, Dims::F); ar(t.src); ar(t.unit); ar(t.use); if (Ar::loading) { std::fill(t.g, t.g + Dims::F, 0.f); t.has_tr = false; } } } }
    ar.section("end");
}
void Organism::save_state(state::Writer& w) { state_io(w); }
void Organism::load_state(state::Reader& r) { state_io(r); }
uint64_t Organism::state_fingerprint() { state::Writer w; state_io(w); return w.fingerprint(); }
template void Organism::state_io<state::Writer>(state::Writer&);
template void Organism::state_io<state::Reader>(state::Reader&);

} // namespace laminar
