// Validation learner: the CPU reference drives the trainer's host weights; the GPU learner runs the same steps on its
// own copy of them and is compared after every step. Bodies whose weights drifted apart are re-synced from the
// reference, so every comparison is of one step's arithmetic, not of accumulated divergence.
#include "laminar/batch_learner.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace laminar {
namespace {

class DualLearner final : public Learner {
public:
    DualLearner(const LearnConfig& c, uint8_t* host_w) : c_(c), hw_(host_w), gw_((size_t)c.units * Dims::UNIT_BYTES) {
        std::memcpy(gw_.data(), hw_, gw_.size());
        ref_ = make_ref_learner(c, hw_); gpu_ = make_gpu_learner(c, gw_.data());
    }
    std::string name() const override { return "dual(" + ref_->name() + " vs " + gpu_->name() + ")"; }
    void upload_all() override { std::memcpy(gw_.data(), hw_, gw_.size()); ref_->upload_all(); gpu_->upload_all(); }
    void upload_units(const uint32_t* ids, uint32_t n) override { copy_units(ids, n); ref_->upload_units(ids, n); gpu_->upload_units(ids, n); }
    void set_head(const float* a, const float* b, const float* hb) override { ref_->set_head(a, b, hb); gpu_->set_head(a, b, hb); }
    void take_head_grads(float* hd) override {
        std::vector<float> r(8 * Dims::F + 4, 0.f), g(8 * Dims::F + 4, 0.f);
        ref_->take_head_grads(r.data()); gpu_->take_head_grads(g.data());
        double num = 0, den = 0; for (size_t i = 0; i < r.size(); ++i) { num += std::fabs((double)r[i] - g[i]); den += std::fabs((double)r[i]); hd[i] += r[i]; }
        if (den > 0) { head_rel_max_ = std::max(head_rel_max_, num / den); ++head_checks_; }
    }
    void read_unit_state(uint32_t u, float* sh, float* m, float* v, float* g) override { ref_->read_unit_state(u, sh, m, v, g); }
    void read_ring_tg(uint32_t s, uint32_t slot, float* tg) override { ref_->read_ring_tg(s, slot, tg); }

    void step(const StepIn& in) override {
        copy_units(in.dirty, in.ndirty);                                  // births and imprints reach both
        ref_->step(in); gpu_->step(in);
        ++steps_;
        // stepped bodies: INT4 weights and float shadows
        std::vector<uint32_t> drift;
        std::vector<float> a(Dims::UNIT_NIBBLES), b(Dims::UNIT_NIBBLES), ma(Dims::UNIT_NIBBLES), mb(Dims::UNIT_NIBBLES);
        for (uint32_t k = 0; k < in.ndue; ++k) {
            const uint32_t u = in.due[k].unit; const uint8_t* wr = hw_ + (size_t)u * Dims::UNIT_BYTES; const uint8_t* wg = gw_.data() + (size_t)u * Dims::UNIT_BYTES;
            uint32_t mism = 0; for (int i = 0; i < Dims::UNIT_BYTES; ++i) mism += ((wr[i] ^ wg[i]) & 15) != 0, mism += ((wr[i] ^ wg[i]) >> 4) != 0;
            nib_checked_ += Dims::UNIT_NIBBLES; nib_mism_ += mism; if (mism) drift.push_back(u);
            if (shadow_checks_ < 4096 || (steps_ % 16) == 0) {           // shadows: every stepped body early on, then a sample
                ref_->read_unit_state(u, a.data(), ma.data(), nullptr, nullptr); gpu_->read_unit_state(u, b.data(), mb.data(), nullptr, nullptr);
                for (int i = 0; i < Dims::UNIT_NIBBLES; ++i) { const double d = std::fabs((double)a[i] - b[i]); sh_max_ = std::max(sh_max_, d); sh_sum_ += d;
                    const double dm = std::fabs((double)ma[i] - mb[i]), sm = std::fabs((double)ma[i]); m_num_ += dm; m_den_ += sm; }
                sh_n_ += Dims::UNIT_NIBBLES; ++shadow_checks_;
            }
        }
        // ring errors: the slot of each stream's previous event (k = 1 of this step's accumulation)
        std::vector<float> ta(Dims::F), tb(Dims::F);
        for (uint32_t s = 0; s < c_.streams && s < 64; ++s) {
            const LStream& ls = in.streams[s]; if (!ls.active || !ls.fill) continue;
            const uint32_t slot = (ls.head + c_.tc_window - 1) % c_.tc_window;
            ref_->read_ring_tg(s, slot, ta.data()); gpu_->read_ring_tg(s, slot, tb.data());
            for (int j = 0; j < Dims::F; ++j) { tg_num_ += std::fabs((double)ta[j] - tb[j]); tg_den_ += std::fabs((double)ta[j]); }
        }
        if (!drift.empty()) { copy_units(drift.data(), (uint32_t)drift.size()); gpu_->upload_units(drift.data(), (uint32_t)drift.size()); resynced_ += drift.size(); }
    }
    std::string report() const override {
        char b[512];
        std::snprintf(b, sizeof b, "dual: %llu steps; stepped-body nibbles differing %llu of %llu (%.2e); shadow |diff| max %.3g mean %.3g over %llu bodies; "
                      "Adam m rel diff %.3g; ring error rel diff %.3g; head gradient rel diff max %.3g (%llu checks); bodies re-synced %llu",
                      (unsigned long long)steps_, (unsigned long long)nib_mism_, (unsigned long long)nib_checked_, nib_checked_ ? (double)nib_mism_ / nib_checked_ : 0.0,
                      sh_max_, sh_n_ ? sh_sum_ / sh_n_ : 0.0, (unsigned long long)shadow_checks_, m_den_ > 0 ? m_num_ / m_den_ : 0.0, tg_den_ > 0 ? tg_num_ / tg_den_ : 0.0,
                      head_rel_max_, (unsigned long long)head_checks_, (unsigned long long)resynced_);
        return b;
    }

private:
    void copy_units(const uint32_t* ids, uint32_t n) { for (uint32_t i = 0; i < n; ++i) std::memcpy(gw_.data() + (size_t)ids[i] * Dims::UNIT_BYTES, hw_ + (size_t)ids[i] * Dims::UNIT_BYTES, Dims::UNIT_BYTES); }
    LearnConfig c_; uint8_t* hw_; std::vector<uint8_t> gw_;
    std::unique_ptr<Learner> ref_, gpu_;
    uint64_t steps_ = 0, nib_checked_ = 0, nib_mism_ = 0, shadow_checks_ = 0, sh_n_ = 0, head_checks_ = 0, resynced_ = 0;
    double sh_max_ = 0, sh_sum_ = 0, m_num_ = 0, m_den_ = 0, tg_num_ = 0, tg_den_ = 0, head_rel_max_ = 0;
};

} // namespace

std::unique_ptr<Learner> make_dual_learner(const LearnConfig& c, uint8_t* host_weights) { return std::make_unique<DualLearner>(c, host_weights); }
std::unique_ptr<Learner> make_learner(LearnerKind k, const LearnConfig& c, uint8_t* host_weights) {
    switch (k) { case LearnerKind::Gpu: return make_gpu_learner(c, host_weights); case LearnerKind::Dual: return make_dual_learner(c, host_weights); case LearnerKind::None: return nullptr; default: return make_ref_learner(c, host_weights); }
}

} // namespace laminar
