// The organism with a hybrid attention head (hybrid.h) after the bodies: online next-byte prediction.
//   laminar_hybrid file=PATH [max_bytes=N] [org=1] [state=1] [base=1] [d=256] [ctx=128] [chunk=128] [hlr=0.001] ...organism params
//   org=0: no organism (the head alone is a small byte transformer: the control for "just added a transformer")
//   state=0: the head does not see the bodies' workspace; base=0: the head predicts alone (not a correction)
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include "laminar/hybrid.h"
#include <chrono>
#include <cstdio>
using namespace laminar;
int main(int argc, char** argv) {
    std::map<std::string, std::string> rest; Params p = Params::from_args(argc, argv, &rest); p.vocab = 256; if (p.future_buckets == 32) p.future_buckets = 64;
    auto geti = [&](const char* k, uint64_t dflt) { auto it = rest.find(k); return it == rest.end() ? dflt : std::strtoull(it->second.c_str(), nullptr, 10); };
    auto getf = [&](const char* k, double dflt) { auto it = rest.find(k); return it == rest.end() ? dflt : std::strtod(it->second.c_str(), nullptr); };
    std::vector<uint8_t> data = read_file_bytes(rest["file"], geti("max_bytes", 0));
    const bool use_org = geti("org", 1), use_head = geti("head", 1); const int chunk = (int)geti("chunk", 128);
    HybridHead hh; hh.d = (int)geti("d", 256); hh.dff = 4 * hh.d; hh.ctx = (int)geti("ctx", 128); hh.use_state = use_org && geti("state", 1); hh.use_base = use_org && geti("base", 1); hh.lr = (float)getf("hlr", 1e-3); hh.init(p.seed);
    std::printf("file=%s bytes=%zu org=%d head=%d state=%d base=%d d=%d ctx=%d params=%zu\n", rest["file"].c_str(), data.size(), (int)use_org, (int)use_head, (int)hh.use_state, (int)hh.use_base, hh.d, hh.ctx, hh.params());
    std::vector<uint8_t> sym, tgt; std::vector<float> st, base; sym.reserve(chunk); tgt.reserve(chunk);
    double org_loss = 0, comb_loss = 0, w_org = 0, w_comb = 0; uint64_t nb = 0, w_n = 0; const uint64_t prog = geti("progress", 1 << 20);
    auto t0 = std::chrono::steady_clock::now(); double head_sec = 0;
    auto flush = [&]() { if (sym.empty()) return; const int n = (int)sym.size();
        auto h0 = std::chrono::steady_clock::now();
        double l = use_head ? hh.step(n, sym.data(), st.data(), use_org ? base.data() : nullptr, tgt.data()) : 0.0;
        head_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count();
        if (!use_head) for (int i = 0; i < n; ++i) l += -base[(size_t)i * 256 + tgt[i]];
        comb_loss += l; w_comb += l; sym.clear(); tgt.clear(); st.clear(); base.clear(); };
    auto observe = [&](uint8_t s, uint8_t t, const int32_t* ws, const float* pr) {
        sym.push_back(s); tgt.push_back(t);
        for (int j = 0; j < 32; ++j) st.push_back(ws ? std::tanh((float)ws[j] / 128.f) : 0.f);
        for (int v = 0; v < 256; ++v) base.push_back(pr ? std::log(std::max(pr[v], 1e-9f)) : 0.f);
        if (pr) { double l = -std::log(std::max(pr[t], 1e-9f)); org_loss += l; w_org += l; }
        ++nb; ++w_n; if ((int)sym.size() == chunk) flush();
        if (w_n >= prog) { const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  %6.2f MB | organism BPB %.4f | combined BPB %.4f | %.1f KB/s (head %.0f%% of time)\n", nb / 1048576.0, use_org ? w_org / w_n / 0.693147 : 0.0, w_comb / w_n / 0.693147, nb / sec / 1024.0, 100.0 * head_sec / sec);
            std::fflush(stdout); w_org = w_comb = 0; w_n = 0; } };
    if (use_org) {
        Organism org(p); org.on_event = [&](uint8_t s, uint8_t t, const int32_t* ws, const float* pr) { observe(s, t, ws, pr); };
        size_t pos = 0; while (pos + 1 < data.size()) { size_t n = std::min<size_t>(p.block, data.size() - 1 - pos); org.run_block(data.data() + pos, n + 1); org.end_block(); org.stats().reset(); pos += n; }
    } else for (size_t i = 0; i + 1 < data.size(); ++i) observe(data[i], data[i + 1], nullptr, nullptr);
    flush();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("TOTAL: %llu bytes | organism BPB %.4f | combined BPB %.4f | %.1f KB/s | head params %zu\n", (unsigned long long)nb, use_org ? org_loss / nb / 0.693147 : 0.0, comb_loss / nb / 0.693147, nb / sec / 1024.0, hh.params());
}
