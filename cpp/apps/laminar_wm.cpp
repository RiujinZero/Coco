// The organism with the "white matter" probe (whitematter.h): cheap attention between motifs across recent bytes.
//   laminar_wm file=PATH [max_bytes=N] [wm=motif|sum|sym|now|off] [wd=16] [wp=32] [wlr=0.003] ...organism params
#include "laminar/organism.h"
#include "laminar/tasks.h"
#include "laminar/whitematter.h"
#include <chrono>
#include <cstdio>
using namespace laminar;
int main(int argc, char** argv) {
    std::map<std::string, std::string> rest; Params p = Params::from_args(argc, argv, &rest); p.vocab = 256; if (p.future_buckets == 32) p.future_buckets = 64;
    auto geti = [&](const char* k, uint64_t dflt) { auto it = rest.find(k); return it == rest.end() ? dflt : std::strtoull(it->second.c_str(), nullptr, 10); };
    auto getf = [&](const char* k, double dflt) { auto it = rest.find(k); return it == rest.end() ? dflt : std::strtod(it->second.c_str(), nullptr); };
    std::vector<uint8_t> data = read_file_bytes(rest["file"], geti("max_bytes", 0));
    const std::string mode = rest.count("wm") ? rest["wm"] : "motif"; const bool on = mode != "off";
    WhiteMatter wm; wm.d = (int)geti("wd", 16); wm.P = (int)geti("wp", 32); wm.lr = (float)getf("wlr", 3e-3); wm.nmotif = (int)std::max<uint32_t>(1, p.units / p.motif_size + 1);
    wm.mode = mode == "sum" ? 1 : mode == "sym" ? 2 : mode == "now" ? 3 : 0; wm.init(p.seed);
    Organism org(p); std::vector<float> base(256); std::vector<std::pair<int, const int8_t*>> em;
    double lo = 0, lc = 0, wo = 0, wc = 0; uint64_t nb = 0, wn = 0; const uint64_t prog = geti("progress", 1 << 20); double wm_sec = 0;
    std::printf("file=%s bytes=%zu wm=%s d=%d P=%d params=%zu motifs=%d\n", rest["file"].c_str(), data.size(), mode.c_str(), wm.d, wm.P, on ? wm.params() : (size_t)0, wm.nmotif);
    auto t0 = std::chrono::steady_clock::now();
    org.on_event = [&](uint8_t s, uint8_t tg, const int32_t* ws, const float* pr) {
        const double l0 = -std::log(std::max(pr[tg], 1e-9f)); lo += l0; wo += l0; double l1 = l0;
        if (on) { auto h0 = std::chrono::steady_clock::now();
            for (int v = 0; v < 256; ++v) base[v] = std::log(std::max(pr[v], 1e-9f));
            em.clear(); org.for_each_emitter([&](uint32_t u, const int8_t* l6) { em.emplace_back((int)(u / p.motif_size), l6); });
            l1 = wm.step(s, em, ws, base.data(), tg); wm_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count(); }
        lc += l1; wc += l1; ++nb; ++wn;
        if (wn >= prog) { const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  %6.2f MB | organism BPB %.4f | combined BPB %.4f | %.1f KB/s (probe %.0f%% of time)\n", nb / 1048576.0, wo / wn / 0.693147, wc / wn / 0.693147, nb / sec / 1024.0, 100.0 * wm_sec / sec); std::fflush(stdout); wo = wc = 0; wn = 0; } };
    size_t pos = 0; while (pos + 1 < data.size()) { size_t n = std::min<size_t>(p.block, data.size() - 1 - pos); org.run_block(data.data() + pos, n + 1); org.end_block(); org.stats().reset(); pos += n; }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("TOTAL: %llu bytes | organism BPB %.4f | combined BPB %.4f | %.1f KB/s | probe %.0f%% of time | params %zu\n", (unsigned long long)nb, lo / nb / 0.693147, lc / nb / 0.693147, nb / sec / 1024.0, 100.0 * wm_sec / sec, on ? wm.params() : (size_t)0);
}
