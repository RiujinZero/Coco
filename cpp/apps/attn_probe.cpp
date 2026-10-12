// The answer-attention block alone (no organism, no workspace): feed a byte stream, learn from every reply's
// first token, then score its answers on held-out questions. For fast iteration on the block, never for results.
//   attn_probe file=STREAM [test=TEST.jsonl] [mode=1] [lr=] [fg_lr=] [warm=] [epochs=] [clip=] [passes=1] [max_bytes=]
#include "laminar/attend.h"
#include <fstream>
#include <map>
#include <cstring>
using namespace laminar;
static std::string json_str(const std::string& s, size_t& i) { std::string o; ++i; for (; i < s.size(); ++i) { char c = s[i]; if (c == '"') { ++i; break; } if (c == '\\' && i + 1 < s.size()) { c = s[++i]; o += c == 'n' ? '\n' : c == 't' ? '\t' : c; } else o += c; } return o; }
int main(int argc, char** argv) {
    std::map<std::string, std::string> a; for (int i = 1; i < argc; ++i) { const char* e = std::strchr(argv[i], '='); if (e) a[std::string(argv[i], (size_t)(e - argv[i]))] = e + 1; }
    auto gs = [&](const char* k, const char* d) { return a.count(k) ? a[k] : std::string(d); };
    std::ifstream f(gs("file", "data/babi_all_train.txt"), std::ios::binary); std::string s((std::istreambuf_iterator<char>(f)), {});
    size_t mb = std::stoull(gs("max_bytes", "0")); if (mb && s.size() > mb) s.resize(mb);
    int passes = std::stoi(gs("passes", "1"));
    AnswerAttention at; at.init(std::stoi(gs("mode", "1")), nullptr, 1);
    at.lr = std::stof(gs("lr", "0.5")); at.fg_lr = std::stof(gs("fg_lr", "0.005")); at.warm = std::stoull(gs("warm", "20000")); at.epochs = std::stoi(gs("epochs", "2")); at.clip = std::stof(gs("clip", "5")); at.fg_k = std::stoi(gs("fg_k", "1")); at.fg_local_out = gs("fg_local_out", "1") != "0"; at.holdout = std::stoi(gs("holdout", "5"));
    for (int p = 0; p < passes; ++p) for (unsigned char c : s) at.observe(c, false);
    at.window_report();
    if (!a.count("test")) { std::printf("%s\n", at.report().c_str()); return 0; }
    std::ifstream t(a["test"]); std::string line; std::map<std::string, int> n, ok, spoke, spoke_ok;
    while (std::getline(t, line)) {
        size_t i = line.find("\"task\""); if (i == std::string::npos) continue; i = line.find('"', i + 6); std::string task = json_str(line, i);
        i = line.find("[[", i) + 2; while (line[i] != '"') ++i; std::string q = json_str(line, i); while (line[i] != '"') ++i; std::string ans = json_str(line, i);
        while (line[i] != '"') ++i; json_str(line, i); i = line.find('[', i) + 1; std::vector<std::string> story; while (i < line.size() && line[i] != ']') { if (line[i] == '"') story.push_back(json_str(line, i)); else ++i; }
        std::string prompt = "\n<|endofdialogue|>\n"; for (auto& x : story) prompt += x + "\n"; prompt += "USER: " + q + "\n\nASSISTANT:";
        for (unsigned char c : prompt) at.observe(c, true);
        std::string got = at.dec_s; for (auto& c : got) c = (char)std::tolower((uint8_t)c);
        std::string want = ans; for (auto& c : want) c = (char)std::tolower((uint8_t)c); { size_t k = 0; while (k < want.size() && std::isalnum((uint8_t)want[k])) ++k; want = want.substr(0, k); }
        bool good = !got.empty() && got == want; if (a.count("show") && n[task] < std::stoi(a["show"])) std::printf("  [%s] %s -> %s (want %s) gate %.2f", task.c_str(), q.c_str(), got.c_str(), want.c_str(), at.dec_gate);
        if (a.count("show") && n[task] < std::stoi(a["show"])) { AnswerAttention::Fw<float> ff; at.forward(at.cur, -1, 0, 0.f, false, ff, nullptr); int bc = 0; for (int c = 1; c < at.nc; ++c) if (ff.lg[c] > ff.lg[bc]) bc = c;
            std::printf(" pi %.2f, classifier says %s (p %.2f), items %d\n", ff.pi, at.cls_surf[bc].c_str(), std::exp(ff.lg[bc] - ff.lse), ff.N); } ++n[task]; ok[task] += good; if (at.use) { ++spoke[task]; spoke_ok[task] += good; }
        at.pending = false; at.use = false;
    }
    if (a.count("gradcheck")) {   // backprop's site gradients . tangent  vs  the forward-mode directional derivative
        auto smp = at.cur; if (!smp.items.empty() && smp.items[0].nw) { smp.prop_h = smp.items[0].w[0]; smp.prop_c = 0.7f; } uint32_t gh = at.cls_h[0]; int gc = 0; double worst = 0.0;
        for (int trial = 0; trial < 5; ++trial) {
            AnswerAttention::Fw<float> f; at.forward(smp, gc, gh, 2.f, true, f, nullptr); AnswerAttention::Sites g; at.backward(smp, f, 2.f, true, g);
            AnswerAttention::Tan u; auto fill = [&](std::vector<float>& v, size_t n) { v.resize(n); for (auto& x : v) x = at.gauss(); };
            size_t nw = 0; for (auto& it : smp.items) nw += it.nw; const int N = (int)smp.items.size();
            fill(u.q, 16); fill(u.k, (size_t)N * 16); fill(u.r, N); fill(u.r2, N); fill(u.pre, N); fill(u.val, N); fill(u.lg, at.nc); fill(u.s, nw); u.lam = at.gauss(); u.gl = at.gauss(); u.pil = at.gauss(); u.ppl = at.gauss();
            AnswerAttention::Fw<Dual> fd; at.forward(smp, gc, gh, 2.f, true, fd, &u);
            auto dot = [](const std::vector<float>& x, const std::vector<float>& y) { double s = 0; for (size_t i = 0; i < x.size(); ++i) s += (double)x[i] * y[i]; return s; };
            double bp = dot(g.q, u.q) + dot(g.k, u.k) + dot(g.r, u.r) + dot(g.r2, u.r2) + dot(g.pre, u.pre) + dot(g.val, u.val) + dot(g.lg, u.lg) + dot(g.s, u.s) + g.lam * u.lam + g.gl * u.gl + g.pil * u.pil + g.ppl * u.ppl;
            double fm = fd.loss.d; worst = std::max(worst, std::fabs(bp - fm) / std::max(1e-6, std::fabs(fm)));
            std::printf("gradcheck: backprop %.6f  forward-mode %.6f  (loss %.4f, items %d)\n", bp, fm, f.loss, N);
        }
        std::printf("gradcheck worst relative error %.2e\n", worst);
    }
    int tn = 0, tok = 0; for (auto& [k, v] : n) { std::printf("%-12s %4d/%-4d  spoke %4d, right when speaking %4d\n", k.c_str(), ok[k], v, spoke[k], spoke_ok[k]); tn += v; tok += ok[k]; }
    std::printf("TOTAL %d/%d = %.1f%%\n", tok, tn, 100.0 * tok / std::max(tn, 1));
}
