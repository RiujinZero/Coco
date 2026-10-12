// Bind-and-recall probe: USEFUL memory = capacity x span x retrieval (docs/PYRAMID_MEMORY.md), not one bit over 8 words.
//
// A story binds k entities to objects ("the red box has the key ."), then N words of random filler sentences, then asks
// about one entity in a DIFFERENT surface form ("in the red box is the key ."). The different form defeats the sequence
// memory's verbatim copy: "in the red box is the " never occurred before, so the answer needs a learned binding from the
// entity to its object, held across the filler and retrieved when asked. Train the organism online on TRAIN_BYTES of
// such stories, then score, frozen, the correct object against a distractor (another object bound in the same story, or
// for k=1 an unbound one), by capacity k and span N.
//
// usage: recall_probe TRAIN_BYTES [key=value organism params...]
//   MEM_ARMS=1: also score with the tile read cut and with the register read cut (which memory holds the binding?)
//   KMAX=4 (bindings per training story, 1..KMAX), DMAX=60 (filler words per training story, 0..DMAX)
#include "laminar/organism.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace laminar;
static const char* COLOR[] = {"red", "blue", "green", "yellow", "black", "white", "brown", "pink"};
static const char* HOLDER[] = {"box", "bag", "cup", "jar"};
static const char* OBJECT[] = {"key", "coin", "ring", "shell", "stone", "card", "pen", "bell", "leaf", "bead", "nut", "pin"};
static const char* FILLW[] = {"the", "a", "man", "woman", "dog", "cat", "bird", "child", "walks", "runs", "sleeps", "sings", "sees", "likes",
    "near", "by", "over", "under", "river", "garden", "house", "town", "road", "hill", "very", "old", "small", "happy", "quiet", "tall",
    "long", "cold", "bright", "slowly", "today", "again", "with", "from", "and", "then"};
static constexpr int NCOLOR = 8, NHOLDER = 4, NOBJ = 12, NFILL = 40;

static void feed_text(Organism& m, const std::string& s) { std::vector<uint8_t> b(s.size() + 1); b[0] = m.last_symbol(); std::memcpy(b.data() + 1, s.data(), s.size()); m.run_block(b.data(), b.size()); }
static std::string entity(int e) { return std::string(COLOR[e % NCOLOR]) + " " + HOLDER[e / NCOLOR]; }

struct Story { std::string ctx; int obj, distractor; };
// k distinct entities bound to k distinct objects, ~d filler words, a question about one of them
static Story story(std::mt19937_64& g, int k, int d) {
    std::vector<int> ents, objs;
    while ((int)ents.size() < k) { const int e = (int)(g() % (NCOLOR * NHOLDER)); bool dup = false; for (int x : ents) dup |= x == e; if (!dup) ents.push_back(e); }
    while ((int)objs.size() < k) { const int o = (int)(g() % NOBJ); bool dup = false; for (int x : objs) dup |= x == o; if (!dup) objs.push_back(o); }
    std::string s;
    for (int i = 0; i < k; ++i) s += "the " + entity(ents[i]) + " has the " + OBJECT[objs[i]] + " .\n";
    int words = 0;
    while (words < d) { const int n = 3 + (int)(g() % 6); for (int w = 0; w < n; ++w) { s += w ? " " : ""; s += FILLW[g() % NFILL]; } s += " .\n"; words += n; }
    const int q = (int)(g() % k); int dis;
    if (k > 1) { do dis = (int)(g() % k); while (dis == q); dis = objs[dis]; }
    else { do dis = (int)(g() % NOBJ); while (dis == objs[0]); }
    s += "in the " + entity(ents[q]) + " is the ";
    return {s, objs[q], dis};
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: recall_probe TRAIN_BYTES [params]");
        const size_t ntr = std::stoul(argv[1]);
        const int kmax = std::getenv("KMAX") ? std::atoi(std::getenv("KMAX")) : 4, dmax = std::getenv("DMAX") ? std::atoi(std::getenv("DMAX")) : 60;
        std::vector<std::string> args{"recall"}; for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
        std::vector<char*> av; for (auto& a : args) av.push_back(a.data());
        Organism model(Params::from_args((int)av.size(), av.data()));
        std::mt19937_64 g(1);
        std::string text; while (text.size() < ntr) { const Story st = story(g, 1 + (int)(g() % kmax), (int)(g() % (dmax + 1))); text += st.ctx + OBJECT[st.obj] + " .\n"; }
        model.set_mode(true, true);
        for (size_t off = 0; off < text.size(); off += 262144) { const size_t n = std::min<size_t>(262144, text.size() - off);
            std::vector<uint8_t> b(text.begin() + off, text.begin() + off + n); b.insert(b.begin(), model.last_symbol()); model.run_block(b.data(), b.size()); model.end_block(); }
        model.set_mode(false, false); feed_text(model, "\n");
        if (std::getenv("DIAG")) {   // where does recall fail? at the question: attention on the object's slot; the object's identity in its tiles
            std::vector<float> cal, cv, ral; std::vector<int> cobj, robj; uint64_t obj_ev = 0;   // the last attend: weights, object-holding slots, that slot's value
            model.on_attend = [&](const TileState& ts) { cal = ts.att_al; cobj.assign(ts.att_slot.size(), 0); cv.clear();
                for (size_t s = 0; s < ts.att_slot.size(); ++s) { const Tile* t = ts.att_slot[s]; const int nl = std::max<int>(1, t->nleaf);
                    for (int i = 0; i < nl; ++i) if ((t->nleaf ? t->leaf[i] : t->src) == obj_ev) cobj[s] = 1;
                    if (cobj[s] && cv.empty()) cv.assign(t->v, t->v + Dims::F); }
                ral.assign(ts.ret_al.begin(), ts.ret_al.begin() + std::min(ts.ret_al.size(), ts.ret.size()));   // head 0
                robj.assign(ral.size(), 0); for (size_t j = 0; j < ral.size(); ++j) robj[j] = ts.ret[j].src == obj_ev; };
            auto decode = [](const std::vector<std::vector<float>>& X, const std::vector<int>& Y) {   // ridge least squares (F + bias -> one-hot) on 2/3, accuracy on 1/3
                const int D = Dims::F + 1, n = (int)X.size(), ntr = n * 2 / 3; if (n < 30) return -1.0;
                std::vector<double> G((size_t)D * D, 0.0), B((size_t)D * NOBJ, 0.0);
                for (int i = 0; i < ntr; ++i) { double x[Dims::F + 1]; for (int j = 0; j < Dims::F; ++j) x[j] = X[i][j]; x[Dims::F] = 1.0;
                    for (int a = 0; a < D; ++a) { for (int b = 0; b < D; ++b) G[(size_t)a * D + b] += x[a] * x[b]; B[(size_t)a * NOBJ + Y[i]] += x[a]; } }
                for (int a = 0; a < D; ++a) G[(size_t)a * D + a] += 1e-3 * ntr;
                for (int c = 0; c < D; ++c) { int piv = c; for (int r = c + 1; r < D; ++r) if (std::fabs(G[(size_t)r * D + c]) > std::fabs(G[(size_t)piv * D + c])) piv = r;   // Gauss-Jordan
                    for (int k = 0; k < D; ++k) std::swap(G[(size_t)c * D + k], G[(size_t)piv * D + k]); for (int k = 0; k < NOBJ; ++k) std::swap(B[(size_t)c * NOBJ + k], B[(size_t)piv * NOBJ + k]);
                    const double iv = 1.0 / G[(size_t)c * D + c]; for (int k = 0; k < D; ++k) G[(size_t)c * D + k] *= iv; for (int k = 0; k < NOBJ; ++k) B[(size_t)c * NOBJ + k] *= iv;
                    for (int r = 0; r < D; ++r) { if (r == c) continue; const double f = G[(size_t)r * D + c]; if (f == 0.0) continue;
                        for (int k = 0; k < D; ++k) G[(size_t)r * D + k] -= f * G[(size_t)c * D + k]; for (int k = 0; k < NOBJ; ++k) B[(size_t)r * NOBJ + k] -= f * B[(size_t)c * NOBJ + k]; } }
                int ok = 0; for (int i = ntr; i < n; ++i) { int best = 0; double bv = -1e30;
                    for (int k = 0; k < NOBJ; ++k) { double z = B[(size_t)Dims::F * NOBJ + k]; for (int j = 0; j < Dims::F; ++j) z += X[i][j] * B[(size_t)j * NOBJ + k]; if (z > bv) { bv = z; best = k; } }
                    ok += best == Y[i]; }
                return 100.0 * ok / std::max(1, n - ntr); };
            std::cout << "DIAG (k=1): identity decodable (12 objects, chance 8.3%) from the object's word tile at its close / from the slot holding it at the\n"
                         "question; attention on the object's slot(s) vs uniform; argmax on it; p(first letter of the object)\n";
            for (int d : {0, 10, 30}) {
                std::mt19937_64 tg(5000 + d); std::vector<std::vector<float>> V, Q; std::vector<int> Y, YQ; double al_obj = 0, al_uni = 0, pfirst = 0; int top = 0, found = 0, n = 0;
                double r_obj = 0, r_uni = 0; int r_top = 0, r_found = 0;
                for (int i = 0; i < 450; ++i) { const Story st = story(tg, 1, d); const std::string ob = OBJECT[st.obj];
                    const size_t p = st.ctx.find(" " + ob + " .") + 1 + ob.size();   // the space after the object: its word closes when that space is read
                    obj_ev = model.events_seen() + p + 1;
                    feed_text(model, st.ctx.substr(0, p + 2));
                    const auto& q0 = model.tile_state().q[0]; if (!q0.empty() && q0.front().src == obj_ev) { V.emplace_back(q0.front().v, q0.front().v + Dims::F); Y.push_back(st.obj); }
                    feed_text(model, st.ctx.substr(p + 2)); model.stats() = BlockStats{}; feed_text(model, ob.substr(0, 1));   // the event that predicts the object
                    pfirst += std::exp(-model.stats().loss_sum); ++n;
                    if (!cal.empty()) { double ao = 0; int no = 0; size_t am = 0; for (size_t s = 0; s < cal.size(); ++s) { if (cobj[s]) { ao += cal[s]; ++no; } if (cal[s] > cal[am]) am = s; }
                        if (no) { ++found; al_obj += ao; al_uni += (double)no / cal.size(); top += cobj[am]; } }
                    if (!cv.empty()) { Q.push_back(cv); YQ.push_back(st.obj); }
                    if (!ral.empty()) { double ro = 0; int no = 0; size_t am = 0; for (size_t j = 0; j < ral.size(); ++j) { if (robj[j]) { ro += ral[j]; ++no; } if (ral[j] > ral[am]) am = j; }
                        if (no) { ++r_found; r_obj += ro; r_uni += (double)no / ral.size(); r_top += robj[am]; } }
                    feed_text(model, ob.substr(1) + " .\n"); }
                std::printf("d=%-3d identity at close %5.1f%% (%zu tiles), in the question's slot %5.1f%% (%zu) | attention on the object %.3f (uniform %.3f), argmax on it %d/%d | p(first letter) %.3f\n",
                            d, decode(V, Y), V.size(), decode(Q, YQ), Q.size(), found ? al_obj / found : 0.0, found ? al_uni / found : 0.0, top, found, pfirst / std::max(1, n));
                if (r_found || !ral.empty()) std::printf("      retrieval (tile_ret) on the object %.3f (uniform %.3f), argmax on it %d/%d\n", r_found ? r_obj / r_found : 0.0, r_found ? r_uni / r_found : 0.0, r_top, r_found);
                std::fflush(stdout);
            }
            return 0;
        }
        std::vector<std::string> arm = {"actual"};
        if (std::getenv("MEM_ARMS")) { if (model.has_tiles()) arm.push_back("no tile read"); if (model.params().ws_regs) arm.push_back("no register read"); }
        const int KS[] = {1, 2, 4}, DS[] = {0, 10, 30, 60, 120};
        std::cout << "recall: % correct object preferred over a distractor, by bindings k (rows) and filler words (columns 0 10 30 60 120)\n";
        for (size_t a = 0; a < arm.size(); ++a) {
            std::cout << "== " << arm[a] << "\n" << std::flush;
            for (int k : KS) {
                int fds[2]; if (pipe(fds) != 0) return 1; std::cout.flush(); pid_t pid = fork();
                if (pid == 0) {   // one (arm, k) row in a child; each item in a grandchild from the same frozen state
                    close(fds[0]);
                    if (arm[a] == "no tile read") model.diag_zero_tiles = true;
                    if (arm[a] == "no register read") model.set_diag_zero_regs(true);
                    char line[256]; int o = std::snprintf(line, sizeof line, "k=%d   ", k);
                    for (int d : DS) { std::mt19937_64 tg(1000 + 97 * k + d); int ok = 0, tot = 0;
                        for (int i = 0; i < 100; ++i) { const Story st = story(tg, k, d); double lp[2];
                            for (int c = 0; c < 2; ++c) { int p2[2]; if (pipe(p2) != 0) _exit(1); pid_t ch = fork();
                                if (ch == 0) { close(p2[0]); feed_text(model, st.ctx); model.stats() = BlockStats{}; feed_text(model, std::string(OBJECT[c ? st.distractor : st.obj]) + " .");
                                    double v = model.stats().loss_sum; ssize_t w = write(p2[1], &v, sizeof v); (void)w; _exit(0); }
                                close(p2[1]); double v = 0; ssize_t r = read(p2[0], &v, sizeof v); (void)r; close(p2[0]); waitpid(ch, nullptr, 0); lp[c] = -v; }
                            ok += lp[0] > lp[1]; ++tot; feed_text(model, "\n"); }
                        o += std::snprintf(line + o, sizeof line - o, " %5.1f", 100.0 * ok / std::max(1, tot)); }
                    std::snprintf(line + o, sizeof line - o, "\n"); ssize_t w = write(fds[1], line, std::strlen(line)); (void)w; _exit(0);
                }
                close(fds[1]); char buf[256] = {0}; ssize_t r = read(fds[0], buf, sizeof buf - 1); (void)r; close(fds[0]); waitpid(pid, nullptr, 0); std::cout << buf;
            }
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
