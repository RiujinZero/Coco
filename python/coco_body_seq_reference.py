"""Float (and INT4-grid) replica of cpp/apps/seq_probe.cpp: Coco-shaped bodies (one per current byte, L1..L6 with the
clamp nonlinearity, same widths) + 4 gated registers written by a shared head from L6, read back as input cells, full BPTT.
Question: can the BODY ARCHITECTURE learn the agreement language (as the GRU does), independent of the C++ update path?

usage: python coco_body_seq_reference.py [sentences=400000] [task=isare|verbs] [variant=float|int4|noregs|gru_like]
"""
import math, random, sys
import torch, torch.nn as nn
import os as _os
SEED = int(_os.environ.get('SEED', '0')); torch.manual_seed(SEED); random.seed(SEED); torch.set_num_threads(2)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
task = sys.argv[2] if len(sys.argv) > 2 else 'isare'
variant = sys.argv[3] if len(sys.argv) > 3 else 'float'
NOUN = [("dog", "dogs"), ("cat", "cats"), ("bird", "birds"), ("child", "children"), ("man", "men"), ("girl", "girls"), ("boy", "boys"),
        ("horse", "horses"), ("teacher", "teachers"), ("farmer", "farmers"), ("mouse", "mice"), ("friend", "friends")]
VERB = [("runs", "run"), ("sings", "sing"), ("walks", "walk"), ("sleeps", "sleep"), ("jumps", "jump"), ("waits", "wait"), ("laughs", "laugh"),
        ("swims", "swim"), ("reads", "read"), ("eats", "eat")]
FILL = "very old small red happy quiet near the big green tired brave from in town garden house river with blue tall kind long cold bright".split()
def sent(rng, d):
    n, num, v = rng.randrange(12), rng.randrange(2), rng.randrange(10)
    return "the " + NOUN[n][num] + "".join(" " + rng.choice(FILL) for _ in range(d)), num, v
def feats(c):   # 16 features of a byte, +-1 (the C++ harness's feats() / 32)
    al = chr(c).isalpha(); f = [((c >> b) & 1) for b in range(8)] + [al, chr(c).isupper(), c == 32, c == 10, c == 46, al and chr(c).lower() in 'aeiou', c == 115, 0]
    return [2.0 * x - 1.0 for x in f]
FEAT = torch.tensor([feats(c) for c in range(256)])
A, B_, C, E, F, NR, WIN = 16, 24, 16, 16, 32, 4, 3
IN = 16 * WIN + NR
regs = variant != 'noregs'; q4 = variant.startswith('int4') or variant == 'int8'
CPP = variant == 'cpp'   # the C++ harness's exact structure: +-32 inputs, integer weights (init {-2..2}, STE, [-8,7]), >> shifts, clamp +-127, readout /128
SH = [int(v) for v in _os.environ.get('SH', '1,2,2,3,3').split(',')]; RSCALE = float(_os.environ.get('RSCALE', '50'))
rowscale = variant == 'int4row'; import os; RGAIN = float(os.environ.get('RGAIN', '1')); LEV = (127, 128) if variant == 'int8' else (7, 8); GRID = 1.5 if variant == 'int4fine' else 2.5
def P(*s): return nn.Parameter(torch.randn(*s) / math.sqrt(s[-1]))
class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.W1, s.W2, s.W3, s.W5, s.W6 = P(256, A, IN), P(256, B_, A), P(256, C, A), P(256, E, B_ + C), P(256, F, E)
        s.out = nn.Linear(F, 256); s.hg = nn.Linear(F, NR); s.hv = nn.Linear(F, NR)
        with torch.no_grad(): s.hg.bias.fill_(-2.0)
        # int4row: a learned float scale per output row (per neuron), the usual INT4 practice; weights stay on the INT4 grid
        if CPP:
            with torch.no_grad():
                for W in (s.W1, s.W2, s.W3, s.W5, s.W6):
                    if _os.environ.get('CPP_INIT') == 'randn': W.copy_(torch.clamp(torch.round(torch.randn(W.shape) * float(_os.environ.get('CPP_ISD', '3.2'))), -8, 7))
                    else: W.copy_(torch.randint(-2, 3, W.shape).float())
        s.rs = nn.ParameterList([nn.Parameter(torch.full((256, W.shape[1], 1), GRID / math.sqrt(W.shape[-1]) / LEV[1])) for W in (s.W1, s.W2, s.W3, s.W5, s.W6)])
    def q(s, W, k=0):
        if CPP: return torch.clamp(torch.round(W).detach() + (W - W.detach()), -8, 7)
        if not q4: return W
        sc = s.rs[k] if rowscale else GRID / math.sqrt(W.shape[-1]) / LEV[1]
        qW = torch.clamp(torch.round(W / sc).detach() + (W / sc - (W / sc).detach()), -LEV[1], LEV[0])   # STE on the rounding
        return qW * sc
    def forward(s, X):   # X: [batch, T] bytes; predicts X[:, t+1] from steps <= t
        Bn, T = X.shape; reg = torch.zeros(Bn, NR); outs = []
        LEAK = float(_os.environ.get('LEAK', '0'))   # backward-only slope outside the clamp (forward stays a hard clamp)
        def act(z):
            if CPP:
                c = torch.clamp(z, -127, 127)
                return c if LEAK <= 0 else c + LEAK * (z.abs() > 127).float() * (z - z.detach())
            if _os.environ.get('ACT') == 'tanh': return torch.tanh(z)
            c = torch.clamp(z, -1, 1)
            if LEAK <= 0: return c
            return c + LEAK * (z.abs() > 1).float() * (z - z.detach())
        for t in range(T):
            cur = X[:, t]; parts = [FEAT[X[:, max(t - k, 0)]] for k in range(WIN)]
            x = torch.cat(parts + ([reg / 100.0 * RGAIN] if regs else [torch.zeros(Bn, NR)]), 1)
            if CPP: x = torch.cat([x[:, :16 * WIN] * float(_os.environ.get('CPP_XS', '32')), torch.clamp(reg * RSCALE / 100, -127, 127) if regs else torch.zeros(Bn, NR)], 1)
            sc = [1.0 / (1 << k) for k in SH] if CPP else [1.0] * 5
            Ws = [s.W1, s.W2, s.W3, s.W5, s.W6]; g = lambda W: s.q(W, next(i for i, w in enumerate(Ws) if w is W))[cur]
            l1 = act(sc[0] * torch.bmm(g(s.W1), x[:, :, None])[..., 0]); l2 = act(sc[1] * torch.bmm(g(s.W2), l1[:, :, None])[..., 0])
            l3 = act(sc[2] * torch.bmm(g(s.W3), l1[:, :, None])[..., 0]); l5 = act(sc[3] * torch.bmm(g(s.W5), torch.cat([l2, l3], 1)[:, :, None])[..., 0])
            l6 = act(sc[4] * torch.bmm(g(s.W6), l5[:, :, None])[..., 0])
            if CPP: s.last = (x / 127, l1 / 127, l2 / 127, l3 / 127, l5 / 127, l6 / 127); l6 = l6 / 128
            if not CPP: s.last = (x, l1, l2, l3, l5, l6)   # DIAG: the last step's activations (clamped; |y| == 1 means saturated)
            outs.append(s.out(l6))
            if regs:
                gt = torch.sigmoid(s.hg(l6)); v = 100 * torch.tanh(s.hv(l6)); reg = (1 - gt) * reg + gt * v
        return torch.stack(outs, 1)
m = Model(); OPT = _os.environ.get('OPT', 'adam')
class RowRMS(torch.optim.Optimizer):   # one running gradient scale per NEURON (row), steps normalised by it: 128 numbers per body
    def __init__(s, params, lr=2e-3, beta=0.99): super().__init__(params, dict(lr=lr, beta=beta))
    @torch.no_grad()
    def step(s):
        for gr in s.param_groups:
            for p_ in gr['params']:
                if p_.grad is None: continue
                st = s.state[p_]; g = p_.grad; r = (g * g).mean(dim=-1, keepdim=True)
                if 'v' not in st: st['v'] = r.clone()
                st['v'].mul_(gr['beta']).add_((1 - gr['beta']) * r)
                p_.add_(-gr['lr'] * g / (st['v'].sqrt() + 1e-8))
body_params = [m.W1, m.W2, m.W3, m.W5, m.W6]; other = [q for q in m.parameters() if all(q is not b for b in body_params)]
if OPT == 'rowrms': opts = [RowRMS(body_params, lr=float(_os.environ.get('RLR', '2e-3'))), torch.optim.Adam(other, lr=2e-3)]
elif OPT == 'adam' and CPP: opts = [torch.optim.Adam(body_params, lr=float(_os.environ.get('ALR', '0.03'))), torch.optim.Adam(other, lr=2e-3)]
elif OPT == 'adam': opts = [torch.optim.Adam(m.parameters(), lr=2e-3)]
else: opts = [torch.optim.SGD(m.parameters(), lr=float(_os.environ.get('SGD_LR', '0.1')))]
class _Multi:
    def zero_grad(s): [o.zero_grad() for o in opts]
    def step(s): [o.step() for o in opts]
opt = _Multi(); rng = random.Random(7 + SEED)
def evaluate():
    rt = random.Random(99); res = []
    with torch.no_grad():
        for d in (0, 1, 2, 4, 6, 8):
            ctxs, nums = [], []
            for _ in range(200):
                s, num, v = sent(rt, d)
                ctxs.append("\n" + s + (" " if task == 'isare' else " " + VERB[v][1])); nums.append(num)
            L = max(len(c) for c in ctxs); ok = 0
            for c, num in zip(ctxs, nums):
                p = m(torch.tensor([list(c.encode())]))[0, -1]
                if task == 'isare': ok += (p[ord('a')] > p[ord('i')]) if num == 1 else (p[ord('i')] > p[ord('a')])
                else: ok += (p[ord(' ')] > p[ord('s')]) if num == 1 else (p[ord('s')] > p[ord(' ')])
            res.append(f"d{d} {100 * ok / 200:.1f}")
            if d == 0 and _os.environ.get('DIAG'):   # per layer at the decision step: saturated fraction and sg/pl separation (in clamp units)
                acts = {0: [], 1: []}
                for c, num in zip(ctxs, nums): m(torch.tensor([list(c.encode())])); acts[num].append([a_[0].clone() for a_ in m.last])
                diag = []
                for L, nm in enumerate(('x', 'l1', 'l2', 'l3', 'l5', 'l6')):
                    A0 = torch.stack([v[L] for v in acts[0]]); A1 = torch.stack([v[L] for v in acts[1]]); Al = torch.cat([A0, A1])
                    diag.append(f"{nm} sat {100 * (Al.abs() >= 0.999).float().mean():.0f}% sep {(A1.mean(0) - A0.mean(0)).abs().mean() * 127:.0f}")
                res.append('[' + ' '.join(diag) + ']')
    return "  ".join(res)
BS = int(_os.environ.get('BS', '32'))
for it in range(0, N, BS):
    batch = []
    for _ in range(BS):
        s, num, v = sent(rng, rng.randrange(9))
        batch.append("\n" + s + " " + (("is", "are")[num] if task == 'isare' else VERB[v][num]) + " .\n")
    L = max(len(b) for b in batch); X = torch.full((BS, L), 10, dtype=torch.long)
    for i, b in enumerate(batch): X[i, :len(b)] = torch.tensor(list(b.encode()))
    tgt = X[:, 1:].clone()
    for i, b in enumerate(batch): tgt[i, len(b) - 1:] = -100   # padding excluded (earlier runs included it: deflated loss)
    loss = nn.functional.cross_entropy(m(X[:, :-1]).reshape(-1, 256), tgt.reshape(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
    if (it // BS + 1) % (N // BS // 8) == 0: print(f"{variant} g{RGAIN:g} s{SEED} {OPT} leak{_os.environ.get('LEAK','0')} {_os.environ.get('ACT','clamp')} {task} {it + BS}: loss {loss.item() / math.log(2):.3f}  {evaluate()}", flush=True)
