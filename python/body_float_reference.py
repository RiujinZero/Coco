"""Float reference for the Coco body: is the body SHAPE or the INT4 LEARNING the bottleneck?

Same inputs as the C++ `in_struct=1` body (16 features for each of the current byte and 4 before it, 16 word/sentence
position features; the workspace-memory cells are zero here), same layer widths and wiring
(L1 128->16; L2 [L1, return 16]->24; L3 [L1, message 24]->16; L5 [L2, L3]->16; L6 [L5, return 8]->32; return/message
inputs are zero as in the no-routing organism), a clamp nonlinearity, one body per current byte (the broad-body test in
`model_unit_check`), a shared 32->256 readout. Trained with float backprop + Adam, then scored on the held-out bytes.

usage: python body_float_reference.py TEXT [train_bytes=2097152] [heldout_bytes=262144] [epochs=1] [variant=...]
variants: clamp (the body's nonlinearity), tanh, wide (hidden widths x4), shared (ONE body for all bytes + a learned
current-byte embedding), mlp (plain 2-layer MLP on the same inputs, 256 hidden),
sgd (clamp body, plain SGD, batch 16: the organism's optimizer class), int4 (clamp body, weights on the INT4 grid
-8..7 x scale with a float shadow + straight-through: the organism's weight class), int4sgd (both).
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(4)
text = open(sys.argv[1], 'rb').read()
ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 2097152
nho = int(sys.argv[3]) if len(sys.argv) > 3 else 262144
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
variant = sys.argv[5] if len(sys.argv) > 5 else 'clamp'
data = np.frombuffer(text[:ntr + nho + 1], dtype=np.uint8)


def byte_feats(c):
    c = np.asarray(c, dtype=np.int64)
    bits = ((c[:, None] >> np.arange(8)) & 1).astype(np.float32)
    ch = c.astype(np.uint8)
    alpha = ((ch >= 65) & (ch <= 90)) | ((ch >= 97) & (ch <= 122))
    upper = (ch >= 65) & (ch <= 90)
    digit = (ch >= 48) & (ch <= 57)
    space = (ch == 32) | (ch == 9)
    punct = np.isin(ch, np.frombuffer(b'!#$%&()*+,-./:;<=>?@[\\]^_`{|}~', dtype=np.uint8))
    low = np.where(upper, ch + 32, ch)
    vowel = alpha & np.isin(low, np.frombuffer(b'aeiou', dtype=np.uint8))
    nl = (ch == 10) | (ch == 13)
    quote = (ch == 39) | (ch == 34)
    cls = np.stack([alpha, upper, digit, space, punct, vowel, nl, quote], 1).astype(np.float32)
    return np.concatenate([bits, cls], 1) * 2 - 1


def features(idx):
    """inputs for predicting data[idx+1] given data[..idx] (idx = position of the current byte)"""
    f = [byte_feats(data[np.maximum(idx - k, 0)]) for k in range(5)]
    hist = np.stack([data[np.maximum(idx - k, 0)] for k in range(16)], 1)
    isal = ((hist >= 65) & (hist <= 90)) | ((hist >= 97) & (hist <= 122))
    pos = np.argmin(np.concatenate([isal, np.zeros((len(idx), 1), bool)], 1), 1)
    sent = np.argmax(np.concatenate([np.isin(hist, [46, 33, 63, 10]), np.ones((len(idx), 1), bool)], 1), 1)
    prev = np.zeros(len(idx))   # previous word length (approximate, as in C++): skip non-letters after the current word
    for r in range(len(idx)):
        j = pos[r]
        while j < 15 and not isal[r, j]: j += 1
        k = 0
        while j + k < 15 and isal[r, j + k]: k += 1
        prev[r] = k
    w = np.concatenate([(pos[:, None] > np.arange(6)), (prev[:, None] > np.arange(1, 6)), (sent[:, None] > (1 << np.arange(5)))], 1).astype(np.float32) * 2 - 1
    return np.concatenate(f + [w], 1).astype(np.float32)   # 96 features


t0 = time.time()
idx_tr = np.arange(0, ntr); idx_ho = np.arange(ntr, ntr + nho)
X_tr = torch.from_numpy(features(idx_tr)); X_ho = torch.from_numpy(features(idx_ho))
c_tr = torch.from_numpy(data[idx_tr].astype(np.int64)); c_ho = torch.from_numpy(data[idx_ho].astype(np.int64))
y_tr = torch.from_numpy(data[idx_tr + 1].astype(np.int64)); y_ho = torch.from_numpy(data[idx_ho + 1].astype(np.int64))
print(f'features {X_tr.shape} in {time.time() - t0:.1f}s', flush=True)

m = 4 if variant == 'wide' else 1
A, B, C, E, F = 16 * m, 24 * m, 16 * m, 16 * m, 32
act = torch.tanh if variant == 'tanh' else (lambda x: torch.clamp(x, -1, 1))
quant = variant.startswith('int4')
cppfmt = variant.startswith('cpp')
import os
SH_ADD = int(os.environ.get('COCO_SHIFT_ADD', '0'))      # extra right shift per layer (less saturation)
INIT_R = int(os.environ.get('COCO_INIT', '2'))           # birth weights uniform in -R..R
NOROUND = os.environ.get('COCO_NOROUND', '0') == '1'     # keep activations continuous (no int8 rounding)
X_SCALE = float(os.environ.get('COCO_XSCALE', '32'))   # 'cpp' / 'cppsgd' / 'cppseq': the exact C++ integer body format


def ste_int(pre):   # round (straight-through) then clamp to +-127 (zero gradient outside, as in C++ backprop)
    r = pre if NOROUND else pre + (torch.round(pre) - pre).detach()
    return torch.clamp(r, -127, 127)   # suffix 'seq' = online in text order (no shuffling), like the organism


def q4(W):   # INT4 grid per layer: 16 levels spanning +-2.5 std of the init, straight-through gradient
    s = 2.5 / math.sqrt(W.shape[-1]) / 8
    return W + (torch.clamp(torch.round(W / s), -8, 7) * s - W).detach()


class Bodies(nn.Module):
    def __init__(self, n):
        super().__init__()
        def P(o, i):
            if cppfmt: return nn.Parameter(torch.randint(-INIT_R, INIT_R + 1, (n, o, i)).float())   # C++ birth weights: integers -2..2 of the INT4 range
            return nn.Parameter(torch.randn(n, o, i) / math.sqrt(i))
        self.W1, self.W2, self.W3, self.W5, self.W6 = P(A, 96), P(B, A), P(C, A), P(E, B + C), P(F, E)
        self.out = nn.Linear(F, 256)
        self.emb = nn.Embedding(256, 96) if n == 1 else None

    def forward(self, x, c):
        if self.emb is not None: x = x + self.emb(c); c = torch.zeros_like(c)
        if cppfmt:   # the C++ integer forward: x in +-32, INT4 weights (-8..7, STE), y = clamp(round(W x / 2^shift), +-127), readout on y/128
            qi = lambda W: (W + (torch.clamp(torch.round(W), -8, 7) - W).detach())[c]
            lay = lambda W, x, sh: ste_int(torch.bmm(qi(W), x[:, :, None])[..., 0] / (1 << (sh + SH_ADD)))
            x = x * X_SCALE
            l1 = lay(self.W1, x, 1); l2 = lay(self.W2, l1, 2); l3 = lay(self.W3, l1, 2)
            l5 = lay(self.W5, torch.cat([l2, l3], 1), 3); l6 = lay(self.W6, l5, 3)
            return self.out(l6 / 128)
        g = lambda W: (q4(W) if quant else W)[c]
        l1 = act(torch.bmm(g(self.W1), x[:, :, None])[..., 0])
        l2 = act(torch.bmm(g(self.W2), l1[:, :, None])[..., 0])
        l3 = act(torch.bmm(g(self.W3), l1[:, :, None])[..., 0])
        l5 = act(torch.bmm(g(self.W5), torch.cat([l2, l3], 1)[:, :, None])[..., 0])
        l6 = act(torch.bmm(g(self.W6), l5[:, :, None])[..., 0])
        return self.out(l6)


if variant == 'mlp':
    model = nn.Sequential(nn.Linear(96 + 256, 256), nn.Tanh(), nn.Linear(256, 256))
    fwd = lambda x, c: model(torch.cat([x, nn.functional.one_hot(c, 256).float()], 1))
else:
    model = Bodies(1 if variant == 'shared' else 256)
    fwd = model
nparam = sum(p.numel() for p in model.parameters())
seq = variant.endswith('seq'); variant_core = variant[:-3] if seq else variant
sgd = variant_core.endswith('sgd')
opt = torch.optim.SGD(model.parameters(), lr=0.05 * (20 if cppfmt else 1)) if sgd else torch.optim.Adam(model.parameters(), lr=2e-3 * (10 if cppfmt else 1))
bs = 16 if sgd else 256   # 'seq': same batch size, but consecutive bytes (text order) instead of shuffled
for ep in range(epochs):
    perm = torch.arange(ntr) if seq else torch.randperm(ntr)
    for s in range(0, ntr, bs):
        b = perm[s:s + bs]
        loss = nn.functional.cross_entropy(fwd(X_tr[b], c_tr[b]), y_tr[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ho = sum(nn.functional.cross_entropy(fwd(X_ho[s:s + 4096], c_ho[s:s + 4096]), y_ho[s:s + 4096], reduction='sum').item() for s in range(0, nho, 4096))
    print(f'{variant} params {nparam:,} epoch {ep + 1}: held-out {ho / nho / math.log(2):.4f} bits/byte ({time.time() - t0:.0f}s)', flush=True)
