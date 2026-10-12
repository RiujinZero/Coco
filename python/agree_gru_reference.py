"""Float GRU reference on the agreement language (agree_probe / seq_probe): is the task learnable by a standard gated
recurrent network with full BPTT, at this data scale?  Prints the verb decision accuracy by distance.

usage: python agree_gru_reference.py [sentences=50000] [hidden=64] [task=verbs|isare]
"""
import math, random, sys
import torch, torch.nn as nn
torch.manual_seed(0); random.seed(0); torch.set_num_threads(4)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
H = int(sys.argv[2]) if len(sys.argv) > 2 else 64
task = sys.argv[3] if len(sys.argv) > 3 else 'verbs'
NOUN = [("dog", "dogs"), ("cat", "cats"), ("bird", "birds"), ("child", "children"), ("man", "men"), ("girl", "girls"), ("boy", "boys"),
        ("horse", "horses"), ("teacher", "teachers"), ("farmer", "farmers"), ("mouse", "mice"), ("friend", "friends")]
VERB = [("runs", "run"), ("sings", "sing"), ("walks", "walk"), ("sleeps", "sleep"), ("jumps", "jump"), ("waits", "wait"), ("laughs", "laugh"),
        ("swims", "swim"), ("reads", "read"), ("eats", "eat")]
FILL = "very old small red happy quiet near the big green tired brave from in town garden house river with blue tall kind long cold bright".split()
def sent(rng, d):
    n, num, v = rng.randrange(12), rng.randrange(2), rng.randrange(10)
    s = "the " + NOUN[n][num] + "".join(" " + rng.choice(FILL) for _ in range(d))
    return s, num, v
class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.e = nn.Embedding(256, 32); s.g = nn.GRU(32, H, batch_first=True); s.o = nn.Linear(H, 256)
    def forward(s, x): return s.o(s.g(s.e(x))[0])
m = LM(); opt = torch.optim.Adam(m.parameters(), lr=3e-3); rng = random.Random(7)
def enc(t): return torch.tensor([list(t.encode())], dtype=torch.long)
def evaluate():
    rt = random.Random(99); out = []
    with torch.no_grad():
        for d in (0, 1, 2, 4, 6, 8):
            ok = 0
            for _ in range(200):
                s, num, v = sent(rt, d)
                if task == 'isare': c = "\n" + s + " "; a, b = ord('i'), ord('a'); good = a if num == 0 else b; bad = b if num == 0 else a
                else: c = "\n" + s + " " + VERB[v][1]; good = ord('s') if num == 0 else ord(' '); bad = ord(' ') if num == 0 else ord('s')
                p = m(enc(c))[0, -1]; ok += p[good] > p[bad]
            out.append(f"d{d} {100 * ok / 200:.1f}")
    return "  ".join(out)
B = 32
for it in range(0, N, B):
    batch = []
    for _ in range(B):
        s, num, v = sent(rng, rng.randrange(9))
        batch.append("\n" + s + " " + (("is", "are")[num] if task == 'isare' else VERB[v][num]) + " .\n")
    L = max(len(b) for b in batch); x = torch.full((B, L), 10, dtype=torch.long)
    for i, b in enumerate(batch): x[i, :len(b)] = torch.tensor(list(b.encode()))
    loss = nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if (it // B + 1) % (N // B // 5) == 0: print(f"{it + B} sentences: loss {loss.item() / math.log(2):.3f} bits  {evaluate()}", flush=True)
