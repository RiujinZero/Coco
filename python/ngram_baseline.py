"""Online byte n-gram baseline on the same stream the organism sees (one pass, learn-as-you-go).

Interpolated Kneser-Ney-free "simple backoff" estimator: for each order k = 0..N, count-based
P_k(next | last k bytes); the prediction blends orders with weights proportional to their evidence
(Witten-Bell style: lambda_k = n_k / (n_k + types_k)). Reports BPB per window and overall.

    python python/ngram_baseline.py data/tinystories_13m.txt --order 5 --window 2097152
"""
import argparse
import math
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--window", type=int, default=2 * 1024 * 1024)
    ap.add_argument("--max_bytes", type=int, default=0)
    a = ap.parse_args()
    data = open(a.file, "rb").read()
    if a.max_bytes:
        data = data[: a.max_bytes]
    N = a.order
    # counts[k][context] -> {byte: count}; types[k][context] -> number of distinct next bytes
    counts = [defaultdict(lambda: defaultdict(int)) for _ in range(N + 1)]
    totals = [defaultdict(int) for _ in range(N + 1)]
    V = 256
    loss = 0.0
    wloss = 0.0
    wn = 0
    correct = 0
    n = len(data)
    for t in range(n - 1):
        y = data[t + 1]
        # predict: blend from highest order down
        p = 1.0 / V
        # start from order 0 up, Witten-Bell interpolation: P_k = lam*ML_k + (1-lam)*P_{k-1}
        for k in range(0, N + 1):
            if t + 1 < k:
                break
            ctx = data[t + 1 - k: t + 1]
            tot = totals[k].get(ctx, 0)
            if tot == 0:
                break
            c = counts[k][ctx]
            types = len(c)
            lam = tot / (tot + types)
            ml = c.get(y, 0) / tot
            p = lam * ml + (1.0 - lam) * p
        loss += -math.log(max(p, 1e-12))
        wloss += -math.log(max(p, 1e-12))
        wn += 1
        # argmax check (approximate: use highest available order's argmax)
        for k in range(N, -1, -1):
            if t + 1 < k:
                continue
            ctx = data[t + 1 - k: t + 1]
            if totals[k].get(ctx, 0) > 0:
                c = counts[k][ctx]
                correct += max(c.items(), key=lambda kv: kv[1])[0] == y
                break
        # learn
        for k in range(0, N + 1):
            if t + 1 < k:
                break
            ctx = data[t + 1 - k: t + 1]
            counts[k][ctx][y] += 1
            totals[k][ctx] += 1
        if a.window and (t + 1) % a.window == 0:
            print(f"  {(t + 1) / 1048576:6.2f} MB | window BPB {wloss / wn / math.log(2):.4f}", flush=True)
            wloss, wn = 0.0, 0
    m = n - 1
    print(f"NGRAM order {N}: {m} bytes, BPB {loss / m / math.log(2):.4f}, acc {100 * correct / m:.2f}%")


if __name__ == "__main__":
    main()
