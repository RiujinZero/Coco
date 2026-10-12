"""Word n-gram reference on BabyLM Strict-Small: how much of BLiMP does purely LOCAL word context explain?
Interpolated absolute discounting (D = 0.75) over orders 1..N, lowercased whitespace/punctuation tokens, sentence starts.
Scores ITEMS_DIR/{blimp,entity_tracking}.jsonl with the pipeline rule (sum of completion log-probs; ties = expected pick).

usage: python3 ngram_blimp_reference.py CORPUS_DIR ITEMS_DIR [N=5]
"""
import collections, glob, json, math, os, re, sys
corpus, items_dir = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 5
tok = re.compile(r"[a-z0-9']+|[.,!?;:]")
def toks(s): return tok.findall(s.lower())
counts = [collections.Counter() for _ in range(N + 1)]      # counts[n][ngram tuple]
ctx = [collections.Counter() for _ in range(N + 1)]         # ctx[n][history of length n-1] -> total
types = [collections.defaultdict(set) for _ in range(N + 1)]
for f in sorted(glob.glob(os.path.join(corpus, '*.train.txt'))):
    for line in open(f, encoding='utf-8', errors='ignore'):
        w = ['<s>'] * (N - 1) + toks(line)
        for i in range(N - 1, len(w)):
            for n in range(1, N + 1):
                g = tuple(w[i - n + 1:i + 1]); counts[n][g] += 1; ctx[n][g[:-1]] += 1
for n in range(2, N + 1):
    for g in counts[n]: types[n][g[:-1]].add(g[-1])
V = len(counts[1]) + 1; D = 0.75
def prob(hist, w):
    p = (counts[1][(w,)] + 0.5) / (ctx[1][()] + 0.5 * V)
    for n in range(2, N + 1):
        h = tuple(hist[-(n - 1):]); c = ctx[n][h]
        if c == 0: continue
        lam = D * len(types[n][h]) / c
        p = max(counts[n][h + (w,)] - D, 0) / c + lam * p
    return p
def logprob(context, completion):
    hist = ['<s>'] * (N - 1) + toks(context); lp = 0.0
    for w in toks(completion): lp += math.log(prob(hist, w)); hist.append(w)
    return lp
for task in ('blimp', 'entity_tracking'):
    path = os.path.join(items_dir, task + '.jsonl')
    if not os.path.exists(path): continue
    items = collections.defaultdict(list)
    for line in open(path, encoding='utf-8'):
        r = json.loads(line); items[(r.get('uid'), r['item'])].append(r)
    per = collections.defaultdict(lambda: [0.0, 0])
    for (uid, _), cands in items.items():
        cands.sort(key=lambda r: r['cand']); s = [logprob(r['context'], r['completion']) for r in cands]; best = max(s)
        tied = [i for i, x in enumerate(s) if x == best]; acc = 1.0 / len(tied) if cands[0]['label'] in tied else 0.0
        per[uid][0] += acc; per[uid][1] += 1
    print(f'{task:18} word {N}-gram: {100 * sum(a / n for a, n in per.values()) / len(per):.2f}  ({len(items)} items)', flush=True)
