#!/usr/bin/env python3
"""A model-free topicality judge for item H: does a continuation stay on its prompt's topic? (numpy; run it with the Windows
Python, which reads the WSL files through \\\\wsl.localhost\\Ubuntu\\...)

Built from BabyLM's training text (the held-out 0.5% of every file excluded, as ms_babylm cuts it): content words (>= 3
letters, seen >= 5 times, not among the 150 most frequent words) and their co-occurrence within non-overlapping windows of 3
consecutive lines. Relatedness of two words = NPMI = log(p(a,b) / (p(a) p(b))) / -log p(a,b), in [-1, 1] (0: independent;
pairs never seen together count 0). A continuation's TOPIC score = the mean, over its content words that are NOT in the prompt,
of their largest NPMI with any prompt content word; topic_late: the same over the continuation's second half. Copying the
prompt earns nothing (its words are excluded), and the judge does not depend on the model being judged.

usage: topic_judge.py build CORPUS_DIR OUT_PREFIX
       topic_judge.py score OUT_PREFIX RUN.jsonl [RUN.jsonl ...]     (per file and mode: mean topic / topic_late; writes RUN.topic.jsonl)
       topic_judge.py paired OUT_PREFIX BASE.jsonl ARM.jsonl [...]  (per prompt and mode: ARM - BASE, mean +- standard error)
"""
import collections
import json
import math
import re
import sys

import numpy as np

FILES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
WORD = re.compile(r"[a-z]+")


def train_part(text):
    cut = len(text) - len(text) // 200
    cut = text.rfind("\n", 0, cut + 1)
    return text[:cut + 1] if cut >= 0 else text


def build(corpus, out):
    lines = []
    for f in FILES:
        lines += [WORD.findall(l.lower()) for l in train_part(open(f"{corpus}/{f}.train.txt", encoding="utf-8", errors="replace").read()).split("\n")]
    freq = collections.Counter(w for ws in lines for w in ws if len(w) >= 3)
    stop = {w for w, _ in freq.most_common(150)}
    words = [w for w, c in freq.most_common() if c >= 5 and w not in stop]
    vocab = {w: i for i, w in enumerate(words)}
    V = len(words)
    ids = [{vocab[w] for w in ws if w in vocab} for ws in lines]
    unit = np.zeros(V, dtype=np.int64)
    chunks, buf, nbuf, n = [], [], 0, 0
    for i in range(0, len(ids) - 2, 3):   # non-overlapping windows of 3 lines
        s = np.array(sorted(ids[i] | ids[i + 1] | ids[i + 2]), dtype=np.int64)
        if len(s) == 0:
            continue
        n += 1
        unit[s] += 1
        if len(s) < 2 or len(s) > 60:
            continue
        a, b = np.triu_indices(len(s), 1)
        buf.append(s[a] * V + s[b]); nbuf += len(a)
        if nbuf > 4_000_000:
            c, k = np.unique(np.concatenate(buf), return_counts=True); chunks.append((c, k)); buf, nbuf = [], 0
    if buf:
        c, k = np.unique(np.concatenate(buf), return_counts=True); chunks.append((c, k))
    codes = np.concatenate([c for c, _ in chunks]); counts = np.concatenate([k for _, k in chunks])
    order = np.argsort(codes, kind="stable"); codes, counts = codes[order], counts[order]
    uniq, start = np.unique(codes, return_index=True)
    counts = np.add.reduceat(counts, start)
    np.savez(out + ".npz", codes=uniq, counts=counts.astype(np.int32), unit=unit.astype(np.int32), n=np.array([n]))
    json.dump(words, open(out + ".words.json", "w", encoding="utf-8"))
    print(f"{V} content words, {n} windows, {len(uniq)} co-occurring pairs -> {out}.npz")


class Judge:
    def __init__(self, prefix):
        d = np.load(prefix + ".npz")
        self.codes, self.counts, self.unit, self.n = d["codes"], d["counts"], d["unit"].astype(np.float64), float(d["n"][0])
        self.vocab = {w: i for i, w in enumerate(json.load(open(prefix + ".words.json", encoding="utf-8")))}
        self.V = len(self.vocab)

    def content(self, text):
        return [self.vocab[w] for w in WORD.findall(text.lower()) if w in self.vocab]

    def best_npmi(self, p, ws):   # for every word in ws: its largest NPMI with any word of p
        P = np.array(sorted(p), dtype=np.int64); W = np.array(ws, dtype=np.int64)
        a = np.minimum(P[:, None], W[None, :]); b = np.maximum(P[:, None], W[None, :])
        code = a * self.V + b
        pos = np.searchsorted(self.codes, code); pos = np.minimum(pos, len(self.codes) - 1)
        c = np.where(self.codes[pos] == code, self.counts[pos], 0).astype(np.float64)
        pab = c / self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log(pab / ((self.unit[a] / self.n) * (self.unit[b] / self.n))) / -np.log(pab)
        v = np.where(c > 0, v, 0.0)
        return v.max(axis=0)

    def score(self, prompt, cont):
        p = set(self.content(prompt))

        def part(t):
            ws = [w for w in self.content(t) if w not in p]
            return float(self.best_npmi(p, ws).mean()) if p and ws else 0.0
        return part(cont), part(cont[len(cont) // 2:])


def score_files(judge, files):
    out = {}
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        for r in rows:
            r["topic"], r["topic_late"] = judge.score(r["prompt"], r["output"])
        with open(f.replace(".jsonl", ".topic.jsonl"), "w", encoding="utf-8") as o:
            for r in rows:
                o.write(json.dumps(r) + "\n")
        out[f] = rows
    return out


def main():
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
        return
    judge = Judge(sys.argv[2])
    files = [f for f in sys.argv[3:] if not f.endswith(".topic.jsonl")]
    rows = score_files(judge, files)
    if sys.argv[1] == "score":
        for f, rs in rows.items():
            for mode in dict.fromkeys(r["mode"] for r in rs):
                m = [r for r in rs if r["mode"] == mode]
                name = f if len(f) <= 44 else "..." + f[-41:]
                print(f"{name:44s} {mode:7s} n={len(m):4d} topic {sum(r['topic'] for r in m) / len(m):.4f}  topic_late {sum(r['topic_late'] for r in m) / len(m):.4f}")
    else:
        base = {(r["prompt"], r["mode"]): r for r in rows[files[0]]}
        for f in files[1:]:
            for mode in dict.fromkeys(r["mode"] for r in rows[f]):
                res = []
                for k in ("topic", "topic_late"):
                    d = [r[k] - base[(r["prompt"], mode)][k] for r in rows[f] if r["mode"] == mode and (r["prompt"], mode) in base]
                    mu = sum(d) / len(d)
                    se = math.sqrt(sum((x - mu) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
                    res.append(f"{k} {mu:+.4f}+-{se:.4f}")
                name = f if len(f) <= 44 else "..." + f[-41:]
                print(f"{name:44s} {mode:7s} " + "  ".join(res))


if __name__ == "__main__":
    main()
