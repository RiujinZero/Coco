#!/usr/bin/env python3
"""Topic-weighted replay for the prompt-following objective (item H), on REAL text: blocks of consecutive BabyLM training lines,
each line weighted by how strongly it continues the topic of the 3 lines before it (the topic judge's score with those lines
as the prompt: corpus NPMI of the line's new content words with theirs). The loss on lines that carry the topic forward
counts more, on lines that drop it less - the model is pushed to use its context where the context decides the content,
with no self-generated text (no drift toward its own register).

Weight = 1 + BETA x z, z = the line's standardised topicality clipped to [-1, 2] (BETA 0.5: weights 0.5 .. 2); lines with no
content word, and the first 3 lines of a block, weight 1. Output: MS_EXTRA text and MS_EXTRA_W records ("0 TLEN W" per line).
FLAT=1 writes the same blocks with every weight 1 (the control). Run with the Windows Python (numpy).

usage: [FLAT=1] topic_weight_corpus.py CORPUS_DIR JUDGE_PREFIX OUT_TEXT OUT_WEIGHTS MBYTES [BETA] [SEED]
"""
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topic_judge  # noqa: E402

BLOCK = 60   # lines per block


def main():
    corpus, jp, out_text, out_w, mbytes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
    beta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    rng = random.Random(int(sys.argv[7]) if len(sys.argv) > 7 else 11)
    flat = os.environ.get("FLAT") == "1"
    judge = topic_judge.Judge(jp)
    blocks = []
    for f in topic_judge.FILES:
        lines = topic_judge.train_part(open(f"{corpus}/{f}.train.txt", encoding="utf-8", errors="replace").read()).split("\n")
        starts = list(range(0, max(1, len(lines) - BLOCK), BLOCK)); rng.shuffle(starts)
        blocks += [(f, lines[s:s + BLOCK]) for s in starts[:400]]
    rng.shuffle(blocks)
    budget, recs, scores = int(mbytes * 1e6), [], []
    for f, ls in blocks:
        if budget <= 0:
            break
        for i, line in enumerate(ls):
            rec = line + "\n"
            s = None
            if i >= 3 and not flat:
                prev = " ".join(ls[i - 3:i]); p = set(judge.content(prev)); ws = [w for w in judge.content(line) if w not in p]
                if p and ws:
                    s = float(judge.best_npmi(p, ws).mean()); scores.append(s)
            recs.append((rec, s)); budget -= len(rec.encode("utf-8"))
    mu = statistics.mean(scores) if scores else 0.0; sd = statistics.pstdev(scores) if len(scores) > 1 else 1.0
    ws_all = []
    with open(out_text, "w", encoding="utf-8", newline="\n") as ot, open(out_w, "w", encoding="utf-8", newline="\n") as ow:
        for rec, s in recs:
            w = 1.0 if s is None else 1.0 + beta * max(-1.0, min(2.0, (s - mu) / (sd + 1e-9)))
            ot.write(rec); ow.write(f"0 {len(rec.encode('utf-8'))} {w:.4f}\n"); ws_all.append(w)
    print(f"{'flat' if flat else 'topic-weighted'}: {len(recs)} lines from {len(blocks)} candidate blocks, {sum(len(r.encode('utf-8')) for r, _ in recs)} bytes; "
          f"line topicality mean {mu:.4f} sd {sd:.4f}; weights mean {statistics.mean(ws_all):.3f} min {min(ws_all):.2f} max {max(ws_all):.2f} -> {out_text}")


if __name__ == "__main__":
    main()
