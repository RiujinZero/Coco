#!/usr/bin/env python3
"""Selection for the prompt-following objective with the model-free topic judge (scripts/topic_judge.py) as the reward.

Same candidates as hloop_select.py (fluent, non-degenerate: no repeated word 4-gram, >= 85% valid words, >= 5 words, no word
3-gram copied from the prompt, lp_self >= the prompt's median); per prompt the kept sample has the largest topic + topic_late
(corpus NPMI of its new content words with the prompt's; real continuations score ~0.10 / 0.09, another line's text ~0.04).
Run with the Windows Python (numpy), paths through \\\\wsl.localhost\\Ubuntu\\...

usage: hloop_topic_select.py JUDGE_PREFIX OUT_TEXT SAMPLES.jsonl [SAMPLES.jsonl ...]
"""
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topic_judge  # noqa: E402


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def ngrams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def main():
    judge = topic_judge.Judge(sys.argv[1])
    out_path, files = sys.argv[2], [f for f in sys.argv[3:] if not f.endswith(".topic.jsonl")]
    by = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            by.setdefault(r["prompt"], []).append(r)
    kept, allr = [], []
    for prompt, rs in by.items():
        med = statistics.median(r["lp_self"] for r in rs)
        p3 = ngrams(words(prompt), 3)
        cands = [r for r in rs if r["loop4"] == 0 and r["valid_words"] >= 0.85 and r["words"] >= 5 and r["lp_self"] >= med
                 and not (ngrams(words(r["output"]), 3) & p3)]
        for r in cands:
            r["topic"], r["topic_late"] = judge.score(r["prompt"], r["output"])
        allr += cands
        if cands:
            kept.append(max(cands, key=lambda r: r["topic"] + r["topic_late"]))
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:   # cut at the last line / sentence end (else word end)
        for r in kept:
            o = r["output"]
            cut = max(o.rfind("\n"), o.rfind(". "), o.rfind("! "), o.rfind("? "))
            o = o[:cut + 1] if cut >= len(o) // 2 else o[:max(1, o.rfind(" "))]
            out.write((r["prompt"] + o).rstrip("\n") + "\n")

    def mean(rs, k):
        return sum(r[k] for r in rs) / max(1, len(rs))
    print(f"topic: {len(by)} prompts, {len(allr)} fluent candidates, {len(kept)} kept -> {out_path}")
    for name, rs in (("candidates", allr), ("kept", kept)):
        print(f"  {name:12s} topic {mean(rs, 'topic'):.4f}  topic_late {mean(rs, 'topic_late'):.4f}  gain {mean(rs, 'gain'):.4f}  "
              f"gain_late {mean(rs, 'gain_late'):.4f}  lp_self {mean(rs, 'lp_self'):.3f}  valid {mean(rs, 'valid_words'):.3f}")


if __name__ == "__main__":
    main()
