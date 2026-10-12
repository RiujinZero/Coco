#!/usr/bin/env python3
"""Selection for the prompt-following objective (item H): filtered self-training (expert iteration).

Reads model_frontier sample runs (one JSON line per prompt, several seeds) and, per prompt, keeps ONE of its samples as
training text "prompt + continuation". Candidates must be fluent and non-degenerate: no repeated word 4-gram (loop4 = 0),
>= 85% valid words, >= 5 words, no word 3-gram copied from the prompt, and a self log-prob (the model's own likelihood of
the sample, lp_self) at least the median of that prompt's samples. Among them the arm picks:
  select   the sample the prompt matters most for: reward = gain + LATE x gain_late (the log-prob the prompt adds per byte,
           over the whole sample and over its second half; LATE, default 3, evens out their spreads so that keeping to the
           prompt past the first line counts as much as the first line), and only if the reward is > 0 - the objective
  random   a uniformly random candidate (the same filters: self-training without the prompt-following pressure)
  fluent   the candidate with the highest lp_self (best-of-N by the model's own likelihood)

usage: [LATE=3] hloop_select.py ARM OUT_TEXT SAMPLES.jsonl [SAMPLES.jsonl ...]
"""
import json
import os
import random
import re
import statistics
import sys


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def ngrams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def main():
    arm, out_path, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    assert arm in ("select", "random", "fluent"), arm
    by_prompt = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            by_prompt.setdefault(r["prompt"], []).append(r)
    rng = random.Random(7)
    kept, allr, n_filtered = [], [], 0
    for prompt, rs in by_prompt.items():
        allr += rs
        med = statistics.median(r["lp_self"] for r in rs)
        p3 = ngrams(words(prompt), 3)
        cands = [r for r in rs if r["loop4"] == 0 and r["valid_words"] >= 0.85 and r["words"] >= 5 and r["lp_self"] >= med
                 and not (ngrams(words(r["output"]), 3) & p3)]
        n_filtered += len(cands)
        if not cands:
            continue
        if arm == "select":
            late = float(os.environ.get("LATE", "3"))
            best = max(cands, key=lambda r: r["gain"] + late * r["gain_late"])
            if best["gain"] + late * best["gain_late"] <= 0:
                continue
        elif arm == "random":
            best = rng.choice(cands)
        else:
            best = max(cands, key=lambda r: r["lp_self"])
        kept.append(best)
    with open(out_path, "w", encoding="utf-8") as out:   # cut at the last line / sentence end (else word end): no torn words
        for r in kept:
            o = r["output"]
            cut = max(o.rfind("\n"), o.rfind(". "), o.rfind("! "), o.rfind("? "))
            o = o[:cut + 1] if cut >= len(o) // 2 else o[:max(1, o.rfind(" "))]
            out.write((r["prompt"] + o).rstrip("\n") + "\n")

    def mean(rs, k):
        return sum(r[k] for r in rs) / max(1, len(rs))
    print(f"{arm}: {len(by_prompt)} prompts, {len(allr)} samples, {n_filtered} pass the filters, {len(kept)} kept "
          f"({sum(len(r['prompt']) + len(r['output']) + 1 for r in kept)} bytes) -> {out_path}")
    for name, rs in (("all samples", allr), ("kept", kept)):
        print(f"  {name:12s} gain {mean(rs, 'gain'):.4f}  gain_late {mean(rs, 'gain_late'):.4f}  lp_self {mean(rs, 'lp_self'):.3f}  "
              f"valid {mean(rs, 'valid_words'):.3f}  loop4 {mean(rs, 'loop4'):.3f}  bytes {mean(rs, 'bytes'):.0f}")


if __name__ == "__main__":
    main()
