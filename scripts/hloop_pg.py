#!/usr/bin/env python3
"""Policy-gradient data for the prompt-following objective (item H): every sample of a prompt, weighted by its ADVANTAGE.

For each prompt, up to K of its samples; reward R = topic + topic_late (the model-free topic judge, scripts/topic_judge.py),
minus 0.1 for a sample with a repeated word 4-gram; advantage A = (R - mean_prompt(R)) / (std_prompt(R) + 0.01), clipped to
+-2. Output: the text (one line per sample: prompt + continuation cut at its last line / sentence / word end) and its weights
file (one record per sample: "PLEN TLEN A" in bytes - the prompt's PLEN bytes learn with weight 1, the rest of the sample's
TLEN bytes (its continuation, any number of lines, and the newline) with weight A), for
ms_babylm's MS_EXTRA / MS_EXTRA_W. The weights of a prompt's samples sum to ~0: probability moves from its off-topic samples
to its on-topic ones, with no net pull toward the model's own text (expert iteration's positive-only data sharpened it).
Run with the Windows Python (numpy).

usage: hloop_pg.py JUDGE_PREFIX K OUT_TEXT OUT_WEIGHTS SAMPLES.jsonl [SAMPLES.jsonl ...]
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topic_judge  # noqa: E402


def trim(o):
    cut = max(o.rfind("\n"), o.rfind(". "), o.rfind("! "), o.rfind("? "))
    return o[:cut + 1] if cut >= len(o) // 2 else o[:max(1, o.rfind(" "))]


def main():
    judge = topic_judge.Judge(sys.argv[1])
    K, out_text, out_w = int(sys.argv[2]), sys.argv[3], sys.argv[4]
    files = [f for f in sys.argv[5:] if not f.endswith(".topic.jsonl")]
    by = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if len(by.setdefault(r["prompt"], [])) < K:
                by[r["prompt"]].append(r)
    n, pos, neg, rs_all = 0, [], [], []
    with open(out_text, "w", encoding="utf-8", newline="\n") as ot, open(out_w, "w", encoding="utf-8", newline="\n") as ow:
        for prompt, rs in by.items():
            if len(rs) < 2:
                continue
            for r in rs:
                t, tl = judge.score(r["prompt"], r["output"])
                r["R"] = t + tl - (0.1 if r["loop4"] else 0.0)
            mu = statistics.mean(r["R"] for r in rs); sd = statistics.pstdev(r["R"] for r in rs)
            for r in rs:
                a = max(-2.0, min(2.0, (r["R"] - mu) / (sd + 0.01)))
                o = trim(r["output"]).rstrip("\n")
                if not o.strip():
                    continue
                rec = prompt + o + "\n"
                ot.write(rec)
                ow.write(f"{len(prompt.encode('utf-8'))} {len(rec.encode('utf-8'))} {a:.4f}\n")
                n += 1; rs_all.append(r); (pos if a > 0 else neg).append(a)
    print(f"pg: {len(by)} prompts, {n} weighted samples ({len(pos)} positive, mean {statistics.mean(pos) if pos else 0:.3f}; "
          f"{len(neg)} negative, mean {statistics.mean(neg) if neg else 0:.3f}); reward mean {statistics.mean(r['R'] for r in rs_all):.4f} -> {out_text}")


if __name__ == "__main__":
    main()
