#!/usr/bin/env python3
"""Generation metrics of model_frontier runs (item H): per file and mode, the means over prompts.

gain / g_late: the log-prob per byte the prompt adds to the output (whole output / second half): the prompt-following
measure; lp_self: the model's own log-prob per byte of its output (fluency by its own standard); valid: in-vocabulary
word rate; loop4: outputs with a repeated word 4-gram; d2: distinct word bigrams; reach: 1 + the last output byte the
prompt still changes.

usage: hloop_report.py RUN.jsonl [RUN.jsonl ...]
       hloop_report.py --paired BASE.jsonl ARM.jsonl [ARM.jsonl ...]   (per prompt and mode: ARM - BASE, mean +- standard error)
"""
import json
import math
import sys


def paired(files):
    base = {(r["prompt"], r["mode"]): r for r in map(json.loads, open(files[0], encoding="utf-8"))}
    print(f"paired vs {files[0]}: mean difference per prompt (+- standard error)")
    for f in files[1:]:
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        for mode in dict.fromkeys(r["mode"] for r in rows):
            out = []
            for k in ("gain", "gain_late", "lp_self", "valid_words", "loop4"):
                d = [float(r[k]) - float(base[(r["prompt"], mode)][k]) for r in rows if r["mode"] == mode and (r["prompt"], mode) in base]
                m = sum(d) / len(d)
                se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
                out.append(f"{k} {m:+.4f}+-{se:.4f}")
            name = f if len(f) <= 40 else "..." + f[-37:]
            print(f"  {name:40s} {mode:7s} n={len(d):3d}  " + "  ".join(out))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--paired":
        paired(sys.argv[2:])
        return
    print(f"{'file':44s} {'mode':7s} {'n':>4s} {'gain':>7s} {'g_late':>7s} {'lp_self':>8s} {'valid':>6s} {'loop4':>6s} {'d2':>6s} {'reach':>6s} {'bytes':>6s}")
    for f in sys.argv[1:]:
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        for mode in dict.fromkeys(r["mode"] for r in rows):
            rs = [r for r in rows if r["mode"] == mode]

            def m(k):
                return sum(float(r[k]) for r in rs) / len(rs)
            name = f if len(f) <= 44 else "..." + f[-41:]
            print(f"{name:44s} {mode:7s} {len(rs):4d} {m('gain'):7.4f} {m('gain_late'):7.4f} {m('lp_self'):8.3f} {m('valid_words'):6.3f} "
                  f"{m('loop4'):6.3f} {m('distinct2'):6.3f} {m('reach'):6.1f} {m('bytes'):6.0f}")


if __name__ == "__main__":
    main()
