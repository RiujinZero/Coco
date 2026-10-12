#!/usr/bin/env python3
"""Accuracy report for model_babylm_score output, following the BabyLM 2026 pipeline's rules.

usage: babylm_report.py SCORED.jsonl [SCORED2.jsonl ...]
- prediction: the candidate with the highest completion log-probability (divided by its byte count for the tasks the
  pipeline length-normalizes); a tie counts as the pipeline's random choice in expectation (1/k if the label is among
  the k tied candidates)
- accuracy: macro average over UIDs; entity tracking averages UIDs within each split (regular, ambiref,
  move_contents) and then the splits, as run.py does
"""
import collections, json, sys

for path in sys.argv[1:]:
    items = collections.defaultdict(list)
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        items[(r['task'], r['item'])].append(r)
    by_task = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for (task, _), cands in items.items():
        cands.sort(key=lambda r: r['cand'])
        score = [r['logprob'] / max(r['bytes'], 1) if r.get('normalize') else r['logprob'] for r in cands]
        best = max(score); tied = [i for i, s in enumerate(score) if s == best]
        label = cands[0]['label']
        acc = (1.0 / len(tied)) if label in tied else 0.0
        by_task[task][cands[0]['uid']][0] += acc; by_task[task][cands[0]['uid']][1] += 1
    for task, uids in by_task.items():
        acc = {u: 100.0 * c / n for u, (c, n) in uids.items()}
        if task == 'entity_tracking':
            splits = [[a for u, a in acc.items() if u.startswith(s)] for s in ('regular', 'ambiref', 'move_contents')]
            avg = sum(sum(s) / len(s) for s in splits if s) / max(1, sum(1 for s in splits if s))
        else:
            avg = sum(acc.values()) / len(acc)
        n = sum(v[1] for v in uids.values())
        print(f'{path.split("/")[-1]:40} {task:18} {avg:6.2f}  ({len(uids)} sub-tasks, {n} items)')
