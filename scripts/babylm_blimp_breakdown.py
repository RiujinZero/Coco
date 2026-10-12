#!/usr/bin/env python3
"""BLiMP accuracy per phenomenon (linguistics_term) and per paradigm for a scored file.

usage: babylm_blimp_breakdown.py SCORED_BLIMP.jsonl BLIMP_DATA_DIR
Ties count as the pipeline's random choice in expectation; each phenomenon is the mean of its paradigms.
"""
import collections, json, pathlib, sys

scored, data = sys.argv[1], pathlib.Path(sys.argv[2])
term = {}
for f in data.glob('*.jsonl'):
    for line in open(f, encoding='utf-8'):
        d = json.loads(line); term[d.get('UID', f.stem)] = d.get('linguistics_term', '?'); break
items = collections.defaultdict(list)
for line in open(scored, encoding='utf-8'):
    r = json.loads(line); items[r['item']].append(r)
par = collections.defaultdict(lambda: [0.0, 0])
for cands in items.values():
    cands.sort(key=lambda r: r['cand']); s = [r['logprob'] for r in cands]; best = max(s)
    tied = [i for i, x in enumerate(s) if x == best]; acc = 1.0 / len(tied) if cands[0]['label'] in tied else 0.0
    par[cands[0]['uid']][0] += acc; par[cands[0]['uid']][1] += 1
by_term = collections.defaultdict(list)
for uid, (c, n) in par.items(): by_term[term.get(uid, '?')].append((100 * c / n, uid))
print(f'{"phenomenon":36} {"acc":>6}  paradigms (best / worst)')
for t, v in sorted(by_term.items(), key=lambda kv: -sum(a for a, _ in kv[1]) / len(kv[1])):
    v.sort(); mean = sum(a for a, _ in v) / len(v)
    print(f'{t:36} {mean:6.1f}  {len(v):2}  best {v[-1][1]} {v[-1][0]:.0f}  worst {v[0][1]} {v[0][0]:.0f}')
print(f'{"overall (macro over paradigms)":36} {sum(100 * c / n for c, n in par.values()) / len(par):6.1f}')
