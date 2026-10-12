"""Score SynLogic replies dumped by laminar_babi (dump=FILE) for learning-curve signs.

Per task:
  exact   normalized exact match with the puzzle's own answer
  own     partial credit: position-wise token agreement between the reply and its own answer (0..1)
  other   the same agreement between the reply and ANOTHER held-out puzzle's answer of the same task
  spec    own - other: puzzle-specific information in the reply (0 when it only copies a format or a memorized answer)
  fmt     own-answer vs other-answer agreement: what a correct-format guess that solves nothing would score (ceiling for
          format-only credit)
    python python/synlogic_score.py dump1.jsonl [dump2.jsonl ...]     (one column block per dump)
"""
import sys, json, re, collections

def toks(s): return re.findall(r'\w+|[^\w\s]', s.lower())
def agree(a, b):                                        # position-wise token agreement, normalized by the longer
    ta, tb = toks(a), toks(b)
    if not ta and not tb: return 1.0
    n = max(len(ta), len(tb)); return sum(x == y for x, y in zip(ta, tb)) / n
def norm(s): return re.sub(r'\s+', '', s.lower())

def score(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8', errors='replace') if l.strip()]   # replies are raw bytes and may split a multi-byte character
    by = collections.defaultdict(list)
    for r in rows: by[r['task']].append(r)
    res = {}
    for t, rs in by.items():
        n = len(rs); ex = own = oth = fmt = 0.0
        for i, r in enumerate(rs):
            o = rs[(i + 1) % n]['gold']
            rep = r['reply'].strip()
            ex += norm(rep) == norm(r['gold'])
            own += agree(rep, r['gold']); oth += agree(rep, o); fmt += agree(o, r['gold'])
        res[t] = dict(n=n, exact=ex / n, own=own / n, other=oth / n, spec=(own - oth) / n, fmt=fmt / n)
    return res

def main():
    paths = sys.argv[1:]; allres = [score(p) for p in paths]
    tasks = sorted(set().union(*[r.keys() for r in allres]))
    hdr = f'{"task":32s}' + ''.join(f' | {"exact":>5s} {"own":>5s} {"spec":>6s}' for _ in paths) + f' | {"fmt":>5s}'
    print('columns per dump:', ', '.join(paths)); print(hdr)
    tot = [collections.Counter() for _ in paths]
    for t in tasks:
        line = f'{t:32s}'
        for k, r in enumerate(allres):
            x = r.get(t)
            if x: line += f' | {x["exact"]:5.2f} {x["own"]:5.2f} {x["spec"]:+6.3f}'; tot[k].update({'exact': x['exact'], 'own': x['own'], 'spec': x['spec'], 'n': 1})
            else: line += ' |     -     -      -'
        fm = next((r[t]['fmt'] for r in allres if t in r), 0.0)
        print(line + f' | {fm:5.2f}')
    line = f'{"MEAN over tasks":32s}'
    for c in tot: line += f' | {c["exact"] / c["n"]:5.3f} {c["own"] / c["n"]:5.3f} {c["spec"] / c["n"]:+6.3f}'
    print(line)
    pos = [sum(1 for t in tasks if t in r and r[t]['spec'] > 0.01) for r in allres]
    print('tasks with spec > +0.01:', pos, 'of', len(tasks))

if __name__ == '__main__': main()
