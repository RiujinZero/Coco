"""Survey SynLogic parquet: tasks, how many carry answers, answer lengths and shapes, a sample per task.
    python python/synlogic_survey.py data/synlogic/synlogic_hard_train.parquet [out.txt]
"""
import sys, collections, io
import pyarrow.parquet as pq
out = io.open(sys.argv[2] if len(sys.argv) > 2 else 'data/synlogic/survey.txt', 'w', encoding='utf-8')
rows = pq.read_table(sys.argv[1]).to_pylist()
by = collections.defaultdict(list)
for r in rows: by[r['data_source']].append(r)
out.write(f'rows {len(rows)}, tasks {len(by)}\n\n')
out.write(f'{"task":28s} {"n":>5s} {"ans":>5s} {"ans p50":>8s} {"ans max":>8s} {"prompt p50":>10s}  answer shape\n')
for t, rs in sorted(by.items(), key=lambda x: -len(x[1])):
    ans = [(r['reward_model'] or {}).get('answer') or '' for r in rs]
    has = sum(bool(a.strip()) for a in ans)
    la = sorted(len(a) for a in ans); lp = sorted(len(r['prompt'][0]['content']) for r in rs)
    shape = collections.Counter(('grid' if a.strip().startswith('[[') or a.strip().startswith('[\n') else 'list' if a.strip().startswith('[') else
             'num' if a.strip().replace('-', '').replace('.', '').isdigit() else 'word' if len(a.split()) == 1 else 'text') for a in ans).most_common(2)
    out.write(f'{t:28s} {len(rs):5d} {has:5d} {la[len(la)//2]:8d} {la[-1]:8d} {lp[len(lp)//2]:10d}  {shape}\n')
out.write('\n')
for t, rs in sorted(by.items(), key=lambda x: -len(x[1])):
    r = rs[0]; a = (r['reward_model'] or {}).get('answer') or ''
    out.write(f'===== {t}\n  answer: {a[:300]!r}\n  prompt tail: {r["prompt"][0]["content"][-300:]!r}\n')
out.close(); print('written', sys.argv[2] if len(sys.argv) > 2 else 'data/synlogic/survey.txt')
