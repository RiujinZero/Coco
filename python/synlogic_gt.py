"""Where does SynLogic keep ground truth? Inspect reward_model.solution and extra_info.game_data_str per task.
    python python/synlogic_gt.py data/synlogic/synlogic_hard_train.parquet [out.txt]
"""
import sys, json, collections, io
import pyarrow.parquet as pq
out = io.open(sys.argv[2] if len(sys.argv) > 2 else 'data/synlogic/gt.txt', 'w', encoding='utf-8')
rows = pq.read_table(sys.argv[1]).to_pylist()
by = collections.defaultdict(list)
for r in rows: by[r['data_source']].append(r)
for t, rs in sorted(by.items(), key=lambda x: x[0]):
    r = rs[0]; sol = (r['reward_model'] or {}).get('solution') or ''
    gd = (r['extra_info'] or {}).get('game_data_str') or ''
    try: g = json.loads(gd); gk = list(g.keys()) if isinstance(g, dict) else type(g).__name__
    except Exception: g = None; gk = 'unparsed'
    ans = g.get('answer') if isinstance(g, dict) else None
    nans = 0
    for x in rs:
        try: gg = json.loads((x['extra_info'] or {}).get('game_data_str') or '{}'); nans += bool(gg.get('answer') not in (None, '', [], {}))
        except Exception: pass
    out.write(f'===== {t}  (n={len(rs)}, with game_data answer: {nans})\n  solution: {sol[:200]!r}\n  game_data keys: {gk}\n  answer: {str(ans)[:300]!r}\n')
out.close(); print('written')
