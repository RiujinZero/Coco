"""Print the schema, field samples and category counts of a JSONL dataset.
    python3 python/inspect_jsonl.py FILE [n_samples]
"""
import sys, json, collections
path = sys.argv[1]; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
rows = []
with open(path, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line: rows.append(json.loads(line))
print('rows', len(rows))
print('keys', list(rows[0].keys()))
def show(v, n=400):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s[:n].replace('\n', ' | ')
for i in range(min(ns, len(rows))):
    print('--- row', i)
    for k, v in rows[i].items(): print(f'  {k} [{type(v).__name__}]: {show(v)}')
for key in rows[0]:
    vals = [r.get(key) for r in rows]
    if all(isinstance(v, (str, int, bool)) or v is None for v in vals):
        c = collections.Counter(vals)
        if len(c) <= 60: print(f'counts of {key}:', c.most_common(60))
