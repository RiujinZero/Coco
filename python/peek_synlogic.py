"""Look inside SynLogic rows: full text of a few, whether an answer is present, task kinds by leading phrase.
    python3 python/peek_synlogic.py FILE
"""
import sys, json, collections, re
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
for i in (0, 1, 7):
    t = rows[i]['text']; print(f'===== row {i}, {len(t)} chars =====')
    print(t[:1800]); print('...'); print(t[-900:])
has_ans = sum('<answer>' in r['text'].split('i.e., <think>')[-1] for r in rows)
print('rows with an <answer> beyond the instruction:', has_ans, 'of', len(rows))
lens = sorted(len(r['text']) for r in rows); print('length p50 / p90 / max:', lens[len(lens) // 2], lens[int(len(lens) * .9)], lens[-1])
cjk = sum(bool(re.search(r'[一-鿿]', r['text'])) for r in rows); print('rows with Chinese text:', cjk)
kinds = collections.Counter()
for r in rows:
    body = r['text'].split('</answer>.', 1)[-1].strip()
    kinds[re.sub(r'\d+', 'N', body[:60])] += 1
print('most common openings:'); [print(f'  {c:6d}  {k!r}') for k, c in kinds.most_common(40)]
