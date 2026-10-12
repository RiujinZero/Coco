"""Does a SynLogic JSONL carry reference answers? Looks for anything after the task statement: a <answer>…</answer>
with real content, an 'answer'/'solution'/'ground_truth' field, or text after the example output format.
    python3 python/synlogic_answers.py FILE
"""
import sys, json, re, collections
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
print('rows', len(rows), 'keys', sorted({k for r in rows[:200] for k in r}))
tmpl = '<answer> answer here</answer>'
real = 0; samples = []
for r in rows:
    t = r.get('text', '')
    for m in re.finditer(r'<answer>(.*?)</answer>', t, flags=re.S):
        if m.group(0) != tmpl and m.group(1).strip() not in ('answer here', ''):
            real += 1
            if len(samples) < 3: samples.append(m.group(1)[:200])
            break
print('rows with a real <answer> body:', real)
for s in samples: print('  sample:', s.replace('\n', ' | '))
tails = collections.Counter(r.get('text', '').rstrip()[-40:] for r in rows)
print('most common endings:'); [print(f'  {c:6d} {k!r}') for k, c in tails.most_common(8)]
