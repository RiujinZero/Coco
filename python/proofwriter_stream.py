"""ProofWriter (tasksource/proofwriter, local JSONL) -> training stream + held-out test in the laminar_babi format.

Each row is "Theory: <facts and rules>\n\nQuestion: <statement>\n\nAnswer: True|False|Unknown". The theory becomes the
story (one sentence per line), the statement becomes a USER question, the label the ASSISTANT answer (lower-cased).
Rows are split by theory text so no test theory appears in training. Answers: true / false / unknown (closed-world
entailment with negation as failure is not assumed — the label set is the dataset's).

    python3 python/proofwriter_stream.py SRC.jsonl data/proofwriter --train 20000 --test 600
"""
import sys, json, random, re, hashlib

def parse(t):
    m = re.match(r'Theory:\s*(.*?)\n\nQuestion:\s*(.*?)\n\nAnswer:\s*(\w+)', t, flags=re.S)
    if not m: return None
    story = [s.strip() for s in re.split(r'(?<=\.)\s+', m.group(1).strip()) if s.strip()]
    return story, m.group(2).strip(), m.group(3).strip().lower()

def main():
    src, out = sys.argv[1], sys.argv[2]; a = sys.argv[3:]
    ntr = int(a[a.index('--train') + 1]) if '--train' in a else 20000
    nte = int(a[a.index('--test') + 1]) if '--test' in a else 600
    tr, te = [], []
    for line in open(src, encoding='utf-8'):
        r = parse(json.loads(line)['text'])
        if not r: continue
        h = int(hashlib.md5(' '.join(r[0]).encode()).hexdigest(), 16) % 10    # split by theory
        (te if h == 0 else tr).append(r)
    rng = random.Random(5); rng.shuffle(tr); rng.shuffle(te)
    with open(out + '_train.txt', 'w', encoding='utf-8', newline='\n') as f:
        for story, q, ans in tr[:ntr]:
            f.write('\n'.join(story) + '\nUSER: ' + q + ' True, false or unknown?\n\nASSISTANT: ' + ans + '\n<|endofdialogue|>\n')
    by = {}
    for r in te: by.setdefault(r[2], []).append(r)
    per = nte // max(1, len(by)); rows = [r for k in sorted(by) for r in by[k][:per]]   # balanced labels
    with open(out + '_test.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for story, q, ans in rows:
            f.write(json.dumps({'task': 'pw_' + ans, 'qa': [[q + ' True, false or unknown?', ans, '', story]]}) + '\n')
    print(f'train {min(ntr, len(tr))} rows, test {len(rows)} ({per} per label), held-out theories {len(te)}')

if __name__ == '__main__': main()
