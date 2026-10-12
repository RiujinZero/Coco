"""CLUTRR (tasksource/clutrr parquet: sentence1 = story with [Name] brackets, sentence2 = "('A', 'B')", labels = relation)
-> Coco chat stream. Each row is one episode: the story, "USER: How is A related to B?" and "ASSISTANT: <relation>".
The answer is a COMPOSED relation (daughter of a sister -> niece), never a word of the story: this is the test
after bAbI — chains of relations whose answer is a function of the relation words along the path, and the
test stories are longer than any in training (systematic generalization).

    python python/clutrr_stream.py data/clutrr data/clutrr_chat [--per-split 400]
"""
import sys, os, json, re
import pyarrow.parquet as pq

NAMES = ["aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law", "granddaughter", "grandfather", "grandmother",
         "grandson", "mother", "mother-in-law", "nephew", "niece", "sister", "son", "son-in-law", "uncle"]

def clean(s): return re.sub(r'\[(\w+)\]', r'\1', s).strip()

def main():
    src, out = sys.argv[1], sys.argv[2]; per = 400
    if '--per-split' in sys.argv: per = int(sys.argv[sys.argv.index('--per-split') + 1])
    ntrain = 0
    with open(out + '_train.txt', 'w', encoding='utf-8', newline='\n') as tr:
        for split in ('train', 'validation'):
            for r in pq.read_table(os.path.join(src, f'{split}-00000-of-00001.parquet')).to_pylist():
                a, b = eval(r['sentence2']); rel = NAMES[int(r['labels'])]
                tr.write(clean(r['sentence1']) + '\nUSER: How is ' + a + ' related to ' + b + '?\n\nASSISTANT: ' + rel + '\n<|endofdialogue|>\n'); ntrain += 1
    rows = pq.read_table(os.path.join(src, 'test-00000-of-00001.parquet')).to_pylist()[:per]
    with open(out + '_test.jsonl', 'w', encoding='utf-8', newline='\n') as te:
        for r in rows:
            a, b = eval(r['sentence2']); rel = NAMES[int(r['labels'])]
            sents = [x.strip() + '.' for x in clean(r['sentence1']).split('.') if x.strip()]
            te.write(json.dumps({'task': 'clutrr%d' % min(len(sents), 9), 'qa': [['How is ' + a + ' related to ' + b + '?', rel, '', sents]]}) + '\n')
    print('train episodes', ntrain, os.path.getsize(out + '_train.txt'), 'bytes; test questions', len(rows))

if __name__ == '__main__': main()
