"""bAbI (Muennighoff/babi jsonl mirror: {"passage","question","answer","task"}) -> Coco chat stream.

Each training row becomes one episode (dialogue): the passage sentences, then the question as a USER turn and the
answer as an ASSISTANT turn, so the sequence memory's episode/commitment machinery applies unchanged.

    python python/babi_stream.py data/babi data/babi_t1-3-15-16 1 2 3 15 16 [--per-task 200]

Writes <out>_train.txt (stream) and <out>_test.jsonl in laminar_babi's format:
    {"task": "qa2", "qa": [[question, answer, "", [sentence, ...]]]}
"""
import sys, os, json, random

def main():
    src, out = sys.argv[1], sys.argv[2]; args = sys.argv[3:]
    per_task = 200
    if '--per-task' in args: i = args.index('--per-task'); per_task = int(args[i + 1]); args = args[:i] + args[i + 2:]
    tasks = set(int(a) for a in args) if args else set(range(1, 21))
    ntrain = 0
    with open(out + '_train.txt', 'w', encoding='utf-8', newline='\n') as tr:
        for split in ('babi_train.jsonl', 'babi_valid.jsonl'):
            for line in open(os.path.join(src, split), encoding='utf-8'):
                r = json.loads(line)
                if r['task'] not in tasks: continue
                tr.write(r['passage'].rstrip('\n') + '\nUSER: ' + r['question'] + '\n\nASSISTANT: ' + r['answer'] + '\n<|endofdialogue|>\n'); ntrain += 1
    rows = [json.loads(l) for l in open(os.path.join(src, 'babi_test.jsonl'), encoding='utf-8')]
    rng = random.Random(1); picked = []
    for t in sorted(tasks):
        rs = [r for r in rows if r['task'] == t]; rng.shuffle(rs); picked += rs[:per_task]
    with open(out + '_test.jsonl', 'w', encoding='utf-8', newline='\n') as te:
        for r in picked:
            te.write(json.dumps({'task': 'qa%d' % r['task'], 'qa': [[r['question'], r['answer'], '', r['passage'].rstrip('\n').split('\n')]]}) + '\n')
    print('train episodes', ntrain, os.path.getsize(out + '_train.txt'), 'bytes; test questions', len(picked))

if __name__ == '__main__': main()
