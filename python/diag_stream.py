"""Compute-depth diagnostic: k-hop containment chains (k = 1..8) with distractors, plus bAbI yes/no (qa6) and
counting (qa7). Accuracy against k shows where the organism's compute per answer runs out.

    python python/diag_stream.py data/babi data/diag [--train 20000] [--test 50] [--rationale]

Each chain story states k containment facts in shuffled order plus distractor chains, and asks where the first
object ultimately is; the answer is the outermost container. With --rationale the training answers spell out the
steps ("the key is in the box. the box is in the drawer. answer: drawer") — the thinking-out-loud lever; test
answers are always scored on the word after "answer:" when present, else on the first word.
"""
import sys, os, json, random

OBJECTS = ['key', 'coin', 'ring', 'letter', 'marble', 'stamp', 'button', 'shell', 'pebble', 'badge', 'ticket', 'feather']
CONTAINERS = ['box', 'bag', 'drawer', 'chest', 'basket', 'jar', 'case', 'crate', 'pouch', 'tin', 'trunk', 'sack',
              'cupboard', 'locker', 'vault', 'barrel', 'bucket', 'carton', 'satchel', 'hamper', 'cabinet', 'safe', 'bin', 'tub']

SHIFT_OBJECTS = ['acorn', 'thimble', 'locket', 'domino', 'quill', 'medal', 'token', 'bead', 'fossil', 'compass', 'whistle', 'spool']
SHIFT_CONTAINERS = ['urn', 'vase', 'kettle', 'pitcher', 'canister', 'suitcase', 'backpack', 'purse', 'wallet', 'envelope', 'folder',
                    'coffer', 'casket', 'hutch', 'dresser', 'wardrobe', 'pantry', 'closet', 'attic', 'cellar', 'garage', 'shed', 'silo', 'barn']
def chain_story(rng, k, n_distract=2, shift=False):
    pool = (SHIFT_CONTAINERS if shift else CONTAINERS)[:]; rng.shuffle(pool)
    objs = SHIFT_OBJECTS if shift else OBJECTS
    obj = rng.choice(objs)
    path = [obj] + pool[:k]; used = k
    facts = [f'The {path[i]} is in the {path[i + 1]}.' for i in range(k)]
    for _ in range(n_distract):                           # distractor chains from other objects over unused containers
        o = rng.choice([x for x in objs if x != obj]); dl = rng.randint(1, 2)
        if used + dl > len(pool): break
        dp = [o] + pool[used:used + dl]; used += dl
        facts += [f'The {dp[i]} is in the {dp[i + 1]}.' for i in range(dl)]
    rng.shuffle(facts)
    steps = ' '.join(f'the {path[i]} is in the {path[i + 1]}.' for i in range(k))
    return facts, f'Where is the {obj}?', path[-1], steps

def episode(facts, q, a, rationale):
    return '\n'.join(facts) + '\nUSER: ' + q + '\n\nASSISTANT: ' + (rationale + ' answer: ' + a if rationale else a) + '\n<|endofdialogue|>\n'

def main():
    src, out = sys.argv[1], sys.argv[2]; args = sys.argv[3:]
    ntrain = int(args[args.index('--train') + 1]) if '--train' in args else 20000
    ntest = int(args[args.index('--test') + 1]) if '--test' in args else 50
    rat = '--rationale' in args; shift = '--shift' in args   # --shift: unseen objects and containers in the TEST chains only
    rng = random.Random(7)
    babi = {6: [], 7: []}
    for split in ('babi_train.jsonl', 'babi_valid.jsonl'):
        for line in open(os.path.join(src, split), encoding='utf-8'):
            r = json.loads(line)
            if r['task'] in babi: babi[r['task']].append(r)
    with open(out + '_train.txt', 'w', encoding='utf-8', newline='\n') as tr:
        for i in range(ntrain):
            k = 1 + i % 8
            facts, q, a, steps = chain_story(rng, k)
            tr.write(episode(facts, q, a, steps if rat else ''))
            if i % 4 == 0:                                   # interleave yes/no and counting
                for t in (6, 7):
                    r = babi[t][(i // 4) % len(babi[t])]
                    tr.write(r['passage'].rstrip('\n') + '\nUSER: ' + r['question'] + '\n\nASSISTANT: ' + r['answer'] + '\n<|endofdialogue|>\n')
    test = [json.loads(l) for l in open(os.path.join(src, 'babi_test.jsonl'), encoding='utf-8')]
    trng = random.Random(99)
    with open(out + '_test.jsonl', 'w', encoding='utf-8', newline='\n') as te:
        for k in range(1, 9):
            for _ in range(ntest):
                facts, q, a, _ = chain_story(trng, k, shift=shift)
                te.write(json.dumps({'task': 'hop%d' % k, 'qa': [[q, a, '', facts]]}) + '\n')
        for t in (6, 7):
            rs = [r for r in test if r['task'] == t]; trng.shuffle(rs)
            for r in rs[:ntest]:
                te.write(json.dumps({'task': 'qa%d' % t, 'qa': [[r['question'], r['answer'], '', r['passage'].rstrip('\n').split('\n')]]}) + '\n')
    print('train episodes', ntrain, os.path.getsize(out + '_train.txt'), 'bytes; rationale' if rat else 'bytes', '; test', 10 * ntest)

if __name__ == '__main__': main()
