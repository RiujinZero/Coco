"""SynLogic (parquet with extra_info.game_data_str) -> Coco episodes and a held-out test set.

Each puzzle becomes one episode: the puzzle text (as given, English or Chinese) as the USER turn and the ground-truth
answer from game_data as the ASSISTANT turn. Tasks without a stored answer (graded by a solver in SynLogic) are skipped.
The last `holdout` puzzles of every task are kept out of the stream and written as the test set.

    python python/synlogic_stream.py data/synlogic/synlogic_hard_train.parquet data/synlogic/sl [--holdout 20]
Writes <out>_train.txt and <out>_test.jsonl ({"task", "qa": [[question, answer, "", []]]}).
"""
import sys, json, collections, os, random
import pyarrow.parquet as pq

def gold(r):
    try: g = json.loads((r['extra_info'] or {}).get('game_data_str') or '{}')
    except Exception: return None
    a = g.get('answer') if isinstance(g, dict) else None
    if a in (None, '', [], {}, 'None'): return None
    return a if isinstance(a, str) else json.dumps(a, ensure_ascii=False, separators=(', ', ': '))

def main():
    src, out = sys.argv[1], sys.argv[2]; args = sys.argv[3:]
    hold = int(args[args.index('--holdout') + 1]) if '--holdout' in args else 20
    rows = pq.read_table(src).to_pylist()
    by = collections.defaultdict(list)
    for r in rows:
        a = gold(r)
        if a is not None: by[r['data_source']].append((r['prompt'][0]['content'].strip(), a.strip()))
    train, test = [], []
    for t, items in by.items():
        train += [(t, q, a) for q, a in items[:-hold]]; test += [(t, q, a) for q, a in items[-hold:]]
    random.Random(3).shuffle(train)                         # tasks interleaved through the stream
    with open(out + '_train.txt', 'w', encoding='utf-8', newline='\n') as f:
        for t, q, a in train: f.write('USER: ' + q + '\n\nASSISTANT: ' + a + '\n<|endofdialogue|>\n')
    with open(out + '_test.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for t, q, a in test: f.write(json.dumps({'task': t, 'qa': [[q, a, '', []]]}, ensure_ascii=False) + '\n')
    print(f'tasks {len(by)}; train episodes {len(train)} ({os.path.getsize(out + "_train.txt") / 1e6:.1f} MB); test {len(test)}')
    for t in sorted(by): print(f'  {t:32s} {len(by[t]):5d}')

if __name__ == '__main__': main()
