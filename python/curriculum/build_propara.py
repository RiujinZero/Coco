"""ProPara (manifest S9) -> sentence-prefix state-tracking episodes, a natural-language transfer check
for Stage 5. "Use sentence-prefix state tracking; withhold future sentences and evaluation state
annotations." Each grids.v1.*.json line has sentence_texts, participants, and a states grid where
states[i][t] is participant i's location BEFORE sentence t (t=0..n_sentences; the last column is the
state after the whole paragraph). '-' = does not exist yet / no longer exists, '?' = exists, location
unknown.

For each prefix length t (1..n_sentences), for each participant, the learner sees only sentences[0:t]
and is asked where that participant is now; the target is states[i][t] (never a later sentence).

    python python/curriculum/build_propara.py SRC_DIR OUT_DIR
"""
import argparse, json, os


def render_answer(loc):
    if loc == '-':
        return 'nowhere yet'
    if loc == '?':
        return 'unknown'
    return loc


def episodes_for_paragraph(rec):
    sents = rec['sentence_texts']
    parts = rec['participants']
    states = rec['states']
    out = []
    for t in range(1, len(sents) + 1):
        prefix = sents[:t]
        for i, p in enumerate(parts):
            ans = render_answer(states[i][t])
            out.append((prefix, f"Where is the {p} now?", ans, rec['para_id'], t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_dir')
    ap.add_argument('out_dir')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    train_path = os.path.join(a.out_dir, 'stage5_propara_train.txt')
    test_path = os.path.join(a.out_dir, 'stage5_propara_test.jsonl')
    n_train = n_test = 0
    with open(train_path, 'w', encoding='utf-8', newline='\n') as tf, \
         open(test_path, 'w', encoding='utf-8', newline='\n') as ef:
        for split, is_test in (('train.json', False), ('dev.json', True), ('test.json', True)):
            path = os.path.join(a.src_dir, split)
            if not os.path.exists(path):
                continue
            for line in open(path, encoding='utf-8'):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for prefix, q, ans, pid, t in episodes_for_paragraph(rec):
                    story = '\n'.join(prefix)
                    if is_test:
                        ef.write(json.dumps({'task': 'propara_state', 'qa': [[q, ans, '', prefix]]}) + '\n')
                        n_test += 1
                    else:
                        tf.write(story + '\nUSER: ' + q + '\n\nASSISTANT: ' + ans + '\n<|endofdialogue|>\n')
                        n_train += 1
    print(f'propara: train {n_train}, test {n_test}')


if __name__ == '__main__':
    main()
