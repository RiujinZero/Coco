"""SQuAD 2.0 (manifest S15, optional) -> passage-grounded QA with unanswerable questions.

"Use supplied passages and unanswerable questions before broad factual chat; do not confuse passage
support with world truth." Each paragraph's context becomes the story; each question an unanswerable
one (is_impossible=True) is answered "unknown" rather than skipped, so the model sees genuine
evidence-insufficiency alongside answerable ones -- not just a corpus of always-answerable trivia.

    python python/curriculum/build_squad.py SRC_DIR OUT_DIR
"""
import argparse, json, os


def build(path, out_txt, out_jsonl, is_test):
    d = json.load(open(path, encoding='utf-8'))
    n = 0
    ftxt = open(out_txt, 'w', encoding='utf-8', newline='\n') if out_txt else None
    fjs = open(out_jsonl, 'w', encoding='utf-8', newline='\n') if out_jsonl else None
    for art in d['data']:
        for para in art['paragraphs']:
            ctx = para['context'].replace('\r\n', '\n').strip()
            for qa in para['qas']:
                q = qa['question'].strip()
                if qa.get('is_impossible'):
                    ans = 'unknown'
                else:
                    answers = qa.get('answers') or []
                    if not answers:
                        continue
                    ans = answers[0]['text'].strip()
                    if not ans:
                        continue
                if is_test:
                    fjs.write(json.dumps({'task': 'squad2', 'qa': [[q, ans, '', [ctx]]]}) + '\n')
                else:
                    ftxt.write(ctx + '\nUSER: ' + q + '\n\nASSISTANT: ' + ans + '\n<|endofdialogue|>\n')
                n += 1
    if ftxt:
        ftxt.close()
    if fjs:
        fjs.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_dir')
    ap.add_argument('out_dir')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    n_train = build(os.path.join(a.src_dir, 'train-v2.0.json'),
                     os.path.join(a.out_dir, 'stage6_squad2_train.txt'), None, is_test=False)
    n_test = build(os.path.join(a.src_dir, 'dev-v2.0.json'),
                    None, os.path.join(a.out_dir, 'stage6_squad2_test.jsonl'), is_test=True)
    print(f'squad2: train {n_train}, test {n_test}')


if __name__ == '__main__':
    main()
