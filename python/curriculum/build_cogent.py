"""CLEVR-CoGenT (manifest S3, "CoGenT specifically tests changed attribute combinations") -> the Stage
4 "new attributes" holdout (manifest section 9.2: "New attributes | Attribute combinations, such as
CoGenT A/B | Individual attributes"). Condition A (cubes gray/blue/brown/yellow, cylinders
red/green/purple/cyan; spheres any color) is the training distribution; condition B swaps the
shape-color pairing. trainA -> training; valA -> a matched in-distribution control; valB -> the actual
attribute-combination transfer test (a shape-color pairing never seen in training).

    python python/curriculum/build_cogent.py COGENT_SRC OUT_DIR
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
from clevr_common import load_scenes, verbalize_scene
from build_clevr import classify, episode


def build_train(src, out_dir):
    scenes = load_scenes(os.path.join(src, 'scenes', 'CLEVR_trainA_scenes.json'))
    qpath = os.path.join(src, 'questions', 'CLEVR_trainA_questions.json')
    data = json.load(open(qpath, encoding='utf-8'))
    out = {'stage2': open(os.path.join(out_dir, 'stage4_cogent_trainA_stage2.txt'), 'w', encoding='utf-8', newline='\n'),
           'stage4': open(os.path.join(out_dir, 'stage4_cogent_trainA_stage4.txt'), 'w', encoding='utf-8', newline='\n')}
    n = {'stage2': 0, 'stage4': 0}
    for q in data['questions']:
        stage = classify(q['program'])
        if stage is None:
            continue
        sents = verbalize_scene(scenes[q['image_index']])
        out[stage].write(episode(sents, q['question'], q['answer']))
        n[stage] += 1
    for f in out.values():
        f.close()
    print('cogent trainA:', n)


def build_eval(src, split, out_dir, per_family):
    scenes = load_scenes(os.path.join(src, 'scenes', f'CLEVR_{split}_scenes.json'))
    qpath = os.path.join(src, 'questions', f'CLEVR_{split}_questions.json')
    data = json.load(open(qpath, encoding='utf-8'))
    rows = {}
    for q in data['questions']:
        stage = classify(q['program'])
        if stage is None:
            continue
        family = q['program'][-1]['function']
        key = (stage, family)
        rows.setdefault(key, [])
        if len(rows[key]) >= per_family:
            continue
        sents = verbalize_scene(scenes[q['image_index']])
        rows[key].append([q['question'], q['answer'], '', sents])
    out_path = os.path.join(out_dir, f'stage4_cogent_{split}_test.jsonl')
    n = 0
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        for (stage, family), items in rows.items():
            for qq, aa, sup, story in items:
                f.write(json.dumps({'task': f'cogent_{split}_{family}', 'qa': [[qq, aa, sup, story]]}) + '\n')
                n += 1
    print(f'cogent {split}: {n} test rows -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out_dir')
    ap.add_argument('--per_family', type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    build_train(a.src, a.out_dir)
    build_eval(a.src, 'valA', a.out_dir, a.per_family)   # matched in-distribution control
    build_eval(a.src, 'valB', a.out_dir, a.per_family)   # the actual new-attribute-combination transfer test


if __name__ == '__main__':
    main()
