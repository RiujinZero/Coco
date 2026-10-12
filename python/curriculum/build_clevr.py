"""CLEVR (manifest S3) -> Coco streams for Stage 2 (individual operations) and Stage 4 (composition).

Classification is by the question's ENTIRE program (teacher-only), not just its final operation.
Rebuild 2026-09-22 (see docs/CURRICULUM_REBUILD_ISSUES.md #5, first bullet): checking only the final
function let a same_shape/equal_<attr>/greater_than/less_than test sitting in the MIDDLE of a program
through into Stage 2 whenever the program happened to end in count/exist/query_<attr> -- e.g. "How
many other things are there of the same shape as the tiny cyan matte object?" is same_shape -> count:
relational (it compares one object's shape against the rest of the scene), yet its final op (count)
put it in Stage 2 under the old rule. Any relational/comparison function ANYWHERE in the program, or
two independent 'scene' roots, now routes the whole question to Stage 4, regardless of what it ends
with. A single-chain program using only filter_*/relate/unique and ending in count/exist/query_<attr>
-- true one-hop retrieval, filtering, spatial relation lookup, counting, existence -- is Stage 2.

The learner stream never sees the program or a query-filtered scene: every episode renders the WHOLE
scene (all objects, the full left-right/front-back order), exactly as clevr_common.verbalize_scene
does for every question of that image.

    python python/curriculum/build_clevr.py SRC_DIR OUT_DIR [--train_cap N] [--test_per_family N]

SRC_DIR is the extracted CLEVR_v1.0 directory (scenes/, questions/). Writes:
  OUT_DIR/stage2_clevr_train.txt   OUT_DIR/stage2_clevr_test.jsonl
  OUT_DIR/stage4_clevr_train.txt   OUT_DIR/stage4_clevr_test.jsonl
  OUT_DIR/clevr_oracle.jsonl       (teacher-only: program, skill_family, group_id, split; NOT fed to the trainer)
"""
import sys, os, json, argparse, collections
sys.path.insert(0, os.path.dirname(__file__))
from clevr_common import load_scenes, verbalize_scene, program_shape, program_depth, n_chains

INDIVIDUAL_FINAL = {'count', 'exist', 'query_size', 'query_color', 'query_material', 'query_shape'}
# any of these ANYWHERE in the program -- not only as the final op -- makes the whole question
# relational/compositional: each one only makes sense by holding two independently-derived object
# sets or attribute values against each other.
RELATIONAL_ANYWHERE = {'equal_size', 'equal_color', 'equal_material', 'equal_shape', 'equal_integer',
                        'greater_than', 'less_than', 'same_size', 'same_color', 'same_material', 'same_shape',
                        'intersect', 'union'}


def classify(program):
    funcs = {p['function'] for p in program}
    if n_chains(program) >= 2 or (funcs & RELATIONAL_ANYWHERE):
        return 'stage4'
    if program[-1]['function'] in INDIVIDUAL_FINAL:
        return 'stage2'
    return None  # rare/other function types: skip rather than mis-stage


def episode(sents, question, answer):
    return '\n'.join(sents) + '\nUSER: ' + question + '\n\nASSISTANT: ' + answer + '\n<|endofdialogue|>\n'


def run_split(src, out_dir, split, train_cap, test_per_family, oracle_fp, is_train):
    scenes = load_scenes(os.path.join(src, 'scenes', f'CLEVR_{split}_scenes.json'))
    qpath = os.path.join(src, 'questions', f'CLEVR_{split}_questions.json')
    print(f'[{split}] loading questions (this file is large) ...', flush=True)
    data = json.load(open(qpath, encoding='utf-8'))
    print(f'[{split}] {len(data["questions"])} questions, {len(scenes)} scenes', flush=True)

    train_f = {}
    test_rows = collections.defaultdict(list)
    test_seen_scene = collections.defaultdict(set)
    counts = collections.Counter()

    if is_train:
        train_f['stage2'] = open(os.path.join(out_dir, 'stage2_clevr_train.txt'), 'w', encoding='utf-8', newline='\n')
        train_f['stage4'] = open(os.path.join(out_dir, 'stage4_clevr_train.txt'), 'w', encoding='utf-8', newline='\n')

    written = collections.Counter()
    for q in data['questions']:
        stage = classify(q['program'])
        if stage is None:
            continue
        scene = scenes[q['image_index']]
        sents = verbalize_scene(scene)
        family = q['program'][-1]['function']
        gid = f"clevr_{split}_{q['image_index']}"

        if is_train:
            if train_cap and written[stage] >= train_cap:
                continue
            train_f[stage].write(episode(sents, q['question'], q['answer']))
            written[stage] += 1
        else:
            key = (stage, family)
            if len(test_rows[key]) >= test_per_family:
                continue
            # one question per scene per family, to keep the held-out set independent-world-grouped
            if q['image_index'] in test_seen_scene[key]:
                continue
            test_seen_scene[key].add(q['image_index'])
            test_rows[key].append([q['question'], q['answer'], '', sents])

        oracle_fp.write(json.dumps({
            'id': f"{gid}_q{q['question_index']}", 'split': split, 'stage': stage,
            'skill_family': family, 'phrase_family': '>'.join(program_shape(q['program'])),
            'program_depth': program_depth(q['program']), 'group_id': gid,
            'question_family_index': q.get('question_family_index'),
        }) + '\n')
        counts[(stage, family)] += 1

    if is_train:
        for f in train_f.values():
            f.close()
        print(f'[{split}] train written: {dict(written)}')
    else:
        for stage in ('stage2', 'stage4'):
            path = os.path.join(out_dir, f'{stage}_clevr_test.jsonl')
            mode = 'a' if os.path.exists(path) and split != 'val' else 'w'
            with open(path, mode, encoding='utf-8', newline='\n') as f:
                for (st, family), rows in test_rows.items():
                    if st != stage:
                        continue
                    for q, a, sup, story in rows:
                        f.write(json.dumps({'task': f'clevr_{family}', 'qa': [[q, a, sup, story]]}) + '\n')
        print(f'[{split}] test groups: { {k: len(v) for k, v in test_rows.items()} }')
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out_dir')
    ap.add_argument('--train_cap', type=int, default=0, help='max train episodes per stage (0 = all)')
    ap.add_argument('--test_per_family', type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, 'clevr_oracle.jsonl'), 'w', encoding='utf-8', newline='\n') as oracle_fp:
        run_split(a.src, a.out_dir, 'train', a.train_cap, a.test_per_family, oracle_fp, is_train=True)
        run_split(a.src, a.out_dir, 'val', a.train_cap, a.test_per_family, oracle_fp, is_train=False)


if __name__ == '__main__':
    main()
