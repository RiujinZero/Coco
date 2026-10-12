"""Stage 3 training data: paraphrase families over the SAME CLEVR worlds already used in Stage 2/4
(manifest: "Reuse the same training worlds and teacher-side operations to make small verified phrase
families ... Assign these a shared teacher-side operation, but do not expose that identifier to Coco
at runtime.")

Rebuild 2026-09-22 (see docs/CURRICULUM_REBUILD_ISSUES.md #3 and #4):

  * Issue #3 (independent holdout): the previous version drew ITS OWN test rows from CLEVR_train_
    scenes -- the exact pool every other CLEVR-based stage trains on -- so 450/450 Stage 3 test
    worlds turned out to already be in Stage 2's training data (449/450 in Stage 4's). Test worlds
    now come exclusively from CLEVR's official VAL split, the same held-out pool build_clevr.py
    already uses for Stage 2/4's own tests, so a Stage 3 test world is *by construction* absent from
    every CLEVR-based stage's training data, not just this file's own training rows.
  * Issue #4 (paired paraphrases): the previous version advanced to a NEW scene for every template,
    so the training file held 8,550 distinct worlds and zero worlds with more than one paraphrase on
    it -- there was never a same-world/same-query pair to check agreement on. Each world's templates
    are now all rendered against THAT SAME world (one true answer, computed once), and the test
    export groups a world's template variants under one id so correctness and cross-template
    agreement can be scored separately (agreement alone can reward being consistently wrong).

One template (the third) is held out of training entirely and used only at test time, on top of the
world holdout: an "unseen construction" test, not just an unseen world for a familiar construction.

    python python/curriculum/build_paraphrase.py CLEVR_SRC OUT_DIR [--per_template N]
"""
import sys, os, json, argparse, random
sys.path.insert(0, os.path.dirname(__file__))
from clevr_common import load_scenes, verbalize_scene

COUNT_TMPL = ["How many things are there?", "What is the number of objects in the scene?", "Count the objects."]
TRAIN_TEMPLATE_IDX = [0, 1]         # trained on
HELDOUT_TEMPLATE_IDX = [2]          # never trained on: an unseen-construction test


def whole_scene_count(scene):
    return str(len(scene['objects']))


def episode(sents, q, a):
    return '\n'.join(sents) + '\nUSER: ' + q + '\n\nASSISTANT: ' + a + '\n<|endofdialogue|>\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clevr_src'); ap.add_argument('out_dir')
    ap.add_argument('--per_template', type=int, default=3000, help='training worlds per trained template')
    ap.add_argument('--n_test_worlds', type=int, default=450)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    oracle_dir = os.path.join(a.out_dir, '..', '_oracle')
    os.makedirs(oracle_dir, exist_ok=True)

    train_scenes = list(load_scenes(os.path.join(a.clevr_src, 'scenes', 'CLEVR_train_scenes.json')).values())
    val_scenes = list(load_scenes(os.path.join(a.clevr_src, 'scenes', 'CLEVR_val_scenes.json')).values())
    rng = random.Random(7)
    rng.shuffle(train_scenes)
    rng.shuffle(val_scenes)

    train_path = os.path.join(a.out_dir, 'stage3_paraphrase_train.txt')
    test_path = os.path.join(a.out_dir, 'stage3_paraphrase_test.jsonl')
    oracle_path = os.path.join(oracle_dir, 'paraphrase_oracle.jsonl')

    n_train = 0
    with open(train_path, 'w', encoding='utf-8', newline='\n') as tf, \
         open(oracle_path, 'w', encoding='utf-8', newline='\n') as of:
        pool = train_scenes[:a.per_template]   # ONE shared pool of worlds; every trained template renders on all of them
        for sc in pool:
            sents = verbalize_scene(sc)
            ans = whole_scene_count(sc)
            for tmpl_i in TRAIN_TEMPLATE_IDX:
                tf.write(episode(sents, COUNT_TMPL[tmpl_i], ans))
                n_train += 1
            of.write(json.dumps({'family': 'whole_scene_count', 'group_id': f"clevr_train_{sc['image_index']}",
                                  'template_indices': TRAIN_TEMPLATE_IDX, 'split': 'train'}) + '\n')

    n_test_groups = 0
    n_test_rows = 0
    with open(test_path, 'w', encoding='utf-8', newline='\n') as ef, \
         open(oracle_path, 'a', encoding='utf-8', newline='\n') as of:
        for sc in val_scenes[:a.n_test_worlds]:
            sents = verbalize_scene(sc)
            ans = whole_scene_count(sc)
            world_group = f"clevr_val_{sc['image_index']}"
            variants = []
            for tmpl_i in TRAIN_TEMPLATE_IDX + HELDOUT_TEMPLATE_IDX:
                unseen = tmpl_i in HELDOUT_TEMPLATE_IDX
                task = 'paraphrase_count_unseen_construction' if unseen else 'paraphrase_count'
                ef.write(json.dumps({'task': task, 'qa': [[COUNT_TMPL[tmpl_i], ans, '', sents]],
                                      'world_group': world_group, 'template_index': tmpl_i}) + '\n')
                variants.append(tmpl_i)
                n_test_rows += 1
            of.write(json.dumps({'family': 'whole_scene_count', 'group_id': world_group,
                                  'template_indices': variants, 'split': 'test'}) + '\n')
            n_test_groups += 1
    print(f'paraphrase: train {n_train} episodes on {len(pool)} worlds x {len(TRAIN_TEMPLATE_IDX)} templates; '
          f'test {n_test_rows} rows across {n_test_groups} worlds x {len(TRAIN_TEMPLATE_IDX) + len(HELDOUT_TEMPLATE_IDX)} templates '
          f'(template {HELDOUT_TEMPLATE_IDX} never trained on); test worlds drawn from CLEVR val (disjoint from all training)')


if __name__ == '__main__':
    main()
