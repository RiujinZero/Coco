"""Schema-Guided Dialogue (manifest S10) -> Stage 6a grounded task dialogue.

Static reconstruction: only each turn's UTTERANCE TEXT is learner-facing (SYSTEM turns become
ASSISTANT turns); the per-turn action/slot/state annotations are teacher-side dialogue-state-tracking
labels, kept in a sidecar and never written into the stream (manifest: "Keep gold dialogue state out
of the learner input"). This is the static-dialogue mode the manifest itself flags as insufficient
proof of interactive task completion on its own ("static dialogue annotations alone do not
demonstrate that Coco can complete a new task") -- an interactive mock-service backend for on-policy
scoring is a separate follow-up, not produced here. Official train/dev/test splits are preserved.

    python python/curriculum/build_sgd.py SGD_DIR OUT_DIR
"""
import argparse, glob, json, os


def render_dialogue(d):
    lines = []
    for turn in d['turns']:
        role = 'USER' if turn['speaker'] == 'USER' else 'ASSISTANT'
        text = turn['utterance'].replace('\n', ' ').strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sgd_dir')
    ap.add_argument('out_dir')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    oracle_dir = os.path.join(a.out_dir, '..', '_oracle')
    os.makedirs(oracle_dir, exist_ok=True)

    for split, is_train in (('train', True), ('dev', False)):
        files = sorted(glob.glob(os.path.join(a.sgd_dir, split, 'dialogues_*.json')))
        out_path = os.path.join(a.out_dir, f'stage6_sgd_{split}.txt')
        oracle_path = os.path.join(oracle_dir, f'sgd_{split}_oracle.jsonl')
        n = 0
        with open(out_path, 'w', encoding='utf-8', newline='\n') as f, \
             open(oracle_path, 'w', encoding='utf-8', newline='\n') as of:
            for fp in files:
                dialogues = json.load(open(fp, encoding='utf-8'))
                for d in dialogues:
                    lines = render_dialogue(d)
                    if len(lines) < 2:
                        continue
                    f.write('\n\n'.join(lines) + '\n<|endofdialogue|>\n')
                    of.write(json.dumps({'dialogue_id': d['dialogue_id'], 'services': d.get('services', [])}) + '\n')
                    n += 1
        print(f'sgd {split}: {n} dialogues -> {out_path}')


if __name__ == '__main__':
    main()
