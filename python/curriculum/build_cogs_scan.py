"""COGS (S4) and SCAN (S5) -> diagnostic-only Coco streams (manifest Stage 3: "Use COGS as a separate
role/structure diagnostic and keep SCAN's official held-out composition splits separate from this
custom data. Semantic parsing success alone is not proof of grounded world knowledge.")

These are formal supervised targets (a logical form / an action sequence), not Coco's own world-
grounded question answering. They are written to their own stream, never mixed into the main
curriculum training stream, and kept split exactly as officially released (COGS: train/gen/test;
SCAN: simple and add-primitive-jump splits) so official generalization scores stay comparable.

    python python/curriculum/build_cogs_scan.py COGS_DIR SCAN_DIR OUT_DIR
"""
import sys, os, json, argparse


def cogs_episode(sent, lf):
    return f"USER: {sent}\n\nASSISTANT: {lf}\n<|endofdialogue|>\n"


def build_cogs(src, out_dir):
    for split, fname in (('train', 'train.tsv'), ('test', 'test.tsv')):
        path = os.path.join(out_dir, f'cogs_{split}.txt')
        n = 0
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            for line in open(os.path.join(src, fname), encoding='utf-8'):
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                f.write(cogs_episode(parts[0], parts[1])); n += 1
        print(f'cogs {split}: {n} episodes')
    # gen.tsv carries its own generalization-type column (3rd field): keep as jsonl for grouped scoring
    rows = []
    for line in open(os.path.join(src, 'gen.tsv'), encoding='utf-8'):
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 2:
            continue
        rows.append({'sentence': parts[0], 'logical_form': parts[1], 'gen_type': parts[2] if len(parts) > 2 else None})
    with open(os.path.join(out_dir, 'cogs_gen.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'cogs gen (generalization, held out): {len(rows)} rows')


def scan_episode(cmd, action):
    return f"USER: {cmd}\n\nASSISTANT: {action}\n<|endofdialogue|>\n"


def parse_scan_line(line):
    line = line.strip()
    if not line.startswith('IN:'):
        return None
    out_i = line.index(' OUT:')
    cmd = line[3:out_i].strip()
    action = line[out_i + 5:].strip()
    return cmd, action


def build_scan(src, out_dir):
    files = {
        'simple_train': 'tasks_train_simple.txt', 'simple_test': 'tasks_test_simple.txt',
        'addprim_jump_train': 'tasks_train_addprim_jump.txt', 'addprim_jump_test': 'tasks_test_addprim_jump.txt',
    }
    for name, fname in files.items():
        fp = os.path.join(src, fname)
        if not os.path.exists(fp):
            print(f'scan {name}: missing {fname}, skipped'); continue
        out_ext = 'txt' if 'train' in name else 'jsonl'
        outp = os.path.join(out_dir, f'scan_{name}.{out_ext}')
        n = 0
        with open(outp, 'w', encoding='utf-8', newline='\n') as f:
            for line in open(fp, encoding='utf-8'):
                parsed = parse_scan_line(line)
                if not parsed:
                    continue
                cmd, action = parsed
                if out_ext == 'txt':
                    f.write(scan_episode(cmd, action))
                else:
                    f.write(json.dumps({'command': cmd, 'action': action}) + '\n')
                n += 1
        print(f'scan {name}: {n} episodes -> {outp}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cogs_dir'); ap.add_argument('scan_dir'); ap.add_argument('out_dir')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    build_cogs(a.cogs_dir, a.out_dir)
    build_scan(a.scan_dir, a.out_dir)


if __name__ == '__main__':
    main()
