"""OpenAssistant OASST1 (manifest S11) -> Stage 6b broad chat.

"Select English, reviewed nondeleted messages and quality branches; split all paths from one
message_tree_id together." Filters to lang=en, deleted=False, review_result=True, then walks each
message tree from its root, enumerating root-to-leaf paths (branching capped per tree so one huge
tree cannot dominate the corpus, higher-ranked replies preferred). Every path from the same tree is
written from the SAME source parquet row group in one pass, so a train/validation split by tree id
(the official HF split already partitions by conversation, not by message) is not further broken
apart here.

    python python/curriculum/build_oasst1.py SRC_DIR OUT_DIR [--max_paths_per_tree 4]
"""
import argparse, json, os, collections
import pyarrow.parquet as pq


def load_rows(path):
    t = pq.read_table(path, columns=['message_id', 'parent_id', 'text', 'role', 'lang', 'deleted',
                                      'review_result', 'rank', 'message_tree_id'])
    return t.to_pylist()


def build_trees(rows):
    by_id = {}
    children = collections.defaultdict(list)
    for r in rows:
        if r['lang'] != 'en' or r['deleted'] or not r['review_result']:
            continue
        by_id[r['message_id']] = r
        children[r['parent_id']].append(r)
    for k in children:
        children[k].sort(key=lambda r: (r['rank'] if r['rank'] is not None else 99))
    roots = [r for r in by_id.values() if r['parent_id'] not in by_id]
    return by_id, children, roots


def enumerate_paths(children, root, max_paths):
    # DFS from root, capping total emitted paths; higher-rank (already-sorted) children first
    paths = []

    def dfs(node, path):
        if len(paths) >= max_paths:
            return
        kids = children.get(node['message_id'], [])
        if not kids:
            paths.append(path)
            return
        for kid in kids:
            if len(paths) >= max_paths:
                break
            dfs(kid, path + [kid])

    dfs(root, [root])
    return paths


def render_path(path):
    lines = []
    for m in path:
        role = 'USER' if m['role'] == 'prompter' else 'ASSISTANT'
        text = m['text'].replace('\r\n', '\n').strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--max_paths_per_tree', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    for split, fname, is_train in (
        ('train', 'train-00000-of-00001-b42a775f407cee45.parquet', True),
        ('validation', 'validation-00000-of-00001-134b8fd0c89408b6.parquet', False),
    ):
        rows = load_rows(os.path.join(a.src_dir, fname))
        by_id, children, roots = build_trees(rows)
        out_path = os.path.join(a.out_dir, f'stage6_oasst1_{split}.txt')
        n_paths = n_trees = 0
        with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
            for root in roots:
                paths = enumerate_paths(children, root, a.max_paths_per_tree)
                if not paths:
                    continue
                n_trees += 1
                for path in paths:
                    lines = render_path(path)
                    if len(lines) < 2 or lines[0].split(':')[0] != 'USER':
                        continue
                    f.write('\n\n'.join(lines) + '\n<|endofdialogue|>\n')
                    n_paths += 1
        print(f'oasst1 {split}: {n_trees} trees, {n_paths} conversation paths -> {out_path}')


if __name__ == '__main__':
    main()
