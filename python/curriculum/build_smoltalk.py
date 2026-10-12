"""SmolTalk (manifest S12) -> Stage 6b broad chat, split back into its labeled subsets.

The "all" config carries a `source` column naming which subset each row came from, so downloading it
once (2.1 GB) and splitting by that column gives every subset without re-downloading each one
separately (they are, by the dataset's own construction, a partition of "all"). Per the manifest,
smol-rewrite and smol-summarize are written first/separately so a training run can start with just
those two ("Begin with smol-rewrite and smol-summarize rather than the entire mixture") before
drawing on the rest. "Its newly generated components and imported components do not all share one
license" -- source is kept in the output filename so any subset can be excluded later without
re-processing.

    python python/curriculum/build_smoltalk.py OUT_DIR
"""
import argparse, os
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

PRIORITY_FIRST = {'smol-rewrite', 'smol-summarize'}


def render(messages):
    lines = []
    for m in messages:
        role = m['role']
        text = (m['content'] or '').replace('\r\n', '\n').strip()
        if not text:
            continue
        if role == 'system':
            lines.append(text)  # context, not a turn: no USER:/ASSISTANT: tag, matches story-prefix convention
        elif role == 'user':
            lines.append(f"USER: {text}")
        else:
            lines.append(f"ASSISTANT: {text}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    api = HfApi()
    info = api.dataset_info('HuggingFaceTB/smoltalk', files_metadata=True)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith('data/all/') and s.rfilename.endswith('.parquet'))
    train_files = [f for f in files if 'train' in f]
    test_files = [f for f in files if 'test' in f]

    handles = {}

    def get_handle(source, is_test):
        key = (source, is_test)
        if key not in handles:
            tag = 'priority' if source in PRIORITY_FIRST else 'rest'
            split = 'test' if is_test else 'train'
            ext = 'jsonl' if is_test else 'txt'
            path = os.path.join(a.out_dir, f'stage6_smoltalk_{tag}_{source}_{split}.{ext}')
            handles[key] = open(path, 'w', encoding='utf-8', newline='\n')
        return handles[key]

    counts = {}
    for fname, is_test in [(f, False) for f in train_files] + [(f, True) for f in test_files]:
        print(f'downloading {fname} ...', flush=True)
        local = hf_hub_download(repo_id='HuggingFaceTB/smoltalk', repo_type='dataset', filename=fname)
        t = pq.read_table(local)
        for row in t.to_pylist():
            source = row['source']
            lines = render(row['messages'])
            if len(lines) < 2:
                continue
            fh = get_handle(source, is_test)
            if is_test:
                import json
                fh.write(json.dumps({'source': source, 'lines': lines}) + '\n')
            else:
                fh.write('\n\n'.join(lines) + '\n<|endofdialogue|>\n')
            counts[source] = counts.get(source, 0) + 1
        print(f'  cumulative counts: {counts}', flush=True)

    for fh in handles.values():
        fh.close()
    print('smoltalk done:', counts)


if __name__ == '__main__':
    main()
