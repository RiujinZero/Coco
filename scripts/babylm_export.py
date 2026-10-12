#!/usr/bin/env python3
"""Export BabyLM 2026 zero-shot items with the official loader, for the byte-level scorer (model_babylm_score).

usage: babylm_export.py EVAL_ROOT OUT_DIR [fast|full]
  EVAL_ROOT: babylm-eval/strict (contains evaluation_pipeline/ and evaluation_data/)
Writes OUT_DIR/<task>.jsonl, one candidate per line:
  {"item": i, "cand": j, "label": correct_index, "uid": ..., "task": ..., "context": ..., "completion": ...}
where context + completion == the official candidate sentence and only the completion is scored (as in the
pipeline's causal backend: summed completion log-probability, length-normalized for the tasks the pipeline lists).
The pipeline's `datasets` import (used only for image tasks) is stubbed so no GPU stack is needed.
"""
import json, pathlib, sys, types
from argparse import Namespace

root = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); split = sys.argv[3] if len(sys.argv) > 3 else 'full'
for stub in ('datasets', 'aiohttp'):   # used only for image tasks and downloads
    sys.modules.setdefault(stub, types.SimpleNamespace(load_dataset=None, Dataset=object))
sys.path.insert(0, str(root))
from evaluation_pipeline.sentence_zero_shot.read_files import read_files  # noqa: E402
import re  # noqa: E402
# the length-normalized task list, read from the pipeline's source (importing it would need torch)
src = (root / 'evaluation_pipeline' / 'sentence_zero_shot' / 'compute_results.py').read_text(encoding='utf-8')
m = re.search(r'LENGTH_NORMALIZED_TASKS\s*=\s*\{([^}]*)\}', src)
normalized = set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()
data = root / 'evaluation_data' / f'{split}_eval'
tasks = [('blimp', 'blimp_filtered'), ('blimp_supplement', 'supplement_filtered'), ('ewok', 'ewok_filtered'),
         ('entity_tracking', 'entity_tracking'), ('comps', 'comps')]
if split == 'fast':   # the fast split's folder names (EWoK fast ships as a protected zip)
    tasks = [('blimp', 'blimp_fast'), ('blimp_supplement', 'supplement_fast'), ('entity_tracking', 'entity_tracking_fast')]
out.mkdir(parents=True, exist_ok=True)
for name, folder in tasks:
    path = data / folder
    if not path.exists():
        print(f'skip {name}: {path} missing'); continue
    task = 'blimp' if name == 'blimp_supplement' else name
    items = read_files(Namespace(data_path=path, task=task, full_sentence_scores=False, images=None, images_path=None, image_split=None))
    n = 0
    with open(out / f'{name}.jsonl', 'w', encoding='utf-8') as f:
        for i, it in enumerate(items):
            for j, (s, c) in enumerate(zip(it['sentences'], it['completions'])):
                if not s.endswith(c):
                    raise SystemExit(f'{name} item {i}: completion is not a suffix of the sentence')
                f.write(json.dumps({'item': i, 'cand': j, 'label': it['label'], 'uid': it['UID'], 'task': name,
                                    'normalize': task in normalized, 'context': s[:len(s) - len(c)], 'completion': c},
                                   ensure_ascii=False) + '\n'); n += 1
    print(f'{name}: {len(items)} items, {n} candidates')
