"""Declared public-operation bootstrap, tapered to zero, through one trainer.

Reuses the capacity screen's exact train/test worlds. This first comparison
tests guided operation experience, not new primitive traces or goal-stack memory.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
from capacity_experiment import run, checksum


def experiment(root, source, seeds, profiles, binary, workers, full, fade):
    root.mkdir(parents=True, exist_ok=False)
    sources = {}
    files = [*Path('cpp/include').rglob('*.h'), *Path('cpp/apps').glob('*.h'), *Path('python').glob('*.py'),
             Path('cpp/apps/laminar_train.cpp'), Path('cpp/src/organism.cpp')]
    for path in files:
        target = root/'source'/path; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes()); sources[str(target)] = checksum(target)
    (root/'source_manifest.json').write_text(json.dumps(sources, indent=2)+'\n')
    for seed in seeds:
        target = root/'data'/str(seed); target.mkdir(parents=True)
        for path in (source/'data'/str(seed)).iterdir():
            if path.is_file(): (target/path.name).write_bytes(path.read_bytes())
        original = (target/'input.txt').read_text()
        (target/'input.txt').write_text(f'GUIDANCE public_control_v1 {full} {fade}\n'+original)
        meta = json.loads((target/'oracle.json').read_text())
        meta['guidance'] = {'teacher': 'public_control_v1', 'full_episodes': full, 'fade_episodes': fade,
                            'terminal_credit_only': True, 'evaluation_hints': False}
        (target/'oracle.json').write_text(json.dumps(meta)+'\n')
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run, root, seed, profile, binary) for seed in seeds for profile in profiles]
        for future in concurrent.futures.as_completed(jobs):
            results.append(future.result())
            (root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    assert all(r['audited'] for r in results), 'Warm-start audit failed; see retained results'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/warm_start_screen'))
    p.add_argument('--source', type=Path, default=Path('logs/capacity_screen'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--profiles', nargs='+', default=['coco_continual_v1', 'controller50k'])
    p.add_argument('--binary', type=Path, default=Path('build-wsl/laminar_train_guidance'))
    p.add_argument('--workers', type=int, default=2); p.add_argument('--full', type=int, default=32); p.add_argument('--fade', type=int, default=128)
    a = p.parse_args(); experiment(a.root, a.source, a.seeds, a.profiles, a.binary, a.workers, a.full, a.fade)
