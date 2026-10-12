"""Exercise canonical controller-only ablations through actual trainer/auditor."""
import argparse
import concurrent.futures
import json
from pathlib import Path
from capacity_experiment import run
from computational_control_probe import build


def check(binary, root):
    root.mkdir(parents=True, exist_ok=False)
    data = root/'data'/'1'
    build(1, data, 8)
    original = (data/'input.txt').read_text().splitlines()
    train = [r for r in original if r.startswith('TRAIN ')][:8]
    boundary = original.index('CHECK stage4')
    tests = [r for r in original[boundary+1:] if r.startswith('TEST ')][:8]
    teaching = [r for r in original if r.split()[0] in ('P', 'C', 'L', 'M', 'N', 'T')]
    (data/'input.txt').write_text('\n'.join(teaching+train+['CHECK stage4']+tests+['ENDCHECK'])+'\n')
    ids = {r.split()[1] for r in train+tests}
    for filename in ('environment.txt', 'worlds.txt'):
        rows = (data/filename).read_text().splitlines()
        rows = [r for r in rows if r.split()[0] not in ('Q', 'W', 'X') or r.split()[1] in ids]
        (data/filename).write_text('\n'.join(rows)+'\n')
    meta = json.loads((data/'oracle.json').read_text())
    meta['cases'] = [c for c in meta['cases'] if c['id'] in ids]
    meta['composed_budget'] = 100000
    (data/'oracle.json').write_text(json.dumps(meta)+'\n')
    profiles = [base+suffix for base in ('controller50k', 'controller50k_trace')
                for suffix in ('', '_frozen', '_shuffled', '_terminal')]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run, root, 1, profile, binary, (), 300) for profile in profiles]
        for job in concurrent.futures.as_completed(jobs):
            results.append(job.result())
            (root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    assert all(r['audited'] for r in results), 'canonical ablation audit failed; preserve results'
    for r in results:
        cfg = r['resolved_config']
        assert cfg['controller_parameters'] == 50013
        for suffix, flag in (('_frozen', 'freeze_control'), ('_shuffled', 'shuffled_credit'), ('_terminal', 'terminal_only_credit')):
            assert cfg.get(flag, False) == r['profile'].endswith(suffix)
        if r['profile'].endswith('_frozen'):
            assert r['control']['changed_parameters'] == 0 and r['control']['updates'] == 0
        else:
            assert r['control']['changed_parameters'] > 0 and r['control']['updates'] == 1024
    result = {'canonical_arms': len(results), 'same_controller_parameters': 50013,
              'per_arm_training_episodes': 8, 'per_arm_frozen_evaluations': 8,
              'all_independently_audited': True,
              'scope': 'Control ablation wiring and evidence accounting, not task-success or long-credit sufficiency.'}
    (root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    check(a.binary, a.root)
