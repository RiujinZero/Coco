"""Verify and aggregate the completed seven-profile, three-seed screen."""
import argparse
import json
from pathlib import Path
from capacity_experiment import checksum, PROFILES


def summarize(root):
    rows = json.loads((root/'summary.json').read_text())
    assert len(rows) == 21 and {(r['profile'], r['seed']) for r in rows} == {(p, s) for p in PROFILES for s in range(3)}
    assert all(r['audited'] and r['native_returncode'] == 0 for r in rows)
    for seed in range(3):
        inputs = [r['inputs_sha256'] for r in rows if r['seed'] == seed]
        assert all(i == inputs[0] for i in inputs)
    aggregate, parity = {}, {}
    for profile in PROFILES:
        selected = [r for r in rows if r['profile'] == profile]
        totals = {stage: [sum(r['control']['metrics'][stage][k] for r in selected) for k in ('correct', 'total')]
                  for stage in ('stage4', 'recovery', 'longer')}
        totals['primitives'] = [sum(r['primitive_scores'][f'checkpoint/task{task}/disabled-1'][key]
                                   for r in selected for task in range(3)) for key in ('correct', 'n')]
        totals['native_wall_seconds'] = [round(r['native_wall_seconds'], 2) for r in sorted(selected, key=lambda r:r['seed'])]
        aggregate[profile] = totals
    for expanded, reference in {'agkern50k':'coco_continual_v1', 'agkern100k':'coco_continual_v1',
                                'both50k':'controller50k', 'both100k':'controller100k'}.items():
        for seed in range(3):
            a, b = root/expanded/str(seed), root/reference/str(seed)
            same = {suffix: checksum(a/f'learned_progress.{suffix}') == checksum(b/f'learned_progress.{suffix}')
                    for suffix in ('txt', 'final.weights', 'stage4.values')}
            assert all(same.values()), (expanded, seed, same)
            parity[f'{expanded}/{seed}'] = {'reference': f'{reference}/{seed}', 'identical': same}
    (root/'aggregate.json').write_text(json.dumps(aggregate, indent=2)+'\n')
    (root/'completed_agkern_parity.json').write_text(json.dumps(parity, indent=2)+'\n')
    print(json.dumps(aggregate, indent=2)); print('All 21 runs audited; all 12 expanded Agkern traces/weights/value tables match their controls.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/capacity_screen'))
    summarize(p.parse_args().root)
