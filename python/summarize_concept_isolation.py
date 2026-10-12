"""Aggregate the growth/retention screen and verify its unchanged warm prefix."""
import argparse
import json
from pathlib import Path
from capacity_experiment import checksum
from check_relational_session import parse_snapshot
from check_relational_control import Reader


def summarize(root, baseline):
    results = json.loads((root/'summary.json').read_text())
    assert len(results) == 3 and {r['seed'] for r in results} == {0, 1, 2}
    assert all(r['audited'] for r in results)
    summary = {'seeds': [0, 1, 2], 'original_profile': 'controller50k', 'added_training_episodes_per_seed': 96,
               'added_teacher_hints': 0, 'scores': {}, 'runs': {}}
    for stage in ('stage4', 'recovery', 'longer', 'uncertain_retention', 'acquired_retention', 'new_concept_transfer'):
        summary['scores'][stage] = [sum(r['control']['metrics'][stage][k] for r in results) for k in ('correct', 'total')]
    summary['scores']['primitives'] = [sum(r['primitive_scores'][f'checkpoint/task{t}/disabled-1'][k]
                                         for r in results for t in range(3)) for k in ('correct', 'n')]
    for result in results:
        seed = result['seed']; directory = root/'controller50k'/str(seed); reference = baseline/'controller50k'/str(seed)
        parity = {}
        for stage in ('stage1', 'stage2', 'stage3', 'stage4', 'removals', 'recovery', 'longer'):
            for suffix in ('weights', 'primitive_weights', 'probe_primitive_weights', 'values'):
                name = f'learned_progress.{stage}.{suffix}'
                if not (reference/name).exists(): continue
                parity[f'{stage}.{suffix}'] = checksum(directory/name) == checksum(reference/name)
        assert parity and all(parity.values()), (seed, parity)
        assert checksum(directory/'learned_progress.final.weights') != checksum(directory/'learned_progress.stage4.weights')
        snapshots = {}; reader = Reader(directory/'learned_progress.txt')
        post_bootstrap = False; hints = 0
        while reader.peek():
            row = reader.pop()
            if row[:3] == ['T', 'C', 'new_sparse_positive']: post_bootstrap = True
            if row[0] == 'H' and post_bootstrap: hints += 1
            if row[0] == 'R':
                rows = [row]
                while rows[-1][0] != 'END': rows.append(reader.pop())
                snap, end = parse_snapshot(rows, 0); assert end == len(rows)
                snapshots[snap['stage']] = snap
        assert hints == 0
        final = snapshots['new_concept_transfer']; new_procedures = sum(
            any(op.startswith('newmotor') for op, _ in code) for _, code in final['procedures'])
        assert new_procedures > 0
        summary['runs'][str(seed)] = {'warm_prefix_parity': parity, 'new_actuator_procedures': new_procedures,
                                     'concepts_before': len(snapshots['stage4']['concepts']), 'concepts_after': len(final['concepts']),
                                     'controller_changed_after_continuation': True,
                                     'new_concept_alternatives': [snapshots[s]['concepts'][f'concept{seed}new']['count']
                                                                for s in ('uncertain_retention', 'new_concept_transfer')]}
    (root/'aggregate.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['scores'], indent=2))
    print('All three runs audited; original warm-prefix weights/values preserved; continued neural updates and new procedures verified.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/concept_isolation_screen_v2'))
    p.add_argument('--baseline', type=Path, default=Path('logs/warm_start_screen'))
    a = p.parse_args(); summarize(a.root, a.baseline)
