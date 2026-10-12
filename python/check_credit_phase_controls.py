"""Actual file-run wiring check for matched post-400 controller ablations.

Repeated short fixtures test state/credit boundaries, not long-task learning.
"""
import argparse
import concurrent.futures
import copy
import json
from pathlib import Path

from capacity_experiment import checksum, run
from computational_control_probe import build


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    original = root/'original'
    build(1, original, 8)
    lines = (original/'input.txt').read_text().splitlines()
    teachers = [line for line in lines if line.split()[0] in ('P', 'C', 'L', 'M', 'N', 'T')]
    source_train = [line for line in lines if line.startswith('TRAIN ')][:8]
    boundary = lines.index('CHECK stage4')
    source_test = [line for line in lines[boundary+1:] if line.startswith('TEST ')][:8]
    meta = json.loads((original/'oracle.json').read_text())
    cases = {c['id']: c for c in meta['cases']}
    old_worlds = (original/'worlds.txt').read_text().splitlines()
    old_environment = (original/'environment.txt').read_text().splitlines()
    rows_for = {}
    for label, rows in (('worlds', old_worlds), ('environment', old_environment)):
        rows_for[label] = {}
        for row in rows:
            parts = row.split(' ', 2)
            if parts[0] in ('Q', 'W', 'X'):
                rows_for[label].setdefault(parts[1], []).append(parts)
    commands, worlds, environment, emitted = teachers[:], [], [r for r in old_environment if not r.startswith('Q ')], []

    def emit(row, index):
        operation, old_id, tail = row.split(' ', 2)
        new_id = f'phase{index}'
        commands.append(f'{operation} {new_id} {tail}')
        case = copy.deepcopy(cases[old_id])
        case.update(id=new_id, training=operation == 'TRAIN')
        emitted.append(case)
        for label, destination in (('worlds', worlds), ('environment', environment)):
            for kind, _, rest in rows_for[label][old_id]:
                destination.append(f'{kind} {new_id} {rest}')

    for i in range(400):
        emit(source_train[i % len(source_train)], i)
    commands.append('CHECK stage4')
    for i, row in enumerate(source_test):
        emit(row, 400+i)
    commands.append('ENDCHECK')
    for i, row in enumerate(source_train):
        emit(row, 408+i)
    commands.append('CHECK after_phase')
    for i, row in enumerate(source_test):
        emit(row, 416+i)
    commands.append('ENDCHECK')
    meta['cases'] = emitted
    meta['composed_budget'] = 100000
    meta['phase_control_contract'] = {'warm_episodes': 400, 'post_boundary_episodes': 8,
                                      'repeated_fixtures': True, 'behavioral_gate': False}
    data = root/'data'/'1'
    data.mkdir(parents=True)
    for name, rows in (('input', commands), ('worlds', worlds), ('environment', environment)):
        (data/f'{name}.txt').write_text('\n'.join(rows)+'\n')
    (data/'oracle.json').write_text(json.dumps(meta)+'\n')


def check(args):
    prepare(args.root)
    profiles = [base+suffix for base in args.bases for suffix in ('', '_frozen_after400', '_shuffled_after400')]
    (args.root/'plan.json').write_text(json.dumps({'profiles': profiles, 'seed': 1, 'seconds': args.seconds,
                                                 'binary_sha256': checksum(args.binary) if args.binary else None,
                                                 'driver_sha256': checksum(Path(__file__))}, indent=2)+'\n')
    (args.root/'driver.py').write_bytes(Path(__file__).read_bytes())
    if args.prepare_only:
        return True
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, args.root, 1, profile, args.binary, (), args.seconds) for profile in profiles]
        for job in concurrent.futures.as_completed(jobs):
            results.append(job.result())
            (args.root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    result = {'all_audited': all(r['audited'] for r in results), 'warm_state_matches': {},
              'frozen_boundary_matches': {}, 'scope': 'Phase wiring and exact bootstrap matching, not long-horizon capability.'}
    for base in args.bases:
        for suffix in ('_frozen_after400', '_shuffled_after400'):
            profile = base+suffix
            current = next(r for r in results if r['profile'] == profile)
            if not current['audited']:
                continue
            reference = args.root/base/'1'
            arm = args.root/profile/'1'
            same = {name: checksum(reference/f'learned_progress.{name}') == checksum(arm/f'learned_progress.{name}')
                    for name in ('stage4.weights', 'stage4.values', 'stage4.primitive_weights')}
            result['warm_state_matches'][profile] = same
            if suffix == '_frozen_after400':
                result['frozen_boundary_matches'][profile] = (
                    checksum(arm/'learned_progress.stage4.weights') == checksum(arm/'learned_progress.final.weights')
                    and checksum(arm/'learned_progress.stage4.values') == checksum(arm/'learned_progress.after_phase.values')
                    and current['control']['checks']['matched_stage4_freeze'] == 1)
    result['passed'] = (result['all_audited'] and len(result['warm_state_matches']) == 2*len(args.bases)
                        and all(all(v.values()) for v in result['warm_state_matches'].values())
                        and all(result['frozen_boundary_matches'].values()))
    (args.root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--bases', nargs='+', choices=['controller50k', 'controller50k_trace'], default=['controller50k', 'controller50k_trace'])
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if min(args.seconds, args.workers) <= 0 or len(set(args.bases)) != len(args.bases):
        parser.error('positive budgets and distinct bases required')
    if not args.prepare_only and args.binary is None:
        parser.error('--binary required to run')
    raise SystemExit(0 if check(args) else 1)
