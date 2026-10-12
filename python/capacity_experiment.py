"""Matched capacity experiment through the canonical trainer, with audits."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path
from computational_control_probe import build
from check_relational_control import audit as audit_control
from check_primitive_discovery import audit as audit_primitives, rows
from check_composed_credit import allowed

PROFILES = ['coco_continual_v1', 'controller50k', 'controller100k', 'agkern50k', 'agkern100k', 'both50k', 'both100k']


def checksum(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, seed, profile, binary, extra_args=(), timeout_seconds=None):
    source = root/'data'/str(seed); directory = root/profile/str(seed)
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('input.txt', 'environment.txt', 'worlds.txt', 'oracle.json'):
        (directory/name).write_bytes((source/name).read_bytes())
    prefix = directory/'learned_progress'
    command = [str(binary), 'trainer=continual', f'profile={profile}', f'seed={seed}',
               f'file={directory/"input.txt"}', f'environment={directory/"environment.txt"}',
               f'worlds={directory/"worlds.txt"}', f'output={prefix}', f'checkpoint={directory/"state.replay"}', *extra_args]
    print(f'Starting capacity profile={profile} seed={seed}', flush=True)
    start = time.monotonic()
    timed_out = False
    with (directory/'learned_progress.txt').open('w') as out, (directory/'stderr.txt').open('w') as err:
        try:
            process = subprocess.run(command, stdout=out, stderr=err, timeout=timeout_seconds)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
    elapsed = time.monotonic()-start
    result = {'seed': seed, 'profile': profile, 'native_returncode': returncode,
              'native_wall_seconds': elapsed, 'command': command,
              'inputs_sha256': {name: checksum(directory/name) for name in ('input.txt', 'environment.txt', 'worlds.txt')},
              'binary_sha256': checksum(binary)}
    if timeout_seconds is not None:
        result.update(timeout_seconds=timeout_seconds, timed_out=timed_out)
    try:
        assert not timed_out, 'declared native wall-time budget exhausted; partial artifacts preserved'
        assert returncode == 0, 'native training failed; see stderr.txt'
        cfg = json.loads((directory/'learned_progress.config.json').read_text())
        result['resolved_config'] = cfg
        expected_control = cfg.get('controller_parameters', 1221)
        expected_skill = cfg.get('skill_allocated_route_weights', 15360)
        assert len((directory/'learned_progress.final.weights').read_text().split()) == expected_control
        assert len((directory/'learned_progress.final.primitive_weights').read_text().split()) == expected_skill
        result['control'] = audit_control(directory, 'learned_progress')
        shape = list(map(str, [cfg.get('skill_units', 512), cfg.get('skill_active', 16), cfg.get('skill_channels', 3)]))
        weights = directory/'learned_progress.final.primitive_weights'
        subprocess.run(['build-wsl/primitive_checkpoint_probe_capacity', str(seed), str(weights), str(directory/'checkpoint'), *shape], check=True, capture_output=True, text=True)
        result['primitive_scores'], _ = audit_primitives(directory/'checkpoint')
        choices = {}
        for row in rows(directory/'checkpoint.decisions.tsv'):
            state, action = int(row['state']), int(row['action'])
            assert state not in choices or choices[state] == action
            choices[state] = action
        wire = ''.join(f'{state} {action} {len(allowed(state))} '+' '.join(map(str, allowed(state)))+'\n' for state, action in sorted(choices.items()))
        subprocess.run(['build-wsl/verify_local_choices_capacity', str(seed), str(weights), *shape], input=wire, check=True, capture_output=True, text=True)
        result['frozen_primitive_states'] = len(choices)
        result['audited'] = True
    except Exception as error:
        result['audited'] = False; result['audit_error'] = str(error)
    result['artifacts_sha256'] = {str(p): checksum(p) for p in directory.iterdir() if p.is_file() and p.name != 'capacity_audit.json'}
    (directory/'capacity_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    metrics = result.get('control', {}).get('metrics', {})
    print(json.dumps({'profile': profile, 'seed': seed, 'seconds': round(elapsed, 2), 'audited': result['audited'],
                      'scores': {k: [metrics[k]['correct'], metrics[k]['total']] for k in ('stage4', 'recovery', 'longer') if k in metrics},
                      'error': result.get('audit_error')}), flush=True)
    return result


def experiment(root, seeds, profiles, episodes, workers, binary):
    root.mkdir(parents=True, exist_ok=False)
    for seed in seeds:
        directory = root/'data'/str(seed); build(seed, directory, episodes)
        meta = json.loads((directory/'oracle.json').read_text())
        meta.update(composed_budget=100000, capacity_curriculum='verified_direct_primitive_warm_start')
        (directory/'oracle.json').write_text(json.dumps(meta)+'\n')
    # Preserve exactly the code used by this run, not later working-tree hashes.
    files = [*Path('cpp/include').rglob('*.h'), *Path('cpp/apps').glob('*.h'), *Path('python').glob('*.py'),
             Path('cpp/apps/laminar_train.cpp'), Path('cpp/src/organism.cpp'),
             Path('cpp/apps/primitive_checkpoint_probe.cpp'), Path('cpp/apps/verify_local_choices.cpp')]
    sources = {}
    for path in files:
        target = root/'source'/path; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes()); sources[str(target)] = checksum(target)
    (root/'source_manifest.json').write_text(json.dumps(sources, indent=2)+'\n')
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run, root, seed, profile, binary) for seed in seeds for profile in profiles]
        for future in concurrent.futures.as_completed(jobs):
            results.append(future.result())
            (root/'summary.json').write_text(json.dumps(sorted(results, key=lambda r: (r['seed'], r['profile'])), indent=2)+'\n')
    assert all(r['audited'] for r in results), 'At least one run failed; failures preserved in summary.json'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/capacity_experiment'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--profiles', nargs='+', choices=PROFILES, default=PROFILES)
    p.add_argument('--episodes', type=int, default=400); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--binary', type=Path, default=Path('build-wsl/laminar_train'))
    a = p.parse_args(); experiment(a.root, a.seeds, a.profiles, a.episodes, a.workers, a.binary)
