"""Matched delayed-credit candidate screen on the preserved interrupted stream.

The host schedules frozen interruptions. This is a task-retention screen for the
credit mechanism, not evidence of learned goal scheduling or long-task acquisition.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path

from capacity_experiment import checksum, run


PROFILES = ('controller50k', 'controller50k_return', 'controller50k_trace')
STAGES = ('stage4', 'recovery', 'longer', 'interrupted', 'interrupted/children')


def experiment(root, source, binary, seeds, workers, seconds, code=Path('.')):
    root.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for seed in seeds:
        target = root/'data'/str(seed)
        target.mkdir(parents=True)
        inputs[str(seed)] = {}
        for name in ('input.txt', 'environment.txt', 'worlds.txt', 'oracle.json'):
            (target/name).write_bytes((source/'data'/str(seed)/name).read_bytes())
            inputs[str(seed)][name] = checksum(target/name)
    plan = {
        'seeds': seeds, 'profiles': PROFILES, 'source': str(source),
        'binary_sha256': checksum(binary), 'native_wall_seconds_per_arm': seconds,
        'input_sha256': inputs,
        'acceptance': {'audits': 'All arms independently audited; preserve all failures and timeouts.',
                       'candidate_per_seed_per_stage_minimum': .95,
                       'candidate_old_task_loss_vs_td_maximum': .02,
                       'primitive_retention': 'All primitive cases pass on every seed.',
                       'default_parity': 'TD behavior matches archived interrupted-goal stream.'},
        'scope': 'Short-task retention and frozen host-scheduled interruption transfer. '
                 'The separate goal-scoped credit contracts are required; this screen '
                 'does not establish autonomous scheduling or long sparse-task learning.'}
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    files = [*(code/'cpp/include').rglob('*.h'), *(code/'cpp/apps').glob('*.h'),
             *(code/'cpp/tests').glob('*.cpp'), *(code/'python').glob('*.py'),
             code/'cpp/apps/laminar_train.cpp', code/'cpp/src/organism.cpp']
    manifest = {}
    for path in files:
        relative = path.relative_to(code)
        saved = root/'source'/relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(path.read_bytes())
        manifest[str(relative)] = checksum(saved)
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run, root, seed, profile, binary, (), seconds)
                for seed in seeds for profile in PROFILES]
        for future in concurrent.futures.as_completed(jobs):
            results.append(future.result())
            (root/'summary.json').write_text(json.dumps(sorted(results, key=lambda r: (r['seed'], r['profile'])), indent=2)+'\n')
    return summarize(root, source)


def summarize(root, source):
    plan = json.loads((root/'plan.json').read_text())
    results = json.loads((root/'summary.json').read_text())
    expected = {(p, s) for p in PROFILES for s in plan['seeds']}
    complete = len(results) == len(expected) and {(r['profile'], r['seed']) for r in results} == expected
    acceptance = {'complete': complete, 'all_audited': complete and all(r['audited'] for r in results),
                  'candidate_pass': complete, 'default_parity': {}, 'scores': {}, 'failures': []}
    indexed = {(r['profile'], r['seed']): r for r in results}
    for seed in plan['seeds']:
        td = indexed.get(('controller50k', seed), {})
        for profile in PROFILES:
            result = indexed.get((profile, seed), {})
            key = f'{profile}/{seed}'
            if not result.get('audited'):
                acceptance['failures'].append(f'{key}: missing or failed audit')
                acceptance['candidate_pass'] = False
                continue
            assert result['inputs_sha256'] == {name: plan['input_sha256'][str(seed)][name]
                                                for name in ('input.txt', 'environment.txt', 'worlds.txt')}
            metrics = result['control']['metrics']
            acceptance['scores'][key] = {stage: [metrics[stage]['correct'], metrics[stage]['total']] for stage in STAGES}
            if profile == 'controller50k':
                suffixes = ('txt', 'final.weights', 'final.primitive_weights', 'stage4.values', 'config.json')
                same = {name: checksum(root/profile/str(seed)/f'learned_progress.{name}') ==
                        checksum(source/profile/str(seed)/f'learned_progress.{name}') for name in suffixes}
                acceptance['default_parity'][str(seed)] = same
                if not all(same.values()):
                    acceptance['candidate_pass'] = False
                    acceptance['failures'].append(f'{key}: baseline changed')
            if profile != 'controller50k_trace':
                continue
            for stage in STAGES:
                score = metrics[stage]['correct']/metrics[stage]['total']
                reference = td.get('control', {}).get('metrics', {}).get(stage)
                loss = reference['correct']/reference['total']-score if reference else 1.
                if score < .95 or loss > .020000001:
                    acceptance['candidate_pass'] = False
                    acceptance['failures'].append(f'{key}/{stage}: accuracy={score:.6f}, loss_vs_td={loss:.6f}')
            for task in range(3):
                score = result['primitive_scores'][f'checkpoint/task{task}/disabled-1']
                if score['correct'] != score['n']:
                    acceptance['candidate_pass'] = False
                    acceptance['failures'].append(f'{key}: primitive {task} forgotten')
    acceptance['candidate_pass'] &= acceptance['all_audited']
    (root/'acceptance.json').write_text(json.dumps(acceptance, indent=2)+'\n')
    print(json.dumps(acceptance, indent=2), flush=True)
    return acceptance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('logs/interrupted_goals'))
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--code', type=Path, default=Path('.'), help='Exact frozen source tree used to build binary')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--seconds', type=int, default=900)
    args = parser.parse_args()
    if args.seconds <= 0 or args.workers <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('positive budgets and distinct seeds required')
    result = experiment(args.root, args.source, args.binary, args.seeds, args.workers, args.seconds, args.code)
    raise SystemExit(0 if result['candidate_pass'] else 1)
