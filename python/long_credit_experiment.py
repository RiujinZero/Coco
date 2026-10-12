"""Longer causal action acquisition after the exact archived warm start.

The existing longer-goal screen needs only three missing effects, even in its
larger worlds. This stream requires 4--6 distinct effects during adaptation and
holds out 7--8. It retains the canonical 48/128 decision limits and costs. A
separate frozen preflight measures those limits before an expensive campaign.
"""
import argparse
import concurrent.futures
import copy
import json
import random
from pathlib import Path

from capacity_experiment import checksum, run
from computational_control_oracle import truth
from relational_session_probe import action_text, apply_effect, observed


BASE_PROFILES = ('controller50k', 'controller50k_trace')
CONTROL_PROFILES = tuple(base+suffix for base in BASE_PROFILES for suffix in ('_frozen', '_shuffled', '_terminal'))
PHASE_PROFILES = tuple(base+suffix for base in BASE_PROFILES for suffix in ('_frozen_after400', '_shuffled_after400'))
PROFILES = BASE_PROFILES+CONTROL_PROFILES+PHASE_PROFILES
FILES = ('input.txt', 'environment.txt', 'worlds.txt', 'oracle.json')


def build(source, target, seed, episodes_per_length=24, preflight=False):
    target.mkdir(parents=True, exist_ok=False)
    commands = (source/'input.txt').read_text().splitlines()
    private = (source/'environment.txt').read_text().splitlines()
    worlds = (source/'worlds.txt').read_text().splitlines()
    meta = json.loads((source/'oracle.json').read_text())
    baseline_train = sum(line.startswith('TRAIN ') for line in commands)
    guidance = [line.split() for line in commands if line.startswith('GUIDANCE ')]
    assert len(guidance) <= 1
    if guidance:
        assert baseline_train >= int(guidance[0][2])+int(guidance[0][3]), 'base warm-start assistance has not ended'
    final_start = commands.index('CHECK stage4')+1
    final_end = commands.index('ENDCHECK', final_start)
    old_ids = {line.split()[1] for line in commands[final_start:final_end] if line.startswith('TEST ')}
    old = [copy.deepcopy(c) for c in meta['cases'] if c['id'] in old_ids]
    assert len(old) == 128, 'expected exact archived final 128-case retention set'
    rng = random.Random(97123+seed)
    serial = 0

    def emit(c, stage, training, missing=None):
        nonlocal serial
        serial += 1
        c.update(id=f'longcredit{seed}x{serial}', stage=6, training=training,
                 disabled=-1, variant=0, long_credit_stage=stage)
        if missing is not None:
            c['minimum_distinct_motor_effects'] = missing
        meta['cases'].append(c)
        commands.append(f'{"TRAIN" if training else "TEST"} {c["id"]} {int(c["effect"])} {c["price"]} -1 '
                        +observed(c['initial'])+' '+json.dumps(c['text'])
                        +f' {len(c["available"])} '+' '.join(action_text(a) for a in c['available']))
        worlds.append(f'W {c["id"]} {observed(c["truth"])} {int(c["memory"] is not None)} {c["memory_revision"]} '
                      +observed(c['memory'] or {'args': [], 'facts': [], 'complete': False}))
        worlds.append(f'X {c["id"]} {c["failures"]} {int(c["sensor_unavailable"])}')
        a, b = c['truth']['args']
        private.append(f'Q {c["id"]} {c["kind"]} {a} {b} {c["r"]} {c["s"]}')

    def fixture(missing, family, price):
        # Include two already satisfied objects and an irrelevant valid motor;
        # world size alone is deliberately different from the causal horizon.
        prefix = f'lc{seed}q{serial+1}e'
        a, b = prefix+'a', prefix+'b'
        objects = [prefix+str(i) for i in range(missing+2)]
        relation = {0: 'left', 3: 'eligible', 4: 'eligible2'}[family]
        motor = {0: 'motor0', 3: 'motor1', 4: 'motor2'}[family]
        facts, available = [], []
        for i, obj in enumerate(objects):
            if family >= 3 or i >= missing:
                facts.append(['left', [a, obj]])
            if family >= 3 and (family == 4 or i >= missing):
                facts.append(['marked', [a, obj]])
            if family == 4 and i >= missing:
                facts.append(['ready', [a, obj]])
            facts.append(['right', [b, obj]])
            available.append({'schema': motor, 'args': [a, obj]})
        facts.append(['noise', [a, b]])
        available.append({'schema': motor, 'args': [b, a]})
        rng.shuffle(facts)
        rng.shuffle(available)
        world = {'args': [a, b], 'facts': facts, 'complete': True}
        c = {'family': family, 'kind': 0, 'text': f'vex {a} with {b} via {relation} and right',
             'r': relation, 's': 'right', 'gold': 0, 'truth': world, 'initial': copy.deepcopy(world),
             'available': available, 'memory': None, 'memory_revision': 1, 'effect': True,
             'price': price, 'visible_mode': 0, 'sensor_unavailable': False, 'failures': 0}
        assert truth(meta, c, world) == 0
        required = [{'schema': motor, 'args': [a, obj]} for obj in objects[:missing]]
        solved = copy.deepcopy(world)
        for action in required:
            solved = apply_effect(solved, action, meta['effects'])
        assert truth(meta, c, solved) == 1
        # Each action adds at most its own one missing literal. Omitting any
        # required action leaves the goal false; no hidden intermediate reward.
        for omitted in range(missing):
            partial = copy.deepcopy(world)
            for i, action in enumerate(required):
                if i != omitted:
                    partial = apply_effect(partial, action, meta['effects'])
            assert truth(meta, c, partial) == 0
        return c

    def check(stage, lengths, repeats=1, families=(0, 3, 4), prices=(0, 1)):
        commands.append(f'CHECK {stage}')
        for missing in lengths:
            for family in families:
                for price in prices:
                    for _ in range(repeats):
                        emit(fixture(missing, family, price), stage, False, missing)
        commands.append('ENDCHECK')

    if preflight:
        check('long_preflight', (4, 5, 6, 7, 8), families=(0,), prices=(0,))
    else:
        check('long_before', (4, 5, 6, 7, 8))
        for missing in (4, 5, 6):
            for block in range(episodes_per_length//6):
                order = [(family, price) for family in (0, 3, 4) for price in (0, 1)]
                rng.shuffle(order)
                for family, price in order:
                    emit(fixture(missing, family, price), f'adaptation{missing}', True, missing)
            check(f'long_acquired{missing}', (missing,), repeats=2)
        check('long_heldout7', (7,), repeats=2)
        check('long_heldout8', (8,), repeats=2)
        commands.append('CHECK long_retention')
        for case in old:
            emit(case, 'long_retention', False)
        commands.append('ENDCHECK')
    meta['long_credit'] = {
        'kind': 'frozen_preflight' if preflight else 'terminal_outcome_adaptation',
        'baseline_training_episodes': baseline_train,
        'adaptation_lengths': [] if preflight else [4, 5, 6],
        'held_out_lengths': [7, 8], 'episodes_per_length': 0 if preflight else episodes_per_length,
        'additional_hints': 0, 'world_objects': 'missing effects plus two already satisfied objects',
        'limits': {'training_decisions': 48, 'evaluation_decisions': 128,
                   'controller_trace': 256, 'planner_depth': 8, 'planner_states': 4096,
                   'motor_attempts': 32, 'entity_names': 32},
        'supplied': ['existing typed language/concepts', 'learned existing motor semantics',
                     'supplied breadth-first planner', 'existing primitive bank'],
        'required_credit_controls': ['frozen_control', 'shuffled_credit', 'terminal_only_credit'],
        'control_scope': 'Whole-stream controls differ during warm start; _after400 controls match the original bootstrap.',
    }
    for name, rows in (('input', commands), ('environment', private), ('worlds', worlds)):
        (target/f'{name}.txt').write_text('\n'.join(rows)+'\n')
    (target/'oracle.json').write_text(json.dumps(meta)+'\n')
    return meta['long_credit']


def summarize(root):
    plan = json.loads((root/'plan.json').read_text())
    results = json.loads((root/'summary.json').read_text())
    expected = {(p, s) for p in plan['profiles'] for s in plan['seeds']}
    summary = {'complete': {(r['profile'], r['seed']) for r in results} == expected,
               'audited': all(r['audited'] for r in results), 'scores': {},
               'credit_gate_established': False,
               'scope': 'Feasibility/acquisition evidence. Whole-stream ablations alone do not isolate acquisition after an identical bootstrap.',
               'selected_control_arms': [p for p in plan['profiles'] if p in CONTROL_PROFILES+PHASE_PROFILES],
               'phase_boundary_parity': {}}
    for result in results:
        label = f'{result["profile"]}/{result["seed"]}'
        if not result['audited']:
            summary['scores'][label] = {'failure': result.get('audit_error'),
                                       'timed_out': result.get('timed_out', False)}
            continue
        metrics = result['control']['metrics']
        scores = {stage: value for stage, value in metrics.items() if stage.startswith('long_') and '/' not in stage}
        # Per-case terminal records preserve failures, actual causal actions,
        # compute costs and whether a successful frozen trace fits 48 decisions.
        directory = root/result['profile']/str(result['seed'])
        meta = json.loads((directory/'oracle.json').read_text())
        cases = {c['id']: c for c in meta['cases'] if 'minimum_distinct_motor_effects' in c}
        episodes, observed_costs, plan_statuses = [], {}, {}
        for line in (directory/'learned_progress.txt').read_text().splitlines():
            fields = line.split()
            if len(fields) > 1 and fields[1] in cases and fields[0] == 'S':
                observed_costs.setdefault(fields[1], []).append(float(fields[5]))
            if len(fields) > 1 and fields[1] in cases and fields[0] == 'D':
                statuses = plan_statuses.setdefault(fields[1], {})
                status = fields[3].strip('"')
                statuses[status] = statuses.get(status, 0)+1
            if fields and fields[0] == 'F' and fields[1] in cases:
                case = cases[fields[1]]
                costs = observed_costs[fields[1]]
                reward = 1 if int(fields[2]) else -1
                discounted = reward*.99**(len(costs)-1)-sum(.99**i*cost for i, cost in enumerate(costs))
                if reward == 1:
                    assert int(fields[7]) >= case['minimum_distinct_motor_effects'], 'scored success did not execute required causal effects'
                episodes.append({'id': fields[1], 'stage': case['long_credit_stage'],
                                 'family': case['family'], 'price': case['price'],
                                 'minimum_motor_effects': case['minimum_distinct_motor_effects'],
                                 'correct': bool(int(fields[2])), 'timeout': bool(int(fields[5])),
                                 'decisions': int(fields[6]), 'motor_attempts': int(fields[7]),
                                 'cost': float(fields[8]),
                                 'reward_minus_cost': reward-float(fields[8]),
                                 'discounted_observed_return': discounted,
                                 'immediate_failure_return': -1,
                                 'observed_return_better_than_immediate_failure': discounted > -1,
                                 'plan_status_counts': plan_statuses.get(fields[1], {}),
                                 'fits_training_decision_limit': int(fields[6]) <= 48})
        scores['episodes'] = episodes
        summary['scores'][label] = scores
        if result['profile'] in PHASE_PROFILES:
            base = result['profile'].rsplit('_', 2)[0]
            baseline = root/base/str(result['seed'])
            matches = {}
            for suffix in ('stage4.weights', 'stage4.primitive_weights', 'stage4.values'):
                reference = baseline/f'learned_progress.{suffix}'
                matches[suffix] = (checksum(reference) == checksum(directory/f'learned_progress.{suffix}')) if reference.exists() else None
            summary['phase_boundary_parity'][label] = matches
    (root/'acceptance.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'scores'}, indent=2), flush=True)
    return summary


def experiment(args):
    args.root.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for seed in args.seeds:
        metadata[str(seed)] = build(args.source/'data'/str(seed), args.root/'data'/str(seed), seed,
                                    args.episodes_per_length, args.preflight)
    if any(p in PHASE_PROFILES for p in args.profiles):
        assert all(m['baseline_training_episodes'] == 400 for m in metadata.values()), 'phase controls require the declared 400-episode bootstrap'
    plan = {'source': str(args.source), 'seeds': args.seeds, 'profiles': args.profiles,
            'native_wall_seconds_per_arm': args.seconds, 'preflight': args.preflight,
            'metadata': metadata, 'input_sha256': {
                str(seed): {name: checksum(args.root/'data'/str(seed)/name) for name in FILES}
                for seed in args.seeds},
            'preflight_evidence': str(args.preflight_evidence) if args.preflight_evidence else None,
            'success_claim_requires': ['all independent audits', 'all declared seeds retained',
                                      'gain over before-acquisition cases', 'old-task retention',
                                      'matched post-bootstrap credit controls']}
    if args.binary:
        plan['binary_sha256'] = checksum(args.binary)
    plan['driver_sha256'] = checksum(Path(__file__))
    if not args.prepare_only:
        manifest = {}
        paths = [*(args.code/'cpp/include').rglob('*.h'), *(args.code/'cpp/apps').glob('*.h'),
                 *(args.code/'cpp/tests').glob('*.cpp'), *(args.code/'python').glob('*.py'),
                 args.code/'cpp/apps/laminar_train.cpp', args.code/'cpp/src/organism.cpp']
        for path in paths:
            relative = path.relative_to(args.code)
            saved = args.root/'source'/relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(path.read_bytes())
            manifest[str(relative)] = checksum(saved)
        (args.root/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    if args.prepare_only:
        print(f'Prepared {len(args.seeds)} streams in {args.root}; no training started.', flush=True)
        return True
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, args.root, seed, profile, args.binary, (), args.seconds)
                   for seed in args.seeds for profile in args.profiles]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            (args.root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    result = summarize(args.root)
    return result['complete'] and result['audited']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('logs/interrupted_goals'))
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--code', type=Path, default=Path('.'), help='Exact frozen source tree used to build binary')
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(BASE_PROFILES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--episodes-per-length', type=int, default=24)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-evidence', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if min(args.workers, args.seconds, args.episodes_per_length) <= 0 or args.episodes_per_length % 6:
        parser.error('positive budgets and episodes-per-length divisible by six required')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.profiles)) != len(args.profiles):
        parser.error('distinct seeds and profiles required')
    if not args.prepare_only and not args.binary:
        parser.error('--binary is required to run')
    if not args.prepare_only and not args.preflight:
        if not args.preflight_evidence:
            parser.error('run and inspect a frozen preflight first, then provide --preflight-evidence')
        evidence = json.loads((args.preflight_evidence/'acceptance.json').read_text())
        plan = json.loads((args.preflight_evidence/'plan.json').read_text())
        if not plan['preflight'] or not evidence['complete'] or not evidence['audited']:
            parser.error('preflight must be complete and audited; preserve failures rather than bypassing the execution-limit review')
    raise SystemExit(0 if experiment(args) else 1)
