"""Canonical frozen-policy transfer across host-scheduled goal interruptions.

Suspension/resumption are declared memory primitives, not learned scheduling.
Only the already trained controller chooses computation and physical operations.
"""
import argparse
import concurrent.futures
import copy
import json
import random
from pathlib import Path
from capacity_experiment import run, checksum
from relational_session_probe import observed, action_text


def build(source, target, seed):
    target.mkdir(parents=True, exist_ok=False)
    for name in ('input.txt', 'environment.txt', 'worlds.txt', 'oracle.json'):
        (target/name).write_bytes((source/name).read_bytes())
    meta = json.loads((target/'oracle.json').read_text())
    commands = (target/'input.txt').read_text().splitlines()
    last = commands.index('CHECK longer')
    templates = {c['id']: c for c in meta['cases']}
    parents = [templates[row.split()[1]] for row in commands[last+1:] if row.startswith('TEST ')]
    children = [c for c in meta['cases'] if c['stage'] == 4 and not c['training'] and not c['effect'] and c['disabled'] == -1]
    worlds = []; private = []; rng = random.Random(29000+seed); serial = 0

    def fresh(original, role):
        nonlocal serial
        serial += 1; entities = set()
        for key in ('truth', 'initial', 'memory'):
            if original[key]:
                entities.update(original[key]['args'])
                for _, args in original[key]['facts']: entities.update(args)
        for a in original['available']: entities.update(a['args'])
        names = {old: f'g{seed}x{serial}x{i}' for i, old in enumerate(sorted(entities))}
        def rename(value):
            if isinstance(value, str): return names.get(value, value)
            if isinstance(value, list): return [rename(v) for v in value]
            if isinstance(value, dict): return {k: rename(v) for k, v in value.items()}
            return value
        c = rename(copy.deepcopy(original)); c['text'] = ' '.join(names.get(w, w) for w in original['text'].split())
        c.update(id=f'interleaved{serial}', stage=5, training=False, interruption_role=role)
        meta['cases'].append(c)
        worlds.append(f'W {c["id"]} {observed(c["truth"])} {int(c["memory"] is not None)} {c["memory_revision"]} '+observed(c['memory'] or {'args': [], 'facts': [], 'complete': False}))
        worlds.append(f'X {c["id"]} {c["failures"]} {int(c["sensor_unavailable"])}')
        a, b = c['truth']['args']; private.append(f'Q {c["id"]} {c["kind"]} {a} {b} {c["r"]} {c["s"]}')
        return c

    def wire(c):
        return f'{c["id"]} {int(c["effect"])} {c["price"]} {c["disabled"]} '+observed(c['initial'])+' '+json.dumps(c['text'])+f' {len(c["available"])} '+' '.join(action_text(a) for a in c['available'])

    commands.append('CHECK interrupted')
    for template in parents:
        for after in ((1,), (3,), (1, 5)):
            parent = fresh(template, 'parent'); tail = []
            for decision in after:
                child = fresh(rng.choice(children), 'child'); tail.append(f'{decision} '+wire(child))
            commands.append('INTERLEAVE_TEST '+wire(parent)+f' {len(after)} '+' '.join(tail))
    commands.append('ENDCHECK')
    (target/'input.txt').write_text('\n'.join(commands)+'\n')
    for name, rows in (('worlds', worlds), ('environment', private)):
        with (target/f'{name}.txt').open('a') as out: out.write('\n'.join(rows)+'\n')
    meta['goal_interruption'] = {'host_schedules_switches': True, 'new_world_information_on_resume': False,
                                 'training_unchanged': True, 'parent_goals': 36, 'child_goals': 48}
    (target/'oracle.json').write_text(json.dumps(meta)+'\n')


def experiment(root, source, seeds, binary, workers):
    root.mkdir(parents=True, exist_ok=False)
    for seed in seeds: build(source/'data'/str(seed), root/'data'/str(seed), seed)
    paths = [*Path('cpp/include').rglob('*.h'), *Path('cpp/apps').glob('*.h'), *Path('python').glob('*.py'),
             Path('cpp/apps/laminar_train.cpp'), Path('cpp/src/organism.cpp')]
    manifest = {}
    for path in paths:
        saved = root/'source'/path; saved.parent.mkdir(parents=True, exist_ok=True); saved.write_bytes(path.read_bytes())
        manifest[str(saved)] = checksum(saved)
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run, root, seed, 'controller50k', binary) for seed in seeds]
        for future in concurrent.futures.as_completed(jobs):
            result = future.result(); results.append(result)
            if result['audited']:
                target = root/'controller50k'/str(result['seed']); baseline = source/'controller50k'/str(result['seed'])
                for name in ('final.weights', 'final.primitive_weights', 'stage4.values'):
                    assert (target/f'learned_progress.{name}').read_bytes() == (baseline/f'learned_progress.{name}').read_bytes(), 'interruption implementation changed baseline training'
                print(json.dumps({'seed': result['seed'], 'metrics': {stage: result['control']['metrics'][stage] for stage in ('interrupted', 'interrupted/children')},
                                  'resumptions': result['control']['checks'].get('goal_resumptions', 0),
                                  'skipped': result['control']['checks'].get('skipped_interrupted_goals', 0)}), flush=True)
            (root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    assert all(r['audited'] for r in results), 'Interruption audit failed; see preserved results'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/interrupted_goals'))
    p.add_argument('--source', type=Path, default=Path('logs/warm_start_screen'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--workers', type=int, default=3)
    p.add_argument('--binary', type=Path, default=Path('build-wsl/laminar_train_interleaved'))
    a = p.parse_args(); experiment(a.root, a.source, a.seeds, a.binary, a.workers)
