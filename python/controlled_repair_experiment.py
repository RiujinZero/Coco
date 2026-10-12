"""Available-skill repair after one canonical learner's language/action warm-up."""
import argparse
import concurrent.futures
import copy
import json
from pathlib import Path
from capacity_experiment import checksum, run
from check_composed_credit import load_questions
from check_primitive_discovery import rows as primitive_rows
from relational_session_probe import observed, action_text


def build(root, seed, source, primitive_source):
    data = root/'data'/str(seed); data.mkdir(parents=True)
    old = source/'data'/str(seed)
    meta = json.loads((old/'oracle.json').read_text())
    commands = (old/'input.txt').read_text().splitlines()
    private = (old/'environment.txt').read_text().splitlines()
    worlds = (old/'worlds.txt').read_text().splitlines()
    # Replay correct but underspecified direct lessons. Import no neural policy.
    ids = {cue: sorted(r['id'] for r in meta['teaching'] if r['type'] == 'P' and int(r['record'].split()[2]) == cue) for cue in range(3)}
    positions = [0, 0, 0]; priors = []
    for r in primitive_rows(primitive_source/f'{seed}.prior.cases.tsv'):
        cue = int(r['task']); rid = ids[cue][positions[cue]]; positions[cue] += 1
        facts = [tuple(map(int, f.split(','))) for f in r['facts'].rstrip(';').split(';') if f]
        if cue == 2: assert len(facts) == len(set(facts))
        fields = [cue, r['relation'], r['subject_type'], r['subject'], r['target_type'], r['target'], r['gold_type'], r['gold'], len(facts)]
        record = f'P {rid} '+' '.join(map(str, fields+[v for fact in facts for v in fact]))
        priors.append(record)
    assert positions == [40, 40, 40]
    commands = priors+[r for r in commands if not r.startswith('P ')]
    meta['teaching'] = [dict(type='P', id=r.split()[1], record=r) for r in priors]+[r for r in meta['teaching'] if r['type'] != 'P']
    finals = [c for c in meta['cases'] if not c['training'] and c['stage'] == 4 and c['disabled'] == -1][:128]
    assert len(finals) == 128
    serial = 0

    def add_case(template, stage, training, duplicate=False):
        nonlocal serial
        serial += 1; c = copy.deepcopy(template); cid = f'repair_case{serial}'
        # Fresh names avoid accidental episode or cached query reuse.
        rename = {x: f'q{serial}x{i}' for i, x in enumerate(sorted({x for _, vs in c['truth']['facts'] for x in vs} | set(c['truth']['args'])))}
        def renamed(w):
            w['args'] = [rename.get(x, x) for x in w['args']]
            w['facts'] = [[p, [rename.get(x, x) for x in vs]] for p, vs in w['facts']]
        renamed(c['truth']); c['text'] = ' '.join(rename.get(x, x) for x in c['text'].split())
        for a in c['available']: a['args'] = [rename.get(x, x) for x in a['args']]
        if duplicate:
            assert c['kind'] == 0 and c['gold'] == 1 and not c['effect']
            fact = next(f for f in c['truth']['facts'] if f[0] == c['r'] and f[1][0] == c['truth']['args'][0])
            c['truth']['facts'].append(copy.deepcopy(fact))
        c.update(id=cid, stage=stage, training=training, disabled=-1, variant=0, initial=copy.deepcopy(c['truth']),
                 memory=None, memory_revision=0, visible_mode=0, sensor_unavailable=False, failures=0)
        meta['cases'].append(c)
        commands.append(f'{"TRAIN" if training else "TEST"} {cid} {int(c["effect"])} {c["price"]} -1 '+observed(c['initial'])+' '+json.dumps(c['text'])+f' {len(c["available"])} '+' '.join(action_text(a) for a in c['available']))
        worlds.extend([f'W {cid} '+observed(c['truth'])+' 0 0 0 0 0', f'X {cid} 0 0'])
        a, b = c['truth']['args']; private.append(f'Q {cid} {c["kind"]} {a} {b} {c["r"]} {c["s"]}')
        return c

    positives = [c for c in finals if c['family'] == 0 and c['gold'] and not c['effect']]
    assert positives
    commands.append('CHECK before_repair')
    for i in range(32): add_case(positives[i % len(positives)], 'before_repair', False, duplicate=True)
    commands.append('ENDCHECK')
    questions = load_questions(primitive_source/f'{seed}.questions.txt')
    counterexamples = 0
    for identifier, q in questions.items():
        if q['phase'] != 'train': continue
        a = 'raw'+str(q['a'][1]); numeric = q['b'][0] == 2
        b = str(q['b'][1]) if numeric else 'raw'+str(q['b'][1]); r, s = ['rel'+str(v) for v in q['relations']]
        world = {'args': [a, b], 'complete': True, 'facts': [['rel'+str(p), ['raw'+str(x), 'raw'+str(y)]] for x, p, y in q['facts']]}
        text = f'{"anchor" if numeric else "vex"} {a} with {b} via {r} and {s}'; rid = f'repair_composed{identifier}'
        op = 'N' if numeric else 'M'; quantities = [b] if numeric else []
        record = f'{op} {rid} {q["answer"][1]} '+observed(world)+' '+json.dumps(text)+f' 2 {r} {s}'
        if numeric: record += f' 1 {b}'
        commands.append(record); counterexamples += 1
        meta['teaching'].append(dict(type=op, id=rid, kind=3 if numeric else 0, family=5 if numeric else 0,
                                    text=text, r=r, s=s, quantities=quantities, gold=q['answer'][1], world=world, truth=world))
    assert counterexamples == 64
    # Continue the same controller without any renewed hints.
    for i in range(32): add_case(finals[(i*7) % len(finals)], 'repair_adaptation', True)
    commands.append('CHECK after_repair')
    for i in range(32): add_case(positives[i % len(positives)], 'after_repair', False, duplicate=True)
    commands.append('ENDCHECK')
    commands.append('CHECK retained')
    for c in finals: add_case(c, 'retained', False)
    commands.append('ENDCHECK')
    meta.update(available_primitive_repair=True, direct_prior_source=str(primitive_source/f'{seed}.prior.cases.tsv'),
                repair_composed_source=str(primitive_source/f'{seed}.questions.txt'), repair_counterexamples=64)
    for name, data_rows in [('input.txt', commands), ('environment.txt', private), ('worlds.txt', worlds)]:
        (data/name).write_text('\n'.join(data_rows)+'\n')
    (data/'oracle.json').write_text(json.dumps(meta)+'\n')


def experiment(root, source, primitive_source, binary, seeds, workers,
               profiles=('controller50k', 'controller50k_repair'), timeout_seconds=None, lifecycle=False):
    root.mkdir(parents=True, exist_ok=False)
    files = [*Path('cpp/include').rglob('*.h'), *Path('cpp/apps').glob('*.h'), *Path('python').glob('*.py'),
             Path('cpp/apps/laminar_train.cpp'), Path('cpp/src/organism.cpp'), Path('cpp/apps/verify_local_choices.cpp')]
    manifest = {}
    for path in files:
        target = root/'source'/path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
        manifest[str(path)] = checksum(target)
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for seed in seeds: build(root, seed, source, primitive_source)
    plan = dict(seeds=seeds, profiles=list(profiles), native_wall_budget_seconds=timeout_seconds,
                binary_sha256=checksum(binary), workers=workers, repair_lifecycle=lifecycle,
                acceptance=dict(new_family_minimum=0.95, maximum_old_task_loss=0.02,
                                primitive_correct=324, primitive_total=324),
                scope=('Repeated observation records remain visible; relational truth counts distinct tuples.'
                       if all(profile.endswith('_records') for profile in profiles) else
                       'Preserved-stream integration/retention; duplicate control scenes normalize to sets and cannot establish visible error repair.'))
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run, root, seed, profile, binary, ['audit_models=1', 'search_progress=1']+
                            (['lifecycle=1'] if lifecycle and '_repair_' in profile else []), timeout_seconds)
                for seed in seeds for profile in profiles]
        for future in concurrent.futures.as_completed(jobs):
            result = future.result(); results.append(result)
            (root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    assert all(r['audited'] for r in results), 'Retained native outputs and audit errors require inspection'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/controlled_repair'))
    p.add_argument('--source', type=Path, default=Path('logs/warm_start_screen'))
    p.add_argument('--primitive-source', type=Path, default=Path('logs/primitive_repair_full/repair'))
    p.add_argument('--binary', type=Path, default=Path('build-wsl/laminar_train_repair_v4'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--profiles', nargs='+', default=['controller50k', 'controller50k_repair'])
    p.add_argument('--seconds', type=int)
    p.add_argument('--lifecycle', action='store_true')
    a = p.parse_args()
    if a.seconds is not None and a.seconds <= 0: p.error('--seconds must be positive')
    experiment(a.root, a.source, a.primitive_source, a.binary, a.seeds, a.workers, a.profiles, a.seconds, a.lifecycle)
