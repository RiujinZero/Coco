"""Continue the canonical learner while a new concept is unresolved, then learn it.

Old-language/control/action training continues during ambiguity. New contrastive
concept evidence resolves a previously untaught conjunction, followed by fresh
physical goals using it. Every frozen evaluation remains unassisted. The shared
audit checks all observations, policy choices, physics and learned definitions.
"""
import argparse
import concurrent.futures
import copy
import json
import random
from pathlib import Path
from capacity_experiment import run, checksum
from computational_control_probe import build as fresh_data
from concept_growth_probe import encode_world
from relational_session_probe import observed, action_text, clause_text, apply_effect


def build(source, target, seed):
    target.mkdir(parents=True)
    original = source/'data'/str(seed)
    meta = json.loads((original/'oracle.json').read_text())
    commands = (original/'input.txt').read_text().splitlines()
    # Match the previously tested warm start; all additions occur after it ends.
    commands.insert(0, 'GUIDANCE public_control_v1 32 128')
    private = (original/'environment.txt').read_text().splitlines()
    worlds = (original/'worlds.txt').read_text().splitlines()
    meta['guidance'] = {'teacher': 'public_control_v1', 'full_episodes': 32, 'fade_episodes': 128,
                        'terminal_credit_only': True, 'evaluation_hints': False}
    extra = target/'fresh'; fresh_data(seed+100, extra, 1)
    fresh = json.loads((extra/'oracle.json').read_text())
    fresh_rows = (extra/'input.txt').read_text().splitlines()
    start = fresh_rows.index('CHECK stage4')+1
    ids = {r.split()[1] for r in fresh_rows[start:fresh_rows.index('ENDCHECK', start)]}
    older = [c for c in fresh['cases'] if c['id'] in ids]
    assert len(older) == 128
    rng = random.Random(9000+seed)
    head, red, blue = f'concept{seed}new', f'relation{seed}red', f'relation{seed}blue'
    definition = [2, [[red, [0, 1]], [blue, [0, 1]]]]
    meta['definitions'][head] = definition
    private.append(f'H {head} '+clause_text(definition))
    meta['concept_alternatives'] = {'uncertain_retention': {head: [[2, [[red, [0, 1]]]], [2, [[blue, [0, 1]]]]]}}

    def concept(rid, facts, value):
        w = {'args': ['a', 'b'], 'facts': [[p, ['a', 'b']] for p in facts], 'complete': True}
        commands.append(f'C {rid} {head} {value} '+encode_world(w))
        meta['teaching'].append({'type': 'C', 'id': rid, 'name': head, 'world': w, 'gold': value})

    serial = 0
    def case(c, stage, training=False):
        nonlocal serial
        c = copy.deepcopy(c); cid = f'isolation{serial}'; serial += 1
        c.update(id=cid, stage=stage, training=training)
        meta['cases'].append(c)
        commands.append(f'{"TRAIN" if training else "TEST"} {cid} {int(c["effect"])} {c["price"]} -1 '+observed(c['initial'])+' '+json.dumps(c['text'])+f' {len(c["available"])} '+' '.join(action_text(a) for a in c['available']))
        worlds.append(f'W {cid} {observed(c["truth"])} {int(c["memory"] is not None)} {c["memory_revision"]} '+observed(c['memory'] or {'args': [], 'facts': [], 'complete': False}))
        worlds.append(f'X {cid} {c["failures"]} {int(c["sensor_unavailable"])}')
        a, b = c['truth']['args']; private.append(f'Q {cid} {c["kind"]} {a} {b} {c["r"]} {c["s"]}')

    # Training and test identities/entities are disjoint even when their world
    # structures match. Rename all entity occurrences, including surface mentions.
    def rename(c, suffix):
        c = copy.deepcopy(c)
        entities = set(c['truth']['args']) | {v for _, vs in c['truth']['facts'] for v in vs}
        mapping = {v: v+suffix for v in entities}
        for key in ('truth', 'initial', 'memory'):
            w = c[key]
            if w is None: continue
            w['args'] = [mapping[v] for v in w['args']]
            w['facts'] = [[p, [mapping[v] for v in vs]] for p, vs in w['facts']]
        for a in c['available']: a['args'] = [mapping[v] for v in a['args']]
        c['text'] = ' '.join(mapping.get(w, w) for w in c['text'].split())
        return c

    concept('new_sparse_positive', [red, blue], 1)
    # Continue updating the same model on established skills despite uncertainty.
    for i in range(32): case(rename(older[i*4], 'train'), 'uncertain_train', True)
    commands.append('CHECK uncertain_retention')
    for c in older: case(rename(c, 'uncertain'), 'uncertain_retention')
    commands.append('ENDCHECK')
    concept('new_contrast_red', [red], 0)
    concept('new_contrast_blue', [blue], 0)
    for index, predicate in enumerate((red, blue)):
        motor = f'newmotor{index}'
        meta['effects'][motor] = {'predicate': predicate, 'subject': 0, 'object': 1, 'kind': 0}
        private.append(f'E {motor} {predicate} 0 1 0')
        a = {'schema': motor, 'args': ['a', 'b']}
        before = {'args': ['a', 'b'], 'facts': [[predicate, ['a', 'old']], [predicate, ['other', 'b']]], 'complete': True}
        after = apply_effect(before, a, meta['effects']); rid = f'new_effect{index}'
        commands.append(f'T {rid} '+action_text(a)+' '+observed(before)+' '+observed(after))
        meta['teaching'].append({'type': 'T', 'id': rid, 'action': a, 'before': before, 'after': after})

    def novel(i):
        tag = f'new{seed}x{i}x'; a, b = tag+'a', tag+'b'
        n = 1+i%3; facts = []; available = []
        missing_red, missing_blue = i%2, (i//2)%2
        for j in range(n):
            x = tag+str(j)
            if j >= missing_red: facts.append([red, [a, x]])
            if j >= missing_blue: facts.append([blue, [a, x]])
            facts.append(['right', [b, x]])
            available.extend({'schema': f'newmotor{k}', 'args': [a, x]} for k in range(2))
        facts.append(['noise', [a, b]]); rng.shuffle(facts); rng.shuffle(available)
        truth = {'args': [a, b], 'facts': facts, 'complete': True}; view = i%4
        visible = copy.deepcopy(truth)
        if view: visible['complete'] = False
        if view == 1: visible['facts'] = []
        if view == 2: visible['facts'] = visible['facts'][::2]
        return {'family': 5, 'kind': 0, 'text': f'vex {a} with {b} via {head} and right', 'r': head, 's': 'right',
                'gold': int(not missing_red and not missing_blue), 'truth': truth, 'available': available,
                'initial': visible, 'memory': copy.deepcopy(truth) if view in (1, 3) else None,
                'memory_revision': 0 if view == 3 else 1, 'effect': True, 'price': (i//4)%2,
                'visible_mode': view, 'sensor_unavailable': view == 1, 'failures': int(i%5 == 0),
                'disabled': -1, 'variant': 0}

    for i in range(32):
        case(novel(i), 'new_train', True)
        case(rename(older[(i*4+1)%128], 'latertrain'), 'old_train', True)
    commands.append('CHECK acquired_retention')
    for c in older: case(rename(c, 'resolved'), 'acquired_retention')
    commands.append('ENDCHECK')
    commands.append('CHECK new_concept_transfer')
    for i in range(32, 96): case(novel(i), 'new_concept_transfer')
    commands.append('ENDCHECK')
    for name, rows in [('input.txt', commands), ('environment.txt', private), ('worlds.txt', worlds)]:
        (target/name).write_text('\n'.join(rows)+'\n')
    (target/'oracle.json').write_text(json.dumps(meta)+'\n')


def experiment(root, source, binary, seeds):
    root.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in [*Path('cpp/include').rglob('*.h'), *Path('cpp/apps').glob('*.h'), *Path('python').glob('*.py'), Path('cpp/apps/laminar_train.cpp'), Path('cpp/src/organism.cpp')]:
        saved = root/'source'/path; saved.parent.mkdir(parents=True, exist_ok=True); saved.write_bytes(path.read_bytes()); manifest[str(path)] = checksum(saved)
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for seed in seeds: build(source, root/'data'/str(seed), seed)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(run, root, seed, 'controller50k', binary) for seed in seeds]
        for future in concurrent.futures.as_completed(jobs):
            results.append(future.result()); (root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    assert all(r['audited'] for r in results), 'See retained audit failure'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('logs/capacity_screen'))
    p.add_argument('--binary', type=Path, default=Path('build-wsl/laminar_train_isolation'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args(); experiment(a.root, a.source, a.binary, a.seeds)
