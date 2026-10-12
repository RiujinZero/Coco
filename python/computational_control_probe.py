"""Evaluator-owned mixed acquisition stream; only raw lessons reach the learner."""
import argparse
import copy
import json
import random
import subprocess
from pathlib import Path
from relational_session_probe import observed, action_text, clause_text, apply_effect
from concept_growth_probe import encode_world


def build(seed, directory, episodes):
    rng = random.Random(16000+seed)
    commands, private, worlds, cases, teaching = [], [], [], [], []
    definitions = {'eligible': [2, [['left', [0, 1]], ['marked', [0, 1]]]],
                   'eligible2': [2, [['left', [0, 1]], ['marked', [0, 1]], ['ready', [0, 1]]]]}
    effects = {f'motor{i}': {'predicate': p, 'subject': 0, 'object': 1, 'kind': 0}
               for i, p in enumerate(('left', 'marked', 'ready'))}
    for name, d in definitions.items(): private.append(f'H {name} '+clause_text(d))
    for name, e in effects.items(): private.append(f'E {name} {e["predicate"]} 0 1 0')
    serial = 0

    def fixture(family, physical=False, size=None, missing=None):
        nonlocal serial
        serial += 1
        fresh = 2
        a, b = f'e{seed}x{serial}x0', f'e{seed}x{serial}x1'
        def entity():
            nonlocal fresh
            fresh += 1
            return f'e{seed}x{serial}x{fresh}'
        n = rng.randrange(1, 5) if size is None else size
        facts = []
        add = lambda p, x, y: facts.append([p, [x, y]])
        relation = ('left', 'left', 'left', 'eligible', 'eligible2')[family]
        r, s = relation, 'right'
        kind = family if family < 3 else 0
        available = []
        if physical or family >= 3:
            absent = rng.randrange(min(n, 2)+1) if missing is None else missing
            for j in range(n):
                obj = entity()
                if family >= 3 or j >= absent: add('left', a, obj)
                if family >= 3 and (family == 4 or j >= absent): add('marked', a, obj)
                if family == 4 and j >= absent: add('ready', a, obj)
                add('right', b, obj)
                motor = 'motor0' if family == 0 else 'motor1' if family == 3 else 'motor2'
                available.append({'schema': motor, 'args': [a, obj]})
            # Irrelevant but valid actions must not be mistaken for progress.
            available.append({'schema': motor, 'args': [b, a]})
            gold = absent == 0
        elif kind == 0:
            m = n if rng.randrange(2) else (n+1) % 5
            if rng.randrange(2): r, s = s, r
            for _ in range(n): add(r, a, entity())
            for _ in range(m): add(s, b, entity())
            gold = n == m
        elif kind == 1:
            n = rng.randrange(4) if size is None else size
            x, y = a, b
            for _ in range(n): z = entity(); add(r, x, z); x = z
            gold = bool(rng.randrange(2)) and n > 0
            depth = max(1, n)
            for i in range(depth): z = x if gold and i+1 == depth else entity(); add(s, y, z); y = z
        else:
            x = a
            for _ in range(1+rng.randrange(3)): z = entity(); add(r, x, z); x = z
            m = n if rng.randrange(2) else (n+1) % 5
            for _ in range(n): add(s, x, entity())
            for _ in range(m): add(s, b, entity())
            gold = n == m
        add('noise', a, b)
        if not available: available = [{'schema': 'motor0', 'args': [b, a]}]
        rng.shuffle(facts); rng.shuffle(available)
        text = f'{("vex", "pim", "zot")[kind]} {a} with {b} via {r} and {s}'
        return {'family': family, 'kind': kind, 'text': text, 'r': r, 's': s, 'gold': int(gold),
                'truth': {'args': [a, b], 'facts': facts, 'complete': True}, 'available': available}

    def language(family):
        for i in range(48):
            c = fixture(family); rid = f'lang{family}x{i}'
            # Mention types are supplied; no family or program is in the command.
            commands.append(f'M {rid} {c["gold"]} '+observed(c['truth'])+' '+json.dumps(c['text'])+f' 2 {c["r"]} {c["s"]}')
            teaching.append(dict(c, id=rid, type='M', world=c['truth']))

    def concept(name):
        base = definitions[name][1]
        positive = {'args': ['u', 'v'], 'facts': [[p, ['u', 'v']] for p, _ in base], 'complete': True}
        for i in range(len(base)+1):
            w = copy.deepcopy(positive)
            if i: w['facts'].pop(i-1)
            rid = f'concept_{name}_{i}'; gold = int(i == 0)
            commands.append(f'C {rid} {name} {gold} '+encode_world(w))
            teaching.append({'type': 'C', 'id': rid, 'name': name, 'world': w, 'gold': gold})

    def effect(i):
        p = effects[f'motor{i}']['predicate']; a = {'schema': f'motor{i}', 'args': ['u', 'v']}
        before = {'args': ['u', 'v'], 'facts': [[p, ['u', 'old']], [p, ['other', 'v']]], 'complete': True}
        after = apply_effect(before, a, effects); rid = f'effect{i}'
        commands.append(f'T {rid} '+action_text(a)+' '+observed(before)+' '+observed(after))
        teaching.append({'type': 'T', 'id': rid, 'action': a, 'before': before, 'after': after})

    def case(stage, family, physical, training, view=None, price=None, template=None, disabled=-1, size=None, missing=None):
        c = copy.deepcopy(template) if template else fixture(family, physical, size, missing)
        cid = f'case{len(cases)}'
        if template is None:
            view = rng.randrange(4) if view is None else view
            price = rng.randrange(2) if price is None else price
            visible = copy.deepcopy(c['truth'])
            if view == 1: visible['facts'] = []; visible['complete'] = False
            elif view == 2: visible['facts'] = visible['facts'][::2]; visible['complete'] = False
            elif view == 3: visible['complete'] = False
            memory = copy.deepcopy(c['truth']) if view in (1, 3) else None
            unavailable = view == 1 and (not training or rng.random() < .5)
            c.update(initial=visible, memory=memory, memory_revision=0 if view == 3 else 1,
                     effect=bool(physical), price=price, visible_mode=view, sensor_unavailable=unavailable,
                     failures=int(training and physical and rng.random() < .25))
        c.update(id=cid, stage=stage, training=training, disabled=disabled, variant=0)
        cases.append(c)
        commands.append(f'{"TRAIN" if training else "TEST"} {cid} {int(c["effect"])} {c["price"]} {disabled} '+observed(c['initial'])+' '+json.dumps(c['text'])+f' {len(c["available"])} '+' '.join(action_text(a) for a in c['available']))
        worlds.append(f'W {cid} {observed(c["truth"])} {int(c["memory"] is not None)} {c["memory_revision"]} '+observed(c['memory'] or {'args': [], 'facts': [], 'complete': False}))
        worlds.append(f'X {cid} {c["failures"]} {int(c["sensor_unavailable"])}')
        a, b = c['truth']['args']; private.append(f'Q {cid} {c["kind"]} {a} {b} {c["r"]} {c["s"]}')
        return c

    raw = subprocess.check_output(['build-wsl/relational_control_probe', '--emit-primitives', str(seed)], text=True).splitlines()
    primitive_blocks = {i: [r for r in raw if int(r.split()[2]) == i] for i in range(3)}
    final = []
    for stage in range(1, 5):
        cues = (1, 2) if stage == 1 else (0,) if stage == 2 else ()
        for cue in cues:
            commands += primitive_blocks[cue]
            teaching += [{'type': 'P', 'id': r.split()[1], 'record': r} for r in primitive_blocks[cue]]
        if stage == 1: language(0); effect(0)
        elif stage == 2: language(1); language(2)
        elif stage == 3: concept('eligible'); language(3); effect(1)
        else: concept('eligible2'); language(4); effect(2)
        families = [0] if stage == 1 else list(range(stage+1))
        for _ in range(episodes):
            family = rng.choice(families); physical = family in (0, 3, 4) and bool(rng.randrange(2))
            case(stage, family, physical, True)
        commands.append(f'CHECK stage{stage}')
        for family in families:
            for physical in ([False, True] if family in (0, 3, 4) else [False]):
                for price in range(2):
                    for view in range(4):
                        for _ in range(2):
                            c = case(stage, family, physical, False, view, price)
                            if stage == 4: final.append(c)
        commands.append('ENDCHECK')
    commands.append('CHECK removals')
    for disabled in range(5):
        for c in final: case(4, 0, False, False, template=c, disabled=disabled)
    commands.append('ENDCHECK')
    commands.append('CHECK recovery')
    for c in final:
        if c['effect'] and not c['gold']:
            c = copy.deepcopy(c); c['failures'] = 3
            case(4, 0, False, False, template=c)
    commands.append('ENDCHECK')
    commands.append('CHECK longer')
    for family in (0, 3, 4):
        for size in (8, 12):
            for price in range(2):
                case(4, family, True, False, 0, price, size=size, missing=3)
    commands.append('ENDCHECK')
    directory.mkdir(parents=True, exist_ok=True)
    for name, lines in [('input', commands), ('worlds', worlds), ('environment', private)]:
        (directory/f'{name}.txt').write_text('\n'.join(lines)+'\n')
    (directory/'oracle.json').write_text(json.dumps({'computational': True, 'reactive': True, 'seed': seed,
        'definitions': definitions, 'effects': effects, 'cases': cases, 'teaching': teaching,
        'episodes_per_stage': episodes, 'stages': 4})+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--episodes', type=int, default=400); p.add_argument('--out', type=Path, default=Path('logs/computational_control'))
    args = p.parse_args()
    for seed in range(args.seeds): build(seed, args.out/str(seed), args.episodes)
    print(f'created {args.seeds} mixed curricula with {args.episodes*4} training episodes each')
