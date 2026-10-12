"""Curriculum and private environments for learned control of the joint model."""
import argparse
import copy
import json
import shlex
from pathlib import Path
from relational_session_probe import JointCurriculum, observed, action_text, clause_text, names
from check_concept_growth import intended, evaluate


def build(seed, destination, episodes, reactive=False):
    course = JointCurriculum(seed); base = destination/'teaching'; course.build(base)
    raw = (base/'input.txt').read_text().splitlines(); teacher = json.loads((base/'oracle.json').read_text())
    blocks = []; pending = []
    for line in raw:
        fields = shlex.split(line)
        if fields[0] == 'R' and fields[1].startswith('stage'):
            blocks.append(pending); pending = []
        elif fields[0] in ('C', 'L', 'T'):
            if fields[1] == 'conflict': break
            pending.append(line)
    assert len(blocks) == 8
    commands, worlds, cases = [], [], []
    private = [' '.join(map(str, ['E', n, e['predicate'], e['subject'], e['object'], e['kind']])) for n, e in course.effects.items()]

    def case(stage, family, effect, price, visible_mode, memory_mode, missing, training, disabled=-1, template=None):
        cid = f'case{len(cases)}'
        if template is None:
            variant = 0
            if not effect: variant = course.rng.randrange(2)
            elif family > 0 and stage >= 6 and course.rng.random() < .2: variant = 1; missing = 0
            truth, actions = course.physical(family, stage, missing)
            text = course.surface(family, truth, variant)
            initial = copy.deepcopy(truth)
            if visible_mode == 1: initial['facts'] = []; initial['complete'] = False
            elif visible_mode == 2:
                initial['facts'] = [f for f in initial['facts'] if course.rng.random() < .4]; initial['complete'] = False
            elif visible_mode == 3: initial['complete'] = False
            inventory = sorted(set(names(truth)) | {v for a in actions for v in a['args']})
            initial['args'] = inventory
            memory = None
            if memory_mode:
                memory = copy.deepcopy(truth)
                if memory_mode == 2:
                    memory['facts'] = memory['facts'][::2]; memory['complete'] = False
            revision = 0 if memory_mode == 3 else 1
            r = {'family': family, 'stage': stage, 'effect': bool(effect), 'price': price,
                 'initial': initial, 'truth': truth, 'available': actions, 'text': text,
                 'variant': variant, 'memory': memory, 'memory_revision': revision,
                 'visible_mode': visible_mode, 'memory_mode': memory_mode, 'missing': missing}
        else: r = copy.deepcopy(template)
        if reactive and template is None:
            r['failures'] = int(bool(effect) and training and course.rng.random() < .25)
            r['sensor_unavailable'] = course.rng.random() < .25 if training else (visible_mode == 1 and family % 2 == 0)
            if r['sensor_unavailable']:
                r['memory'] = copy.deepcopy(r['truth']); r['memory_revision'] = 1; r['memory_mode'] = 1
        r.update(id=cid, training=training, disabled=disabled); cases.append(r)
        commands.append(f'{"TRAIN" if training else "TEST"} {cid} {int(r["effect"])} {r["price"]} {disabled} '+
                        observed(r['initial'])+' '+json.dumps(r['text'])+f' {len(r["available"])} '+' '.join(action_text(a) for a in r['available']))
        worlds.append(f'W {cid} {observed(r["truth"])} {int(r["memory"] is not None)} {r["memory_revision"]} '+observed(r['memory'] or {'args': [], 'facts': [], 'complete': False}))
        if reactive: worlds.append(f'X {cid} {r["failures"]} {int(r["sensor_unavailable"])}')
        private.append(' '.join(map(str, ['G', cid, len(r['truth']['args']), *r['truth']['args'], r['variant']]))+' '+clause_text(intended(teacher, r['family'])))
        return r

    final = []
    for stage, block in enumerate(blocks, 1):
        commands += block
        for _ in range(episodes):
            family = course.rng.randrange(stage); effect = course.rng.randrange(2); price = course.rng.randrange(2)
            case(stage, family, effect, price, course.rng.randrange(4), course.rng.randrange(4), course.rng.randrange(3), True)
        commands.append(f'CHECK stage{stage}')
        for family in range(stage):
            for effect in range(2):
                for price in range(2):
                    for mode in range(4):
                        record = case(stage, family, effect, price, mode, (0, 1, 0, 3)[mode], mode, False)
                        if stage == 8: final.append(record)
        commands.append('ENDCHECK')
    commands.append('CHECK removals')
    for disabled in range(5):
        for original in final:
            case(8, 0, 0, 0, 0, 0, 0, False, disabled, original)
    commands.append('ENDCHECK')
    if reactive:
        commands.append('CHECK recovery')
        for original in final:
            if not original['effect'] or (evaluate(intended(teacher, original['family']), original['truth']) ^ original['variant']): continue
            interrupted = copy.deepcopy(original); interrupted['failures'] = 3; interrupted['sensor_unavailable'] = False
            case(8, 0, 0, 0, 0, 0, 0, False, template=interrupted)
        commands.append('ENDCHECK')
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'input.txt').write_text('\n'.join(commands)+'\n')
    (destination/'environment.txt').write_text('\n'.join(private)+'\n')
    (destination/'worlds.txt').write_text('\n'.join(worlds)+'\n')
    meta = {k: v for k, v in teacher.items() if k != 'records'}
    meta.update(cases=cases, teaching=[r for r in teacher['records'] if r['kind'] in ('C', 'L', 'T') and r['id'] != 'conflict'], episodes_per_stage=episodes, reactive=reactive)
    (destination/'oracle.json').write_text(json.dumps(meta))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--seeds', type=int, default=3);p.add_argument('--episodes', type=int, default=200);p.add_argument('--out', default='logs/relational_control')
    p.add_argument('--reactive', action='store_true'); args = p.parse_args()
    for seed in range(args.seeds): build(seed, Path(args.out)/str(seed), args.episodes, args.reactive)
    print(f'created {args.seeds} curricula with {args.episodes*8} control episodes each')
