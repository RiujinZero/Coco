"""Interleaved evaluator-owned curriculum for the continuing relational session."""
import argparse
import copy
import json
import random
from pathlib import Path
from concept_growth_probe import Curriculum, ARITY, PATTERNS, truth, encode_world
from check_concept_growth import intended


def names(world):
    return sorted(set(world['args']) | {v for _, args in world['facts'] for v in args})


def observed(world):
    return encode_world(dict(world, args=names(world)))


def action_text(action):
    return ' '.join(map(str, [action['schema'], len(action['args']), *action['args']]))


def clause_text(clause):
    arity, body = clause; tokens = [arity, len(body)]
    for pred, variables in body: tokens += [pred, len(variables), *variables]
    return ' '.join(map(str, tokens))


def apply_effect(world, action, effects):
    result = copy.deepcopy(world); e = effects[action['schema']]
    a, b = action['args'][e['subject']], action['args'][e['object']]
    kept = []
    for pred, values in world['facts']:
        remove = pred == e['predicate'] and ((e['kind'] == 1 and values == [a, b]) or
                    (e['kind'] == 2 and values[0] == a) or (e['kind'] == 3 and values[1] == b))
        if not remove: kept.append((pred, tuple(values)))
    if e['kind'] != 1: kept.append((e['predicate'], (a, b)))
    result['facts'] = [[p, list(vs)] for p, vs in sorted(set(kept))]
    return result


class JointCurriculum(Curriculum):
    def __init__(self, seed):
        super().__init__(seed)
        self.seed = seed; self.surface_roles = []
        self.verbs = [[f'w{seed}x{family}x{variant}' for variant in range(2)] for family in range(8)]
        for n in ARITY:
            variants = []
            for _ in range(2):
                roles = list(range(n)); self.rng.shuffle(roles); variants.append(roles)
            self.surface_roles.append(variants)
        self.effect_names = [f'm{seed}x{i}' for i in self.rng.sample(range(30), 4)]
        self.motor_roles = []; self.effects = {}
        for i, pred in enumerate(('loc', 'holds', 'needs', 'holds')):
            permutation = [0, 1]; self.rng.shuffle(permutation); self.motor_roles.append(permutation)
            subject = permutation.index(self.permutations[pred][0]); obj = permutation.index(self.permutations[pred][1])
            self.effects[self.effect_names[i]] = {'predicate': self.names[pred], 'subject': subject, 'object': obj, 'kind': 1 if i == 3 else 0}
        self.commands = []; self.records = []; self.private = []; self.cid = 0

    def surface(self, family, world, variant=0):
        args = [world['args'][i] for i in self.surface_roles[family][variant]]
        tokens = ['please', args[0], self.verbs[family][variant]]
        for arg in args[1:]: tokens += [arg, 'with']
        return ' '.join(tokens[:-1])

    def base_meta(self):
        return {'base': self.names, 'base_roles': self.permutations, 'heads': self.heads, 'head_roles': self.roles}

    def decode(self, family, world):
        args = [None]*ARITY[family]
        for old, value in zip(self.roles[family], world['args']): args[old] = value
        inverse = {v: k for k, v in self.names.items()}; facts = []
        for p, values in world['facts']:
            p = inverse[p]; decoded = [None]*len(values)
            for old, value in zip(self.permutations[p], values): decoded[old] = value
            facts.append((p, tuple(decoded)))
        return args, facts

    def emit(self, command, record=None):
        self.commands.append(command)
        if record is not None: self.records.append(record)

    def question(self, group, family, world, variant=0, expected=None):
        qid = f'q{self.cid}'; self.cid += 1; text = self.surface(family, world, variant)
        args, facts = self.decode(family, world)
        base = int(truth(family, facts, args)); gold = base ^ variant
        if not world.get('complete', True) and base == 0: gold = -1
        if expected is not None: gold = expected
        self.emit(f'Q {qid} {observed(world)} {json.dumps(text)}',
                  {'kind': 'Q', 'id': qid, 'group': group, 'family': family, 'variant': variant, 'world': world, 'text': text, 'gold': gold})

    def physical(self, family, stage, missing=1, bound=False):
        world = self.world(family, True, stage, anchor=True)
        args, facts = self.decode(family, world)
        persons = sorted({vs[0] for p, vs in facts if p in ('loc', 'holds', 'needs')})
        rooms = sorted({vs[1] for p, vs in facts if p == 'loc'})
        items = sorted({vs[1] for p, vs in facts if p in ('holds', 'needs')})
        prefix = f'extra{self.serial}'
        persons += [prefix+'p']; rooms += [prefix+'r', prefix+'s']; items += [prefix+'i']
        indices = [i for i, (p, vs) in enumerate(world['facts']) if p in ({self.names['holds'], self.names['needs']} if bound else
                    {self.names[p] for p in ('loc', 'holds', 'needs')[:min(stage, 3)]})]
        self.rng.shuffle(indices)
        for i in sorted(indices[:missing], reverse=True): world['facts'].pop(i)
        available = []
        for motor in range(4 if stage >= 6 else min(stage, 3)):
            objects = rooms if motor == 0 else items
            for person in persons:
                for obj in objects:
                    canonical = [person, obj]
                    available.append({'schema': self.effect_names[motor], 'args': [canonical[i] for i in self.motor_roles[motor]]})
        self.rng.shuffle(available)
        return world, available

    def goal(self, group, family, world, available, variant=0, budget=32768, explore=False):
        gid = f'g{self.cid}'; self.cid += 1; text = self.surface(family, world, variant)
        self.emit(f'{"A" if explore else "G"} {gid} {budget} 4 {observed(world)} {json.dumps(text)} {len(available)} '+' '.join(action_text(a) for a in available),
                  {'kind': 'G', 'id': gid, 'group': group, 'family': family, 'variant': variant, 'world': world, 'text': text, 'available': available, 'budget': budget, 'explore': explore})
        clause = intended(self.base_meta(), family)
        # The evaluator receives the intended meaning separately. It cannot be
        # queried by the learner except for the terminal satisfaction bit.
        self.private.append(' '.join(map(str, ['G', gid, len(world['args']), *world['args'], variant]))+' '+clause_text(clause))
        return gid

    def renamed(self, world, available):
        self.serial += 1
        values = sorted(set(names(world)) | {v for a in available for v in a['args']})
        mapping = {v: f'transfer{self.serial}x{i}' for i, v in enumerate(values)}
        w = {'args': [mapping[v] for v in world['args']], 'facts': [[p, [mapping[v] for v in vs]] for p, vs in world['facts']], 'complete': True}
        actions = [{'schema': a['schema'], 'args': [mapping[v] for v in a['args']]} for a in available]
        self.rng.shuffle(actions); return w, actions

    def build(self, path):
        self.private = [' '.join(map(str, ['E', name, e['predicate'], e['subject'], e['object'], e['kind']])) for name, e in self.effects.items()]
        for family in range(8):
            stage = family+1
            training = [(self.world(family, i % 2 == 0, stage, anchor=i == 0), i % 2 == 0) for i in range(48)]
            training += [(w, False) for w in self.contrasts(family)]
            for i, (world, value) in enumerate(training):
                rid = f'c{family}x{i}'
                self.emit(f'C {rid} {self.heads[family]} {int(value)} {encode_world(world)}',
                          {'kind': 'C', 'id': rid, 'family': family, 'world': world, 'gold': int(value)})
                for variant in range(2):
                    lid = f'l{family}x{i}x{variant}'; text = self.surface(family, world, variant)
                    self.emit(f'L {lid} {int(value)^variant} {observed(world)} {json.dumps(text)}',
                              {'kind': 'L', 'id': lid, 'family': family, 'variant': variant, 'world': world, 'text': text, 'gold': int(value)^variant})
            # Independently observed effects arrive during concept/language growth.
            motor = {1: 0, 2: 1, 3: 2, 6: 3}.get(stage)
            if motor is not None:
                pred = ('loc', 'holds', 'needs', 'holds')[motor]
                for i in range(4):
                    a, b, c, d = [f'motor{stage}x{i}x{j}' for j in range(4)]
                    raw = [(pred, (a, d)), (pred, (c, b))]
                    if motor == 3: raw.append((pred, (a, b)))
                    before = self.encode(0, [a, b], raw, anchor=True)
                    action = {'schema': self.effect_names[motor], 'args': [[a, b][j] for j in self.motor_roles[motor]]}
                    after = apply_effect(before, action, self.effects); rid = f't{motor}x{i}'
                    self.emit(f'T {rid} {action_text(action)} {observed(before)} {observed(after)}',
                              {'kind': 'T', 'id': rid, 'action': action, 'before': before, 'after': after})
            self.emit(f'R stage{stage}')
            for previous in range(stage):
                for i in range(20):
                    self.question(f'stage{stage}', previous, self.world(previous, i % 2 == 0, stage), (i//2)%2)
                for missing in (1, 2):
                    w, actions = self.physical(previous, stage, missing)
                    self.goal('interleaved', previous, w, actions)
            # Exploration returns observed effects and one satisfaction bit;
            # both effect and language banks learn from this same receipt.
            w, actions = self.physical(family, stage, 1)
            self.goal('exploration', family, w, actions, explore=True)
        self.emit('R final')
        final_questions = []
        for family in range(8):
            for i in range(96):
                w = self.world(family, i % 2 == 0, 8); variant = (i//2)%2
                final_questions.append((family, w, variant)); self.question('final', family, w, variant)
            for i in range(6):
                w, actions = self.physical(family, 8, 1+i%3)
                self.goal('final_goals', family, w, actions)
        # Learn a bound two-step procedure from own search; then test fresh names
        # with too little work for breadth-first search to reach a second layer.
        for family in (3, 7):
            w, actions = self.physical(family, 8, 2, bound=True)
            self.goal('procedure_acquisition', family, w, actions)
            for _ in range(12):
                renamed, menu = self.renamed(w, actions)
                self.goal('limited_procedure', family, renamed, menu, budget=16)
        for family in (1, 2, 3, 4, 5, 6, 7):
            w, actions = self.physical(family, 8, 0)
            self.goal('negative_goals', family, w, actions, variant=1)
        first = next(r for r in self.records if r['kind'] == 'C')
        self.emit(f'C conflict {self.heads[0]} 0 {encode_world(first["world"])}',
                  {'kind': 'C', 'id': 'conflict', 'family': 0, 'world': first['world'], 'gold': 0, 'deliberate_conflict': True})
        self.question('dirty', 0, first['world'], expected=-1);self.emit('R conflict')
        for family in range(8):
            w = self.world(family, True, 8);self.question('conflict', family, w, expected=-1)
            w, actions = self.physical(family, 8, 1);self.goal('conflict_goals', family, w, actions)
        self.emit('VC conflict 1 1');self.emit('R recovered')
        for family, w, variant in final_questions: self.question('recovered', family, w, variant)
        for family, w, variant in final_questions[::5]:
            partial = dict(w, complete=False);self.question('partial', family, partial, variant)
        path.mkdir(parents=True, exist_ok=True)
        (path/'input.txt').write_text('\n'.join(self.commands)+'\n')
        (path/'environment.txt').write_text('\n'.join(self.private)+'\n')
        meta = self.base_meta() | {'records': self.records, 'effects': self.effects, 'surface_roles': self.surface_roles, 'verbs': self.verbs}
        (path/'oracle.json').write_text(json.dumps(meta))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--seeds', type=int, default=3);parser.add_argument('--out', default='logs/relational_session')
    args = parser.parse_args()
    for seed in range(args.seeds): JointCurriculum(seed).build(Path(args.out)/str(seed))
    print(f'created {args.seeds} interleaved relational-session curricula')
