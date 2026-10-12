"""Evaluator-only curriculum; definitions and query answers never enter native inference.

Sparse positive witnesses, deletion/substitution contrasts, distractors and random
argument aliases. This is a deliberately structured concept-learning curriculum,
not the user's public language dataset or a claim about spontaneous discovery.
"""
import argparse
import json
import random
from pathlib import Path

ARITY = [2, 2, 2, 3, 3, 4, 2, 3]
PATTERNS = [
    [('loc', (0, 2)), ('loc', (1, 2))],
    [('holds', (0, 2)), ('needs', (1, 2))],
    [('loc', (0, 2)), ('loc', (1, 2)), ('holds', (0, 3)), ('needs', (1, 3))],
    [('loc', (0, 3)), ('loc', (1, 3)), ('holds', (0, 2)), ('needs', (1, 2))],
    [('loc', (0, 3)), ('loc', (1, 3)), ('holds', (0, 4)), ('needs', (1, 4)),
     ('loc', (1, 5)), ('loc', (2, 5)), ('holds', (1, 6)), ('needs', (2, 6))],
    [('loc', (0, 4)), ('loc', (1, 4)), ('holds', (0, 3)), ('needs', (1, 3)),
     ('loc', (1, 5)), ('loc', (2, 5)), ('holds', (1, 3)), ('needs', (2, 3))],
    [('loc', (0, 2)), ('loc', (1, 2)), ('holds', (0, 3)), ('needs', (1, 3)), ('marker', (1,))],
    [('loc', (0, 3)), ('loc', (1, 3)), ('holds', (0, 2)), ('needs', (1, 2)), ('permit', (0, 1, 2))],
]
BASE_ARITY = {'loc': 2, 'holds': 2, 'needs': 2, 'noise': 2, 'marker': 1, 'permit': 3}


def truth(family, facts, args):
    """Independent direct relational oracle, not execution of PATTERNS."""
    rows = {}
    for name, values in facts:
        rows.setdefault(name, set()).add(tuple(values))
    def objects(name, subject):
        return {b for a, b in rows.get(name, ()) if a == subject}
    def present(a, b):
        return bool(objects('loc', a) & objects('loc', b))
    def matched(a, b):
        return bool(objects('holds', a) & objects('needs', b))
    def eligible(a, b):
        return present(a, b) and matched(a, b)
    def handoff(a, b, x):
        return present(a, b) and (a, x) in rows.get('holds', ()) and (b, x) in rows.get('needs', ())
    if family == 0: return present(*args)
    if family == 1: return matched(*args)
    if family == 2: return eligible(*args)
    if family == 3: return handoff(*args)
    if family == 4:
        a, b, c = args
        return eligible(a, b) and eligible(b, c)
    if family == 5:
        a, b, c, x = args
        return handoff(a, b, x) and handoff(b, c, x)
    if family == 6: return eligible(*args) and (args[1],) in rows.get('marker', ())
    if family == 7: return handoff(*args) and tuple(args) in rows.get('permit', ())
    raise ValueError(family)


def encode_world(world):
    args, facts, complete = world['args'], world['facts'], world.get('complete', True)
    tokens = [int(complete), len(args), *args, len(facts)]
    for pred, values in facts:
        tokens += [pred, len(values), *values]
    return ' '.join(map(str, tokens))


class Curriculum:
    def __init__(self, seed):
        self.rng = random.Random(48711 + seed)
        words = list(range(100, 150)); self.rng.shuffle(words)
        self.names = {p: f'p{words.pop()}' for p in BASE_ARITY}
        self.heads = [f'c{words.pop()}' for _ in ARITY]
        self.permutations = {}
        for p, n in BASE_ARITY.items():
            v = list(range(n)); self.rng.shuffle(v); self.permutations[p] = v
        self.roles = []
        for n in ARITY:
            v = list(range(n)); self.rng.shuffle(v); self.roles.append(v)
        self.serial = 0

    def encode(self, family, args, facts, anchor=False):
        encoded = [(self.names[p], [values[i] for i in self.permutations[p]]) for p, values in sorted(facts)]
        if not anchor and encoded and self.rng.random() < .3: encoded.append(self.rng.choice(encoded))
        self.rng.shuffle(encoded)
        return {'args': [args[i] for i in self.roles[family]], 'facts': encoded, 'complete': True}

    def contrasts(self, family):
        # Generic role-preserving decoys: an alternate tuple satisfies the whole
        # relation while the queried tuple lacks one required fact. This catches
        # hypotheses that existentially forget a head role inside a learned macro.
        for dropped in range(len(PATTERNS[family])):
            for role in range(ARITY[family]):
                self.serial += 1
                binding = [f'e{self.serial}_contrast_{i}' for i in range(8)]
                original = [(p, tuple(binding[v] for v in vs)) for p, vs in PATTERNS[family]]
                alternate = list(binding); alternate[role] = f'e{self.serial}_alternate'
                facts = set(original[:dropped]+original[dropped+1:])
                facts.update((p, tuple(alternate[v] for v in vs)) for p, vs in PATTERNS[family])
                args = binding[:ARITY[family]]
                if not truth(family, facts, args): yield self.encode(family, args, facts)

    def world(self, family, positive, stage, anchor=False):
        self.serial += 1
        for attempt in range(1000):
            entities = [f'e{self.serial}_{attempt}_{i}' for i in range(18)]
            self.rng.shuffle(entities)
            binding = entities[:8]
            if not anchor and self.rng.random() < .2:
                a, b = self.rng.sample(range(ARITY[family]), 2); binding[b] = binding[a]
            facts = {(p, tuple(binding[v] for v in vs)) for p, vs in PATTERNS[family]}
            if not anchor:
                available = ['loc', 'holds', 'needs', 'noise']
                if stage >= 7: available.append('marker')
                if stage >= 8: available.append('permit')
                for _ in range(self.rng.randrange(3, 10)):
                    p = self.rng.choice(available)
                    facts.add((p, tuple(self.rng.choice(entities[:12]) for _ in range(BASE_ARITY[p]))))
            args = binding[:ARITY[family]]
            if not positive:
                if self.rng.random() < .7:
                    alternate = list(binding)
                    role = self.rng.randrange(ARITY[family]); alternate[role] = self.rng.choice(entities[12:])
                    facts.update((p, tuple(alternate[v] for v in vs)) for p, vs in PATTERNS[family])
                # Both missing edges and separated shared witnesses are needed:
                # mere degree/existence rules must receive counterexamples.
                for _ in range(12):
                    if not truth(family, facts, args): break
                    p, vs = self.rng.choice(PATTERNS[family]); edge = (p, tuple(binding[v] for v in vs))
                    if edge not in facts: continue
                    facts.remove(edge)
                    if self.rng.random() < .65:
                        changed = list(edge[1]); changed[self.rng.randrange(len(changed))] = self.rng.choice(entities[12:])
                        facts.add((p, tuple(changed)))
            if truth(family, facts, args) != positive: continue
            return self.encode(family, args, facts, anchor)
        raise RuntimeError('failed to generate balanced case')

    def build(self, path):
        commands, cases = [], []
        teaching = []
        def query(group, family, world, gold):
            qid = f'q{len(cases)}'
            commands.append(f'Q {qid} {self.heads[family]} {encode_world(world)}')
            cases.append({'id': qid, 'group': group, 'family': family, 'world': world, 'gold': gold})
        for family in range(8):
            stage = family + 1
            training = [(self.world(family, i % 2 == 0, stage, anchor=i == 0), i % 2 == 0) for i in range(48)]
            training += [(world, False) for world in self.contrasts(family)]
            for i, (world, positive) in enumerate(training):
                rid = f't{family}_{i:03}'
                commands.append(f'T {rid} {self.heads[family]} {int(positive)} {encode_world(world)}')
                teaching.append({'id': rid, 'family': family, 'world': world, 'gold': int(positive)})
            commands.append(f'R stage{stage}')
            for previous in range(stage):
                for i in range(40):
                    query(f'stage{stage}', previous, self.world(previous, i % 2 == 0, stage), i % 2 == 0)
        commands.append('R final')
        final = []
        for family in range(8):
            for i in range(200):
                world = self.world(family, i % 2 == 0, 8)
                final.append((family, world, i % 2 == 0))
                query('final', family, world, i % 2 == 0)
        first = teaching[0]
        commands.append(f'T conflict {self.heads[0]} 0 {encode_world(first["world"])}')
        for family in range(8): query('dirty_conflict', family, final[family * 200][1], -1)
        commands.append('R conflict')
        for family in range(8):
            for i in range(32): query('conflict', family, self.world(family, i % 2 == 0, 8), i % 2 == 0)
        commands.append('V conflict 1 1 1')
        for family in range(8): query('dirty_recovery', family, final[family * 200][1], -1)
        commands.append('R recovered')
        for family, world, gold in final: query('recovered', family, world, gold)
        for family, world, gold in final[::5]:
            partial = dict(world, complete=False)
            query('partial', family, partial, 1 if gold else -1)
        path.mkdir(parents=True, exist_ok=True)
        (path/'input.txt').write_text('\n'.join(commands)+'\n')
        meta = {'base': self.names, 'base_roles': self.permutations, 'heads': self.heads, 'head_roles': self.roles,
                'teaching': teaching, 'cases': cases}
        (path/'oracle.json').write_text(json.dumps(meta))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', default='logs/concept_growth'); parser.add_argument('--seeds', type=int, default=3)
    opts = parser.parse_args()
    for seed in range(opts.seeds): Curriculum(seed).build(Path(opts.out)/str(seed))
    print(f'created {opts.seeds} opaque-vocabulary concept curricula')
