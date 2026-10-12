"""Reconstruct learned (possibly wrong) answers separately from world truth."""
import csv
import json
import shlex
import subprocess
from pathlib import Path
from check_composed_credit import execute
from check_concept_growth import evaluate, require
from check_relational_session import parse_snapshot
from computational_control_oracle import binding


class LearnedComputations:
    def __init__(self, directory, mode, seed, shape):
        self.directory, self.mode, self.seed, self.shape = directory, mode, seed, shape
        config = directory/f'{mode}.config.json'
        self.observed_relation_types = json.loads(config.read_text()).get('observed_relation_types', False) if config.exists() else False
        with (directory/f'{mode}.models.tsv').open() as f:
            self.records = {(r['episode'], int(r['decision'])): r for r in csv.DictReader(f, delimiter='\t')}
        self.used, self.weights = set(), {}

    def rebuilt(self, episode, decision, version):
        key = (episode, decision); require(key not in self.used, 'model readout reused for another rebuild')
        self.used.add(key); record = self.records[key]
        require(int(record['version']) == version, 'model readout has stale evidence version')
        path = self.directory/f'{self.mode}.model{record["model"]}.txt'
        rows = [shlex.split(r) for r in path.read_text().splitlines()]
        snapshot, at = parse_snapshot(rows, 0)
        require(snapshot['version'] == version and snapshot['ready'], 'unreconstructed model readout')
        snapshot['schemas'] = set()
        snapshot['observed_relation_types'] = self.observed_relation_types
        for r in rows[at:]:
            require(len(r) == 2 and r[0] == 'SCHEMA', 'invalid model schema row')
            snapshot['schemas'].add(r[1])
        wid = record['weights']
        if wid not in self.weights:
            weights = self.directory/f'{self.mode}.model_weights{wid}'
            command = ['build-wsl/verify_local_choices_readout', str(self.seed), str(weights), *map(str, self.shape), '--export']
            output = subprocess.check_output(command, text=True)
            choices = dict(tuple(map(int, line.split())) for line in output.splitlines())
            require(len(choices) == 3*64*3*11, 'incomplete frozen neural readout')
            self.weights[wid] = choices
        return snapshot, self.weights[wid]

    @staticmethod
    def answer(model, text, world):
        if model is None or not world['complete']: return -1
        snapshot, choices = model
        matched = [(key, frame, binding(key, text)) for key, frame in snapshot['computations'].items() if binding(key, text) is not None]
        if len(matched) != 1: return -1
        key, frame, (args, relations) = matched[0]
        if not frame['learned']: return -1
        types = [2 if p.startswith('#') else 0 for p in key.removeprefix('computed:').split() if p.startswith(('$', '#'))]
        names = sorted({x for _, vs in world['facts'] for x in vs} | {x for x, t in zip(args, types) if t == 0})
        if len(names) > 32: return -1
        entities = {name: i for i, name in enumerate(names)}
        rels = {name: i for i, name in enumerate(dict.fromkeys(relations))}
        facts = []
        for name in sorted(rels):
            relation = rels[name]
            if name in snapshot['concepts']:
                concept = snapshot['concepts'][name]
                if concept['arity'] != 2: return -1
                for a in names:
                    for b in names:
                        votes = [evaluate(d, dict(world, args=[a, b])) for d in concept['expanded']]
                        if not votes or len(set(votes)) != 1 or votes[0] < 0: return -1
                        if votes[0] == 1: facts.append((entities[a], relation, entities[b]))
            else:
                if name not in snapshot['schemas'] and not (snapshot.get('observed_relation_types') and
                        any(pred == name and len(vs) == 2 for pred, vs in world['facts'])): return -1
                for pred, vs in world['facts']:
                    if pred == name:
                        require(len(vs) == 2, 'nonbinary fact uses a binary schema')
                        facts.append((entities[vs[0]], relation, entities[vs[1]]))
        values = [(t, int(a) if t == 2 else entities[a]) for t, a in zip(types, args)]
        query = {'a': values[0], 'b': values[1], 'relations': [rels[r] for r in relations], 'facts': facts}
        available = snapshot['primitives'][1:]
        def choose(state, actions):
            task = next(t for t in range(3) if state & (1 << t))
            return choices[state] if available[task] else -1
        value, _ = execute(query, frame['program'], choose)
        return value[1] if value[0] == 1 else -1

    def finish(self):
        require(self.used == set(self.records), 'unused or omitted model rebuilds')
