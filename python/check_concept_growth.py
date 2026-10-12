"""Independent source-evidence, role semantics, definition expansion and retention audit."""
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
from concept_growth_probe import truth, PATTERNS, ARITY


def require(condition, message):
    if not condition: raise AssertionError(message)


def parse_world(tokens, at):
    complete, n = int(tokens[at]), int(tokens[at+1]); at += 2
    args = tokens[at:at+n]; at += n
    nf = int(tokens[at]); at += 1; facts = []
    for _ in range(nf):
        p, n = tokens[at], int(tokens[at+1]); at += 2
        facts.append((p, tokens[at:at+n])); at += n
    return {'complete': bool(complete), 'args': args, 'facts': facts}, at


def parse_clause(tokens):
    arity, n = map(int, tokens[:2]); at = 2; body = []
    for _ in range(n):
        p, k = tokens[at], int(tokens[at+1]); at += 2
        body.append((p, tuple(map(int, tokens[at:at+k])))); at += k
    require(at == len(tokens), 'trailing clause fields')
    require(all(v >= 0 for _, vs in body for v in vs), 'negative variable')
    require(set(range(arity)) <= {v for _, vs in body for v in vs}, 'unbound head')
    return arity, body


def normalize(clause):
    arity, body = clause
    existential = sorted({v for _, vs in body for v in vs if v >= arity})
    best = None
    for permutation in itertools.permutations(range(arity, arity+len(existential))):
        mapping = dict(zip(existential, permutation))
        terms = tuple(sorted({(p, tuple(mapping.get(v, v) for v in vs)) for p, vs in body}))
        if best is None or terms < best: best = terms
    return arity, best


def expand(local, stage, order, name):
    arity, body = local
    fresh = max([arity-1]+[v for _, vs in body for v in vs])+1
    expanded = []
    for p, args in body:
        if p not in stage:
            expanded.append((p, args)); continue
        require(order.index(p) < order.index(name), 'forward/cyclic learned predicate')
        definitions = stage[p]['expanded']
        require(len(definitions) == 1, 'ambiguous dependency reused')
        dep_arity, dep = next(iter(definitions.values()))
        require(dep_arity == len(args), 'dependency arity mismatch')
        mapping = dict(enumerate(args))
        for v in sorted({v for _, vs in dep for v in vs if v >= dep_arity}):
            mapping[v] = fresh; fresh += 1
        expanded += [(q, tuple(mapping[v] for v in vs)) for q, vs in dep]
    return arity, expanded


def evaluate(clause, world):
    if isinstance(clause, dict):
        require(set(clause) == {'any'} and clause['any'], 'invalid rule union')
        votes = [evaluate(branch, world) for branch in clause['any']]
        return 1 if 1 in votes else -1 if -1 in votes else 0
    arity, body = clause
    require(arity == len(world['args']), 'inference arity mismatch')
    rows = collections.defaultdict(set)
    for pred, values in world['facts']: rows[pred].add(tuple(values))
    states = [dict(enumerate(world['args']))]
    for pred, variables in body:
        next_states = []
        for state in states:
            for values in rows[pred]:
                require(len(values) == len(variables), 'fact/schema arity mismatch')
                bound = dict(state)
                for variable, value in zip(variables, values):
                    if variable in bound and bound[variable] != value: break
                    bound[variable] = value
                else: next_states.append(bound)
        states = next_states
        if not states: return 0 if world['complete'] else -1
    return 1


def equivalent(a, b):
    if isinstance(a, dict) or isinstance(b, dict):
        aa = a['any'] if isinstance(a, dict) else [a]
        bb = b['any'] if isinstance(b, dict) else [b]
        def covered(stronger, weaker):
            n, body = stronger
            database = {'args': [f'v{i}' for i in range(n)], 'complete': True,
                        'facts': [(p, [f'v{v}' for v in vs]) for p, vs in body]}
            return any(n == branch[0] and evaluate(branch, database) == 1 for branch in weaker)
        return all(covered(c, bb) for c in aa) and all(covered(c, aa) for c in bb)
    def implies(stronger, weaker):
        n, body = stronger
        database = {'args': [f'v{i}' for i in range(n)], 'complete': True,
                    'facts': [(p, [f'v{v}' for v in vs]) for p, vs in body]}
        return evaluate(weaker, database) == 1
    return a[0] == b[0] and implies(a, b) and implies(b, a)


def intended(meta, family):
    n = ARITY[family]
    mapping = {old: new for new, old in enumerate(meta['head_roles'][family])}
    body = []
    for p, vs in PATTERNS[family]:
        args = [mapping.get(v, v) for v in vs]
        body.append((meta['base'][p], tuple(args[i] for i in meta['base_roles'][p])))
    return n, body


def oracle(meta, family, world):
    inverse = {v: k for k, v in meta['base'].items()}
    facts = []
    for opaque, values in world['facts']:
        name = inverse[opaque]; unpermuted = [None]*len(values)
        for old, value in zip(meta['base_roles'][name], values): unpermuted[old] = value
        facts.append((name, tuple(unpermuted)))
    args = [None]*len(world['args'])
    for old, value in zip(meta['head_roles'][family], world['args']): args[old] = value
    return int(truth(family, facts, args))


def check_run(directory, mode):
    meta = json.loads((directory/'oracle.json').read_text())
    commands = [line.split() for line in (directory/'input.txt').read_text().splitlines()]
    output = [line.split() for line in (directory/f'{mode}.txt').read_text().splitlines()]
    cases = {q['id']: q for q in meta['cases']}
    lessons = {q['id']: q for q in meta['teaching']}
    by_stage, predictions = {}, {}; receipts = {}; active = {}; stage = 'none'; dirty = False; out_at = 0
    metrics = collections.defaultdict(lambda: collections.Counter(total=0, correct=0, wrong=0, withheld=0))
    state_checks = 0
    for cmd in commands:
        kind = cmd[0]; row = output[out_at]; out_at += 1
        require(row[0] == {'Q': 'A'}.get(kind, kind), f'output misaligned: {cmd[:2]} {row}')
        if kind == 'T':
            rid, name, label = cmd[1], cmd[2], int(cmd[3]); world, at = parse_world(cmd, 4)
            require(at == len(cmd) and row == ['T', rid, '1'], 'teaching rejected/malformed')
            if rid in lessons:
                lesson = lessons[rid]
                require(world == dict(lesson['world'], facts=[tuple(a) for a in lesson['world']['facts']]), 'oracle/teaching world differs')
                require(label == oracle(meta, lesson['family'], world), 'incorrect generated teaching label')
            receipts[rid] = [name, label, world, 1, True]; dirty = True
        elif kind == 'V':
            rid, version, value, enabled = cmd[1], *map(int, cmd[2:])
            require(row == ['V', rid, '1'] and receipts[rid][3] == version, 'receipt edit rejected/stale')
            receipts[rid][1] = value; receipts[rid][3] += 1; receipts[rid][4] = bool(enabled); dirty = True
        elif kind == 'R':
            stage = cmd[1]; dirty = False; models = {}; by_stage[stage] = models
            require(row[1] == stage and int(row[3]) == len(receipts), 'wrong rebuild inventory')
            while out_at < len(output) and output[out_at][0] in ('M', 'D', 'E'):
                r = output[out_at]; out_at += 1; require(r[1] == stage, 'wrong model stage'); name = r[2]
                if r[0] == 'M':
                    require(name not in models, 'duplicate model')
                    models[name] = {'status': r[3], 'arity': int(r[4]), 'count': int(r[5]), 'candidates': int(r[6]), 'work': int(r[7]), 'local': {}, 'expanded': {}}
                else: models[name]['local' if r[0] == 'D' else 'expanded'][int(r[3])] = parse_clause(r[4:])
            for name, model in models.items():
                require(len(model['local']) == len(model['expanded']) == model['count'], 'missing definition')
                require(model['work'] <= 20000000 and model['candidates'] <= 200001, 'unbounded search')
                for hid, local in model['local'].items():
                    require(len(local[1]) <= (8 if mode == 'wide' else 3), 'local body budget exceeded')
                    if mode != 'reuse': require(all(p not in meta['heads'] for p, _ in local[1]), 'flat control reused a definition')
                    require(equivalent(expand(local, models, meta['heads'], name), model['expanded'][hid]), 'native expansion differs from independent expansion')
                    for target, label, world, version, enabled in receipts.values():
                        if enabled and target == name:
                            require(evaluate(model['expanded'][hid], world) == label, 'definition contradicts active receipt'); state_checks += 1
            active = models
        else:
            qid, name = cmd[1:3]; world, at = parse_world(cmd, 3); case = cases[qid]
            require(at == len(cmd) and row[1:3] == [stage, qid], 'query/output identity differs')
            require(world == dict(case['world'], facts=[tuple(a) for a in case['world']['facts']]), 'oracle/query world differs')
            value = int(row[3]); require(value in (-1, 0, 1), 'invalid prediction')
            model = active.get(name); votes = [] if dirty or not model else [evaluate(h, world) for h in model['expanded'].values()]
            expected = votes[0] if votes and len(set(votes)) == 1 else -1
            require(value == expected, 'native inference differs from independent relational join')
            target = oracle(meta, case['family'], world)
            if case['group'].startswith('dirty'): target = -1
            elif not world['complete'] and target == 0: target = -1
            require(target == int(case['gold']), 'incorrect held-out oracle label')
            bucket = metrics[f'{case["group"]}/{case["family"]}']; bucket['total'] += 1
            bucket['correct'] += value == target; bucket['wrong'] += value != -1 and value != target; bucket['withheld'] += value == -1
            predictions[qid] = value
    require(out_at == len(output) and len(predictions) == len(cases), 'extra output / omitted queries')
    for name in meta['heads']:
        require(by_stage['final'][name]['expanded'] == by_stage['recovered'][name]['expanded'], 'correction did not restore representations')
    semantic_matches = {}
    for family, name in enumerate(meta['heads']):
        definitions = by_stage['final'][name]['expanded']
        semantic_matches[str(family)] = bool(definitions) and all(equivalent(c, intended(meta, family)) for c in definitions.values())
    result = {'metrics': dict(metrics), 'state_checks': state_checks, 'models': by_stage, 'semantic_matches': semantic_matches}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', default='logs/concept_growth'); parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--modes', nargs='+', default=['reuse', 'flat', 'wide']); parser.add_argument('--exploratory', action='store_true')
    opts = parser.parse_args(); root = Path(opts.out); report = {'runs': {}}
    for seed in range(opts.seeds):
        for mode in opts.modes:
            result = check_run(root/str(seed), mode); report['runs'][f'{seed}/{mode}'] = result
            final = [m for key, m in result['metrics'].items() if key.startswith('final/')]
            total = sum(m['total'] for m in final); correct = sum(m['correct'] for m in final); wrong = sum(m['wrong'] for m in final)
            print(f'seed={seed} mode={mode} final={correct}/{total} wrong={wrong} audited_receipt_checks={result["state_checks"]}')
            if not opts.exploratory:
                if mode in ('reuse', 'wide'):
                    require(all(result['semantic_matches'].values()), 'learned definition differs from intended relational semantics')
                    for key, m in result['metrics'].items():
                        if not key.startswith('conflict/'): require(m['correct'] == m['total'], f'incomplete reuse result: {seed} {key} {m}')
                        else: require(m['wrong'] == 0, 'conflict produced incorrect committed answer')
                else:
                    require(0 < correct < total, 'flat control did not show bounded composition gap')
    paths = [Path('cpp/include/laminar/concept_memory.h'), Path('cpp/apps/concept_growth_probe.cpp'), Path('cpp/tests/test_concept_memory.cpp'),
             Path('python/concept_growth_probe.py'), Path('python/check_concept_growth.py'), Path('scripts/concept_growth.sh'),
             Path('build-wsl/concept_growth_probe'), Path('build-asan/test_concept_memory')]
    paths += [p for p in root.rglob('*') if p.is_file() and p.suffix in ('.txt', '.json') and p.name not in ('audit.json',)]
    report['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (root/'audit.json').write_text(json.dumps(report, indent=2))
    print(f'independent concept audit passed: {root / "audit.json"}')


if __name__ == '__main__': main()
