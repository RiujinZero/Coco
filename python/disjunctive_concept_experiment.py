"""Public-outcome union induction with an independent relational semantic audit."""
import argparse
import hashlib
import itertools
import json
import random
import subprocess
import time
from pathlib import Path


def require(okay, message):
    if not okay:
        raise AssertionError(message)


def proves(clause, facts, args):
    arity, body = clause
    require(arity == len(args), 'wrong head arity')
    rows = {}
    for pred, values in facts:
        rows.setdefault(pred, set()).add(tuple(values))
    bindings = [dict(enumerate(args))]
    for pred, variables in body:
        next_bindings = []
        for binding in bindings:
            for values in rows.get(pred, ()):
                assigned = dict(binding)
                for variable, value in zip(variables, values):
                    if variable in assigned and assigned[variable] != value:
                        break
                    assigned[variable] = value
                else:
                    next_bindings.append(assigned)
        bindings = next_bindings
    return bool(bindings)


def truth(formula, facts, args):
    return any(proves(clause, facts, args) for clause in formula)


def contains(left, right):
    # CQ canonical databases independently establish exact UCQ containment:
    # every left disjunct must imply some right disjunct.
    for arity, body in left:
        facts = [(p, [f'v{x}' for x in variables]) for p, variables in body]
        if not truth(right, facts, [f'v{i}' for i in range(arity)]):
            return False
    return True


def equivalent(left, right):
    return contains(left, right) and contains(right, left)


def expand(formula, models, order, current):
    result = []
    for arity, body in formula:
        fresh = max([arity-1] + [v for _, vs in body for v in vs]) + 1
        products = [[]]
        for pred, variables in body:
            if pred not in models:
                options = [[(pred, variables)]]
            else:
                require(order.index(pred) < order.index(current), 'recursive/forward dependency escaped class')
                definitions = models[pred]['expanded']
                require(len(definitions) == 1, 'ambiguous concept reused')
                options = []; allocated = 0
                for dependency_arity, dependency in definitions[0]:
                    require(dependency_arity == len(variables), 'dependency role mismatch')
                    existential = sorted({v for _, vs in dependency for v in vs if v >= dependency_arity})
                    mapping = dict(enumerate(variables))
                    mapping.update({v: fresh+i for i, v in enumerate(existential)})
                    allocated = max(allocated, len(existential))
                    options.append([(p, [mapping[v] for v in vs]) for p, vs in dependency])
                fresh += allocated
            products = [prefix + suffix for prefix in products for suffix in options]
        result.extend((arity, body) for body in products)
    return result


def serialize_world(facts, args, complete=True):
    values = [int(complete), len(args), *args, len(facts)]
    for pred, names in facts:
        values.extend([pred, len(names), *names])
    return ' '.join(map(str, values))


def parse_formula(tokens):
    count = int(tokens[0]); at = 1; formula = []
    for _ in range(count):
        arity, size = map(int, tokens[at:at+2]); at += 2; body = []
        for _ in range(size):
            pred, n = tokens[at], int(tokens[at+1]); at += 2
            body.append((pred, list(map(int, tokens[at:at+n])))); at += n
        formula.append((arity, body))
    require(at == len(tokens), 'trailing formula data')
    return formula


def curriculum(seed):
    rng = random.Random(seed); names = [f'p{n}' for n in rng.sample(range(100, 999), 5)]
    r, s, t, link, relay = names
    flips = {p: bool(rng.randrange(2)) for p in (r, s, t)}
    head_order = [0, 1, 2]; rng.shuffle(head_order)
    def atom(pred, args):
        return pred, list(reversed(args)) if flips[pred] else list(args)
    def term(pred, variables):
        return atom(pred, variables)
    link_formula = [(2, [term(r, [0, 2]), term(s, [2, 1])]), (2, [term(t, [0, 1])])]
    position = {role: head_order.index(role) for role in range(3)}
    relay_formula = []
    for first, second in itertools.product(range(2), repeat=2):
        body = []
        for branch, a, b, z in ((first, 0, 1, 3), (second, 1, 2, 4)):
            a, b = position[a], position[b]
            body += [term(t, [a, b])] if branch else [term(r, [a, z]), term(s, [z, b])]
        relay_formula.append((3, body))
    targets = {link: link_formula, relay: relay_formula}
    commands = []; queries = {}; lessons = {}; query_count = 0; lesson_count = 0
    def teach(name, facts, args):
        nonlocal lesson_count
        identity = f'l{lesson_count}'; lesson_count += 1
        answer = truth(targets[name], facts, args)
        commands.append(f'T {identity} {name} {int(answer)} {serialize_world(facts, args)}')
        lessons[identity] = dict(name=name, facts=facts, args=args, answer=answer)
        return identity
    def query(stage, family, name, facts, args, complete=True):
        nonlocal query_count
        identity = f'q{query_count}'; query_count += 1
        commands.append(f'Q {identity} {name} {serialize_world(facts, args, complete)}')
        queries[identity] = dict(stage=stage, family=family, name=name, facts=facts, args=args, complete=complete)
    def edge(branch, a, b, z):
        return [atom(t, [a, b])] if branch else [atom(r, [a, z]), atom(s, [z, b])]
    def world(name, index, noise=True):
        entities = [f'e{seed}_{index}_{j}' for j in range(8)]
        if name == link:
            facts = edge(index % 2, *entities[:3]); args = entities[:2]
        else:
            facts = edge(index % 2, entities[0], entities[1], entities[3])
            facts += edge((index // 2) % 2, entities[1], entities[2], entities[4])
            args = [entities[i] for i in head_order]
        if index % 3 == 0:
            facts.pop(rng.randrange(len(facts)))
        if noise:
            for _ in range(4):
                pred = rng.choice([r, s, t]); values = rng.sample(entities, 2); facts.append(atom(pred, values))
            if facts:
                facts.append(rng.choice(facts))
            rng.shuffle(facts)
        return facts, args
    # Two irreducible positives make single-conjunction impossibility visible.
    teach(link, edge(0, 'a', 'b', 'z'), ['a', 'b'])
    direct_id = teach(link, edge(1, 'a', 'b', 'z'), ['a', 'b'])
    teach(link, [atom(r, ['a', 'x']), atom(s, ['y', 'b'])], ['a', 'b'])
    teach(link, [], ['a', 'b'])
    commands.append('R first')
    for i in range(128):
        facts, args = world(link, i); query('first', 'new_union', link, facts, args)
    for first, second in itertools.product(range(2), repeat=2):
        facts = edge(first, 'a', 'b', 'u') + edge(second, 'b', 'c', 'v')
        args = [['a', 'b', 'c'][i] for i in head_order]
        teach(relay, facts, args)
        for at in range(len(facts)):
            teach(relay, facts[:at]+facts[at+1:], args)
            for side in range(len(facts[at][1])):
                detached = [(pred, list(values)) for pred, values in facts]
                detached[at][1][side] = 'detached'
                teach(relay, detached, args)
        # The specified middle role must join both legs. Mere existence of an
        # unrelated incoming/outgoing link must not explain the queried tuple.
        separated = edge(first, 'a', 'b', 'u') + edge(second, 'c', 'd', 'v')
        for head in (['a', 'b', 'd'], ['a', 'c', 'd']):
            teach(relay, separated, [head[i] for i in head_order])
    for i in range(24):
        facts, args = world(relay, 1000+i); teach(relay, facts, args)
    commands.append('R final')
    for name, family, total in [(link, 'retained_union', 128), (relay, 'new_composition', 256)]:
        for i in range(total):
            facts, args = world(name, 2000+i); query('final', family, name, facts, args)
    for i in range(64):
        name = link if i % 2 else relay; facts, args = world(name, 3000+i)
        query('final', 'partial', name, facts, args, False)
    conflict = dict(lessons[direct_id]); conflict['answer'] = False
    commands.append(f'T conflict {link} 0 {serialize_world(conflict["facts"], conflict["args"])}')
    lessons['conflict'] = conflict
    commands.append('R conflict')
    for name in (link, relay):
        facts, args = world(name, 4001); query('conflict', 'conflict_withholding', name, facts, args)
    commands += ['V conflict 1 0 0', 'R restored']
    for name in (link, relay):
        facts, args = world(name, 4001); query('restored', 'restoration', name, facts, args)
    return '\n'.join(commands)+'\n', dict(seed=seed, link=link, relay=relay, targets=targets, queries=queries, lessons=lessons,
                                              supplied=dict(base_schemas='observed tuples', head_arity='public query', body=2, clauses=2, expanded_clauses=8,
                                                            candidate_budget=200000, work_budget=20000000))


def audit(output, meta, mode):
    snapshots = {}; answers = {}; accepted = 0
    for line in output.splitlines():
        token = line.split()
        if token[0] in ('T', 'V'):
            require(token[-1] == '1', 'native rejected verified input'); accepted += 1
        elif token[0] == 'R':
            snapshots[token[1]] = {}
        elif token[0] == 'M':
            snapshots[token[1]][token[2]] = dict(status=token[3], count=int(token[4]), candidates=int(token[5]), work=int(token[6]), local=[], expanded=[])
        elif token[0] in ('D', 'E'):
            field = 'local' if token[0] == 'D' else 'expanded'
            snapshots[token[1]][token[2]][field].append(parse_formula(token[4:]))
        elif token[0] == 'A':
            require(token[2] not in answers, 'duplicate query output'); answers[token[2]] = int(token[3])
        else:
            raise AssertionError(f'unknown native output: {line}')
    require(set(answers) == set(meta['queries']), 'missing or extra native answers')
    counts = {}; replayed = 0; receipt_checks = 0; expansion_checks = 0
    for stage, models in snapshots.items():
        for name, model in models.items():
            require(model['count'] == len(model['expanded']) == len(model['local']), 'lost model alternatives')
            for local, expanded in zip(model['local'], model['expanded']):
                reconstructed = expand(local, models, list(meta['targets']), name)
                require(equivalent(reconstructed, expanded), 'native local expansion changed union semantics'); expansion_checks += 1
                for identity, receipt in meta['lessons'].items():
                    if receipt['name'] != name or (identity == 'conflict' and stage != 'conflict'):
                        continue
                    require(truth(expanded, receipt['facts'], receipt['args']) == receipt['answer'], 'model violates active outcome evidence')
                    receipt_checks += 1
    for identity, query in meta['queries'].items():
        model = snapshots[query['stage']][query['name']]
        votes = [1 if truth(formula, query['facts'], query['args']) else 0 if query['complete'] else -1 for formula in model['expanded']]
        expected = votes[0] if votes and len(set(votes)) == 1 else -1
        require(answers[identity] == expected, 'native and independent model inference differ'); replayed += 1
        golden = 1 if truth(meta['targets'][query['name']], query['facts'], query['args']) else 0 if query['complete'] else -1
        if query['stage'] == 'conflict' and mode == 'union':
            golden = -1
        tally = counts.setdefault(query['family'], dict(correct=0, total=0, withheld=0))
        tally['correct'] += int(answers[identity] == golden); tally['total'] += 1; tally['withheld'] += int(answers[identity] == -1)
    semantics = {}
    for name, target in meta['targets'].items():
        final = snapshots['final'][name]
        require(final['count'] == len(final['expanded']) == len(final['local']), 'lost hypothesis branches')
        semantics[name] = bool(final['expanded']) and all(equivalent(formula, target) for formula in final['expanded'])
        if mode in ('union', 'wide_union'):
            require(semantics[name], 'learned union is not independently equivalent to target')
            require(snapshots['restored'][name]['expanded'] == final['expanded'], 'evidence restoration changed learned semantics')
    if mode == 'union':
        require(all(v['correct'] == v['total'] for v in counts.values()), 'union arm fails declared family')
        local = snapshots['final'][meta['relay']]['local']
        require(len(local) == 1 and any(p == meta['link'] for _, body in local[0] for p, _ in body), 'composition did not reuse learned union')
    if mode == 'wide':
        require(snapshots['final'][meta['link']]['status'] == 'no_bounded_hypothesis', 'single conjunction control has unexpected solution')
    if mode == 'flat':
        require(semantics[meta['link']] and not semantics[meta['relay']], 'no-reuse control has unexpected expressivity')
    return dict(mode=mode, families=counts, exact_semantics=semantics, independent_predictions=replayed, accepted_records=accepted,
                active_outcome_checks=receipt_checks, independent_expansion_checks=expansion_checks, models=snapshots, passed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', default='build-wsl/disjunctive_concept_probe_v1')
    parser.add_argument('--out', default='logs/disjunctive_concepts_v1')
    args = parser.parse_args(); root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in (0, 1, 2):
        folder = root / str(seed); folder.mkdir(); corpus, meta = curriculum(seed)
        source = folder / 'input.txt'; source.write_text(corpus); (folder / 'oracle.json').write_text(json.dumps(meta, indent=2))
        for mode in ('union', 'flat', 'wide', 'wide_union'):
            started = time.monotonic()
            result = subprocess.run([str(Path(args.binary).resolve()), str(source), mode], capture_output=True, text=True, timeout=120)
            (folder / f'{mode}.out').write_text(result.stdout); (folder / f'{mode}.err').write_text(result.stderr)
            require(result.returncode == 0, f'native {seed} {mode} failed: {result.stderr}')
            report = audit(result.stdout, meta, mode); report.update(seed=seed, seconds=time.monotonic()-started)
            (folder / f'{mode}.audit.json').write_text(json.dumps(report, indent=2)); results.append(report)
            print(seed, mode, round(report['seconds'], 3), report['families'], flush=True)
    files = ['cpp/include/laminar/concept_memory.h', 'cpp/apps/disjunctive_concept_probe.cpp', 'cpp/tests/test_disjunctive_concepts.cpp',
             'python/disjunctive_concept_experiment.py', args.binary]
    summary = dict(passed=True, seeds=[0, 1, 2], modes=['union', 'flat', 'wide', 'wide_union'],
                   runs=[{k: v for k, v in result.items() if k != 'models'} for result in results],
                   sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files})
    (root / 'audit.json').write_text(json.dumps(summary, indent=2)); print(root / 'audit.json')


if __name__ == '__main__':
    main()
