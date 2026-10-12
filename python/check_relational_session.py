"""Independent audit of the continuing model, learned dependencies and actions.

Reconstruct receipts from the exogenous stream and actual execution observations;
check each published model against those receipts. Use an independent relational
interpreter and explicit effect semantics, never the native reported answer as gold.
"""
import argparse
import collections
import copy
import hashlib
import json
import shlex
from pathlib import Path
from check_concept_growth import parse_world, parse_clause, evaluate, equivalent, intended, oracle, require
from computational_control_oracle import binding as computational_binding, program_answer


def canonical(world):
    return bool(world['complete']), tuple(sorted({(p, tuple(vs)) for p, vs in world['facts']}))


def action(tokens, at):
    name, n = tokens[at], int(tokens[at+1]); at += 2
    return {'schema': name, 'args': tokens[at:at+n]}, at+n


def effect(rule, command, world):
    pred, subject, obj, kind = rule
    a, b = command['args'][subject], command['args'][obj]
    facts = {(p, tuple(vs)) for p, vs in world['facts']}
    if kind == 1: facts.discard((pred, (a, b)))
    elif kind == 2: facts = {(p, vs) for p, vs in facts if p != pred or vs[0] != a}
    elif kind == 3: facts = {(p, vs) for p, vs in facts if p != pred or vs[1] != b}
    if kind != 1: facts.add((pred, (a, b)))
    return {'args': [], 'facts': [(p, list(vs)) for p, vs in sorted(facts)], 'complete': True}


def bind(pattern, text):
    if pattern.startswith('computed:'):
        b = computational_binding(pattern, text)
        return None if b is None else b[0]
    literal, tokens = pattern.split(), text.split()
    if len(literal) != len(tokens): return None
    args = []
    for p, t in zip(literal, tokens):
        if p.startswith('$'):
            require(p == f'${len(args)}', 'invalid surface slot'); args.append(t)
        elif p != t: return None
    return args


def renamed_clause(clause, slots):
    if isinstance(clause, dict):
        return {'any': [renamed_clause(branch, slots) for branch in clause['any']]}
    n, body = clause
    return n, [(p, tuple(slots[v] if v < n else v for v in vs)) for p, vs in body]


def language_answer(snapshot, text, world):
    if not snapshot or not snapshot['ready']: return -1
    matches = [(f, bind(key, text)) for key, f in snapshot['language'].items() if bind(key, text) is not None]
    computations = [(key, f) for key, f in snapshot.get('computations', {}).items() if computational_binding(key, text) is not None]
    if computations:
        if len(computations) != 1 or matches: return -1
        key, frame = computations[0]
        definitions = {name: model['expanded'][0] for name, model in snapshot['concepts'].items() if len(model['expanded']) == 1}
        return program_answer(frame['program'], text, world, key, definitions) if frame['learned'] else -1
    if len(matches) != 1: return -1
    f, args = matches[0]; values = []
    for pred, negate, slots in f['meanings']:
        raw = dict(world, args=[args[i] for i in slots]); definitions = snapshot['concepts'][pred]['expanded']
        votes = [evaluate(d, raw) for d in definitions]
        value = votes[0] if votes and len(set(votes)) == 1 else -1
        values.append(-1 if value < 0 else value ^ negate)
    return values[0] if values and len(set(values)) == 1 else -1


def predict(snapshot, command, world):
    f = snapshot['motors'].get(command['schema'])
    if not f or not f['rules'] or f['arity'] != len(command['args']) or not world['complete']: return None
    worlds = [effect(r, command, world) for r in f['rules']]
    return worlds[0] if len({canonical(w) for w in worlds}) == 1 else None


def instantiate(program, text, world):
    frame, code = program; args = bind(frame, text)
    if args is None: return None
    result = []
    for name, operands in code:
        values = []
        for index, pred, source, target in operands:
            require(0 <= index < len(args) and source in (0, 1) and target in (0, 1), 'invalid procedure operand')
            if not pred: values.append(args[index]); continue
            candidates = {vs[target] for p, vs in world['facts'] if p == pred and len(vs) == 2 and vs[source] == args[index]}
            if len(candidates) != 1: return None
            values.append(next(iter(candidates)))
        result.append({'schema': name, 'args': values})
        # The caller supplies the per-step world when checking a whole execution.
        if len(code) != 1: return result
    return result


def supports(program, execution, snapshot):
    frame, code = program
    done = execution.get('executed', [True]*len(execution['actions']))
    actual_steps = [(a, w) for a, w, executed in zip(execution['actions'], execution['observed'], done) if executed]
    if not execution['success'] or len(code) != len(actual_steps) or bind(frame, execution['text']) is None: return False
    world = execution['before']
    for instruction, (actual, observed) in zip(code, actual_steps):
        commands = instantiate((frame, [instruction]), execution['text'], world)
        if not commands or commands[0] != actual: return False
        predicted = predict(snapshot, actual, world)
        if predicted is None or canonical(predicted) != canonical(observed): return False
        world = observed
    return language_answer(snapshot, execution['text'], world) == 1


def effective_program(program, text, world, snapshot):
    frame, code = program; result = []
    if bind(frame, text) is None: return None
    for instruction in code:
        commands = instantiate((frame, [instruction]), text, world)
        if not commands: return None
        after = predict(snapshot, commands[0], world)
        if after is None: return None
        if canonical(after) != canonical(world): result.append(commands[0])
        world = after
    return result if language_answer(snapshot, text, world) == 1 else None


def parse_snapshot(rows, at):
    r = rows[at]; require(r[0] == 'R', 'missing snapshot'); at += 1
    snapshot = {'stage': r[1], 'version': int(r[2]), 'ready': bool(int(r[3])), 'counts': list(map(int, r[4:])),
                'concepts': {}, 'language': {}, 'motors': {}, 'procedures': [], 'computations': {}, 'primitives': None}
    while rows[at][0] != 'END':
        r = rows[at]; at += 1; name = r[1]
        if r[0] == 'C': snapshot['concepts'][name] = {'status': r[2], 'arity': int(r[3]), 'count': int(r[4]), 'local': [], 'expanded': []}
        elif r[0] in ('D', 'E'): snapshot['concepts'][name]['local' if r[0] == 'D' else 'expanded'].append(parse_clause(r[2:]))
        elif r[0] in ('DX', 'EX'):
            definitions = snapshot['concepts'][name]['local' if r[0] == 'DX' else 'expanded']
            require(bool(definitions), 'rule branch without a hypothesis')
            if not isinstance(definitions[-1], dict): definitions[-1] = {'any': [definitions[-1]]}
            definitions[-1]['any'].append(parse_clause(r[2:]))
        elif r[0] == 'L': snapshot['language'][name] = {'support': int(r[2]), 'count': int(r[3]), 'meanings': []}
        elif r[0] == 'I':
            require(int(r[4]) == len(r[5:]), 'bad language arity')
            snapshot['language'][name]['meanings'].append((r[2], int(r[3]), list(map(int, r[5:]))))
        elif r[0] == 'K':
            require(int(r[5]) == len(r[6:]), 'invalid computation program length')
            snapshot['computations'][name] = {'cue': int(r[2]), 'support': int(r[3]), 'learned': bool(int(r[4])), 'program': list(map(int, r[6:]))}
        elif r[0] == 'V':
            require(len(r) == 5, 'invalid primitive snapshot')
            snapshot['primitives'] = [int(v) for v in r[1:]]
        elif r[0] == 'F': snapshot['motors'][name] = {'arity': int(r[2]), 'support': int(r[3]), 'count': int(r[4]), 'rules': []}
        elif r[0] == 'B': snapshot['motors'][name]['rules'].append((r[2], *map(int, r[3:])))
        elif r[0] == 'P':
            length = int(r[2]); cursor = 3; code = []
            for _ in range(length):
                op, count = r[cursor], int(r[cursor+1]); cursor += 2; operands = []
                for _ in range(count):
                    index, pred, source, target = r[cursor:cursor+4]; cursor += 4
                    operands.append((int(index), pred, int(source), int(target)))
                code.append((op, operands))
            require(cursor == len(r), 'trailing procedure fields'); snapshot['procedures'].append((name, code))
        else: raise AssertionError(f'unknown snapshot row: {r}')
    return snapshot, at+1


def audit_snapshot(snapshot, concept_receipts, language, transitions, executions, meta, procedure_capacity=256):
    require(snapshot['counts'] == [len(concept_receipts), len(language), len(transitions), len(executions), len(snapshot['procedures'])], 'raw evidence inventory mismatch')
    require(len(snapshot['procedures']) <= procedure_capacity, 'procedure capacity escaped')
    checks = collections.Counter()
    for name, model in snapshot['concepts'].items():
        require(model['count'] == len(model['local']) == len(model['expanded']), 'missing concept alternatives')
        for definition in model['expanded']:
            for r in concept_receipts.values():
                if r['name'] == name:
                    require(evaluate(definition, r['world']) == r['gold'], 'concept contradicts active evidence'); checks['concept'] += 1
        if snapshot['ready']:
            target = meta['definitions'][name] if meta.get('computational') else intended(meta, meta['heads'].index(name))
            alternatives = meta.get('concept_alternatives', {}).get(snapshot['stage'], {}).get(name, [target])
            require(model['count'] == len(alternatives) and all(
                sum(equivalent(actual, expected) for actual in model['expanded']) == 1
                for expected in alternatives), 'concept semantics mismatch')
    for key, frame in snapshot.get('computations', {}).items():
        fitting = [r for r in language.values() if computational_binding(key, r['text']) is not None]
        require(frame['support'] == len(fitting), 'incorrect computational-language support')
        require(frame['learned'] == bool(frame['program']) and len(frame['program']) <= 10, 'invalid learned computation')
        if frame['learned']:
            for r in fitting:
                require(language_answer(snapshot, r['text'], r['world']) == r['gold'], 'computation contradicts raw language answer')
                checks['computational_language'] += 1
    for key, frame in snapshot['language'].items():
        require(frame['count'] == len(frame['meanings']), 'missing language alternatives')
        fitting = [r for r in language.values() if bind(key, r['text']) is not None]
        require(frame['support'] == len(fitting), 'wrong language support')
        for pred, negate, slots in frame['meanings']:
            for r in fitting:
                bound = bind(key, r['text']); world = dict(r['world'], args=[bound[i] for i in slots])
                values = [evaluate(c, world) for c in snapshot['concepts'][pred]['expanded']]
                require(values and all(v >= 0 and (v ^ negate) == r['gold'] for v in values), 'language interpretation contradicts raw answer'); checks['language'] += 1
            if fitting:
                r = fitting[0]; target = intended(meta, r['family']); target = renamed_clause(target, [meta['surface_roles'][r['family']][r['variant']].index(i) for i in range(target[0])])
                actual = renamed_clause(snapshot['concepts'][pred]['expanded'][0], slots)
                require(negate == r['variant'] and equivalent(actual, target), 'language learned a semantic shortcut')
    for name, motor in snapshot['motors'].items():
        require(motor['count'] == len(motor['rules']), 'missing effect alternatives')
        evidence = [r for r in transitions.values() if r['action']['schema'] == name and r.get('executed', True)]
        require(motor['support'] == len(evidence), 'wrong effect support')
        for rule in motor['rules']:
            for r in evidence:
                require(canonical(effect(rule, r['action'], r['before'])) == canonical(r['after']), 'effect contradicts observed transition'); checks['effect'] += 1
            expected = meta['effects'][name]
            require(rule == (expected['predicate'], expected['subject'], expected['object'], expected['kind']), 'wrong learned actuator semantics')
    for program in snapshot['procedures']:
        require(any(supports(program, r, snapshot) for r in executions.values()), 'procedure lacks a currently valid own execution witness'); checks['procedure'] += 1
    return checks


def audit_run(directory, mode='reuse'):
    meta = json.loads((directory/'oracle.json').read_text()); records = {r['id']: r for r in meta['records']}
    commands = [shlex.split(line) for line in (directory/'input.txt').read_text().splitlines()]
    rows = [shlex.split(line) for line in (directory/f'{mode}.txt').read_text().splitlines()]
    concept_receipts, language, transitions, executions = {}, {}, {}, {}
    at = version = 0; dirty = False; snapshot = None; pre_conflict = None; snapshots = {}; metrics = collections.defaultdict(collections.Counter); checks = collections.Counter()
    def consume_snapshot():
        nonlocal at, snapshot, dirty, checks
        snapshot, at = parse_snapshot(rows, at); require(snapshot['version'] == version, 'model version differs from evidence')
        checks.update(audit_snapshot(snapshot, concept_receipts, language, transitions, executions, meta)); snapshots[snapshot['stage']] = snapshot; dirty = False
    def steps(qid, tag, world, available, require_prediction):
        nonlocal at
        actions, observed = [], []
        while at < len(rows) and rows[at][0] == tag:
            r = rows[at]; at += 1; require(r[1] == qid and int(r[2]) == len(actions), 'execution step identity mismatch')
            command, cursor = action(r, 3); after, cursor = parse_world(r, cursor); require(cursor == len(r), 'trailing execution fields')
            require(command in available, 'executed unavailable action')
            e = meta['effects'][command['schema']]; expected = effect((e['predicate'], e['subject'], e['object'], e['kind']), command, world)
            require(canonical(expected) == canonical(after), 'native environment transition differs from independent physics')
            if require_prediction:
                predicted = predict(snapshot, command, world)
                require(predicted is not None and canonical(predicted) == canonical(after), 'planned action lacks matching learned effect')
            actions.append(command); observed.append(after); world = after
        return actions, observed, world
    for cmd in commands:
        op = cmd[0]
        if op in ('C', 'L', 'T'):
            rid = cmd[1]; r = copy.deepcopy(records[rid]); result = rows[at]; at += 1
            require(result[:4] == ['T', op, rid, '1'], 'teaching receipt rejected'); version += 1
            require(int(result[4]) == version, 'receipt did not advance version'); dirty = True
            if op == 'C':
                if r.get('deliberate_conflict'): pre_conflict = copy.deepcopy(snapshot)
                actual_world, cursor = parse_world(cmd, 4)
                require(cursor == len(cmd) and canonical(actual_world) == canonical(r['world']) and actual_world['args'] == r['world']['args'] and int(cmd[3]) == r['gold'], 'concept input differs from audited raw record')
                r['name'] = cmd[2]; concept_receipts[rid] = r
                if not r.get('deliberate_conflict'): require(r['gold'] == oracle(meta, r['family'], r['world']), 'incorrect concept teacher label')
            elif op == 'L':
                actual_world, cursor = parse_world(cmd, 3)
                entities = sorted(set(r['world']['args']) | {v for _, vs in r['world']['facts'] for v in vs})
                require(cursor+1 == len(cmd) and cmd[cursor] == r['text'] and canonical(actual_world) == canonical(r['world']) and actual_world['args'] == entities and int(cmd[2]) == r['gold'], 'language input differs from audited raw record')
                require(r['gold'] == (oracle(meta, r['family'], r['world']) ^ r['variant']), 'incorrect language teacher label'); language[rid] = r
            else:
                actual_action, cursor = action(cmd, 2); actual_before, cursor = parse_world(cmd, cursor); actual_after, cursor = parse_world(cmd, cursor)
                require(cursor == len(cmd) and actual_action == r['action'] and canonical(actual_before) == canonical(r['before']) and canonical(actual_after) == canonical(r['after']), 'transition input differs from audited record')
                e = meta['effects'][r['action']['schema']]; expected = effect((e['predicate'], e['subject'], e['object'], e['kind']), r['action'], r['before'])
                require(canonical(expected) == canonical(r['after']), 'incorrect external transition'); transitions[rid] = r
        elif op == 'R':
            consume_snapshot(); require(snapshot['stage'] == cmd[1], 'wrong snapshot stage')
        elif op == 'Q':
            rid = cmd[1]; r = records[rid]; result = rows[at]; at += 1
            actual_world, cursor = parse_world(cmd, 2)
            require(cursor+1 == len(cmd) and cmd[cursor] == r['text'] and canonical(actual_world) == canonical(r['world']), 'query input differs from audited world/text')
            require(result[:3] == ['Q', rid, str(version)], 'question/version identity differs')
            value = int(result[3]); expected = -1 if dirty else language_answer(snapshot, r['text'], r['world'])
            require(value == expected, 'native language answer differs from independent execution')
            base = oracle(meta, r['family'], r['world']); gold = base ^ r['variant']
            if not r['world']['complete'] and not base: gold = -1
            if r['group'] in ('conflict', 'dirty'): gold = -1
            require(gold == r['gold'], 'incorrect query gold')
            m = metrics['Q/'+r['group']]; m['total'] += 1; m['correct'] += value == gold; m['withheld'] += value == -1; m['wrong'] += value != gold and value != -1
        elif op in ('G', 'A'):
            rid = cmd[1]; r = records[rid]; before = r['world']; control = None
            actual_world, cursor = parse_world(cmd, 4); actual_text = cmd[cursor]; n = int(cmd[cursor+1]); cursor += 2; available = []
            for _ in range(n):
                a, cursor = action(cmd, cursor); available.append(a)
            require(cursor == len(cmd) and int(cmd[2]) == r['budget'] and int(cmd[3]) == 4 and canonical(actual_world) == canonical(before) and actual_text == r['text'] and available == r['available'], 'goal interface differs from audited public inputs')
            if op == 'G':
                ca, co, cw = steps(rid, 'Y', before, r['available'], True); result = rows[at]; at += 1
                require(result[:2] == ['K', rid] and int(result[4]) == len(ca), 'bad matched control')
                gold = oracle(meta, r['family'], dict(cw, args=before['args'])) ^ r['variant']
                require(gold == int(result[5]) and int(result[3]) <= r['budget'], 'incorrect/unbounded matched control')
                control = gold; m = metrics['K/'+r['group']]; m['total'] += 1; m['success'] += gold; m['predictions'] += int(result[3])
            actions, observed, world = steps(rid, 'X', before, r['available'], op == 'G'); result = rows[at]; at += 1
            require(result[:3] == ['G', rid, str(version)], 'decision version differs')
            status, reused, work, count, success, token, next_version, accepted = result[3], *map(int, result[4:])
            require(count == len(actions) <= 4 and work <= r['budget'], 'execution exceeded bound')
            gold = oracle(meta, r['family'], dict(world, args=before['args'])) ^ r['variant']
            require(gold == success, 'native reward differs from independent goal semantics')
            execution = {'id': f'execution{token}', 'text': r['text'], 'before': before, 'actions': actions, 'observed': observed, 'success': bool(success)}
            if reused: require(any(effective_program(p, execution['text'], before, snapshot) == actions for p in snapshot['procedures']), 'claimed procedure reuse lacks matching learned program')
            if accepted:
                require(token > 0 and count > 0 and next_version == version+1, 'non-atomic execution feedback')
                version += 1; eid = execution['id']; require(eid not in executions, 'duplicate execution'); executions[eid] = execution
                language[eid] = {'text': r['text'], 'world': dict(world, args=before['args']), 'gold': gold, 'family': r['family'], 'variant': r['variant']}
                previous = before
                for i, (a, after) in enumerate(zip(actions, observed)):
                    transitions[f'{eid}_step{i}'] = {'action': a, 'before': previous, 'after': after}; previous = after
                consume_snapshot()
            else: require(token == 0 and count == 0 and next_version == version, 'unacknowledged model mutation')
            m = metrics['G/'+r['group']]; m['total'] += 1; m['success'] += success; m['reused'] += reused; m['actions'] += count; m['predictions'] += work
        elif op == 'VC':
            rid = cmd[1]; result = rows[at]; at += 1; version += 1
            require(result == ['V', 'VC', rid, '1', str(version)], 'concept revision rejected')
            concept_receipts[rid]['gold'] = int(cmd[3]); dirty = True
        else: raise AssertionError(f'unsupported audit operation {op}')
    require(at == len(rows), 'unconsumed native outputs')
    require(snapshots['conflict']['counts'][-1] == 0 and not snapshots['conflict']['ready'], 'concept conflict failed to invalidate dependent procedures')
    require(snapshots['recovered']['ready'] and snapshots['recovered']['counts'][-1] > 0, 'concept recovery lost procedures')
    require(pre_conflict is not None and pre_conflict['concepts'] == snapshots['recovered']['concepts']
            and pre_conflict['language'] == snapshots['recovered']['language']
            and pre_conflict['motors'] == snapshots['recovered']['motors']
            and pre_conflict['procedures'] == snapshots['recovered']['procedures'], 'correction failed to restore the coupled model')
    return {'metrics': dict(metrics), 'checks': dict(checks), 'checkpoints': {s: {k: v for k, v in m.items() if k in ('version', 'ready', 'counts')} for s, m in snapshots.items()},
            'final_definitions': snapshots['recovered']['concepts'], 'final_language': snapshots['recovered']['language'], 'final_effects': snapshots['recovered']['motors'],
            'final_procedures': snapshots['recovered']['procedures']}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--seeds', type=int, default=3); p.add_argument('--out', default='logs/relational_session');p.add_argument('--exploratory', action='store_true')
    args = p.parse_args(); root = Path(args.out); report = {'runs': {}}
    for seed in range(args.seeds):
        result = audit_run(root/str(seed)); report['runs'][str(seed)] = result; m = result['metrics']
        print(f'seed={seed} final_questions={m["Q/final"]} goals={m["G/final_goals"]} limited={m["G/limited_procedure"]} matched_search={m["K/limited_procedure"]}', flush=True)
        if not args.exploratory:
            for key, value in m.items():
                if key.startswith('Q/'): require(value['correct'] == value['total'], f'question gap: {key} {value}')
                if key.startswith('G/') and key not in ('G/exploration', 'G/conflict_goals'): require(value['success'] == value['total'], f'goal gap: {key} {value}')
            require(m['G/conflict_goals']['actions'] == 0, 'acted under semantic conflict')
            require(m['G/limited_procedure']['success'] > m['K/limited_procedure']['success'], 'procedure control did not distinguish reuse')
    files = [Path(p) for p in ['cpp/include/laminar/relational_session.h', 'cpp/include/laminar/concept_memory.h', 'cpp/include/laminar/request_grounding.h',
        'cpp/include/laminar/event_grounding.h', 'cpp/apps/relational_session_probe.cpp', 'cpp/tests/test_relational_session.cpp',
        'python/relational_session_probe.py', 'python/check_relational_session.py', 'python/concept_growth_probe.py', 'python/check_concept_growth.py',
        'scripts/relational_session.sh', 'build-wsl/relational_session_probe', 'build-asan/test_relational_session']]
    files += [f for f in root.rglob('*') if f.is_file() and f.suffix in ('.json', '.txt') and f.name != 'audit.json']
    report['sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()}
    (root/'audit.json').write_text(json.dumps(report, indent=2)); print(f'independent continuing-model audit passed: {root / "audit.json"}')


if __name__ == '__main__': main()
