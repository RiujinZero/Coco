"""Independently replay primitive VM traces and score computational worlds/actions."""
import argparse
import collections
import hashlib
import json
import math
import shutil
from pathlib import Path
from check_primitive_discovery import audit, policy, rows

ROOT = Path(__file__).resolve().parents[1]


def tuples(row):
    return {tuple(a.split(',')) for a in row['facts'].split(';') if a}


def edges(facts, relation):
    if relation == 'eligible':
        return {(a, b) for r, a, b in facts if r == 'edge'} & {(a, b) for r, a, b in facts if r == 'marked'}
    return {(a, b) for r, a, b in facts if r == relation}


def follow(pairs, start):
    seen = set()
    while True:
        assert start not in seen, 'cycle outside the evaluation domain'
        seen.add(start)
        values = {b for a, b in pairs if a == start}
        if not values:
            return start
        assert len(values) == 1
        start = next(iter(values))


def count(pairs, start):
    return len({b for a, b in pairs if a == start})


def evaluate(row, program):
    facts = tuples(row)
    a, b = row['a'], row['b']
    r, s = edges(facts, row['r']), edges(facts, row['s'])
    family = int(row['family'])
    if family == 0:
        truth = count(r, a) == count(s, b)
    elif family == 1:
        truth = follow(r, a) == follow(s, b)
    else:
        assert family == 2
        truth = count(s, follow(r, a)) == count(s, b)
    # Replay the learned composition independently with typed primitive results.
    regs = [(0, a), (0, b)]
    result = None
    for op in program:
        if op == 0: regs[0] = (0, a)
        elif op == 1: regs[0] = (0, b)
        elif op == 2: regs[1] = regs[0]
        elif op == 3: regs.reverse()
        elif op in (4, 5):
            assert regs[0][0] == 0
            regs[0] = (0, follow((r, s)[op % 2], regs[0][1]))
        elif op in (6, 7):
            assert regs[0][0] == 0
            regs[0] = (2, count((r, s)[op % 2], regs[0][1]))
        elif op == 8: regs[0] = (1, regs[0] == regs[1])
        elif op == 9:
            result = regs[0]
            break
        else: raise AssertionError(op)
    assert result == (1, truth)
    assert int(row['gold']) == truth
    assert int(row['predicted']) == truth
    assert int(row['work']) > 0
    return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'logs/computational_session')
    args = parser.parse_args()
    args.out = args.out.resolve()
    reports = {}
    for seed in range(3):
        prefix = args.out / str(seed)
        report = {}
        for task in range(3):
            base = Path(str(prefix)+f'.skill{task}')
            score, _ = audit(base, rules=policy(Path(str(base)+'.policy.txt')), training_budget=128)
            value = score[f'training/task{task}/disabled-1']
            assert value['correct'] == value['n'] == 40
            report[f'acquisition{task}'] = value
        score, _ = audit(Path(str(prefix)+'.neural'))
        for task in range(3):
            value = score[f'retained/task{task}/disabled-1']
            assert value['correct'] == value['n'] == 72
            report[f'retention{task}'] = value
        checkpoints = [Path(str(prefix)+'.initial.weights.txt')] + [Path(str(prefix)+f'.stage{i}.weights.txt') for i in (1, 2, 0)]
        weights = [[float(x) for x in p.read_text().splitlines()] for p in checkpoints]
        assert all(len(w) == 15360 and all(map(math.isfinite, w)) for w in weights)
        assert not any(weights[0]) and all(a != b for a, b in zip(weights, weights[1:]))
        names = {'vex': 0, 'pim': 1, 'zot': 2}
        programs = {}
        for row in rows(Path(str(prefix)+'.programs.tsv')):
            family = names[row['frame'].split()[0].removeprefix('computed:')]
            programs[family] = [int(a) for a in row['instructions'].strip(',').split(',')]
        assert set(programs) == {0, 1, 2}
        counts = collections.Counter()
        before = None
        for row in rows(Path(str(prefix)+'.cases.tsv')):
            assert int(row['seed']) == seed
            facts = evaluate(row, programs[int(row['family'])])
            counts[row['phase']] += 1
            if row['phase'] == 'physical_before':
                assert before is None and row['predicted'] == '0'
                before = facts
            elif row['phase'] == 'physical_after':
                assert before is not None and row['predicted'] == '1'
                motor, a, b = row['action'].split(',')
                assert motor == 'motor' and facts == before | {('marked', a, b)}
                before = None
            else:
                assert row['phase'] == 'heldout' and not row['action']
        assert before is None and counts == {'heldout': 336, 'physical_before': 40, 'physical_after': 40}
        report['questions_and_goals'] = dict(counts)
        report['programs'] = programs
        # Contract assertions are native tests, not independently replayed claims.
        log = Path(str(prefix)+'.txt').read_text()
        assert f'seed={seed} questions=336/336 physical_goals=40/40 contracts=passed' in log
        reports[str(seed)] = report
    source = ['cpp/include/laminar/primitive_store.h', 'cpp/include/laminar/procedure_memory.h',
              'cpp/include/laminar/relational_session.h', 'cpp/include/laminar/relational_control.h',
              'cpp/apps/computational_session_probe.cpp', 'cpp/apps/primitive_discovery_support.h',
              'python/check_computational_session.py', 'python/check_primitive_discovery.py',
              'scripts/computational_session.sh', 'cpp/CMakeLists.txt', 'cpp/tests/test_computational_session.cpp']
    source += [str(p.relative_to(ROOT)) for p in (ROOT/'cpp/include/laminar').glob('*.h')]
    artifacts = [p for p in args.out.iterdir() if p.is_file() and p.name != 'audit.json']
    paths = set(ROOT/p for p in source) | set(artifacts)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    snapshots = {}
    for original in sorted(set(source)):
        saved = args.out/'source'/original
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/original, saved)
        snapshots[original] = {'path': str(saved.relative_to(ROOT)), 'sha256': hashes[original]}
    (args.out/'audit.json').write_text(json.dumps({'runs': reports, 'sha256': hashes, 'source_snapshots': snapshots,
        'scope': 'Independent world labels, physical updates, learned composition execution and primitive VM traces; native assertions for revision/clone/control contracts. No learned-controller benchmark in this experiment.'}, indent=2)+'\n')
    print('Audited: 1008 questions, 120 goals, 648 retained primitive cases; three seeds.')


if __name__ == '__main__':
    main()
