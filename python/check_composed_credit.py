"""Replay composed VM executions and reload the actual frozen neural policies."""
import argparse
import collections
import hashlib
import json
import math
import subprocess
from pathlib import Path
from check_primitive_discovery import ABSENT, audit, gold, key, policy, rows


def allowed(state):
    result = [0, 4, 9]
    if state & (1 << 4): result += [1, 5, 6]
    if state & (1 << 6): result += [2, 3, 7]
    if state & ((1 << 16) | (1 << 17)): result += [8]
    return result


def vm(task, subject, target, relation, facts, choose, budget=256):
    focus = value = output = ABSENT; cursor = pred = last = -1
    exhausted = False; seen = set(); count = steps = 0
    for _ in range(budget):
        state = key(task, focus, value, subject, exhausted, seen, count, pred, last)
        actions = allowed(state); a = choose(state, actions)
        if a == -1: return ABSENT, steps
        assert a in actions
        if a == 0: focus, value, cursor, exhausted = subject, ABSENT, -1, False
        elif a == 1:
            value = ABSENT
            if task != 1 and focus[0] == 0:
                for index in range(cursor+1, len(facts)):
                    src, rel, dst = facts[index]
                    if src == focus[1] and rel == relation:
                        cursor, value = index, (0, dst); break
            exhausted = value == ABSENT
            if exhausted: cursor = len(facts)
        elif a == 2: focus, value, cursor, exhausted = value, ABSENT, -1, False
        elif a == 3: seen.add(value)
        elif a == 4: count += 1
        elif a == 5: pred = int(focus == target)
        elif a == 6: output = focus
        elif a == 7: output = value
        elif a == 8: output = (1, pred)
        elif a == 9: output = (2, count)
        steps += 1; last = a
        if a >= 6: return output, steps
    return ABSENT, steps


def execute(q, program, choose):
    regs = [q['a'], q['b']]; work = 0
    for op in program:
        if op == 9: return regs[0], work
        if op == 0: regs[0] = q['a']
        elif op == 1: regs[0] = q['b']
        elif op == 2: regs[1] = regs[0]
        elif op == 3: regs.reverse()
        else:
            assert 4 <= op <= 8
            if op != 8 and regs[0][0] != 0: return ABSENT, work
            task = 1 if op == 8 else 2 if op >= 6 else 0
            output, steps = vm(task, regs[0], regs[1] if op == 8 else ABSENT, q['relations'][op % 2], q['facts'], choose)
            work += steps
            if output[0] != (1 if op == 8 else 2 if op >= 6 else 0): return ABSENT, work
            regs[0] = output
    return ABSENT, work


def question_gold(q):
    a, b, (r, s), facts = q['a'], q['b'], q['relations'], q['facts']
    family = q['cue']-71
    if family == 0:
        left = gold(2, a, ABSENT, r, facts); assert left[1] == q['size']
        return (1, int(left == gold(2, b, ABSENT, s, facts)))
    endpoint = gold(0, a, ABSENT, r, facts)
    count = gold(2, endpoint, ABSENT, s, facts)
    assert count[1] == q['size']
    if family == 1: return (1, int(count == b))
    if family == 2: return count
    assert family == 3
    return (1, int(count == gold(2, gold(0, b, ABSENT, r, facts), ABSENT, s, facts)))


def load_questions(path):
    result = {}
    for line in path.read_text().splitlines():
        r = line.split(); identifier = int(r[0]); v = list(map(int, r[2:]))
        size, cue, at, av, bt, bv, r0, r1, yt, yv, n = v[:11]
        assert len(v) == 11+3*n and identifier == len(result)+1
        q = dict(phase=r[1], size=size, cue=cue, a=(at, av), b=(bt, bv), relations=(r0, r1), answer=(yt, yv), facts=list(zip(v[11::3], v[12::3], v[13::3])))
        assert q['answer'] == question_gold(q)
        result[identifier] = q
    return result


def check(prefix, seed, profile):
    summary = list(map(int, Path(str(prefix)+'.summary.txt').read_text().split()))
    acquired, cue, correct, total, evaluations, first, search_steps, macro_states = summary
    questions = load_questions(Path(str(prefix)+'.questions.txt'))
    training = {i: q for i, q in questions.items() if q['phase'] == 'train'}
    repair = profile == 'repair'
    assert len(training) == (64 if profile in ('boolean', 'repair') else 32)
    assert total == len(training)+(40 if repair else 0)
    assert {q['cue'] for q in training.values()} == ({71, 72} if profile in ('boolean', 'repair') else {73})
    curve = [list(map(int, r.split())) for r in Path(str(prefix)+'.curve.tsv').read_text().splitlines()]
    if repair:
        attempts = [{k: int(v) for k, v in r.items()} for r in rows(Path(str(prefix)+'.attempts.tsv'))]
        assert [r['cue'] for r in attempts] == [0, 1, 2] and all(r['available'] for r in attempts)
        selected = next(i for i, r in enumerate(attempts) if r['cue'] == cue)
        assert attempts[selected]['evaluations'] == evaluations and attempts[selected]['steps'] == search_steps
        if acquired: assert all(r['correct'] < r['total'] for r in attempts[:selected])
        starts = [i for i, r in enumerate(curve) if r[0] == 0]+[len(curve)]
        assert len(starts) == 4
        for index, attempt in enumerate(attempts):
            part = curve[starts[index]:starts[index+1]]
            assert part[-1][1] == attempt['correct'] and all(r[2] == attempt['total'] and r[0] < attempt['evaluations'] for r in part)
        curve = curve[starts[selected]:starts[selected+1]]
    assert curve and curve[-1][1] == correct and all(r[2] == total and 0 <= r[0] < evaluations for r in curve)
    assert first == next((r[0] for r in curve if r[1] == total), -1)
    assert bool(acquired) == (correct == total) and search_steps > 0 and macro_states > 0
    prior, _ = audit(Path(str(prefix)+'.prior'))
    prior_cases = list(rows(Path(str(prefix)+'.prior.cases.tsv')))
    assert len(prior_cases) == (120 if repair else 80) and {int(r['task']) for r in prior_cases} == ({0, 1, 2} if repair else {0, 1})
    neural = collections.defaultdict(dict)

    def record(model, state, action, actions):
        pair = (state, tuple(actions))
        assert pair not in neural[model] or neural[model][pair] == action
        neural[model][pair] = action

    for r in rows(Path(str(prefix)+'.prior.decisions.tsv')):
        state, action = int(r['state']), int(r['action']); record('initial', state, action, allowed(state))
    result = {'acquired': bool(acquired), 'cue': cue, 'correct': correct, 'total': total, 'evaluations': evaluations,
              'first_solution': first, 'search_microsteps': search_steps, 'macro_states': macro_states, 'prior': prior}
    if repair:
        for r in prior_cases:
            if int(r['task']) == 2:
                facts = [tuple(map(int, f.split(','))) for f in r['facts'].rstrip(';').split(';') if f]
                assert len(facts) == len(set(facts)), 'duplicate-free warm start was not declared truthfully'
        warm, _ = audit(Path(str(prefix)+'.warm'))
        for r in rows(Path(str(prefix)+'.warm.decisions.tsv')):
            state, action = int(r['state']), int(r['action']); record('initial', state, action, allowed(state))
        result.update(attempts=attempts, all_candidate_evaluations=sum(r['evaluations'] for r in attempts),
                      all_candidate_microsteps=sum(r['steps'] for r in attempts), warm=warm)
    if acquired:
        rules = policy(Path(str(prefix)+'.policy.txt')); assert cue == 2
        derived = {}
        for line in Path(str(prefix)+'.derived_samples.txt').read_text().splitlines():
            state, action, n, *actions = map(int, line.split()); assert len(actions) == n and actions == allowed(state)
            assert state not in derived and rules[state] == action; derived[state] = (action, tuple(actions))
        assert set(derived) == set(rules)
        programs = {}; program_rows = []
        for line in Path(str(prefix)+'.programs.txt').read_text().splitlines():
            model, cue_text, found, states, n, *ops = line.split(); cue_id, found, states, n = map(int, (cue_text, found, states, n)); p = list(map(int, ops))
            assert bool(p) == bool(found) and len(p) == n <= 10 and states <= 50001 and all(0 <= a <= 9 for a in p)
            assert (model, cue_id) not in programs; programs[model, cue_id] = p
            program_rows.append({'model': model, 'cue': cue_id, 'found': found, 'states': states, 'program': p})
        decisions = collections.defaultdict(list)
        for line in Path(str(prefix)+'.micro.txt').read_text().splitlines():
            cid, model, step, state, action = line.split(); pair = (int(cid), model)
            assert int(step) == len(decisions[pair]); decisions[pair].append((int(state), int(action)))
        scores = collections.defaultdict(collections.Counter); observed = {}; answered = set()
        for line in Path(str(prefix)+'.answers.txt').read_text().splitlines():
            cid, model, typ, answer, work = line.split(); cid, typ, answer, work = map(int, (cid, typ, answer, work)); q = questions[cid]
            pair = (cid, model); assert pair not in answered; answered.add(pair)
            trace = iter(decisions.pop(pair, [])); consumed = 0

            def choose(state, actions):
                nonlocal consumed
                logged, action = next(trace); assert logged == state; consumed += 1
                if model == 'symbolic' and state & (1 << 2):
                    assert action == rules.get(state, -1)
                    if q['phase'] == 'train': observed[state] = (action, tuple(actions))
                else: record(model, state, action, actions)
                return action

            output, computed_work = execute(q, programs[model, q['cue']], choose)
            assert next(trace, None) is None and output == (typ, answer) and work == computed_work
            for name in (f'{model}/{q["phase"]}', f'{model}/{q["phase"]}/family{q["cue"]-71}', f'{model}/{q["phase"]}/size{q["size"]}'):
                m = scores[name]; m.update(total=1, correct=int(output == q['answer']), microsteps=work)
                m['max_microsteps'] = max(m['max_microsteps'], work)
        if repair:
            for r in prior_cases:
                if int(r['task']) != cue: continue
                facts = [tuple(map(int, f.split(','))) for f in r['facts'].rstrip(';').split(';') if f]
                subject, target = (int(r['subject_type']), int(r['subject'])), (int(r['target_type']), int(r['target']))
                def repaired_choice(state, actions):
                    action = rules.get(state, -1)
                    assert action in actions; observed[state] = (action, tuple(actions)); return action
                output, _ = vm(cue, subject, target, int(r['relation']), facts, repaired_choice)
                assert output == (int(r['gold_type']), int(r['gold']))
        assert not decisions and observed == derived
        models = ('symbolic', 'rehearsal', 'no_rehearsal', 'unacquired')
        assert set(programs) == {(m, c) for m in models for c in range(71, 75)}
        assert answered == {(i, m) for i, q in questions.items() if q['phase'] != 'transfer_train' for m in models}
        primitive = {}
        for model in models:
            primitive[model], _ = audit(Path(str(prefix)+'.'+model))
            for row in rows(Path(str(prefix)+'.'+model+'.decisions.tsv')):
                state, action = int(row['state']), int(row['action'])
                if model == 'symbolic' and state & (1 << 2): assert action == rules.get(state, -1)
                else: record(model, state, action, allowed(state))
        for model in ('symbolic', 'unacquired'):
            assert Path(str(prefix)+f'.{model}.weights.txt').read_bytes() == Path(str(prefix)+'.initial.weights.txt').read_bytes()
        for model in ('rehearsal', 'no_rehearsal'):
            assert Path(str(prefix)+f'.{model}.weights.txt').read_bytes() != Path(str(prefix)+'.initial.weights.txt').read_bytes()
        result.update(scores=dict(scores), primitives=primitive, programs=program_rows, derived_states=len(derived))
    else:
        assert all(q['phase'] == 'train' for q in questions.values())
    choices = {}
    for model, samples in neural.items():
        text = ''.join(f'{state} {a} {len(actions)} '+' '.join(map(str, actions))+'\n' for (state, actions), a in sorted(samples.items()))
        p = Path(str(prefix)+f'.{model}.weights.txt')
        values = list(map(float, p.read_text().split())); assert len(values) == 15360 and all(map(math.isfinite, values))
        verified = subprocess.run(['build-wsl/verify_local_choices', str(seed), str(p)], input=text, text=True, capture_output=True, check=True)
        choices[model] = len(samples)
    result['frozen_neural_states_verified'] = choices
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/composed_credit')); p.add_argument('--seeds', type=int, default=3); p.add_argument('--profiles', nargs='+', default=['boolean', 'numeric']); a = p.parse_args()
    result = {}
    for profile in a.profiles:
        for seed in range(a.seeds):
            r = check(a.root/profile/str(seed), seed, profile); result[f'{profile}/{seed}'] = r
            print(profile, seed, 'acquired', r['acquired'], 'training', r['correct'], '/', r['total'],
                  'test', r.get('scores', {}).get('rehearsal/test'), flush=True)
    paths = [Path(x) for x in ['cpp/apps/composed_credit_probe.cpp','cpp/apps/verify_local_choices.cpp','cpp/apps/primitive_discovery_support.h','python/check_composed_credit.py','python/check_primitive_discovery.py','scripts/composed_credit_seed.sh']]
    paths += list(Path('cpp/include/laminar').glob('*.h'))
    paths += [x for x in a.root.rglob('*') if x.is_file() and x.name != 'audit.json']
    for manifest in a.root.glob('*source_manifest.json'):
        for name, expected in json.loads(manifest.read_text()).items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, 'archived source changed'
    hashes = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    (a.root/'audit.json').write_text(json.dumps({'runs': result, 'sha256': hashes}, indent=2)+'\n')
    print('independent composed semantics, VM traces, derived targets and frozen neural replay passed')
