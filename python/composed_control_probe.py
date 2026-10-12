"""Use raw composed answers, never direct count lessons, in the mixed learner."""
import argparse
import json
from pathlib import Path
from computational_control_probe import build as mixed
from check_composed_credit import load_questions
from relational_session_probe import observed


def build(seed, directory, episodes, budget, source):
    mixed(seed, directory, episodes)
    meta = json.loads((directory/'oracle.json').read_text())
    lines = (directory/'input.txt').read_text().splitlines()
    prior = [line for line in lines if line.startswith('P ') and int(line.split()[2]) != 2]
    other = [line for line in lines if not line.startswith('P ')]
    meta['teaching'] = [r for r in meta['teaching'] if r['type'] != 'P' or int(r['record'].split()[2]) != 2]
    lessons, records = [], []
    for identifier, q in load_questions(source/f'{seed}.questions.txt').items():
        if q['phase'] != 'train': continue
        a = 'raw'+str(q['a'][1])
        b = str(q['b'][1]) if q['b'][0] == 2 else 'raw'+str(q['b'][1])
        r, s = ['rel'+str(v) for v in q['relations']]
        numeric = q['b'][0] == 2
        text = f'{"anchor" if numeric else "vex"} {a} with {b} via {r} and {s}'
        world = {'args': [a, b], 'complete': True,
                 'facts': [['rel'+str(p), ['raw'+str(x), 'raw'+str(y)]] for x, p, y in q['facts']]}
        rid = f'composed{identifier}'; op = 'N' if numeric else 'M'
        quantities = [b] if numeric else []
        line = f'{op} {rid} {q["answer"][1]} '+observed(world)+' '+json.dumps(text)+f' 2 {r} {s}'
        if numeric: line += f' 1 {b}'
        lessons.append(line)
        records.append({'type': op, 'id': rid, 'kind': 3 if numeric else 0, 'family': 5 if numeric else 0,
                        'text': text, 'r': r, 's': s, 'quantities': quantities, 'gold': q['answer'][1],
                        'world': world, 'truth': world})
    assert len(prior) == 80 and len(lessons) == 64
    meta['teaching'] = [r for r in meta['teaching'] if r['type'] == 'P'] + records + [r for r in meta['teaching'] if r['type'] != 'P']
    meta['composed_budget'] = budget
    meta['composed_source'] = str(source/f'{seed}.questions.txt')
    (directory/'input.txt').write_text('\n'.join(prior+lessons+other)+'\n')
    (directory/'oracle.json').write_text(json.dumps(meta)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=Path('logs/composed_control'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--episodes', type=int, default=400)
    p.add_argument('--budget', type=int, default=100000); p.add_argument('--source', type=Path, default=Path('logs/composed_credit/boolean'))
    a = p.parse_args()
    for seed in a.seeds: build(seed, a.out/str(seed), a.episodes, a.budget, a.source)
    print('created mixed curriculum with 80 direct follow/equality priors and 64 composed final answers per seed')
