"""Measure support for stored preferences without supplying new policy targets."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from check_relational_control import policy


def diagnose(directory, mode):
    counts = collections.defaultdict(collections.Counter)
    returns = collections.defaultdict(lambda: collections.defaultdict(list))
    checking = False; episode = []; chosen = cost = None
    with (directory/f'{mode}.txt').open() as stream:
        for line in stream:
            r = line.split()
            if r[0] == 'CHECK': checking = True
            if checking:
                if r[0] == 'END': checking = False
                continue
            if r[0] == 'S': chosen, cost = int(r[4]), float(r[5])
            elif r[0] == 'O':
                obs = tuple(map(float, r[3:])); counts[obs][chosen] += 1; episode.append((obs, chosen, cost))
            elif r[0] == 'J':
                result = float(r[3])
                for i, (obs, act, paid) in enumerate(reversed(episode)):
                    result = (.99*result if i else result)-paid
                    returns[obs][act].append(result)
                episode = []
    assert not checking and not episode
    rows = [list(map(float, r.split())) for r in (directory/f'{mode}.stage4.values').read_text().splitlines()]
    _, scores = policy(directory/f'{mode}.stage4.weights')
    groups = collections.defaultdict(collections.Counter)
    examples = []
    for row in rows:
        obs, q = tuple(row[:32]), row[32:]
        assert obs in counts
        best = max(range(5), key=lambda a: q[a]); neural = max(range(5), key=lambda a: scores(obs)[a])
        categories = ['all']
        if obs[9]: categories.append('plan_available')
        if obs[10]: categories.append('search_exhausted')
        for category in categories:
            m = groups[category]
            m['states'] += 1; m['neural_matches'] += best == neural
            m['best_untried'] += counts[obs][best] == 0
            m[f'preference{best}'] += 1; m[f'neural{neural}'] += 1
            m['experienced_choices'] += sum(counts[obs].values())
        if obs[9] and best != 3:
            examples.append({'observation': obs, 'values': q, 'preference': best, 'neural': neural,
                             'action_counts': dict(counts[obs]),
                             'experienced_return_means': {a: sum(v)/len(v) for a, v in returns[obs].items()}})
    return {'groups': dict(groups), 'non_act_plan_preferences': examples}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', default='logs/computational_control'); p.add_argument('--out', default='logs/computational_readout/preference_support.json'); a = p.parse_args()
    root = Path(a.source); result = {}; hashes = {}
    for seed in range(3):
        for mode in ('learned', 'scaled'):
            result[f'{seed}/{mode}'] = diagnose(root/str(seed), mode)
            for suffix in ('txt', 'stage4.values', 'stage4.weights'):
                path = root/str(seed)/f'{mode}.{suffix}'; hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            print(seed, mode, result[f'{seed}/{mode}']['groups'], flush=True)
    Path(a.out).write_text(json.dumps({'runs': result, 'sha256': hashes}, indent=2)+'\n')
