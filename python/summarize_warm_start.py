"""Summarize completed audited pairs without dropping missing/failed runs."""
import argparse
import collections
import json
from pathlib import Path
from capacity_experiment import checksum


def summarize(warm, baseline):
    result = {'complete': True, 'pairs': [], 'pending': [], 'failed': []}
    for seed in range(3):
        for profile in ('coco_continual_v1', 'controller50k'):
            directory = warm/profile/str(seed); reference = baseline/profile/str(seed)
            if not (directory/'capacity_audit.json').exists() or not (reference/'capacity_audit.json').exists():
                result['pending'].append([profile, seed]); result['complete'] = False; continue
            a = json.loads((reference/'capacity_audit.json').read_text())
            b = json.loads((directory/'capacity_audit.json').read_text())
            if not a['audited'] or not b['audited']:
                result['failed'].append([profile, seed]); result['complete'] = False; continue
            assert (directory/'input.txt').read_text().splitlines()[1:] == (reference/'input.txt').read_text().splitlines()
            for name in ('environment.txt', 'worlds.txt'):
                assert (directory/name).read_bytes() == (reference/name).read_bytes()
            meta = json.loads((directory/'oracle.json').read_text())
            train = {c['id'] for c in meta['cases'] if c['training']}
            phases = {}; counts = collections.defaultdict(collections.Counter)
            for line in (directory/'learned_progress.txt').read_text().splitlines():
                r = line.split()
                if r[0] == 'G': phases[r[1]] = r[2]
                if len(r) < 2 or r[1] not in train: continue
                if r[0] == 'H':
                    assert phases[r[1]] != 'autonomous'; counts[phases[r[1]]]['hints'] += 1
                if r[0] == 'F':
                    counts[phases[r[1]]].update(episodes=1, correct=int(r[2]), decisions=int(r[6]))
            assert set(phases) == train and counts['autonomous']['episodes'] == 240
            pair = {'profile': profile, 'seed': seed, 'matched_worlds': True, 'training_phases': dict(counts),
                    'baseline': {}, 'warm': {}, 'audit_sha256': [checksum(reference/'capacity_audit.json'), checksum(directory/'capacity_audit.json')]}
            for key, audit in (('baseline', a), ('warm', b)):
                pair[key]['metrics'] = {stage: audit['control']['metrics'][stage] for stage in ('stage1', 'stage2', 'stage3', 'stage4', 'recovery', 'longer')}
                pair[key]['updates'] = audit['control']['updates']
                pair[key]['primitive_correct'] = sum(audit['primitive_scores'][f'checkpoint/task{cue}/disabled-1']['correct'] for cue in range(3))
                pair[key]['native_wall_seconds'] = audit['native_wall_seconds']
            result['pairs'].append(pair)
    result['warm_aggregate'] = {}
    result['warm_complete'] = True
    for profile in ('coco_continual_v1', 'controller50k'):
        aggregate = {'seeds': [], 'stage4': [0, 0], 'recovery': [0, 0], 'longer': [0, 0], 'primitives': [0, 0]}
        for seed in range(3):
            path = warm/profile/str(seed)/'capacity_audit.json'
            if not path.exists(): result['warm_complete'] = False; continue
            audit = json.loads(path.read_text())
            if not audit['audited']: result['warm_complete'] = False; continue
            aggregate['seeds'].append(seed)
            for stage in ('stage4', 'recovery', 'longer'):
                m = audit['control']['metrics'][stage]
                aggregate[stage][0] += m['correct']; aggregate[stage][1] += m['total']
            for cue in range(3):
                m = audit['primitive_scores'][f'checkpoint/task{cue}/disabled-1']
                aggregate['primitives'][0] += m['correct']; aggregate['primitives'][1] += m['n']
        result['warm_aggregate'][profile] = aggregate
    result['summary_script_sha256'] = checksum(Path(__file__))
    (warm/'paired_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'complete': result['complete'], 'completed_pairs': len(result['pairs']), 'pending': result['pending'],
                      'warm_complete': result['warm_complete'], 'warm_aggregate': result['warm_aggregate'],
                      'scores': [[p['profile'], p['seed'], {arm: {s: p[arm]['metrics'][s]['correct'] for s in ('stage4', 'recovery', 'longer')} for arm in ('baseline', 'warm')}]
                                 for p in result['pairs']]}), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--warm', type=Path, default=Path('logs/warm_start_screen'))
    p.add_argument('--baseline', type=Path, default=Path('logs/capacity_screen'))
    a = p.parse_args(); summarize(a.warm, a.baseline)
