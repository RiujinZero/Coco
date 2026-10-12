"""Post-evaluation diagnosis only; private case labels never feed training."""
import argparse
import collections
import json
from pathlib import Path
from check_relational_control import policy


def diagnose(directory, stage='stage4'):
    meta = json.loads((directory/'oracle.json').read_text()); cases = {c['id']: c for c in meta['cases']}
    values = {}
    for row in (directory/f'learned_progress.{stage}.values').read_text().splitlines():
        v = list(map(float, row.split())); values[tuple(v[:32])] = v[32:]
    _, scores = policy(directory/f'learned_progress.{stage}.weights')
    training = collections.defaultdict(collections.Counter); teachers = collections.defaultdict(collections.Counter)
    states = collections.defaultdict(collections.Counter); failed = []
    selected = {}; hinted = set(); current_stage = None
    for row in (directory/'learned_progress.txt').read_text().splitlines():
        r = row.split()
        if r[0] == 'CHECK': current_stage = r[1]
        if r[0] == 'ENDCHECK': current_stage = None
        if r[0] == 'H': hinted.add((r[1], r[2]))
        if r[0] == 'S': selected[r[1], r[2]] = int(r[4])
        if r[0] == 'O':
            obs = tuple(map(float, r[3:])); cid, decision = r[1:3]
            if cases[cid]['training']:
                training[obs][selected[cid, decision]] += 1
                if (cid, decision) in hinted: teachers[obs][selected[cid, decision]] += 1
            elif current_stage == stage: states[cid][obs] += 1
        if r[0] == 'F' and current_stage == stage and r[2] == '0': failed.append(r[1])
    result = []
    for cid in failed:
        obs, repeats = states[cid].most_common(1)[0]; q = values.get(obs); v = scores(obs)
        result.append({'episode': cid, 'family': cases[cid]['family'], 'effect': cases[cid]['effect'],
                       'price': cases[cid]['price'], 'sensor_unavailable': cases[cid]['sensor_unavailable'],
                       'repeated_state': obs, 'repeats': repeats, 'neural_scores': v,
                       'neural_choice': max(range(5), key=v.__getitem__),
                       'experienced_values': q, 'table_choice': max(range(5), key=q.__getitem__) if q else None,
                       'training_choices_at_state': dict(training[obs]), 'teacher_choices_at_state': dict(teachers[obs])})
    (directory/f'{stage}_failure_diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{'episode': r['episode'], 'repeats': r['repeats'], 'neural': r['neural_choice'], 'table': r['table_choice'],
                       'training': r['training_choices_at_state'], 'teacher': r['teacher_choices_at_state']} for r in result]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', type=Path); p.add_argument('--stage', default='stage4')
    a = p.parse_args(); diagnose(a.directory, a.stage)
