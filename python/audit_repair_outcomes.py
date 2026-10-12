"""Independent frozen-VM audit of earlier direct lessons and matched count cases."""
import argparse
import json
import subprocess
from pathlib import Path
from capacity_experiment import checksum
from check_composed_credit import vm
from check_primitive_discovery import gold, rows


def audit(directory):
    config = json.loads((directory/'learned_progress.config.json').read_text())
    shape = [config.get('skill_units', 512), config.get('skill_active', 16), config.get('skill_channels', 3)]
    def choices(stage):
        weights = directory/f'learned_progress.{stage}.primitive_weights'
        text = subprocess.check_output(['build-wsl/verify_local_choices_readout', str(config['seed']), str(weights), *map(str, shape), '--export'], text=True)
        policy = dict(tuple(map(int, line.split())) for line in text.splitlines())
        assert len(policy) == 3*64*3*11
        return lambda state, available: policy[state]
    before, after = choices('before_repair'), choices('final')
    prior_scores = {str(cue): dict(correct=0, total=0) for cue in range(3)}
    for line in (directory/'input.txt').read_text().splitlines():
        fields = line.split()
        if fields[0] != 'P': continue
        v = list(map(int, fields[2:])); cue, relation, st, sv, tt, tv, yt, yv, n = v[:9]
        facts = list(zip(v[9::3], v[10::3], v[11::3])); assert len(facts) == n
        expected = gold(cue, (st, sv), (tt, tv), relation, facts)
        assert expected == (yt, yv)
        outcome, _ = vm(cue, (st, sv), (tt, tv), relation, facts, after, budget=256)
        prior_scores[str(cue)]['total'] += 1
        prior_scores[str(cue)]['correct'] += outcome == expected
    matched = {stage: dict(correct=0, total=0) for stage in ('before', 'after')}
    for r in rows(directory/'checkpoint.cases.tsv'):
        if int(r['task']) != 2: continue
        subject = (int(r['subject_type']), int(r['subject']))
        target = (int(r['target_type']), int(r['target']))
        facts = [tuple(map(int, x.split(','))) for x in r['facts'].rstrip(';').split(';') if x]
        expected = gold(2, subject, target, int(r['relation']), facts)
        for stage, policy in (('before', before), ('after', after)):
            outcome, _ = vm(2, subject, target, int(r['relation']), facts, policy, budget=1024)
            matched[stage]['total'] += 1; matched[stage]['correct'] += outcome == expected
    report = dict(prior_outcomes=prior_scores, matched_count=matched,
                  weights_sha256={s: checksum(directory/f'learned_progress.{s}.primitive_weights') for s in ('before_repair', 'final')})
    (directory/'repair_outcomes_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', type=Path)
    print(json.dumps(audit(p.parse_args().directory), indent=2))
