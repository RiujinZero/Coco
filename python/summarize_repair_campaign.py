"""Apply the prespecified integration thresholds, retaining every negative arm."""
import argparse
import csv
import json
from pathlib import Path
from audit_repair_outcomes import audit as outcome_audit
from audit_repair_language import audit as language_audit
from capacity_experiment import checksum


def lifecycle(directory):
    prefix = directory/'learned_progress'
    with Path(str(prefix)+'.lifecycle.tsv').open() as stream:
        stages = list(csv.DictReader(stream, delimiter='\t'))
    expected = ['before', 'contradiction', 'restored', 'concept_changed',
                'concept_restored', 'credit_restored', 'withdrawn', 'final']
    assert [r['stage'] for r in stages] == expected
    repaired = Path(str(prefix)+'.lifecycle.before.weights').read_bytes()
    direct = Path(str(prefix)+'.before_repair.primitive_weights').read_bytes()
    assert repaired != direct
    for r in stages:
        assert int(r['ready']) == 1 and int(r['count_available']) == 1
        s = r['stage']; weights = Path(str(prefix)+f'.lifecycle.{s}.weights').read_bytes()
        assert weights == (direct if s in ('contradiction', 'withdrawn') else repaired)
        if s == 'concept_changed':
            assert int(r['withheld_answers']) > 0 and int(r['verified_answers']) > 0
        elif s in ('contradiction', 'withdrawn'):
            assert int(r['verified_answers']) == 0
        else:
            assert r['verified_answers'] == stages[0]['verified_answers'] and int(r['withheld_answers']) == 0
    return stages


def summarize(root):
    plan = json.loads((root/'plan.json').read_text())
    results = json.loads((root/'summary.json').read_text())
    expected = {(seed, profile) for seed in plan['seeds'] for profile in plan['profiles']}
    observed = {(r['seed'], r['profile']) for r in results}
    assert len(observed) == len(results) and observed <= expected
    report = dict(complete=observed == expected, all_repairs_pass=False, results=[],
                  plan_sha256=checksum(root/'plan.json'), native_summary_sha256=checksum(root/'summary.json'))
    report['auditor_sha256'] = {}
    for source in (Path(__file__), Path('python/audit_repair_outcomes.py'), Path('python/audit_repair_language.py'), Path('python/check_composed_credit.py'),
                   Path('python/check_primitive_discovery.py'), Path('cpp/apps/verify_local_choices.cpp')):
        target = root/'acceptance_source'/source.name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
        report['auditor_sha256'][source.name] = checksum(target)
    report['readout_binary_sha256'] = checksum(Path('build-wsl/verify_local_choices_readout'))
    for r in results:
        directory = root/r['profile']/str(r['seed'])
        entry = {k:r.get(k) for k in ('seed', 'profile', 'audited', 'timed_out', 'native_wall_seconds', 'audit_error')}
        repairing = '_repair_' in r['profile']
        if r['audited']:
            metrics = r['control']['metrics']
            scores = {k:[metrics[k]['correct'], metrics[k]['total']] for k in ('stage4', 'before_repair', 'after_repair', 'retained')}
            entry['control'] = scores
            try:
                entry['outcomes'] = outcome_audit(directory)
                metadata = json.loads((directory/'oracle.json').read_text())
                entry['held_out_language_diagnostic'] = language_audit(directory, Path(metadata['repair_composed_source']))
            except Exception as error:
                entry['acceptance_error'] = repr(error); entry['pass'] = False
                report['results'].append(entry); continue
            new = scores['after_repair']; old = scores['retained']; warm = scores['stage4']
            primitive = [r['primitive_scores'][f'checkpoint/task{cue}/disabled-1'] for cue in range(3)]
            entry['primitive'] = [sum(v['correct'] for v in primitive), sum(v['n'] for v in primitive)]
            if repairing:
                before = entry['outcomes']['matched_count']['before']; after = entry['outcomes']['matched_count']['after']
                priors = entry['outcomes']['prior_outcomes'].values()
                passed = (new[0]/new[1] >= plan['acceptance']['new_family_minimum']
                          and warm[0]/warm[1]-old[0]/old[1] <= plan['acceptance']['maximum_old_task_loss']
                          and entry['primitive'] == [324, 324]
                          and all(p['correct'] == p['total'] for p in priors)
                          and before['correct'] < before['total'] and after['correct'] == after['total'])
                if r['resolved_config'].get('observation_records'):
                    passed &= scores['before_repair'][0] < scores['before_repair'][1]
                if plan.get('repair_lifecycle'):
                    try:
                        entry['lifecycle'] = lifecycle(directory)
                    except Exception as error:
                        entry['acceptance_error'] = repr(error); passed = False
                entry['pass'] = bool(passed)
        elif repairing:
            entry['pass'] = False
        report['results'].append(entry)
    for seed in plan['seeds']:
        arms = [r for r in results if r['seed'] == seed]
        if len(arms) == len(plan['profiles']):
            assert all(r['inputs_sha256'] == arms[0]['inputs_sha256'] for r in arms), 'unmatched inputs'
    repairs = [r for r in report['results'] if '_repair_' in r['profile']]
    report['all_repairs_pass'] = report['complete'] and len(repairs) == len(plan['seeds']) and all(r.get('pass', False) for r in repairs)
    (root/'acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    r = summarize(p.parse_args().root)
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r['all_repairs_pass'] else 1)
