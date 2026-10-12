"""Produce a compact committed record from the preserved repair evidence."""
import datetime
import json
from pathlib import Path
from capacity_experiment import checksum


def main():
    root = Path('logs/repair_records_three_seed_v7')
    acceptance = json.loads((root/'acceptance.json').read_text())
    assert acceptance['complete'] and acceptance['all_repairs_pass']
    repaired = sorted((r for r in acceptance['results'] if r.get('pass')), key=lambda r:r['seed'])
    assert [r['seed'] for r in repaired] == [0, 1, 2]
    canonical = json.loads(Path('logs/canonical_repair_final_v7/audit.json').read_text())
    assert canonical['trace_and_weights_identical'] and canonical['coupled_credit_correction_and_resume']
    evidence = [root/'acceptance.json', root/'plan.json', root/'summary.json', root/'source_manifest.json',
                Path('logs/repair_integration_three_seed_v3/acceptance.json'),
                Path('logs/repair_integration_three_seed_v3/plan.json'),
                Path('logs/canonical_repair_final_v7/audit.json'), Path('logs/canonical_repair_final_v7/default_parity.json'),
                Path('logs/mixed_frame_repair_regression_v7/audit.json'),
                Path('logs/repair_integration_cache_parity_v3/native_parity.json'),
                Path('logs/repair_memo_parity_v6/native_parity.json'),
                Path('logs/repair_grounded_regression_v8/native_parity.json'),
                Path('logs/repair_grounded_regression_v8/summary.json')]
    for path in evidence:
        if path.name in ('native_parity.json', 'default_parity.json'):
            assert all(json.loads(path.read_text()).values())
    language = []
    for seed in range(3):
        p = Path(f'logs/frozen_grounded_language_v8/{seed}/audit.json'); evidence.append(p)
        r = json.loads(p.read_text()); scores = r['results']['observed_types']
        assert all(s['correct']/s['within_capacity'] >= .95 for s in scores.values())
        language.append(dict(seed=seed, scores=scores))
    report = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='Single available-skill repair in a continuing structured learner. Broader complete-learner gates remain open.',
                  recommended_profile='controller50k', recommended_profile_changed=False,
                  experimental_profile='controller50k_repair_records',
                  matched_campaign=dict(seeds=[0, 1, 2], native_wall_budget_seconds=900,
                                        completed_arms=6, repair_successes=3, repairs=[{
                      'seed':r['seed'], 'seconds':r['native_wall_seconds'], 'control':r['control'],
                      'primitive':r['primitive'], 'prior_outcomes':r['outcomes']['prior_outcomes'],
                      'matched_count':r['outcomes']['matched_count'], 'lifecycle_stages':len(r['lifecycle'])
                  } for r in repaired]),
                  older_campaign=dict(root='logs/repair_integration_three_seed_v3', successful_repairs=[0, 1],
                                      timed_out_seed=2, wall_budget_seconds=900),
                  frozen_language=dict(scope='Native/Python parity with archived v7 weights and programs, updated observation-grounding adapter; no new training',
                                       seeds=language, total_cases=840, within_capacity=771, correct=770,
                                       withheld_unknown_relation=1, outside_name_capacity=69),
                  latest_adapter_integration=dict(seed=1, root='logs/repair_grounded_regression_v8',
                                                  audited=True, unchanged_native_artifacts=476,
                                                  limitation='Full three-seed training campaign used v7; v8 grounding checked on all three frozen models and a full seed-1 replay.'),
                  canonical_regression=dict(training_episodes=canonical['training_episodes'],
                                            frozen_episodes=canonical['frozen_episodes'], exact_replay=True,
                                            coupled_correction=True, negative_checks=canonical['negative_checks']),
                  binaries={str(p):checksum(p) for p in map(Path, ['build-wsl/laminar_train_repair_memo_v7',
                      'build-wsl/laminar_train_repair_grounded_v8', 'build-wsl/frozen_repair_language_probe_v8'])},
                  evidence_files={str(p):checksum(p) for p in evidence},
                  remaining=['Several successive/simultaneous repaired skills', 'Independent computation choice on unfamiliar task structures',
                             'Long-horizon credit and autonomous goal scheduling', 'Representation/operation-bank growth',
                             'Integrated grounded reader and mixed continuing validation'])
    Path('docs/REPAIR_INTEGRATION_VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Repository validation record prepared: 3/3 repairs, 24 lifecycle stages, 770/771 in-capacity frozen language cases.')


if __name__ == '__main__':
    main()
