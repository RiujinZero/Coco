"""Collect completed credit evidence without broadening its capability claim."""
import argparse
import json
from pathlib import Path
from capacity_experiment import checksum


def finalize(args):
    acceptance = json.loads((args.screen/'acceptance.json').read_text())
    assert acceptance['complete'] and acceptance['all_audited'] and acceptance['candidate_pass']
    results = json.loads((args.screen/'summary.json').read_text())
    totals = {}
    for profile in ('controller50k', 'controller50k_return', 'controller50k_trace'):
        runs = [r for r in results if r['profile'] == profile]
        assert sorted(r['seed'] for r in runs) == [0, 1, 2]
        totals[profile] = {stage: [sum(r['control']['metrics'][stage][k] for r in runs) for k in ('correct', 'total')]
                           for stage in ('stage4', 'recovery', 'longer', 'interrupted', 'interrupted/children')}
        totals[profile]['primitives'] = [sum(r['primitive_scores'][f'checkpoint/task{cue}/disabled-1'][k]
                                            for r in runs for cue in range(3)) for k in ('correct', 'n')]
    contract_path = Path('logs/greedy_trace_credit_contract_v1/validation.json')
    contracts = json.loads(contract_path.read_text())
    assert all(r['exit_code'] == 0 and checksum(Path(r['binary'])) == r['sha256'] for r in contracts['results'])
    phase_contract_path = Path('logs/credit_phase_contract_v1/validation.json')
    phase_contracts = json.loads(phase_contract_path.read_text())
    assert all(r['exit_code'] == 0 and checksum(Path(r['binary'])) == r['sha256'] for r in phase_contracts['results'])
    exact_path = Path('logs/greedy_trace_exact_suspension_v1/validation.json')
    exact = json.loads(exact_path.read_text())
    assert exact['exit_code'] == 0 and checksum(exact_path.parent/'test_goal_suspension_greedy.cpp') == exact['source_sha256']
    phase = json.loads((args.phase/'audit.json').read_text())
    assert phase['passed']
    whole = json.loads(Path('logs/control_ablations_v3/audit.json').read_text())
    assert whole['all_independently_audited'] and whole['canonical_arms'] == 8
    canonical = json.loads(Path('logs/canonical_credit_controls_v2/audit.json').read_text())
    assert canonical['trace_and_weights_identical'] and canonical['coupled_credit_correction_and_resume']
    suffixes = ('final.weights', 'final.primitive_weights', 'stage1.weights', 'stage1.primitive_weights',
                'stage1.probe_primitive_weights', 'stage1.values', 'config.json')
    baseline_parity = {suffix: checksum(Path('logs/canonical_credit_controls_v2/full')/f'learner.{suffix}') ==
                       checksum(Path('logs/canonical_handoff_regression/full')/f'learner.{suffix}') for suffix in suffixes}
    baseline_parity['trace'] = checksum(Path('logs/canonical_credit_controls_v2/full/trace.txt')) == checksum(Path('logs/canonical_handoff_regression/full/trace.txt'))
    assert all(baseline_parity.values())
    preflight = json.loads(Path('logs/long_credit_preflight_v1/acceptance.json').read_text())
    assert preflight['complete'] and preflight['audited'] and not preflight['credit_gate_established']
    paths = [args.screen/'plan.json', args.screen/'summary.json', args.screen/'acceptance.json',
             args.screen/'source_manifest.json', contract_path, phase_contract_path, exact_path, args.phase/'audit.json', args.phase/'plan.json',
             Path('logs/control_ablations_v3/audit.json'), Path('logs/canonical_credit_controls_v2/audit.json'),
             Path('logs/long_credit_preflight_v1/acceptance.json'), Path('logs/long_credit_preflight_v1/plan.json')]
    record = {'scope': 'Delayed propagation contracts plus matched short-task/host-interruption retention; not closure of the long-credit gate.',
              'three_seed_totals': totals, 'candidate_passed_prespecified_screen': True,
              'default_screen_parity': acceptance['default_parity'], 'canonical_handoff_parity': baseline_parity,
              'contracts': contracts, 'original_suspension_contract_with_greedy_credit': exact,
              'phase_contracts': phase_contracts, 'phase_controls': phase, 'whole_stream_control_wiring': whole,
              'canonical_replay_regression': {'training_episodes': 201, 'frozen_episodes': 2,
                                             'exact_replay_and_coupled_correction': True, 'rejection_checks': 6},
              'preflight': {'both_arms_already_solve_lengths': [4, 5, 6, 7, 8],
                            'seven_and_eight_exceed_training_decision_limit': True,
                            'eight_effect_success_return_worse_than_immediate_failure': True,
                            'supports_missing_strategy_acquisition_claim': False},
              'preserved_harness_failure': 'logs/control_ablations_v2: unused private cases absent from filtered oracle; corrected generator rerun in v3.',
              'binary_source_snapshots': {'build-wsl/laminar_train_greedy_trace_v1': 'logs/greedy_trace_build_v1',
                                         'build-wsl/laminar_train_credit_phase_v3': 'logs/credit_phase_build_v3'},
              'evidence_sha256': {str(path): checksum(path) for path in paths},
              'remaining': ['Acquisition of initially missing longer causal strategies', 'Autonomous goal/subgoal and interruption choices',
                            'Changed task/cost adaptation with matched post-bootstrap controls', 'Combined reader/growth/repair continuation']}
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'record': str(args.output), 'candidate': totals['controller50k_trace'], 'full_credit_gate_closed': False}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--screen', type=Path, default=Path('logs/greedy_trace_screen_v1'))
    p.add_argument('--phase', type=Path, default=Path('logs/credit_phase_controls_v3'))
    p.add_argument('--output', type=Path, default=Path('docs/GREEDY_TRACE_VALIDATION.json'))
    finalize(p.parse_args())
