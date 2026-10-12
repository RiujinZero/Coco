"""Verify compatibility of the frozen combined trainer build."""
import json
from pathlib import Path
from capacity_experiment import checksum


snapshot = Path('logs/learner_followup_build_v1')
binary = Path('build-wsl/laminar_train_learner_followup_v1')
canonical = Path('logs/canonical_learner_followup_v1')
audit = json.loads((canonical/'audit.json').read_text())
assert audit['sha256'][str(binary)] == checksum(binary)
assert audit['trace_and_weights_identical'] and audit['coupled_credit_correction_and_resume']
assert audit['training_episodes'] == 201 and audit['frozen_episodes'] == 2 and audit['negative_checks'] == 6
suffixes = ('final.weights', 'final.primitive_weights', 'stage1.weights', 'stage1.primitive_weights',
            'stage1.probe_primitive_weights', 'stage1.values', 'config.json')
parity = {suffix: checksum(canonical/'full'/f'learner.{suffix}') ==
          checksum(Path('logs/canonical_handoff_regression/full')/f'learner.{suffix}') for suffix in suffixes}
parity['trace'] = checksum(canonical/'full/trace.txt') == checksum(Path('logs/canonical_handoff_regression/full/trace.txt'))
assert all(parity.values()), parity
(canonical/'default_parity.json').write_text(json.dumps(parity, indent=2)+'\n')
smoke_path = Path('logs/repair_bank_profile_v1/audit.json')
smoke = json.loads(smoke_path.read_text())
assert smoke['passed'] and smoke['independently_audited'] and smoke['binary_sha256'] == checksum(binary)
inputs = [*snapshot.joinpath('cpp').rglob('*.h'), snapshot/'cpp/apps/laminar_train.cpp', snapshot/'cpp/src/organism.cpp']
current_matches = {str(p.relative_to(snapshot)): checksum(p) == checksum(p.relative_to(snapshot)) for p in inputs}
assert all(current_matches.values()), 'A compiled input changed after the frozen build; preserve the result and rebuild under a new name'
manifest = {str(p.relative_to(snapshot)): checksum(p) for p in snapshot.rglob('*')
            if p.is_file() and p.suffix in ('.h', '.cpp', '.py')}
(snapshot/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
record = {'scope': 'Combined trainer build compatibility and opt-in repair-bank profile wiring. Component capability evidence remains separate.',
          'binary': str(binary), 'binary_sha256': checksum(binary), 'frozen_source': str(snapshot),
          'compile_flags': '-std=c++20 -O3 -march=native -DNDEBUG',
          'current_compiled_inputs_match_snapshot': current_matches,
          'source_manifest_sha256': checksum(snapshot/'source_manifest.json'),
          'canonical': {'training_episodes': 201, 'frozen_episodes': 2, 'exact_resume_and_correction': True,
                        'rejection_checks': 6, 'handoff_artifact_parity': parity},
          'bank_profile': {'independently_audited': True, 'training_episodes': 8, 'frozen_episodes': 8,
                           'not_a_successive_repair_campaign': True},
          'evidence_sha256': {str(p): checksum(p) for p in (canonical/'audit.json', canonical/'default_parity.json', smoke_path)},
          'recommended_profile_and_normal_binary_unchanged': True, 'full_capability_gates_closed': []}
target = Path('docs/LEARNER_FOLLOWUP_BUILD_VALIDATION.json')
target.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({'record': str(target), 'canonical_parity_artifacts': len(parity), 'bank_profile_audited': True}, indent=2))
