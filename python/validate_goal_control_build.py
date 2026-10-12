"""Freeze a fresh trainer and verify default compatibility after mission support.

The goal wrapper is separately tested; this check makes no goal-learning claim.
Run from the repository root under WSL. Existing run directories are immutable.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v1')
    args = parser.parse_args()
    if not args.version.isalnum():
        parser.error('version must be alphanumeric')
    root = Path('logs') / f'goal_control_build_{args.version}'
    binary = Path('build-wsl') / f'laminar_train_goal_control_{args.version}'
    canonical = Path('logs') / f'canonical_goal_control_{args.version}'
    if root.exists() or binary.exists() or canonical.exists():
        raise RuntimeError('Preserve existing evidence; choose a new version')
    root.mkdir(parents=True)
    for directory in ('cpp', 'python'):
        shutil.copytree(directory, root/directory, ignore=shutil.ignore_patterns('__pycache__'))
    manifest = {str(p.relative_to(root)): checksum(p) for p in root.rglob('*') if p.is_file()}
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    command = ['g++', '-std=c++20', '-O3', '-march=native', '-DNDEBUG', '-I'+str(root/'cpp/include'),
               str(root/'cpp/apps/laminar_train.cpp'), str(root/'cpp/src/organism.cpp'), '-o', str(binary)]
    (root/'compile_command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (root/'compile.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f'Frozen trainer built: {binary}', flush=True)
    subprocess.run([sys.executable, str(root/'python/check_canonical_trainer.py'), '--binary', str(binary),
                    '--out', str(canonical)], check=True)
    audit = json.loads((canonical/'audit.json').read_text())
    assert audit['training_episodes'] == 201 and audit['frozen_episodes'] == 2
    assert audit['trace_and_weights_identical'] and audit['coupled_credit_correction_and_resume']
    assert audit['negative_checks'] == 6 and audit['sha256'][str(binary)] == checksum(binary)
    baseline = Path('logs/canonical_handoff_regression/full')
    names = ['trace.txt'] + ['learner.'+s for s in ('final.weights', 'final.primitive_weights',
             'stage1.weights', 'stage1.primitive_weights', 'stage1.probe_primitive_weights',
             'stage1.values', 'config.json')]
    parity = {name: checksum(canonical/'full'/name) == checksum(baseline/name) for name in names}
    assert all(parity.values()), parity
    (canonical/'default_parity.json').write_text(json.dumps(parity, indent=2)+'\n')
    record = {'scope': 'Default trainer compatibility after templated mission-control substrate; no goal acquisition claim.',
              'binary': str(binary), 'binary_sha256': checksum(binary), 'frozen_source': str(root),
              'compile_command': command, 'source_manifest_sha256': checksum(root/'source_manifest.json'),
              'canonical': {'training_episodes': 201, 'frozen_episodes': 2, 'exact_resume_and_correction': True,
                            'rejection_checks': 6, 'handoff_artifact_parity': parity},
              'evidence_sha256': {str(p): checksum(p) for p in (canonical/'audit.json', canonical/'default_parity.json')},
              'normal_binary_unchanged': True, 'full_capability_gates_closed': []}
    target = Path('docs') / f'GOAL_CONTROL_BUILD_{args.version.upper()}_VALIDATION.json'
    target.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'validation': str(target), 'exact_parity_artifacts': len(parity)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
