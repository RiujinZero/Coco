"""Freeze and test goal-control protocol; this does not measure acquisition."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('logs/goal_control_protocol_v1'))
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('include', 'tests'):
        shutil.copytree(Path('cpp')/folder, root/'cpp'/folder)
    sources = {str(p.relative_to(root)): checksum(p) for p in root.joinpath('cpp').rglob('*') if p.is_file()}
    (root/'source_manifest.json').write_text(json.dumps(sources, indent=2)+'\n')
    runs = {}
    for mode, flags in [('optimized', ['-O2', '-DNDEBUG']),
                        ('sanitizer', ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'])]:
        binary = root/f'test_goal_control_{mode}'
        command = ['g++', '-std=c++20', *flags, '-I'+str(root/'cpp/include'),
                   str(root/'cpp/tests/test_goal_control.cpp'), '-o', str(binary)]
        (root/f'{mode}.command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (root/f'{mode}.compile.log').open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        environment = dict(os.environ)
        environment['ASAN_OPTIONS'] = 'detect_leaks=1:halt_on_error=1'
        environment['UBSAN_OPTIONS'] = 'halt_on_error=1:print_stacktrace=1'
        with (root/f'{mode}.stdout').open('w') as out, (root/f'{mode}.stderr').open('w') as err:
            subprocess.run([str(binary)], stdout=out, stderr=err, env=environment, check=True)
        runs[mode] = {'command': command, 'binary': str(binary), 'sha256': checksum(binary),
                      'stdout_sha256': checksum(root/f'{mode}.stdout'),
                      'stderr_sha256': checksum(root/f'{mode}.stderr'), 'passed': True}
        print(f'Goal protocol {mode}: passed', flush=True)
    compiled = ['cpp/include/laminar/goal_control.h', 'cpp/include/laminar/relational_control.h',
                'cpp/include/laminar/relational_session.h', 'cpp/include/laminar/relational_goal_preview.h',
                'cpp/tests/test_goal_control.cpp', 'cpp/tests/goal_control_fixture.h']
    current = {p: checksum(p) == sources[p] for p in compiled}
    assert all(current.values()), 'Relevant source changed during verification; preserve this run and rebuild a fresh snapshot'
    record = {'scope': 'Goal-control protocol, paid computation provenance and coupled replay; forced protocol choices, not learned acquisition.',
              'frozen_source': str(root), 'source_manifest_sha256': checksum(root/'source_manifest.json'),
              'current_relevant_sources_match': current, 'runs': runs,
              'evidence': ['policy-issued token switching and resume', 'derived child interrupt and resume',
                           'owned callbacks with public arrivals and isolated worlds',
                           'correct False query and deadlines', 'one terminal mission credit',
                           'coupled correction and withdrawal exact replay', 'frozen evaluation and five-output warm import',
                           'shared decision and motor budgets', 'immutable paid computation receipts and cost accounting',
                           'internal proposals do not set external arrival flag'],
              'full_capability_gates_closed': []}
    (root/'validation.json').write_text(json.dumps(record, indent=2)+'\n')
    Path('docs/GOAL_CONTROL_PROTOCOL_VALIDATION.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'validation': str(root/'validation.json')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
