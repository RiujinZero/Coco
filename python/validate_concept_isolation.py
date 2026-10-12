"""Sanitizer contracts for retaining known skills during concept growth."""
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path


def check(name):
    source = Path(f'cpp/tests/test_{name}.cpp')
    binary = Path(f'build-wsl/test_{name}_isolation')
    subprocess.run(['g++', '-std=c++20', '-O1', '-g', '-fsanitize=address,undefined',
                    '-fno-omit-frame-pointer', '-Icpp/include', str(source), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    root = Path('logs/concept_isolation'); root.mkdir(exist_ok=True, parents=True)
    (root/f'{name}.txt').write_text(result.stdout+result.stderr)
    record = {'returncode': result.returncode, 'output': result.stdout+result.stderr,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    print(name, record['returncode'], record['output'], flush=True)
    return name, record


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(check, ['concept_isolation', 'relational_session', 'computational_session', 'goal_suspension']))
    Path('logs/concept_isolation/contracts.json').write_text(json.dumps(results, indent=2)+'\n')
    assert all(r['returncode'] == 0 for r in results.values())
