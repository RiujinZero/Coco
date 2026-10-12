"""Declared underspecified warm start, followed by composed-outcome skill repair."""
import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from capacity_experiment import checksum
from check_composed_credit import check


def experiment(root, binary, seeds, proposals, quantum=0):
    root.mkdir(parents=True, exist_ok=False); (root/'repair').mkdir()
    sources = [*Path('cpp/include').rglob('*.h'), Path('cpp/apps/composed_credit_probe.cpp'),
               Path('cpp/apps/primitive_discovery_support.h'), Path('cpp/apps/verify_local_choices.cpp'),
               Path('python/check_composed_credit.py'), Path('python/check_primitive_discovery.py'), Path(__file__)]
    manifest = {}
    for path in sources:
        if path.is_absolute(): path = path.relative_to(Path.cwd())
        target = root/'source'/path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
        manifest[str(path)] = checksum(target)
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def run(seed):
        command = [str(binary), str(seed), str(root/'repair'), str(proposals), 'repair']
        if quantum: command.append(str(quantum))
        start = time.monotonic()
        with (root/f'{seed}.stdout.txt').open('w') as out, (root/f'{seed}.stderr.txt').open('w') as err:
            completed = subprocess.run(command, stdout=out, stderr=err)
        result = {'seed': seed, 'command': command, 'returncode': completed.returncode,
                  'native_wall_seconds': time.monotonic()-start, 'binary_sha256': checksum(binary), 'audited': False,
                  'proposal_quantum': quantum, 'maximum_proposals_per_slot': proposals}
        try:
            assert completed.returncode == 0
            result['audit'] = check(root/'repair'/str(seed), seed, 'repair'); result['audited'] = True
        except Exception as error:
            result['audit_error'] = repr(error)
        (root/f'{seed}.audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: result[k] for k in ('seed', 'returncode', 'audited', 'native_wall_seconds')}) , flush=True)
        return result

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, seed) for seed in seeds]):
            results.append(future.result()); (root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    assert all(r['audited'] for r in results), 'See preserved audit failure'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/primitive_repair'))
    p.add_argument('--binary', type=Path, default=Path('build-wsl/composed_credit_probe_repair_v2'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--proposals', type=int, default=5000)
    p.add_argument('--quantum', type=int, default=0)
    a = p.parse_args(); experiment(a.root, a.binary, a.seeds, a.proposals, a.quantum)
