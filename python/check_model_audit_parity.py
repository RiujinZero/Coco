"""The model audit must observe, not change, canonical learning and inference."""
import json
import shlex
import subprocess
from pathlib import Path
from capacity_experiment import checksum
from check_relational_control import audit


def check():
    source = Path('logs/canonical_model_audit_regression')
    root = Path('logs/model_audit_parity'); root.mkdir(exist_ok=False)
    commands = (source/'data/input.txt').read_text().splitlines()
    ids = {r.split()[1] for r in commands if r.startswith(('TRAIN ', 'TEST '))}
    teacher_ids = {r.split()[1] for r in commands if r.startswith(('P ', 'M ', 'N ', 'L ', 'T ', 'C '))}
    meta = json.loads((source/'data/oracle.json').read_text())
    meta['cases'] = [c for c in meta['cases'] if c['id'] in ids]
    meta['teaching'] = [r for r in meta['teaching'] if r['id'] in teacher_ids]
    meta['composed_budget'] = 100000
    (root/'oracle.json').write_text(json.dumps(meta)+'\n')
    (root/'input.txt').write_text('\n'.join(commands)+'\n')
    for name in ('worlds.txt', 'environment.txt'):
        rows = [r for r in (source/'data'/name).read_text().splitlines() if r.split()[0] in ('E', 'H') or r.split()[1] in ids]
        (root/name).write_text('\n'.join(rows)+'\n')
    binary = Path('build-wsl/laminar_train_repair_v4')
    command = [str(binary), 'trainer=continual', 'seed=1', 'audit_models=1', f'file={root/"input.txt"}',
               f'environment={root/"environment.txt"}', f'worlds={root/"worlds.txt"}', f'output={root/"learned_progress"}']
    with (root/'learned_progress.txt').open('w') as out, (root/'stderr.txt').open('w') as err:
        result = subprocess.run(command, stdout=out, stderr=err)
    assert result.returncode == 0, (root/'stderr.txt').read_text()
    pairs = [('trace.txt', 'learned_progress.txt')]
    pairs += [(f'learner.{suffix}', f'learned_progress.{suffix}') for suffix in
              ('final.weights', 'final.primitive_weights', 'stage1.weights', 'stage1.primitive_weights',
               'stage1.probe_primitive_weights', 'stage1.values', 'config.json')]
    for before, after in pairs: assert (source/'full'/before).read_bytes() == (root/after).read_bytes(), after
    report = audit(root, 'learned_progress')
    report.update(byte_identical_artifacts=[after for _, after in pairs], binary_sha256=checksum(binary))
    (root/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Model audit: 8 canonical artifacts unchanged; independently reconstructed every answer and public control input.')


if __name__ == '__main__': check()
