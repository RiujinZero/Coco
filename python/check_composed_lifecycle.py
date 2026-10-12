"""Check raw evidence, checkpoint retention/rollback and actual neural behavior."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from check_primitive_discovery import audit, rows
from check_composed_credit import allowed
from composed_lifecycle_probe import build


def check(root, source):
    with tempfile.TemporaryDirectory() as tmp:
        regenerated = Path(tmp)/'input.txt'; build(source, regenerated)
        assert regenerated.read_bytes() == (root/'input.txt').read_bytes()
    stages = ('acquired', 'extended', 'contradiction', 'restored', 'withdrawn', 'final')
    weights = {s: root/f'1.{s}.weights.txt' for s in stages}
    for stage in ('extended', 'restored', 'final'):
        assert weights[stage].read_bytes() == weights['acquired'].read_bytes()
    assert weights['contradiction'].read_bytes() == weights['withdrawn'].read_bytes() == Path(str(source)+'.initial.weights.txt').read_bytes()
    assert weights['acquired'].read_bytes() != weights['withdrawn'].read_bytes()
    scores, _ = audit(root/'1.checkpoint'); choices = {}
    for r in rows(root/'1.checkpoint.decisions.tsv'):
        state, action = int(r['state']), int(r['action'])
        assert state not in choices or choices[state] == action; choices[state] = action
    commands = ''.join(f'{s} {a} {len(allowed(s))} '+' '.join(map(str, allowed(s)))+'\n' for s, a in sorted(choices.items()))
    subprocess.run(['build-wsl/verify_local_choices', '1', str(weights['final'])], input=commands, text=True, check=True)
    paths = [p for p in root.rglob('*') if p.is_file() and p.name != 'audit.json']
    paths += [Path(p) for p in ['cpp/apps/composed_lifecycle_probe.cpp', 'cpp/apps/primitive_checkpoint_probe.cpp', 'cpp/apps/verify_local_choices.cpp', 'python/composed_lifecycle_probe.py', 'python/check_composed_lifecycle.py', 'python/check_composed_credit.py', 'python/check_primitive_discovery.py', 'scripts/composed_lifecycle.sh', 'cpp/tests/test_composed_learner.cpp']]
    paths += list(Path('cpp/include/laminar').glob('*.h'))
    return {'primitive_scores': scores, 'frozen_neural_states': len(choices), 'checkpoint_sha256': {s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in weights.items()},
            'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/composed_lifecycle')); p.add_argument('--source', type=Path, default=Path('logs/composed_credit/boolean/1')); a = p.parse_args()
    r = check(a.root, a.source); (a.root/'audit.json').write_text(json.dumps(r, indent=2)+'\n')
    print('raw evidence, checkpoint identity/rollback, primitive VM and frozen neural replay passed')
