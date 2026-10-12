"""Re-audit completed native artifacts without repeating or overwriting runs."""
import argparse
import json
import subprocess
from pathlib import Path
from capacity_experiment import checksum
from check_relational_control import audit
from check_primitive_discovery import audit as primitives, rows
from check_composed_credit import allowed


def check(root):
    results = []
    for path in sorted(root.glob('controller*/[0-9]*/capacity_audit.json')):
        original = json.loads(path.read_text()); directory = path.parent; seed = original['seed']
        result = {'profile': original['profile'], 'seed': seed, 'audited': False, 'original_audit_sha256': checksum(path)}
        try:
            assert original['native_returncode'] == 0
            result['control'] = audit(directory, 'learned_progress')
            cfg = json.loads((directory/'learned_progress.config.json').read_text())
            shape = list(map(str, [cfg['skill_units'], cfg['skill_active'], cfg['skill_channels']]))
            for stage in ('before_repair', 'after_repair'):
                weights = directory/f'learned_progress.{stage}.primitive_weights'; prefix = directory/f'reaudited_{stage}'
                subprocess.run(['build-wsl/primitive_checkpoint_probe_capacity', str(seed), str(weights), str(prefix), *shape], check=True, capture_output=True, text=True)
                result[stage], _ = primitives(prefix)
                choices = {}
                for row in rows(Path(str(prefix)+'.decisions.tsv')):
                    state, action = int(row['state']), int(row['action'])
                    assert state not in choices or choices[state] == action; choices[state] = action
                wire = ''.join(f'{s} {a} {len(allowed(s))} '+' '.join(map(str, allowed(s)))+'\n' for s, a in choices.items())
                subprocess.run(['build-wsl/verify_local_choices_readout', str(seed), str(weights), *shape], input=wire, check=True, capture_output=True, text=True)
            result['audited'] = True
        except Exception as error: result['audit_error'] = repr(error)
        (directory/'reaudit.json').write_text(json.dumps(result, indent=2)+'\n'); results.append(result)
        print(json.dumps({'profile': result['profile'], 'seed': seed, 'audited': result['audited'], 'error': result.get('audit_error')}), flush=True)
    assert results
    (root/'reaudited_summary.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); a = p.parse_args(); check(a.root)
