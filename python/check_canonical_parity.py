"""Compare canonical training against an archived, fully audited control run."""
import argparse
import hashlib
import json
from pathlib import Path
from check_composed_control import check


def audit(reference, candidate):
    result = check(candidate)
    names = ['input.txt', 'environment.txt', 'worlds.txt', 'learned_progress.txt',
             'learned_progress.acquisition.tsv', 'learned_progress.final.weights']
    for path in reference.glob('learned_progress.*'):
        if any(path.name.endswith(s) for s in ('.primitive_weights', '.probe_primitive_weights', '.weights', '.values')):
            names.append(path.name)
    for name in sorted(set(names)):
        assert (reference/name).read_bytes() == (candidate/name).read_bytes(), f'canonical parity failed: {name}'
    config = json.loads((candidate/'learned_progress.config.json').read_text())
    seed = json.loads((candidate/'oracle.json').read_text())['seed']
    assert config['profile'] == 'coco_continual_v1' and config['seed'] == seed
    result['canonical_parity'] = {'reference': str(reference), 'byte_identical_files': sorted(set(names)), 'config': config}
    result['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in candidate.iterdir() if p.is_file() and p.name != 'canonical_audit.json'}
    (candidate/'canonical_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Canonical full-run parity: {len(set(names))} files byte-identical; independent control and primitive audits passed.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('reference', type=Path); p.add_argument('candidate', type=Path)
    a = p.parse_args(); audit(a.reference, a.candidate)
