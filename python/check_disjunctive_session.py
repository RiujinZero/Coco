"""Independently read the actual continuing-session rule-union snapshot."""
import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from check_concept_growth import equivalent
from check_relational_session import language_answer, parse_snapshot


def audit(binary, root):
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root/'session.txt'
    process = subprocess.run([str(binary), str(snapshot)], capture_output=True, text=True)
    (root/'stdout.txt').write_text(process.stdout)
    (root/'stderr.txt').write_text(process.stderr)
    assert process.returncode == 0, process.stderr
    rows = [shlex.split(line) for line in snapshot.read_text().splitlines()]
    model, end = parse_snapshot(rows, 0)
    assert end == len(rows) and model['ready']
    learned = model['concepts']['permitted']['expanded']
    assert len(learned) == 1 and equivalent(learned[0], {'any': [(2, [('red', (0, 1))]), (2, [('blue', (0, 1))])]})
    universe = [('red', ['x', 'z']), ('blue', ['x', 'z']), ('red', ['z', 'x']),
                ('blue', ['z', 'x']), ('red', ['x', 'other']), ('blue', ['other', 'z']), ('edge', ['x', 'z'])]
    checked = 0
    for mask in range(128):
        for complete in (False, True):
            world = {'facts': [atom for bit, atom in enumerate(universe) if mask & (1 << bit)], 'complete': complete}
            expected = 1 if mask & 3 else 0 if complete else -1
            assert language_answer(model, 'permit x z', world) == expected
            old_expected = 1 if mask & 64 else 0 if complete else -1
            assert language_answer(model, 'join x z', world) == old_expected
            checked += 2
    result = {'native_contract_passed': True, 'independent_language_cases': checked,
              'learned_union_semantically_equivalent': True, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'snapshot_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              'scope': 'Continuing symbolic session: new rule union, old language/procedure retention, action planning and evidence revision. '
                       'No autonomous control training or unrestricted representation-growth claim.'}
    (root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    audit(a.binary, a.root)
