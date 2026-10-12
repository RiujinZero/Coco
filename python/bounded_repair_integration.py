"""Instrumented continuation of a preserved stream, with a declared wall budget.

This pilot measures integration and retention; the archived duplicate-observation
stream cannot establish visible control-error repair (observations are sets).
"""
import argparse
import json
from pathlib import Path
from capacity_experiment import checksum, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--seconds', type=int, default=300)
    p.add_argument('--profile', default='controller50k_repair_interleaved')
    p.add_argument('--lifecycle', action='store_true')
    a = p.parse_args()
    if a.seconds <= 0:
        p.error('--seconds must be positive')
    a.root.mkdir(parents=True, exist_ok=False)
    data = a.root/'data'/str(a.seed)
    data.mkdir(parents=True)
    for name in ('input.txt', 'environment.txt', 'worlds.txt', 'oracle.json'):
        (data/name).write_bytes((a.source/name).read_bytes())
    files = [*Path('cpp/include').rglob('*.h'), *Path('cpp/apps').glob('*.h'),
             *Path('python').glob('*.py'), Path('cpp/apps/laminar_train.cpp'), Path('cpp/src/organism.cpp')]
    manifest = {}
    for source in files:
        target = a.root/'source'/source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        manifest[str(source)] = checksum(target)
    (a.root/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    plan = dict(seed=a.seed, profile=a.profile, native_wall_budget_seconds=a.seconds,
                source=str(a.source), binary_sha256=checksum(a.binary),
                scope=('Repeated observation records remain visible; relational truth counts distinct tuples.'
                       if a.profile.endswith('_records') else
                       'Preserved-stream cost/retention pilot; normalized duplicates do not demonstrate control error repair.'))
    (a.root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    extras = ['audit_models=1', 'search_progress=1']
    if a.lifecycle: extras.append('lifecycle=1')
    result = run(a.root, a.seed, a.profile, a.binary, extras, timeout_seconds=a.seconds)
    (a.root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['audited'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
