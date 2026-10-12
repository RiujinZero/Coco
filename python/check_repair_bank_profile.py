"""Small canonical-profile wiring check, separate from repair capability tests."""
import argparse
import json
import shutil
from pathlib import Path
from capacity_experiment import run, checksum


def check(args):
    args.root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.source, args.root/'data')
    profile = 'controller50k_repair_bank_records'
    result = run(args.root, 1, profile, args.binary, (), 300)
    (args.root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['audited'], 'Native profile or independent audit failed; preserve output'
    config = result['resolved_config']
    assert config['experimental_repair_bank'] and config['experimental_primitive_repair']
    assert config['observation_records'] == 'sorted_with_multiplicity'
    assert config['observed_relation_types'] and config['controller_parameters'] == 50013
    assert config['discovery_interleave_quantum'] == 1000
    assert result['control']['updates'] == 1024
    record = {'scope': 'Canonical bank-profile construction, eight training episodes and eight frozen evaluations; not a successive-repair capability campaign.',
              'passed': True, 'profile': profile, 'independently_audited': True,
              'binary_sha256': checksum(args.binary),
              'source_data_sha256': {str(p): checksum(p) for p in sorted(args.source.rglob('*')) if p.is_file()},
              'resolved_config': config, 'control': result['control']}
    (args.root/'audit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'passed': True, 'profile': profile, 'scope': record['scope']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('logs/control_ablations_v3/data'))
    check(parser.parse_args())
