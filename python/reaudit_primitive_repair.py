"""Re-audit preserved native results, including unsuccessful search candidates."""
import argparse
import json
from pathlib import Path
from check_composed_credit import check
from capacity_experiment import checksum


def reaudit(root):
    rows = json.loads((root/'summary.json').read_text())
    reports = []
    for result in sorted(rows, key=lambda r:r['seed']):
        result = dict(result); result['audit'] = check(root/'repair'/str(result['seed']), result['seed'], 'repair')
        result['audited'] = True; result.pop('audit_error', None)
        result['auditor_sha256'] = checksum(Path('python/check_composed_credit.py'))
        reports.append(result)
    (root/'reaudited_summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    print([(r['seed'], r['audit']['acquired'], r['audit'].get('scores', {}).get('rehearsal/test')) for r in reports])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/primitive_repair'))
    reaudit(p.parse_args().root)
