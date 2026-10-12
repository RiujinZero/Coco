"""Actual canonical learner: partial concept withdrawal keeps unrelated evidence."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
from relational_session_probe import observed
from capacity_experiment import checksum


def check(root, binary):
    root.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in [*Path('cpp/include').rglob('*.h'), *Path('cpp/apps').glob('*.h'), Path('cpp/apps/laminar_train.cpp'), Path('cpp/src/organism.cpp'), Path(__file__)]:
        if path.is_absolute(): path = path.relative_to(Path.cwd())
        target = root/'source'/path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
        manifest[str(path)] = checksum(target)
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    source = Path('logs/composed_control_raw/1')
    metadata = json.loads((source/'oracle.json').read_text())
    source_rows = (source/'input.txt').read_text().splitlines()
    priors = [r for r in source_rows if r.startswith('P ')]
    raw = [r for r in source_rows if r.startswith(('M ', 'N ')) and r.split()[1].startswith('composed')]
    assert len(priors) == 80 and len(raw) == 64
    base = next(r for r in metadata['teaching'] if r['id'] == 'composed1')
    worlds, environment = [], []
    def trial(name):
        truth = observed(base['world']); a, b = base['world']['args']
        worlds.extend([f'W {name} {truth} 0 0 0 0 0', f'X {name} 0 0'])
        environment.append(f'Q {name} {base["kind"]} {a} {b} {base["r"]} {base["s"]}')
        return f'TRAIN {name} 0 0 -1 {truth} '+json.dumps(base['text'])+' 0'
    rows = priors+raw+[trial('before_concept')]
    rows += [r for r in source_rows if r.startswith('C ')]
    for yes in (0, 1):
        world = {'args': ['a', 'b'], 'complete': True, 'facts': [['left', ['a', 'x']], ['right', ['b', 'x']]]}
        if yes: world['facts'].append(['marked', ['a', 'x']])
        rows.append(f'M concept_use{yes} {yes} '+observed(world)+' "vex a with b via eligible and right" 2 eligible right')
    rows.append(trial('after_concept'))
    for name, content in (('input.txt', rows), ('worlds.txt', worlds), ('environment.txt', environment)):
        (root/name).write_text('\n'.join(content)+'\n')
    command = [str(binary), 'trainer=continual', 'profile=controller50k', 'seed=1', 'lifecycle=1',
               f'file={root/"input.txt"}', f'worlds={root/"worlds.txt"}', f'environment={root/"environment.txt"}', f'output={root/"learner"}']
    (root/'run.json').write_text(json.dumps({'command': command, 'binary_sha256': checksum(binary)}, indent=2)+'\n')
    with (root/'stdout.txt').open('w') as out, (root/'stderr.txt').open('w') as err:
        result = subprocess.run(command, stdout=out, stderr=err)
    assert result.returncode == 0, (root/'stderr.txt').read_text()
    with (root/'learner.lifecycle.tsv').open() as f: records = list(csv.DictReader(f, delimiter='\t'))
    names = ['before', 'contradiction', 'restored', 'concept_changed', 'concept_restored', 'credit_restored', 'withdrawn', 'final']
    assert [r['stage'] for r in records] == names
    before = (root/'learner.lifecycle.before.weights').read_bytes()
    for r in records:
        present = r['stage'] not in ('contradiction', 'withdrawn')
        assert int(r['ready']) == 1 and int(r['count_available']) == present
        assert (int(r['verified_answers']) > 0) == present
        if present: assert (root/f'learner.lifecycle.{r["stage"]}.weights').read_bytes() == before
        if r['stage'] == 'concept_changed': assert int(r['withheld_answers']) == 2
        elif present: assert int(r['withheld_answers']) == 0
    assert (root/'learner.lifecycle.contradiction.weights').read_bytes() == (root/'learner.lifecycle.withdrawn.weights').read_bytes() != before
    report = {'stages': records, 'command': command, 'binary_sha256': checksum(binary),
              'checks': 'relevant answers withheld, unrelated answers/primitive weights retained, corrections and withdrawal restored'}
    (root/'audit.json').write_text(json.dumps(report, indent=2)+'\n'); print(report['checks'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/mixed_frame_lifecycle'))
    p.add_argument('--binary', type=Path, default=Path('build-wsl/laminar_train_repair_v2'))
    a = p.parse_args(); check(a.root, a.binary)
