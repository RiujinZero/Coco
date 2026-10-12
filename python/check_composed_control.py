"""Audit composed acquisition inside the actual continuing controlled session."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from check_relational_control import audit as control_audit
from check_primitive_discovery import audit as primitive_audit, rows
from check_composed_credit import allowed, load_questions


def check(directory, mode='learned_progress'):
    meta = json.loads((directory/'oracle.json').read_text()); seed = meta['seed']
    priors = [r for r in meta['teaching'] if r['type'] == 'P']
    assert len(priors) == 80 and {int(r['record'].split()[2]) for r in priors} == {0, 1}
    source = {f'composed{i}': q for i, q in load_questions(Path(meta['composed_source'])).items() if q['phase'] == 'train'}
    supplied = {r['id']: r for r in meta['teaching'] if r['id'] in source}
    assert set(supplied) == set(source) and len(source) == 64
    for rid, q in source.items():
        r = supplied[rid]
        assert r['gold'] == q['answer'][1]
        assert r['world']['facts'] == [['rel'+str(p), ['raw'+str(x), 'raw'+str(y)]] for x, p, y in q['facts']]
        assert r['world']['args'] == ['raw'+str(q['a'][1]), str(q['b'][1]) if q['b'][0] == 2 else 'raw'+str(q['b'][1])]
    control = control_audit(directory, mode)
    with (directory/f'{mode}.acquisition.tsv').open() as f:
        acquisition = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f, delimiter='\t')]
    for r in acquisition:
        assert r['acquired'] == int(r['correct'] == r['total'])
        assert r['cue'] == 2 and 0 <= r['first'] < r['evaluations'] if r['acquired'] else r['first'] == -1
        assert r['steps'] > 0
    result = {'control': control, 'acquisition': acquisition, 'direct_prior_tasks': [0, 1], 'raw_composed_lessons': len(source)}
    lifecycle = directory/f'{mode}.lifecycle.tsv'
    if lifecycle.exists():
        with lifecycle.open() as f: stages = list(csv.DictReader(f, delimiter='\t'))
        names = ['before', 'contradiction', 'restored', 'concept_changed', 'concept_restored', 'credit_restored', 'withdrawn', 'final']
        assert [r['stage'] for r in stages] == names
        assert all(int(b['version']) > int(a['version']) for a, b in zip(stages, stages[1:]))
        checkpoint = {s: directory/f'{mode}.lifecycle.{s}.weights' for s in names}
        learned = checkpoint['before'].read_bytes()
        direct = Path('logs/composed_credit/boolean')/f'{seed}.initial.weights.txt'
        for r in stages:
            s = r['stage']; isolated = 'withheld_answers' in r
            present = s not in (('contradiction', 'withdrawn') if isolated else ('contradiction', 'concept_changed', 'withdrawn'))
            assert int(r['count_available']) == present
            assert int(r['verified_answers']) > 0 if present else int(r['verified_answers']) == 0
            assert checkpoint[s].read_bytes() == (learned if present else direct.read_bytes())
            if isolated:
                assert int(r['ready']) == 1
                if s == 'concept_changed': assert int(r['withheld_answers']) > 0
                elif present: assert int(r['withheld_answers']) == 0
        assert checkpoint['before'].read_bytes() == (directory/f'{mode}.stage4.primitive_weights').read_bytes()
        primitive, _ = primitive_audit(directory/'checkpoint')
        choices = {}
        for r in rows(directory/'checkpoint.decisions.tsv'):
            state, action = int(r['state']), int(r['action'])
            assert state not in choices or choices[state] == action; choices[state] = action
        wire = ''.join(f'{s} {a} {len(allowed(s))} '+' '.join(map(str, allowed(s)))+'\n' for s, a in sorted(choices.items()))
        subprocess.run(['build-wsl/verify_local_choices', str(seed), str(checkpoint['final'])], input=wire, text=True, check=True)
        result.update(lifecycle=stages, primitive_scores=primitive, frozen_primitive_states=len(choices))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('logs/composed_control'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); a = p.parse_args()
    for seed in a.seeds:
        directory = a.root/str(seed); result = check(directory)
        files = [f for f in directory.rglob('*') if f.is_file() and f.name != 'composed_audit.json']
        files += list(Path('cpp/include/laminar').glob('*.h'))
        files += [Path(s) for s in ('cpp/apps/relational_control_probe.cpp', 'cpp/apps/composed_session_contracts.h', 'cpp/apps/control_probe_support.h', 'cpp/apps/primitive_discovery_support.h', 'cpp/apps/primitive_checkpoint_probe.cpp', 'cpp/apps/verify_local_choices.cpp', 'python/composed_control_probe.py', 'python/check_composed_control.py', 'python/check_relational_control.py', 'python/check_relational_session.py', 'python/computational_control_oracle.py', 'scripts/composed_control_seed.sh', 'cpp/tests/test_composed_session.cpp')]
        result['sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        (directory/'composed_audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(seed, {s: (result['control']['metrics'][s]['correct'], result['control']['metrics'][s]['total']) for s in ('stage4', 'recovery', 'longer')}, flush=True)
