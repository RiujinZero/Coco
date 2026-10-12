"""Native/Python parity for old and observed-predicate grounding on frozen models."""
import argparse
import collections
import copy
import csv
import io
import json
import subprocess
from pathlib import Path
from capacity_experiment import checksum
from check_composed_credit import load_questions
from computational_control_oracle import binding
from learned_computation_audit import LearnedComputations
from concept_growth_probe import encode_world


def check(directory, questions, binary, root):
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads((directory/'learned_progress.config.json').read_text())
    models = LearnedComputations(directory, 'learned_progress', config['seed'], [512, 16, 3])
    key, record = max(models.records.items(), key=lambda item:int(item[1]['model']))
    original = models.rebuilt(*key, int(record['version']))
    assert all(original[0]['primitives'][1:])
    cases, wire = [], []
    for serial, q in load_questions(questions).items():
        if q['phase'] != 'test' or q['cue'] not in (71, 72): continue
        numeric = q['cue'] == 72
        a = 'held'+str(q['a'][1]); b = str(q['b'][1]) if numeric else 'held'+str(q['b'][1])
        r, s = ['rel'+str(x) for x in q['relations']]
        text = f'{"anchor" if numeric else "vex"} {a} with {b} via {r} and {s}'
        world = dict(complete=True, args=[], facts=[['rel'+str(rel), ['held'+str(x), 'held'+str(y)]] for x, rel, y in q['facts']])
        names = {x for _, vs in world['facts'] for x in vs} | {a} | (set() if numeric else {b})
        frames = [f for k, f in original[0]['computations'].items() if binding(k, text) is not None]
        assert len(frames) == 1 and frames[0]['learned']
        program = frames[0]['program']
        wire.append(f'{serial} '+encode_world(world)+' '+json.dumps(text)+f' {len(program)} '+' '.join(map(str, program)))
        cases.append((serial, 'typed_quantity' if numeric else 'entity_comparison', text, world, q['answer'][1], len(names) <= 32))
    inputs = root/'cases.txt'; inputs.write_text('\n'.join(wire)+'\n')
    weights = directory/f'learned_progress.model_weights{record["weights"]}'
    report = dict(source_directory=str(directory), binary_sha256=checksum(binary), weights_sha256=checksum(weights),
                  questions_sha256=checksum(questions), cases_sha256=checksum(inputs), results={},
                  scope='Frozen native language adapter plus archived learned programs/weights; no training or controller rollout')
    for grounded in (False, True):
        model = copy.deepcopy(original); model[0]['observed_relation_types'] = grounded
        command = [str(binary), str(config['seed']), str(weights), str(directory/'input.txt'), str(inputs), str(int(grounded))]
        process = subprocess.run(command, capture_output=True, text=True)
        label = 'observed_types' if grounded else 'registered_types'
        (root/f'{label}.stdout.tsv').write_text(process.stdout); (root/f'{label}.stderr.txt').write_text(process.stderr)
        assert process.returncode == 0, process.stderr
        rows = list(csv.DictReader(io.StringIO(process.stdout), delimiter='\t')); assert len(rows) == len(cases)
        scores = collections.defaultdict(lambda:dict(total=0, within_capacity=0, correct=0, withheld=0, outside_capacity=0))
        for row, (serial, family, text, world, gold, inside) in zip(rows, cases):
            predicted = LearnedComputations.answer(model, text, world)
            assert int(row['case']) == serial and int(row['output']) == predicted, (serial, row, predicted)
            score = scores[family]; score['total'] += 1
            if inside:
                score['within_capacity'] += 1; score['correct'] += predicted == gold; score['withheld'] += predicted == -1
            else:
                score['outside_capacity'] += 1; assert predicted == -1
        report['results'][label] = dict(scores)
    source_files = [Path(__file__), Path('python/learned_computation_audit.py'), Path('cpp/apps/frozen_repair_language_probe.cpp'),
                    *Path('cpp/include').rglob('*.h')]
    report['source_sha256'] = {}
    for source in source_files:
        if source.is_absolute(): source = source.relative_to(Path.cwd())
        target = root/'source'/source; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
        report['source_sha256'][str(source)] = checksum(target)
    (root/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', type=Path); p.add_argument('questions', type=Path)
    p.add_argument('--root', required=True, type=Path); p.add_argument('--binary', required=True, type=Path)
    a = p.parse_args(); print(json.dumps(check(a.directory, a.questions, a.binary, a.root)['results'], indent=2))
