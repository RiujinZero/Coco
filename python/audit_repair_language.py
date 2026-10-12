"""Held-out raw-language bindings executed by the continuing model's frozen readout.

No control policy is run here; this complements the native controlled trials.
All cases, including those outside the adapter's 32-name capacity, are counted.
"""
import argparse
import collections
import json
from pathlib import Path
from capacity_experiment import checksum
from check_composed_credit import load_questions
from learned_computation_audit import LearnedComputations


def audit(directory, questions):
    config = json.loads((directory/'learned_progress.config.json').read_text())
    shape = [config.get('skill_units', 512), config.get('skill_active', 16), config.get('skill_channels', 3)]
    models = LearnedComputations(directory, 'learned_progress', config['seed'], shape)
    key, record = max(models.records.items(), key=lambda item:int(item[1]['model']))
    model = models.rebuilt(*key, int(record['version']))
    results = collections.defaultdict(lambda:dict(total=0, within_capacity=0, correct=0, withheld=0,
                                                 outside_capacity=0, unknown_schema=0))
    for q in load_questions(questions).values():
        if q['phase'] != 'test' or q['cue'] not in (71, 72): continue
        numeric = q['cue'] == 72; family = 'typed_quantity' if numeric else 'entity_comparison'
        a = 'held'+str(q['a'][1]); b = str(q['b'][1]) if numeric else 'held'+str(q['b'][1])
        r, s = ['rel'+str(x) for x in q['relations']]
        text = f'{"anchor" if numeric else "vex"} {a} with {b} via {r} and {s}'
        world = dict(complete=True, args=[a, b], facts=[['rel'+str(rel), ['held'+str(x), 'held'+str(y)]] for x, rel, y in q['facts']])
        names = {x for _, vs in world['facts'] for x in vs} | {a} | (set() if numeric else {b})
        score = results[family]; score['total'] += 1
        answer = models.answer(model, text, world)
        if len(names) > 32:
            score['outside_capacity'] += 1
            assert answer == -1, 'adapter exceeded its declared name limit'
            continue
        score['within_capacity'] += 1
        score['unknown_schema'] += any(rel not in model[0]['schemas'] for rel in (r, s))
        score['correct'] += answer == q['answer'][1]; score['withheld'] += answer == -1
    report = dict(model=int(record['model']), results=dict(results), question_source=str(questions),
                  questions_sha256=checksum(questions), scope='Frozen continuing-model language execution; no controller rollout')
    (directory/'repair_language_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', type=Path); p.add_argument('questions', type=Path)
    a = p.parse_args(); print(json.dumps(audit(a.directory, a.questions), indent=2))
