"""Reformat verified raw lessons for the continuing composed learner."""
import argparse
import csv
from pathlib import Path
from check_composed_credit import load_questions


def build(source, destination):
    lines = []
    with Path(str(source)+'.prior.cases.tsv').open() as f:
        for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
            facts = [list(map(int, x.split(','))) for x in row['facts'].strip(';').split(';') if x]
            fields = [int(row[k]) for k in ['task', 'relation', 'subject_type', 'subject', 'target_type', 'target', 'gold_type', 'gold']]
            assert fields[0] in (0, 1)
            lines.append('P prior'+str(i).zfill(5)+' '+' '.join(map(str, fields+[len(facts)]+[v for f in facts for v in f])))
    for i, q in load_questions(Path(str(source)+'.questions.txt')).items():
        if q['phase'] == 'train' or (q['phase'] == 'transfer_train' and q['cue'] == 73):
            fields = [q['cue'], *q['a'], *q['b'], *q['relations'], *q['answer'], len(q['facts']), *[v for f in q['facts'] for v in f]]
            lines.append(('Q' if q['phase'] == 'train' else 'L')+' q'+str(i).zfill(5)+' '+' '.join(map(str, fields)))
    assert len(lines) == 176
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, default=Path('logs/composed_credit/boolean/1')); p.add_argument('--out', type=Path, default=Path('logs/composed_lifecycle/input.txt')); a = p.parse_args()
    build(a.source, a.out)
