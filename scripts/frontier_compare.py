"""Check that two model_frontier runs (JSON lines, same prompts and modes) are identical: every field present in both
records except the wall time must be equal (outputs, log-probs, match statistics, event and proposal counts).

usage: python scripts/frontier_compare.py REFERENCE.jsonl NEW.jsonl
Used to verify that pmi=0 reproduces the runs made before the prompt-dependence judge existed.
"""
import json
import sys


def main(ref_path, new_path):
    ref = [json.loads(line) for line in open(ref_path, encoding='utf-8')]
    new = [json.loads(line) for line in open(new_path, encoding='utf-8')]
    if len(ref) != len(new):
        print('DIFFER: %d vs %d records' % (len(ref), len(new)))
        return 1
    diffs, keys = [], set()
    for a, b in zip(ref, new):
        common = [k for k in a if k in b and k != 'seconds']
        keys.update(common)
        diffs += [(a['prompt'], a['mode'], k, a[k], b[k]) for k in common if a[k] != b[k]]
    if diffs:
        print('DIFFER in %d fields, e.g.:' % len(diffs))
        for d in diffs[:12]:
            print('  prompt %r mode %s field %s: %r vs %r' % d)
        return 1
    print('IDENTICAL: %d records, %d shared fields compared (%s)' % (len(ref), len(keys), ', '.join(sorted(keys))))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
