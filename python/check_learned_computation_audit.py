"""Check that the new audit reconstructs errors, rather than assuming gold."""
import json
import subprocess
from pathlib import Path
from capacity_experiment import checksum
from check_composed_credit import allowed
from check_primitive_discovery import rows
from learned_computation_audit import LearnedComputations


def check():
    source = Path('logs/primitive_repair_full/repair/1')
    verifier = Path('build-wsl/verify_local_choices_readout')
    policies = {}
    for name in ('initial', 'rehearsal'):
        weights = Path(str(source)+f'.{name}.weights.txt')
        output = subprocess.check_output([str(verifier), '1', str(weights), '--export'], text=True)
        policies[name] = dict(tuple(map(int, line.split())) for line in output.splitlines())
        assert len(policies[name]) == 6336
        trace = Path(str(source)+('.warm.decisions.tsv' if name == 'initial' else '.rehearsal.decisions.tsv'))
        for r in rows(trace): assert policies[name][int(r['state'])] == int(r['action'])
    frame = 'computed:vex $0 with $1 via %0 and %1'
    snapshot = {'computations': {frame: {'learned': True, 'program': [6, 3, 7, 8, 9]}},
                'concepts': {}, 'schemas': {'left', 'right'}, 'primitives': [120, 1, 1, 1]}
    world = {'complete': True, 'facts': [['left', ['a', 'x']], ['left', ['a', 'x']], ['right', ['b', 'y']], ['unary', ['a']]]}
    text = 'vex a with b via left and right'
    before = LearnedComputations.answer((snapshot, policies['initial']), text, world)
    after = LearnedComputations.answer((snapshot, policies['rehearsal']), text, world)
    assert before == 0 and after == 1, (before, after)
    assert LearnedComputations.answer((snapshot, policies['rehearsal']), text, dict(world, complete=False)) == -1
    changed = dict(snapshot, computations={frame: {'learned': False, 'program': []}})
    assert LearnedComputations.answer((changed, policies['rehearsal']), text, world) == -1
    state, action = next(iter(policies['rehearsal'].items())); actions = allowed(state)
    wrong = next(a for a in actions if a != action)
    wire = f'{state} {wrong} {len(actions)} '+' '.join(map(str, actions))+'\n'
    rejected = subprocess.run([str(verifier), '1', str(source)+'.rehearsal.weights.txt'], input=wire, text=True, capture_output=True)
    assert rejected.returncode != 0 and 'differs from saved neural weights' in rejected.stderr
    report = {'wrong_answer_reconstructed': before, 'repaired_answer_reconstructed': after,
              'frozen_states_per_policy': 6336, 'matches_archived_native_execution_choices': True,
              'incomplete_world_and_unlearned_program_withheld': True, 'tampered_choice_rejected': True,
              'sha256': {str(p): checksum(p) for p in (verifier, Path(__file__), Path('python/learned_computation_audit.py'))}}
    root = Path('logs/learned_computation_audit'); root.mkdir(exist_ok=True)
    (root/'unit.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Learned answer audit: wrong/repaired answers, archived neural choices and tamper rejection passed.')


if __name__ == '__main__': check()
