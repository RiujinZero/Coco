"""Real executable regression for the canonical schedule and replay checkpoints.

This uses direct primitive teaching to keep persistence tests inexpensive. The
separate full composed-control audit verifies answer-only primitive acquisition.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from computational_control_probe import build


def check(binary, directory):
    directory.mkdir(parents=True, exist_ok=True)
    data = directory/'data'
    build(1, data, 201)
    rows = (data/'input.txt').read_text().splitlines()
    boundary = rows.index('CHECK stage1')
    # Keep 201 real training episodes (cross both schedule boundaries), then
    # two frozen evaluation episodes, on the generator's original worlds.
    rows = rows[:boundary+3]+['ENDCHECK']
    (data/'input.txt').write_text('\n'.join(rows)+'\n')
    stop = [i+1 for i, line in enumerate(rows) if line.startswith('TRAIN ')][198]

    def run(name, extra=(), okay=True, curriculum=None, seed='1'):
        target = directory/name; target.mkdir(exist_ok=True)
        command = [str(binary), 'trainer=continual', f'file={curriculum or data/"input.txt"}',
                   f'environment={data/"environment.txt"}', f'worlds={data/"worlds.txt"}',
                   f'seed={seed}', f'output={target/"learner"}', *map(str, extra)]
        with (target/'trace.txt').open('w') as out:
            result = subprocess.run(command, stdout=out, stderr=subprocess.PIPE, text=True)
        (target/'stderr.txt').write_text(result.stderr)
        assert (result.returncode == 0) == okay, (name, result.returncode, result.stderr)
        return target, result.stderr

    full, _ = run('full', [f'checkpoint={directory/"full.replay"}'])
    partial, _ = run('partial', [f'checkpoint={directory/"partial.replay"}', f'stop_after={stop}'])
    resumed, message = run('resumed', [f'resume={directory/"partial.replay"}', f'checkpoint={directory/"resumed.replay"}'])
    assert 'Checkpoint replay verified' in message
    assert full.joinpath('trace.txt').read_bytes() == resumed.joinpath('trace.txt').read_bytes()
    assert b'CONSOLIDATE 200 ' in full.joinpath('trace.txt').read_bytes()
    for suffix in ['final.weights', 'final.primitive_weights', 'stage1.weights', 'stage1.primitive_weights', 'stage1.probe_primitive_weights', 'stage1.values', 'config.json']:
        assert (full/f'learner.{suffix}').read_bytes() == (resumed/f'learner.{suffix}').read_bytes(), suffix
    assert (directory/'full.replay').read_bytes() == (directory/'resumed.replay').read_bytes()
    assert (partial/'learner.final.weights').read_bytes() != (full/'learner.final.weights').read_bytes()
    resolved = json.loads((full/'learner.config.json').read_text())
    assert resolved['profile'] == 'coco_continual_v1' and resolved['discovery_proposals_per_attempt'] == 100000
    assert resolved['updates_per_episode'] == 128 and resolved['consolidation_passes'] == 64
    assert not resolved['training_preserves_search'] and resolved['evaluation_preserves_search']

    corrupt = directory/'corrupt.replay'
    corrupt.write_bytes((directory/'partial.replay').read_bytes()[:-12])
    _, err = run('truncated', [f'resume={corrupt}'], okay=False)
    assert 'checkpoint is incomplete, corrupt' in err
    modified = data/'modified.txt'; modified.write_text((data/'input.txt').read_text()+'\n')
    _, err = run('changed_input', [f'resume={directory/"partial.replay"}'], okay=False, curriculum=modified)
    assert 'different binary/config/data' in err
    _, err = run('unknown_setting', ['updates_per_episode=1'], okay=False)
    assert 'unknown or duplicate continual argument' in err
    _, err = run('bad_seed', okay=False, seed='-1')
    assert 'invalid seed' in err
    _, err = run('changed_seed', [f'resume={directory/"partial.replay"}'], okay=False, seed='2')
    assert 'different binary/config/data' in err

    # Coupled controller corrections must restore both controller updates and
    # owned raw evidence; they must remain part of the replayable curriculum.
    outcomes = [line.split() for line in (full/'trace.txt').read_text().splitlines() if line.startswith('F ')][0]
    _, cid, correct, satisfaction, *_ = outcomes
    revisions = data/'revisions.txt'
    revisions.write_text('\n'.join(rows)+f'\nREVISE_CREDIT {cid} 1 {1-int(correct)} {1-int(satisfaction)}\nREVISE_CREDIT {cid} 2 {correct} {satisfaction}\n')
    corrected, _ = run('corrected', [f'checkpoint={directory/"corrected.replay"}'], curriculum=revisions)
    assert (corrected/'learner.final.weights').read_bytes() == (full/'learner.final.weights').read_bytes()
    corrected_resume, err = run('corrected_resume', [f'resume={directory/"corrected.replay"}'], curriculum=revisions)
    assert 'Checkpoint replay verified' in err
    assert (corrected/'trace.txt').read_bytes() == (corrected_resume/'trace.txt').read_bytes()
    stale = data/'stale.txt'; stale.write_text(revisions.read_text()+f'REVISE_CREDIT {cid} 1 {correct} {satisfaction}\n')
    _, err = run('stale', okay=False, curriculum=stale)
    assert 'stale, unauthorized or invalid' in err
    result = {'profile': resolved, 'training_episodes': 201, 'frozen_episodes': 2,
              'resume_after_training_episode': 199, 'trace_and_weights_identical': True,
              'coupled_credit_correction_and_resume': True, 'negative_checks': 6,
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [binary, *directory.rglob('*.replay')]}}
    (directory/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Canonical trainer: schedule, frozen evaluation, exact resume, coupled correction and rejection checks passed.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, default=Path('build-wsl/laminar_train'))
    p.add_argument('--out', type=Path, default=Path('logs/canonical_checkpoint'))
    a = p.parse_args(); check(a.binary, a.out)
