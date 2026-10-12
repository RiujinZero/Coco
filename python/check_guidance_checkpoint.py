"""Exercise taper withdrawal and resume through the real canonical executable."""
import argparse
import json
import subprocess
from pathlib import Path
from computational_control_probe import build


def check(binary, root):
    root.mkdir(parents=True, exist_ok=False)
    data = root/'data'; build(1, data, 20)
    rows = (data/'input.txt').read_text().splitlines()
    end = rows.index('CHECK stage1')
    rows = ['GUIDANCE public_control_v1 4 4']+rows[:end+3]+['ENDCHECK']
    (data/'input.txt').write_text('\n'.join(rows)+'\n')

    def run(name, extra=(), okay=True, file=None):
        directory = root/name; directory.mkdir()
        command = [str(binary), 'trainer=continual', 'seed=1', f'file={file or data/"input.txt"}',
                   f'environment={data/"environment.txt"}', f'worlds={data/"worlds.txt"}', f'output={directory/"learner"}', *extra]
        with (directory/'trace.txt').open('w') as out:
            p = subprocess.run(command, stdout=out, stderr=subprocess.PIPE, text=True)
        (directory/'stderr.txt').write_text(p.stderr)
        assert (p.returncode == 0) == okay, (name, p.stderr)
        return directory

    full = run('full')
    resumes = []
    for episode in (5, 10):
        cursor = [i+1 for i, row in enumerate(rows) if row.startswith('TRAIN ')][episode-1]
        checkpoint = root/f'after{episode}.replay'
        run(f'partial{episode}', [f'checkpoint={checkpoint}', f'stop_after={cursor}'])
        resumed = run(f'resumed{episode}', [f'resume={checkpoint}'])
        assert 'Checkpoint replay verified' in (resumed/'stderr.txt').read_text()
        for filename in ('trace.txt', 'learner.final.weights', 'learner.final.primitive_weights', 'learner.stage1.weights'):
            assert (full/filename).read_bytes() == (resumed/filename).read_bytes(), filename
        resumes.append(episode)
    trace = (full/'trace.txt').read_text().splitlines()
    autonomous = set(); evaluation = False; hint_count = 0
    for row in trace:
        fields = row.split()
        if fields[0] == 'G' and fields[2] == 'autonomous': autonomous.add(fields[1])
        if fields[0] == 'CHECK': evaluation = True
        if fields[0] == 'H':
            assert not evaluation and fields[1] not in autonomous
            hint_count += 1
    assert len(autonomous) == 12 and hint_count > 0
    for name, badrows in [
        ('repeat', rows+['GUIDANCE public_control_v1 4 4']),
        ('during_evaluation', rows[:-1]+['GUIDANCE public_control_v1 4 4', 'ENDCHECK']),
        ('overflow', ['GUIDANCE public_control_v1 9223372036854775807 1']+rows[1:]),
        ('invalid_source', ['GUIDANCE private_answers 4 4']+rows[1:])]:
        path = data/f'{name}.txt'; path.write_text('\n'.join(badrows)+'\n')
        run(name, okay=False, file=path)
    result = {'resume_after_episodes': resumes, 'trace_and_weights_identical': True,
              'autonomous_training_episodes': len(autonomous), 'teacher_interventions': hint_count,
              'evaluation_interventions': 0, 'negative_checks': 4}
    (root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--binary', type=Path, default=Path('build-wsl/laminar_train_guidance'))
    p.add_argument('--out', type=Path, default=Path('logs/guidance_checkpoint'))
    args = p.parse_args(); check(args.binary, args.out)
