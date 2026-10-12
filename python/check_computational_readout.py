"""Verify that tree evaluation reuses identical training, values and weights."""
import argparse
import hashlib
import json
from pathlib import Path


def training_digest(path):
    digest = hashlib.sha256(); evaluation = False; lines = 0
    with path.open('rb') as stream:
        for raw in stream:
            if raw.startswith(b'CHECK '): evaluation = True
            if not evaluation and not raw.startswith((b'META preference_tree ', b'META evaluation_search_retention')):
                digest.update(raw); lines += 1
            if evaluation and raw.strip() == b'END': evaluation = False
    assert not evaluation
    return {'sha256': digest.hexdigest(), 'lines': lines}


def compare(source, root, seeds, modes=('learned_tree', 'scaled_tree')):
    result = {}
    for seed in range(seeds):
        src, dst = source/str(seed), root/str(seed)
        for name in ('input.txt', 'environment.txt', 'worlds.txt', 'oracle.json'):
            assert (src/name).read_bytes() == (dst/name).read_bytes(), name
        for mode in modes:
            base = mode.split('_')[0]
            before, after = training_digest(src/f'{base}.txt'), training_digest(dst/f'{mode}.txt')
            assert before == after, ('training changed', seed, mode, before, after)
            matched = []
            for p in src.glob(base+'.*'):
                if p.suffix not in ('.weights', '.values', '.primitive_weights'): continue
                q = dst/(mode+p.name[len(base):])
                assert p.read_bytes() == q.read_bytes(), ('checkpoint changed', str(p), str(q))
                matched.append({'source': str(p), 'readout': str(q), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
            result[f'{seed}/{mode}'] = {'identical_training': after, 'identical_checkpoints': matched}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', default='logs/computational_control'); p.add_argument('--root', default='logs/computational_readout'); p.add_argument('--seeds', type=int, default=3); p.add_argument('--modes', nargs='+', default=['learned_tree', 'scaled_tree']); a = p.parse_args()
    result = compare(Path(a.source), Path(a.root), a.seeds,a.modes)
    (Path(a.root)/'paired_training.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'{len(result)} readout arms have byte-identical training traces, neural/primitive weights and value tables')
