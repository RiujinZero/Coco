"""Reconstruct CART partitions and leaf votes solely from saved public values."""
import collections
import functools
import math
import struct


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def readout(path, values):
    nodes = []
    for line in path.read_text().splitlines():
        r = line.split()
        assert len(r) == 9
        nodes.append((int(r[0]), float(r[1]), int(r[2]), int(r[3]), list(map(int, r[4:]))))
    assert nodes
    samples = [(tuple(map(f32, obs)), max(range(5), key=lambda a: f32(q[a]))) for obs, q in sorted(values.items())]
    assert samples
    visited = set()

    def counts(ids):
        c = collections.Counter(samples[i][1] for i in ids)
        return [c[a] for a in range(5)]

    def purity(c, n):
        return sum(v*v for v in c)/n

    def best_split(ids):
        total = counts(ids); best = purity(total, len(ids)); split = None
        for feature in range(32):
            order = sorted(ids, key=lambda i: (samples[i][0][feature], i))
            left = [0]*5; right = total.copy()
            for k, i in enumerate(order[:-1]):
                label = samples[i][1]; left[label] += 1; right[label] -= 1
                nl, nr = k+1, len(order)-k-1
                lo, hi = samples[i][0][feature], samples[order[k+1]][0][feature]
                if min(nl, nr) < 2 or lo == hi: continue
                score = purity(left, nl)+purity(right, nr)
                if score > best+1e-12:
                    best, split = score, (feature, (lo+hi)/2)
        return split

    def verify(index, ids, depth):
        assert 0 <= index < len(nodes) and index not in visited and ids and depth <= 12
        visited.add(index)
        feature, threshold, left, right, c = nodes[index]
        assert c == counts(ids) and math.isfinite(threshold)
        stop = depth == 12 or len(ids) < 4 or max(c) == len(ids)
        split = None if stop else best_split(ids)
        if split is None:
            assert feature == left == right == -1 and threshold == 0
            return
        assert feature == split[0] and abs(threshold-split[1]) < 1e-12
        lower = [i for i in ids if samples[i][0][feature] <= threshold]
        upper = [i for i in ids if samples[i][0][feature] > threshold]
        assert min(len(lower), len(upper)) >= 2
        verify(left, lower, depth+1); verify(right, upper, depth+1)

    verify(0, list(range(len(samples))), 0)
    assert len(visited) == len(nodes)

    @functools.lru_cache(None)
    def scores(obs):
        index = 0
        while nodes[index][0] >= 0:
            feature, threshold, left, right, _ = nodes[index]
            index = left if f32(obs[feature]) <= threshold else right
        return nodes[index][4]

    matched = sum(max(range(5), key=lambda a: scores(obs)[a]) == label for obs, label in samples)
    return scores, {'nodes': len(nodes), 'states': len(samples), 'preference_matches': matched}
