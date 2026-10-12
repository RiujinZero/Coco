"""Adaptive address tree, locks and split evidence. Mirrors cpp/include/laminar/address.h."""
import math
from .config import TUMBLER_BITS, SENSORY_BITS


class Lock:
    __slots__ = ("value", "mask", "unit", "hits", "last_hit_block", "born_block", "err_ema", "evidence", "alive")

    def __init__(self, value, mask, unit, block):
        self.value, self.mask, self.unit = value, mask, unit
        self.hits = 0
        self.last_hit_block = block
        self.born_block = block
        self.err_ema = 0
        self.evidence = -1
        self.alive = 1


class EvidencePool:
    def __init__(self, nslots, buckets):
        self.buckets = buckets
        self.counts = [None] * nslots
        self.n = [None] * nslots
        self.lock = [None] * nslots
        self.free = [nslots - 1 - i for i in range(nslots)]

    def acquire(self, lock):
        if not self.free:
            return -1
        s = self.free.pop()
        self.counts[s] = [[[0] * self.buckets for _ in range(2)] for _ in range(TUMBLER_BITS)]
        self.n[s] = [[0, 0] for _ in range(TUMBLER_BITS)]
        self.lock[s] = lock
        return s

    def release(self, s):
        if s >= 0:
            self.lock[s] = None
            self.free.append(s)

    def record(self, s, key, y):
        yb = y % self.buckets
        c, n = self.counts[s], self.n[s]
        for b in range(TUMBLER_BITS):
            v = (key >> (b + SENSORY_BITS)) & 1
            if c[b][v][yb] < 65535:
                c[b][v][yb] += 1
            n[b][v] += 1

    def best_split(self, s, min_ev):
        c, n = self.counts[s], self.n[s]
        best, bestg = -1, 0.0
        for b in range(TUMBLER_BITS):
            n0, n1 = n[b]
            if n0 < min_ev or n1 < min_ev:
                continue
            N = float(n0 + n1)
            g = 0.0
            for y in range(self.buckets):
                c0, c1 = float(c[b][0][y]), float(c[b][1][y])
                tot = c0 + c1
                if tot == 0.0:
                    continue
                e0, e1 = tot * n0 / N, tot * n1 / N
                if c0 > 0:
                    g += c0 * math.log(c0 / e0)
                if c1 > 0:
                    g += c1 * math.log(c1 / e1)
            g *= 2.0
            if g > bestg:
                bestg, best = g, b
        return best, bestg


class AddressTree:
    def __init__(self, leaf_max):
        self.leaf_max = leaf_max
        self.bit = [0]
        self.left = [-1]
        self.right = [-1]
        self.leaves = [[]]            # lock ids per node (only leaves non-empty)
        self.locks = []
        self.free_locks = []
        self.lock_leaves = []

    def num_nodes(self):
        return len(self.bit)

    def live_locks(self):
        return sum(l.alive for l in self.locks)

    # hot path
    def walk(self, key):
        n, depth = 0, 0
        while self.left[n] >= 0:
            n = self.right[n] if (key >> self.bit[n]) & 1 else self.left[n]
            depth += 1
        return n, depth

    def match(self, leaf, key, cap=16):
        out = []
        for lid in self.leaves[leaf]:
            L = self.locks[lid]
            if ((key ^ L.value) & L.mask) == 0:
                out.append(lid)
                if len(out) >= cap:
                    break
        return out

    # structural
    def new_lock(self, value, mask, unit, block):
        if self.free_locks:
            lid = self.free_locks.pop()
            self.locks[lid] = Lock(value, mask, unit, block)
            self.lock_leaves[lid] = []
        else:
            lid = len(self.locks)
            self.locks.append(Lock(value, mask, unit, block))
            self.lock_leaves.append([])
        self._insert(0, lid)
        return lid

    def kill_lock(self, lid):
        for leaf in self.lock_leaves[lid]:
            lv = self.leaves[leaf]
            i = lv.index(lid)
            lv[i] = lv[-1]
            lv.pop()
        self.lock_leaves[lid] = []
        self.locks[lid].alive = 0
        self.free_locks.append(lid)

    def _insert(self, n, lid):
        L = self.locks[lid]
        while self.left[n] >= 0:
            if (L.mask >> self.bit[n]) & 1:
                n = self.right[n] if (L.value >> self.bit[n]) & 1 else self.left[n]
            else:
                self._insert(self.left[n], lid)
                n = self.right[n]
        self.leaves[n].append(lid)
        self.lock_leaves[lid].append(n)

    def _add_leaf(self):
        self.bit.append(0)
        self.left.append(-1)
        self.right.append(-1)
        self.leaves.append([])
        return len(self.bit) - 1

    def maintain(self):
        splits = 0
        n = 0
        while n < len(self.bit):
            if self.left[n] >= 0 or len(self.leaves[n]) <= self.leaf_max:
                n += 1
                continue
            ids = list(self.leaves[n])
            c = len(ids)
            all_mask = (1 << 64) - 1
            for lid in ids:
                all_mask &= self.locks[lid].mask
            bestb, bestimb = -1, 1 << 40
            for b in range(64):
                if not (all_mask >> b) & 1:
                    continue
                n1 = sum((self.locks[lid].value >> b) & 1 for lid in ids)
                if n1 == 0 or n1 == c:
                    continue
                imb = abs(n1 - (c - n1))
                if imb < bestimb:
                    bestimb, bestb = imb, b
            if bestb < 0:
                n += 1
                continue
            l, r = self._add_leaf(), self._add_leaf()
            self.bit[n], self.left[n], self.right[n] = bestb, l, r
            self.leaves[n] = []
            for lid in ids:
                self.lock_leaves[lid].remove(n)
                side = r if (self.locks[lid].value >> bestb) & 1 else l
                self.leaves[side].append(lid)
                self.lock_leaves[lid].append(side)
            splits += 1
            n += 1
        return splits

    def depth_stats(self):
        ds = []

        def rec(n, d):
            if self.left[n] < 0:
                ds.append(d)
                return
            rec(self.left[n], d + 1)
            rec(self.right[n], d + 1)
        rec(0, 0)
        ds.sort()
        occ = [len(self.leaves[n]) for n in range(len(self.bit)) if self.left[n] < 0]
        return sum(ds) / len(ds), ds[int(0.95 * (len(ds) - 1))], sum(occ) / len(occ)
