"""Task streams. Mirrors cpp/include/laminar/tasks.h (same RNG consumption order)."""
from .rng import Rng


class SyntheticVocab:
    def __init__(self, V=32, C=8, noise=0.08, seed=1001, run_len=64):
        self.V, self.C, self.noise, self.run_len = V, C, noise, run_len
        self.rng = Rng(seed)
        self.maps = [0] * (C * V)
        for k in range(C):
            p = list(range(V))
            for i in range(V - 1, 0, -1):
                j = self.rng.below(i + 1)
                p[i], p[j] = p[j], p[i]
            for i in range(V):
                self.maps[k * V + i] = p[i]
        self.cur = self.rng.below(V)
        self.ctx = 0
        self.left = 0

    def generate(self, n):
        out = bytearray(n)
        for i in range(n):
            if self.left == 0:
                self.ctx = self.rng.below(self.C)
                self.left = self.run_len
            out[i] = self.cur
            nxt = self.maps[self.ctx * self.V + self.cur]
            if self.rng.uniform() < self.noise:
                nxt = self.rng.below(self.V)
            self.cur = nxt
            self.left -= 1
        return bytes(out)
