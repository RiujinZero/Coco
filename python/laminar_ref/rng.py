"""splitmix64 / murmur mixer, bit-for-bit identical to cpp/include/laminar/rng.h."""
M64 = (1 << 64) - 1


class Rng:
    def __init__(self, seed=1):
        self.s = seed & M64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def uniform(self):
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)


def mix64(x):
    x &= M64
    x ^= x >> 33
    x = (x * 0xff51afd7ed558ccd) & M64
    x ^= x >> 33
    x = (x * 0xc4ceb9fe1a85ec53) & M64
    x ^= x >> 33
    return x
