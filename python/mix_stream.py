"""Mix several byte streams into one training stream, by proportion, in episode-sized pieces.

    python python/mix_stream.py OUT SRC1 SRC2 ... [--piece 65536] [--seed 1]

Each source is cut into pieces (at a delimiter "<|endofdialogue|>" or "<|endoftext|>" boundary when one occurs
within the piece, else at a newline) and the output draws pieces from the sources in proportion to their
remaining size, so a 2 MB source is spread across a 600 MB stream rather than appearing once. Every piece from a
dialogue/episode source keeps its delimiters; prose pieces get a "<|endoftext|>\n" boundary so the episode store
sees them as episodes too.
"""
import sys, os, random

DELIMS = [b'\n<|endofdialogue|>\n', b'<|endoftext|>']

def pieces(path, piece):
    data = open(path, 'rb').read(); out = []; i = 0
    while i < len(data):
        j = min(len(data), i + piece)
        if j < len(data):
            cut = -1
            for d in DELIMS:
                k = data.rfind(d, i + piece // 2, j)
                if k > cut: cut = k + len(d)
            if cut < 0:
                k = data.rfind(b'\n', i + piece // 2, j); cut = k + 1 if k > 0 else j
            j = cut
        p = data[i:j]
        if not any(d in p for d in DELIMS): p = p.rstrip(b'\n') + b'\n<|endoftext|>\n'
        out.append(p); i = j
    return out

def main():
    args = sys.argv[1:]; piece = 65536; seed = 1
    if '--piece' in args: i = args.index('--piece'); piece = int(args[i + 1]); del args[i:i + 2]
    if '--seed' in args: i = args.index('--seed'); seed = int(args[i + 1]); del args[i:i + 2]
    out, srcs = args[0], args[1:]
    rng = random.Random(seed)
    queues = [pieces(s, piece) for s in srcs]
    for q, s in zip(queues, srcs): print(os.path.basename(s), len(q), 'pieces', sum(len(p) for p in q), 'bytes')
    total = 0
    with open(out, 'wb') as o:
        while any(queues):
            weights = [sum(len(p) for p in q[:64]) + (len(q) - 64) * piece if len(q) > 64 else sum(len(p) for p in q) for q in queues]
            i = rng.choices(range(len(queues)), weights=[max(w, 1) if q else 0 for w, q in zip(weights, queues)])[0]
            p = queues[i].pop(0); o.write(p); total += len(p)
    print('wrote', out, total, 'bytes')

if __name__ == '__main__': main()
