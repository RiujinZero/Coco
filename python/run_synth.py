"""Run the NumPy reference on the synthetic vocabulary task and optionally dump a trace for parity checks.

    python python/run_synth.py --units 4096 --train 20000 --eval 4000 --trace ref_trace.jsonl
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from laminar_ref import Params, Organism, SyntheticVocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=4096)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--eval", type=int, default=4000)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--contexts", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--run", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--hidden", default="none", choices=["none", "perturb"])
    ap.add_argument("--trace", default=None)
    ap.add_argument("--report", type=int, default=4096)
    a = ap.parse_args()

    p = Params(units=a.units, vocab=a.vocab, seed=a.seed, block=a.block, hidden=a.hidden)
    task = SyntheticVocab(a.vocab, a.contexts, a.noise, a.seed + 1000, a.run)
    stream = task.generate(a.train + a.eval + 1)
    org = Organism(p)
    if a.trace:
        org.trace_fp = open(a.trace, "w")

    def drive(begin, end, phase):
        pos, since, agg = begin, 0, {"events": 0, "correct": 0, "loss": 0.0, "macs": 0}
        t0 = time.time()
        while pos < end:
            n = min(p.block, end - pos)
            org.run_block(stream[pos:pos + n + 1], n + 1)
            org.end_block()
            s = org.st
            since += s.events
            if since >= a.report:
                print(f"[{phase}] {s.report(org.block_idx - 1)}  ({s.events / (time.time() - t0):.0f} ev/s)", flush=True)
                since = 0
                t0 = time.time()
            agg["events"] += s.events; agg["correct"] += s.correct; agg["loss"] += s.loss_sum; agg["macs"] += s.macs
            s.reset()
            pos += n
        return agg

    tr = drive(0, a.train, "train")
    print(f"TRAIN: {tr['events']} events, acc {100 * tr['correct'] / tr['events']:.2f}%, loss {tr['loss'] / tr['events']:.4f}, {tr['macs'] / tr['events']:.0f} MACs/event")
    org.set_mode(False, False)
    ev = drive(a.train, a.train + a.eval, "frozen")
    print(f"FROZEN: {ev['events']} events, acc {100 * ev['correct'] / ev['events']:.2f}% (chance {100 / a.vocab:.3f}%), loss {ev['loss'] / ev['events']:.4f}, "
          f"live locks {org.tree.live_locks()}, live units {p.units - len(org.free_units)}")
    if org.trace_fp:
        org.trace_fp.close()


if __name__ == "__main__":
    main()
