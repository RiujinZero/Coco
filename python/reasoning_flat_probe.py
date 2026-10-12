"""No operation graph: learn procedures from a flat goal cue and final answers.

The frozen individual skill policies are learned, not oracle executors. Composite
trials supply facts, two entities, relation handles and an opaque question cue.
No subgoals, programs, supporting-fact indices or instruction targets are supplied.
"""
from dataclasses import dataclass
from collections import defaultdict
from collections import deque
import argparse
import json
from pathlib import Path
import random
import time

from reasoning_primitives_probe import Controller, Example, Machine
from reasoning_composition_probe import train_skills

FAMILIES = ("follow", "count", "equal", "count_through", "equal_counts", "equal_counts_through", "equal_ends")
ACTIONS = ("LOAD_A", "LOAD_B", "SAVE", "SWAP", "FOLLOW_0", "FOLLOW_1", "COUNT_0", "COUNT_1", "EQUAL", "ANSWER")


@dataclass(frozen=True)
class FlatQuestion:
    cue: int
    a: tuple
    b: tuple
    relations: tuple[str, str]
    facts: tuple
    answer: tuple


def question(rng, cue, serial, length, number):
    serial = f"{serial}/{rng.getrandbits(64)}"
    uid = 0
    def fresh():
        nonlocal uid
        uid += 1
        return f"{serial}/{uid}"
    a, b = fresh(), fresh()
    r0, r1 = fresh(), fresh()
    same = bool(rng.randrange(2))
    facts = []
    if cue == 2 and same:
        b = a
    ea, eb = fresh(), fresh()
    if cue == 6 and same:
        eb = ea
    seen_subjects = set()
    for src, end in ((a, ea), (b, eb)):
        if src in seen_subjects:
            continue
        seen_subjects.add(src)
        path = [src] + [fresh() for _ in range(length - 1)] + [end]
        facts += [(x, r0, y) for x, y in zip(path, path[1:])]
    n0 = number
    n1 = number if same else number + 1
    # Root holdings and endpoint holdings are independent. Giving both the same
    # counts would let COUNT_THROUGH pass without ever following a relation.
    noise = lambda: rng.randint(0, max(3, number + 1))
    root_counts = (n0, n1) if cue in (1, 4) else (noise(), noise())
    end_counts = (n0, n1) if cue in (3, 5) else (noise(), noise())
    if cue == 3:
        while root_counts[0] == n0:
            root_counts = (noise(), root_counts[1])
    seen_subjects.clear()
    for src, num in ((a, root_counts[0]), (b, root_counts[1]), (ea, end_counts[0]), (eb, end_counts[1])):
        if src in seen_subjects:
            continue
        seen_subjects.add(src)
        for _ in range(num):
            val = fresh()
            facts += [(src, r1, val)] * rng.randint(1, 3)
    for _ in range(4):
        facts.append((fresh(), rng.choice((r0, r1)), fresh()))
    rng.shuffle(facts)
    answers = [("entity", ea), ("int", n0), ("bool", same), ("int", n0),
               ("bool", same), ("bool", same), ("bool", same)]
    return FlatQuestion(cue, ("entity", a), ("entity", b), (r0, r1), tuple(facts), answers[cue])


class FlatMachine:
    def __init__(self, q, skills):
        self.q, self.skills = q, skills
        self.r0, self.r1 = q.a, q.b
        self.last = -1
        self.changed = False
        self.done = False
        self.output = None

    def state(self):
        return (self.q.cue, self.r0[0], self.r1[0], self.r0 == self.q.a,
                self.r0 == self.q.b, self.r1 == self.q.a, self.r1 == self.q.b,
                self.r0 == self.r1, self.last, self.changed)

    def available(self):
        a = [0, 1, 2, 3, 8, 9]
        if self.r0[0] == "entity":
            a += [4, 5, 6, 7]
        return a

    def step(self, a):
        before = self.r0
        if a == 0:
            self.r0 = self.q.a
        elif a == 1:
            self.r0 = self.q.b
        elif a == 2:
            self.r1 = self.r0
        elif a == 3:
            self.r0, self.r1 = self.r1, self.r0
        elif a == 9:
            self.output = self.r0
            self.done = True
        else:
            task = "compare" if a == 8 else "count" if a >= 6 else "follow"
            facts = () if a == 8 else self.q.facts
            subject = repr(self.r0) if a == 8 else self.r0[1]
            target = repr(self.r1) if a == 8 else "unused"
            rel = self.q.relations[a % 2] if a != 8 else "unused"
            vm = Machine(Example(task, facts, subject, rel, target, ("hidden", None), 0))
            for _ in range(256):
                vm.step(self.skills.choose(vm.state(), vm.available()))
                if vm.done:
                    break
            if not vm.done:
                self.done = True
                return
            expected_type = "bool" if a == 8 else "int" if a >= 6 else "entity"
            if vm.output[0] != expected_type:
                self.done = True
                return
            self.r0 = vm.output
        self.last = a
        self.changed = self.r0 != before


def run(p, q, skills, learn=False, epsilon=0, trace=False):
    vm = FlatMachine(q, skills)
    hist, shown = [], []
    for _ in range(24):
        s = vm.state()
        a = p.choose(s, vm.available(), epsilon)
        vm.step(a)
        hist.append((s, a, vm.state(), vm.available()))
        if trace:
            shown.append(ACTIONS[a])
        if vm.done:
            break
    hit = vm.done and vm.output == q.answer
    if learn:
        for i in range(len(hist) - 1, -1, -1):
            s, a, ns, avail = hist[i]
            target = (1.0 if hit else -1.0) if i + 1 == len(hist) else -0.01 + 0.99 * max(p.values(ns)[j] for j in avail)
            v = p.values(s)
            v[a] += 0.1 * (target - v[a])
    return hit, shown


def execute(program, q, skills):
    vm = FlatMachine(q, skills)
    for a in program:
        if a not in vm.available():
            return None
        vm.step(a)
        if vm.done:
            break
    return vm.output


def synthesize(worlds, skills, max_depth=8, max_states=50000, no_bindings=False):
    """Search a common sequence against verified TRAIN outcomes only.

    There is no question-cue -> recipe switch. Each candidate is executed through
    the learned skills. Requiring multiple worlds prevents lucky Boolean answers
    from serving as evidence for a procedure.
    """
    initial = tuple((w.a, w.b) for w in worlds)
    targets = tuple(w.answer for w in worlds)
    queue = deque([(initial, ())])
    visited = {initial}
    cache = {}
    while queue:
        state, program = queue.popleft()
        if tuple(s[0] for s in state) == targets:
            return program + (9,), len(visited)
        if len(program) >= max_depth - 1:
            continue
        available = [0, 1, 2, 3, 8] + ([4, 5, 6, 7] if all(s[0][0] == "entity" for s in state) else [])
        if no_bindings:
            available = [a for a in available if a not in (2, 3)]
        for a in available:
            next_state = []
            for i, regs in enumerate(state):
                key = (i, regs, a)
                if key not in cache:
                    vm = FlatMachine(worlds[i], skills)
                    vm.r0, vm.r1 = regs
                    vm.step(a)
                    cache[key] = None if vm.done else (vm.r0, vm.r1)
                nxt = cache[key]
                if nxt is None:
                    break
                next_state.append(nxt)
            if len(next_state) != len(worlds):
                continue
            nxt = tuple(next_state)
            if nxt in visited:
                continue
            visited.add(nxt)
            if len(visited) > max_states:
                return None, len(visited)
            queue.append((nxt, program + (a,)))
    return None, len(visited)


def search_experiment(seed, skills, examples=12, no_bindings=False):
    rng = random.Random(77000 + seed)
    metrics, programs, search_sizes = {}, {}, {}
    for cue, family in enumerate(FAMILIES):
        worlds = [question(rng, cue, f"search/{i}", rng.randint(1, 3), rng.randint(0, 3)) for i in range(examples)]
        program, size = synthesize(worlds, skills, no_bindings=no_bindings)
        programs[family] = [ACTIONS[a] for a in program] if program is not None else None
        search_sizes[family] = size
        trng = random.Random(199000 + 100 * seed + cue)
        hit = 0
        for i in range(200):
            q = question(trng, cue, f"test/{i}", trng.choice([4, 8, 12, 32]), trng.randint(4, 24))
            hit += program is not None and execute(program, q, skills) == q.answer
        metrics[family] = {"correct": hit, "n": 200}
    return {"seed": seed, "examples": examples, "no_bindings": no_bindings,
            "metrics": metrics, "programs": programs, "search_states": search_sizes}


def export_native(out, seed):
    def encoded(q):
        names = list(dict.fromkeys([q.a[1], q.b[1], *q.relations] + [v for f in q.facts for v in f]))
        ids = {s: i for i, s in enumerate(names)}
        typ = {"entity": 0, "bool": 1, "int": 2}[q.answer[0]]
        target = ids[q.answer[1]] if typ == 0 else int(q.answer[1])
        vals = [q.cue, ids[q.a[1]], ids[q.b[1]], ids[q.relations[0]], ids[q.relations[1]], typ, target, len(q.facts)]
        vals += [ids[v] for fact in q.facts for v in fact]
        return " ".join(map(str, vals)) + "\n"
    rng = random.Random(77000 + seed)
    with (out / f"train_{seed}.txt").open("w") as f:
        for cue in range(7):
            for i in range(12):
                f.write(encoded(question(rng, cue, f"search/{i}", rng.randint(1, 3), rng.randint(0, 3))))
    with (out / f"test_{seed}.txt").open("w") as f:
        for cue in range(7):
            trng = random.Random(199000 + 100 * seed + cue)
            for i in range(200):
                f.write(encoded(question(trng, cue, f"test/{i}", trng.choice([4, 8, 12, 32]), trng.randint(4, 24))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=70000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="logs/reasoning_flat")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--export-native", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {"config": vars(args), "runs": []}
    start = time.time()
    for seed in range(args.seeds):
        if args.export_native:
            export_native(out, seed)
            continue
        skills = train_skills(seed)
        if args.search:
            for examples, no_bindings in ((1, False), (12, False), (12, True)):
                result = search_experiment(seed, skills, examples, no_bindings)
                data["runs"].append(result)
                (out / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
                print(json.dumps({"seed": seed, "examples": examples, "no_bindings": no_bindings,
                                  "accuracy": {k: v['correct']/v['n'] for k,v in result['metrics'].items()},
                                  "seconds": round(time.time() - start, 1)}), flush=True)
            continue
        p, rng = Controller(seed + 55), random.Random(seed + 456)
        for i in range(args.episodes):
            cue = i % 7
            q = question(rng, cue, f"train/{i}", rng.randint(1, 3), rng.randint(0, 3))
            run(p, q, skills, True, max(0.04, 0.5 * (1 - i / args.episodes)))
        before = p.digest(), skills.digest()
        metrics, traces = {}, {}
        rng = random.Random(99000 + seed)
        for cue, name in enumerate(FAMILIES):
            good = 0
            for i in range(100):
                q = question(rng, cue, f"test/{i}", rng.choice([4, 8, 12]), rng.randint(4, 12))
                hit, trace = run(p, q, skills, trace=i == 0)
                good += hit
                if i == 0:
                    traces[name] = trace
            metrics[name] = {"correct": good, "n": 100}
        assert before == (p.digest(), skills.digest())
        data["runs"].append({"seed": seed, "metrics": metrics, "traces": traces})
        (out / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        (out / f"policy_{seed}.json").write_text(json.dumps([{"state": s, "values": v} for s, v in p.q.items()]), encoding="utf-8")
        print(json.dumps({"seed": seed, "metrics": metrics, "seconds": round(time.time() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()
