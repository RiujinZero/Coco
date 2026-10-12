"""Isolated test of a learned read/act/write loop; NOT the Coco organism.

Structured queries and triples deliberately bypass language understanding. A small
tabular Q learner chooses individual VM instructions using terminal answer reward.
No demonstration programs, intermediate rewards, or target answers enter its state.
The finite observation abstracts away entity IDs and numeric accumulator values:
this is a supplied representation, not evidence of learned representation discovery.

python python/reasoning_primitives_probe.py --out logs/reasoning_primitives
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time


ACTIONS = ("BIND", "READ", "COPY", "MARK", "INC", "COMPARE",
           "RETURN_FOCUS", "RETURN_VALUE", "RETURN_BOOL", "RETURN_COUNT")
TASKS = ("follow", "compare", "count")
RETURNS = set(range(6, 10))


@dataclass(frozen=True)
class Example:
    task: str
    facts: tuple[tuple[str, str, str], ...]
    subject: str
    relation: str
    target: str
    answer: tuple[str, object]
    size: int


def example(rng: random.Random, task: str, size: int, domain: str,
            serial: int, duplicates: bool = True, distractors: int = 4) -> Example:
    # No name is shared between episodes, or between training and evaluation.
    ids = [f"{domain}/{serial}/{i}" for i in range(3 * size + 30)]
    rng.shuffle(ids)
    rel, other = "r", "s"
    subject = ids[0]
    if task in ("follow", "compare"):
        facts = [(ids[i], rel, ids[i + 1]) for i in range(size)]
        endpoint = ids[size]
        target = endpoint if rng.randrange(2) else rng.choice(ids[:size] + ids[size + 1:])
        answer = ("entity", endpoint) if task == "follow" else ("bool", endpoint == target)
    else:
        values = ids[1:size + 1]
        facts = [(subject, rel, v) for v in values]
        if duplicates:
            # Repeated evidence is not an additional distinct object.
            facts += [(subject, rel, v) for v in values for _ in range(rng.randrange(3))]
        target = ids[size + 2]
        answer = ("int", len(set(values)))
    for i in range(distractors):
        a, b = ids[size + 2 + i], ids[size + 12 + i]
        facts.append((a, rel, b))
        facts.append((rng.choice(ids[:size + 1]), other, b))
    rng.shuffle(facts)
    return Example(task, tuple(facts), subject, rel, target, answer, size)


class Machine:
    """Generic instructions. Only the external scorer consults Example.answer.

    READ returns the next triple matching the focus and query relation. COPY
    makes its value the new focus and resets the cursor. Neither follows a path
    or selects an endpoint automatically. MARK and INC are separate operations.
    """
    def __init__(self, ex: Example, ablation: str = "full"):
        self.ex = ex
        self.ablation = ablation
        self.focus = None
        self.value = None
        self.cursor = -1
        self.exhausted = False
        self.seen = set()
        self.acc = 0
        self.predicate = None
        self.last = -1
        self.done = False
        self.output = None
        self.steps = 0

    def state(self):
        # The query kind is an opaque categorical input, NOT a selected program.
        # Register contents and lengths are never policy-table keys.
        return (self.ex.task, self.focus is not None, self.value is not None,
                self.focus == self.ex.subject, self.exhausted,
                self.value is not None and self.value in self.seen,
                self.acc == 0, self.predicate, self.last)

    def available(self):
        # Only operand/type validity gates instructions; no task-specific masks.
        a = [0, 4, 9]
        if self.focus is not None:
            a += [1, 5, 6]
        if self.value is not None:
            a += [2, 3, 7]
        if self.predicate is not None:
            a += [8]
        if self.ablation == "no_compare":
            a = [v for v in a if v != 5]
        if self.ablation == "no_mark":
            a = [v for v in a if v != 3]
        if self.ablation == "no_copy":
            a = [v for v in a if v != 2]
        return a

    def step(self, a: int):
        assert a in self.available()
        self.steps += 1
        if a == 0:
            self.focus = self.ex.subject
            self.value = None
            self.cursor = -1
            self.exhausted = False
        elif a == 1:
            self.value = None
            start = self.cursor + 1 if self.ablation != "no_cursor" else 0
            for i in range(start, len(self.ex.facts)):
                src, rel, dst = self.ex.facts[i]
                if src == self.focus and rel == self.ex.relation:
                    self.cursor = i
                    self.value = dst
                    break
            self.exhausted = self.value is None
            if self.exhausted:
                self.cursor = len(self.ex.facts)
        elif a == 2:
            self.focus = self.value
            self.value = None
            self.cursor = -1
            self.exhausted = False
        elif a == 3:
            self.seen.add(self.value)
        elif a == 4:
            self.acc += 1
        elif a == 5:
            self.predicate = self.focus == self.ex.target
        elif a == 6:
            self.output = ("entity", self.focus)
        elif a == 7:
            self.output = ("entity", self.value)
        elif a == 8:
            self.output = ("bool", self.predicate)
        elif a == 9:
            self.output = ("int", self.acc)
        self.last = a
        self.done = a in RETURNS
        return self.done


class Controller:
    def __init__(self, seed: int):
        self.q = {}
        self.rng = random.Random(seed)

    def values(self, state):
        if state not in self.q:
            self.q[state] = [0.0] * len(ACTIONS)
        return self.q[state]

    def choose(self, state, available, epsilon=0.0):
        if self.rng.random() < epsilon:
            return self.rng.choice(available)
        # Evaluation is read-only and has deterministic tie breaking.
        values = self.q.get(state, [0.0] * len(ACTIONS))
        best = max(values[a] for a in available)
        opts = [a for a in available if values[a] == best]
        return self.rng.choice(opts) if epsilon else opts[0]

    def digest(self):
        return hashlib.sha256(repr(sorted(self.q.items(), key=repr)).encode()).hexdigest()


class PolicyGradient(Controller):
    """Second learner: Monte Carlo policy gradient, with no TD bootstrapping.

    This control matters because the finite observation is not a Markov state
    for arbitrary erroneous programs (e.g. different nonzero accumulator values).
    """
    def __init__(self, seed):
        super().__init__(seed)
        self.baseline = defaultdict(float)

    def probabilities(self, state, available):
        vals = self.q.get(state, [0.0] * len(ACTIONS))
        top = max(vals[a] for a in available)
        weights = [math.exp(vals[a] - top) for a in available]
        total = sum(weights)
        return [w / total for w in weights]

    def choose(self, state, available, epsilon=0.0):
        if not epsilon:
            return super().choose(state, available)
        return self.rng.choices(available, weights=self.probabilities(state, available))[0]

    def update(self, history, terminal):
        updates = defaultdict(lambda: [0.0] * len(ACTIONS))
        reward = terminal
        for s, a, _, avail, probs in reversed(history):
            advantage = reward - self.baseline[s]
            self.baseline[s] += 0.03 * advantage
            delta = updates[s]
            for j, pr in zip(avail, probs):
                delta[j] += 0.025 * advantage * ((j == a) - pr)
                # Small entropy term preserves exploration.
                delta[j] += 0.0001 * (1 / len(avail) - pr)
            reward = -0.01 + 0.99 * reward
        for s, delta in updates.items():
            q = self.values(s)
            for a in range(len(ACTIONS)):
                q[a] = min(8.0, max(-8.0, q[a] + delta[a]))


def oracle_action(vm):
    """Handwritten positive control ONLY. Never called by answer-only learning."""
    if vm.focus is None:
        return 0
    if vm.ex.task == "count":
        if vm.exhausted:
            return 9
        if vm.last == 3:
            return 4
        if vm.last == 4:
            return 1
        return 3 if vm.value is not None and vm.value not in vm.seen else 1
    if vm.exhausted:
        if vm.ex.task == "follow":
            return 6
        return 5 if vm.predicate is None else 8
    return 2 if vm.value is not None else 1


def demonstration_control(seed, n=300):
    """Fit the SAME table from supplied action demonstrations; diagnostic only."""
    policy = Controller(seed)
    rng = random.Random(seed + 8000)
    for i in range(n):
        task = TASKS[i % 3]
        ex = example(rng, task, rng.randint(0 if task == "count" else 1, 3), "demo", i)
        vm = Machine(ex)
        for _ in range(48):
            a = oracle_action(vm)
            q = policy.values(vm.state())
            assert not any(v > 0 and j != a for j, v in enumerate(q)), "inconsistent demonstration"
            q[a] = 1.0
            if vm.step(a):
                break
        assert vm.output == ex.answer
    return policy


def episode(policy, ex, *, epsilon=0.0, learn=False, ablation="full",
            max_steps=128, reward_rng=None, trace=False):
    vm = Machine(ex, ablation)
    history = []
    shown = []
    for _ in range(max_steps):
        s = vm.state()
        avail = vm.available()
        probs = policy.probabilities(s, avail) if isinstance(policy, PolicyGradient) else None
        a = policy.choose(s, avail, epsilon)
        vm.step(a)
        ns = vm.state()
        history.append((s, a, ns, avail if probs is not None else vm.available(), probs))
        if trace:
            shown.append({"op": ACTIONS[a], "focus": vm.focus, "value": vm.value,
                          "count": vm.acc, "seen": len(vm.seen),
                          "exhausted": vm.exhausted, "predicate": vm.predicate})
        if vm.done:
            break
    correct = vm.done and vm.output == ex.answer
    if learn:
        # Reward is observed ONLY after termination or timeout. Reverse replay
        # propagates it using local one-step TD updates; no gradient/backprop.
        terminal = (1.0 if correct else -1.0)
        if reward_rng is not None:
            terminal = 1.0 if reward_rng.randrange(2) else -1.0
        if isinstance(policy, PolicyGradient):
            policy.update(history, terminal)
        else:
            for i in range(len(history) - 1, -1, -1):
                s, a, ns, avail, _ = history[i]
                target = terminal if i == len(history) - 1 else -0.01 + 0.99 * max(policy.values(ns)[j] for j in avail)
                q = policy.values(s)
                q[a] += 0.15 * (target - q[a])
    return {"correct": correct, "steps": vm.steps, "output": vm.output,
            "timeout": not vm.done, "trace": shown}


def train(seed, episodes, ablation="full", random_reward=False, method="td"):
    policy = (PolicyGradient if method == "pg" else Controller)(seed + 20)
    rng = random.Random(seed + 100)
    rrng = random.Random(seed + 900) if random_reward else None
    curves = []
    wins = defaultdict(int)
    totals = defaultdict(int)
    for i in range(episodes):
        task = TASKS[i % len(TASKS)]
        # Size curriculum only; no instruction or trajectory is supplied.
        ceiling = 1 if i < episodes // 6 else 2 if i < episodes // 3 else 3
        size = rng.randint(0 if task == "count" else 1, ceiling)
        ex = example(rng, task, size, "train", i)
        epsilon = max(0.04, 0.4 * (1 - i / episodes))
        result = episode(policy, ex, learn=True, epsilon=epsilon,
                         ablation=ablation, max_steps=48, reward_rng=rrng)
        wins[task] += result["correct"]
        totals[task] += 1
        if (i + 1) % max(1, episodes // 10) == 0:
            curves.append({"episode": i + 1, "accuracy": {t: wins[t] / totals[t] for t in TASKS}})
            wins.clear()
            totals.clear()
    return policy, curves


def test_set(seed, n=100):
    rng = random.Random(70000 + seed)
    out = []
    serial = 0
    for task in TASKS:
        sizes = (0, 1, 2, 3, 4, 6, 8, 12) if task == "count" else (1, 2, 3, 4, 6, 8, 12)
        for size in sizes:
            for _ in range(n):
                out.append(example(rng, task, size, "test", serial))
                serial += 1
    return out


def evaluate(policy, examples, ablation="full", max_steps=128):
    before = policy.digest()
    metrics = defaultdict(lambda: {"correct": 0, "n": 0, "steps": 0, "timeouts": 0})
    traces = {}
    for ex in examples:
        key = f"{ex.task}/{ex.size}"
        take_trace = ex.size in (3, 12) and key not in traces
        r = episode(policy, ex, ablation=ablation, max_steps=max_steps, trace=take_trace)
        row = metrics[key]
        row["correct"] += r["correct"]
        row["n"] += 1
        row["steps"] += r["steps"]
        row["timeouts"] += r["timeout"]
        if take_trace:
            traces[key] = {"example": ex.__dict__, "result": r}
    assert before == policy.digest(), "evaluation changed learned parameters"
    return dict(metrics), traces


def self_check():
    """Check VM semantics and split invariants, not learned performance."""
    ex = Example("count", (("a", "r", "b"), ("a", "s", "x"),
                          ("a", "r", "b"), ("a", "r", "c")),
                 "a", "r", "c", ("int", 2), 2)
    vm = Machine(ex)
    vm.step(0)
    vm.step(1)
    assert vm.value == "b"
    vm.step(3)
    vm.step(4)
    vm.step(1)
    assert vm.value in vm.seen
    vm.step(1)
    assert vm.value == "c" and vm.value not in vm.seen
    vm.step(3)
    vm.step(4)
    vm.step(1)
    assert vm.exhausted
    vm.step(9)
    assert vm.output == ex.answer
    # Changing the hidden answer must not change observations or transitions.
    altered = Example(**{**ex.__dict__, "answer": ("int", 99)})
    a, b = Machine(ex), Machine(altered)
    for op in (0, 1, 3, 4, 1):
        assert a.state() == b.state() and a.available() == b.available()
        a.step(op)
        b.step(op)
    assert a.state() == b.state()


def export_agency(out, seed, episodes, ntest):
    """Integerized structured episodes for the native AgencyKernel probe."""
    out.mkdir(parents=True, exist_ok=True)
    def line(ex):
        names = list(dict.fromkeys([ex.subject, ex.target] + [v for a, _, b in ex.facts for v in (a, b)]))
        ids = {v: i for i, v in enumerate(names)}
        typ = {"entity": 0, "bool": 1, "int": 2}[ex.answer[0]]
        ans = ids[ex.answer[1]] if typ == 0 else int(ex.answer[1])
        values = [TASKS.index(ex.task), ex.size, ids[ex.subject], ids[ex.target], typ, ans, len(ex.facts)]
        values += [v for a, r, b in ex.facts for v in (ids[a], 0 if r == ex.relation else 1, ids[b])]
        return " ".join(map(str, values)) + "\n"
    rng = random.Random(seed + 100)
    with (out / f"train_{seed}.txt").open("w", encoding="utf-8") as f:
        for i in range(episodes):
            task = TASKS[i % 3]
            ceiling = 1 if i < episodes // 6 else 2 if i < episodes // 3 else 3
            size = rng.randint(0 if task == "count" else 1, ceiling)
            f.write(line(example(rng, task, size, "train", i)))
    with (out / f"test_{seed}.txt").open("w", encoding="utf-8") as f:
        for ex in test_set(seed, ntest):
            f.write(line(ex))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=60000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--test-per-size", type=int, default=100)
    p.add_argument("--out", default="logs/reasoning_primitives")
    p.add_argument("--controls", action="store_true")
    p.add_argument("--method", choices=("td", "pg"), default="td")
    p.add_argument("--export-agency", action="store_true")
    args = p.parse_args()
    self_check()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.export_agency:
        for seed in range(args.seeds):
            export_agency(out, seed, args.episodes, args.test_per_size)
        return
    results = {"config": vars(args), "actions": ACTIONS, "runs": [],
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    t0 = time.time()
    for seed in range(args.seeds):
        tests = test_set(seed, args.test_per_size)
        conditions = [("full", False)]
        if args.controls:
            conditions += [("full", True), ("no_mark", False), ("no_copy", False), ("no_compare", False)]
        for condition, shuffled in conditions:
            name = "random_reward" if shuffled else condition
            policy, curve = train(seed, args.episodes, condition, shuffled, args.method)
            metrics, traces = evaluate(policy, tests, condition)
            run = {"seed": seed, "condition": name, "states": len(policy.q),
                   "curve": curve, "metrics": metrics, "traces": traces,
                   "policy_sha256": policy.digest()}
            (out / f"policy_{seed}_{name}.json").write_text(json.dumps(
                [{"state": s, "values": v} for s, v in policy.q.items()]), encoding="utf-8")
            if name == "full":
                run["frozen_ablations"] = {a: evaluate(policy, tests, a)[0]
                                           for a in ("no_copy", "no_mark", "no_compare", "no_cursor")}
                run["one_step"] = evaluate(policy, tests, max_steps=1)[0]
                demo = demonstration_control(seed)
                run["demonstration_control"] = evaluate(demo, tests)[0]
                run["demonstration_ablations"] = {a: evaluate(demo, tests, a)[0]
                                                for a in ("no_copy", "no_mark", "no_compare", "no_cursor")}
            results["runs"].append(run)
            results["seconds"] = time.time() - t0
            (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            summary = {}
            for task in TASKS:
                for label, lo, hi in (("short", 0, 3), ("long", 4, 12)):
                    rows = [r for k, r in metrics.items() if k.startswith(task + "/") and lo <= int(k.split("/")[1]) <= hi]
                    summary[f"{task}_{label}"] = round(sum(r["correct"] for r in rows) / sum(r["n"] for r in rows), 4)
            print(json.dumps({"seed": seed, "condition": name, "accuracy": summary,
                              "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
