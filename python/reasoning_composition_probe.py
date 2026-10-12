"""Composition after individual skills are learned, with no composite action teacher.

Goals are supplied as semantic expression graphs, NOT natural-language questions.
The controller learns traversal, skill choice, result binding and stopping from
terminal answer reward. Test expression shapes mix skills in unseen combinations.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

from reasoning_primitives_probe import (Controller, Example, Machine, TASKS,
                                        example, oracle_action)

OPS = ("LEFT", "RIGHT", "UP", "FOLLOW", "COUNT", "EQUAL", "ANSWER")


@dataclass(frozen=True)
class Node:
    kind: str
    children: tuple[int, ...] = ()
    relation: str = ""
    literal: tuple | None = None


@dataclass(frozen=True)
class Problem:
    nodes: tuple[Node, ...]
    root: int
    facts: tuple[tuple[str, str, str], ...]
    answer: tuple
    family: str


def reference(p, root=None):
    """Independent evaluator, used ONLY to validate/export benchmark labels."""
    def value(i):
        n = p.nodes[i]
        if n.kind == "literal":
            return n.literal
        args = [value(c) for c in n.children]
        if n.kind == "equal":
            return ("bool", args[0] == args[1])
        assert args[0][0] == "entity"
        entity = args[0][1]
        if n.kind == "count":
            return ("int", len({b for a, r, b in p.facts if a == entity and r == n.relation}))
        visited = set()
        while True:
            if entity in visited:
                raise ValueError("cyclic benchmark relation")
            visited.add(entity)
            successors = {b for a, r, b in p.facts if a == entity and r == n.relation}
            if not successors:
                return ("entity", entity)
            assert len(successors) == 1
            entity = next(iter(successors))
    return value(p.root if root is None else root)


class Builder:
    def __init__(self, rng, prefix, length):
        self.rng, self.prefix, self.length = rng, prefix, length
        self.nodes, self.facts = [], []
        self.serial = 0

    def fresh(self):
        self.serial += 1
        return f"{self.prefix}:{self.serial}"

    def add(self, node):
        self.nodes.append(node)
        return len(self.nodes) - 1

    def entity(self, depth=0, endpoint=None):
        if depth == 0:
            v = endpoint or self.fresh()
            return self.add(Node("literal", literal=("entity", v))), v
        child, start = self.entity(depth - 1)
        rel = self.fresh()
        end = endpoint or self.fresh()
        path = [start] + [self.fresh() for _ in range(self.length - 1)] + [end]
        self.facts += [(a, rel, b) for a, b in zip(path, path[1:])]
        self.facts += [(start, self.fresh(), self.fresh()), (self.fresh(), rel, self.fresh())]
        return self.add(Node("follow", (child,), rel)), end

    def count(self, depth, number):
        child, subject = self.entity(depth)
        rel = self.fresh()
        values = [self.fresh() for _ in range(number)]
        for v in values:
            self.facts += [(subject, rel, v)] * self.rng.randint(1, 3)
        self.facts += [(subject, self.fresh(), self.fresh()), (self.fresh(), rel, self.fresh())]
        return self.add(Node("count", (child,), rel)), number

    def finish(self, root, answer, family):
        self.rng.shuffle(self.facts)
        # Scramble storage order: index arithmetic cannot provide the schedule.
        perm = list(range(len(self.nodes)))
        self.rng.shuffle(perm)
        remap = {old: new for new, old in enumerate(perm)}
        nodes = tuple(Node(self.nodes[i].kind, tuple(remap[c] for c in self.nodes[i].children),
                           self.nodes[i].relation, self.nodes[i].literal) for i in perm)
        return Problem(nodes, remap[root], tuple(self.facts), answer, family)


def problem(rng, family, serial, length=2, depth=2, number=None):
    b = Builder(rng, f"{family}/{serial}/{rng.getrandbits(64)}", length)
    number = rng.randint(0, 3) if number is None else number
    equal = bool(serial % 2)
    if family == "atomic_follow":
        root, v = b.entity(1)
        ans = ("entity", v)
    elif family == "atomic_count":
        root, v = b.count(0, number)
        ans = ("int", v)
    elif family == "atomic_equal":
        left, a = b.entity(0)
        right, _ = b.entity(0, a if equal else None)
        root = b.add(Node("equal", (left, right)))
        ans = ("bool", equal)
    elif family in ("follow_follow", "deep_follow"):
        root, v = b.entity(depth)
        ans = ("entity", v)
    elif family in ("equal_follow", "deep_equal_follow"):
        left, a = b.entity(depth)
        right, _ = b.entity(depth, a if equal else None)
        root = b.add(Node("equal", (left, right)))
        ans = ("bool", equal)
    elif family in ("count_follow", "deep_count_follow"):
        root, v = b.count(depth, number)
        ans = ("int", v)
    elif family in ("equal_count", "equal_count_follow"):
        n2 = number if equal else number + 1
        left, _ = b.count(depth, number)
        right, _ = b.count(depth, n2)
        root = b.add(Node("equal", (left, right)))
        ans = ("bool", equal)
    else:
        raise ValueError(family)
    return b.finish(root, ans, family)


def train_skills(seed):
    """Short verified individual-skill demonstrations; no composite goals."""
    p = Controller(seed)
    rng = random.Random(seed + 44)
    for i in range(600):
        task = TASKS[i % 3]
        # EQUAL is truly atomic here, rather than FOLLOW followed by comparison.
        size = 0 if task == "compare" else rng.randint(0 if task == "count" else 1, 3)
        ex = example(rng, task, size, "skill", i)
        vm = Machine(ex)
        while not vm.done:
            a = oracle_action(vm)
            q = p.values(vm.state())
            assert not any(v and j != a for j, v in enumerate(q))
            q[a] = 1.0
            vm.step(a)
        assert vm.output == ex.answer
    return p


class Composer:
    def __init__(self, p, skills, ablation="full"):
        self.p, self.skills, self.ablation = p, skills, ablation
        self.cursor = p.root
        self.stack = []
        self.values = {i: n.literal for i, n in enumerate(p.nodes) if n.kind == "literal"}
        self.done = False
        self.output = None
        self.skill_steps = 0
        self.trace = []
        self.shared_seen, self.shared_acc = set(), 0

    def state(self):
        n = self.p.nodes[self.cursor]
        flags = [c in self.values for c in n.children]
        return (n.kind, len(flags) >= 1, flags[0] if flags else False,
                len(flags) == 2, flags[1] if len(flags) == 2 else False,
                self.cursor in self.values, bool(self.stack))

    def available(self):
        a = [3, 4, 5, 6]
        n = self.p.nodes[self.cursor]
        if n.children:
            a += [0]
        if len(n.children) == 2:
            a += [1]
        if self.stack:
            a += [2]
        return a

    def call(self, op):
        n = self.p.nodes[self.cursor]
        needed = 2 if op == 5 else 1
        if len(n.children) < needed or any(c not in self.values for c in n.children[:needed]):
            self.done = True
            return
        args = [self.values[c] for c in n.children[:needed]]
        if op in (3, 4) and args[0][0] != "entity":
            self.done = True
            return
        if self.ablation == "root_binding":
            # Corrupt only caller-to-callee binding, leaving learned skills intact.
            first = next(v.literal for v in self.p.nodes if v.kind == "literal")
            args[0] = first
        task = {3: "follow", 4: "count", 5: "compare"}[op]
        # Box typed arguments for EQUAL so integers and entities cannot alias.
        subject = args[0][1] if op != 5 else repr(args[0])
        target = repr(args[1]) if op == 5 else "unused"
        facts = self.p.facts if op != 5 else ()
        # Placeholder answer is never read by the VM or skill policy.
        ex = Example(task, facts, subject, n.relation, target, ("hidden", None), 0)
        vm = Machine(ex)
        if self.ablation == "shared_scratch":
            vm.seen, vm.acc = self.shared_seen, self.shared_acc
        for _ in range(256):
            vm.step(self.skills.choose(vm.state(), vm.available()))
            self.skill_steps += 1
            if vm.done:
                break
        if not vm.done:
            self.done = True
            return
        if self.ablation == "shared_scratch":
            self.shared_seen, self.shared_acc = vm.seen, vm.acc
        self.values[self.cursor] = vm.output

    def step(self, a):
        n = self.p.nodes[self.cursor]
        if a in (0, 1):
            self.stack.append(self.cursor)
            self.cursor = n.children[a]
        elif a == 2:
            child = self.cursor
            self.cursor = self.stack.pop()
            if self.ablation == "no_bindings" and self.p.nodes[child].kind != "literal":
                self.values.pop(child, None)
        elif a in (3, 4, 5):
            self.call(a)
        else:
            self.output = self.values.get(self.p.root)
            self.done = True


def run(policy, p, skills, learn=False, epsilon=0, ablation="full", record=False):
    vm = Composer(p, skills, ablation)
    history, shown = [], []
    for step in range(160):
        s = vm.state()
        avail = vm.available()
        if hasattr(policy, "dispatch"):
            s = ("control",) + s[1:]
            avail = [a for a in avail if a not in (4, 5)]
        a = policy.choose(s, avail, epsilon)
        actual = policy.dispatch.choose((vm.p.nodes[vm.cursor].kind,), [3, 4, 5]) if hasattr(policy, "dispatch") and a == 3 else a
        if record:
            shown.append({"node": vm.cursor, "state": s, "action": OPS[actual]})
        vm.step(actual)
        ns, navail = vm.state(), vm.available()
        if hasattr(policy, "dispatch"):
            ns = ("control",) + ns[1:]
            navail = [a for a in navail if a not in (4, 5)]
        history.append((s, a, ns, navail))
        if vm.done:
            break
    correct = vm.done and vm.output == p.answer
    if learn:
        for i in range(len(history) - 1, -1, -1):
            s, a, ns, avail = history[i]
            y = (1.0 if correct else -1.0) if i == len(history) - 1 else -0.005 + 0.99 * max(policy.values(ns)[j] for j in avail)
            q = policy.values(s)
            q[a] += 0.10 * (y - q[a])
    return {"correct": correct, "steps": len(history), "skill_steps": vm.skill_steps,
            "timeout": not vm.done, "trace": shown, "output": vm.output}


def train(seed, n, skills, regime="composite"):
    p, rng = Controller(seed + 121), random.Random(seed + 330)
    families = ["atomic_follow", "atomic_count", "atomic_equal"]
    if regime != "atomic":
        families += ["follow_follow", "equal_follow"]
    if regime.startswith("factorized"):
        # Learn which skill a goal cue denotes from single-goal outcomes, not
        # from a coded name -> operation switch in the controller.
        p.dispatch = Controller(seed + 555)
        for i in range(600):
            ex = problem(rng, families[i % 3], i)
            vm = Composer(ex, skills)
            s = (ex.nodes[ex.root].kind,)
            a = p.dispatch.choose(s, [3, 4, 5], 0.3)
            vm.step(a)
            r = 1.0 if vm.values.get(ex.root) == ex.answer else -1.0
            q = p.dispatch.values(s)
            q[a] += 0.1 * (r - q[a])
    curves = []
    hits = 0
    for i in range(n):
        fam = families[i % len(families)]
        depth = 1 if fam == "equal_follow" else (rng.choice([2, 3]) if regime == "factorized_deep" else 2)
        ex = problem(rng, fam, i, length=rng.randint(1, 3), depth=depth)
        eps = max(0.02, 0.4 * (1 - i / n))
        hits += run(p, ex, skills, True, eps)["correct"]
        if (i + 1) % max(1, n // 10) == 0:
            curves.append([i + 1, hits / max(1, n // 10)])
            hits = 0
    return p, curves


def evaluate(policy, skills, seed, n=100, ablation="full"):
    rng = random.Random(90000 + seed)
    before = policy.digest(), skills.digest()
    groups = [("atomic_follow", 0), ("atomic_count", 0), ("atomic_equal", 0),
              ("follow_follow", 2), ("equal_follow", 1), ("count_follow", 1),
              ("equal_count", 0), ("deep_follow", 6), ("deep_count_follow", 6),
              ("deep_equal_follow", 4), ("equal_count_follow", 4)]
    metrics, traces = {}, {}
    for fam, depth in groups:
        row = dict(correct=0, n=0, steps=0, skill_steps=0, timeouts=0)
        for i in range(n):
            ex = problem(rng, fam, i, length=rng.choice([4, 6, 8, 12]), depth=depth, number=rng.randint(4, 12))
            r = run(policy, ex, skills, ablation=ablation, record=i == 0)
            row["correct"] += r["correct"]
            row["n"] += 1
            row["steps"] += r["steps"]
            row["skill_steps"] += r["skill_steps"]
            row["timeouts"] += r["timeout"]
            if i == 0:
                traces[fam] = r
        metrics[fam] = row
    assert before == (policy.digest(), skills.digest())
    return metrics, traces


def micro_obs(vm):
    o = [0] * 32
    o[TASKS.index(vm.ex.task)] = 1
    flags = [vm.focus is not None, vm.value is not None, vm.focus == vm.ex.subject,
             vm.exhausted, vm.value is not None and vm.value in vm.seen, vm.acc == 0]
    for i, flag in enumerate(flags):
        o[3 + 2 * i + int(flag)] = 1
    o[15 + (0 if vm.predicate is None else 2 if vm.predicate else 1)] = 1
    o[18 + vm.last + 1] = 1
    o[31] = 1
    return o


def control_obs(s):
    o = [0] * 32
    for i, flag in enumerate(s[1:]):
        o[2 * i + int(flag)] = 1
    o[31] = 1
    return o


def dispatch_obs(kind):
    o = [0] * 32
    o[{"follow": 0, "count": 1, "equal": 2, "literal": 3}[kind]] = 1
    o[31] = 1
    return o


def export_native(out, seed, policy, skills, test_n, episodes=20000):
    """Only successful TRAIN trajectories teach the native composition routes.

    The micro-skill teacher is the same explicit primitive supervision as above.
    The compositor teacher is a reward-trained policy; no composite oracle.
    """
    def sample(f, obs, available, action):
        f.write(" ".join(map(str, [action, len(available)] + available + obs)) + "\n")

    rng = random.Random(seed + 44)
    with (out / f"micro_{seed}.txt").open("w") as f:
        for i in range(600):
            task = TASKS[i % 3]
            size = 0 if task == "compare" else rng.randint(0 if task == "count" else 1, 3)
            vm = Machine(example(rng, task, size, "skill", i))
            while not vm.done:
                a = oracle_action(vm)
                sample(f, micro_obs(vm), vm.available(), a)
                vm.step(a)
    with (out / f"dispatch_{seed}.txt").open("w") as f:
        for _ in range(100):
            for kind in ("follow", "count", "equal"):
                sample(f, dispatch_obs(kind), [3, 4, 5], policy.dispatch.choose((kind,), [3, 4, 5]))
    rng = random.Random(40400 + seed)
    with (out / f"control_{seed}.txt").open("w") as f:
        for i in range(1000):
            fam = ("atomic_follow", "atomic_count", "atomic_equal", "follow_follow", "equal_follow")[i % 5]
            depth = 1 if fam == "equal_follow" else rng.choice([2, 3])
            ex = problem(rng, fam, i, length=rng.randint(1, 3), depth=depth)
            result = run(policy, ex, skills, record=True)
            if not result["correct"]:
                continue
            for rec in result["trace"]:
                s = rec["state"]
                avail = [3, 6] + ([0] if s[1] else []) + ([1] if s[3] else []) + ([2] if s[6] else [])
                a = OPS.index(rec["action"])
                if a in (4, 5):
                    a = 3
                sample(f, control_obs(s), avail, a)
    def encode(ex):
        assert reference(ex) == ex.answer, "incorrect benchmark label"
        names = list(dict.fromkeys([v for a, r, b in ex.facts for v in (a, r, b)] +
                                   [n.literal[1] for n in ex.nodes if n.literal and n.literal[0] == "entity"]))
        ids = {v: j for j, v in enumerate(names)}
        def val(v):
            return [{"entity": 0, "bool": 1, "int": 2}[v[0]], ids[v[1]] if v[0] == "entity" else int(v[1])]
        fields = [ex.family, len(ex.nodes), ex.root, len(ex.facts)] + val(ex.answer)
        for n in ex.nodes:
            children = list(n.children) + [-1, -1]
            fields += [{"follow": 0, "count": 1, "equal": 2, "literal": 3}[n.kind],
                       children[0], children[1], ids.get(n.relation, -1)]
            fields += val(n.literal) if n.literal else [-1, -1]
        fields += [v for a, r, b in ex.facts for v in (ids[a], ids[r], ids[b])]
        return " ".join(map(str, fields)) + "\n"
    rng = random.Random(55000 + seed)
    with (out / f"train_{seed}.txt").open("w") as f:
        for i in range(episodes):
            fam = ("atomic_follow", "atomic_count", "atomic_equal", "follow_follow", "equal_follow")[i % 5]
            depth = 1 if fam == "equal_follow" else rng.choice([2, 3])
            f.write(encode(problem(rng, fam, i, length=rng.randint(1, 3), depth=depth)))
    rng = random.Random(90000 + seed)
    groups = [("atomic_follow", 0), ("atomic_count", 0), ("atomic_equal", 0),
              ("follow_follow", 2), ("equal_follow", 1), ("count_follow", 1),
              ("equal_count", 0), ("deep_follow", 6), ("deep_count_follow", 6),
              ("deep_equal_follow", 4), ("equal_count_follow", 4)]
    with (out / f"test_{seed}.txt").open("w") as f:
        for family, depth in groups:
            for i in range(test_n):
                ex = problem(rng, family, i, length=rng.choice([4, 6, 8, 12]), depth=depth, number=rng.randint(4, 12))
                f.write(encode(ex))
    rng = random.Random(220000 + seed)
    with (out / f"stress_{seed}.txt").open("w") as f:
        for fam in ("deep_follow", "deep_count_follow", "deep_equal_follow", "equal_count_follow"):
            for depth in (8, 12):
                for length in (16, 32):
                    for i in range(30):
                        ex = problem(rng, fam, i, length=length, depth=depth, number=24)
                        f.write(encode(Problem(ex.nodes, ex.root, ex.facts, ex.answer, f"{fam}_d{depth}_l{length}")))
        for i in range(100):
            ex = problem(rng, "equal_count_follow", i, length=12, depth=6, number=8)
            root = ex.nodes[ex.root]
            left = ex.nodes[root.children[0]]
            owner = reference(ex, left.children[0])[1]
            extra = (owner, left.relation, f"new-evidence/{seed}/{i}")
            changed = Problem(ex.nodes, ex.root, ex.facts + (extra, extra),
                              ("bool", not ex.answer[1]), "counterfactual_changed")
            f.write(encode(Problem(ex.nodes, ex.root, ex.facts, ex.answer, "counterfactual_original")))
            f.write(encode(changed))
            nodes = list(ex.nodes)
            nodes[ex.root] = Node("equal", (root.children[0], root.children[0]))
            shared = Problem(tuple(nodes), ex.root, ex.facts, ("bool", True), "shared_subgoal")
            f.write(encode(shared))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-n", type=int, default=100)
    ap.add_argument("--out", default="logs/reasoning_composition")
    ap.add_argument("--export-native", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {"config": vars(args), "runs": [], "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    start = time.time()
    for seed in range(args.seeds):
        skills = train_skills(seed)
        for regime in ("atomic", "composite", "factorized", "factorized_deep"):
            p, curve = train(seed, args.episodes, skills, regime)
            metrics, traces = evaluate(p, skills, seed, args.test_n)
            r = {"seed": seed, "regime": regime, "metrics": metrics, "curve": curve, "traces": traces}
            if regime != "atomic":
                r["ablations"] = {a: evaluate(p, skills, seed, args.test_n, a)[0]
                                  for a in ("no_bindings", "root_binding", "shared_scratch")}
            data["runs"].append(r)
            data["seconds"] = time.time() - start
            (out / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            (out / f"policy_{seed}_{regime}.json").write_text(json.dumps([{"state": s, "values": v} for s, v in p.q.items()]), encoding="utf-8")
            if hasattr(p, "dispatch"):
                (out / f"dispatch_{seed}_{regime}.json").write_text(json.dumps([{"state": s, "values": v} for s, v in p.dispatch.q.items()]), encoding="utf-8")
            if args.export_native and regime == "factorized_deep":
                export_native(out, seed, p, skills, args.test_n, args.episodes)
            print(json.dumps({"seed": seed, "regime": regime, "accuracy": {k: v["correct"] / v["n"] for k, v in metrics.items()}, "seconds": round(time.time() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()
