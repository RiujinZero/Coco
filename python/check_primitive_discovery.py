"""Independent primitive VM, discovered-policy and composition audit."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/primitive_discovery"
ABSENT = (-1, -1)


def rows(path):
    with path.open() as f:
        yield from csv.DictReader(f, delimiter="\t")


def policy(path):
    lines = path.read_text().splitlines()
    assert lines[0].split()[1] == "1"
    result = dict(tuple(map(int, line.split())) for line in lines[1:])
    assert len(result) == len(lines) - 1
    return result


def gold(task, subject, target, relation, facts):
    if task == 1:
        return (1, int(subject == target))
    if task == 2:
        return (2, len({b for a, r, b in facts if a == subject[1] and r == relation}))
    focus = subject[1]
    visited = set()
    while True:
        assert focus not in visited
        visited.add(focus)
        values = {b for a, r, b in facts if a == focus and r == relation}
        if not values:
            return (0, focus)
        assert len(values) == 1
        focus = next(iter(values))


def key(task, focus, value, subject, exhausted, seen, acc, pred, last):
    result = (1 << task) | (1 << 31) | (1 << (16 + pred)) | (1 << (19 + last))
    flags = (focus[0] >= 0, value[0] >= 0, focus == subject, exhausted, value in seen, acc == 0)
    for i, flag in enumerate(flags):
        result |= 1 << (3 + 2*i + int(flag))
    return result


def audit(prefix, rules=None, source_policies=None, training_budget=48):
    scores = collections.defaultdict(collections.Counter)
    observed_training = {}
    self_decisions = collections.defaultdict(dict)
    with Path(str(prefix) + ".decisions.tsv").open() as f:
        decisions = iter(csv.DictReader(f, delimiter="\t"))
        current = next(decisions, None)
        cases = 0
        for case in rows(Path(str(prefix) + ".cases.tsv")):
            serial = int(case["case"])
            assert serial == cases
            cases += 1
            task, size, relation, disabled = (int(case[k]) for k in ("task", "size", "relation", "disabled"))
            subject = (int(case["subject_type"]), int(case["subject"]))
            target = (int(case["target_type"]), int(case["target"]))
            facts = [tuple(map(int, t.split(","))) for t in case["facts"].rstrip(";").split(";") if t]
            expected = gold(task, subject, target, relation, facts)
            assert expected == (int(case["gold_type"]), int(case["gold"]))
            if task == 2: assert expected[1] == size
            focus, value, output = ABSENT, ABSENT, ABSENT
            cursor = pred = last = -1
            acc = steps = 0
            exhausted = done = unavailable = False
            seen = set()
            phase = case["phase"]
            model = source_policies[task] if phase == "self_generation" else rules
            available_by_state = {}
            while current is not None and int(current["case"]) == serial:
                assert not done and not unavailable and int(current["step"]) == steps
                state = key(task, focus, value, subject, exhausted, seen, acc, pred, last)
                assert int(current["state"]) == state
                a = int(current["action"])
                allowed = [0, 4, 9]
                if focus[0] >= 0: allowed += [1, 5, 6]
                if value[0] >= 0: allowed += [2, 3, 7]
                if pred >= 0: allowed += [8]
                if disabled in allowed: allowed.remove(disabled)
                if model is not None:
                    wanted = model.get(state, -1)
                    if wanted not in allowed: wanted = -1
                    assert a == wanted, (prefix, serial, state, a, wanted)
                current = next(decisions, None)
                if a == -1:
                    unavailable = True
                    continue
                assert a in allowed
                if phase == "training": observed_training[state] = a
                if phase == "self_generation":
                    pair = (a, tuple(allowed))
                    assert state not in self_decisions[task] or self_decisions[task][state] == pair
                    self_decisions[task][state] = pair
                if a == 0:
                    focus, value, cursor, exhausted = subject, ABSENT, -1, False
                elif a == 1:
                    value = ABSENT
                    begin = 0 if disabled == -2 else cursor + 1
                    if task != 1 and focus[0] == 0:
                        for j in range(begin, len(facts)):
                            src, rel, dst = facts[j]
                            if src == focus[1] and rel == relation:
                                cursor, value = j, (0, dst)
                                break
                    exhausted = value == ABSENT
                    if exhausted: cursor = len(facts)
                elif a == 2: focus, value, cursor, exhausted = value, ABSENT, -1, False
                elif a == 3: seen.add(value)
                elif a == 4: acc += 1
                elif a == 5: pred = int(focus == target)
                elif a == 6: output = focus
                elif a == 7: output = value
                elif a == 8: output = (1, pred)
                elif a == 9: output = (2, acc)
                last, done = a, a >= 6
                steps += 1
            assert output == (int(case["output_type"]), int(case["output"])), (prefix, serial, output, case)
            assert steps == int(case["steps"]) and unavailable == bool(int(case["unavailable"]))
            assert (not done and not unavailable) == bool(int(case["timeout"]))
            budget = training_budget if phase == "training" else 128 if phase == "self_generation" else 1024
            assert steps <= budget
            if not done and not unavailable: assert steps == budget
            for group in (f"{phase}/task{task}/disabled{disabled}", f"{phase}/task{task}/size{size}/disabled{disabled}"):
                score = scores[group]
                score["n"] += 1
                score["correct"] += output == expected
                score["steps"] += steps
                score["timeout"] += not done and not unavailable
                score["unavailable"] += unavailable
                score["max_steps"] = max(score["max_steps"], steps)
        assert current is None
    if rules is not None:
        assert observed_training == rules
    print(prefix.name, {k: dict(v) for k, v in scores.items() if "/size" not in k and k.endswith("disabled-1")}, flush=True)
    return {k: dict(v) for k, v in scores.items()}, self_decisions


def flat(path):
    out = []
    for line in path.read_text().splitlines():
        v = list(map(int, line.split()))
        cue, a, b, r0, r1, typ, answer, n = v[:8]
        assert len(v) == 8 + n*3
        facts = list(zip(v[8::3], v[9::3], v[10::3]))
        out.append(dict(cue=cue, a=(0,a), b=(0,b), relations=(r0,r1), facts=facts, answer=(typ,answer)))
    return out


def flat_gold(q):
    a, b, facts, (r0,r1) = q["a"], q["b"], q["facts"], q["relations"]
    follow = lambda v: gold(0, v, ABSENT, r0, facts)
    count = lambda v: gold(2, v, ABSENT, r1, facts)
    cue = q["cue"]
    if cue == 0: return follow(a)
    if cue == 1: return count(a)
    if cue == 2: return (1, int(a == b))
    if cue == 3: return count(follow(a))
    if cue == 4: return (1, int(count(a) == count(b)))
    if cue == 5: return (1, int(count(follow(a)) == count(follow(b))))
    assert cue == 6
    return (1, int(follow(a) == follow(b)))


def execute(q, program):
    regs = [q["a"], q["b"]]
    for op in program:
        if op == 0: regs[0] = q["a"]
        elif op == 1: regs[0] = q["b"]
        elif op == 2: regs[1] = regs[0]
        elif op == 3: regs.reverse()
        elif op in (4,5): regs[0] = gold(0, regs[0], ABSENT, q["relations"][op%2], q["facts"])
        elif op in (6,7): regs[0] = gold(2, regs[0], ABSENT, q["relations"][op%2], q["facts"])
        elif op == 8: regs[0] = (1, int(regs[0] == regs[1]))
        elif op == 9: return regs[0]
        else: raise AssertionError(op)
    return ABSENT


def transfer(seed, mode):
    prefix = OUT / f"{seed}.{mode}"
    models = [policy(OUT / f"{seed}.{task}.search.policy.txt") for task in range(3)]
    scores, observed = audit(prefix, source_policies=models)
    saved = collections.defaultdict(dict)
    for row in rows(Path(str(prefix) + ".self_samples.tsv")):
        task, state, action = (int(row[k]) for k in ("stage","state","action"))
        available = tuple(map(int,row["available"].rstrip(",").split(",")))
        assert state not in saved[task]
        saved[task][state] = (action, available)
    assert saved == observed
    weight_paths = [Path(str(prefix) + ".initial.weights.txt")] + [Path(str(prefix) + f".stage{t}.weights.txt") for t in range(3)]
    weights = [[float(x) for x in p.read_text().splitlines()] for p in weight_paths]
    assert all(len(w) == 10*512*3 and all(map(math.isfinite,w)) for w in weights)
    assert not any(weights[0]) and all(a != b for a,b in zip(weights,weights[1:]))
    programs = {}
    for row in rows(Path(str(prefix) + ".programs.tsv")):
        cue = int(row["cue"])
        p = list(map(int,row["instructions"].rstrip(",").split(","))) if row["instructions"] else []
        assert bool(p) == bool(int(row["found"])) and len(p) <= 8
        programs[cue] = p
    assert set(programs) == set(range(7))
    training = flat(ROOT/f"logs/reasoning_flat_corpus/train_{seed}.txt")
    tests = flat(ROOT/f"logs/reasoning_flat_corpus/test_{seed}.txt")
    assert len(training) == 84 and len(tests) == 1400
    assert all(flat_gold(q) == q["answer"] for q in training + tests)
    if mode == "replay":
        assert all(execute(q,programs[q["cue"]]) == q["answer"] for q in training)
    records = list(rows(Path(str(prefix) + ".composition.tsv")))
    assert len(records) == len(tests)
    composition = collections.defaultdict(collections.Counter)
    for i, (q, row) in enumerate(zip(tests,records)):
        assert (int(row["case"]),int(row["cue"])) == (i,q["cue"])
        output = (int(row["output_type"]),int(row["output"]))
        if mode == "replay": assert output == execute(q,programs[q["cue"]])
        composition[q["cue"]]["n"] += 1
        composition[q["cue"]]["correct"] += output == q["answer"]
    print(prefix.name, "composition", dict(composition), flush=True)
    return dict(primitives=scores,composition={k:dict(v) for k,v in composition.items()},
                weight_changes=[sum(abs(x-y) for x,y in zip(a,b)) for a,b in zip(weights,weights[1:])])


def main():
    results = {}
    prefixes = []
    for seed in range(3):
        for task in range(3):
            for mode in (("search","first","terminal","random","frozen","backprop") if task == 2 else ("search",)):
                prefix = OUT / f"{seed}.{task}.{mode}"
                model = None if mode == "backprop" else policy(Path(str(prefix)+".policy.txt"))
                scores, _ = audit(prefix, model)
                results[prefix.name] = scores
                prefixes.append(prefix)
                if mode != "backprop":
                    curve = list(rows(Path(str(prefix)+".curve.tsv")))
                    count = scores[f"training/task{task}/disabled-1"]["correct"]
                    assert int(curve[-1]["total"]) == 32
                    if mode != "random": assert int(curve[-1]["correct"]) == count
        for mode in ("replay","no_replay"):
            results[f"{seed}.{mode}"] = transfer(seed,mode)
            prefixes.append(OUT/f"{seed}.{mode}")
        samples = [(OUT/f"{seed}.{mode}.self_samples.tsv").read_bytes() for mode in ("replay","no_replay")]
        assert samples[0] == samples[1], "rehearsal arms received different self-generated action examples"
        contexts = [[r for r in rows(OUT/f"{seed}.{mode}.cases.tsv") if r["phase"] == "self_generation"] for mode in ("replay","no_replay")]
        assert contexts[0] == contexts[1], "rehearsal changed the new-skill world stream"
    paths = [ROOT/p for p in (
        "cpp/apps/primitive_discovery_probe.cpp","cpp/apps/primitive_discovery_support.h","cpp/apps/primitive_transfer_probe.cpp",
        "cpp/include/laminar/policy_discovery.h","cpp/include/laminar/reasoning_runtime.h","cpp/include/laminar/agency.h",
        "cpp/include/laminar/backprop_choice.h","cpp/include/laminar/procedure_memory.h","cpp/tests/test_policy_discovery.cpp",
        "build-wsl/primitive_discovery_probe","build-wsl/primitive_transfer_probe","build-asan/test_policy_discovery",
        "scripts/primitive_discovery.sh","cpp/CMakeLists.txt","scripts/build_wsl.sh",
        "python/reasoning_flat_probe.py","python/reasoning_primitives_probe.py","python/reasoning_composition_probe.py")]
    paths.append(Path(__file__))
    for prefix in prefixes: paths += sorted(prefix.parent.glob(prefix.name + ".*"))
    for seed in range(3):
        paths += [ROOT/f"logs/reasoning_flat_corpus/{split}_{seed}.txt" for split in ("train","test")]
    (OUT/"audit.json").write_text(json.dumps(dict(results=results,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+"\n")


if __name__ == "__main__": main()
