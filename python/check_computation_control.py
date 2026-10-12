"""Independent transition/outcome/cost audit; evaluator truth never enters policy."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/computation_control"


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def mapping(text):
    return {int(a): int(b) for a, b in (entry.split(":") for entry in text.rstrip(";").split(";"))}


def audit(seed, mode):
    prefix = OUT / f"{seed}.{mode}"
    cases = rows(Path(str(prefix) + ".cases.tsv"))
    steps = collections.defaultdict(list)
    for step in rows(Path(str(prefix) + ".steps.tsv")):
        steps[int(step["trial"])].append(step)
    assert len(cases) == 1200 and len(steps) == 1200
    groups = collections.defaultdict(collections.Counter)
    decisions = collections.defaultdict(collections.Counter)
    for trial, case in enumerate(cases):
        assert trial == int(case["trial"])
        length, effect, price, cached, disabled, root, goal = (int(case[k]) for k in ("length", "effect", "price", "cached", "disabled", "root", "goal"))
        truth, archive = mapping(case["truth"]), mapping(case["archive"])
        assert all(a >= 100000 for a in truth) and all(truth[a] == b for a, b in archive.items())
        path = [root]
        while truth[path[-1]] >= 0:
            path.append(truth[path[-1]])
            assert len(path) <= length + 1
        assert len(path) == length + 1 and path[-1] == goal
        costs = [0.05 if price else 0.01, 0.005, 0.01 if price else 0.05, 0.02, 0]
        focus, next_value, scope, marker, done, cost = root, -1, 0, -1, False, 0
        if cached:
            next_value = truth[focus];scope = 2 if next_value < 0 else 1
        counts = collections.Counter()
        for t, row in enumerate(steps.pop(trial)):
            assert not done
            assert [int(row[k]) for k in ("step", "focus", "scope", "next", "marker")] == [t, focus, scope, next_value, marker]
            action = int(row["action"])
            assert 0 <= action < 5 and action != disabled and (action != 1 or scope == 1)
            assert abs(float(row["cost"]) - costs[action]) < 1e-6
            if disabled == -1:
                decisions[f"scope={scope},effect={effect},marked={marker==focus},price={price}"][str(action)] += 1
            cost += costs[action];counts[str(action)] += 1
            if action in (0, 2):
                store = archive if action == 0 else truth
                if focus in store:
                    next_value = store[focus];scope = 2 if next_value < 0 else 1
                elif scope not in (1, 2): scope = 3
            elif action == 1: focus, next_value, scope = next_value, -1, 0
            elif action == 3: marker = focus
            else: done = True
        correct = done and focus == goal and (not effect or marker == goal)
        assert correct == bool(int(case["correct"])) and (not done) == bool(int(case["timeout"]))
        assert sum(counts.values()) == int(case["steps"]) and abs(cost - float(case["cost"])) < 1e-4
        for group in (f"disabled={disabled}", f"disabled={disabled},length={length}", f"disabled={disabled},price={price},effect={effect}"):
            score = groups[group];score["n"] += 1;score["correct"] += correct;score["timeout"] += not done
            score["steps"] += sum(counts.values());score["cost"] += cost
            for a, n in counts.items(): score[f"action_{a}"] += n
            # Evaluator-only comparison: a fixed public-state policy retrieving
            # before asking at cheap-retrieval prices, asking directly otherwise.
            acquisition = sum(0 if i == 0 and cached else costs[2] if price else costs[0] + (0 if node in archive else costs[2]) for i, node in enumerate(path))
            score["reference_cost"] += acquisition + length*costs[1] + effect*costs[3]
        if disabled == 1 and length > 0: assert not correct
        if disabled == 3 and effect: assert not correct
    assert not steps
    weights = [float(v) for v in Path(str(prefix) + ".weights.txt").read_text().splitlines()]
    initial = [float(v) for v in Path(str(prefix) + ".initial_weights.txt").read_text().splitlines()]
    assert len(weights) == len(initial) == (1221 if mode.startswith("backprop") else 5*512*3)
    assert (weights == initial) == ("frozen" in mode)
    result = dict(groups={k:dict(v) for k,v in groups.items()},decisions={k:dict(v) for k,v in decisions.items()},weight_l1_change=sum(abs(a-b) for a,b in zip(weights,initial)))
    print(seed, mode, dict(groups["disabled=-1"]), flush=True)
    return result


def main():
    results = {f"{seed}.{mode}":audit(seed,mode) for seed in range(3) for mode in ("learned","frozen","shuffled","terminal","backprop","backprop_frozen","backprop_shuffled","backprop_terminal")}
    paths = [Path(__file__), ROOT/"cpp/apps/computation_control_probe.cpp", ROOT/"cpp/include/laminar/computation_control.h", ROOT/"cpp/include/laminar/backprop_choice.h", ROOT/"cpp/tests/test_backprop_choice.cpp", ROOT/"cpp/include/laminar/reasoning_runtime.h", ROOT/"cpp/include/laminar/agency.h", ROOT/"build-wsl/computation_control_probe", ROOT/"build-asan/test_backprop_choice", ROOT/"scripts/computation_control.sh"]
    paths += [p for p in OUT.iterdir() if p.suffix in (".tsv",".txt")]
    (OUT/"audit.json").write_text(json.dumps(dict(results=results,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+"\n")


if __name__ == "__main__":main()
