"""Read-only check for distinct mission histories behind equal public states.

An observed reward/context collision is not a proof that no shared feasible
policy exists. No private diagnostic field is delivered to the controller.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def records(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def analyze(root):
    missions = {row["id"]: row for row in records(root / "pilot_missions.jsonl")
                if row["arm"] == "warm"}
    states = {}
    requests, submissions = {}, {}
    active = ""
    for row in records(root / "pilot_trace.jsonl"):
        if row["mission"] not in missions:
            continue
        if row["mission"] != active:
            active = row["mission"]
            requests, submissions = {}, {}
        if row["event"] == "public_request":
            requests[row["id"]] = row["deadline"]
        if row["event"] != "decision":
            continue
        before_clock = row["clock"] - 1
        failed = any(not outcome for outcome in submissions.values()) or any(
            deadline and before_clock >= deadline and goal not in submissions
            for goal, deadline in requests.items())
        key = row["observation_bits"], tuple(row["menu"])
        if key not in states:
            states[key] = {"observation": row["observation"], "contexts": Counter(),
                           "examples": {}, "action_outcomes": defaultdict(Counter)}
        state = states[key]
        context = "already_failed" if failed else "not_yet_failed"
        state["contexts"][context] += 1
        state["examples"].setdefault(context, {"mission": active, "clock": row["clock"],
                                               "operation": row["operation"],
                                               "mission_success": missions[active]["success"]})
        state["action_outcomes"][str(row["operation"])]["success" if missions[active]["success"] else "failure"] += 1
        if row["operation"] == 4:
            goal = row["active_before"]
            submissions[goal] = row["terminal_correct"] and (
                not requests[goal] or row["clock"] <= requests[goal])
    collisions = []
    satisfied_physical = []
    for (bits, menu), state in states.items():
        item = {"observation_bits": bits, "menu": menu, **state}
        if len(state["contexts"]) > 1:
            collisions.append(item)
        o = state["observation"]
        if not o[29] and o[4] and o[6] and not o[30]:
            satisfied_physical.append(item)
    return {"schema": "goal_credit_context_diagnostic_v1", "warm_missions": len(missions),
            "unique_states": len(states), "context_collision_states": len(collisions),
            "context_collisions": collisions, "satisfied_physical_no_arrival_states": satisfied_physical,
            "scope": "Equal exported observation/menu with different public mission histories; not a proof of incompatible optimal actions or inevitable failure."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "refusing to overwrite earlier diagnostic"
    report = analyze(args.root)
    report["input_sha256"] = {name: hashlib.sha256((args.root / name).read_bytes()).hexdigest()
                               for name in ("pilot_missions.jsonl", "pilot_trace.jsonl")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("warm_missions", "unique_states", "context_collision_states")}))
    print(json.dumps(report["satisfied_physical_no_arrival_states"], indent=2))
