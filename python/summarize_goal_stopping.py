"""Separate physical completion, policy-issued stopping and mission success.

Read-only, for the declared six-edge public-root environment. Reconstructs the
physical route from public initial facts and acknowledged environment actions.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def summarize(prefix):
    missions = {row["id"]: row for row in rows(Path(str(prefix) + "_missions.jsonl"))}
    metrics = {key: {"arm": value["arm"], "mission_success": value["success"],
                     "physical_satisfied_at": None, "physical_finished_at": None,
                     "physical_correct_finish": False, "urgent_correct_ontime": False,
                     "known_satisfied_nonfinish_operations": Counter(),
                     "physical_operations_after_satisfaction": Counter()}
               for key, value in missions.items()}
    worlds = {}
    deadlines = {}
    required = {("n" + str(i), "n" + str(i + 1)) for i in range(6)}
    for row in rows(Path(str(prefix) + "_trace.jsonl")):
        key, event = row["mission"], row["event"]
        if event == "mission_begin":
            assert row["family"] == "raw_six_edge_public_root_protocol", "unsupported task family"
            worlds[key] = set()
            deadlines[key] = {}
        elif event == "public_request":
            deadlines[key][row["id"]] = row["deadline"]
            if row["id"] == "physical":
                worlds[key] = {tuple(fact["args"]) for fact in row["visible"]["facts"]
                               if fact["predicate"] == "edge"}
                assert not required <= worlds[key], "six-edge root already satisfied"
        elif event == "decision":
            item = metrics[key]
            if row["active_before"] == "physical":
                if item["physical_satisfied_at"] is not None:
                    item["physical_operations_after_satisfaction"][str(row["operation"])] += 1
                if row["observation"][6] and not row["observation"][29] and row["operation"] != 4:
                    item["known_satisfied_nonfinish_operations"][str(row["operation"])] += 1
            if row["operation"] == 3 and row["actions"]:
                assert row["callback_source"] == "verified_environment"
                for action, executed in zip(row["actions"], row["executed"], strict=True):
                    assert action["schema"] == "edge_motor"
                    if executed:
                        worlds[key].add(tuple(action["args"]))
                if required <= worlds[key] and item["physical_satisfied_at"] is None:
                    item["physical_satisfied_at"] = row["clock"]
            if row["operation"] == 4:
                if row["active_before"] == "physical":
                    assert row["terminal_satisfaction"] == (required <= worlds[key])
                    item["physical_finished_at"] = row["clock"]
                    item["physical_correct_finish"] = row["terminal_correct"] and row["terminal_satisfaction"]
                elif row["active_before"] == "arrival":
                    item["urgent_correct_ontime"] = row["terminal_correct"] and row["clock"] <= deadlines[key]["arrival"]
    aggregate = defaultdict(Counter)
    for key, item in metrics.items():
        group = aggregate[item["arm"]]
        group["missions"] += 1
        group["mission_successes"] += item["mission_success"]
        group["physical_world_completed"] += item["physical_satisfied_at"] is not None
        group["physical_correct_finishes"] += item["physical_correct_finish"]
        group["urgent_correct_ontime"] += item["urgent_correct_ontime"]
        group["completed_world_without_finish"] += item["physical_satisfied_at"] is not None and item["physical_finished_at"] is None
        group["known_satisfied_nonfinish_decisions"] += sum(item["known_satisfied_nonfinish_operations"].values())
        if item["physical_satisfied_at"] is not None and item["physical_finished_at"] is not None:
            item["decisions_until_physical_finish"] = item["physical_finished_at"] - item["physical_satisfied_at"]
        if item["mission_success"]:
            assert item["physical_correct_finish"] and item["urgent_correct_ontime"], "success bypassed learned completion"
    return {"schema": "goal_stopping_summary_v1", "prefix": str(prefix),
            "scope": "Six-edge public-root task; no host completion rule introduced. Physical satisfaction, cached belief and actual policy finish are distinct.",
            "arms": dict(aggregate), "missions": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "refusing to overwrite previous stopping analysis"
    report = summarize(args.prefix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["arms"], indent=2))
