"""Preserve compact completed-run records for the usage-limited handoff.

This only reads existing artifacts. It runs no learner or new experiment.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line]


def record(kind):
    root = Path(f"logs/goal_continuing_bootstrap_{kind}_20260923_v1")
    prefix = root / kind
    config = load(str(prefix) + "_config.json")
    summary = load(str(prefix) + "_summary.json")
    evaluations = lines(str(prefix) + "_evaluations.jsonl")
    missions = lines(str(prefix) + "_missions.jsonl")
    phases = lines(str(prefix) + "_phases.jsonl")
    completed = load(root / "completed_evidence.json")
    sources = load(root / "source_manifest.json")
    assert summary["complete"] and completed["missions"] == len(missions)
    for row in completed["files"]:
        path = root / row["path"]
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], path
    for row in sources["files"]:
        assert sha(root / row["path"]) == row["sha256"], row
    training = [m for m in missions if m["learning"]]
    frozen = [m for m in missions if not m["learning"]]
    counts = Counter(m["arm"] for m in missions)
    assert len(training) == config["warm_missions"] + config["autonomous_missions"]
    assert len(frozen) == sum(e["cases"] for e in evaluations)
    assert len({m["id"] for m in missions}) == len(missions)
    assert all(m["updates_after"] - m["updates_before"] == config["terminal_updates"] for m in training)
    assert all(m["hints"] == 0 and m["updates_after"] == m["updates_before"]
               and m["credits_after"] == m["credits_before"] and m["model_after"] == m["model_before"] for m in frozen)
    assert all(m["hints"] == 0 for m in missions if m["arm"] == "autonomous")
    fits = [p for p in phases if p["event"] == "demonstration_fit"]
    assert all(p["fits"] == p["representatives"] * p["rounds"] and p["empirical_values_unchanged"] for p in fits)
    assert sum(p["fits"] for p in fits) == summary["demonstration_fits"] <= config["fit_cap"]
    assert summary["updates"] == len(training) * config["terminal_updates"] + summary["demonstration_fits"]
    assert summary["credits"] == len(training)
    for evaluation in evaluations:
        group = [m for m in frozen if m["arm"] == "eval_" + evaluation["checkpoint"]]
        assert len(group) == evaluation["cases"]
        assert sum(m["success"] for m in group) == evaluation["successes"]
    extension = [p for p in phases if p.get("stage") in ("before_skill_extension", "after_skill_extension")]
    if kind == "pilot":
        assert len(extension) == 2
        assert extension[0]["updates"] == extension[1]["updates"]
        assert extension[0]["credits"] == extension[1]["credits"] == 100
        assert len(missions) == 738
    descriptive_path = root / "descriptive_results.json"
    descriptive = ({"results": load(descriptive_path), "sha256": sha(descriptive_path)}
                   if descriptive_path.exists() else None)
    return {"directory": str(root), "configuration": config, "summary": summary,
            "evaluations": evaluations, "mission_counts": dict(counts),
            "demonstration_fits": fits, "skill_extension": extension,
            "descriptive_results": descriptive,
            "artifact_manifest": completed, "source_manifest": sources,
            "source_manifest_sha256": sha(root / "source_manifest.json"),
            "consistency_checks_passed": True}


result = {"schema": "continuing_learner_handoff_results_v1",
          "scope": "Completed declared runs and artifact/count consistency; no new trial or independent larger-pilot audit.",
          "smoke": record("smoke"), "pilot": record("pilot")}
target = Path("docs/GOAL_CONTINUING_BOOTSTRAP_RESULTS.json")
target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
for kind in ("smoke", "pilot"):
    run = result[kind]
    print(kind, json.dumps(run["summary"]))
    print("evaluations", json.dumps(run["evaluations"]))
print("record", target, sha(target))
