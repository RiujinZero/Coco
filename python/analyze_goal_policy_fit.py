"""Read-only diagnostic of the frozen goal pilot's neural policy fit.

Compares neural choices with logged warm assistance, never uses those labels to
train. Python uses double arithmetic; exported evaluation choices are checked
and near-tie disagreements would be reported separately.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def context(observation):
    if observation[29]:
        return "agenda"
    if observation[30]:
        return "execution_with_uninspected_arrival"
    return "execution_without_uninspected_arrival"


def scores(policy, observation):
    width, weights = policy["hidden"], policy["weights"]
    hidden = [math.tanh(weights[width * 32 + j] + sum(
        weights[j * 32 + i] * value for i, value in enumerate(observation)))
        for j in range(width)]
    output_start = width * 33
    bias_start = output_start + policy["actions"] * width
    return [weights[bias_start + action] + sum(
        weights[output_start + action * width + j] * value
        for j, value in enumerate(hidden)) for action in range(policy["actions"])]


def analyze(root):
    labels = {"initial": "before", "warm100": "warm100",
              "adaptive_m299": "adaptive300",
              "frozen_policy_m299": "frozen_policy300",
              "random_credit_m299": "random_credit300"}
    policies = {labels[row["label"]]: row for row in rows(root / "pilot_weights.jsonl")
                if row["label"] in labels}
    assert len(policies) == len(labels), "missing checkpoint weights"
    support = {}
    evaluation = defaultdict(list)
    coverage = defaultdict(Counter)
    for row in rows(root / "pilot_trace.jsonl"):
        if row["event"] != "decision":
            continue
        key = (row["observation_bits"], tuple(row["menu"]))
        if row["mission"].startswith("warm_") and row["teacher"]:
            if key not in support:
                support[key] = {"observation": row["observation"],
                                "menu": row["menu"], "operations": Counter(), "count": 0}
            support[key]["operations"][row["operation"]] += 1
            support[key]["count"] += 1
        if row["mission"].startswith("eval_"):
            checkpoint = row["mission"][5:].rsplit("_m", 1)[0]
            evaluation[checkpoint].append((key, row["observation"], row["menu"], row["operation"]))
    assert all(len(row["operations"]) == 1 for row in support.values()), "contradictory assistance"
    report = {"schema": "goal_policy_fit_diagnostic_v1", "assisted_unique_states": len(support),
              "assisted_decisions": sum(row["count"] for row in support.values()),
              "policies": {}, "scope": "Read-only fit diagnosis; assistance agreement is not acquisition or optimality."}
    for label, policy in policies.items():
        cache = {}
        def choose(key, observation, menu):
            if key not in cache:
                values = scores(policy, observation)
                ordered = sorted(menu, key=lambda action: values[action], reverse=True)
                cache[key] = ordered[0], (values[ordered[0]] - values[ordered[1]] if len(ordered) > 1 else None)
            return cache[key]
        groups = defaultdict(Counter)
        disagreements = []
        for key, row in support.items():
            choice, margin = choose(key, row["observation"], row["menu"])
            assisted = next(iter(row["operations"]))
            group = groups[context(row["observation"])]
            group["unique_states"] += 1
            group["decisions"] += row["count"]
            group["matching_unique_states"] += choice == assisted
            group["matching_decisions"] += row["count"] * (choice == assisted)
            group[f"neural_operation_{choice}"] += row["count"]
            if choice != assisted and row["observation"][30] and not row["observation"][29]:
                disagreements.append({"observation_bits": key[0], "neural": choice,
                                      "assisted": assisted, "margin": margin, "count": row["count"]})
        parity = Counter()
        for key, observation, menu, actual in evaluation[label]:
            choice, margin = choose(key, observation, menu)
            parity["decisions"] += 1
            parity["matching_choices"] += choice == actual
            if choice != actual:
                parity["near_tie_disagreements" if margin is not None and margin < 1e-5 else "clear_disagreements"] += 1
            coverage[label][context(observation) + ("_seen_assisted" if key in support else "_not_seen_assisted")] += 1
        assert not parity["clear_disagreements"], "neural decoder disagrees with exported evaluation"
        report["policies"][label] = {"fit_on_warm_assistance": dict(groups),
                                     "arrival_disagreements": disagreements,
                                     "evaluation_decoder_parity": parity,
                                     "evaluation_support": coverage[label]}
    report["input_sha256"] = {name: digest(root / name) for name in
                              ("pilot_config.json", "pilot_weights.jsonl", "pilot_trace.jsonl")}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists(), "refusing to overwrite earlier diagnostic"
    result = analyze(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: data["fit_on_warm_assistance"] for label, data in result["policies"].items()}, indent=2))
