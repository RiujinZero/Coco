"""Independent bounded audit of the frozen continuing mixed-skill smoke."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from audit_goal_learning_pilot import neural_choice
from audit_goal_preflights import apply_actions, bit_key, jsonl, need, near, read_json, scene_facts, sha


def public_demonstration(o, menu):
    action = 5
    if o[29]:
        if 6 not in menu: action = 5
        elif o[12] and o[17] >= o[16]: action = 5
        elif not o[0] or not o[4] or (o[13] and o[11] and o[6]): action = 6
        elif o[13] and o[10] and o[15]: action = 5
        elif (o[13] and o[10]) or not o[5]: action = 1
        elif (not o[13] and (o[6] or o[9])) or (o[13] and o[18] and o[19] > 0): action = 6
    elif o[30]: action = 5
    elif not o[5]: action = 1
    elif (not o[4] and (o[6] or o[7])) or (o[4] and o[6]): action = 4
    elif not o[0] or o[8]: action = 0 if o[12] else 2
    elif o[4] and o[9]: action = 3
    elif o[10] and o[19] < 1: action = 1
    need(action in menu, "public demonstration action unavailable")
    return action


def independent_truth(text, facts):
    tokens = text.split()
    pairs = lambda relation: {(args[0], args[1]) for predicate, args in facts if predicate == relation and len(args) == 2}
    if tokens[0] in {"inspect", "route", "span", "connect"}:
        length = {"inspect": 1, "route": 3, "span": 6, "connect": 9}[tokens[0]]
        frontier = {tokens[1]}
        for _ in range(length):
            frontier = {b for a, b in pairs("edge") if a in frontier}
        return tokens[3] in frontier
    if tokens[0] == "permit":
        return (tokens[1], tokens[2]) in pairs("red") | pairs("blue")
    if tokens[0] == "locate":
        # This declared task generates a positive raw report about these exact
        # arguments. Its truth is not inferred from a potentially partial view.
        need(len(tokens) == 4 and tokens[2] == "with", "reader query schema")
        return True
    need(len(tokens) == 8 and tokens[2] == "beside" and tokens[4] == "across" and tokens[6] == "and", "computation schema")
    left, right = pairs(tokens[5]), pairs(tokens[7])
    count = lambda relation, subject: sum(a == subject for a, b in relation)
    if tokens[0] == "loma": return count(left, tokens[1]) == count(right, tokens[3])
    if tokens[0] == "navi": return count(left, tokens[1]) == int(tokens[3])
    need(tokens[0] == "telo", "unknown computation surface")
    at, visited = tokens[1], set()
    while at not in visited:
        visited.add(at)
        nexts = {b for a, b in left if a == at}
        if not nexts: return at == tokens[3]
        if len(nexts) != 1: return False
        at = next(iter(nexts))
    return False


def audit(root):
    manifest, declaration = read_json(root / "source_manifest.json"), read_json(root / "predeclared_run.json")
    for row in manifest["files"] + declaration["files"]:
        need(sha(root / row["path"]) == row["sha256"], "frozen/predeclared hash: " + row["path"])
    config, summary = read_json(root / "smoke_config.json"), read_json(root / "smoke_summary.json")
    need(config["smoke"] and config["hidden"] == 32 and config["warm_missions"] == 20 and summary["complete"], "wrong smoke variant")
    missions = jsonl(root / "smoke_missions.jsonl")
    traces, computations = defaultdict(list), defaultdict(list)
    for row in jsonl(root / "smoke_trace.jsonl"): traces[row["mission"]].append(row)
    for row in jsonl(root / "smoke_computations.jsonl"): computations[row["mission"]].append(row)
    weights = {row["label"]: row for row in jsonl(root / "smoke_weights.jsonl")}
    phases = jsonl(root / "smoke_phases.jsonl")
    fits = {row["warm_index"]: row for row in phases if row["event"] == "demonstration_fit"}
    need(len(weights) == 21 and len(missions) == 36 and Counter(m["arm"] for m in missions) == {"warm": 20, "eval_before": 8, "eval_warm": 8}, "smoke inventory")
    need(set(fits) == {9, 19}, "unexpected direct-fitting boundaries")
    need(set(traces) == {m["id"] for m in missions}, "trace mission inventory")
    labels, greedy_cache, tokens = {}, {}, {"warm": 0, "eval_before": 0}
    stats = defaultdict(Counter)
    results = []
    decisions = neural_checks = callback_checks = terminal_checks = arrival_checks = 0
    total_fits = 0
    for mission in missions:
        identity, arm, index = mission["id"], mission["arm"], mission["index"]
        frozen = arm.startswith("eval_")
        start = weights["initial"] if arm == "eval_before" or (arm == "warm" and index == 0) else weights["warm_m19"] if arm == "eval_warm" else weights[f"warm_m{index - 1}"]
        if arm == "eval_warm" and arm not in tokens: tokens[arm] = tokens["warm"]
        rows = traces[identity]
        need(rows[0]["event"] == "mission_begin" and rows[-1]["result"] == mission, "mission trace boundaries")
        begin = rows[0]
        need(begin["hint_probability"] == int(not frozen) and begin["epsilon"] == 0 and not begin["random_credit"] and not begin["memory_policy"], "declared smoke choices")
        need(begin["terminal_updates"] == (0 if frozen else 512) and begin["learning"] == (not frozen), "declared learning mode")
        need(mission["updates_before"] == start["updates"] and mission["credits_before"] == start["credits"], "continuing checkpoint counters")
        requests, world, truth_world, revisions, seen, closed = {}, {}, {}, {}, set(), {}
        derived = {row["candidate"]: row for row in reversed(computations[identity]) if row["parent"]}
        public_proposals = {}
        for receipt in computations[identity]:
            for text in receipt.get("proposals", {}).get("texts", []):
                public_proposals.setdefault(text, receipt["token"])
        active, inspected = "", ""
        motors = count = hints = predictions = primitives = 0
        credit = []
        stopping = Counter()
        for event in rows[1:-1]:
            kind = event["event"]
            if kind == "public_request":
                need(event["clock"] == count and event["source"] == "verified_request" and not event["generated"], "external request provenance/clock")
                requests[event["id"]] = event
                wid = event["world_id"]
                facts = scene_facts(event["visible"])
                if wid in world: need(world[wid] == facts, "arrival world differs without callback")
                world[wid] = facts
                truth_world[wid] = facts.copy()
                revisions[wid] = event["visible"]["revision"]
                if wid == "stopping_world" and not event["visible"]["complete"]:
                    need(not frozen and index == 8, "unexpected partial stopping fixture")
                    words = event["text"].split()
                    truth_world[wid].add(("edge", (words[1], words[3])))
                    prefix = words[1][:-2]
                    truth_world[wid].add(("edge", (prefix + "d2", prefix + "d3")))
                continue
            if kind == "mission_credit": credit.append(event); continue
            need(kind == "decision" and not credit, "nondecision after terminal credit")
            count += 1;decisions += 1
            need(event["clock"] == count and event["token"] == tokens[arm] + 1, "owned chronological token")
            tokens[arm] = event["token"]
            o, menu, operation = event["observation"], event["menu"], event["operation"]
            need(bit_key(o, [])[:-1] == event["observation_bits"] and operation in menu, "exact public observation/menu")
            need(event["active_before"] == active and event["inspected_before"] == inspected, "goal lifecycle")
            if not o[29]:
                need(o[30] == int(bool(set(requests) - seen - set(closed))), "external arrival flag")
                arrival_checks += 1
            if frozen:
                need(not event["teacher"], "frozen evaluation hint")
                key = arm, event["observation_bits"], tuple(menu)
                if key not in greedy_cache: greedy_cache[key] = neural_choice(start, o, menu)
                need(operation == greedy_cache[key], "independent greedy neural choice")
                neural_checks += 1
            else:
                need(event["teacher"] == "continuing_curriculum_v1" and operation == public_demonstration(o, menu), "public-only bootstrap demonstration")
                key = event["observation_bits"], tuple(menu)
                need(labels.setdefault(key, operation) == operation, "conflicting demonstration labels")
                hints += 1
            need(event["cost"] == 0 and not event["callback_generated"], "cost/callback provenance")
            predictions += event["predictions"];primitives += event["primitive_steps"]
            if active and active not in requests:
                receipt = derived[active]
                need(receipt["token"] < event["token"] and public_proposals.get(receipt["candidate_text"], event["token"]) < event["token"], "derived target lacks earlier paid public proposal")
            goal = requests.get(active)
            if active and goal is None:
                goal = {"text": derived[active]["candidate_text"], "world_id": next(iter(world)), "effect": True}
            if goal and goal["effect"] and independent_truth(goal["text"], truth_world[goal["world_id"]]):
                stopping["effect_true_before_operation"] += 1
                if operation != 4: stopping["true_effect_nonfinish"] += 1
            if operation in (0, 2, 3):
                wid = goal["world_id"]
                if operation == 3:
                    need(len(event["actions"]) == len(event["observed_worlds"]), "owned action callback count")
                    allowed = [a for request in requests.values() for a in request["actions"]]
                    for action, observed in zip(event["actions"], event["observed_worlds"]):
                        need(action in allowed, "motor not publicly afforded")
                        truth_world[wid] = apply_actions(truth_world[wid], [action]);revisions[wid] += 1;motors += 1
                        need(scene_facts(observed) == truth_world[wid] and observed["revision"] == revisions[wid], "independent physical transition")
                        world[wid] = truth_world[wid].copy();callback_checks += 1
                    if event["actions"]: need(event["executed"] == [True] * len(event["actions"]), "unverified motors")
                else:
                    need(len(event["observed_worlds"]) == 1, "read callback count")
                    observed = event["observed_worlds"][0]
                    if wid == "stopping_world" and world[wid] != truth_world[wid]: revisions[wid] += 1
                    need(scene_facts(observed) == truth_world[wid] and observed["revision"] == revisions[wid], f"independent read disclosure {identity}/{count}: expected {truth_world[wid]}/{revisions[wid]}, got {scene_facts(observed)}/{observed['revision']}")
                    world[wid] = truth_world[wid].copy();callback_checks += 1
            if operation == 4:
                truth = independent_truth(goal["text"], truth_world[goal["world_id"]])
                correct = event["reply"] == int(truth) and (truth or not goal["effect"])
                need(event["terminal_satisfaction"] == truth and event["terminal_correct"] == correct and event["callback_source"] == "verified_environment", "independent actual finish/outcome")
                closed[active] = correct, count
                terminal_checks += 1
                if goal["effect"] and truth and correct: stopping["correct_effect_finishes"] += 1
            active, inspected = event["active_after"], event["inspected_after"]
            if operation == 5 and inspected: seen.add(inspected)
            if frozen: need(event["model_before"] == event["model_after"] == mission["model_before"], "frozen semantic mutation")
        success = bool(requests) and all(goal in closed and closed[goal][0] and closed[goal][1] <= request["deadline"] for goal, request in requests.items())
        need(success == mission["success"] and mission["updates_after"] - mission["updates_before"] == (0 if frozen else 512), "mission result/terminal updates")
        need(mission["credits_after"] - mission["credits_before"] == len(credit) == int(not frozen), "one chronological credit")
        if credit: need(credit[0]["actual_correct"] == success and credit[0]["reward"] == (1 if success else -1), "factual mission credit")
        for field, value in {"decisions": count, "motors": motors, "hints": hints, "predictions": predictions, "primitive_steps": primitives, "posted_requests": len(requests)}.items():
            need(mission[field] == value, "mission count: " + field)
        need(mission["cost"] == mission["discounted_cost"] == 0 and mission["observed_reward"] == (1 if success else -1), "reward/cost arithmetic")
        near(mission["observed_utility"], .99 ** (count - 1) * (1 if success else -1), "utility arithmetic")
        if not frozen:
            extra = 0
            if index in fits:
                fit = fits[index]
                need(fit["representatives"] == len(labels) and fit["rounds"] == 8 and fit["fits"] == len(labels) * 8, "trace-derived demonstration fits")
                extra = fit["fits"];total_fits += extra
                need(fit["cumulative_fits"] == total_fits <= 10000, "bounded demonstration fitting")
            need(weights[identity]["updates"] == mission["updates_after"] + extra and weights[identity]["credits"] == mission["credits_after"] and weights[identity]["model_version"] == mission["model_after"], "postfit continuing snapshot")
        stats[arm].update(missions=1, successes=int(success), motors=motors, decisions=count, hints=hints)
        results.append({"id": identity, "success": success, "motors": motors, "decisions": count, "root_finishes": {key: closed.get(key) for key in requests}, "stopping": dict(stopping)})
    need(summary["warm_successes"] == stats["warm"]["successes"] == 19 and summary["before_successes"] == stats["eval_before"]["successes"] == 0 and summary["warm_probe_successes"] == stats["eval_warm"]["successes"] == 6, "reported smoke results")
    need(summary["demonstration_fits"] == total_fits == 3840 and summary["updates"] == 20 * 512 + total_fits == 14080 and summary["credits"] == 20, "final update/credit totals")
    return {"passed": True, "scope": "Assisted mixed integration and a small before/after neural probe; no autonomous continuation or full capability closure",
        "source_files": len(manifest["files"]), "missions": len(missions), "decisions": decisions,
        "greedy_choices_independently_recomputed": neural_checks, "world_callbacks": callback_checks,
        "terminal_labels": terminal_checks, "arrival_flags": arrival_checks, "demonstration_fits": total_fits,
        "updates": summary["updates"], "arms": dict(stats), "results": results,
        "errata": ["Frozen reader_control_language_teaching rows export source verified_environment, while the actual concept and language receipts retain verified_teacher. These are independently authored raw lessons; the source string export is inaccurate and is preserved."],
        "limits": ["Initial reader atomic question program [17,9] is explicitly supplied substrate, not acquired here.",
            "Actual demo-fit gradients and empirical-value invariance are covered by frozen-source assertions and focused core tests; this audit recomputes label support, fit counts and resulting greedy choices, not every training gradient.",
            "Partial reader positive reports are evaluated from the declared raw-report environment; they are not promoted to closed-scene teaching. Current/historical negative and ambiguous conclusions remain Unknown.",
            "The full mixed core has separate helper tests; this smoke introduces raw repair/union upfront, while the later pilot introduces them after100 continuing missions."],
        "artifacts": {p.name: sha(p) for p in root.glob("smoke_*.json*")}, "auditor_sha256": sha(__file__)}


if __name__ == "__main__":
    root = Path("logs/goal_continuing_bootstrap_smoke_20260923_v1")
    output = Path("logs/goal_continuing_smoke_independent_audit_v1/audit.json")
    need(not output.exists(), "refusing to overwrite prior audit")
    result = audit(root)
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("artifacts", "results")}, indent=2))
