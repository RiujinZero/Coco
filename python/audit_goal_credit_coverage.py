"""Independent audit of 512-update credit coverage and separate readout consolidation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from audit_goal_preflights import (apply_actions, bit_key, f32, jsonl, near, need,
                                   read_json, satisfaction, scene_facts, sha, switch_reference)


def neural_choice(parameters, observation, menu):
    """Independent scalar float32 forward pass; does not invoke native policy."""
    weights, hidden, actions = parameters["weights"], parameters["hidden"], parameters["actions"]
    need(len(weights) == hidden * 32 + hidden + actions * hidden + actions, "network shape")
    activations = []
    for j in range(hidden):
        value = weights[hidden * 32 + j]
        for i in range(32):
            value = f32(value + f32(weights[j * 32 + i] * observation[i]))
        activations.append(f32(math.tanh(value)))
    output = hidden * 33
    scores = {}
    for action in menu:
        value = weights[output + actions * hidden + action]
        for j, activation in enumerate(activations):
            value = f32(value + f32(weights[output + action * hidden + j] * activation))
        scores[action] = value
    return max(menu, key=lambda action: scores[action])


def audit_endpoints(root, prefix, summary, weights, supports, stats):
    result = {}
    for arm, support in supports.items():
        before = weights[arm, arm + "_m299"]
        after = weights[arm, arm + "_consolidated"]
        representatives = len({bits for bits, menu in support})
        rounds = min(256, 100000 // representatives)
        fits = rounds * representatives
        need(before["updates"] == 115200 and before["credits"] == 300, "endpoint training counters")
        need(after["updates"] - before["updates"] == fits <= 100000, "bounded consolidation fit count")
        for field in ("credits", "model_version", "hidden", "actions"):
            need(before[field] == after[field], "consolidation changed " + field)
        need(summary[arm] == {"neural_before": stats["eval_" + arm + "_before"]["successes"],
            "neural_after": stats["eval_" + arm + "_after"]["successes"], "cases": 12,
            "representatives": representatives, "rounds": rounds, "additional_fits": fits,
            "values_and_semantics_unchanged": True}, "endpoint summary arithmetic")
        paths = [root / f"{prefix}_{arm}_{stage}_empirical.jsonl" for stage in ("before", "after")]
        need(paths[0].read_bytes() == paths[1].read_bytes(), "consolidation empirical values changed")
        empirical_rows = jsonl(paths[0])
        empirical = {row["observation_bits"]: row["values"] for row in empirical_rows}
        need(len(empirical) == len(empirical_rows), "duplicate empirical observations")
        for row in empirical_rows:
            need(bit_key(row["observation"], [])[:-1] == row["observation_bits"], "empirical float encoding")
        for stage, parameters in (("before", before), ("after", after)):
            rows = jsonl(root / f"{prefix}_{arm}_{stage}_readouts.jsonl")
            by_key = {(row["observation_bits"], tuple(row["menu"])): row for row in rows}
            need(len(by_key) == len(rows) and set(by_key) == set(support), "exact trace/readout support")
            for key, row in by_key.items():
                expected = support[key]
                for field in ("visits", "assisted"):
                    need(row[field] == expected[field], "trace-derived support " + field)
                need(set(row["assisted_operations"]) == expected["assisted_operations"], "trace-derived teacher support")
                need(bit_key(row["observation"], [])[:-1] == row["observation_bits"], "readout float encoding")
                need(row["arrival"] == bool(not row["observation"][29] and row["observation"][30]), "readout active-execution arrival field")
                need(row["neural_operation"] == neural_choice(parameters, row["observation"], row["menu"]), "independent endpoint neural readout")
                values = empirical.get(row["observation_bits"])
                need(row["memory_found"] == (values is not None) and row["empirical_values"] == values, "readout empirical linkage")
                memory_choice = max(row["menu"], key=lambda action: values[action]) if values is not None else row["neural_operation"]
                need(row["memory_operation"] == memory_choice, "independent empirical menu argmax")
        result[arm] = {"representatives": representatives, "support_pairs": len(support), "rounds": rounds,
            "fits": fits, "empirical_bytes_unchanged": True, "semantic_version_unchanged": True,
            "teacher_conflicting_support_pairs": sum(len(x["assisted_operations"]) > 1 for x in support.values())}
    return result


def audit_frozen_comparator(original_prefix, warm, summary):
    original_config = read_json(Path(str(original_prefix) + "_config.json"))
    need(original_config["limits"] == {"roots": 2, "derived": 0, "decisions": 256, "motors": 32}, "original task limits")
    rows = [row for row in jsonl(Path(str(original_prefix) + "_weights.jsonl")) if row["arm"] == "frozen_policy"]
    need(len(rows) == 201, "frozen comparator snapshot count")
    for row in rows:
        for field in ("weights", "updates", "hidden", "actions"):
            need(row[field] == warm[field], "frozen comparator warm parameters")
    missions = [row for row in jsonl(Path(str(original_prefix) + "_missions.jsonl")) if row["arm"] == "frozen_policy"]
    need(len(missions) == 200 and {row["index"] for row in missions} == set(range(100, 300)), "original comparator task indices")
    need(all(row["updates_before"] == row["updates_after"] == 12800 and row["hints"] == 0 and row["cost"] == 0 for row in missions), "frozen comparator updates/hints/cost")
    evaluations = [row for row in jsonl(Path(str(original_prefix) + "_evaluations.jsonl")) if row["checkpoint"] == "frozen_policy300"]
    need(len(evaluations) == 1 and evaluations[0]["cases"] == 12, "original comparator evaluation")
    measured = {"sampled_successes": sum(row["success"] for row in missions), "missions": len(missions),
        "neural_evaluation": evaluations[0]["successes"], "cases": 12, "verified_warm_weight_vectors": len(rows)}
    need(measured == summary["original_frozen_policy_comparator"], "reused frozen comparator result")
    return measured


def audit(root, prefix="continuation"):
    declaration = read_json(root / "predeclared_run.json")
    manifest = read_json(root / "source_manifest.json")
    for entry in declaration["files"] + manifest["files"]:
        need(sha(root / entry["path"]) == entry["sha256"], "predeclared/source hash: " + entry["path"])
    for entry in declaration["original_inputs"]:
        need(sha(Path(entry["path"])) == entry["sha256"], "predeclared original input hash: " + entry["path"])
    config, summary = read_json(root / (prefix + "_config.json")), read_json(root / (prefix + "_summary.json"))
    need(summary["complete"] and config["limits"] == {"roots": 2, "derived": 0, "decisions": 256, "motors": 32}, "pilot declaration")
    missions = jsonl(root / (prefix + "_missions.jsonl"))
    parameters = jsonl(root / (prefix + "_weights.jsonl"))
    evaluations = jsonl(root / (prefix + "_evaluations.jsonl"))
    events = defaultdict(list)
    for row in jsonl(root / (prefix + "_trace.jsonl")):
        events[row["mission"]].append(row)
    weights = {(row["arm"], row["label"]): row for row in parameters}
    need(len(weights) == len(parameters) == 505, "weight snapshot inventory")
    initial, warm = weights["reconstruction", "initial"], weights["shared", "warm100_verified"]
    original_prefix = Path(config["reference_prefix"])
    if not Path(str(original_prefix) + "_config.json").exists():
        original_prefix = root / original_prefix
    original_missions = {row["id"]: row for row in jsonl(Path(str(original_prefix) + "_missions.jsonl"))}
    original_weights = jsonl(Path(str(original_prefix) + "_weights.jsonl"))
    original_warm = next(row for row in original_weights if row["label"] == "warm100")
    for key in ("weights", "updates", "credits", "model_version", "hidden", "actions"):
        need(warm[key] == original_warm[key], "exact original warm reconstruction: " + key)
        need(weights["random_credit512", "branch_start"][key] == warm[key], "random control warm clone mismatch")
    original_by_key = {(row["arm"], row["label"]): row for row in original_weights}
    for index in range(100):
        need(weights["warm", f"warm_m{index}"] == original_by_key["warm", f"warm_m{index}"], "warm intermediate weights differ")
    checkpoints = ["warm100", "adaptive512_before", "adaptive512_after", "random_credit512_before", "random_credit512_after"]
    expected_counts = {"warm": 100, "adaptive512": 200, "random_credit512": 200,
                       **{"eval_" + checkpoint: 12 for checkpoint in checkpoints}}
    need(Counter(row["arm"] for row in missions) == expected_counts and len(missions) == 560, "mission inventory")
    need(len(events) == len(missions) and set(events) == {row["id"] for row in missions}, "trace mission inventory")
    need(len(evaluations) == len(checkpoints) and {row["checkpoint"] for row in evaluations} == set(checkpoints), "evaluation checkpoint inventory")
    for mission in missions:
        if mission["arm"] in {"warm", "eval_warm100"}:
            need(mission == original_missions[mission["id"]], "original warm mission/evaluation result differs")
    supports = {arm: defaultdict(lambda: {"visits": 0, "assisted": 0, "assisted_operations": set()})
                for arm in ("adaptive512", "random_credit512")}
    stopping = {}
    last_state = {}
    last_token = {}
    stats = defaultdict(lambda: {"missions": 0, "successes": 0, "hints": 0, "decisions": 0, "motors": 0,
                                 "predictions": 0, "operations": Counter(), "credited_reward": Counter(),
                                 "physical_correct": 0, "urgent_correct": 0, "urgent_on_time": 0})
    policy_cache, task_schedules = {}, {}
    decision_checks = neural_checks = arrival_checks = callback_checks = terminal_checks = 0
    zero_cost_checks = 0
    for mission in missions:
        identity, arm, index = mission["id"], mission["arm"], mission["index"]
        rows = events[identity]
        begin = rows[0]
        need(begin["event"] == "mission_begin" and rows[-1] == {"mission": identity, "event": "mission_end", "result": mission}, "mission event boundaries")
        learning = not arm.startswith("eval_")
        need(mission["learning"] == begin["learning"] == learning, "mission learning mode")
        if arm == "warm":
            start = initial if index == 0 else weights[arm, f"warm_m{index - 1}"]
        elif arm == "eval_warm100":
            start = warm
        elif arm.startswith("eval_"):
            target_arm, stage = arm[5:].rsplit("_", 1)
            start = weights[target_arm, target_arm + ("_m299" if stage == "before" else "_consolidated")]
        else:
            start = warm if index == 100 else weights[arm, f"{arm}_m{index - 1}"]
        if arm not in last_state:
            last_state[arm] = {"updates": start["updates"], "credits": start["credits"], "model": start["model_version"]}
            last_token[arm] = 0 if arm == "warm" else last_token["warm"] if arm in {"adaptive512", "random_credit512", "eval_warm100"} else last_token[arm[5:].rsplit("_", 1)[0]]
        for key in ("updates", "credits", "model"):
            need(mission[key + "_before"] == last_state[arm][key], "continuing learner state: " + key)
        expected_probability = (1 if index < 50 else (99 - index) / 50) if arm == "warm" else 0
        near(begin["hint_probability"], expected_probability, "declared hint taper")
        need(begin["epsilon"] == (f32(.1) if learning and arm != "warm" else 0), "declared epsilon")
        need(begin["assistance_seed"] == 50000 + index, "assistance seed")
        expected_updates = (128 if arm == "warm" else 512) if learning else 0
        need(begin["terminal_updates"] == expected_updates and not begin["memory_policy"], "terminal update/readout request")
        need(begin["random_credit"] == mission["random_credit"] == (arm == "random_credit512"), "random credit arm")
        arrival_tick = config["training_arrivals"][index % 3] if learning else config["evaluation_arrivals"][index % 3]
        expected_order = 1000 + index if learning else config["evaluation_affordance_order_seeds"][index // 3]
        need(begin["task_order_seed"] == expected_order, "matched task ordering seed")
        requests, worlds, closed, seen = {}, {}, {}, set()
        active = inspected = ""
        agenda = True
        interrupted = set()
        did_interrupt = did_resume = arrival_execution = False
        decisions, motors, predictions, primitive, hints = 0, 0, 0, 0, 0
        credits = []
        model = mission["model_before"]
        stopping[identity] = stop = {"arm": arm, "mission_success": mission["success"], "physical_satisfied_at": None,
            "physical_finished_at": None, "physical_correct_finish": False, "urgent_correct_ontime": False,
            "known_satisfied_nonfinish_operations": Counter(), "physical_operations_after_satisfaction": Counter()}
        for event in rows[1:-1]:
            kind = event["event"]
            if kind == "public_request":
                goal = event["id"]
                need(goal not in requests and event["clock"] == decisions, "request sequence")
                need(event["clock"] == (0 if goal == "physical" else arrival_tick), "declared arrival time")
                need(event["deadline"] == (256 if goal == "physical" else arrival_tick + 12), "declared root deadline")
                need(event["generated"] is False and event["source"] == "verified_request", "request provenance")
                if goal == "arrival":
                    arrival_execution = not agenda
                world = event["world_id"]
                state = (scene_facts(event["visible"]), event["visible"]["revision"])
                if world in worlds:
                    need(worlds[world] == state, "arrival callback world")
                else:
                    worlds[world] = state
                requests[goal] = event
                if learning:
                    schedule_key = (index, goal)
                    public_task = {key: value for key, value in event.items() if key not in {"mission", "agenda_order_key_audit_only", "visible"}}
                    if schedule_key in task_schedules:
                        need(task_schedules[schedule_key] == public_task, "control task schedules differ")
                    else:
                        task_schedules[schedule_key] = public_task
                continue
            if kind == "mission_credit":
                credits.append(event)
                need(event["source"] == "verified_environment" and event["random_credit"] == (arm == "random_credit512"), "mission credit source")
                continue
            need(kind == "decision", "unexpected pilot event")
            need(not credits, "decision follows terminal mission credit")
            decisions += 1
            decision_checks += 1
            need(event["clock"] == decisions and event["token"] == last_token[arm] + 1, "owned chronological token")
            last_token[arm] = event["token"]
            o, menu, op = event["observation"], event["menu"], event["operation"]
            need(bit_key(o, [])[:-1] == event["observation_bits"], "exact float observation encoding")
            expected_menu = ([1, 5, 6] if inspected else [5]) if agenda else ([0, 1, 2, 4, 5] if motors >= 32 else [0, 1, 2, 3, 4, 5])
            need(menu == expected_menu and op in menu and bool(o[29]) == agenda, "public legal menu")
            need(event["active_before"] == active and event["inspected_before"] == inspected, "pre-decision goal lifecycle")
            if not agenda:
                need(o[30] == int(bool(set(requests) - seen - set(closed))), "event-derived arrival flag")
                arrival_checks += 1
            if event["teacher"]:
                need(arm == "warm" and expected_probability > 0 and event["teacher"] == "protocol_bootstrap", "assistance outside declared taper")
                need(op == switch_reference(o, menu), "hint not derived from public reference")
                hints += 1
            else:
                need(expected_probability != 1, "missing fully guided hint")
                if begin["epsilon"] == 0:
                    key = (start["arm"], start["label"], event["observation_bits"], tuple(menu))
                    if key not in policy_cache:
                        policy_cache[key] = neural_choice(start, o, menu)
                    need(op == policy_cache[key], "unassisted greedy action differs from independently evaluated checkpoint")
                    neural_checks += 1
            need(event["cost"] == 0 and event["callback_generated"] is False, "declared zero cost/genuine callback")
            zero_cost_checks += 1
            need(event["model_before"] == model, "semantic version chronology")
            model = event["model_after"]
            predictions += event["predictions"]
            primitive += event["primitive_steps"]
            if learning:
                for support_arm in (supports if arm == "warm" else [arm]):
                    item = supports[support_arm][event["observation_bits"], tuple(menu)]
                    item["visits"] += 1
                    if event["teacher"]:
                        item["assisted"] += 1
                        item["assisted_operations"].add(op)
            if active == "physical":
                if stop["physical_satisfied_at"] is not None:
                    stop["physical_operations_after_satisfaction"][str(op)] += 1
                if o[6] and not o[29] and op != 4:
                    stop["known_satisfied_nonfinish_operations"][str(op)] += 1
            previous_active = active
            if op == 5:
                inspected = event["inspected_after"]
                need(inspected in requests and inspected not in closed, "next-goal selection")
                seen.add(inspected)
                agenda = True
            elif op == 6:
                active, agenda = inspected, False
                need(event["callback_source"] == "owned_activation_token", "activation ownership")
                if previous_active and previous_active != active:
                    interrupted.add(previous_active)
                    did_interrupt = True
                did_resume |= active in interrupted
            elif op in (0, 2, 3):
                world = requests[active]["world_id"]
                facts, revision = worlds[world]
                if op == 3:
                    need(len(event["observed_worlds"]) == len(event["actions"]), "motor callback count")
                    for action, observed in zip(event["actions"], event["observed_worlds"]):
                        need(action in requests["physical"]["actions"], "nonpublic motor action")
                        facts = apply_actions(facts, [action])
                        revision += 1
                        motors += 1
                        need(scene_facts(observed) == facts and observed["revision"] == revision, "independent per-motor world transition")
                        callback_checks += 1
                    if event["actions"]:
                        need(event["executed"] == [True] * len(event["actions"]) and event["callback_source"] == "verified_environment", "motor verification")
                else:
                    need(len(event["observed_worlds"]) == 1 and scene_facts(event["observed_worlds"][0]) == facts and event["observed_worlds"][0]["revision"] == revision, "independent read callback")
                    need(event["callback_source"] == ("trusted_memory" if op == 0 else "verified_environment"), "read provenance")
                    callback_checks += 1
                worlds[world] = (facts, revision)
                if op == 3 and event["actions"] and stop["physical_satisfied_at"] is None:
                    if satisfaction(requests["physical"]["text"], worlds[requests["physical"]["world_id"]][0]):
                        stop["physical_satisfied_at"] = decisions
            elif op == 4:
                request = requests[active]
                truth = satisfaction(request["text"], worlds[request["world_id"]][0])
                correct = event["reply"] == int(truth) and (truth or not request["effect"])
                need(event["terminal_correct"] == correct and event["terminal_satisfaction"] == truth, "independent terminal truth")
                need(event["callback_source"] == "verified_environment", "terminal provenance")
                closed[active] = {"correct": correct, "clock": decisions, "deadline": request["deadline"]}
                terminal_checks += 1
                if active == "physical":
                    stop["physical_finished_at"] = decisions
                    stop["physical_correct_finish"] = correct and truth
                elif active == "arrival":
                    stop["urgent_correct_ontime"] = correct and decisions <= request["deadline"]
                active = inspected = ""
                agenda = True
            need(event["active_after"] == active and event["inspected_after"] == inspected, "post-decision goal lifecycle")
            if not learning:
                need(model == mission["model_before"], "frozen evaluation semantic mutation")
            stats[arm]["operations"][op] += 1
        success = set(closed) == {"physical", "arrival"} and all(row["correct"] and row["clock"] <= row["deadline"] for row in closed.values())
        need(success == mission["success"] and decisions <= 256 and motors <= 32, "mission reward/budget")
        for field, value in {"decisions": decisions, "motors": motors, "predictions": predictions, "primitive_steps": primitive,
                             "hints": hints, "posted_requests": len(requests), "interrupted": did_interrupt, "resumed": did_resume,
                             "arrival_during_execution": arrival_execution}.items():
            need(mission[field] == value, "mission count: " + field)
        need(mission["observed_reward"] == (1 if success else -1) and mission["cost"] == mission["discounted_cost"] == 0, "mission arithmetic")
        near(mission["observed_utility"], (.99 ** (decisions - 1)) * (1 if success else -1), "mission observed utility")
        need(mission["updates_after"] - mission["updates_before"] == expected_updates, "terminal-only update count")
        need(mission["credits_after"] - mission["credits_before"] == len(credits) == int(learning), "one chronological mission credit")
        if learning:
            need(credits[0]["actual_correct"] == success and credits[0]["reward"] == mission["credited_reward"], "credited outcome")
            need(mission["credited_reward"] in (-1, 1), "invalid credit signal")
            if arm != "random_credit512":
                need(mission["credited_reward"] == mission["observed_reward"], "factual credit differs from true outcome")
            after = weights[arm, identity]
            need(after["updates"] == mission["updates_after"] and after["credits"] == mission["credits_after"] and after["model_version"] == mission["model_after"], "terminal weight snapshot state")
        else:
            need(mission["credited_reward"] is None and mission["model_after"] == mission["model_before"], "frozen evaluation learned")
        for key in ("updates", "credits", "model"):
            last_state[arm][key] = mission[key + "_after"]
        for key, value in {"missions": 1, "successes": int(success), "hints": hints, "decisions": decisions, "motors": motors, "predictions": predictions}.items():
            stats[arm][key] += value
        stats[arm]["credited_reward"][str(mission["credited_reward"])] += 1
        stats[arm]["physical_correct"] += int(closed.get("physical", {}).get("correct", False))
        stats[arm]["urgent_correct"] += int(closed.get("arrival", {}).get("correct", False))
        stats[arm]["urgent_on_time"] += int("arrival" in closed and closed["arrival"]["correct"] and closed["arrival"]["clock"] <= closed["arrival"]["deadline"])
        if stop["physical_satisfied_at"] is not None and stop["physical_finished_at"] is not None:
            stop["decisions_until_physical_finish"] = stop["physical_finished_at"] - stop["physical_satisfied_at"]
    for evaluation in evaluations:
        arm = "eval_" + evaluation["checkpoint"]
        need(evaluation["successes"] == stats[arm]["successes"], "checkpoint result")
        need(evaluation["cases"] == stats[arm]["missions"] == 12 and evaluation["hints"] == evaluation["epsilon"] == 0 and evaluation["frozen"], "checkpoint conditions")
        need(not evaluation["memory_policy"], "evaluation used empirical readout")
        state = last_state[arm]
        need((evaluation["updates"], evaluation["credits"], evaluation["model_version"]) == (state["updates"], state["credits"], state["model"]), "frozen evaluation counters")
    need(set(stats["random_credit512"]["credited_reward"]) == {"1", "-1"}, "random-credit distribution degenerate")
    endpoints = audit_endpoints(root, prefix, summary, weights, supports, stats)
    comparator = audit_frozen_comparator(original_prefix, warm, summary)
    published_stopping = read_json(root / "stopping_summary.json")
    need(stopping == published_stopping["missions"], "independent route-based stopping reconstruction differs")
    stopping_aggregate = defaultdict(Counter)
    for row in stopping.values():
        total = stopping_aggregate[row["arm"]]
        total["missions"] += 1
        total["mission_successes"] += row["mission_success"]
        total["physical_world_completed"] += row["physical_satisfied_at"] is not None
        total["physical_correct_finishes"] += row["physical_correct_finish"]
        total["urgent_correct_ontime"] += row["urgent_correct_ontime"]
        total["completed_world_without_finish"] += row["physical_satisfied_at"] is not None and row["physical_finished_at"] is None
        total["known_satisfied_nonfinish_decisions"] += sum(row["known_satisfied_nonfinish_operations"].values())
    need(stopping_aggregate == published_stopping["arms"], "stopping aggregate counts differ")
    teacher_labels = defaultdict(set)
    teacher_rows = 0
    for mission in missions:
        if mission["arm"] == "warm":
            for event in events[mission["id"]]:
                if event["event"] == "decision" and event["teacher"]:
                    teacher_labels[event["observation_bits"], tuple(event["menu"])].add(event["operation"])
                    teacher_rows += 1
    return {"schema": "independent_goal_credit_coverage_audit_v1", "passed": True, "policy_acquisition_established": False,
            "root": str(root), "source_files_verified": len(manifest["files"]), "predeclared_files_verified": len(declaration["files"]),
            "predeclared_original_inputs_verified": len(declaration["original_inputs"]),
            "missions": len(missions), "decision_events_checked": decision_checks, "zero_cost_charges_checked": zero_cost_checks,
            "greedy_actions_recomputed_from_exported_weights": neural_checks, "distinct_greedy_forward_inputs": len(policy_cache),
            "arrival_flags_recomputed": arrival_checks, "independent_world_callbacks": callback_checks,
            "independent_terminal_labels": terminal_checks, "identical_warm_control_starts": True,
            "frozen_comparator": comparator, "consolidation": endpoints, "stopping": dict(stopping_aggregate),
            "warm_demonstrations": {"rows": teacher_rows, "unique_observation_menu_pairs": len(teacher_labels),
                "conflicting_pairs": sum(len(labels) > 1 for labels in teacher_labels.values()),
                "unique_pairs_by_operation": dict(Counter(next(iter(labels)) for labels in teacher_labels.values() if len(labels) == 1))},
            "arms": dict(stats),
            "artifacts": {str(path): sha(path) for path in sorted(root.glob(prefix + "_*.json*"))},
            "limits": ["Single seed and one six-edge public-root task family; no derived-goal or long-task acquisition claim.",
                       "The original 12 evaluation cases are reused for development, not untouched confirmation.",
                       "Consolidation fits the same empirical targets on separate endpoint clones; before/after empirical exports are byte-identical and semantic versions unchanged.",
                       "The same continuing controller lineage and terminal-only weight changes are supported by full mission-boundary snapshots plus reviewed native intra-mission assertions; no per-decision weight array is exported.",
                       "All unassisted zero-epsilon choices are independently reproduced from the exported prior checkpoint weights; this does not independently reproduce training gradients or epsilon RNG draws.",
                       "Random-credit labels use the controller RNG, so later epsilon draws can differ from the adaptive arm; this is declared, not a perfectly matched exploration realization.",
                       "Checkpoint clones use one continuing RNG across their 12 fixed cases, not independently reseeded case trials."],
            "auditor_sha256": sha(__file__),
            "helper_sha256": sha(Path(__file__).with_name("audit_goal_preflights.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/goal_credit_coverage_continuation_20260923_v1"))
    parser.add_argument("--output", type=Path, default=Path("logs/goal_credit_coverage_independent_audit_v1/audit.json"))
    args = parser.parse_args()
    need(not args.output.exists(), "refusing to overwrite previous independent audit")
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "missions": result["missions"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
