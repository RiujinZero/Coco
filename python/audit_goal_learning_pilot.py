"""Independent bounded audit of frozen seven-action learning-pilot evidence."""
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


def audit(root):
    declaration = read_json(root / "predeclared_run.json")
    manifest = read_json(root / "source_manifest.json")
    for entry in declaration["files"] + manifest["files"]:
        need(sha(root / entry["path"]) == entry["sha256"], "predeclared/source hash: " + entry["path"])
    config, summary = read_json(root / "pilot_config.json"), read_json(root / "pilot_summary.json")
    need(summary["complete"] and config["limits"] == {"roots": 2, "derived": 0, "decisions": 256, "motors": 32}, "pilot declaration")
    missions = jsonl(root / "pilot_missions.jsonl")
    parameters = jsonl(root / "pilot_weights.jsonl")
    evaluations = jsonl(root / "pilot_evaluations.jsonl")
    events = defaultdict(list)
    for row in jsonl(root / "pilot_trace.jsonl"):
        events[row["mission"]].append(row)
    weights = {(row["arm"], row["label"]): row for row in parameters}
    need(len(weights) == len(parameters) == 704, "weight snapshot inventory")
    initial, warm = weights["shared", "initial"], weights["shared", "warm100"]
    for arm in ("frozen_policy", "random_credit"):
        branch = weights[arm, "branch_start"]
        need(all(branch[key] == warm[key] for key in ("weights", "updates", "credits", "model_version", "hidden", "actions")), "warm control clone mismatch")
    expected_counts = {"warm": 100, "adaptive": 200, "frozen_policy": 200, "random_credit": 200,
                       **{"eval_" + checkpoint: 12 for checkpoint in config["checkpoints"]}}
    need(Counter(row["arm"] for row in missions) == expected_counts and len(missions) == declaration["mission_budget"] == 760, "mission inventory")
    need(len(events) == len(missions) and set(events) == {row["id"] for row in missions}, "trace mission inventory")
    need(len(evaluations) == len(config["checkpoints"]) and {row["checkpoint"] for row in evaluations} == set(config["checkpoints"]), "evaluation checkpoint inventory")
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
        elif arm.startswith("eval_"):
            checkpoint = arm[5:]
            start = initial if checkpoint == "before" else warm if checkpoint == "warm100" else weights[checkpoint[:-3], checkpoint[:-3] + "_m299"]
        else:
            start = warm if index == 100 else weights[arm, f"{arm}_m{index - 1}"]
        if arm not in last_state:
            last_state[arm] = {"updates": start["updates"], "credits": start["credits"], "model": start["model_version"]}
            last_token[arm] = 0 if arm in {"warm", "eval_before"} else last_token["warm"] if arm in {"adaptive", "frozen_policy", "random_credit", "eval_warm100"} else last_token[arm[5:-3]]
        for key in ("updates", "credits", "model"):
            need(mission[key + "_before"] == last_state[arm][key], "continuing learner state: " + key)
        expected_probability = (1 if index < 50 else (99 - index) / 50) if arm == "warm" else 0
        near(begin["hint_probability"], expected_probability, "declared hint taper")
        need(begin["epsilon"] == (f32(.1) if learning and arm != "warm" else 0), "declared epsilon")
        need(begin["assistance_seed"] == 50000 + index, "assistance seed")
        expected_updates = 128 if learning and arm != "frozen_policy" else 0
        need(begin["terminal_updates"] == expected_updates, "terminal update request")
        need(begin["random_credit"] == mission["random_credit"] == (arm == "random_credit"), "random credit arm")
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
                need(event["source"] == "verified_environment" and event["random_credit"] == (arm == "random_credit"), "mission credit source")
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
            elif op == 4:
                request = requests[active]
                truth = satisfaction(request["text"], worlds[request["world_id"]][0])
                correct = event["reply"] == int(truth) and (truth or not request["effect"])
                need(event["terminal_correct"] == correct and event["terminal_satisfaction"] == truth, "independent terminal truth")
                need(event["callback_source"] == "verified_environment", "terminal provenance")
                closed[active] = {"correct": correct, "clock": decisions, "deadline": request["deadline"]}
                terminal_checks += 1
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
            if arm != "random_credit":
                need(mission["credited_reward"] == mission["observed_reward"], "factual credit differs from true outcome")
            after = weights[arm, identity]
            need(after["updates"] == mission["updates_after"] and after["credits"] == mission["credits_after"] and after["model_version"] == mission["model_after"], "terminal weight snapshot state")
            if arm == "frozen_policy":
                need(after["weights"] == warm["weights"], "frozen control changed warm weights")
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
    for evaluation in evaluations:
        arm = "eval_" + evaluation["checkpoint"]
        need(evaluation["successes"] == stats[arm]["successes"] == summary[evaluation["checkpoint"]], "checkpoint result")
        need(evaluation["cases"] == stats[arm]["missions"] == 12 and evaluation["hints"] == evaluation["epsilon"] == 0 and evaluation["frozen"], "checkpoint conditions")
    need(set(stats["random_credit"]["credited_reward"]) == {"1", "-1"}, "random-credit distribution degenerate")
    return {"schema": "independent_goal_learning_pilot_audit_v1", "passed": True, "policy_acquisition_established": False,
            "root": str(root), "source_files_verified": len(manifest["files"]), "predeclared_files_verified": len(declaration["files"]),
            "missions": len(missions), "decision_events_checked": decision_checks, "zero_cost_charges_checked": zero_cost_checks,
            "greedy_actions_recomputed_from_exported_weights": neural_checks, "distinct_greedy_forward_inputs": len(policy_cache),
            "arrival_flags_recomputed": arrival_checks, "independent_world_callbacks": callback_checks,
            "independent_terminal_labels": terminal_checks, "identical_warm_control_starts": True,
            "frozen_control_all_200_weight_snapshots_match_warm": True, "arms": dict(stats),
            "artifacts": {str(path): sha(path) for path in sorted(root.glob("pilot_*.json*"))},
            "limits": ["Single seed and one six-edge public-root task family; no derived-goal or long-task acquisition claim.",
                       "The same continuing controller lineage and terminal-only weight changes are supported by full mission-boundary snapshots plus reviewed native intra-mission assertions; no per-decision weight array is exported.",
                       "All unassisted zero-epsilon choices are independently reproduced from the exported prior checkpoint weights; this does not independently reproduce training gradients or epsilon RNG draws.",
                       "Random-credit labels use the controller RNG, so later epsilon draws can differ from the adaptive arm; this is declared, not a perfectly matched exploration realization.",
                       "Checkpoint clones use one continuing RNG across their12 fixed cases, not independently reseeded case trials."],
            "auditor_sha256": sha(__file__),
            "helper_sha256": sha(Path(__file__).with_name("audit_goal_preflights.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/goal_control_learning_pilot_20260923_v1"))
    parser.add_argument("--output", type=Path, default=Path("logs/goal_learning_independent_audit_v1/audit.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "missions": result["missions"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
