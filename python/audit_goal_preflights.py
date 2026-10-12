"""Independent arithmetic/event audit of bounded goal-control preflights.

This does not establish feature sufficiency outside the exported visited traces,
nor learned policy acquisition. It consumes frozen artifacts, never trains.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def near(left, right, label, tolerance=2e-11):
    need(math.isfinite(left) and math.isfinite(right), f"{label}: nonfinite")
    need(abs(left - right) <= tolerance * max(1, abs(left), abs(right)),
         f"{label}: {left!r} != {right!r}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def f32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def bit_key(observation, menu):
    need(len(observation) == 32, "observation width")
    return "".join(f"{struct.unpack('!I', struct.pack('!f', x))[0]:08x}" for x in observation) + ":" + "".join(map(str, menu))


def scene_facts(scene):
    result = set()
    for atom in scene["facts"]:
        predicate, args = (atom["predicate"], atom["args"]) if isinstance(atom, dict) else atom
        result.add((predicate, tuple(args)))
    return result


def action_parts(action):
    return (action["schema"], action["args"]) if isinstance(action, dict) else action


def apply_actions(facts, actions):
    result = set(facts)
    for action in actions:
        schema, args = action_parts(action)
        need(schema == "edge_motor" and len(args) == 2, "unexpected evaluator motor")
        result.add(("edge", tuple(args)))
    return result


def route(facts, start, end, length):
    frontier = {start}
    for _ in range(length):
        frontier = {args[1] for predicate, args in facts if predicate == "edge" and len(args) == 2 and args[0] in frontier}
    return end in frontier


def satisfaction(text, facts):
    words = text.split()
    need(len(words) == 4 and words[0] in {"span", "connect", "route", "inspect"}, "unknown declared fixture query")
    return route(facts, words[1], words[3], {"span": 6, "connect": 9, "route": 3, "inspect": 1}[words[0]])


def route_definition_length(definition):
    edges = [(int(a), int(b)) for a, b in re.findall(r"edge\((\d+),(\d+),\)", definition)]
    need(definition.count("[") == 1 and definition.count("(") == len(edges) and edges,
         "preview definition is outside independently audited positive-route family")
    successors = dict(edges)
    need(len(successors) == len(edges) and len({b for _, b in edges}) == len(edges), "preview route definition branches")
    current, visited = 0, set()
    for _ in edges:
        need(current in successors and current not in visited, "preview route definition disconnected/cyclic")
        visited.add(current)
        current = successors[current]
    need(current == 1 and 1 not in successors, "preview definition does not connect its head arguments")
    return len(edges)


def minimum_missing(potential, facts, args, length):
    # Independent dynamic program on this fixture's acyclic public potential
    # graph. Acyclicity ensures no repeated atom can be counted twice on a walk.
    edges = {values for predicate, values in potential if predicate == "edge"}
    adjacency = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
    visiting, done = set(), set()
    def visit(node):
        need(node not in visiting, "cyclic potential graph requires a different independent minima audit")
        if node in done:
            return
        visiting.add(node)
        for target in adjacency[node]:
            visit(target)
        visiting.remove(node)
        done.add(node)
    for node in list(adjacency):
        visit(node)
    distances = {args[0]: 0}
    for _ in range(length):
        following = {}
        for (source, target) in edges:
            if source in distances:
                cost = distances[source] + int(("edge", (source, target)) not in facts)
                following[target] = min(cost, following.get(target, math.inf))
        distances = following
    need(args[1] in distances, "no compatible complete parent grounding in public potential domain")
    return distances[args[1]]


def switch_reference(observation, menu):
    # Independently executed rule over the exported public inputs. The source
    # review separately establishes that the native function has no other input.
    o = observation
    if o[29]:
        operation = 6 if 6 in menu and (not o[4] or not o[12] or not o[14]) else 5
    elif o[30]:
        operation = 5
    elif not o[5]:
        operation = 1
    elif not o[4] or o[6]:
        operation = 4
    elif o[9]:
        operation = 3
    elif o[10]:
        operation = 1
    elif o[8]:
        operation = 2
    else:
        operation = 4
    need(operation in menu, "switch reference action outside public menu")
    return operation


def long_reference(o, menu, direct, reference_variant=0):
    if o[29]:
        if not o[0] or (o[12] and o[17] >= o[16]):
            operation = 5
        elif not o[4] or direct:
            operation = 6
        elif reference_variant == 1 and o[13] and o[11] and o[6]:
            operation = 6
        elif reference_variant == 1 and o[13] and o[10]:
            operation = 1
        elif not o[5]:
            operation = 1
        elif not o[13] and (o[6] or o[9]):
            operation = 6
        elif o[13] and o[18] and o[19] > 0:
            operation = 6
        else:
            operation = 5
    elif o[30]:
        operation = 5
    elif not o[5]:
        operation = 1
    elif not o[4] or o[6] or o[20]:
        operation = 4
    elif o[9] or direct == 1:
        operation = 3
    elif o[10] and o[19] < (f32(12 / 15) if direct == 2 else 1):
        operation = 1
    else:
        operation = 3 if direct else 5
    need(operation in menu, "long reference action outside public menu")
    return operation


def arithmetic(trace, success):
    need(trace, "empty trace")
    costs = [row["cost"] for row in trace]
    need(all(math.isfinite(c) and c >= 0 for c in costs), "invalid cost")
    cost = sum(costs)
    discounted = sum((.99 ** i) * cost for i, cost in enumerate(costs))
    reward = 1 if success else -1
    return {"reward": reward, "cost": cost, "discounted_cost": discounted,
            "utility": (.99 ** (len(trace) - 1)) * reward - discounted}


def check_manifest(root):
    manifest_path = root / "source_manifest.json"
    manifest = read_json(manifest_path)
    for row in manifest["files"]:
        need(sha(root / row["path"]) == row["sha256"], f"source hash mismatch: {row['path']}")
    return {"path": str(manifest_path), "sha256": sha(manifest_path), "verified_files": len(manifest["files"])}


def audit_switch(root):
    summary_path, trace_path, aliases_path = [root / ("measurement_" + suffix) for suffix in ("summary.json", "trace.jsonl", "aliases.jsonl")]
    summary = read_json(summary_path)
    need(summary["schema"] == "goal_switch_preflight_v2", "wrong authoritative switch schema")
    need(not summary["training_run"] and not summary["policy_learning"] and not summary["warm_control_policy"], "unsupported switch learning claim")
    events = defaultdict(list)
    for event in jsonl(trace_path):
        events[event["run"]].append(event)
    aliases = defaultdict(lambda: {"rows": 0, "operations": set(), "contexts": set(), "runs": set()})
    audited = []
    arrival_checks = cost_checks = observation_checks = 0
    for run in summary["runs"]:
        name = run["id"]
        requests, worlds, seen, closed, trace = {}, {}, set(), {}, []
        active = inspected = ""
        agenda, motors = True, 0
        for event in events[name]:
            if event["event"] == "public_request":
                need(event["clock"] == len(trace), f"{name}: request event clock")
                need(event["id"] not in requests, f"{name}: repeated request")
                need(event["clock"] == (0 if event["id"] == "physical" else run["arrival"]), f"{name}: arrival schedule")
                requests[event["id"]] = event
                world_id = event["world_id"]
                if world_id in worlds:
                    need(scene_facts(event["visible"]) == worlds[world_id][0] and event["visible"]["revision"] == worlds[world_id][1], f"{name}: request supplied unobserved world")
                else:
                    worlds[world_id] = (scene_facts(event["visible"]), event["visible"]["revision"])
                continue
            need(event["event"] == "decision", f"{name}: unknown event")
            trace.append(event)
            o, menu, op = event["observation"], event["menu"], event["operation"]
            need(event["clock"] == len(trace) and event["token"] == len(trace), f"{name}: chronological ownership")
            need(event["active_before"] == active and event["inspected_before"] == inspected, f"{name}: lifecycle mismatch")
            need(bool(o[29]) == agenda, f"{name}: phase observation")
            expected_menu = ([1, 5, 6] if inspected else [5]) if agenda else ([0, 1, 2, 4, 5] if motors >= 32 else [0, 1, 2, 3, 4, 5])
            need(menu == expected_menu and op in menu, f"{name}: public menu")
            key = bit_key(o, menu)
            need(event["observation_bits"] == key, f"{name}: rounded feature key")
            need(event["teacher"] == "", f"{name}: hidden teacher receipt")
            if not agenda:
                expected_arrival = bool(set(requests) - seen - set(closed))
                need(o[30] == int(expected_arrival), f"{name}: arrival flag at {len(trace)}")
                arrival_checks += 1
            observation_checks += 1
            if run["strategy"] in {"reference", "reference_semantic"}:
                need(op == switch_reference(o, menu), f"{name}: non-public reference choice")
                group = aliases[key]
                group["rows"] += 1
                group["operations"].add(op)
                group["contexts"].add(active + "/" + inspected)
                group["runs"].add(name)
            cost = f32([.01, .005, .06, .01, 0, .002, .01][op])
            if op == 1:
                if o[2]:
                    cost = f32(cost + f32(.005))
                cost = f32(cost + f32(f32(.00001) * f32(event["predictions"])))
                cost = f32(cost + f32(f32(.00001) * f32(event["primitive_steps"])))
            if op == 3:
                cost = f32(cost + f32(f32(.02) * f32(len(event["actions"]))))
            need(cost == event["cost"], f"{name}: charged cost at {len(trace)}: {cost} != {event['cost']}")
            cost_checks += 1
            if op == 5:
                inspected = event["inspected_after"]
                need(inspected in requests and inspected not in closed, f"{name}: inspection target")
                seen.add(inspected)
                agenda = True
            elif op == 6:
                active, agenda = inspected, False
                need(event["callback"], f"{name}: missing activation callback")
            elif op in (0, 2, 3):
                world_id = requests[active]["world_id"]
                facts, revision = worlds[world_id]
                if op == 3:
                    facts = apply_actions(facts, event["actions"])
                    revision += len(event["actions"])
                    motors += len(event["actions"])
                if event["actions"] or op in (0, 2):
                    observed = event["observed_world"]
                    need(event["callback"] and scene_facts(observed) == facts and observed["revision"] == revision, f"{name}: independent evaluator callback")
                worlds[world_id] = (facts, revision)
            elif op == 4:
                request = requests[active]
                truth = satisfaction(request["text"], worlds[request["world_id"]][0])
                correct = event["reply"] == int(truth) and (truth or not request["effect"])
                need(event["terminal_correct"] == correct and event["terminal_satisfaction"] == truth, f"{name}: terminal label")
                closed[active] = {"correct": correct, "clock": event["clock"], "deadline": request["deadline"]}
                active = inspected = ""
                agenda = True
            need(event["active_after"] == active and event["inspected_after"] == inspected, f"{name}: callback state")
            if not run["semantic_learning"]:
                need(event["model_before"] == event["model_after"] == run["model_start"], f"{name}: frozen semantic mutation")
        success = "arrival" in requests and set(closed) == set(requests) and all(row["correct"] and (not row["deadline"] or row["clock"] <= row["deadline"]) for row in closed.values())
        need(success == run["success"] and len(trace) == run["decisions"] and motors == run["motors"], f"{name}: run totals")
        need(sum(row["predictions"] for row in trace) == run["predictions"], f"{name}: predictions")
        need(sum(row["primitive_steps"] for row in trace) == run["primitive_steps"], f"{name}: primitive count")
        need(run["final_credit_receipts"] == int(run["semantic_learning"]), f"{name}: terminal credit count")
        need((run["model_end"] > run["model_start"]) if run["semantic_learning"] else (run["model_end"] == run["model_start"]), f"{name}: semantic mode")
        values = arithmetic(trace, success)
        for key, value in values.items():
            near(value, run[key], name + ": " + key)
        audited.append({"id": name, "strategy": run["strategy"], "success": success, "decisions": len(trace), **values})
    exported_aliases = {row["key"]: row for row in jsonl(aliases_path)}
    need(set(exported_aliases) == set(aliases), "switch alias inventory")
    for key, group in aliases.items():
        need(exported_aliases[key]["rows"] == group["rows"], "switch alias row count")
        for field in ("operations", "contexts", "runs"):
            need(set(exported_aliases[key][field]) == group[field], "switch alias " + field)
    references = [row for row in audited if row["strategy"].startswith("reference")]
    failures = [row for row in audited if row["strategy"] in {"immediate_failure", "delayed_failure"}]
    strongest = max(row["utility"] for row in failures)
    bound = -(.99 ** 95)
    margins = {"strongest_checked_failure_utility": strongest, "optimistic_zero_cost_failure_bound": bound,
               "worst_reference_margin_vs_checked": min(row["utility"] - strongest for row in references),
               "worst_reference_margin_vs_bound": min(row["utility"] - bound for row in references)}
    for key, value in margins.items():
        near(value, summary[key], "switch " + key)
    need(len(references) == summary["reference_runs"] and sum(row["success"] for row in references) == summary["reference_successes"], "switch reference counts")
    return {"passed": True, "artifacts": {str(path): sha(path) for path in (summary_path, trace_path, aliases_path)},
            "source_manifest": check_manifest(root), "runs": len(audited), "reference_successes": sum(row["success"] for row in references),
            "decision_costs_reconstructed": cost_checks, "public_observations_checked": observation_checks,
            "arrival_bits_reconstructed_from_events": arrival_checks, "alias_classes": len(aliases),
            "cross_context_alias_classes": sum(len(g["contexts"]) > 1 for g in aliases.values()),
            "action_conflicts_on_visited_support": sum(len(g["operations"]) > 1 for g in aliases.values()),
            **margins, "run_arithmetic": audited,
            "limits": ["Alias action agreement is expected for a deterministic reference; it is not proof over counterfactual or unvisited decision points.",
                       "There are no cross-context alias groups in this narrow root-type fixture.",
                       "Zero policy updates and restored weights are native runtime assertions reviewed in frozen source; weights are not exported for a second independent comparison.",
                       "The initial-policy control is a zero-output policy, not a trained warm checkpoint."]}


def audit_long(path, config_path=None):
    data = read_json(path)
    need(data["schema"] == "goal_long_preflight_v1", "unsupported long schema")
    need(data["training_run"] is False, "preflight unexpectedly reports training")
    config_path = config_path or path.parent / "phase3_config.json"
    config = read_json(config_path)
    if "variants" in config:
        variants = [variant for variant in config["variants"] if Path(variant["output"]).resolve() == path.resolve()]
        need(len(variants) == 1, "long artifact must match exactly one declared variant")
        variant = variants[0]
        grid = config["unchanged"]
        horizon, price_scale = variant.get("decision_limit", grid.get("decision_limit")), variant["price_scale"]
        ticks = grid["arrival_ticks"]
        need(data["decision_limit"] == horizon and data["price_scale"] == price_scale,
             "long header disagrees with declared variant")
    else:
        horizon, price_scale = config["mission"]["decisions"], 1
        grid, ticks = config, config["reference_arrival_ticks"]
    reference_variant = config.get("reference_variant", 0)
    need(data.get("reference_variant", 0) == reference_variant, "long reference variant header")
    expected_prices = [f32(f32(value) * f32(price_scale)) for value in (.01, .005, .06, .01, 0, .002, .01)]
    need(data["operation_prices"] == expected_prices, "long declared operation prices")
    need(data["motor_price"] == f32(f32(.02) * f32(price_scale)), "long declared motor price")
    need(data["execution_prediction_price"] == f32(f32(.00001) * f32(price_scale)), "long declared prediction price")
    need(data["rebuild_price"] == f32(f32(.005) * f32(price_scale)), "long declared rebuild price")
    need(data["preview_prices"] == [value * price_scale for value in (.00001, .0000001, .0000001, .005, .005)], "long declared preview prices")
    groups = {"references": 0, "semantic_learning_references": 0, "actual_direct_uniform": 1, "actual_direct_deliberate": 2}
    results = {}
    aliases = defaultdict(lambda: {"rows": 0, "operations": set(), "contexts": set()})
    arrival_checks = terminal_checks = computation_checks = preview_minima_checks = preview_feature_checks = 0
    for group, direct in groups.items():
        expected_grid = ({(seed, tick) for seed in grid.get("reference_seeds", grid.get("seeds")) for tick in ticks} if direct == 0 else
                         {(seed, grid.get("direct_control_arrival_tick", 12)) for seed in grid.get("direct_control_seeds", grid.get("seeds"))})
        actual_grid = [(run["seed"], run["arrival_tick"]) for run in data[group]]
        need(len(actual_grid) == len(expected_grid) and set(actual_grid) == expected_grid, "long declared seed/arrival grid")
        runs = []
        for run in data[group]:
            need(run.get("decision_limit", horizon) == horizon and run.get("price_scale", price_scale) == price_scale,
                 "long run disagrees with variant")
            need(run["semantic_learning"] == (group == "semantic_learning_references"), "long group semantic mode")
            need(run.get("reference_variant", 0) == reference_variant, "long run reference variant")
            trace = run["trace"]
            need(len(trace) == run["decisions"] and len(run["computations"]) == run["paid_agenda_computations"], "long row counts")
            need("public_requests" in run and "submissions" in run and "goal_bindings" in run, "long final event export required")
            bindings = {row["id"]: row for row in run["goal_bindings"]}
            requests_by_clock = defaultdict(list)
            for request in run["public_requests"]:
                requests_by_clock[request["clock"]].append(request)
            requests, worlds, seen, closed = {}, {}, set(), {}
            submissions = {row["id"]: row for row in run["submissions"]}
            need(len(submissions) == len(run["submissions"]), "duplicate long submission")
            computations = {row["token"]: row for row in run["computations"]}
            derived_computations = [row for row in run["computations"] if "preview" in row]
            max_allowance = max((row["preview_prediction_limit"] for row in derived_computations), default=0)
            cap_exhaustions = sum(row["preview_prediction_limit"] == data["preview_planner_cap"] and
                                  row["plan_status"] == "search_budget" for row in derived_computations)
            if "global_preview_exhaustions" in run:
                need(run["global_preview_exhaustions"] == cap_exhaustions and run["max_derived_preview_allowance"] == max_allowance,
                     "long independently counted preview allowance/exhaustion")
            paid_previews = {}
            motors = 0
            active_unfinished_at_arrival = False
            for index, row in enumerate(trace, 1):
                need(row["clock"] == row["token"] == index, "long chronology")
                for request in requests_by_clock[index - 1]:
                    need(request["id"] not in requests, "duplicate long public request")
                    need(request["clock"] == (0 if request["id"] == "root" else run["arrival_tick"]), "long posted clock")
                    world_id = request["world_id"]
                    if world_id in worlds:
                        need(scene_facts(request["visible"]) == worlds[world_id][0] and request["visible_revision"] == worlds[world_id][1], "long arrival world differs from prior action replay")
                    else:
                        worlds[world_id] = (scene_facts(request["visible"]), request["visible_revision"])
                    requests[request["id"]] = request
                o, menu = row["features"], row["menu"]
                need(row["operation"] == long_reference(o, menu, direct, reference_variant), "long reference used extra input")
                if o[29]:
                    if o[18]:
                        prior = paid_previews.get(row["inspected"])
                        need(prior and prior["preview"]["valid"] and f32(prior["preview"]["normalized_delta"]) == o[19],
                             "public parent-progress feature lacks a matching prior paid preview")
                        preview_feature_checks += 1
                    else:
                        need(o[19] == 0, "invalid public preview retained progress")
                if not o[29]:
                    need(o[30] == int(bool(set(requests) - seen - set(closed))), "long public arrival bit")
                    arrival_checks += 1
                op = row["operation"]
                if op != 1:
                    expected_cost = expected_prices[op]
                    if op == 3:
                        expected_cost = f32(expected_cost + f32(data["motor_price"] * f32(len(row["actions"]))))
                    need(expected_cost == row["cost"], "long non-reason operation charge")
                if price_scale == 0:
                    need(row["cost"] == 0, "long success-first trace retains a charge")
                if op == 5 and index < len(trace):
                    seen.add(trace[index]["inspected"])
                if op == 3:
                    need(row["goal"] in bindings, "unknown active long goal")
                    world_id = bindings[row["goal"]]["world_id"]
                    facts, revision = worlds[world_id]
                    worlds[world_id] = (apply_actions(facts, row["actions"]), revision + len(row["actions"]))
                    motors += len(row["actions"])
                if op == 4:
                    binding = bindings[row["goal"]]
                    truth = satisfaction(binding["text"], worlds[binding["world_id"]][0])
                    correct = truth if binding["effect"] else row["reply"] == int(truth)
                    expected = {"id": row["goal"], "parent": binding["parent"], "clock": index,
                                "deadline": binding["deadline"], "correct": correct, "satisfaction": truth}
                    need(submissions.get(row["goal"]) == expected, "long independently reconstructed terminal submission")
                    closed[row["goal"]] = expected
                    terminal_checks += 1
                if index == run["arrival_tick"]:
                    active_after = row["inspected"] if op == 6 else "" if op == 4 else row["goal"]
                    if active_after:
                        binding = bindings[active_after]
                        active_unfinished_at_arrival = binding["effect"] and not satisfaction(binding["text"], worlds[binding["world_id"]][0])
                if op == 1 and o[29]:
                    computation = computations[row["token"]]
                    candidate = bindings[computation["candidate"]]
                    need(computation["candidate"] == row["inspected"] and computation["candidate_text"] == candidate["text"], "long computation candidate identity")
                    facts, revision = worlds[candidate["world_id"]]
                    need(scene_facts(computation["world"]) == facts and computation["revisions"][0] == revision,
                         "long paid computation used a non-public world")
                    if "preview" in computation:
                        paid_previews[row["inspected"]] = computation
                    computation_checks += 1
                if direct == 0:
                    key = bit_key(o, menu)
                    aliases[key]["rows"] += 1
                    aliases[key]["operations"].add(row["operation"])
                    aliases[key]["contexts"].add(row["goal"] + "/" + row["inspected"])
            need(set(closed) == set(submissions), "untraced long submission")
            roots = {key: row for key, row in closed.items() if not row["parent"]}
            success = len(roots) == 2 and all(row["correct"] and (not row["deadline"] or row["clock"] <= row["deadline"]) for row in roots.values())
            need(success == run["success"] and motors == run["motors"], "long independently reconstructed success/motors")
            urgent_on_time = "arrival" in roots and roots["arrival"]["correct"] and roots["arrival"]["clock"] <= roots["arrival"]["deadline"]
            need(urgent_on_time == run["urgent_on_time"], "long urgent deadline")
            need(active_unfinished_at_arrival == run["arrival_active_unfinished"], "long arrival context")
            need(run["policy_updates"] == 0 and run["unchanged_policy"], "long policy update")
            need(run["credit_receipts"] == int(run["semantic_learning"]), "long credit count")
            need(run["knowledge_unchanged"] == (run["semantic_revisions"] == 0), "long semantic count")
            need((run["semantic_revisions"] > 0) if run["semantic_learning"] else (run["semantic_revisions"] == 0), "long semantic mode")
            values = arithmetic(trace, success)
            for key in ("cost", "discounted_cost", "utility"):
                near(values[key], run[key], "long " + key)
            need(sum(row["predictions"] for row in trace) == run["predictions"], "long prediction count")
            for row in trace:
                if row["operation"] == 1 and row["features"][29]:
                    computation = computations[row.get("token", row["clock"])]
                    costs = computation["cost_components"]
                    need(costs[0] == expected_prices[1], "long agenda operation price")
                    if price_scale == 0:
                        need(all(value == 0 for value in costs), "long success-first computation retains a charge")
                    cost = f32(costs[0] + costs[1])
                    cost = f32(cost + costs[2])
                    if "preview" in computation:
                        cost = f32(cost + f32(computation["preview"]["cost"]))
                    if "proposals" in computation:
                        cost = f32(cost + f32(computation["proposals"]["cost"]))
                    cost = f32(cost + costs[3])
                    need(cost == row["cost"] == computation["total_cost"], "long computation charge")
                    if "preview" in computation:
                        preview = computation["preview"]
                        need(sum(preview["counts"]) == preview["work"], "long preview work count")
                        near(sum(count * price for count, price in zip(preview["counts"], data["preview_prices"])),
                             preview["cost"], "long independent preview work price")
                        if preview["valid"]:
                            need(preview["delta"] == preview["before_missing"] - preview["after_missing"], "long preview delta")
                            near(preview["normalized_delta"], preview["delta"] / computation["normalization_atoms"], "long normalized delta")
                            before = scene_facts(computation["world"])
                            after = scene_facts(preview["predicted"])
                            need(after == apply_actions(before, computation["plan_actions"]), "preview predicted scene disagrees with independently replayed learned edge effects")
                            potential = scene_facts({"facts": preview["potential"]})
                            need(potential == apply_actions(before | after, computation["actions"]), "preview potential domain differs from public before/after affordance effects")
                            length = route_definition_length(preview["parent_definition"])
                            need(minimum_missing(potential, before, preview["parent_args"], length) == preview["before_missing"], "independent before missing-atom minimum")
                            need(minimum_missing(potential, after, preview["parent_args"], length) == preview["after_missing"], "independent after missing-atom minimum")
                            preview_minima_checks += 1
            runs.append({"seed": run["seed"], "arrival_tick": run["arrival_tick"], "success": run["success"],
                         "arrival_active_unfinished": active_unfinished_at_arrival, "decisions": len(trace), "motors": motors,
                         "max_derived_preview_allowance": max_allowance, "global_preview_exhaustions": cap_exhaustions, **values})
        results[group] = {"runs": len(runs), "successes": sum(row["success"] for row in runs),
                          "arrival_active_unfinished": sum(row["arrival_active_unfinished"] for row in runs),
                          "maximum_decisions": max(row["decisions"] for row in runs), "maximum_motors": max(row["motors"] for row in runs),
                          "maximum_derived_preview_allowance": max(row["max_derived_preview_allowance"] for row in runs),
                          "global_preview_exhaustions": sum(row["global_preview_exhaustions"] for row in runs), "arithmetic": runs}
    manifest_path = Path(config["source_manifest"])
    manifest = read_json(manifest_path)
    for rel, expected in manifest.items():
        need(sha(Path(config["frozen_source"]) / rel) == expected, "long frozen source mismatch: " + rel)
    for group in groups:
        for run in data[group]:
            roots = [request for request in run["public_requests"] if request["id"] == "root"]
            need(len(roots) == 1 and roots[0]["deadline"] == horizon and run["decisions"] <= horizon,
                 "long variant horizon/deadline mismatch")
    bound = -(.99 ** (horizon - 1))
    near(data["zero_cost_failure_upper_bound"], bound, "long failure bound")
    return {"passed": True, "artifact": str(path), "sha256": sha(path), "groups": results,
            "config": {"path": str(config_path), "sha256": sha(config_path), "horizon": horizon,
                       "price_scale": price_scale, "reference_variant": reference_variant},
            "source_manifest": {"path": str(manifest_path), "sha256": sha(manifest_path), "verified_files": len(manifest)},
            "arrival_bits_reconstructed_from_events": arrival_checks, "terminal_submissions_reconstructed": terminal_checks,
            "paid_computation_public_worlds_checked": computation_checks,
            "preview_minima_recomputed_by_independent_dynamic_program": preview_minima_checks,
            "public_progress_features_linked_to_prior_paid_previews": preview_feature_checks,
            "alias_classes": len(aliases), "cross_context_alias_classes": sum(len(g["contexts"]) > 1 for g in aliases.values()),
            "action_conflicts_on_visited_support": sum(len(g["operations"]) > 1 for g in aliases.values()),
            "worst_frozen_reference_utility": min(row["utility"] for row in results["references"]["arithmetic"]),
            "worst_semantic_reference_utility": min(row["utility"] for row in results["semantic_learning_references"]["arithmetic"]),
            "zero_cost_failure_upper_bound": bound,
            "reward_and_arrival_audit": "Recomputed from public requests, full action trace, declared edge-add evaluator physics, goal bindings and terminal replies/submissions.",
            "limits": ["Only visited reference support is grouped; direct controls are not treated as desired action labels.",
                       "Agreement with a deterministic public rule is not a proof that every reachable feature class has a feasible common action.",
                       "Long export omits per-action observed-world callbacks; replay uses the frozen evaluator's declared deterministic successful edge-add physics, cross-checked at request and paid-computation snapshots.",
                       "Physical reason rows omit raw primitive-step counts; total return and agenda cost components are reconstructed, but all physical inference work cannot be independently remetered from this export.",
                       "No trained warm-policy deficit, acquisition, or held-out generalization is established."]}


def compare_long_prices(original_path, free_path):
    original, free = read_json(original_path), read_json(free_path)
    need(original["price_scale"] == 1 and free["price_scale"] == 0, "expected original/zero-price twins")
    groups = ("references", "semantic_learning_references", "actual_direct_uniform", "actual_direct_deliberate")
    row_checks = computation_checks = 0
    success_utilities, failure_utilities, failed_reference_diagnostics = [], [], []
    for group in groups:
        need(len(original[group]) == len(free[group]), "price twin group length")
        for paid, zero in zip(original[group], free[group]):
            for field in paid:
                if field not in {"price_scale", "cost", "discounted_cost", "utility", "trace", "computations"}:
                    need(paid[field] == zero[field], "price twin run change: " + field)
            need(zero["cost"] == zero["discounted_cost"] == 0, "zero-price twin cost")
            need((zero["utility"] > 0) == zero["success"], "zero-price objective fails to prefer completion")
            (success_utilities if zero["success"] else failure_utilities).append(zero["utility"])
            need(len(paid["trace"]) == len(zero["trace"]), "price twin trace count")
            for a, b in zip(paid["trace"], zero["trace"]):
                for field in a:
                    if field == "features":
                        price_end = 28 if a[field][29] else 29
                        need(a[field][:21] + a[field][price_end:] == b[field][:21] + b[field][price_end:], "price twin non-price observation change")
                        need(all(value == 0 for value in b[field][21:price_end]), "zero-price observation retains a price")
                    elif field != "cost":
                        need(a[field] == b[field], "price twin trace change: " + field)
                row_checks += 1
            need(len(paid["computations"]) == len(zero["computations"]), "price twin computation count")
            for a, b in zip(paid["computations"], zero["computations"]):
                for field in a:
                    if field in {"preview", "proposals"}:
                        need({key: value for key, value in a[field].items() if key != "cost"} ==
                             {key: value for key, value in b[field].items() if key != "cost"}, "price twin raw computation change")
                        need(b[field]["cost"] == 0, "zero-price nested computation charge")
                    elif field not in {"cost_components", "total_cost"}:
                        need(a[field] == b[field], "price twin computation change: " + field)
                computation_checks += 1
            if group == "semantic_learning_references" and not paid["success"]:
                skipped = [row for row in paid["trace"] if row["features"][29] and row["features"][13] and
                           row["features"][5] and row["features"][10] and row["operation"] == 5]
                failed_reference_diagnostics.append({"seed": paid["seed"], "arrival_tick": paid["arrival_tick"],
                    "decisions": paid["decisions"], "derived_cached_search_budget_skips": len(skipped),
                    "skips_after96": sum(row["clock"] > 96 for row in skipped),
                    "maximum_paid_derived_preview_allowance": max((row["preview_prediction_limit"] for row in paid["computations"] if row["parent"]), default=0)})
    return {"passed": True, "artifacts": {str(path): sha(path) for path in (original_path, free_path)},
            "unchanged_trace_rows": row_checks, "unchanged_raw_computations": computation_checks,
            "zero_price_success_utility_minimum": min(success_utilities),
            "zero_price_failure_utility_maximum": max(failure_utilities),
            "failed_semantic_reference_diagnostics": failed_reference_diagnostics,
            "interpretation": "Only declared prices, public price features, charges and returns differ. The failure diagnostics identify observed reference skips, not learned-policy incapacity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switch-root", type=Path, default=Path("logs/goal_switch_preflight_20260923_v3"))
    parser.add_argument("--long-json", type=Path)
    parser.add_argument("--long-config", type=Path)
    parser.add_argument("--long-price-twin", type=Path, help="Compare with a separately declared zero-price variant of the same long run.")
    parser.add_argument("--output", type=Path, default=Path("logs/goal_preflight_independent_audit_v1/audit.json"))
    args = parser.parse_args()
    result = {"schema": "independent_goal_preflight_audit_v1", "training_run": False,
              "switch": audit_switch(args.switch_root)}
    if args.long_json:
        result["long"] = audit_long(args.long_json, args.long_config)
        if args.long_price_twin:
            result["price_twin"] = compare_long_prices(args.long_json, args.long_price_twin)
    result["passed"] = all(result[key]["passed"] for key in ("switch", "long") if key in result)
    result["auditor_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(args.output),
                      "switch_runs": result["switch"]["runs"], "long_included": "long" in result}, indent=2))


if __name__ == "__main__":
    main()
