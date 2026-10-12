"""Independent physical and hypothesis replay for live report-word learning."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
from check_online_grounding import world, menu, task, satisfied, execute
from check_request_grounding import words, observe, parse_facts, expected_facts, qa_target

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/live_observation_grounding"


def observe_report(owners, places, text):
    w = words(text)
    if len(w) == 9 and w[0] == "the" and w[2:4] == ["passed", "from"]:
        assert w[5] == "into" and w[7:] == ["s", "keeping"] and owners[w[1]] == w[4]
        owners[w[1]] = w[6]
    elif len(w) == 8 and w[1:3] == ["relinquished", "the"]:
        assert w[4] == "into" and w[6:] == ["s", "care"] and owners[w[3]] == w[0]
        owners[w[3]] = w[5]
    elif len(w) == 4 and w[1:3] == ["relocated", "into"]:
        places[w[0]] = w[3]
    elif len(w) == 6 and w[:3] == ["the", "destination", "of"]:
        assert w[4] == "became"
        places[w[3]] = w[5]
    else:
        observe(owners, places, text)


def bind(text, names):
    args, pattern = [], []
    for token in words(text):
        if token in names:
            if token not in args:
                args.append(token)
            pattern.append("$"+str(args.index(token)))
        else:
            pattern.append(token)
    return " ".join(pattern)+" ", args


def update(facts, args, rule):
    relation, subject, obj, effect = rule
    a, b = args[subject], args[obj]
    changed = {f for f in facts if not (f[1] == relation and ((effect == 1 and f[0] == a and f[2] == b) or (effect == 2 and f[0] == a) or (effect == 3 and f[2] == b)))}
    if effect != 1:
        changed.add((a, relation, b))
    return changed


def question_answer(question, facts, names, seed):
    w = words(question);name = w[-1]
    if "items" in w:
        return len({a for a, r, b in facts if r == 1-seed%2 and b == name})
    if "destinations" in w:
        return len({b for a, r, b in facts if r == seed%2 and a == name})
    assert w[:2] == ["where", "is"]
    # Follow the first native-ID-ordered edge. Cycles cannot complete the bounded
    # learned follow call. This is an independent symbolic interpreter, not a
    # call into the native learner or its scoring outputs.
    order = {n: i for i, n in enumerate(names)}
    current, seen = name, set()
    while current not in seen:
        seen.add(current)
        edges = [b for a, r, b in facts if a == current and r == seed%2]
        if not edges:
            return current
        current = min(edges, key=order.__getitem__)
    return None


def parse_rules(text):
    return {tuple(map(int, r.split(","))) for r in text.split(";") if r}


def rows(seed, suffix):
    return list(csv.DictReader((OUT/f"{seed}.{suffix}.tsv").open(), delimiter="\t"))


def main():
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    totals = {}
    for seed in seeds:
        score = collections.Counter()
        reqs, trace, answers = rows(seed, "requests"), rows(seed, "trace"), rows(seed, "answers")
        assert [int(r["request_id"]) for r in reqs] == list(range(160))
        assert [int(r["event_id"]) for r in trace] == list(range(len(trace)))
        assert [(int(r["request_id"]), int(r["attempt"])) for r in trace] == [(i, j) for i, req in enumerate(reqs) for j in range(int(req["attempts"]))]
        by_request, by_event = collections.defaultdict(list), collections.defaultdict(list)
        predictions = {}
        for row in rows(seed, "predictions"):
            key = (int(row["event_id"]), int(row["answer_id"]), int(row["question_index"]), next(iter(parse_rules(row["rule"]))))
            assert key not in predictions
            kind = int(row["type"])
            assert kind in (-1, 0, 2)
            if kind == -1:
                assert row["value"] == ""
            predictions[key] = None if kind == -1 else int(row["value"]) if kind == 2 else row["value"]
        used_predictions = set()
        for row in trace:
            by_request[int(row["request_id"])].append(row)
        for row in answers:
            by_event[int(row["event_id"])].append(row)
        assert [(int(r["event_id"]), int(r["answer_id"])) for r in answers] == [(int(t["event_id"]), i) for t in trace for i in range(int(t["questions"]))]
        people, rooms, objects, owners, places = world("evolving")
        names = [v for pair in zip(people, rooms) for v in pair]+objects
        questions = [q for name in names for q in (f"What is the number of items held by {name}?", f"What is the number of destinations for {name}?")]+[f"Where is {p}?" for p in people]
        hypotheses, scenes, states, checkpoints = {}, collections.defaultdict(set), {}, {}
        for i, req in enumerate(reqs):
            stage, within = divmod(i, 40);family, version = within%(stage+1), int(stage == 3)
            assert int(req["family"]) == family and req["request"] == task(family, i, people, rooms, objects, owners, places)
            assert not satisfied(req["request"], owners, places) and req["initial_prediction"] in ("-1", "0")
            for attempt, row in enumerate(by_request[i]):
                event_id = int(row["event_id"]);assert row["request"] == req["request"]
                before = expected_facts(owners, places, seed)
                assert parse_facts(row["before"]) == before
                offered = [tuple(a.split(",")) for a in row["menu"].split(";") if a]
                assert len(set(offered)) == len(offered) and set(offered) == menu(people, rooms, owners, places, seed, version)
                action = tuple(row["action"].split(","));assert offered[int(row["index"])] == action
                executed = row["executed"] == "1";assert executed == (event_id%7 != 0)
                physical_owner, physical_place = dict(owners), dict(places)
                execute(action, executed, physical_owner, physical_place, seed, version)
                observe_report(owners, places, row["report"])
                assert (owners, places) == (physical_owner, physical_place)
                actual_after = expected_facts(owners, places, seed)
                staged = row["staged"] == "1";assert staged == executed
                if staged:
                    frame, args = bind(row["report"], names);assert frame == row["frame"]
                    if frame not in hypotheses:
                        hypotheses[frame] = {(r, a, b, e) for r in (0, 1) for a in range(len(args)) for b in range(len(args)) for e in range(4)}
                    candidates = hypotheses[frame]
                    if len(scenes[frame]) < 4:
                        assert row["initial_status"] == "unverified_observation" and int(row["questions"]) > 0
                    else:
                        assert row["initial_status"] == "verified_observation" and row["questions"] == "0"
                    for qa in by_event[event_id]:
                        assert parse_rules(qa["before_rules"]) == candidates
                        worlds = {frozenset(update(before, args, rule)) for rule in candidates}
                        assert int(qa["before_worlds"]) == len(worlds)
                        options = qa["menu"].rstrip("|").split("|");assert options == questions
                        scores, ideal_scores = [], []
                        for qi, q in enumerate(questions):
                            group, ideal_group = collections.Counter(), collections.Counter()
                            for rule in candidates:
                                key = (event_id, int(qa["answer_id"]), qi, rule)
                                assert key not in used_predictions
                                used_predictions.add(key)
                                value = predictions[key]
                                ideal = question_answer(q, update(before, args, rule), names, seed)
                                # A bounded learned skill may fail to return.
                                # Every definite prediction must still match
                                # the independent symbolic interpretation.
                                assert value is None or value == ideal, (seed, key, q, value, ideal)
                                score["counterfactual_predictions"] += 1
                                score["counterfactual_policy_abstentions"] += value is None and ideal is not None
                                group[value] += 1
                                ideal_group[ideal] += 1
                            scores.append((len(candidates)-max(group.values()), len(group), any(v is not None for v in group)))
                            ideal_scores.append(len(candidates)-max(ideal_group.values()))
                        eligible = [n for n in range(len(scores)) if scores[n][2]]
                        selected = max(eligible, key=lambda n: scores[n][0])
                        assert int(qa["choice"]) == selected and qa["question"] == questions[selected]
                        assert (int(qa["score"]), int(qa["partitions"])) == scores[selected][:2], (seed, event_id, qa, scores[selected])
                        score["question_choices_different_from_ideal"] += selected != max(range(len(ideal_scores)), key=ideal_scores.__getitem__)
                        gold = question_answer(qa["question"], actual_after, names, seed)
                        assert gold is not None and qa["answer"] == str(gold)
                        candidates = {rule for rule in candidates if question_answer(qa["question"], update(before, args, rule), names, seed) == gold}
                        assert candidates and parse_rules(qa["after_rules"]) == candidates
                        assert int(qa["after_worlds"]) == len({frozenset(update(before, args, rule)) for rule in candidates})
                        hypotheses[frame] = candidates;scenes[frame].add(event_id)
                        score["verified_answers"] += 1
                        score["distinguishing_answers"] += int(qa["score"]) > 0
                    predicted = {frozenset(update(before, args, rule)) for rule in candidates}
                    assert predicted == {frozenset(actual_after)}
                    score["new_report_events"] += 1
                else:
                    assert not row["frame"] and row["questions"] == "0" and row["initial_status"] == "prior_observation"
                    score["familiar_refusals"] += 1
                assert parse_facts(row["after"]) == actual_after and row["agree"] == "1" and row["accepted"] == "1"
                truth = int(satisfied(req["request"], owners, places))
                assert int(row["satisfied"]) == truth and int(row["prediction"]) in (-1, truth)
                if attempt+1 < len(by_request[i]):
                    assert not truth
                score["transitions"] += 1
            assert int(req["complete"]) == int(satisfied(req["request"], owners, places)) == 1
            score["completed"] += 1;states[i] = dict(owners), dict(places)
            if within == 39:
                checkpoints[stage] = {key: (len(scenes[key]), set(rules)) for key, rules in hypotheses.items()}
        seen = set()
        for row in rows(seed, "retention"):
            id = int(row["request_id"]);key = id, row["question"]
            assert key not in seen;seen.add(key)
            o, p = states[id]
            assert row["reply"] == qa_target(row["question"], o, p)
            assert parse_facts(row["facts"]) == expected_facts(o, p, seed)
        assert seen == {(i, q) for i in range(160) for q in (f"Who has the evolvingitem{i%3} now?", f"Where is evolvingperson{i%4}?")}
        score["retained_qa"] = len(seen)
        serialized, stage = collections.defaultdict(dict), None
        for line in (OUT/f"{seed}.models.tsv").read_text().splitlines():
            if line.startswith("checkpoint\t"):
                stage = int(line.split("\t")[1])
            else:
                key, n, rules = line.split(":")
                assert key not in serialized[stage]
                serialized[stage][key] = int(n), parse_rules(rules)
        assert dict(serialized) == checkpoints
        assert used_predictions == set(predictions)
        expected = {
            "$0 relinquished the $1 into $2 s care ": (1-seed%2, 1, 2, 2),
            "the $0 passed from $1 into $2 s keeping ": (1-seed%2, 0, 2, 2),
            "$0 relocated into $1 ": (seed%2, 0, 1, 2),
            "the destination of $0 became $1 ": (seed%2, 0, 1, 2),
        }
        assert set(hypotheses) == set(expected)
        for key, wanted in expected.items():
            assert hypotheses[key] == {wanted} and len(scenes[key]) == 4
        held = rows(seed, "holdout")
        assert [(r["mode"], int(r["version"]), int(r["case"])) for r in held] == [(m, v, c) for m in ("grounded", "no_new_reports") for v in range(2) for c in range(48)]
        for row in held:
            mode, v, c = row["mode"], int(row["version"]), int(row["case"])
            hp, hr, ho, owners, places = world(f"newreport{v}case{c}")
            assert row["request"] == task(c%4, c, hp, hr, ho, owners, places)
            before = expected_facts(owners, places, seed);assert parse_facts(row["before"]) == before
            offered = [tuple(a.split(",")) for a in row["menu"].split(";") if a]
            assert len(set(offered)) == len(offered) and set(offered) == menu(hp, hr, owners, places, seed, v)
            action = tuple(row["action"].split(","));assert offered[int(row["index"])] == action
            physical_owner, physical_place = dict(owners), dict(places)
            execute(action, True, physical_owner, physical_place, seed, v)
            observe_report(owners, places, row["report"])
            assert (owners, places) == (physical_owner, physical_place) and satisfied(row["request"], owners, places) and row["completed"] == "1"
            if mode == "grounded":
                assert row["truth"] == row["agree"] == "1" and row["status"] == "complete"
                assert parse_facts(row["after"]) == expected_facts(owners, places, seed)
                score["frozen_transfers"] += 1
            else:
                assert row["truth"] == "-1" and row["agree"] == "0" and row["status"] == "unknown_event"
                assert parse_facts(row["after"]) == before
                score["without_report_meanings_withheld"] += 1
        totals[str(seed)] = score
    paths = list(OUT.glob("*.tsv"))
    paths += [ROOT/name for name in ("cpp/include/laminar/live_observation_grounding.h", "cpp/include/laminar/answer_grounding.h", "cpp/include/laminar/online_grounding.h", "cpp/include/laminar/reasoning_language.h", "cpp/apps/live_observation_probe.cpp", "cpp/apps/interaction_probe_support.h", "cpp/tests/test_live_observation_grounding.cpp", "python/check_live_observation_grounding.py", "python/check_online_grounding.py", "python/check_request_grounding.py", "scripts/live_observation_grounding.sh", "build-wsl/live_observation_probe", "logs/bootstrap_grounding/train.txt", "logs/incremental_grounding/train.txt", "logs/dialogue_grounding/train.txt")]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = dict(seeds=seeds, totals=totals, failures=[], sha256=hashes)
    (OUT/("audit.json" if seeds == [0, 1, 2] else "partial_audit.json")).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(totals, indent=2))
    print("Verified every live report hypothesis, selected question/answer, physical transition, learned frame, frozen transfer and ablation.")


if __name__ == "__main__":
    main()
