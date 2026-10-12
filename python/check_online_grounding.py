"""Independently replay the continuing world and feedback, including failures.

Goal bindings and expected outcomes are reconstructed here, outside the learner.
No correctness claim relies on the native process merely exiting successfully.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
from check_request_grounding import observe, words, parse_facts, expected_facts, qa_target

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/online_grounding"


def world(prefix):
    people = [prefix+f"person{i}" for i in range(4)]
    rooms = [prefix+f"room{i}" for i in range(4)]
    objects = [prefix+f"item{i}" for i in range(3)]
    return people, rooms, objects, {o: people[i%2] for i, o in enumerate(objects)}, dict(zip(people, rooms))


def ports(seed, version):
    base = 1000+version*1000
    transfer, movement = ((31, 73) if seed%2 else (73, 31))
    return f"socket{base+transfer}", f"socket{base+movement}"


def menu(people, rooms, owners, places, seed, version):
    transfer, movement = ports(seed, version)
    return {(transfer, p, o, t) for o, p in owners.items() for t in people if t != p} | {(movement, p, t) for p, r in places.items() for t in rooms if t != r}


def task(family, serial, people, rooms, objects, owners, places):
    if family == 1:
        p, target = people[(serial//3)%4], rooms[(serial+1)%4]
        if places[p] == target:
            target = rooms[(serial+2)%4]
        return f"Please have {p} go to {target}."
    obj = objects[(serial//4)%3]
    giver, target = owners[obj], people[(serial+1)%4]
    if target == giver:
        target = people[(serial+2)%4]
    if family == 0:
        return f"Please have {giver} give the {obj} to {target}."
    if family == 2:
        return f"Arrange for {obj} to be held by {target}."
    assert family == 3
    return f"Ensure {obj} is not held by {giver}."


def satisfied(request, owners, places):
    w = words(request)
    if w[:2] == ["please", "have"]:
        if w[3:5] == ["go", "to"]:
            assert len(w) == 6
            return places[w[2]] == w[5]
        assert w[3:5] == ["give", "the"] and w[6] == "to" and len(w) == 8
        return owners[w[5]] == w[7]
    if w[:2] == ["arrange", "for"]:
        assert w[3:7] == ["to", "be", "held", "by"] and len(w) == 8
        return owners[w[2]] == w[7]
    assert w[0] == "ensure" and w[2:6] == ["is", "not", "held", "by"] and len(w) == 7
    return owners[w[1]] != w[6]


def execute(action, executed, owners, places, seed, version):
    transfer, movement = ports(seed, version)
    if action[0] == transfer:
        _, giver, obj, target = action
        assert owners[obj] == giver and giver != target
        if executed:
            owners[obj] = target
    else:
        assert action[0] == movement
        _, person, room = action
        assert places[person] != room
        if executed:
            places[person] = room


def records(root, suffix):
    return list(csv.DictReader(Path(str(root)+f".{suffix}.tsv").open(), delimiter="\t"))


def check_models(path, seed):
    expected = {
        "please have $0 give the $1 to $2 ": (0, (1-seed%2, 1, 2, 1)),
        "please have $0 go to $1 ": (1, (seed%2, 0, 1, 1)),
        "arrange for $0 to be held by $1 ": (2, (1-seed%2, 0, 1, 1)),
        "ensure $0 is not held by $1 ": (3, (1-seed%2, 0, 1, 0)),
    }
    groups = collections.defaultdict(dict)
    stage = None
    for line in path.read_text().splitlines():
        if line.startswith("checkpoint\t"):
            stage = int(line.split("\t")[1])
        else:
            fields = line.split(":")
            assert fields[0] not in groups[stage]
            groups[stage][fields[0]] = fields[1:]
    assert set(groups) == {0, 1, 2, 3}
    for stage, entries in groups.items():
        wanted = {key for key, (introduced, _) in expected.items() if introduced <= stage}
        for version in range(2 if stage == 3 else 1):
            wanted.update(ports(seed, version))
        assert set(entries) == wanted
        for key, fields in entries.items():
            candidates = {tuple(map(int, v.split(","))) for v in fields[-1].split(";") if v}
            if key in expected:
                assert int(fields[0]) >= 4 and candidates == {expected[key][1]}, (stage, key, fields)
            else:
                version = 0 if key in ports(seed, 0) else 1
                is_transfer = key == ports(seed, version)[0]
                wanted_rule = (1-seed%2, 1, 2, 2) if is_transfer else (seed%2, 0, 1, 2)
                assert int(fields[0]) == (3 if is_transfer else 2) and int(fields[1]) >= 4 and candidates == {wanted_rule}


def main():
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    stats = {}
    for seed in seeds:
        for mode in ("information", "uniform"):
            root = OUT/f"{seed}.{mode}"
            score = collections.Counter()
            reqs, trace = records(root, "requests"), records(root, "trace")
            assert [int(r["request_id"]) for r in reqs] == list(range(160))
            steps = collections.defaultdict(list)
            for row in trace:
                steps[int(row["request_id"])].append(row)
            assert [(int(r["request_id"]), int(r["attempt"])) for r in trace] == [(i, a) for i, r in enumerate(reqs) for a in range(int(r["attempts"]))]
            assert [int(r["token"]) for r in trace] == list(range(1, len(trace)+1))
            people, rooms, objects, owners, places = world("live")
            states = {}
            clock = 0
            for id, req in enumerate(reqs):
                stage, within = divmod(id, 40)
                family, version = within%(stage+1), int(stage == 3)
                assert int(req["stage"]) == stage and int(req["family"]) == family
                expected_request = task(family, id, people, rooms, objects, owners, places)
                assert req["request"] == expected_request and not satisfied(req["request"], owners, places)
                assert req["initial_feedback"] == "1" and req["initial_prediction"] in ("-1", "0")
                score["initial_unknown"] += req["initial_prediction"] == "-1"
                for attempt, row in enumerate(steps[id]):
                    assert int(row["stage"]) == stage and row["request"] == req["request"]
                    assert parse_facts(row["before"]) == expected_facts(owners, places, seed)
                    offered = [tuple(a.split(",")) for a in row["menu"].split(";") if a]
                    assert len(set(offered)) == len(offered) and set(offered) == menu(people, rooms, owners, places, seed, version)
                    action = tuple(row["action"].split(","))
                    assert offered[int(row["index"])] == action
                    executed = row["executed"] == "1"
                    assert executed == (clock%7 != 0)
                    clock += 1
                    physical_owner, physical_place = dict(owners), dict(places)
                    execute(action, executed, physical_owner, physical_place, seed, version)
                    observe(owners, places, row["observation"])
                    assert (owners, places) == (physical_owner, physical_place)
                    assert parse_facts(row["after"]) == expected_facts(owners, places, seed) and row["agree"] == "1"
                    truth = int(satisfied(req["request"], owners, places))
                    assert int(row["satisfied"]) == truth
                    assert int(row["prediction_before_feedback"]) in (-1, truth)
                    assert int(row["prediction_after_feedback"]) in (-1, truth)
                    assert row["accepted"] == "1" and row["status"] == "learned"
                    assert row["policy"] in ("exploit", "explore_actuator", "explore_meaning", "explore_uniform")
                    if mode == "uniform":
                        assert row["policy"] != "explore_meaning"
                    if attempt+1 < len(steps[id]):
                        assert truth == 0  # feedback-completed requests must stop
                    score[row["policy"]] += 1
                    score["refusals"] += not executed
                complete = int(satisfied(req["request"], owners, places))
                assert int(req["complete"]) == complete and int(req["final_prediction"]) in (-1, complete)
                assert complete or int(req["attempts"]) == 32
                score["completed"] += complete
                score["requests"] += 1
                score["transitions"] += int(req["attempts"])
                states[id] = dict(owners), dict(places)
            retained = records(root, "retention")
            expected_qa = {(i, q) for i in range(160) for q in (f"Who has the liveitem{i%3} now?", f"Where is liveperson{i%4}?")}
            seen = set()
            for row in retained:
                id = int(row["request_id"]);key = id, row["question"]
                assert key not in seen
                seen.add(key)
                assert int(row["stage"]) == id//40
                o, p = states[id]
                assert row["reply"] == qa_target(row["question"], o, p)
                assert parse_facts(row["facts"]) == expected_facts(o, p, seed)
            assert seen == expected_qa
            score["retained_qa"] = len(seen)
            held = records(root, "holdout")
            expected_keys = [(checkpoint, version, c) for checkpoint in range(-1, 4) for version in range(2 if checkpoint == 3 else 1) for c in range(48)]
            assert [(int(r["checkpoint"]), int(r["version"]), int(r["case"])) for r in held] == expected_keys
            current = None
            for row in held:
                checkpoint, version, c = (int(row[k]) for k in ("checkpoint", "version", "case"))
                if current != (checkpoint, version):
                    current = checkpoint, version
                    hp, hr, ho, owners, places = world(f"held{checkpoint+1}v{version}")
                family = c%4
                assert int(row["family"]) == family and row["request"] == task(family, c, hp, hr, ho, owners, places)
                assert not satisfied(row["request"], owners, places)
                assert parse_facts(row["before"]) == expected_facts(owners, places, seed)
                offered = [tuple(a.split(",")) for a in row["menu"].split(";") if a]
                assert len(set(offered)) == len(offered) and set(offered) == menu(hp, hr, owners, places, seed, version)
                trained = family <= checkpoint
                if trained:
                    assert row["status"] == "action" and int(row["index"]) >= 0, (seed, mode, row)
                    selected = tuple(row["action"].split(","));assert offered[int(row["index"])] == selected
                    physical_owner, physical_place = dict(owners), dict(places)
                    execute(selected, True, physical_owner, physical_place, seed, version)
                    observe(owners, places, row["observation"])
                    assert (owners, places) == (physical_owner, physical_place)
                    assert row["completed"] == row["truth"] == "1" and satisfied(row["request"], owners, places), (seed, mode, row)
                    score[f"holdout_{checkpoint}_correct"] += 1
                else:
                    assert row["status"] == "unknown_request" and row["index"] == "-1" and row["truth"] == "-1"
                    assert not row["action"] and not row["observation"] and row["completed"] == "0"
                    score["untaught_withheld"] += 1
                assert parse_facts(row["after"]) == expected_facts(owners, places, seed) and row["agree"] == "1"
            check_models(Path(str(root)+".models.tsv"), seed)
            stats[f"{seed}/{mode}"] = score
    paths = list(OUT.glob("*.tsv"))+[ROOT/"cpp/apps/interaction_probe_support.h"]
    paths += [ROOT/name for name in ("cpp/include/laminar/online_grounding.h", "cpp/include/laminar/request_grounding.h", "cpp/include/laminar/reasoning_language.h", "cpp/apps/online_grounding_probe.cpp", "cpp/tests/test_online_grounding.cpp", "python/check_online_grounding.py", "python/check_request_grounding.py", "scripts/online_grounding.sh", "build-wsl/online_grounding_probe", "logs/bootstrap_grounding/train.txt", "logs/incremental_grounding/train.txt", "logs/dialogue_grounding/train.txt")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(seeds=seeds, totals=stats, failures=[], sha256=hashes)
    target = OUT/("audit.json" if seeds == [0, 1, 2] else "partial_audit.json")
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(stats, indent=2))
    print("Verified continuing physical state, every feedback bit/action, frozen checkpoint transfer, exact learned meanings and retained QA.")


if __name__ == "__main__":
    main()
