"""Independent replay of satisfaction feedback, affordances and physical actions.

This checker never calls the C++ grounding/actuator code. Native outputs are
matched to complete expected key sets, not just counted as successful rows.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/request_grounding"
V2 = Path("/home/user/curriculum_v2/stage6")


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def observe(owners, places, text):
    w = words(text)
    if len(w) == 5 and w[1:4] == ["moved", "to", "the"]:
        places[w[0]] = w[4]
    elif len(w) == 4 and w[1:3] == ["has", "the"]:
        owners[w[3]] = w[0]
    else:
        assert w[2] == "the" and w[4] == "to", text
        giver, obj, receiver = w[0], w[3], w[5]
        assert owners[obj] == giver, (text, owners)
        if w[1] == "gives":
            assert w[6:] == [receiver, "takes", "it"], text
            owners[obj] = receiver
        else:
            assert w[1] == "offers" and w[6:] == ["but", receiver, "does", "not", "take", "it"], text


def satisfied(request, owners, places):
    w = words(request)
    if w[:2] == ["make", "sure"]:
        assert w[3:6] == ["is", "holding", "the"] and len(w) == 7
        return owners.get(w[6]) == w[2]
    assert w[:2] == ["please", "have"]
    if w[3:5] == ["go", "to"]:
        assert len(w) == 6
        return places.get(w[2]) == w[5]
    assert w[3:5] == ["give", "the"] and w[6] == "to" and len(w) == 8
    return owners.get(w[5]) == w[7]


def parse_facts(text):
    return {(a, int(r), b) for a, r, b in (s.split(",") for s in text.split(";") if s)}


def expected_facts(owners, places, seed):
    return {(o, 1-seed%2, p) for o, p in owners.items()} | {(p, seed%2, r) for p, r in places.items()}


def arena(prefix):
    people = [prefix+f"person{i}" for i in range(4)]
    rooms = [prefix+f"room{i}" for i in range(4)]
    objects = [prefix+f"item{i}" for i in range(3)]
    return people, rooms, objects, {o: people[i%2] for i, o in enumerate(objects)}, {p: rooms[i%2] for i, p in enumerate(people)}


def ports(seed):
    return ("port91", "port17") if seed%2 else ("port17", "port91")


def actions(people, rooms, owners, places, seed):
    transfer, movement = ports(seed)
    return {(transfer, p, o, t) for o, p in owners.items() for t in people if t != p} | {(movement, p, t) for p, r in places.items() for t in rooms if t != r}


def apply_action(action, executed, owners, places, seed):
    transfer, movement = ports(seed)
    if action[0] == transfer:
        _, giver, obj, receiver = action
        assert owners[obj] == giver and receiver != giver
        if executed:
            owners[obj] = receiver
    else:
        assert action[0] == movement
        _, person, room = action
        assert places[person] != room
        if executed:
            places[person] = room


def qa_target(question, owners, places):
    w = words(question)
    if w[:2] == ["where", "is"]:
        assert len(w) == 3
        return places[w[2]]
    assert w[:3] == ["who", "has", "the"] and w[-1] == "now"
    return owners[w[3]]


def rows(seed, suffix):
    return list(csv.DictReader((OUT/f"{seed}.{suffix}.tsv").open(), delimiter="\t"))


def main():
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    stats = {}
    train = json.loads((OUT/"teacher_train.json").read_text())
    native = iter((OUT/"train.txt").read_text().splitlines())
    for i, record in enumerate(train):
        assert next(native).split("\t") == [str(i), record["phase"], str(len(record["story"])), str(int(record["satisfied"]))]
        assert next(native) == record["request"]
        assert [next(native) for _ in record["story"]] == record["story"]
        owners, places = {}, {}
        for observation in record["story"]:
            observe(owners, places, observation)
        assert satisfied(record["request"], owners, places) == record["satisfied"], i
    assert next(native, None) is None
    canonical = [json.loads(line) for line in (V2/"stage6_grounded_dialogue_test.jsonl").read_text().splitlines()]
    native = iter((OUT/"canonical.txt").read_text().splitlines())
    expected_canonical = {}
    for d in canonical:
        assert next(native).split("\t") == [d["dialogue_id"], str(len(d["turns"]))]
        owners, places, request, question = {}, {}, "", ""
        for i, t in enumerate(d["turns"]):
            assert next(native).split("\t") == [t["role"], t["text"]]
            if t["role"] == "USER":
                request = "" if t["text"].endswith("?") else t["text"]
                question = t["text"] if not request else ""
            if t["role"] == "ENVIRONMENT":
                expected_canonical[d["dialogue_id"], str(i), "before"] = (request, int(satisfied(request, owners, places)), "", dict(owners))
            if t["role"] in ("ENVIRONMENT", "CONTEXT"):
                observe(owners, places, t["text"])
            if t["role"] == "ENVIRONMENT":
                expected_canonical[d["dialogue_id"], str(i), "after"] = (request, int(satisfied(request, owners, places)), "", dict(owners))
            if t["role"] == "ASSISTANT" and question:
                answer = qa_target(question, owners, places)
                assert answer == t["text"].lower()
                expected_canonical[d["dialogue_id"], str(i), "qa"] = ("", -1, answer, dict(owners))
                question = ""
    assert next(native, None) is None
    for seed in seeds:
        score = collections.Counter()
        feedback = rows(seed, "feedback")
        assert [int(r["id"]) for r in feedback] == list(range(len(train)))
        for r, teaching in zip(feedback, train):
            assert r["phase"] == teaching["phase"] and r["accepted"] == "1" and r["status"] == "learned"
            assert int(r["truth"]) in (-1, int(teaching["satisfied"]))
        assert feedback[117]["candidates"] == "2" and feedback[118]["candidates"] == "1"
        score["feedback"] = len(feedback)
        seen = set()
        for row in rows(seed, "canonical"):
            key = row["dialogue"], row["turn"], row["phase"]
            assert key not in seen
            seen.add(key)
            request, truth, reply, owners = expected_canonical[key]
            assert (row["request"], int(row["truth"]), row["reply"]) == (request, truth, reply), (seed, key, row)
            assert parse_facts(row["facts"]) == expected_facts(owners, {}, seed) and row["agree"] == "1"
            score["canonical_"+row["phase"]] += 1
        assert seen == set(expected_canonical)
        exploration = rows(seed, "exploration")
        assert [(int(r["episode"]), int(r["step"])) for r in exploration] == [(e, s) for e in range(16) for s in range(8)]
        for e in range(16):
            people, rooms, _, owners, places = arena(f"explore{e}")
            for row in exploration[e*8:(e+1)*8]:
                step = int(row["step"])
                assert parse_facts(row["before"]) == expected_facts(owners, places, seed)
                action = tuple(row["action"].split(","));executed = row["executed"] == "1"
                assert action in actions(people, rooms, owners, places, seed) and executed == ((e+step)%7 != 0)
                physical_owner, physical_place = dict(owners), dict(places)
                apply_action(action, executed, physical_owner, physical_place, seed)
                observe(owners, places, row["observation"])
                assert (owners, places) == (physical_owner, physical_place)
                assert parse_facts(row["after"]) == expected_facts(owners, places, seed) and row["agree"] == "1"
                score["exploration_success" if executed else "exploration_failure"] += 1
        interactive = rows(seed, "interactive")
        expected_keys = [(e, s, a) for e in range(48) for s in range(12) for a in range(2 if (e*37+s*11)%5 == 0 else 1)]
        assert [(int(r["episode"]), int(r["step"]), int(r["attempt"])) for r in interactive] == expected_keys
        final_states, worlds = {}, {}
        for row in interactive:
            e, step, attempt = (int(row[k]) for k in ("episode", "step", "attempt"))
            if e not in worlds:
                worlds[e] = arena(f"unseen{e}")
            people, rooms, objects, owners, places = worlds[e]
            assert parse_facts(row["before"]) == expected_facts(owners, places, seed)
            menu = [tuple(a.split(",")) for a in row["menu"].split(";") if a]
            assert len(set(menu)) == len(menu) and set(menu) == actions(people, rooms, owners, places, seed)
            selected = tuple(row["action"].split(","))
            assert menu[int(row["index"])] == selected
            assert row["before_truth"] == "0" and not satisfied(row["request"], owners, places)
            executed = row["executed"] == "1"
            assert executed == (attempt != 0 or (e*37+step*11)%5 != 0)
            physical_owner, physical_place = dict(owners), dict(places)
            apply_action(selected, executed, physical_owner, physical_place, seed)
            observe(owners, places, row["observation"])
            assert (owners, places) == (physical_owner, physical_place)
            assert parse_facts(row["after"]) == expected_facts(owners, places, seed) and row["agree"] == "1"
            assert int(row["after_truth"]) == int(satisfied(row["request"], owners, places)) == int(executed)
            if executed:
                final_states[e, step] = dict(owners), dict(places)
            score["interactive_success" if executed else "interactive_failure"] += 1
        assert set(final_states) == {(e, s) for e in range(48) for s in range(12)}
        retained = rows(seed, "retention")
        seen = set()
        for row in retained:
            e, step = int(row["episode"]), int(row["step"])
            key = e, step, row["question"]
            assert key not in seen
            seen.add(key)
            owners, places = final_states[e, step]
            assert row["reply"] == qa_target(row["question"], owners, places)
            assert parse_facts(row["facts"]) == expected_facts(owners, places, seed)
        expected_retention = {(e, s, q) for e in range(48) for s in range(12) for q in (f"Who has the unseen{e}item{s%3} now?", f"Where is unseen{e}person{s%4}?")}
        assert seen == expected_retention
        score["retention"] = len(seen)
        controls = rows(seed, "controls")
        assert [(int(r["episode"]), int(r["step"])) for r in controls] == [(e, s) for e in range(48) for s in range(12)]
        for row in controls:
            assert row["no_requests"] == "unknown_request" and row["no_actuators"] == "no_satisfying_action"
            assert row["blind_truth"] == "0" and row["finished"] == "already_satisfied"
            if int(row["step"])%3 == 0:
                assert row["ambiguous_wrong"] == "-1" and row["resolved_wrong"] == "0"
                score["ambiguity_contrasts"] += 1
            else:
                assert row["ambiguous_wrong"] == row["resolved_wrong"] == "-2"
        score["control_requests"] = len(controls)
        stats[str(seed)] = score
    files = list(OUT.glob("*.tsv"))+[OUT/n for n in ("train.txt", "canonical.txt", "teacher_train.json", "preparation.json")]
    files += [ROOT/n for n in ("cpp/include/laminar/request_grounding.h", "cpp/include/laminar/reasoning_language.h", "cpp/apps/request_grounding_probe.cpp", "cpp/tests/test_request_grounding.cpp", "python/request_grounding_probe.py", "python/check_request_grounding.py", "scripts/request_grounding.sh", "build-wsl/request_grounding_probe", "logs/bootstrap_grounding/train.txt", "logs/incremental_grounding/train.txt", "logs/dialogue_grounding/train.txt")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = dict(seeds=seeds, totals=stats, failures=[], sha256=hashes)
    (OUT/("audit.json" if seeds == [0, 1, 2] else "partial_audit.json")).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(stats, indent=2))
    print("Verified request feedback, all canonical goals, actual actuator outcomes, complete states, retention and controls.")


if __name__ == "__main__":
    main()
