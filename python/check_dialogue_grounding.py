"""Audit full native dialogue states and predictions before target exposure."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/dialogue_grounding"
V2 = Path("/home/user/curriculum_v2/stage6")


def observe(world, text, locations=None):
    words = re.findall(r"\w+", text.lower())
    if len(words) == 5 and words[1:4] == ["moved", "to", "the"]:
        assert locations is not None
        locations[words[0]] = words[4]
        return
    if len(words) == 4 and words[1:3] == ["has", "the"]:
        world[words[3]] = words[0]
        return
    assert words[1] in ("gives", "offers") and words[2] == "the" and words[4] == "to", text
    giver, obj, receiver = words[0], words[3], words[5]
    assert world[obj] == giver and words[6 if words[1] == "gives" else 7] == receiver, text
    if words[1] == "gives":
        assert words[7:] == ["takes", "it"], text
        world[obj] = receiver
    else:
        assert words[6:] == ["but", receiver, "does", "not", "take", "it"], text


def target(world, question, locations=None):
    q = question.lower().rstrip("?")
    m = re.fullmatch(r"where is (\w+)", q)
    if m:
        return locations[m[1]]
    m = re.fullmatch(r"what is the number of destinations for (\w+)", q)
    if m:
        return "1" if m[1] in locations else "0"
    m = re.fullmatch(r"who has the (\w+) now", q)
    if m:
        return world[m[1]]
    m = re.fullmatch(r"what is the number of items held by (\w+)", q)
    assert m, q
    return str(sum(p == m[1] for p in world.values()))


def main():
    gold = json.loads((OUT / "gold.json").read_text())
    assert gold == [json.loads(line) for line in (V2 / "stage6_grounded_dialogue_test.jsonl").read_text().splitlines()]
    train = json.loads((OUT / "teacher_train.json").read_text())
    for row in train:
        world, locations = {}, {}
        for text in row["story"]:
            observe(world, text, locations)
        assert target(world, row["question"], locations) == row["answer"].lower(), row
    # Match the complete native input, including every original turn and role.
    lines = iter((OUT / "test.txt").read_text().splitlines())
    expected = {}
    expected_keys = set()
    reveals = []
    expected_source_questions = 0
    for d in gold:
        objects = sorted(re.fullmatch(r"\w+ has the (\w+)\.", t["text"])[1].lower() for t in d["turns"] if t["role"] == "CONTEXT")
        assert next(lines).split("\t") == [d["dialogue_id"], str(len(d["turns"])), str(len(objects))]
        assert [next(lines) for _ in objects] == [f"Who has the {o} now?" for o in objects]
        world, initial, prefix = {}, {}, []
        for i, t in enumerate(d["turns"]):
            role, text = t["role"], t["text"]
            assert next(lines).split("\t") == [role, text]
            before = dict(world)
            if role in ("CONTEXT", "ENVIRONMENT"):
                observe(world, text)
                if role == "CONTEXT":
                    initial = dict(world)
            if role == "ENVIRONMENT":
                reveals.append((list(prefix), text))
            original = role == "ASSISTANT" and i > 0 and d["turns"][i-1]["role"] == "USER" and d["turns"][i-1]["text"].endswith("?")
            if original:
                assert target(world, d["turns"][i-1]["text"]) == text.lower()
                expected_source_questions += 1
            expected[(d["dialogue_id"], i)] = dict(before=before, state=dict(world), after=dict(world), initial=dict(initial), role=role, original=original)
            for mode in ("rendered", "canonical", "poisoned_claims", "no_environment"):
                expected_keys.add((mode, d["dialogue_id"], i, "state", ""))
                if role == "ENVIRONMENT":
                    for phase in ("before", "after"):
                        for obj in objects:
                            expected_keys.add((mode, d["dialogue_id"], i, phase, f"Who has the {obj} now?"))
                if original:
                    expected_keys.add((mode, d["dialogue_id"], i, "original", d["turns"][i-1]["text"][:-1]))
            prefix.append(text if role == "CONTEXT" else role + ": " + text)
    assert next(lines, None) is None
    export = []
    for line in (V2 / "stage6_grounded_dialogue_before_reveal_test.jsonl").read_text().splitlines():
        q, outcome, _, prefix = json.loads(line)["qa"][0]
        assert q == "What happens?"
        export.append((prefix, outcome))
    assert reveals == export
    stats = collections.defaultdict(collections.Counter)
    failures = []
    hashes = {}
    for seed in range(3):
        seen = set()
        for row in csv.DictReader((OUT / f"{seed}.trace.tsv").open(), delimiter="\t"):
            mode, phase, question = row["mode"], row["phase"], row["question"]
            key = (mode, row["dialogue"], int(row["turn"]), phase, question)
            assert key not in seen
            seen.add(key)
            e = expected[(row["dialogue"], int(row["turn"]))]
            world = e["initial"] if mode == "no_environment" else e["state" if phase == "original" else phase]
            wanted = {(o, 1-seed%2, p) for o, p in world.items()}
            actual = {tuple(f.split(",")) for f in row["facts"].split(";") if f}
            actual = {(a, int(r), b) for a, r, b in actual}
            score = stats[f"{mode}/{phase}"]
            score["n"] += 1
            score["state"] += actual == wanted
            score["all_worlds_agree"] += row["worlds_agree"] == "1"
            score["max_worlds"] = max(score["max_worlds"], int(row["worlds"]))
            ok = actual == wanted and row["worlds_agree"] == "1"
            if phase == "state":
                ok &= row["status"] == "complete"
            else:
                answer = target(world, question)
                answered = row["status"] in ("answer", "grounded_answer")
                exact = answered and row["reply"] == answer
                score["answered"] += answered
                score["exact_observed"] += exact
                score["exact_actual"] += answered and row["reply"] == target(e["before" if phase == "before" else "state"], question)
                ok &= exact
            if not ok and len(failures) < 12:
                failures.append(dict(seed=seed, row=row, expected=sorted(wanted)))
        assert seen == expected_keys, (seed, len(seen), len(expected_keys), list(expected_keys-seen)[:3])
        for suffix in ("trace.tsv", "model.tsv"):
            p = OUT / f"{seed}.{suffix}"
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("cpp/include/laminar/reasoning_language.h", "cpp/apps/dialogue_grounding_probe.cpp", "cpp/tests/test_dialogue_grounding.cpp", "python/dialogue_grounding_probe.py", "python/check_dialogue_grounding.py", "scripts/dialogue_grounding.sh", "build-wsl/dialogue_grounding_probe", "logs/bootstrap_grounding/train.txt", "logs/incremental_grounding/train.txt"):
        hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in ("preparation.json", "train.txt", "test.txt", "teacher_train.json", "gold.json"):
        p = OUT/name
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(dialogues=len(gold), original_questions=expected_source_questions, before_reveal_cases=len(reveals), totals=stats, failures=failures, sha256=hashes)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(totals=stats, failures=failures[:2]), indent=2))
    assert not failures, "Native dialogue validation failed; see audit.json"
    print("Verified all canonical roles, complete states, before/after possession probes and original answers.")


if __name__ == "__main__":
    main()
