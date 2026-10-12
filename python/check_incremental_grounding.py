"""Independent visible-state replay and native output audit; no learner input."""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/incremental_grounding"


def replay(story):
    location, owner = {}, {}
    for sentence in story:
        s = sentence.lower()
        m = re.fullmatch(r"(\w+) (?:moved to|went to|went back to|journeyed to|travelled to) the (\w+)\.", s)
        if m:
            location[m[1]] = m[2]
            continue
        m = re.fullmatch(r"(\w+) has the (\w+)\.", s)
        if m:
            owner[m[2]] = m[1]
            continue
        m = re.fullmatch(r"(\w+) (?:gave|handed|passed) the (\w+) to (\w+)\.", s)
        assert m and owner[m[2]] == m[1], sentence
        owner[m[2]] = m[3]
    return location, owner


def answer(question, location, owner):
    q = question.lower().rstrip("?")
    def term(t):
        if t.startswith("the location of "):
            return location[term(t[len("the location of "):])]
        if t.startswith("the number of items held by "):
            entity = term(t[len("the number of items held by "):])
            return sum(x == entity for x in owner.values())
        assert re.fullmatch(r"\w+", t), t
        return t
    m = re.fullmatch(r"who has the (\w+) now", q)
    if m:
        return owner[m[1]]
    m = re.fullmatch(r"where is (\w+)", q)
    if m:
        return location[m[1]]
    for prefix in ("what is ", "name "):
        if q.startswith(prefix):
            return str(term(q[len(prefix):]))
    m = re.fullmatch(r"is (.+) equal to (.+)", q)
    if m:
        return "yes" if term(m[1]) == term(m[2]) else "no"
    m = re.fullmatch(r"does (\w+) have the (\w+)", q)
    if m:
        return "yes" if owner.get(m[2]) == m[1] else "no"
    m = re.fullmatch(r"is the (\w+) held by (\w+)", q)
    assert m, q
    return "yes" if owner.get(m[1]) == m[2] else "no"


def native_records(path, targets):
    lines = iter(path.read_text().splitlines())
    rows = []
    for header in lines:
        i, n = map(int, header.split("\t"))
        assert i == len(rows)
        r = dict(story=[next(lines) for _ in range(n)], question=next(lines))
        if targets:
            r["answer"] = next(lines)
        rows.append(r)
    return rows


def main():
    train = json.loads((OUT / "teacher_train.json").read_text())
    gold = json.loads((OUT / "gold.json").read_text())
    for rows, name, targets in ((train, "train.txt", True), (gold, "test.txt", False)):
        exported = native_records(OUT / name, targets)
        assert exported == [{k: r[k] for k in ("story", "question", "answer") if targets or k != "answer"} for r in rows]
        for r in rows:
            loc, own = replay(r["story"])
            assert answer(r["question"], loc, own) == r["answer"].lower(), r
            if "locations" in r:
                assert loc == r["locations"] and own == r["owners"]
    # Entire composed question forms are withheld, including across renamings.
    def form(r):
        loc, own = replay(r["story"])
        entities = set(loc) | set(loc.values()) | set(own) | set(own.values())
        return re.sub(r"\w+", lambda m: "$" if m[0] in entities else m[0], r["question"].lower())
    training_forms = {form(r) for r in train}
    assert all(form(r) not in training_forms for r in gold if r["group"].startswith("composition_"))
    groups = {r["group"] for r in gold}
    for group in groups:
        labels = {r["answer"] for r in gold if r["group"] == group}
        if group.startswith("composition_equal") or group == "composition_nested":
            assert labels == {"yes", "no"}, (group, labels)
        if group == "composition_count_at_location":
            assert labels == {"0", "1", "2"}, labels
    stats = collections.defaultdict(collections.Counter)
    hashes = {}
    # Verify that selected external examples really occur in the source splits.
    v2 = Path("/home/user/curriculum_v2/stage5")
    source_train = set()
    for episode in (v2 / "stage5_events_train.txt").read_text().split("<|endofdialogue|>"):
        if not episode.strip():
            continue
        story, rest = episode.split("USER: ", 1)
        question, target = rest.split("ASSISTANT: ", 1)
        source_train.add((tuple(story.strip().splitlines()), question.strip(), target.strip()))
    selected_train = {(tuple(r["story"]), r["question"], r["answer"]) for r in train if r["group"] == "v2_training"}
    assert selected_train <= source_train
    source_test = set()
    for line in (v2 / "stage5_events_test.jsonl").read_text().splitlines():
        for question, target, _, story in json.loads(line)["qa"]:
            source_test.add((tuple(story), question, {"true": "yes", "false": "no"}.get(target, target.lower())))
    selected_test = {(tuple(r["story"]), r["question"], r["answer"]) for r in gold if r["group"] == "v2_transfer"}
    assert selected_test <= source_test
    overlap = len(selected_train & selected_test)
    assert overlap == 0, ("selected v2 train/test QA overlap", overlap)
    for seed in range(3):
        location, possession = seed % 2, 1 - seed % 2
        model = list(csv.reader((OUT / f"{seed}.model.tsv").open(), delimiter="\t"))
        assert model[0] == ["status", "learned", "accepted", str(len(train)), "rejected", "0", "checked", str(len(train)), "skipped", "0", "rounds", "2"]
        event = {r[1]: r[2] for r in model if r[0] == "event"}
        assert len(event) == 9
        assert event["$0 has the $1 "] == f"{possession},1,0,2"
        for verb in ("gave", "handed", "passed"):
            assert event[f"$0 {verb} the $1 to $2 "] == f"{possession},1,2,2"
        seen = set()
        for r in csv.DictReader((OUT / f"{seed}.trace.tsv").open(), delimiter="\t"):
            i = int(r["id"])
            key = (r["mode"], i)
            assert key not in seen
            seen.add(key)
            g = gold[i]
            expected = {(a, location, b) for a, b in g["locations"].items()} | {(o, possession, p) for o, p in g["owners"].items()}
            facts = {tuple(f.split(",")) for f in r["facts"].split(";") if f}
            facts = {(a, int(rel), b) for a, rel, b in facts}
            score = stats[r["mode"] + "/" + g["group"]]
            score["n"] += 1
            score["exact"] += r["reply"] == g["answer"] and r["status"] in ("answer", "grounded_answer")
            score["state"] += facts == expected
            if r["mode"] == "no_shared_meanings" and g["group"].startswith("composition_"):
                assert r["status"] not in ("answer", "grounded_answer"), (seed, i, r)
            else:
                assert r["reply"] == g["answer"] and r["status"] in ("answer", "grounded_answer"), (seed, i, g, r)
            assert facts == expected, (seed, i, expected, facts)
        assert len(seen) == 2 * len(gold)
        for name in (f"{seed}.trace.tsv", f"{seed}.model.tsv"):
            p = OUT / name
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("train.txt", "test.txt", "teacher_train.json", "gold.json"):
        p = OUT / name
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("cpp/include/laminar/incremental_grounding.h", "cpp/include/laminar/procedure_memory.h", "cpp/include/laminar/semantic_grounding.h", "cpp/include/laminar/event_grounding.h", "cpp/apps/incremental_grounding_probe.cpp", "cpp/tests/test_incremental_grounding.cpp", "python/check_incremental_grounding.py", "python/incremental_grounding_probe.py", "build-wsl/incremental_grounding_probe"):
        p = ROOT / name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "audit.json").write_text(json.dumps(dict(totals=stats, selected_v2_qa_overlap=overlap, sha256=hashes), indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print("Verified mixed QA induction, withheld compositions, original v2 subset and bAbI retention.")


if __name__ == "__main__":
    main()
