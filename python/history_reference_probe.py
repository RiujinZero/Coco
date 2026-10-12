"""Historical QA teaches record references; target IDs stay outside teaching input."""
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/history_reference"
PHRASES = [
    "the earliest active report about {a}",
    "the latest active report about {a}",
    "the previous active report about {a}",
    "the earliest active report from {a}",
    "the latest active report from {a}",
    "the active report about {a}",
    "the latest active report from {a} about {b}",
]


def names(prefix):
    return ([prefix + f"person{i}" for i in range(4)],
            [prefix + f"room{i}" for i in range(4)],
            [prefix + f"item{i}" for i in range(3)])


def initial(prefix):
    p, r, o = names(prefix)
    return {(a, 0): b for a, b in zip(p, r)} | {(obj, 1): p[i % 2] for i, obj in enumerate(o)}


def apply(rows, text):
    move = re.fullmatch(r"(\w+) moved to the (\w+)\.", text)
    has = re.fullmatch(r"(\w+) has the (\w+)\.", text)
    assert move or has
    if move: rows[move[1], 0] = move[2]
    else: rows[has[2], 1] = has[1]


def history(prefix, reports):
    rows = initial(prefix)
    out = {}
    for record in reports:
        if record["active"]:
            apply(rows, record["text"])
        out[record["id"]] = dict(rows)
    return out


def questions(prefix):
    p, _, o = names(prefix)
    return [(f"Who has the {obj} now?", (obj, 1)) for obj in o] + [(f"Where is {a}?", (a, 0)) for a in p]


def mention(record, entity):
    return entity in re.findall(r"[a-z0-9]+", record["text"])


def eligible(record, family, a, b):
    if family in (0, 1, 2, 5): return mention(record, a)
    if family in (3, 4): return record["source"] == a
    return record["source"] == a and mention(record, b)


def make_case(rng, id, prefix, family, group="resolved"):
    p, rooms, objects = names(prefix)
    sources = [prefix + f"source{i}" for i in range(3)]
    # Test field identity rather than treating source-looking names as a type.
    if prefix.startswith("test") and id % 4 == 0:
        sources[0] = p[0]
    entity = objects[id % 3] if id % 2 == 0 else p[id % 4]
    a = entity if family in (0, 1, 2, 5) else sources[0]
    b = entity
    def draw(target=False):
        source = sources[0] if target and family in (3, 4, 6) else rng.choice(sources)
        if target and family in (0, 1, 2, 5, 6):
            if entity in objects: text = f"{rng.choice(p)} has the {entity}."
            elif rng.randrange(2): text = f"{entity} has the {rng.choice(objects)}."
            else: text = f"{entity} moved to the {rng.choice(rooms)}."
        elif rng.randrange(2): text = f"{rng.choice(p)} has the {rng.choice(objects)}."
        else: text = f"{rng.choice(p)} moved to the {rng.choice(rooms)}."
        return dict(source=source, text=text, active=True)
    records = [draw() for _ in range(12)] + [draw(True) for _ in range(6)]
    rng.shuffle(records)
    matching = [r for r in records if eligible(r, family, a, b)]
    assert len(matching) >= 6
    # Vary inactivity independently of chronological rank. Always retiring the
    # first/last match would confound first-active with unfiltered second (and
    # last-active with unfiltered penultimate) even under perfect QA accuracy.
    leading, trailing = id % 3, (id // 3) % 3
    for r in matching[:leading]: r["active"] = False
    if trailing:
        for r in matching[-trailing:]: r["active"] = False
    if family == 5 and group == "train":
        for r in matching: r["active"] = False
        matching[len(matching) // 2]["active"] = True
    if group == "missing":
        for r in matching: r["active"] = False
    ids = rng.sample(range(100, 9000), len(records))
    for record, key in zip(records, ids): record["id"] = key
    candidates = [r for r in records if r["active"] and eligible(r, family, a, b)]
    index = 0 if family in (0, 3, 5) else -2 if family == 2 else -1
    selected = candidates[index] if candidates else None
    reference = PHRASES[family].format(a=a, b=b)
    if group == "unknown": reference = "the obscure forgotten note concerning " + entity
    receiver = p[(id + 2) % 4]
    replacement = f"{receiver} has the {objects[id % 3]}." if id % 2 == 0 else f"{p[id % 4]} moved to the {rooms[(id + 2) % 4]}."
    source = selected["source"] if selected else sources[0]
    if group == "denied": source = "unrelated_source"
    active = id % 3 != 0
    if not active: replacement = ""
    return dict(id=id, prefix=prefix, family=family, group=group, reports=records,
                reference=reference, target=selected["id"] if selected else None,
                expected_candidates=[r["id"] for r in candidates], replacement=replacement,
                source=source, active=active,
                stale_text=f"{p[0]} moved to the {rooms[3]}.")


def write_reports(handle, case):
    for r in case["reports"]:
        handle.write(f"{r['id']}\t{r['source']}\t{int(r['active'])}\t{r['text']}\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(671932)
    train = []
    for family in range(7):
        for i in range(16):
            id = len(train)
            case = make_case(rng, id, f"train{id}", family, "train")
            states = history(case["prefix"], case["reports"])
            case["lessons"] = [dict(reference=case["reference"], question=q, answer=states[case["target"]][scope]) for q, scope in questions(case["prefix"])]
            train.append(case)
    test = []
    for family in (0, 1, 2, 3, 4, 6):
        for _ in range(12):
            id = len(test)
            test.append(make_case(rng, id, f"test{id}", family))
    for group in ("ambiguous", "stale", "missing", "unknown", "denied", "generated"):
        for i in range(12):
            id = len(test)
            family = 5 if group in ("ambiguous", "stale") else (0, 1, 2, 3, 4, 6)[i % 6]
            test.append(make_case(rng, id, f"test{id}", family, group))
    (OUT / "gold.json").write_text(json.dumps(dict(train=train, test=test), indent=2) + "\n")
    with (OUT / "train.txt").open("w") as handle:
        handle.write(f"{len(train)}\n")
        for case in train:
            handle.write(f"{case['id']}\t{case['prefix']}\t{len(case['reports'])}\t{len(case['lessons'])}\n")
            write_reports(handle, case)
            for e in case["lessons"]: handle.write(f"{e['reference']}\t{e['question']}\t{e['answer']}\n")
    with (OUT / "test.txt").open("w") as handle:
        handle.write(f"{len(test)}\n")
        for case in test:
            handle.write(f"{case['id']}\t{case['prefix']}\t{len(case['reports'])}\t{case['group']}\n")
            write_reports(handle, case)
            handle.write(f"{case['reference']}\t{case['source']}\t{int(case['active'])}\t{case['replacement']}\n")
            # This is a later explicit clarification message, consumed only
            # after the initial resolution/output. Definite tests have no ID.
            choice = case["target"] if case["group"] in ("ambiguous", "stale") else -1
            handle.write(f"{choice}\n{case['stale_text']}\n")
    print(f"Prepared {len(train)} histories, {sum(len(c['lessons']) for c in train)} answer lessons and {len(test)} frozen cases.")


if __name__ == "__main__":
    main()
