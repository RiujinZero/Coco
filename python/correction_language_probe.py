"""Whole correction requests, taught only through post-correction QA outcomes."""
import copy
import json
from pathlib import Path
import random
from history_reference_probe import make_case, names, initial, questions, history, write_reports

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/correction_language"
TEMPLATES = ["replace {ref} with {report}", "use {report} instead of {ref}",
             "withdraw {ref}", "leave {ref} unchanged", "for {ref} use {report} instead",
             "consider {ref}"]
OPS = [2, 2, 1, 0, 2, 0]


def command(family, reference, report):
    return TEMPLATES[family].format(ref=reference, report=report.rstrip("."))


def outcomes(case):
    records = copy.deepcopy(case["reports"])
    target = next(r for r in records if r["id"] == case["target"])
    if case["operation"]:
        target["active"] = case["operation"] == 2
        if target["active"]:
            target["text"] = case["replacement"]
    return history(case["prefix"], records)[records[-1]["id"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(860214)
    train = []
    for construction in range(6):
        for i in range(12):
            id = len(train)
            prefix = f"edittrain{id}"
            p, rooms, objects = names(prefix)
            obj = objects[i % 3]
            other = objects[(i + 1) % 3]
            before = initial(prefix)[obj, 1]
            actor = before if construction == 5 else p[2]
            texts = [f"{p[1]} has the {other}.", f"{p[2]} moved to the {rooms[0]}.",
                     f"{before} has the {obj}.", f"{actor} has the {obj}.",
                     f"{p[3]} moved to the {rooms[2]}.", f"{p[0]} has the {other}."]
            ids = rng.sample(range(100, 9000), len(texts))
            records = [dict(id=rid, source=prefix + f"source{j % 3}", active=True, text=text)
                       for j, (rid, text) in enumerate(zip(ids, texts))]
            ref = f"the latest active report about {obj}"
            replacement = f"{p[3]} has the {obj}."
            case = dict(id=id, prefix=prefix, construction=construction, operation=OPS[construction],
                        reports=records, reference=ref, replacement=replacement, target=ids[3],
                        utterance=command(construction, ref, replacement))
            state = outcomes(case)
            case["answers"] = [dict(question=q, answer=state[scope]) for q, scope in questions(prefix)]
            train.append(case)
    test = []
    families = (0, 1, 2, 3, 4, 6)

    def add(construction, family, group):
        id = len(test)
        original_group = "missing" if group == "missing" else "resolved"
        case = make_case(rng, id, f"edittest{id}", family, original_group)
        case.update(group=group, construction=construction, operation=OPS[construction])
        p, rooms, objects = names(case["prefix"])
        replacement = f"{p[(id + 1) % 4]} has the {objects[id % 3]}." if id % 2 else f"{p[id % 4]} moved to the {rooms[(id + 1) % 4]}."
        case["replacement"] = replacement
        case["utterance"] = command(construction, case["reference"], replacement)
        if group == "denied": case["source"] = "unrelated_source"
        if group == "unknown": case["utterance"] = "rethink " + case["reference"]
        if group == "unknown_payload":
            case["utterance"] = command(construction, case["reference"], f"{p[0]} mysteriously teleports near {rooms[0]}")
        case["choice"] = case["target"] if group in ("ambiguous", "stale") else -1
        test.append(case)

    for construction in range(5):
        for family in families:
            for _ in range(4): add(construction, family, "resolved")
    for i in range(24): add((0, 1, 2, 4)[i % 4], 5, "ambiguous")
    for group in ("stale", "missing", "denied", "generated", "unknown", "weak_meaning", "unknown_payload", "no_reference_model", "no_command_model"):
        for i in range(12):
            construction = 5 if group == "weak_meaning" else (0, 1, 4)[i % 3] if group == "unknown_payload" else (0, 1, 2, 4)[i % 4]
            add(construction, 5 if group == "stale" else families[i % 6], group)
    (OUT / "gold.json").write_text(json.dumps(dict(train=train, test=test), indent=2) + "\n")
    with (OUT / "train.txt").open("w") as f:
        f.write(f"{len(train)}\n")
        for c in train:
            f.write(f"{c['id']}\t{c['prefix']}\t{len(c['reports'])}\t{len(c['answers'])}\n")
            write_reports(f, c)
            f.write(c["utterance"] + "\n")
            for qa in c["answers"]: f.write(qa["question"] + "\t" + qa["answer"] + "\n")
    with (OUT / "test.txt").open("w") as f:
        f.write(f"{len(test)}\n")
        for c in test:
            f.write(f"{c['id']}\t{c['prefix']}\t{len(c['reports'])}\t{c['group']}\n")
            write_reports(f, c)
            f.write(c["source"] + "\t" + c["utterance"] + "\n")
            f.write(f"{c['choice']}\n{c['stale_text']}\n")
    print(f"Prepared {len(train)} correction histories, {sum(len(c['answers']) for c in train)} outcome answers and {len(test)} frozen requests.", flush=True)


if __name__ == "__main__":
    main()
