"""Independent replay of raw evidence revisions, language hypotheses and history."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

from check_online_grounding import world
from check_live_evidence import apply, hypotheses, answer, public_knowledge, question_menu, freeze, parse_rules
from check_observation_language import parse_knowledge
from revision_grounding_probe import curriculum

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/revision_grounding"
TRANSFER = "the $0 was logged beside $1 "
GAP = "the location record for $0 went dark "


def records(path):
    return list(csv.DictReader(path.open(), delimiter="\t"))


def initial(prefix):
    people, rooms, objects, owners, places = world(prefix)
    rows = {(a, 0): b for a, b in places.items()} | {(o, 1): a for o, a in owners.items()}
    names = [n for pair in zip(people, rooms) for n in pair] + objects
    bindings = tuple(sorted({"$0 moved to the $1 ": (0, 0, 1, 2), "$0 has the $1 ": (1, 1, 0, 2)}.items()))
    return names, public_knowledge(rows), bindings


def parse_bindings(text, seed):
    out = {}
    for raw in text.split(";"):
        if not raw:
            continue
        key, value = raw.split("#")
        rule = list(map(int, value.split(",")))
        relation = 1 if key.startswith("observation:") else 0
        if rule[relation] >= 0:
            rule[relation] ^= seed % 2
        assert key not in out
        out[key] = tuple(rule)
    return tuple(sorted(out.items()))


def known(text):
    move = re.fullmatch(r"(\w+) moved to the (\w+)\.", text.lower())
    has = re.fullmatch(r"(\w+) has the (\w+)\.", text.lower())
    if move:
        return (9, 0, 0, 2), move.groups(), "$0 moved to the $1 ", (0, 0, 1, 2)
    if has:
        return (9, 1, 0, 2), (has[2], has[1]), "$0 has the $1 ", (1, 1, 0, 2)
    return None


def replay(entries):
    names, k, meanings = initial("revision")
    queries = dict(question_menu(names))
    worlds = {(k, meanings)}
    models, support, history = {}, collections.defaultdict(set), {}
    for id, entry in entries.items():
        if entry["active"]:
            parsed = known(entry["text"])
            if parsed:
                rule, args, key, choice = parsed
                options = [(rule, args, key, choice)]
            else:
                words = re.findall(r"[a-z0-9]+", entry["text"].lower())
                args = [w for w in words if w in names]
                slots = iter(range(len(args)))
                frame = " ".join(f"${next(slots)}" if w in names else w for w in words) + " "
                assert frame in (TRANSFER, GAP)
                models.setdefault(frame, hypotheses(len(args)))
                if any(e["active"] for e in entry["evidence"].values()):
                    support[frame].add(id)
                options = [(r, args, "observation:" + frame, r) for r in models[frame]]
            next_worlds = set()
            for old, bindings in worlds:
                bindings = dict(bindings)
                for rule, args, key, choice in options:
                    if key in bindings and bindings[key] != choice:
                        continue
                    new = apply(old, rule, args)
                    if new is not None:
                        next_worlds.add((new, tuple(sorted((bindings | {key: choice}).items()))))
            if not next_worlds:
                return None, {}, {}, "conflicting_report_history"
            worlds = next_worlds
            for _, e in sorted(entry["evidence"].items()):
                if e["active"]:
                    worlds = {(k, m) for k, m in worlds if answer(k, queries[e["question"]]) == e["answer"]}
                    if not worlds:
                        return None, {}, {}, "conflicting_journal_evidence"
        history[id] = worlds
    for id, before in history.items():
        history[id] = {(k, m) for k, m in before if any(set(m) <= set(final_m) for _, final_m in worlds)}
    for frame in models:
        models[frame] = ({dict(m)["observation:" + frame] for _, m in worlds}, len(support[frame]))
    return worlds, history, models, "replayed_evidence"


def semantic_state(prefix, entries, stop=None):
    _, k, _ = initial(prefix)
    for id, entry in entries.items():
        if entry["active"]:
            parsed = known(entry["text"])
            if parsed:
                rule, args, _, _ = parsed
            else:
                transfer = re.fullmatch(r"The (\w+) was logged beside (\w+)\.", entry["text"])
                gap = re.fullmatch(r"The location record for (\w+) went dark\.", entry["text"])
                assert transfer or gap
                rule, args = ((9, 1, 0, 2), transfer.groups()) if transfer else ((5, 0, 0, 0), gap.groups())
            k = apply(k, rule, args)
            assert k is not None
        if stop == id:
            break
    return k


def audit(seed, operations):
    score = collections.Counter()
    steps = records(OUT / f"{seed}.steps.tsv")
    assert [int(r["step"]) for r in steps] == list(range(len(operations)))
    exported_worlds, exported_models, outputs = collections.defaultdict(set), collections.defaultdict(dict), collections.defaultdict(list)
    for row in records(OUT / f"{seed}.worlds.tsv"):
        key = int(row["step"]), row["scope"]
        value = freeze(parse_knowledge(row["knowledge"], seed)), parse_bindings(row["bindings"], seed)
        assert value not in exported_worlds[key]
        exported_worlds[key].add(value)
    for row in records(OUT / f"{seed}.models.tsv"):
        step = int(row["step"])
        assert row["frame"] not in exported_models[step]
        exported_models[step][row["frame"]] = parse_rules(row["rules"], seed), int(row["support"])
    for row in records(OUT / f"{seed}.answers.tsv"):
        outputs[int(row["step"])].append(row)
    names, _, _ = initial("revision")
    options = question_menu(names)
    queries = dict(options)
    entries, published = {}, {}
    revisions, changes, seen_world_keys = 0, [], set()
    for step, (op, native) in enumerate(zip(operations, steps)):
        id, evidence = op["report"], op["evidence"]
        kind = op["op"]
        if kind == "report":
            assert id not in entries
            entries[id] = dict(text=op["text"], source=op["source"], active=True, revision=0, evidence={})
            change_kind, revision, payload = "report", 0, op["text"]
        elif kind == "feedback":
            assert evidence not in entries[id]["evidence"]
            entries[id]["evidence"][evidence] = dict(question=op["question"], answer=op["answer"], source=op["source"], active=True, revision=0)
            revisions += id != next(reversed(entries))
            change_kind, revision, payload = "feedback", 0, op["answer"]
        else:
            old = entries[id] if kind == "revise_report" else entries[id]["evidence"][evidence]
            assert old["revision"] == op["revision"] and old["source"] == op["source"]
            old["revision"] += 1
            old["active"] = bool(op["active"])
            revision = old["revision"]
            field = "text" if kind == "revise_report" else "answer"
            old[field] = payload = op[field]
            change_kind = kind if op["active"] else "withdraw_" + kind.removeprefix("revise_")
            revisions += 1
        changes.append(dict(kind=change_kind, report=str(id), evidence=str(evidence), revision=str(revision), source=op["source"], payload=payload))
        worlds, history, model, status = replay(entries)
        consistent = worlds is not None
        assert int(native["consistent"]) == consistent and native["status"] == status, (seed, step, status, native)
        assert int(native["history_revision"]) == revisions and native["op"] == kind
        assert int(native["reports"]) == len(entries)
        assert native["episode_revision"] == steps[0]["episode_revision"]
        if consistent:
            published = model
            assert int(native["worlds"]) == len(worlds)
        assert exported_models[step] == published, (seed, step, "model")
        scopes = {"current": worlds} | {str(id): ws for id, ws in history.items()}
        expected_keys = [("current", q) for q, _ in options]
        if consistent:
            expected_keys += [(str(id), q) for id in entries for q in ("Who has the revisionitem0 now?", "Where is revisionperson2?")]
        assert [(r["scope"], r["question"]) for r in outputs[step]] == expected_keys
        for scope, ws in scopes.items():
            key = step, scope
            if consistent:
                assert exported_worlds[key] == ws, (seed, step, scope, "retained worlds")
                seen_world_keys.add(key)
                score["audited_world_states"] += len(ws)
            else:
                assert key not in exported_worlds
        for row in outputs[step]:
            if not consistent:
                assert row["reply"] == "I do not know." and row["status"] == status
                score["conflict_abstentions"] += 1
                continue
            values = {answer(k, queries[row["question"]]) for k, _ in scopes[row["scope"]]}
            expected = next(iter(values)) if len(values) == 1 and "unknown" not in values else "I do not know."
            assert row["reply"] == expected, (seed, step, row, values)
            assert row["status"] in ({"answer", "grounded_answer"} if expected != "I do not know." else {"missing_observation", "nonunique_result", "uncertain_answer"})
            score["definite_answers" if expected != "I do not know." else "uncertain_answers"] += 1
        score["conflict_states"] += not consistent
    assert seen_world_keys == set(exported_worlds)
    assert records(OUT / f"{seed}.changes.tsv") == changes
    assert published == {TRANSFER: ({(9, 1, 0, 2)}, 4), GAP: ({(5, 0, 0, 0)}, 4)}
    assert {k for k, _ in worlds} == {semantic_state("revision", entries)}
    # Crucial recovery checks: a wrong initial answer is asserted, a later
    # contradiction blocks use, then revision repairs current and earlier views.
    assert steps[4]["consistent"] == "0" and steps[5]["consistent"] == "1"
    def response(step, scope, q):
        return next(r["reply"] for r in outputs[step] if r["scope"] == scope and r["question"] == q)
    assert response(1, "current", "Who has the revisionitem0 now?") == "revisionperson0"
    assert response(5, "current", "Who has the revisionitem0 now?") == "revisionperson2"
    assert response(5, "0", "Who has the revisionitem0 now?") == "revisionperson2"
    assert response(5, "current", "Where is revisionperson2?") == "revisionroom0"
    assert response(5, "0", "Where is revisionperson2?") == "revisionroom2"
    frozen = records(OUT / f"{seed}.frozen.tsv")
    expected_keys = []
    for c in range(48):
        prefix = f"revheld{c}"
        p, r, o, _, _ = world(prefix)
        obj, actor = o[c % 3], p[c % 4]
        transfer = lambda delta: f"The {obj} was logged beside {p[(c+delta)%4]}."
        gap = f"The location record for {actor} went dark."
        reports = {}
        for phase in range(8):
            if phase in (0, 1, 3):
                reports[phase] = dict(text=transfer(1) if phase == 0 else gap if phase == 1 else f"{p[(c+3)%4]} has the {obj}.", active=True)
            elif phase == 2: reports[0]["text"] = transfer(2)
            elif phase == 4: reports[0]["text"] = transfer(0)
            elif phase == 5: reports[1]["active"] = False
            elif phase == 6: reports[3]["active"] = False
            else: reports[1]["active"] = True
            for scope in ("current", "0"):
                k = semantic_state(prefix, reports, None if scope == "current" else 0)
                for q, query in ((f"Who has the {obj} now?", (obj, 1, None)), (f"Where is {actor}?", (actor, 0, None))):
                    row = frozen[len(expected_keys)]
                    expected_keys.append((str(c), str(phase), scope, q))
                    assert freeze(parse_knowledge(row["knowledge"], seed)) == k
                    gold = answer(k, query)
                    assert row["reply"] == ("I do not know." if gold == "unknown" else gold)
                    assert row["status"] in ({"missing_observation", "uncertain_answer"} if gold == "unknown" else {"answer", "grounded_answer"})
                    score["frozen_unknown" if gold == "unknown" else "frozen_definite"] += 1
    assert [(r["case"], r["phase"], r["scope"], r["question"]) for r in frozen] == expected_keys
    score.update(operations=len(operations), evidence_revisions=revisions, frozen_questions=len(frozen))
    print(f"seed={seed}: {dict(score)}", flush=True)
    return dict(score)


def main():
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    operations = curriculum()
    assert records(OUT / "train.tsv") == [{k: str(v) for k, v in op.items()} for op in operations]
    results = {seed: audit(seed, operations) for seed in seeds}
    paths = [Path(__file__), ROOT / "python/revision_grounding_probe.py", ROOT / "python/check_live_evidence.py",
             ROOT / "python/check_observation_language.py", ROOT / "python/check_online_grounding.py",
             ROOT / "scripts/revision_grounding.sh", ROOT / "cpp/apps/revision_grounding_probe.cpp",
             ROOT / "cpp/apps/interaction_probe_support.h", ROOT / "cpp/apps/grounding_probe_support.h",
             ROOT / "cpp/tests/test_revisable_grounding.cpp", ROOT / "build-wsl/revision_grounding_probe",
             ROOT / "build-asan/test_revisable_grounding", OUT / "train.tsv"]
    paths += list((ROOT / "cpp/include/laminar").glob("*.h")) + [ROOT / "cpp/src/organism.cpp"]
    paths += [ROOT / f"logs/{name}/train.txt" for name in ("bootstrap_grounding", "incremental_grounding", "dialogue_grounding", "observation_language")]
    paths += [ROOT / f"logs/reasoning_composition_verified/micro_{s}.txt" for s in seeds]
    paths += [p for s in seeds for p in OUT.glob(f"{s}.*.tsv")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT / "audit.json").write_text(json.dumps(dict(results=results, sha256=hashes), indent=2) + "\n")


if __name__ == "__main__":
    main()
