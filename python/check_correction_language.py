"""Independent semantics and state replay for whole correction requests."""
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from check_observation_language import parse_knowledge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/correction_language"
PATTERNS = ["replace @reference with @report ", "use @report instead of @reference ",
            "withdraw @reference ", "leave @reference unchanged ",
            "for @reference use @report instead ", "consider @reference "]
MEANINGS = [{2}, {2}, {1}, {0}, {2}, {0, 1}]


def rows(path):
    with path.open() as f: return list(csv.DictReader(f, delimiter="\t"))


def vocabulary(prefix):
    return ([prefix + "person" + str(i) for i in range(4)],
            [prefix + "room" + str(i) for i in range(4)],
            [prefix + "item" + str(i) for i in range(3)])


def replay(prefix, records):
    people, rooms, objects = vocabulary(prefix)
    state = {(people[i], 0): rooms[i] for i in range(4)}
    state.update({(o, 1): people[i % 2] for i, o in enumerate(objects)})
    states = {}
    for record in records:
        if record["active"]:
            move = re.fullmatch(r"(\w+) moved to the (\w+)\.", record["text"])
            owner = re.fullmatch(r"(\w+) has the (\w+)\.", record["text"])
            assert move or owner, record
            if move: state[move[1], 0] = move[2]
            else: state[owner[2], 1] = owner[1]
        states[str(record["id"])] = dict(state)
    states["current"] = dict(state)
    return states


def query(text, state):
    location = re.fullmatch(r"Where is (\w+)\?", text)
    owner = re.fullmatch(r"Who has the (\w+) now\?", text)
    assert owner or location
    return state[(owner or location)[1], int(bool(owner))]


def reference(case):
    about = re.fullmatch(r"the (earliest|latest|previous) active report about (\w+)", case["reference"])
    source = re.fullmatch(r"the (earliest|latest) active report from (\w+)(?: about (\w+))?", case["reference"])
    weak = re.fullmatch(r"the active report about (\w+)", case["reference"])
    assert about or source or weak
    matching = []
    for r in case["reports"]:
        words = set(re.findall(r"[a-z0-9]+", r["text"]))
        if not r["active"]: continue
        if about and about[2] not in words: continue
        if source and (r["source"] != source[2] or source[3] and source[3] not in words): continue
        if weak and weak[1] not in words: continue
        matching.append(r["id"])
    if not matching: return set()
    if weak: return set(matching)
    rank = (about or source)[1]
    index = {"earliest": 0, "latest": -1, "previous": -2}[rank]
    return {matching[index]} if len(matching) >= abs(index) else set()


def revised(case, operation, target=None):
    result = copy.deepcopy(case["reports"])
    if operation:
        selected = target if target is not None else case["target"]
        record = next(r for r in result if r["id"] == selected)
        record["active"] = operation == 2
        if operation == 2: record["text"] = case["replacement"]
    return result


def check_input(gold):
    for mode in ("train", "test"):
        lines = iter((OUT / (mode + ".txt")).read_text().splitlines())
        assert int(next(lines)) == len(gold[mode])
        for case in gold[mode]:
            expected = [str(case["id"]), case["prefix"], str(len(case["reports"])), str(len(case["answers"])) if mode == "train" else case["group"]]
            assert next(lines).split("\t") == expected
            for r in case["reports"]:
                assert next(lines).split("\t") == [str(r["id"]), r["source"], str(int(r["active"])), r["text"]]
            if mode == "train":
                assert next(lines) == case["utterance"]
                selected = reference(case)
                assert selected == {case["target"]}
                expected_state = replay(case["prefix"], revised(case, case["operation"]))["current"]
                for qa in case["answers"]:
                    assert next(lines).split("\t") == [qa["question"], qa["answer"]]
                    assert query(qa["question"], expected_state) == qa["answer"]
            else:
                assert next(lines).split("\t") == [case["source"], case["utterance"]]
                # Only a later clarification turn carries a target ID.
                assert int(next(lines)) == case["choice"]
                assert next(lines) == case["stale_text"]
                assert (case["choice"] >= 0) == (case["group"] in ("ambiguous", "stale"))
        assert next(lines, None) is None
    train_names = {n for c in gold["train"] for group in vocabulary(c["prefix"]) for n in group}
    test_names = {n for c in gold["test"] for group in vocabulary(c["prefix"]) for n in group}
    assert train_names.isdisjoint(test_names)
    assert {c["reference"].split(" report")[0] for c in gold["train"]} == {"the latest active"}
    assert all(" has the " in c["replacement"] for c in gold["train"])
    assert any(" moved to the " in c["replacement"] and c["operation"] == 2 for c in gold["test"])


def knowledge(state):
    return dict(positive={(a, r, b) for (a, r), b in state.items()}, negative=set(), rows=set(state), columns=set())


def audit(seed, gold):
    prefix = OUT / str(seed)
    def file(suffix): return Path(str(prefix) + suffix)
    teaching = rows(file(".teaching.tsv"))
    assert len(teaching) == len(gold["train"])
    candidates, support = {}, collections.defaultdict(set)
    for observed, case in zip(teaching, gold["train"]):
        key = PATTERNS[case["construction"]]
        assert observed["scene"] == str(case["id"]) and observed["utterance"] == case["utterance"] and observed["frame"] == key
        possible = {0, 1, 2} if "@report" in key else {0, 1}
        agrees = set()
        for op in possible:
            state = replay(case["prefix"], revised(case, op))["current"]
            if all(query(qa["question"], state) == qa["answer"] for qa in case["answers"]): agrees.add(op)
        candidates[key] = candidates.get(key, possible) & agrees
        support[key].add(case["id"])
        assert set(map(int, filter(None, observed["operations"].split(",")))) == candidates[key]
        assert int(observed["support"]) == len(support[key])
    model = {}
    for line in file(".model.txt").read_text().splitlines():
        key, n, ops = line.split(":")
        model[key] = set(map(int, filter(None, ops.split(","))))
        assert int(n) == 12
    assert model == dict(zip(PATTERNS, MEANINGS)) == candidates
    native_cases = rows(file(".cases.tsv"))
    state_rows, record_rows, answer_rows = (rows(file(s)) for s in (".states.tsv", ".records.tsv", ".answers.tsv"))
    state_index = {(int(r["case"]), r["phase"], r["scope"]): r for r in state_rows}
    record_index = {(int(r["case"]), r["phase"], int(r["id"])): r for r in record_rows}
    answer_index = collections.defaultdict(list)
    for r in answer_rows: answer_index[int(r["case"])].append(r)
    assert len(state_index) == len(state_rows) and len(record_index) == len(record_rows)
    assert len(native_cases) == len(gold["test"])
    score = collections.Counter(teaching_answers=sum(len(c["answers"]) for c in gold["train"]), frames=len(model))
    for case, observed in zip(gold["test"], native_cases):
        id, group, op = case["id"], case["group"], case["operation"]
        assert observed["case"] == str(id) and observed["group"] == group
        score[group] += 1
        uninterpreted = group in ("unknown", "unknown_payload", "no_reference_model", "no_command_model", "weak_meaning")
        if uninterpreted:
            assert int(observed["operation"]) == -1 and not observed["reference"] and not observed["payload"] and not observed["candidates"]
            meaning = "ambiguous_correction_meaning" if group == "weak_meaning" else "unknown_correction"
            assert observed["meaning"] == meaning and observed["status"] == meaning
            assert observed["initial_reply"] == "I cannot interpret that correction."
        else:
            assert observed["meaning"] == "grounded_correction" and int(observed["operation"]) == op
            assert observed["reference"] == case["reference"]
            assert observed["payload"] == (case["replacement"] if op == 2 else "")
            selected = reference(case)
            assert set(map(int, filter(None, observed["candidates"].split(",")))) == selected
            expected_resolution = "missing_reference" if not selected else "resolved_reference" if len(selected) == 1 else "ambiguous_reference"
            assert observed["resolution"] == expected_resolution
            if group in ("ambiguous", "stale"):
                assert case["choice"] in selected and int(observed["invalid_preserved"]) == 1
                assert observed["initial_reply"] == "Which report? Candidates:" + "".join(f" {r}." for r in sorted(selected))
                assert observed["final_reply"] == ("The report has been revised." if group == "ambiguous" else "That correction needs a fresh reference.")
                assert observed["status"] == ("revised_report" if group == "ambiguous" else "stale_correction")
            else:
                statuses = {"missing": ("missing_reference", "I cannot identify that report."),
                            "denied": ("stale_report_revision", "The report could not be revised."),
                            "generated": ("self_generated_correction", "")}
                status, reply = statuses.get(group, ("kept_report", "The report is unchanged.") if op == 0 else ("revised_report", "The report has been revised."))
                assert observed["status"] == status and observed["initial_reply"] == reply
        applied = group in ("resolved", "ambiguous") and op != 0
        if group == "resolved":
            score["held_out_reference_combinations"] += case["family"] != 1
            score["held_out_payload_constructions"] += op == 2 and " moved to the " in case["replacement"]
        assert bool(int(observed["applied"])) == applied
        target = case["choice"] if group == "ambiguous" else case["target"]
        assert int(observed["selected"]) == (target if applied else -1)
        if group not in ("ambiguous", "stale"):
            assert not observed["final_reply"] and int(observed["invalid_preserved"]) == 0
        initial_revision = sum(not r["active"] for r in case["reports"])
        assert int(observed["history_before"]) == initial_revision
        assert int(observed["history_after"]) == initial_revision + applied
        assert observed["episode_before"] == observed["episode_after"]
        expected_records = revised(case, op, target) if applied else copy.deepcopy(case["reports"])
        if group == "stale": expected_records.append(dict(id=99999, source="late_sensor", active=True, text=case["stale_text"]))
        original = {r["id"]: r for r in case["reports"]}
        for phase, records in (("before", case["reports"]), ("after", expected_records)):
            for record in records:
                native = record_index.pop((id, phase, record["id"]))
                initial = original.get(record["id"])
                revision = int(initial is not None and not initial["active"]) + int(phase == "after" and applied and record["id"] == target)
                assert int(native["revision"]) == revision and bool(int(native["active"])) == record["active"]
                assert native["source"] == record["source"] and native["text"] == record["text"]
            for scope, state in replay(case["prefix"], records).items():
                native = state_index.pop((id, phase, scope))
                assert parse_knowledge(native["knowledge"], seed) == knowledge(state), (seed, id, phase, scope)
                score["audited_states"] += 1
        final = replay(case["prefix"], expected_records)["current"]
        answers = answer_index.pop(id)
        assert len(answers) == 7 and len({r["question"] for r in answers}) == 7
        expected_questions = {f"Who has the {o} now?" for o in vocabulary(case["prefix"])[2]} | {f"Where is {p}?" for p in vocabulary(case["prefix"])[0]}
        assert {r["question"] for r in answers} == expected_questions
        for answer in answers:
            assert answer["reply"] == query(answer["question"], final)
            score["current_answers"] += 1
        score["applied"] += applied
    assert not state_index and not record_index and not answer_index
    assert score["applied"] == 120 and score["audited_states"] == 9588 and score["current_answers"] == 1764
    assert score["held_out_reference_combinations"] == 100 and score["held_out_payload_constructions"] == 36
    print(f"seed={seed}: {dict(score)}", flush=True)
    return dict(score)


def main():
    gold = json.loads((OUT / "gold.json").read_text())
    check_input(gold)
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    results = {s: audit(s, gold) for s in seeds}
    paths = [Path(__file__), ROOT / "python/correction_language_probe.py", ROOT / "python/history_reference_probe.py",
             ROOT / "python/check_observation_language.py", ROOT / "cpp/apps/correction_language_probe.cpp",
             ROOT / "cpp/apps/interaction_probe_support.h", ROOT / "cpp/apps/grounding_probe_support.h",
             ROOT / "cpp/tests/test_correction_language_grounding.cpp", ROOT / "scripts/correction_language.sh",
             ROOT / "cpp/src/organism.cpp", ROOT / "build-wsl/correction_language_probe", ROOT / "build-asan/test_correction_language_grounding"]
    paths += list((ROOT / "cpp/include/laminar").glob("*.h"))
    paths += [ROOT / f"logs/{name}/train.txt" for name in ("bootstrap_grounding", "incremental_grounding", "dialogue_grounding", "observation_language", "history_reference")]
    paths += [ROOT / f"logs/reasoning_composition_verified/micro_{s}.txt" for s in seeds]
    paths += [p for p in OUT.iterdir() if p.name != "audit.json" and p.suffix in (".txt", ".tsv", ".json")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT / "audit.json").write_text(json.dumps(dict(results=results, sha256=hashes), indent=2) + "\n")


if __name__ == "__main__":
    main()
