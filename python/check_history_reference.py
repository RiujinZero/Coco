"""Independent record-reference hypotheses, temporal selection and edit audit."""
import collections
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
from check_observation_language import parse_knowledge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/history_reference"


def read(path):
    return list(csv.DictReader(path.open(), delimiter="\t"))


def vocabulary(prefix):
    p = [prefix + f"person{i}" for i in range(4)]
    r = [prefix + f"room{i}" for i in range(4)]
    o = [prefix + f"item{i}" for i in range(3)]
    return p, r, o


def interpret(prefix, reports):
    p, r, o = vocabulary(prefix)
    state = {(a, 0): b for a, b in zip(p, r)} | {(obj, 1): p[i % 2] for i, obj in enumerate(o)}
    history = {}
    for row in reports:
        if row["active"]:
            moved = re.fullmatch(r"(\w+) moved to the (\w+)\.", row["text"])
            held = re.fullmatch(r"(\w+) has the (\w+)\.", row["text"])
            assert moved or held
            if moved: state[moved[1], 0] = moved[2]
            else: state[held[2], 1] = held[1]
        history[row["id"]] = dict(state)
    return history, state


def answer(question, state):
    held = re.fullmatch(r"Who has the (\w+) now\?", question)
    moved = re.fullmatch(r"Where is (\w+)\?", question)
    assert held or moved
    return state[(held or moved)[1], int(bool(held))]


def knowledge(state):
    return dict(positive={(a, r, b) for (a, r), b in state.items()}, negative=set(), rows=set(state), columns=set())


def descriptors(case):
    names = set(sum((list(v) for v in vocabulary(case["prefix"])), []))
    return [(r["id"], bool(r["active"]), {0: set(re.findall(r"[a-z0-9]+", r["text"])) & names, 1: {r["source"]}}) for r in case["reports"]]


def bind(case, reference):
    words = re.findall(r"[a-z0-9]+", reference)
    values = set(sum((list(v) for v in vocabulary(case["prefix"])), [])) | {r["source"] for r in case["reports"]}
    args, pattern = [], []
    for w in words:
        if w in values:
            pattern.append(f"${len(args)}")
            args.append(w)
        else: pattern.append(w)
    return " ".join(pattern) + " ", args


def hypotheses(arity):
    return {(fields, active, rank) for fields in itertools.product((-1, 0, 1), repeat=arity) for active in (-1, 0, 1) for rank in (-99, -2, -1, 0, 1)}


def select(records, rule, args):
    fields, active, rank = rule
    selected = [id for id, state, values in records if (active < 0 or state == bool(active)) and all(f < 0 or args[i] in values[f] for i, f in enumerate(fields))]
    if rank == -99:
        return set(selected)
    position = rank if rank >= 0 else len(selected) + rank
    return {selected[position]} if 0 <= position < len(selected) else set()


def parse_rules(text, seed):
    rules = set()
    for raw in text.split(";"):
        if raw:
            f, active, rank = raw.split("/")
            fields = tuple(int(v) ^ (seed % 2) if int(v) >= 0 else -1 for v in f.split(",") if v)
            rules.add((fields, int(active), int(rank)))
    return rules


EXPECTED = {
    "the earliest active report about $0 ": {((0,), 1, 0)},
    "the latest active report about $0 ": {((0,), 1, -1)},
    "the previous active report about $0 ": {((0,), 1, -2)},
    "the earliest active report from $0 ": {((1,), 1, 0)},
    "the latest active report from $0 ": {((1,), 1, -1)},
    "the latest active report from $0 about $1 ": {((1, 0), 1, -1)},
    "the active report about $0 ": {((0,), 1, -99), ((0,), 1, -1), ((0,), 1, 0)},
}


def input_integrity(gold):
    for mode in ("train", "test"):
        lines = iter((OUT / f"{mode}.txt").read_text().splitlines())
        assert int(next(lines)) == len(gold[mode])
        for case in gold[mode]:
            h = next(lines).split("\t")
            assert h[:3] == [str(case["id"]), case["prefix"], str(len(case["reports"]))]
            assert h[3] == (str(len(case["lessons"])) if mode == "train" else case["group"])
            for r in case["reports"]:
                assert next(lines) == f"{r['id']}\t{r['source']}\t{int(r['active'])}\t{r['text']}"
            if mode == "train":
                history, _ = interpret(case["prefix"], case["reports"])
                frame, args = bind(case, case["reference"])
                targets = set().union(*(select(descriptors(case), rule, args) for rule in EXPECTED[frame]))
                assert targets == {case["target"]}
                for lesson in case["lessons"]:
                    assert answer(lesson["question"], history[case["target"]]) == lesson["answer"]
                    assert next(lines) == f"{lesson['reference']}\t{lesson['question']}\t{lesson['answer']}"
            else:
                assert next(lines) == f"{case['reference']}\t{case['source']}\t{int(case['active'])}\t{case['replacement']}"
                choice = int(next(lines))
                assert choice == (case["target"] if case["group"] in ("ambiguous", "stale") else -1)
                assert next(lines) == case["stale_text"]
        assert next(lines, None) is None
    train_names = {n for c in gold["train"] for v in vocabulary(c["prefix"]) for n in v}
    test_names = {n for c in gold["test"] for v in vocabulary(c["prefix"]) for n in v}
    assert not train_names & test_names


def audit(seed, gold):
    score = collections.Counter()
    teaching = iter(read(OUT / f"{seed}.teaching.tsv"))
    model, support = {}, collections.defaultdict(set)
    for case in gold["train"]:
        history, _ = interpret(case["prefix"], case["reports"])
        for id, lesson in enumerate(case["lessons"]):
            row = next(teaching)
            frame, args = bind(case, lesson["reference"])
            candidates = hypotheses(len(args))
            keep = set()
            for rule in candidates:
                selected = select(descriptors(case), rule, args)
                if selected and all(answer(lesson["question"], history[r]) == lesson["answer"] for r in selected):
                    keep.add(rule)
            model[frame] = model.get(frame, candidates) & keep
            support[frame].add(case["id"])
            assert row == dict(scene=str(case["id"]), evidence=str(id), reference=lesson["reference"], question=lesson["question"], answer=lesson["answer"], frame=frame, support=str(len(support[frame])), rules=row["rules"])
            assert parse_rules(row["rules"], seed) == model[frame] and model[frame], (seed, case["id"], id)
            score["teaching_answers"] += 1
    assert next(teaching, None) is None
    exported = {}
    for line in (OUT / f"{seed}.model.txt").read_text().splitlines():
        frame, n, rules = line.split(":")
        assert frame not in exported and int(n) == len(support[frame]) == 16
        exported[frame] = parse_rules(rules, seed)
    assert exported == model
    assert model == EXPECTED, (seed, "selectors remain observationally confounded", model)
    cases = read(OUT / f"{seed}.cases.tsv")
    assert [int(c["case"]) for c in cases] == list(range(len(gold["test"])))
    outputs = collections.defaultdict(list)
    for row in read(OUT / f"{seed}.answers.tsv"):
        outputs[int(row["case"])].append(row)
    states, record_versions = {}, collections.defaultdict(list)
    for row in read(OUT / f"{seed}.states.tsv"):
        key = int(row["case"]), row["phase"], row["scope"]
        assert key not in states
        states[key] = parse_knowledge(row["knowledge"], seed)
    for row in read(OUT / f"{seed}.records.tsv"):
        record_versions[int(row["case"]), row["phase"]].append(row)
    seen_states = set()
    for case, result in zip(gold["test"], cases):
        id, group = case["id"], case["group"]
        frame, args = bind(case, case["reference"])
        targets, missing = set(), False
        for rule in model.get(frame, set()):
            selected = select(descriptors(case), rule, args)
            missing |= not selected
            targets |= selected
        resolved = not missing and len(targets) == 1
        resolution = "unknown_reference" if frame not in model else "resolved_reference" if resolved else "missing_reference" if not targets else "ambiguous_reference"
        assert result["resolution"] == resolution and {int(v) for v in result["candidates"].split(",") if v} == targets
        assert result["group"] == group
        if group in ("resolved", "denied", "generated"):
            assert resolved and targets == {case["target"]}
        elif group in ("ambiguous", "stale"):
            assert not resolved and targets == set(case["expected_candidates"]) and len(targets) > 1
        else: assert not targets
        applied = group in ("resolved", "ambiguous")
        assert bool(int(result["applied"])) == applied and int(result["selected"]) == (case["target"] if applied else -1)
        expected_status = {"resolved": "revised_report", "ambiguous": "revised_report", "stale": "stale_correction", "missing": "missing_reference", "unknown": "unknown_reference", "denied": "stale_report_revision", "generated": "self_generated_correction"}[group]
        assert result["status"] == expected_status
        assert result["episode_before"] == result["episode_after"]
        revisions = sum(not r["active"] for r in case["reports"])
        assert int(result["history_before"]) == revisions and int(result["history_after"]) == revisions + int(applied)
        if group in ("ambiguous", "stale"):
            assert result["initial_reply"] == "Which report? Candidates:" + "".join(f" {n}." for n in sorted(targets))
            assert result["invalid_preserved"] == "1"
            assert result["final_reply"] == ("The report has been revised." if applied else "That correction needs a fresh reference.")
        else:
            wanted = "The report has been revised." if applied else "The report could not be revised." if group == "denied" else "" if group == "generated" else "I cannot identify that report."
            assert result["initial_reply"] == wanted and not result["final_reply"] and result["invalid_preserved"] == "0"
        revised = copy.deepcopy(case["reports"])
        for r in revised: r["revision"] = int(not r["active"])
        before_history, before = interpret(case["prefix"], revised)
        history_answers = before_history[next(iter(targets))] if resolved else None
        def check_state(phase, reports):
            histories, current = interpret(case["prefix"], reports)
            for scope, state in {"current": current, **{str(i): k for i, k in histories.items()}}.items():
                key = id, phase, scope
                assert states[key] == knowledge(state), (seed, id, phase, scope)
                seen_states.add(key)
            expected_records = [dict(case=str(id), phase=phase, id=str(r["id"]), revision=str(r["revision"]), active=str(int(r["active"])), source=r["source"], text=r["text"]) for r in reports]
            assert record_versions[id, phase] == expected_records
            return current
        check_state("before", revised)
        if applied:
            record = next(r for r in revised if r["id"] == case["target"])
            assert record["source"] == case["source"]
            record["active"] = bool(case["active"])
            if case["active"]: record["text"] = case["replacement"]
            record["revision"] += 1
        elif group == "stale":
            revised.append(dict(id=99999, source="late_sensor", text=case["stale_text"], active=True, revision=0))
        after = check_state("after", revised)
        p, _, o = vocabulary(case["prefix"])
        questions = [f"Who has the {obj} now?" for obj in o] + [f"Where is {a}?" for a in p]
        expected_qa = [("reference", q) for q in questions] if resolved and group != "generated" else []
        expected_qa += [("current", q) for q in questions]
        assert [(r["scope"], r["question"]) for r in outputs[id]] == expected_qa
        for row in outputs[id]:
            assert row["reply"] == answer(row["question"], history_answers if row["scope"] == "reference" else after)
            assert row["status"] in ("answer", "grounded_answer")
            score["historical_answers" if row["scope"] == "reference" else "current_answers"] += 1
        score[group] += 1
        score["applied"] += applied
    assert seen_states == set(states)
    score.update(frames=len(model), audited_states=len(states))
    print(f"seed={seed}: {dict(score)}", flush=True)
    return dict(score)


def main():
    gold = json.loads((OUT / "gold.json").read_text())
    input_integrity(gold)
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    results = {seed: audit(seed, gold) for seed in seeds}
    paths = [Path(__file__), ROOT / "python/history_reference_probe.py", ROOT / "python/check_observation_language.py",
             ROOT / "cpp/apps/history_reference_probe.cpp", ROOT / "cpp/apps/interaction_probe_support.h", ROOT / "cpp/apps/grounding_probe_support.h",
             ROOT / "cpp/tests/test_history_reference_grounding.cpp", ROOT / "scripts/history_reference.sh",
             ROOT / "cpp/src/organism.cpp", ROOT / "build-wsl/history_reference_probe", ROOT / "build-asan/test_history_reference_grounding"]
    paths += list((ROOT / "cpp/include/laminar").glob("*.h"))
    paths += [ROOT / f"logs/{name}/train.txt" for name in ("bootstrap_grounding", "incremental_grounding", "dialogue_grounding", "observation_language")]
    paths += [ROOT / f"logs/reasoning_composition_verified/micro_{s}.txt" for s in seeds]
    paths += [p for p in OUT.iterdir() if p.name != "audit.json" and p.suffix in (".txt", ".tsv", ".json")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT / "audit.json").write_text(json.dumps(dict(results=results, sha256=hashes), indent=2) + "\n")


if __name__ == "__main__":
    main()
