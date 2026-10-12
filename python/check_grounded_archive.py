"""Independent episode replay, receipt revision and projected-state audit."""
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from check_observation_language import answer, parse_knowledge
from check_grounded_teaching import empty, observe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/grounded_archive"
OWNER = "the $0 was logged beside $1 "
GAP = "the location record for $0 went dark "


def read(path):
    with path.open() as f: return list(csv.DictReader(f, delimiter="\t"))


def replay(episodes, ownership):
    states, histories, valid = {}, {}, True
    def apply(state, text):
        if ownership == "ignore" and re.fullmatch(r"the \w+ was logged beside \w+\.", text.lower()): return
        observe(state, text)
    for id, episode in episodes.items():
        state = empty()
        for text in episode["context"]: apply(state, text)
        for n, record in enumerate(episode["reports"], 1):
            if record["active"]:
                apply(state, record["text"])
                for e in record["evidence"].values():
                    if e["active"]: valid &= answer(e["question"], state) == e["answer"]
            histories[f"{id}:{n}"] = copy.deepcopy(state)
        states[id] = state
    return valid, states, histories


def expectations(turns):
    receipts = collections.defaultdict(list)
    for line in (OUT / "feedback.tsv").read_text().splitlines():
        f = line.split("\t");assert len(f) == 10
        receipts[int(f[0])].append(f)
    episodes, expected, count = {}, [], 0
    for turn in turns:
        id = turn["episode"];request = None;route = None
        for line in turn["text"].splitlines():
            if not line: continue
            if line == "<|endofdialogue|>": episodes[id] = dict(context=[], reports=[])
            elif line.startswith("CONTEXT: "): episodes[id]["context"].append(line[9:])
            elif line.startswith("ENVIRONMENT: "):
                episodes[id]["reports"].append(dict(text=line[13:], active=True, revision=0, evidence={}))
            elif line.startswith("USER: "):
                request = line[6:];route = "question"
                replace = re.fullmatch(r"replace the earliest active report about (\w+) with (.+)", request)
                withdraw = re.fullmatch(r"withdraw the latest active report about (\w+)", request)
                if replace or withdraw:
                    entity = (replace or withdraw)[1]
                    matches = [r for r in episodes[id]["reports"] if r["active"] and entity in re.findall(r"\w+", r["text"])]
                    assert matches;record = matches[0] if replace else matches[-1]
                    if replace: record["text"] = replace[2] + "."
                    else: record["active"] = False
                    record["revision"] += 1;route = "correction"
            else: assert line == "ASSISTANT:", line
        for f in receipts.pop(turn["offset"], []):
            _, ep, report, report_revision, evidence, previous, operation, source, q, a = f
            record = episodes[int(ep)]["reports"][int(report)-1];key = int(evidence)
            assert record["revision"] == int(report_revision) and record["active"]
            if operation == "add":
                assert key not in record["evidence"] and previous == "0"
                record["evidence"][key] = dict(question=q, answer=a, revision=0, active=True, source=source)
            else:
                e = record["evidence"][key]
                assert e["question"] == q and e["source"] == source and e["revision"] == int(previous)
                e["revision"] += 1;e["active"] = operation == "revise"
                if e["active"]: e["answer"] = a
            count += 1
        # Two independent semantic witnesses determine projected states here:
        # the initially taught no-change interpretation and the intended update.
        # The native model may retain other candidates before repair; its full
        # projected knowledge must agree with these exhaustive public QA contrasts.
        survivors = [r for mode in ("ignore", "update") if (r := replay(episodes, mode))[0]]
        consistent = bool(survivors)
        def intersect(values): return {key: set.intersection(*(v[key] for v in values)) for key in empty()}
        state = intersect([r[1][id] for r in survivors]) if consistent else None
        histories = {scope: intersect([r[2][scope] for r in survivors]) for scope in survivors[0][2]} if consistent else {}
        reply = ""
        if request:
            if route == "correction": reply = "The report has been revised." if consistent else "The remaining evidence conflicts."
            elif not consistent: reply = "I do not know."
            else:
                outcomes = {answer(request, r[1][id]) for r in survivors}
                result = next(iter(outcomes)) if len(outcomes) == 1 else "unknown"
                reply = "I do not know." if result == "unknown" else result
        support = collections.Counter();evidence_rows = {}
        for ep, episode in episodes.items():
            for n, record in enumerate(episode["reports"], 1):
                if record["active"] and any(e["active"] for e in record["evidence"].values()):
                    frame = OWNER if "was logged beside" in record["text"].lower() else GAP if "went dark" in record["text"].lower() else None
                    if frame: support[frame] += 1
                for key, e in record["evidence"].items(): evidence_rows[(ep, n, key)] = dict(e)
        expected.append(dict(reply=reply, route=route, consistent=consistent, state=state, histories=histories, evidence=evidence_rows, support=dict(support), feedback=count, restored=consistent and survivors[0][1][1]["positive"] != replay({1: episodes[1]}, "ignore")[1][1]["positive"]))
    assert not receipts and count == 313
    return expected


def audit(seed, turns, expected):
    prefix = OUT / str(seed)
    trace = read(Path(str(prefix) + ".turns.tsv"));assert len(trace) == len(turns)
    states = read(Path(str(prefix) + ".states.tsv"));models = read(Path(str(prefix) + ".models.tsv"));evidence = read(Path(str(prefix) + ".evidence.tsv"))
    state_index = {(int(r["turn"]), r["scope"]): r for r in states}
    model_index = collections.defaultdict(dict)
    for row in models:
        t = int(row["turn"]);assert row["frame"] not in model_index[t];model_index[t][row["frame"]] = row
    evidence_index = {(int(r["turn"]), int(r["episode"]), int(r["report"]), int(r["id"])): r for r in evidence}
    assert len(state_index) == len(states) and len(evidence_index) == len(evidence)
    score = collections.Counter()
    for t, (turn, gold, row) in enumerate(zip(turns, expected, trace)):
        assert (int(row["turn"]), int(row["offset"])) == (t, turn["offset"])
        assert row["reply"] == gold["reply"], (seed, t, row["reply"], gold["reply"])
        assert bool(int(row["consistent"])) == gold["consistent"], (seed, t, row["status"])
        assert int(row["feedback"]) == gold["feedback"]
        if gold["route"]: assert row["route"] == gold["route"]
        score["turns"] += 1;score["replies"] += bool(gold["reply"]);score["corrections"] += gold["route"] == "correction"
        score["withheld_conflict_replies"] += bool(gold["reply"]) and not gold["consistent"]
        score["transfer_replies"] += bool(gold["reply"]) and turn["episode"] >= 9
        current = state_index.pop((t, "current"))
        if not gold["consistent"]: assert current["knowledge"] == "unavailable"
        else:
            assert parse_knowledge(current["knowledge"], seed) == gold["state"], (seed, t, "current")
            score["audited_states"] += 1
        for scope, state in gold["histories"].items():
            row = state_index.pop((t, scope));assert parse_knowledge(row["knowledge"], seed) == state, (seed, t, scope)
            score["audited_states"] += 1
        for key, wanted in gold["evidence"].items():
            row = evidence_index.pop((t, *key))
            assert int(row["revision"]) == wanted["revision"] and bool(int(row["active"])) == wanted["active"]
            assert all(row[field] == wanted[field] for field in ("question", "answer", "source"))
            score["audited_receipt_versions"] += 1
        if gold["consistent"]:
            for frame, support in gold["support"].items():
                row = model_index[t][frame];assert int(row["support"]) == support
                rules = {tuple(map(int, rule.split(","))) for rule in row["rules"].rstrip(";").split(";")}
                rules = {(kind, rel ^ (seed % 2) if rel >= 0 else rel, focus, mask) for kind, rel, focus, mask in rules}
                if frame == OWNER:
                    if gold["restored"]: assert rules == {(9, 1, 0, 2)}, (seed, t, rules)
                    else: assert (0, -1, -1, 0) in rules and (9, 1, 0, 2) not in rules
                elif frame == GAP: assert rules == {(5, 0, 0, 0)}, (seed, t, rules)
                score["audited_frame_states"] += 1
    assert not state_index and not evidence_index
    print(f"seed={seed}: {dict(score)}", flush=True);return dict(score)


def main():
    turns = json.loads((OUT / "turns.json").read_text());expected = expectations(turns)
    assert (OUT / "trainer.txt").read_bytes() == "".join(t["text"] for t in turns).encode()
    with (OUT / "turns.inputs").open("rb") as f:
        assert int(f.readline()) == len(turns)
        for turn in turns: assert f.read(int(f.readline())).decode() == turn["text"] and f.read(1) == b"\n"
        assert not f.read()
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    results = {seed: audit(seed, turns, expected) for seed in seeds}
    logs = [(OUT / f"trainer_{name}.log").read_text() for name in ("probe", "plain")]
    models = [re.findall(r"GROUNDED observation_model BEGIN\n(.*?)GROUNDED observation_model END", log, re.S) for log in logs]
    assert models[0] == models[1] == [(OUT / "0.model.txt").read_text()]
    states = [re.findall(r"GROUNDED state (.+)", log) for log in logs]
    assert len(states[0]) == 1 and states[0] == states[1]
    names, _, projected, *_ = states[0][0].split("|")
    names = names.rstrip(",").split(",")
    final = empty()
    for marker, key in (("p", "positive"), ("n", "negative")):
        final[key] = {(names[int(a)], int(r), names[int(b)]) for a, r, b in re.findall(marker + r"(\d+),(\d+),(\d+);", projected)}
    final["rows"] = {(names[int(a)], int(r)) for a, r in re.findall(r"r(\d+),(\d+);", projected)}
    final["columns"] = {(int(r), names[int(b)]) for r, b in re.findall(r"c(\d+),(\d+);", projected)}
    assert final == expected[-1]["state"]
    for pattern in (r"GROUNDED stream (.+)", r"GROUNDED archive (.+)"):
        records = [re.findall(pattern, log) for log in logs]
        assert len(records[0]) == 1 and records[0] == records[1]
    for log in logs:
        assert "GROUNDED teaching accepted=313 rejected=0 duplicates=0 delivered=313 pending=0" in log
        assert re.search(r"GROUNDED archive episodes=12 revision=\d+ consistent=1 status=replayed_archive", log)
    checks = re.findall(r"session_probe (\d+)/(\d+)", logs[0])
    assert len(checks) >= 3 and checks[-1] == ("1", "1"), checks
    assert all(b == "1" and a in ("0", "1") for a, b in checks)
    probe_status = re.findall(r"GROUNDED probe [^\n]*status=(\S+)", logs[0])
    assert len(probe_status) == len(checks)
    assert all(status == ("answer" if a == "1" else "unavailable_evidence_history") for (a, _), status in zip(checks, probe_status))
    results["trainer"] = dict(receipts=313, episodes=12, final_state_and_model_equal=True,
                              checkpoint_probes=len(checks), correct_probes=sum(a == "1" for a, _ in checks))
    print(f"trainer: {results['trainer']}", flush=True)
    paths = [Path(__file__), ROOT / "python/grounded_archive_probe.py", ROOT / "python/check_grounded_teaching.py", ROOT / "python/check_observation_language.py",
             ROOT / "cpp/apps/grounded_archive_probe.cpp", ROOT / "cpp/apps/grounded_feedback.h", ROOT / "cpp/apps/grounded_bootstrap.h",
             ROOT / "cpp/tests/test_grounded_archive.cpp", ROOT / "cpp/apps/laminar_train.cpp", ROOT / "cpp/src/organism.cpp",
             ROOT / "build-wsl/grounded_archive_probe", ROOT / "build-asan/test_grounded_archive",
             ROOT / "build-wsl/laminar_train_grounded", ROOT / "scripts/grounded_archive.sh",
             ROOT / "cpp/CMakeLists.txt", ROOT / "scripts/build_wsl.sh", ROOT / "logs/grounded_session/probe.jsonl"]
    paths += list((ROOT / "cpp/include/laminar").glob("*.h"))
    paths += [p for p in OUT.iterdir() if p.name != "audit.json" and p.suffix in (".json", ".inputs", ".txt", ".tsv", ".log")]
    manifest = ROOT / "logs/grounded_session/bootstrap.paths"
    paths.append(manifest)
    for line in manifest.read_text().splitlines():
        key, name = line.split("\t")
        paths += [Path(name) / f"micro_{seed}.txt" for seed in seeds] if key == "micro" else [Path(name)]
    (OUT / "audit.json").write_text(json.dumps(dict(results=results, sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), indent=2) + "\n")


if __name__ == "__main__": main()
