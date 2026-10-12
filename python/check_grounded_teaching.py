"""Independent evidence replay for the continuing byte-session learner."""
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from check_observation_language import answer, parse_knowledge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/grounded_teaching"
RULES = {
    "the $0 was logged beside $1 ": "9,1,0,2;",
    "the location record for $0 went dark ": "5,0,0,0;",
    "the whereabouts of $0 shifted into $1 ": "9,0,0,2;",
    "the ownership record for $0 went dark ": "5,1,0,0;",
}


def empty(): return dict(positive=set(), negative=set(), rows=set(), columns=set())


def observe(k, text):
    s = text.lower().rstrip(".")
    move = re.fullmatch(r"(\w+) moved to the (\w+)", s)
    has = re.fullmatch(r"(\w+) has the (\w+)", s)
    logged = re.fullmatch(r"the (\w+) was logged beside (\w+)", s)
    shifted = re.fullmatch(r"the whereabouts of (\w+) shifted into (\w+)", s)
    gap = re.fullmatch(r"the (location|ownership) record for (\w+) went dark", s)
    if gap:
        r, a = int(gap[1] == "ownership"), gap[2]
        k["positive"] = {f for f in k["positive"] if f[:2] != (a, r)}
        k["negative"] = {f for f in k["negative"] if f[:2] != (a, r)}
        k["rows"].discard((a, r)); k["columns"] = {f for f in k["columns"] if f[0] != r}
        return "the " + gap[1] + " record for $0 went dark "
    if move: a, r, b = move[1], 0, move[2]
    elif has: a, r, b = has[2], 1, has[1]
    elif logged: a, r, b = logged[1], 1, logged[2]
    else:
        assert shifted, text
        a, r, b = shifted[1], 0, shifted[2]
    k["positive"] = {f for f in k["positive"] if f[:2] != (a, r)} | {(a, r, b)}
    k["negative"] = {f for f in k["negative"] if f[:2] != (a, r)}
    k["rows"].add((a, r))
    return "the $0 was logged beside $1 " if logged else "the whereabouts of $0 shifted into $1 " if shifted else None


def read(path):
    with path.open() as f: return list(csv.DictReader(f, delimiter="\t"))


def expectations(turns):
    base, reports, support = empty(), [], collections.Counter()
    lessons = collections.defaultdict(list)
    for line in (OUT / "feedback.tsv").read_text().splitlines():
        fields = line.split("\t"); assert len(fields) == 10
        lessons[int(fields[0])].append(fields)
    expected, scored_lessons = [], 0
    def replay(limit=None):
        state = copy.deepcopy(base)
        for record in reports[:limit]:
            if record["active"]: observe(state, record["text"])
        return state
    for t, turn in enumerate(turns):
        reply = ""; route = None
        for line in turn["text"].splitlines():
            if not line: continue
            if line == "<|endofdialogue|>": base, reports = empty(), []
            elif line.startswith("CONTEXT: "): observe(base, line[9:])
            elif line.startswith("ENVIRONMENT: "):
                reports.append(dict(text=line[13:], active=True))
                if turn["training"]:
                    frame = observe(copy.deepcopy(base), line[13:]);support[frame] += 1
            elif line.startswith("USER: "):
                text = line[6:]
                replace = re.fullmatch(r"replace the (earliest|latest) active report about (\w+) with (.+)", text)
                withdraw = re.fullmatch(r"withdraw the latest active report about (\w+)", text)
                if replace or withdraw:
                    entity = replace[2] if replace else withdraw[1]
                    selected = [i for i, r in enumerate(reports) if r["active"] and entity in re.findall(r"\w+", r["text"])]
                    assert selected
                    chosen = selected[0] if replace and replace[1] == "earliest" else selected[-1]
                    if replace: reports[chosen]["text"] = replace[3] + "."
                    else: reports[chosen]["active"] = False
                    reply, route = "The report has been revised.", "correction"
                else:
                    gold = answer(text, replay());reply = "I do not know." if gold == "unknown" else gold;route = "question"
            else: assert line == "ASSISTANT:", line
        state = replay()
        for row in lessons.pop(turn["offset"], []):
            offset, episode, report, revision, evidence, previous, op, source, q, a = row
            assert turn["training"] and int(episode) == turn["episode"] + 1
            assert int(report) == len(reports) and revision == previous == "0" and op == "add" and source == "verified_teacher"
            assert answer(q, state) == a, (t, q, a, answer(q, state))
            scored_lessons += 1
        states = {"current": state} | {str(i + 1): replay(i + 1) for i in range(len(reports))}
        expected.append(dict(reply=reply, route=route, states=states, support=dict(support), feedback=scored_lessons))
    assert not lessons and scored_lessons == 288
    return expected


def audit(seed, turns, expected):
    root = OUT / str(seed)
    trace = read(Path(str(root) + ".turns.tsv"));states = read(Path(str(root) + ".states.tsv"));models = read(Path(str(root) + ".models.tsv"))
    assert len(trace) == len(turns)
    state_index = {(int(row["turn"]), row["scope"]): row for row in states}
    model_index = {(int(row["turn"]), row["frame"]): row for row in models}
    assert len(state_index) == len(states) and len(model_index) == len(models)
    score = collections.Counter()
    for t, (turn, gold, row) in enumerate(zip(turns, expected, trace)):
        assert int(row["turn"]) == t and int(row["offset"]) == turn["offset"]
        assert row["reply"] == gold["reply"], (seed, t, row["reply"], gold["reply"])
        if gold["route"]: assert row["route"] == gold["route"]
        assert int(row["feedback"]) == gold["feedback"]
        if turn["label"] != "start": assert row["consistent"] == "1"
        score["turns"] += 1
        if gold["reply"]:
            score["teaching_replies" if turn["training"] else "transfer_replies"] += 1
            score["unknown_answers"] += gold["reply"] == "I do not know."
            score["corrections"] += gold["route"] == "correction"
        for scope, state in gold["states"].items():
            row = state_index.pop((t, scope))
            assert parse_knowledge(row["knowledge"], seed) == state, (seed, t, scope)
            score["audited_states"] += 1
        for frame, support in gold["support"].items():
            row = model_index.pop((t, frame))
            assert int(row["support"]) == support, (seed, t, frame, row)
            rules = row["rules"]
            if seed % 2:
                rules = "".join(",".join([parts[0], str(int(parts[1]) ^ 1), *parts[2:]]) + ";" for parts in (r.split(",") for r in rules.rstrip(";").split(";")))
            assert rules == RULES[frame], (seed, t, frame, rules)
            score["audited_frame_states"] += 1
    assert not state_index and not model_index
    print(f"seed={seed}: {dict(score)}", flush=True)
    return dict(score)


def main():
    turns = json.loads((OUT / "turns.json").read_text())
    data = (OUT / "trainer.txt").read_bytes();assert data == "".join(t["text"] for t in turns).encode()
    with (OUT / "turns.inputs").open("rb") as f:
        assert int(f.readline()) == len(turns)
        for turn in turns:
            assert f.read(int(f.readline())).decode() == turn["text"] and f.read(1) == b"\n"
        assert not f.read()
    expected = expectations(turns)
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    results = {s: audit(s, turns, expected) for s in seeds}
    if len(seeds) == 3:
        baseline = read(OUT / "0.no-feedback.turns.tsv");assert len(baseline) == len(turns)
        correct = total = 0
        for turn, gold, row in zip(turns, expected, baseline):
            assert row["feedback"] == "0"
            if not turn["training"] and gold["reply"]:
                total += 1;correct += row["reply"] == gold["reply"]
        assert correct < total
        results["without_feedback"] = dict(correct=correct, total=total)
        print(f"without feedback: {correct}/{total} transfer replies")
        logs = [(OUT / f"trainer_{name}.log").read_text() for name in ("probe", "plain")]
        signatures = [re.findall(r"GROUNDED observation_model BEGIN\n(.*?)GROUNDED observation_model END", log, re.S) for log in logs]
        assert signatures[0] == signatures[1] == [(OUT / "0.model.txt").read_text()]
        states = [re.findall(r"GROUNDED state (.+)", log) for log in logs]
        assert states[0] and states[0] == states[1]
        names, _, projected, *_ = states[0][0].split("|")
        names = names.rstrip(",").split(",")
        final = empty()
        for marker, key in (("p", "positive"), ("n", "negative")):
            final[key] = {(names[int(a)], int(r), names[int(b)]) for a, r, b in re.findall(marker + r"(\d+),(\d+),(\d+);", projected)}
        final["rows"] = {(names[int(a)], int(r)) for a, r in re.findall(r"r(\d+),(\d+);", projected)}
        final["columns"] = {(int(r), names[int(b)]) for r, b in re.findall(r"c(\d+),(\d+);", projected)}
        assert final == expected[-1]["states"]["current"]
        counters = [re.findall(r"GROUNDED stream (.+)", log) for log in logs]
        assert counters[0] and counters[0] == counters[1]
        for log in logs: assert "GROUNDED teaching accepted=288 rejected=0 duplicates=0 delivered=288 pending=0" in log
        checks = re.findall(r"session_probe (\d+)/(\d+)", logs[0]);assert len(checks) >= 3 and all(a == b == "1" for a, b in checks), checks
    paths = [ROOT / "python/grounded_teaching_probe.py", Path(__file__), ROOT / "python/check_observation_language.py",
             ROOT / "cpp/apps/grounded_teaching_probe.cpp", ROOT / "cpp/apps/grounded_feedback.h", ROOT / "cpp/apps/grounded_bootstrap.h",
             ROOT / "cpp/apps/laminar_train.cpp", ROOT / "cpp/src/organism.cpp", ROOT / "cpp/tests/test_grounded_teaching.cpp",
             ROOT / "scripts/grounded_teaching.sh", ROOT / "cpp/CMakeLists.txt", ROOT / "scripts/build_wsl.sh",
             ROOT / "logs/grounded_session/probe.jsonl", ROOT / "build-asan/test_grounded_teaching",
             ROOT / "build-wsl/grounded_teaching_probe", ROOT / "build-wsl/laminar_train_grounded"]
    paths += list((ROOT / "cpp/include/laminar").glob("*.h"))
    paths += [p for p in OUT.iterdir() if p.name != "audit.json" and p.suffix in (".tsv", ".txt", ".json", ".inputs", ".log")]
    paths += [ROOT / "logs/grounded_session/bootstrap.paths"]
    for line in paths[-1].read_text().splitlines():
        key, name = line.split("\t");paths += [Path(name) / f"micro_{seed}.txt" for seed in seeds] if key == "micro" else [Path(name)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT / "audit.json").write_text(json.dumps(dict(results=results, sha256=hashes), indent=2) + "\n")


if __name__ == "__main__": main()
