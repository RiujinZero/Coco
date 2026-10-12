"""Independent replay of unlabeled mixed dialogue turns and trainer checks."""
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
OUT = ROOT / "logs/grounded_session"


def empty(): return dict(positive=set(), negative=set(), rows=set(), columns=set())


def observe(state, text):
    text = text.lower().rstrip(".")
    held = re.fullmatch(r"(\w+) has the (\w+)", text)
    moved = re.fullmatch(r"(\w+) moved to the (\w+)", text)
    gap = re.fullmatch(r"no one reported anything about the (\w+) for a while", text)
    positive = re.fullmatch(r"it is visible that (\w+) holds the (\w+)", text)
    complete = re.fullmatch(r"the complete holder list for the (\w+) is (\w+)", text)
    assert held or moved or gap or positive or complete, text
    if held or moved:
        a, relation, b = (held[2], 1, held[1]) if held else (moved[1], 0, moved[2])
        for key in ("positive", "negative"): state[key] = {f for f in state[key] if f[:2] != (a, relation)}
        state["positive"].add((a, relation, b)); state["rows"].add((a, relation))
    elif gap:
        for key in ("positive", "negative"): state[key] = {f for f in state[key] if f[:2] != (gap[1], 1)}
        state["rows"].discard((gap[1], 1)); state["columns"] = {scope for scope in state["columns"] if scope[0] != 1}
    elif positive: state["positive"].add((positive[2], 1, positive[1]))
    else:
        assert all(a != complete[1] or rel != 1 or b == complete[2] for a, rel, b in state["positive"])
        state["positive"].add((complete[1], 1, complete[2])); state["rows"].add((complete[1], 1))


def answer(state, question):
    text = question.lower().rstrip("?")
    who = re.fullmatch(r"who has the (\w+) now", text)
    where = re.fullmatch(r"where is (\w+)", text)
    does = re.fullmatch(r"does (\w+) have the (\w+)", text)
    if who or where:
        a, rel = ((who or where)[1], int(bool(who)))
        values = {b for s, r, b in state["positive"] if (s, r) == (a, rel)}
        return next(iter(values)) if (a, rel) in state["rows"] and len(values) == 1 else "I do not know."
    if does:
        fact = (does[2], 1, does[1])
        return "yes" if fact in state["positive"] else "no" if fact in state["negative"] or fact[:2] in state["rows"] else "I do not know."
    return None


class Replay:
    def __init__(self):
        self.base = empty(); self.records = []; self.started = False; self.pending = None
        self.counts = collections.Counter(); self.status = "session_ready"; self.route = ""

    def states(self):
        state = copy.deepcopy(self.base); out = {}
        for record in self.records:
            if record["active"]: observe(state, record["text"])
            out[str(record["id"])] = copy.deepcopy(state)
        out["current"] = state
        return out

    def select(self, reference):
        about = re.fullmatch(r"the (earliest|latest|previous) active report about (\w+)", reference)
        source = re.fullmatch(r"the (earliest|latest) active report from (\w+)(?: about (\w+))?", reference)
        weak = re.fullmatch(r"the active report about (\w+)", reference)
        if not (about or source or weak): return None
        matches = []
        for r in self.records:
            if not r["active"]: continue
            words = set(re.findall(r"[a-z0-9]+", r["text"].lower()))
            if about and about[2] not in words: continue
            if source and (r["source"] != source[2] or source[3] and source[3] not in words): continue
            if weak and weak[1] not in words: continue
            matches.append(r["id"])
        if weak: return set(matches)
        rank = {"earliest": 0, "latest": -1, "previous": -2}[(about or source)[1]]
        return {matches[rank]} if matches and len(matches) >= abs(rank) else set()

    def edit(self, id):
        p = self.pending
        record = next(r for r in self.records if r["id"] == id)
        if record["source"] != p["source"]:
            self.status = "stale_report_revision"; self.counts["refusals"] += 1
            return "The report could not be revised."
        record["revision"] += 1; record["active"] = p["operation"] == 2
        if record["active"]: record["text"] = p["payload"]
        self.pending = None; self.status = "revised_report"; self.counts["edits"] += 1; self.counts["history"] += 1
        return "The report has been revised."

    def user(self, text, source):
        self.started = True; self.counts["requests"] += 1
        if text.isdecimal() and self.pending:
            self.route = "clarification"
            if source != self.pending["source"]:
                self.status = "clarification_source_mismatch"; self.counts["refusals"] += 1
                return "That choice has a different source."
            if (self.counts["observation"], self.counts["history"]) != self.pending["stamp"]:
                self.pending = None; self.status = "stale_correction"; self.counts["refusals"] += 1
                return "That correction needs a fresh reference."
            if int(text) not in self.pending["candidates"]:
                self.status = "invalid_report_choice"; self.counts["refusals"] += 1
                return "Choose one of the listed reports."
            return self.edit(int(text))
        result = answer(self.states()["current"], text)
        if result is not None:
            self.route = "question"; self.counts["questions"] += 1
            # Reader statuses are checked as a category below; independent
            # semantics do not depend on the internal choice of question executor.
            self.status = "QUESTION"
            return result
        text = text.lower().rstrip("?")
        replace = re.fullmatch(r"replace (.+) with (.+)", text)
        withdraw = re.fullmatch(r"withdraw (.+)", text)
        keep = re.fullmatch(r"leave (.+) unchanged", text)
        weak = re.fullmatch(r"consider (.+)", text)
        command = replace or withdraw or keep or weak
        selected = self.select(command[1]) if command else None
        if selected is None:
            self.route = "unknown"; self.status = "unknown_grounded_request"; self.counts["refusals"] += 1
            return "I do not understand that request."
        self.route = "correction"; self.pending = None
        if weak:
            self.status = "ambiguous_correction_meaning"; self.counts["refusals"] += 1
            return "I cannot interpret that correction."
        if not selected:
            self.status = "missing_reference"; self.counts["refusals"] += 1
            return "I cannot identify that report."
        if keep:
            assert len(selected) == 1; self.status = "kept_report"
            return "The report is unchanged."
        self.pending = dict(candidates=selected, source=source, operation=2 if replace else 1,
                            payload=replace[2].rstrip(".") + "." if replace else "",
                            stamp=(self.counts["observation"], self.counts["history"]))
        if len(selected) == 1: return self.edit(next(iter(selected)))
        self.status = "ambiguous_reference"; self.counts["clarifications"] += 1
        return "Which report? Candidates:" + "".join(f" {id}." for id in sorted(selected))

    def turn(self, turn):
        if turn["generated"]: return ""
        response = ""
        for line in turn["text"].splitlines():
            if not line: continue
            if line == "<|endofdialogue|>":
                self.base = empty(); self.records = []; self.pending = None; self.started = False
                self.status = "episode_reset"; self.route = ""; self.counts["resets"] += 1
            elif line.startswith("CONTEXT: "):
                if self.started: self.status = "late_context_rejected"
                else: observe(self.base, line[9:]); self.counts["observation"] += 1
            elif line.startswith("ENVIRONMENT: "):
                self.started = True; self.counts["observation"] += 1; self.counts["reports"] += 1
                self.records.append(dict(id=len(self.records) + 1, revision=0, active=True, source=turn["source"], text=line[13:]))
                self.status = "replayed_evidence"
            elif line.startswith("USER: "): response = self.user(line[6:], turn["source"])
            else: assert line == "ASSISTANT:", line
        return response


def read(path):
    with path.open() as f: return list(csv.DictReader(f, delimiter="\t"))


def audit(seed, data):
    root = OUT / str(seed)
    def path(s): return Path(str(root) + s)
    expected_model = (ROOT / f"logs/history_reference/{seed}.model.txt").read_text() + (ROOT / f"logs/correction_language/{seed}.model.txt").read_text()
    assert path(".model.txt").read_text() == expected_model
    turns = read(path(".turns.tsv")); states = read(path(".states.tsv")); records = read(path(".records.tsv"))
    state_index = {(int(r["episode"]), int(r["turn"]), r["scope"]): r for r in states}
    record_index = {(int(r["episode"]), int(r["turn"]), int(r["id"])): r for r in records}
    assert len(state_index) == len(states) and len(record_index) == len(records)
    assert len(turns) == sum(len(e["turns"]) for e in data)
    engine = Replay(); index = 0; score = collections.Counter()
    for e, episode in enumerate(data):
        for t, turn in enumerate(episode["turns"]):
            row = turns[index]; index += 1
            reply = engine.turn(turn)
            assert (int(row["episode"]), int(row["turn"])) == (e, t)
            assert row["reply"] == reply, (seed, e, t, row["reply"], reply)
            assert row["route"] == engine.route, (seed, e, t, row["route"], engine.route)
            if engine.status == "QUESTION":
                assert row["status"] in ("answer", "grounded_answer", "unknown_answer", "missing_evidence", "missing_observation", "nonunique_result"), row
            else: assert row["status"] == engine.status, (seed, e, t, row["status"], engine.status)
            for key in ("requests", "questions", "reports", "edits", "clarifications", "refusals", "resets", "observation", "history"):
                assert int(row[key]) == engine.counts[key], (seed, e, t, key, row[key], engine.counts[key])
            assert bool(int(row["pending"])) == bool(engine.pending)
            assert bool(int(row["probe_isolated"])) == (t == 13)
            score["probe_isolation"] += t == 13
            score["turns"] += 1; score["replies"] += bool(reply); score["unknown_answers"] += reply == "I do not know."
            for scope, state in engine.states().items():
                record = state_index.pop((e, t, scope))
                assert parse_knowledge(record["knowledge"], seed) == state, (seed, e, t, scope)
                score["audited_states"] += 1
            for expected in engine.records:
                record = record_index.pop((e, t, expected["id"]))
                assert int(record["revision"]) == expected["revision"] and bool(int(record["active"])) == expected["active"]
                assert record["source"] == expected["source"] and record["text"] == expected["text"]
                score["audited_records"] += 1
    assert not state_index and not record_index
    score.update({"requests": engine.counts["requests"], "questions": engine.counts["questions"], "edits": engine.counts["edits"], "reports": engine.counts["reports"]})
    print(f"seed={seed}: {dict(score)}", flush=True)
    return dict(score)


def inputs(data):
    with (OUT / "dialogues.inputs").open("rb") as f:
        assert int(f.readline()) == len(data)
        for e in data:
            assert f.readline().decode().strip().split("\t") == [str(e["id"]), str(len(e["turns"]))]
            for turn in e["turns"]:
                source, generated, count = f.readline().decode().strip().split("\t")
                assert source == turn["source"] and int(generated) == turn["generated"]
                assert f.read(int(count)).decode() == turn["text"] and f.read(1) == b"\n"
        assert not f.read()


def main():
    data = json.loads((OUT / "dialogues.json").read_text()); inputs(data)
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    results = {seed: audit(seed, data) for seed in seeds}
    if len(seeds) == 3:
        probe = (OUT / "trainer_probe.log").read_text(); plain = (OUT / "trainer_plain.log").read_text()
        pattern = r"GROUNDED stream (.+)"
        assert re.findall(pattern, probe) == re.findall(pattern, plain)
        final_states = re.findall(r"GROUNDED state (.+)", probe)
        assert len(final_states) == 1 and final_states == re.findall(r"GROUNDED state (.+)", plain)
        names = final_states[0].split("|")[0].rstrip(",").split(",")
        positive = {(names[int(a)], int(rel), names[int(b)]) for a, rel, b in re.findall(r"p(\d+),(\d+),(\d+);", final_states[0])}
        assert positive == {("key", 1, "cy"), ("coin", 1, "ben"), ("pen", 1, "cy")}
        assert "requests=3 questions=2 reports=1 edits=1 clarifications=0 refusals=0 episodes=1" in probe
        checks = re.findall(r"QA .+?session_probe (\d+)/(\d+)", probe)
        assert len(checks) >= 3 and all(a == b == "1" for a, b in checks), checks
        disabled = (OUT / "trainer_disabled.log").read_text()
        assert "trained " in disabled and "GROUNDED" not in disabled
    paths = [Path(__file__), ROOT / "python/grounded_session_probe.py", ROOT / "python/build_grounded_bundle.py",
             ROOT / "python/check_observation_language.py", ROOT / "cpp/apps/grounded_bootstrap.h", ROOT / "cpp/apps/grounded_session_probe.cpp",
             ROOT / "cpp/apps/laminar_train.cpp", ROOT / "cpp/tests/test_grounded_session.cpp", ROOT / "cpp/tests/test_packed_allocation.cpp", ROOT / "cpp/src/organism.cpp",
             ROOT / "cpp/CMakeLists.txt", ROOT / "scripts/build_wsl.sh", ROOT / "scripts/grounded_session.sh", ROOT / "build-wsl/grounded_session_probe", ROOT / "build-wsl/laminar_train_grounded", ROOT / "build-asan/test_grounded_session", ROOT / "build-asan/test_packed_allocation"]
    paths += list((ROOT / "cpp/include/laminar").glob("*.h"))
    paths += [p for p in OUT.iterdir() if p.name != "audit.json" and p.suffix in (".txt", ".tsv", ".json", ".jsonl", ".inputs", ".paths", ".log")]
    for line in (OUT / "bootstrap.paths").read_text().splitlines():
        key, name = line.split("\t")
        paths += [Path(name) / f"micro_{seed}.txt" for seed in seeds] if key == "micro" else [Path(name)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT / "audit.json").write_text(json.dumps(dict(results=results, sha256=hashes), indent=2) + "\n")


if __name__ == "__main__":
    main()
