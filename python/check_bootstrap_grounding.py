"""Independent native-output/state audit; never used by the learner."""
import collections
import csv
import hashlib
import json
from bootstrap_grounding_probe import ROOT, OUT, VERBS, opaque, truth


def main():
    totals = collections.defaultdict(collections.Counter)
    hashes = {}
    def read_train(path):
        lines = iter(path.read_text().splitlines())
        records = []
        for header in lines:
            serial, n = header.split("\t")
            story = [next(lines) for _ in range(int(n))]
            question, answer = next(lines), next(lines)
            assert int(serial) == len(records)
            records.append(dict(story=story, question=question, answer=answer))
        return records
    source_train = [dict(story=r["passage"].strip().splitlines(), question=r["question"], answer=r["answer"])
                    for line in (ROOT / "data/babi/babi_train.jsonl").read_text().splitlines()
                    if (r := json.loads(line))["task"] == 1]
    assert read_train(OUT / "train.txt") == source_train
    assert read_train(OUT / "opaque/train.txt") == [dict(story=[opaque(s) for s in r["story"]],
        question=opaque(r["question"]), answer=opaque(r["answer"])) for r in source_train]
    for r in source_train:
        assert truth(r["story"], r["question"])[0] == r["answer"].lower()
    base_gold = json.loads((OUT / "gold.json").read_text())
    for r in base_gold:
        if r["answer"] is not None:
            answer, state = truth(r["story"], r["question"])
            assert answer == r["answer"] and state == r["state"]
    for directory, seeds in ((OUT, range(3)), (OUT / "opaque", [0])):
        variant = "opaque" if directory.name == "opaque" else "english"
        golds = json.loads((directory / "gold.json").read_text())
        if variant == "opaque":
            for base, r in zip(base_gold, golds):
                assert r["story"] == [opaque(s) for s in base["story"]]
                assert r["question"] == opaque(base["question"])
                assert r["answer"] == (opaque(base["answer"]) if base["answer"] else None)
                assert r["state"] == ({opaque(k):opaque(v) for k,v in base["state"].items()} if base["state"] else None)
        for seed in seeds:
            learned = list(csv.reader((directory / f"{seed}.learned.tsv").open(), delimiter="\t"))
            expected_form = "where is $0 " if variant == "english" else opaque("where is") + " $0 "
            assert learned[0] == ["question", expected_form, "4,9,"]
            expected_events = {"$0 " + (verb + " the" if variant == "english" else opaque(verb + " the")) + " $1 " for verb in VERBS}
            assert {r[1] for r in learned[1:]} == expected_events
            assert len(learned[1:]) == len(VERBS)
            assert all(r[0] == "event" and r[2] == "0,0,1,2" for r in learned[1:])
            model = (directory / f"{seed}.model.txt").read_text()
            assert model.startswith("status=learned accepted=900 rejected=0 rounds=")
            assert model.count("viable=1 checked=900 skipped=0") == 2
            seen = set()
            with (directory / f"{seed}.trace.tsv").open() as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    i = int(r["id"]); mode = r["mode"]; key = (mode, i)
                    assert key not in seen; seen.add(key)
                    gold = golds[i]
                    facts = set()
                    for fact in r["facts"].split(";"):
                        if fact:
                            a, rel, b = fact.split(","); facts.add((a, int(rel), b))
                    supported = gold["answer"] is not None
                    answered = r["status"] in ("answer", "grounded_answer")
                    score = totals[f"{variant}/{mode}/" + (gold["group"] if supported else "untaught_verb")]
                    score["n"] += 1
                    score["answered"] += answered
                    score["exact"] += supported and answered and r["reply"] == gold["answer"]
                    score["state"] += supported and facts == {(a, 0, b) for a, b in gold["state"].items()}
                    if mode == "learned" and supported:
                        assert answered and r["reply"] == gold["answer"], (variant, seed, i, r, gold)
                        assert facts == {(a, 0, b) for a, b in gold["state"].items()}, (variant, seed, i, r)
                    if not supported or mode == "no_observation_grounding":
                        assert not answered and r["status"] == "unknown_event" and r["reply"] == "I do not know."
            assert len(seen) == len(golds) * 3
            for suffix in ("trace.tsv", "model.txt", "learned.tsv"):
                p = directory / f"{seed}.{suffix}"
                hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        for filename in ("train.txt", "test.txt", "gold.json"):
            p = directory / filename
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("cpp/include/laminar/bootstrap_grounding.h", "cpp/apps/bootstrap_grounding_probe.cpp",
                 "cpp/tests/test_bootstrap_grounding.cpp", "python/bootstrap_grounding_probe.py",
                 "python/check_bootstrap_grounding.py", "scripts/bootstrap_grounding.sh",
                 "build-wsl/bootstrap_grounding_probe", "cpp/include/laminar/procedure_memory.h",
                 "cpp/include/laminar/reasoning_language.h", "cpp/include/laminar/event_grounding.h"):
        p = ROOT / name; hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "audit.json").write_text(json.dumps(dict(totals=totals, sha256=hashes), indent=2) + "\n")
    compact = collections.defaultdict(collections.Counter)
    for key, value in totals.items():
        variant, mode, group = key.split("/")
        compact[f"{variant}/{mode}/" + ("unsupported" if group == "untaught_verb" else "supported")].update(value)
    print(json.dumps(compact, indent=2))
    print("Verified QA-only learned frames, held-out answers, complete states and opaque language control.")


if __name__ == "__main__":
    main()
