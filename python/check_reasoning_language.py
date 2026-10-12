"""Independent corpus-label and native-result audit for the text bridge."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "logs/reasoning_language_corpus"


def read_cases(path):
    lines = iter(path.read_text().splitlines())
    for h in lines:
        group, family, gold, n = h.split("\t")
        question = next(lines)
        facts = [next(lines).split("\t", 3) for _ in range(int(n))]
        yield group, int(family), gold, question, facts


def audit_case(case):
    group, family, gold, question, facts = case
    names = {x for a, _, b, _ in facts for x in (a, b)}
    args = [w.strip("?,.").lower() for w in question.split() if w.strip("?,.").lower() in names]
    assert 1 <= len(args) <= 2
    a, b = args[0], args[-1]
    edges = collections.defaultdict(set)
    for src, rel, dst, _ in facts:
        edges[(src, int(rel))].add(dst)

    def end(src):
        seen = set()
        while edges[(src, 0)]:
            assert src not in seen and len(edges[(src, 0)]) == 1
            seen.add(src)
            src = next(iter(edges[(src, 0)]))
        return src

    def count(src):
        return len(edges[(src, 1)])

    ea, eb = end(a), end(b)
    result = [ea, count(a), a == b, count(ea), count(a) == count(b),
              count(ea) == count(eb), ea == eb][family]
    answer = ("yes" if result else "no") if isinstance(result, bool) else str(result)
    assert answer == gold, (group, family, answer, gold)
    if family == 3:
        assert count(a) != count(ea), "source-count shortcut leaked"
    return names


def main():
    totals = collections.defaultdict(collections.Counter)
    all_inputs = []
    for seed in range(3):
        all_inputs.append(ROOT / f"logs/reasoning_composition_verified/micro_{seed}.txt")
        split_names = {}
        for split in ("train", "adapt", "test"):
            path = CORPUS / f"{split}_{seed}.txt"; all_inputs.append(path)
            names = set()
            for case in read_cases(path):
                names |= audit_case(case)
            split_names[split] = names
        assert not (split_names["train"] & split_names["test"])
        assert not (split_names["adapt"] & split_names["test"])
        path = ROOT / f"logs/reasoning_language_native/{seed}.json"; all_inputs.append(path)
        all_inputs.append(Path(str(path) + ".examples.txt"))
        result = json.loads(path.read_text())
        assert result["seed"] == seed and result["initial_procedures"] == 14
        assert result["adapted_procedures"] == 21 and result["fact_frames"] == 6
        assert len(result["scores"]) == 98
        for key, score in result["scores"].items():
            condition, group, family = key.split("/"); family = int(family)
            n = score["n"]
            assert n == (100 if group == "known" else 20 if group == "digits" else 40)
            for metric in ("exact", "extracted", "interpreted", "covered"):
                assert 0 <= score[metric] <= n
            if condition == "fully_adapted" or (condition in ("initial", "fact_adapted") and group in ("known", "digits")) or (condition == "fact_adapted" and group == "new_facts"):
                assert all(score[k] == n for k in ("exact", "extracted", "interpreted", "covered")), key
            if condition in ("initial", "fact_adapted") and group == "new_question":
                assert score["extracted"] == n and score["interpreted"] == score["covered"] == 0
            if condition == "initial" and group == "new_facts":
                assert score["extracted"] == score["covered"] == 0
            if condition == "no_bindings":
                assert score["extracted"] == score["interpreted"] == n
                assert score["covered"] == (n if family < 4 else 0)
            totals[f"{condition}/{group}"].update(score)
    paths = [ROOT / p for p in ("cpp/include/laminar/reasoning_language.h", "cpp/include/laminar/sequence.h", "cpp/include/laminar/organism.h", "cpp/src/organism.cpp", "cpp/include/laminar/config.h", "cpp/include/laminar/reasoning_runtime.h", "cpp/include/laminar/procedure_memory.h", "cpp/apps/reasoning_language_probe.cpp", "cpp/tests/test_reasoning_language.cpp", "python/reasoning_language_probe.py", "python/check_reasoning_language.py", "scripts/reasoning_language.sh", "build-wsl/reasoning_language_probe")]
    audit = {"totals": totals, "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in all_inputs + paths}}
    path = ROOT / "logs/reasoning_language_audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(totals, indent=2))
    print(f"Corpus labels, disjoint splits, all native seeds and ablations verified: {path}")


if __name__ == "__main__":
    main()
