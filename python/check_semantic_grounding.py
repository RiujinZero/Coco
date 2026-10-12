"""Independent audit of shared-meaning transfer and withheld compositions."""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cases(path):
    lines = iter(path.read_text().splitlines())
    for header in lines:
        group, gold, meaning, n = header.split("\t")
        question = next(lines)
        facts = [next(lines).split("\t", 3) for _ in range(int(n))]
        yield group, gold, meaning, question, facts


def expression(text):
    i = 0
    def parse():
        nonlocal i
        if text[i] == "$":
            result = int(text[i + 1]); i += 2; return result
        op = int(text[i]); i += 1; assert text[i] == "("; i += 1; children = []
        while text[i] != ")":
            children.append(parse()); assert text[i] == ","; i += 1
        i += 1; return (op, *children)
    tree = parse(); assert i == len(text); return tree


def inspect(case):
    group, gold, meaning, question, facts = case
    names = {x for a, _, b, _ in facts for x in (a, b)}
    words = re.findall(r"[a-z0-9]+", question.lower()); args = []; form = []
    for word in words:
        if word in names:
            form.append(f"${len(args)}"); args.append(word)
        else:
            form.append(word)
    assert 1 <= len(args) <= 2
    relations = collections.defaultdict(set)
    for a, r, b, _ in facts:
        relations[(a, int(r))].add(b)
    def value(tree):
        if isinstance(tree, int): return args[tree]
        op = tree[0]
        if op == 8: return value(tree[1]) == value(tree[2])
        x = value(tree[1])
        if op == 7: return len(relations[(x, 1)])
        assert op == 4
        visited = set()
        while relations[(x, 0)]:
            assert len(relations[(x, 0)]) == 1 and x not in visited
            visited.add(x); x = next(iter(relations[(x, 0)]))
        return x
    result = value(expression(meaning))
    expected = ("yes" if result else "no") if isinstance(result, bool) else str(result)
    assert gold == expected or (group == "unknown_words" and gold == "_")
    return " ".join(form), names


def main():
    totals = collections.defaultdict(collections.Counter); paths = []
    for seed in range(3):
        forms = {}; entity_sets = {}; count = collections.Counter()
        for split in ("train", "adapt", "test"):
            p = ROOT / f"logs/semantic_grounding_corpus/{split}_{seed}.txt"; paths.append(p)
            forms[split] = set(); entity_sets[split] = set(); previous = None
            for case in cases(p):
                group, gold, meaning, question, facts = case
                form, names = inspect(case); forms[split].add(form); entity_sets[split] |= names
                if split != "test":
                    assert meaning in ("$0", "4($0,)", "7($0,)", "8($0,$1,)")
                else:
                    count[group] += 1
                    if group == "attachment_pair":
                        assert previous[0] == "cross_counts" and previous[1] != gold
                        assert sorted(x[:3] for x in previous[4]) == sorted(x[:3] for x in facts)
                previous = case
        assert len(forms["train"]) == 18 and len(forms["adapt"]) == 4
        assert not forms["test"] & (forms["train"] | forms["adapt"]), "a complete held-out question was trained"
        assert not entity_sets["test"] & (entity_sets["train"] | entity_sets["adapt"])
        p = ROOT / f"logs/semantic_grounding_native/{seed}.json"; paths += [p, Path(str(p) + ".examples.txt")]
        result = json.loads(p.read_text())
        assert result["seed"] == seed and result["initial_procedures"] == 18 and result["adapted_procedures"] == 22
        assert len(result["scores"]) == 36
        for key, sc in result["scores"].items():
            condition, group = key.split("/")
            assert sc["n"] == count[group] and sc["extracted"] == sc["n"] and sc["known_form"] == 0
            assert sc["exhausted"] == 0
            if group == "unknown_words" or condition in ("no_fragments", "whole_question_lookup") or (condition == "shared" and group == "new_alias"):
                assert sc["covered"] == 0, (key, sc)
            else:
                assert sc["covered"] == sc["meaning"] == sc["exact"] == sc["n"], (key, sc)
            totals[key].update({k: v for k, v in sc.items() if k != "max_states"})
            totals[key]["max_states"] = max(totals[key]["max_states"], sc["max_states"])
    paths += [ROOT / p for p in ("cpp/include/laminar/semantic_grounding.h", "cpp/include/laminar/reasoning_language.h", "cpp/apps/semantic_grounding_probe.cpp", "cpp/tests/test_semantic_grounding.cpp", "python/semantic_grounding_probe.py", "python/check_semantic_grounding.py", "scripts/semantic_grounding.sh", "build-wsl/semantic_grounding_probe")]
    output = ROOT / "logs/semantic_grounding_audit.json"
    output.write_text(json.dumps({"totals": totals, "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}, indent=2) + "\n")
    print(json.dumps(totals, indent=2)); print(f"Verified withheld forms, atomic-only training, labels, paired attachments and native results: {output}")


if __name__ == "__main__": main()
