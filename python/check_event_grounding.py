"""Independent state, curriculum-exposure and native-answer audit."""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_world(lines):
    facts = set()
    for _ in range(int(next(lines))):
        a, r, b = next(lines).split("\t"); facts.add((a, int(r), b))
    return facts


def update(state, r, effect, a, b):
    result = set()
    for source, relation, target in state:
        if relation != r:
            result.add((source, relation, target)); continue
        if effect == 1 and (source, target) == (a, b): continue
        if effect == 2 and source == a: continue
        if effect == 3 and target == b: continue
        result.add((source, relation, target))
    if effect != 1: result.add((a, r, b))
    return result


def count_at_endpoint(question, state, seed):
    names = {x for a, _, b in state for x in (a, b)}
    args = [w for w in re.findall("[a-z0-9]+", question.lower()) if w in names]
    assert len(args) == 1
    x = args[0]; visited = set()
    while True:
        values = {b for a, r, b in state if a == x and r == seed % 2}
        if not values: break
        assert len(values) == 1 and x not in visited
        visited.add(x); x = next(iter(values))
    return str(len({b for a, r, b in state if a == x and r == 1 - seed % 2}))


def main():
    totals = collections.defaultdict(collections.Counter); paths = []
    for seed in range(3):
        paths += [ROOT / f"logs/semantic_grounding_corpus/train_{seed}.txt", ROOT / f"logs/reasoning_composition_verified/micro_{seed}.txt"]
        p = ROOT / f"logs/event_grounding_corpus/sensors_{seed}.txt"; paths.append(p)
        lines = iter(p.read_text().splitlines()); seen = set(); exposure = collections.Counter(); sensor_names = set()
        for header in lines:
            phase, scene = header.split("\t"); assert scene not in seen; seen.add(scene); exposure[phase] += 1
            text = next(lines); before, after = read_world(lines), read_world(lines)
            assert before != after
            names = {x for world in (before, after) for a, _, b in world for x in (a, b)}
            sensor_names |= names
            assert len([w for w in re.findall("[a-z0-9]+", text.lower()) if w in names]) == 2
        assert exposure == {"weak": 144, "repeat": 144, "contrast": 144}
        p = ROOT / f"logs/event_grounding_corpus/test_{seed}.txt"; paths.append(p)
        lines = iter(p.read_text().splitlines()); expected_n = collections.Counter(); test_names = set()
        for header in lines:
            group, n, steps = header.split("\t"); question = next(lines); initial = set()
            for _ in range(int(n)):
                a, r, b, text = next(lines).split("\t"); initial.add((a, int(r), b))
            gold_before = next(lines); assert count_at_endpoint(question, initial, seed) == gold_before
            expected_n[(group, "before")] += 1; state = initial; last_gold = gold_before
            test_names |= {x for a, _, b in state for x in (a, b)}
            for i in range(int(steps)):
                r, effect, a, b, text = next(lines).split("\t"); r, effect = int(r), int(effect)
                gold, wanted = next(lines), read_world(lines)
                new = update(state, r, effect, a, b); assert new == wanted
                if group == "unknown_event": assert gold == "_"
                else:
                    assert count_at_endpoint(question, new, seed) == gold
                    if group in ("link_reassignment", "ownership_transfer", "retraction", "correction") or (group == "old_owner_release" and i == 0): assert last_gold != gold
                    if group in ("duplicate_event", "unrelated_relation") or (group == "old_owner_release" and i == 1): assert last_gold == gold
                if group == "old_owner_release" and i == 1: assert new == state
                if group == "correction" and i == 1: assert gold == gold_before
                expected_n[(group, "after")] += 1; state = new; last_gold = gold
        assert not sensor_names & test_names
        p = ROOT / f"logs/event_grounding_native/{seed}.json"; paths.extend([p, Path(str(p) + ".examples.txt")])
        result = json.loads(p.read_text()); assert result["seed"] == seed
        assert result["weak_frames"] == 2 and result["grounded_frames"] == 12 and result["question_procedures"] == 18
        assert len(result["scores"]) == 48
        for key, sc in result["scores"].items():
            condition, group, phase = key.split("/"); n = expected_n[(group, phase)]
            assert sc["n"] == n and sc["covered"] + sc["declined"] == n
            if condition == "contrast_grounded":
                if group == "unknown_event" and phase == "after": assert sc["covered"] == 0
                else: assert sc["covered"] == sc["state"] == sc["exact"] == n, (key, sc)
            elif condition == "weak_scenes": assert sc["covered"] == sc["state"] == 0
            elif phase == "before": assert sc["state"] == sc["exact"] == n
            elif group in ("duplicate_event", "unrelated_relation"): assert sc["state"] == sc["exact"] == n
            else: assert sc["state"] == 0
            totals[key].update(sc)
    paths += [ROOT / p for p in ("cpp/include/laminar/event_grounding.h", "cpp/include/laminar/reasoning_language.h", "cpp/apps/event_grounding_probe.cpp", "cpp/tests/test_event_grounding.cpp", "python/event_grounding_probe.py", "python/check_event_grounding.py", "scripts/event_grounding.sh", "build-wsl/event_grounding_probe")]
    out = ROOT / "logs/event_grounding_audit.json"
    out.write_text(json.dumps({"totals": totals, "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}, indent=2) + "\n")
    compact = collections.defaultdict(collections.Counter)
    for k, v in totals.items(): compact[k.split('/')[0]].update(v)
    print(json.dumps(compact, indent=2)); print(f"Verified matched curricula, state transitions, answer changes and native results: {out}")


if __name__ == "__main__": main()
