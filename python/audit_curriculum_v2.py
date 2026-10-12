"""Read-only independent checks of the rebuilt curriculum. No learner imports.
Run under WSL; data stays in /home/user/curriculum_v2. Source builders are not run.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
from validate_curriculum_grounding import episodes

ROOT = Path(__file__).resolve().parents[1]


def current(story):
    state = {}
    for s in story:
        if s.startswith("He "):
            return None
        m = re.fullmatch(r"(\w+) has the (\w+)\.", s)
        if m:
            state[m[2]] = m[1]; continue
        m = re.fullmatch(r"(\w+) (?:gave|handed|passed) the (\w+) to (\w+)\.", s)
        if m:
            state[m[2]] = m[3]; continue
        # The rebuilt generator declares these to be failed transfers. This
        # checker uses that declared controlled-language convention explicitly.
        m = re.fullmatch(r"(\w+) (?:tried to give|tried to hand|offered) the (\w+) to (\w+)\.", s)
        if m:
            state[m[2]] = m[1]; continue
        m = re.fullmatch(r"No one reported anything about the (\w+) for a while\.", s)
        if m:
            state.pop(m[1], None); continue
        m = re.fullmatch(r"Correction: (\w+), not (\w+), has the (\w+)\.", s)
        if m:
            state[m[3]] = m[1]; continue
        if re.fullmatch(r"Someone said \w+ has the \w+\.", s):
            continue
        if re.fullmatch(r"\w+ (?:is tired|is happy|went to the market|is reading|is waiting outside)\.", s):
            continue
        return None
    return state


def temporal_witness(story, question, gold):
    """Score only explicit named, gap-free predecessor histories.

    A failed attempt repeats an observation; it is not an ownership change.
    The giver in a successful transfer is visible immediately before its receiver.
    Always-claims additionally admit a strong disproof: a different named holder.
    Gapped predecessor histories are deliberately outside this narrow check.
    """
    before = re.fullmatch(r"Who had the (\w+) before (\w+)\?", question)
    always = re.fullmatch(r"Who has always had the (\w+)\?", question)
    if not (before or always) or current(story) is None:
        return None
    obj = (before or always)[1]
    holder = None
    predecessor = None
    witnessed = set()
    gap = False
    for sentence in story:
        if sentence == f"No one reported anything about the {obj} for a while.":
            gap = True
        m = re.fullmatch(r"(\w+) has the " + re.escape(obj) + r"\.", sentence)
        if m:
            holder = m[1]; witnessed.add(holder)
        m = re.fullmatch(r"(\w+) (gave|handed|passed) the " + re.escape(obj) + r" to (\w+)\.", sentence)
        if m:
            predecessor, holder = m[1], m[3]
            witnessed.update((predecessor, holder))
        m = re.fullmatch(r"(\w+) (tried to give|tried to hand|offered) the " + re.escape(obj) + r" to (\w+)\.", sentence)
        if m:
            holder = m[1]; witnessed.add(holder)
    if always:
        if gold != "unknown" and witnessed - {gold}:
            return dict(kind="always_counterexample", issue="another holder is explicitly observed", expected="not always " + gold)
        return None
    if gap:
        return dict(kind="before_gap_not_scored")
    if holder != before[2]:
        return None
    expected = predecessor or "unknown"
    return dict(kind="before_scored", expected=expected, error=gold != expected,
                issue="previous distinct holder, ignoring failed attempts")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", type=Path, nargs="?", default=Path("/home/user/curriculum_v2")); a = ap.parse_args()
    result = {"root": str(a.root), "events": {}}; temporal_issues = []
    for split in ("train", "test"):
        path = a.root / f"stage5/stage5_events_{split}.{'txt' if split=='train' else 'jsonl'}"
        def rows():
            if split == "train":
                for ep in episodes(path):
                    story, rest = ep.split("USER: ", 1); q, answer = rest.split("ASSISTANT: ", 1)
                    yield story.strip().splitlines(), q.strip(), answer.strip()
            else:
                for line in path.open():
                    for q, answer, _, story in json.loads(line)["qa"]:
                        yield story, q, answer
        stats = collections.Counter(); examples = []; temporal_examples = []
        for story, q, gold in rows():
            stats["total"] += 1
            witness = temporal_witness(story, q, gold)
            if witness:
                stats["temporal_" + witness["kind"]] += 1
                wrong = witness.get("error", witness["kind"] == "always_counterexample")
                stats["temporal_witness_errors"] += bool(wrong)
                if wrong:
                    temporal_issues.append(dict(split=split, index=stats["total"]-1, story=story, question=q, gold=gold, **witness))
                if wrong and len(temporal_examples) < 6:
                    temporal_examples.append(dict(story=story, question=q, gold=gold, **witness))
            # Strong leak witness: the queried object's history never appears
            # anywhere in the learner's observations.
            m = re.fullmatch(r"Who has always had the (\w+)\?", q)
            if m and not any(re.search(r"\b" + re.escape(m[1]) + r"\b", s) for s in story) and gold != "unknown":
                stats["unmentioned_object_history_target"] += 1
                if len(examples)<3: examples.append(dict(story=story, question=q, gold=gold, issue="no observation mentions the object"))
            state = current(story)
            if state is None:
                stats["skipped_story"] += 1; continue
            m = re.fullmatch(r"Who has the (\w+) now\?", q)
            n = re.fullmatch(r"Does (\w+) have the (\w+)\?", q)
            if m: predicted = state.get(m[1], "unknown")
            elif n: predicted = "unknown" if n[2] not in state else "true" if state[n[2]] == n[1] else "false"
            else: stats["temporal_not_scored"] += 1; continue
            stats["current_state_audited"] += 1
            stats["current_state_label_errors"] += predicted != gold
        result["events"][split] = {"counts": dict(stats), "temporal_leak_examples": examples,
                                    "temporal_witness_examples": temporal_examples,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    def key(story): return hashlib.sha256("\n".join(story).strip().encode()).hexdigest()
    tests = collections.defaultdict(list)
    for line in (a.root / "stage3/stage3_paraphrase_test.jsonl").open():
        r = json.loads(line)
        for q, gold, _, story in r["qa"]: tests[key(story)].append((q, gold, r.get("world_group")))
    training = collections.defaultdict(set)
    for ep in episodes(a.root / "stage3/stage3_paraphrase_train.txt"):
        s, rest = ep.split("USER: ",1); training[key(s.strip().splitlines())].add(rest.split("ASSISTANT:",1)[0].strip())
    overlaps = {}
    for filename in ("stage2/stage2_clevr_train.txt", "stage4/stage4_clevr_train.txt"):
        seen = set()
        for ep in episodes(a.root / filename):
            k = key(ep.split("USER: ",1)[0].strip().splitlines())
            if k in tests: seen.add(k)
        overlaps[filename] = len(seen)
    result["paraphrases"] = dict(test_worlds=len(tests), test_variants=dict(collections.Counter(map(len,tests.values()))),
        inconsistent_answers=sum(len({x[1] for x in xs})!=1 for xs in tests.values()),
        train_worlds=len(training), train_worlds_with_multiple_phrases=sum(len(x)>1 for x in training.values()), overlaps=overlaps)
    roles = collections.Counter()
    for line in (a.root / "stage6/stage6_grounded_dialogue_test.jsonl").open():
        r = json.loads(line)
        for turn in r.get("turns", []): roles[turn["role"]] += 1
    result["canonical_dialogue_roles"] = dict(roles)
    out = ROOT / "logs/curriculum_grounding/v2_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True);out.write_text(json.dumps(result, indent=2)+"\n")
    (out.parent/"v2_temporal_issues.jsonl").write_text("".join(json.dumps(r)+"\n" for r in temporal_issues))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
