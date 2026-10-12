"""Prepare raw QA-only bootstrap lessons and independent held-out scoring data.

The task ID selects the bAbI location pilot; it is not serialized to the native
learner. No entities, facts, question program or event labels are teaching inputs.
Source curriculum/builders are read-only. The grammar below is an evaluator only.
"""
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/bootstrap_grounding"
VERBS = ("moved to", "went to", "went back to", "journeyed to", "travelled to")
EVENT = re.compile(r"(\w+) (?:moved to|went to|went back to|journeyed to|travelled to) the (\w+)\.", re.I)


def truth(story, question):
    state = {}
    for sentence in story:
        m = EVENT.fullmatch(sentence)
        assert m, sentence
        a, b = (x.lower() for x in m.groups())
        state[a] = b
    q = re.fullmatch(r"Where is (\w+)\?", question, re.I)
    assert q, question
    return state[q[1].lower()], state


def write_episodes(path, rows, targets):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for i, row in enumerate(rows):
            f.write(f"{i}\t{len(row['story'])}\n")
            for sentence in row["story"]:
                f.write(sentence + "\n")
            f.write(row["question"] + "\n")
            if targets:
                f.write(row["answer"] + "\n")


def opaque(text):
    return re.sub(r"\w+", lambda m: "word" + hashlib.sha256(m[0].lower().encode()).hexdigest()[:12], text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    training = []
    path = ROOT / "data/babi/babi_train.jsonl"
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["task"] == 1:
            training.append(dict(story=r["passage"].strip().splitlines(), question=r["question"], answer=r["answer"]))
    held = []
    source = ROOT / "data/babi/babi_test.jsonl"
    for line in source.read_text().splitlines():
        r = json.loads(line)
        if r["task"] != 1:
            continue
        story, question = r["passage"].strip().splitlines(), r["question"]
        answer, state = truth(story, question)
        assert answer == r["answer"].lower()
        held.append(dict(group="babi", story=story, question=question, answer=answer, state=state))
    original = list(held)
    for i, r in enumerate(original):
        names = sorted(set(r["state"]) | set(r["state"].values()) |
                       {v.lower() for s in r["story"] for v in EVENT.fullmatch(s).groups()})
        mapping = {name: f"entity{i}x{j}" for j, name in enumerate(names)}
        rename = lambda s: re.sub(r"\w+", lambda m: mapping.get(m[0].lower(), m[0]), s)
        story, question = [rename(s) for s in r["story"]], rename(r["question"])
        answer, state = truth(story, question)
        held.append(dict(group="renamed", story=story, question=question, answer=answer, state=state))
    rng = random.Random(822)
    for i in range(120):
        names = [f"probe{i}z{j}" for j in range(8)]
        a, b, c, x, y, z, w, v = names
        def event(p, loc):
            return f"{p} {rng.choice(VERBS)} the {loc}."
        story = [event(a, x), event(b, z), event(a, y), event(c, y), event(b, w), event(c, v)]
        # Shared location, changed actor, unchanged actor and reversed time.
        for group, sentences, actor in [("changed", story, a), ("persistence", story, a),
                                         ("shared_location", story[:4], a), ("reverse_time", list(reversed(story)), a)]:
            if group == "persistence":
                sentences = [event(a, x)] + [event(b, t) for t in (x, y, z, w)] + [event(c, v)]
            q = f"Where is {actor}?"
            answer, state = truth(sentences, q)
            held.append(dict(group=group, story=sentences, question=q, answer=answer, state=state))
    for i in range(30):
        held.append(dict(group="untaught_verb", story=[f"someone{i} teleported into the somewhere{i}."],
                         question=f"Where is someone{i}?", answer=None, state=None))
    write_episodes(OUT / "train.txt", training, True)
    write_episodes(OUT / "test.txt", held, False)
    (OUT / "gold.json").write_text(json.dumps(held, indent=2) + "\n")
    variant = OUT / "opaque"
    variant.mkdir(exist_ok=True)
    hidden_train = [dict(story=[opaque(s) for s in r["story"]], question=opaque(r["question"]), answer=opaque(r["answer"])) for r in training]
    hidden_test = [dict(group=r["group"], story=[opaque(s) for s in r["story"]], question=opaque(r["question"]),
                       answer=opaque(r["answer"]) if r["answer"] else None,
                       state={opaque(k): opaque(v) for k,v in r["state"].items()} if r["state"] else None) for r in held]
    write_episodes(variant / "train.txt", hidden_train, True)
    write_episodes(variant / "test.txt", hidden_test, False)
    (variant / "gold.json").write_text(json.dumps(hidden_test, indent=2) + "\n")
    (OUT / "preparation.json").write_text(json.dumps(dict(training=len(training), test=len(held),
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, source)},
        protocol="Training: text + final answer only. Test: text only. Metadata/evaluator not linked into learner."), indent=2) + "\n")
    print(f"bootstrap training={len(training)} heldout={len(held)}")


if __name__ == "__main__":
    main()
