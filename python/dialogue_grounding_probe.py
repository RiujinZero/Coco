"""Prepare raw environment-message QA teaching and untouched canonical dialogues.

The interpreter here supplies verified training answers and scoring targets only.
Native evaluation receives the original turn stream and extra questions, never
fact annotations. Source curriculum is read-only.
"""
import collections
import hashlib
import json
from pathlib import Path
import re
from bootstrap_grounding_probe import write_episodes
from validate_curriculum_grounding import episodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/dialogue_grounding"
V2 = Path("/home/user/curriculum_v2/stage6")
HAS = re.compile(r"(\w+) has the (\w+)\.")
SUCCESS = re.compile(r"(\w+) gives the (\w+) to (\w+)\. (\w+) takes it\.")
FAILURE = re.compile(r"(\w+) offers the (\w+) to (\w+), but (\w+) does not take it\.")


def observe(owners, text):
    m = HAS.fullmatch(text)
    if m:
        owners[m[2].lower()] = m[1].lower()
        return {m[1].lower()}, "context"
    m = SUCCESS.fullmatch(text) or FAILURE.fullmatch(text)
    assert m and m[3] == m[4] and owners[m[2].lower()] == m[1].lower(), text
    kind = "success" if SUCCESS.fullmatch(text) else "failure"
    if kind == "success":
        owners[m[2].lower()] = m[3].lower()
    return {m[1].lower(), m[3].lower()}, kind


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    teaching = []
    counts = collections.Counter()
    for i, episode in enumerate(episodes(V2 / "stage6_grounded_dialogue_train.txt")):
        if i >= 32:
            break
        story, owners, people = [], {}, set()
        original_question = None
        for line in episode.strip().splitlines():
            role, text = line.split(": ", 1) if ": " in line else ("CONTEXT", line)
            if role in ("CONTEXT", "ENVIRONMENT"):
                names, kind = observe(owners, text)
                people |= names
                story.append(text)
                counts["training_" + kind] += 1
                if role == "ENVIRONMENT":
                    # Query all objects and people to constrain consequences,
                    # including unchanged facts and zero/count alternatives.
                    for obj, owner in sorted(owners.items()):
                        teaching.append(dict(story=list(story), question=f"Who has the {obj} now?", answer=owner, source="verified_consequence"))
                    for person in sorted(people):
                        teaching.append(dict(story=list(story), question=f"What is the number of items held by {person}?", answer=str(sum(p == person for p in owners.values())), source="verified_consequence"))
                    # Joint consequences also constrain preservation of the old
                    # relation. Possession-only feedback says nothing about
                    # candidate side effects on locations.
                    entities = sorted(people | set(owners))
                    locations = {name: f"shelter{j}" for j, name in enumerate(entities)}
                    contrasted = story[:-1] + [f"{a} moved to the {b}." for a, b in locations.items()] + [text]
                    for name, room in locations.items():
                        teaching.append(dict(story=contrasted, question=f"Where is {name}?", answer=room, source="preservation_contrast"))
                    for name in list(locations) + list(locations.values()):
                        teaching.append(dict(story=contrasted, question=f"What is the number of destinations for {name}?", answer="1" if name in locations else "0", source="preservation_contrast"))
            elif role == "USER":
                original_question = text if text.endswith("?") else None
            elif role == "ASSISTANT" and original_question:
                obj = re.fullmatch(r"Who has the (\w+) now\?", original_question)[1].lower()
                assert owners[obj] == text.lower()
                teaching.append(dict(story=list(story), question=original_question, answer=text, source="original_qa"))
                counts["original_training_qa"] += 1
                original_question = None
    write_episodes(OUT / "train.txt", teaching, True)
    (OUT / "teacher_train.json").write_text(json.dumps(teaching, indent=2) + "\n")
    records = [json.loads(line) for line in (V2 / "stage6_grounded_dialogue_test.jsonl").read_text().splitlines()]
    with (OUT / "test.txt").open("w") as out:
        for r in records:
            objects = sorted(HAS.fullmatch(t["text"])[2].lower() for t in r["turns"] if t["role"] == "CONTEXT")
            out.write(f"{r['dialogue_id']}\t{len(r['turns'])}\t{len(objects)}\n")
            for obj in objects:
                out.write(f"Who has the {obj} now?\n")
            owners = {}
            for t in r["turns"]:
                out.write(f"{t['role']}\t{t['text']}\n")
                counts["test_" + t["role"].lower()] += 1
                if t["role"] in ("CONTEXT", "ENVIRONMENT"):
                    _, kind = observe(owners, t["text"])
                    if t["role"] == "ENVIRONMENT":
                        counts["test_" + kind] += 1
                        counts["before_after_queries"] += 2 * len(objects)
    (OUT / "gold.json").write_text(json.dumps(records, indent=2) + "\n")
    meta = dict(teaching=len(teaching), dialogues=len(records), counts=dict(counts), source_sha256={})
    for name in ("stage6_grounded_dialogue_train.txt", "stage6_grounded_dialogue_test.jsonl", "stage6_grounded_dialogue_before_reveal_test.jsonl"):
        meta["source_sha256"][name] = hashlib.sha256((V2 / name).read_bytes()).hexdigest()
    (OUT / "preparation.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
