"""Prepare request satisfaction feedback; never export correct actions to Coco.

Original Stage 6 requests/outcomes supply the first lessons. Wrong-recipient
contrasts supply the missing distinguishing evidence. The curriculum is read-only.
"""
import collections
import hashlib
import json
from pathlib import Path
import re
from validate_curriculum_grounding import episodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/request_grounding"
V2 = Path("/home/user/curriculum_v2/stage6")
REQUEST = re.compile(r"Please have (\w+) give the (\w+) to (\w+)\.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    natural, contrasts, aliases, moves = [], [], [], []
    for i, episode in enumerate(episodes(V2 / "stage6_grounded_dialogue_train.txt")):
        if i >= 32:
            break
        turns = [line.split(": ", 1) if ": " in line else ("CONTEXT", line) for line in episode.strip().splitlines()]
        names = set()
        for role, text in turns:
            match = REQUEST.fullmatch(text)
            if role == "USER" and match:
                names.update(match.groups())
        outsider = f"outsidertrain{i}"
        names.add(outsider)
        # Established location meanings identify all referents and provide an
        # unrelated relation against which a request must also be discriminated.
        story = [f"{name} moved to the shelter{j}." for j, name in enumerate(sorted(names))]
        request = None
        for role, text in turns:
            if role == "USER":
                request = text if REQUEST.fullmatch(text) else None
            elif role in ("CONTEXT", "ENVIRONMENT"):
                before = list(story)
                story.append(text)
                if role != "ENVIRONMENT":
                    continue
                a, obj, target = REQUEST.fullmatch(request).groups()
                satisfied = " takes it." in text
                natural.append(dict(phase="original", request=request, story=list(story), satisfied=satisfied))
                wrong = f"{a} gives the {obj} to {outsider}. {outsider} takes it."
                contrasts.append(dict(phase="wrong_recipient", request=request, story=before+[wrong], satisfied=False))
                aliases.append(dict(phase="alias", request=f"Make sure {target} is holding the {obj}.", story=list(story), satisfied=satisfied))
                aliases.append(dict(phase="alias_contrast", request=f"Make sure {target} is holding the {obj}.", story=before+[wrong], satisfied=False))
    for i in range(12):
        a, b, obj = f"traveler{i}", f"companion{i}", f"package{i}"
        initial = [f"{a} moved to the origin{i}.", f"{b} moved to the destination{i}.", f"{a} has the {obj}."]
        request = f"Please have {a} go to destination{i}."
        for satisfied in (False, True):
            story = initial + ([f"{a} moved to the destination{i}."] if satisfied else [])
            moves.append(dict(phase="movement", request=request, story=story, satisfied=satisfied))
    lessons = natural + contrasts + aliases + moves
    with (OUT / "train.txt").open("w") as out:
        for i, row in enumerate(lessons):
            out.write(f"{i}\t{row['phase']}\t{len(row['story'])}\t{int(row['satisfied'])}\n{row['request']}\n")
            out.write("\n".join(row["story"])+"\n")
    (OUT / "teacher_train.json").write_text(json.dumps(lessons, indent=2)+"\n")
    # The full unfiltered canonical test is retained verbatim, with no targets
    # attached to individual request turns in the native input.
    records = [json.loads(line) for line in (V2 / "stage6_grounded_dialogue_test.jsonl").read_text().splitlines()]
    with (OUT / "canonical.txt").open("w") as out:
        for r in records:
            out.write(f"{r['dialogue_id']}\t{len(r['turns'])}\n")
            for t in r["turns"]:
                out.write(f"{t['role']}\t{t['text']}\n")
    summary = dict(lessons=len(lessons), phases=dict(collections.Counter(r["phase"] for r in lessons)),
                   canonical_dialogues=len(records), source_sha256={})
    for name in ("stage6_grounded_dialogue_train.txt", "stage6_grounded_dialogue_test.jsonl"):
        summary["source_sha256"][name] = hashlib.sha256((V2/name).read_bytes()).hexdigest()
    (OUT / "preparation.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
