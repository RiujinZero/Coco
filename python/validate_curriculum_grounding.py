"""Read-only curriculum readiness audit and bounded native probe inputs.

No oracle files, source programs, supporting-fact indices or held-out answers are
exported to the learner. Gold is kept in a separate scoring file. The Stage 5
checker below is an evaluation-only text interpreter, never a learner adapter.
TextWorld is excluded while its builder is running. No source files are changed.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/curriculum"
OUT = ROOT / "logs/curriculum_grounding"
FILES = [
    "stage1/babi_s1_test.jsonl", "stage2/babi_s2_test.jsonl",
    "stage2/stage2_clevr_test.jsonl", "stage3/stage3_paraphrase_test.jsonl",
    "stage4/stage4_clevr_test.jsonl", "stage4/stage4_cogent_valA_test.jsonl",
    "stage4/stage4_cogent_valB_test.jsonl", "stage5/stage5_events_test.jsonl",
    "stage5/stage5_propara_test.jsonl", "stage5/stage5_babybench_test.jsonl",
]


def episodes(path):
    lines = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip() == "<|endofdialogue|>":
                yield "".join(lines)
                lines = []
            else:
                lines.append(line)
    if any(s.strip() for s in lines):
        raise ValueError(f"Unterminated episode: {path}")


def event_answer(story, question):
    """Only score unambiguous named-entity stories; decline any pronoun giver.

    Possession is exclusive in this environment. All objects' initial holders
    and every successful transfer are visible; failed attempts preserve state.
    Unknown is justified only if that state cannot be recovered from the text.
    """
    holder, history = {}, {}
    for text in story:
        if text.startswith("He "):
            return None, "pronoun_not_audited"
        m = re.fullmatch(r"(\w+) has the (\w+)\.", text)
        if m:
            p, obj = m.groups()
            holder[obj], history[obj] = p, [p]
            continue
        m = re.fullmatch(r"(\w+) (?:gave|handed|passed) the (\w+) to (\w+)\.", text)
        if m:
            p, obj, receiver = m.groups()
            if holder.get(obj) != p:
                return None, "inconsistent_giver"
            holder[obj] = receiver
            history[obj].append(receiver)
            continue
        m = re.fullmatch(r"Correction: (\w+), not (\w+), has the (\w+)\.", text)
        if m:
            p, wrong, obj = m.groups()
            if holder.get(obj) != p or p == wrong:
                return None, "inconsistent_correction"
            continue
        if re.fullmatch(r"\w+ (?:tried to give|tried to hand|offered) the \w+ to \w+\.", text):
            continue
        if re.fullmatch(r"\w+ did not take it\.", text):
            continue
        if re.fullmatch(r"\w+ (?:is tired|is happy|went to the market|is reading|is waiting outside)\.", text):
            continue
        return None, "unsupported_sentence"
    m = re.fullmatch(r"Who has the (\w+) now\?", question)
    if m:
        return holder.get(m[1], "unknown"), "retrieval"
    m = re.fullmatch(r"Does (\w+) have the (\w+)\?", question)
    if m:
        return ("unknown" if m[2] not in holder else "true" if holder[m[2]] == m[1] else "false"), "tf"
    m = re.fullmatch(r"Who had the (\w+) before (\w+)\?", question)
    if m and len(history.get(m[1], [])) > 1 and holder[m[1]] == m[2]:
        return history[m[1]][-2], "temporal"
    m = re.fullmatch(r"Who has always had the (\w+)\?", question)
    if m and len(history.get(m[1], [])) == 1:
        return holder[m[1]], "temporal"
    return None, "unsupported_question"


def check_events():
    report = {}
    for split in ("train", "test"):
        path = DATA / f"stage5/stage5_events_{split}.{'txt' if split == 'train' else 'jsonl'}"
        if split == "train":
            def rows():
                for episode in episodes(path):
                    story, rest = episode.split("USER: ", 1)
                    q, gold = rest.split("ASSISTANT: ", 1)
                    yield story.strip().splitlines(), q.strip(), gold.strip()
        else:
            def rows():
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        for q, gold, _, story in json.loads(line)["qa"]:
                            yield story, q, gold
        counts = collections.Counter()
        examples = []
        for i, (story, question, gold) in enumerate(rows()):
            counts["total"] += 1
            predicted, kind = event_answer(story, question)
            if predicted is None:
                counts["skipped_" + kind] += 1
                continue
            counts["audited"] += 1
            counts[kind + "_audited"] += 1
            if predicted.lower() != gold.lower():
                counts["label_contradictions"] += 1
                counts[f"gold_{gold.lower()}_expected_{predicted.lower()}"] += 1
                if len(examples) < 3:
                    examples.append(dict(episode=i, story=story, question=question, gold=gold, expected=predicted))
        report[split] = {"counts": dict(counts), "examples": examples}
    return report


def check_paraphrase_worlds():
    def world_key(sentences):
        return hashlib.sha256("\n".join(sentences).strip().encode()).hexdigest()
    test_worlds = set()
    with (DATA / "stage3/stage3_paraphrase_test.jsonl").open(encoding="utf-8") as f:
        for line in f:
            for q, gold, _, story in json.loads(line)["qa"]:
                test_worlds.add(world_key(story))
    train_phrases = collections.defaultdict(set)
    for ep in episodes(DATA / "stage3/stage3_paraphrase_train.txt"):
        story, rest = ep.split("USER: ", 1)
        question = rest.split("ASSISTANT:", 1)[0].strip()
        train_phrases[world_key(story.strip().splitlines())].add(question)
    overlaps = {}
    for filename in ("stage2/stage2_clevr_train.txt", "stage4/stage4_clevr_train.txt"):
        found = set()
        for ep in episodes(DATA / filename):
            story = ep.split("USER: ", 1)[0]
            key = world_key(story.strip().splitlines())
            if key in test_worlds:
                found.add(key)
        overlaps[filename] = len(found)
    return dict(test_worlds=len(test_worlds), test_worlds_seen_in_other_training=overlaps,
                train_unique_worlds=len(train_phrases),
                train_worlds_with_multiple_paraphrases=sum(len(v)>1 for v in train_phrases.values()))


def prepare(cap):
    OUT.mkdir(parents=True, exist_ok=True)
    golds, inventory, inputs, adaptation = [], {}, [], []
    for filename in FILES:
        source = DATA / filename
        seen = collections.Counter()
        digest = hashlib.sha256()
        with source.open("rb") as f:
            for line_number, raw in enumerate(f, 1):
                digest.update(raw)
                row = json.loads(raw)
                for q, gold, supporting, story in row["qa"]:
                    family = row["task"]
                    seen[family] += 1
                    if seen[family] > cap:
                        continue
                    index = len(inputs)
                    prompt = "\n<|endofdialogue|>\n" + "\n".join(story) + "\nUSER: " + q + "\n\nASSISTANT:"
                    inputs.append((index, prompt))
                    golds.append(dict(id=index, source=filename, line=line_number, family=family, gold=gold))
        inventory[filename] = {"sha256": digest.hexdigest(), "questions_by_family": dict(seen)}
    # A bounded feedback-acceptance check, using training answers only.
    # No surface translation, inferred facts, entity labels or event recipes.
    for i, ep in enumerate(episodes(DATA / "stage1/babi_s1_train.txt")):
        if i == 128:
            break
        prompt, target = ep.rsplit("ASSISTANT:", 1)
        adaptation.append((i, "\n<|endofdialogue|>\n" + prompt + "ASSISTANT:", target.strip()))
    def write_records(path, records):
        with path.open("wb") as f:
            for index, prompt in records:
                encoded = prompt.encode()
                f.write(f"{index}\t{len(encoded)}\n".encode() + encoded + b"\n")
    write_records(OUT / "heldout.inputs", inputs)
    write_records(OUT / "training.inputs", [(i, prompt) for i, prompt, target in adaptation])
    (OUT / "training.targets").write_text("\n".join(target for _, _, target in adaptation) + "\n", encoding="utf-8")
    (OUT / "heldout.gold.json").write_text(json.dumps(golds, indent=2) + "\n", encoding="utf-8")
    result = dict(cap_per_family=cap, total_sampled=len(inputs), inventory=inventory, events=check_events(),
                  paraphrase=check_paraphrase_worlds())
    # SCAN and stage6 use different record schemas; report that explicitly.
    result["excluded"] = {"textworld": "builder still running", "cogs_scan": "diagnostic semantic/action outputs, not grounded QA",
                          "stage6": "dialogue/fluency; environment role boundary separately inspected"}
    (OUT / "dataset_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sampled": len(inputs), "events": {s: v["counts"] for s, v in result["events"].items()}}, indent=2))


def score():
    golds = {r["id"]: r for r in json.loads((OUT / "heldout.gold.json").read_text())}
    summaries = collections.defaultdict(collections.Counter)
    seen = set()
    with (OUT / "native.tsv").open(encoding="utf-8") as f:
        assert next(f).strip() == "id\tstatus\tcomplete\tfacts\treply"
        for line in f:
            index, status, complete, facts, reply = line.rstrip("\n").split("\t")
            index = int(index)
            assert index not in seen
            seen.add(index)
            r = golds[index]
            for key in ("all", r["source"]):
                s = summaries[key]
                s["n"] += 1
                s["status_" + status] += 1
                s["complete_state"] += int(complete)
                s["answered"] += status in ("answer", "grounded_answer")
                s["exact_literal"] += reply.lower() == r["gold"].strip().lower()
                s["unsupported"] += status not in ("answer", "grounded_answer")
    assert seen == golds.keys(), "Missing or extra native records"
    report = {k: dict(v) for k, v in summaries.items()}
    (OUT / "native_scores.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--cap", type=int, default=32)
    args = parser.parse_args()
    score() if args.score else prepare(args.cap)
