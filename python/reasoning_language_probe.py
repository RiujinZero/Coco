"""Text-only inference corpus plus separate audit annotations.

Training fact alignments teach sentence frames. Question procedures see only raw
text and verified answers, never task IDs or program traces. The task IDs and
triples in these files are for evaluation outside the native inference path.
"""
import argparse
import random
from pathlib import Path
from reasoning_flat_probe import question, FAMILIES

FACTS = (("{a} leads to {b}.", "{a} holds {b}."),
         ("{b} is reached from {a}.", "{b} is held by {a}."),
         ("From {a} the next stop is {b}.", "An item belonging to {a} is {b}."))
QUESTIONS = (
    ("Where does the path from {a} end?", "How many items does {a} hold?",
     "Are {a} and {b} the same entity?", "How many items are held at the end of the path from {a}?",
     "Do {a} and {b} hold equal numbers of items?",
     "Do the endpoints of the paths from {a} and {b} hold equal numbers of items?",
     "Do the paths from {a} and {b} end at the same entity?"),
    ("Name the last stop reached from {a}?", "Count the items held by {a}?",
     "Is {a} identical to {b}?", "Count the items at the last stop reached from {a}?",
     "Are the item counts for {a} and {b} equal?",
     "Are the item counts at the last stops reached from {a} and {b} equal?",
     "Are the last stops reached from {a} and {b} identical?"),
    ("Which endpoint is reachable from {a}?", "What is the number of items belonging to {a}?",
     "Does {a} denote the same entity as {b}?", "What is the item total at the endpoint reachable from {a}?",
     "Does the number of items belonging to {a} match the number belonging to {b}?",
     "Does the item total at the endpoint reachable from {a} match the total at the endpoint from {b}?",
     "Does the endpoint reachable from {a} match the endpoint from {b}?"))


def write_case(out, rng, seed, serial, cue, group, style, length, number, fact_style=None):
    q = question(rng, cue, serial, length, number)
    all_names = list(dict.fromkeys([q.a[1], q.b[1]] + [x for a, _, b in q.facts for x in (a, b)]))
    def letters(n):
        s = ""
        while n:
            s += chr(97 + n % 26); n //= 26
        return s or "a"
    # Fair alphabetic-token baseline; digits are tested as a separate stress arm.
    # Random prefixes also avoid manufacturing collisions in the legacy reader's
    # three-letter stem hashes. The suffix guarantees disjoint episode identities.
    suffix = "z" + letters(seed + 1) + "x" + letters(serial + 1)
    names = {v: letters(rng.getrandbits(40)) + suffix + "e" + letters(i + 1) for i, v in enumerate(all_names)}
    if group == "digits":
        names = {v: f"N{seed}x{serial}z{rng.randrange(1000000)}e{i}" for i, v in enumerate(all_names)}
    names = {k: v.lower() for k, v in names.items()}
    a, b = names[q.a[1]], names[q.b[1]]
    answer = names[q.answer[1]] if q.answer[0] == "entity" else str(int(q.answer[1])) if q.answer[0] == "int" else "yes" if q.answer[1] else "no"
    out.write(f"{group}\t{cue}\t{answer}\t{len(q.facts)}\n")
    out.write(QUESTIONS[style][cue].format(a=a, b=b) + "\n")
    for x, r, y in q.facts:
        relation = q.relations.index(r)
        fs = fact_style if fact_style is not None else rng.randrange(2)
        text = FACTS[fs][relation].format(a=names[x], b=names[y])
        # Case is deliberately varied after grounding; entity equality is lexical.
        if rng.randrange(3) == 0:
            text = text.upper()
        out.write(f"{names[x]}\t{relation}\t{names[y]}\t{text}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/reasoning_language_corpus")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-n", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        rng = random.Random(423000 + seed); serial = 0
        with (out / f"train_{seed}.txt").open("w") as f:
            for style in range(2):
                for cue in range(len(FAMILIES)):
                    for _ in range(12):
                        write_case(f, rng, seed, serial, cue, "train", style, rng.randint(1, 3), rng.randint(0, 3)); serial += 1
        with (out / f"adapt_{seed}.txt").open("w") as f:
            for cue in range(len(FAMILIES)):
                for _ in range(12):
                    write_case(f, rng, seed, serial, cue, "adapt", 2, rng.randint(1, 3), rng.randint(0, 3), 2); serial += 1
        with (out / f"test_{seed}.txt").open("w") as f:
            for group in ("known", "digits", "new_question", "new_facts"):
                for cue in range(len(FAMILIES)):
                    for _ in range(args.test_n if group == "known" else 20 if group == "digits" else 40):
                        style = 2 if group == "new_question" else rng.randrange(2)
                        write_case(f, rng, seed, serial, cue, group, style, rng.choice([4, 8, 12, 32]), rng.randint(4, 24), 2 if group == "new_facts" else None); serial += 1
    print(f"Wrote grounded training, separate adaptation, and held-out text corpora to {out}")


if __name__ == "__main__":
    main()
