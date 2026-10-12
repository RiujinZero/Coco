"""Before/after perceptual grounding and native stateful language evaluation."""
import argparse
import random
from pathlib import Path
from semantic_grounding_probe import world, answer, letters, term
from reasoning_language_probe import FACTS

# This language belongs to the environment/tutor, never the native learner.
PHRASES = ((0, 2, "{a} leads to {b}."), (0, 2, "{b} is reached from {a}."),
           (1, 3, "{a} holds {b}."), (1, 3, "{b} is held by {a}."),
           (0, 2, "{a} now connects to {b}."), (0, 2, "{b} is now reached from {a}."),
           (1, 3, "{a} picked up {b}."), (1, 3, "{b} was taken by {a}."),
           (1, 1, "{a} no longer holds {b}."), (1, 1, "{b} was released by {a}."),
           (2, 0, "{a} notices {b}."), (2, 0, "{b} is noticed by {a}."))


def update(facts, relation, effect, a, b):
    facts = set(facts)
    if effect == 1:
        facts.discard((a, relation, b))
    elif effect == 2:
        facts = {f for f in facts if f[:2] != (a, relation)}
    elif effect == 3:
        facts = {f for f in facts if f[1:] != (relation, b)}
    if effect != 1:
        facts.add((a, relation, b))
    return facts


def write_world(out, facts):
    out.write(str(len(facts)) + "\n")
    for a, r, b in sorted(facts): out.write(f"{a}\t{r}\t{b}\n")


def channel(relation, seed):
    return 1 - relation if seed % 2 and relation < 2 else relation


def observed(facts, seed):
    return {(a, channel(r, seed), b) for a, r, b in facts}


def make_sensor(out, seed, rng):
    serial = 0
    for phase in ("weak", "repeat", "contrast"):
        for relation, effect, phrase in PHRASES:
            for _ in range(12):
                serial += 1
                names = [letters(rng.getrandbits(40)) + "s" + letters(seed + 1) + "e" + letters(serial) + "v" + letters(i + 1) for i in range(7)]
                a, b, c, d, e, f, g = names
                if phase in ("weak", "repeat"):
                    before = {(a, relation, b)} if effect == 1 else set()
                else:
                    before = {(a, relation, c), (d, relation, b), (f, relation, g), (a, (relation + 1) % 3, e)}
                    if effect == 1:
                        before.discard((d, relation, b)); before.add((a, relation, b))
                after = update(before, relation, effect, a, b)
                out.write(f"{phase}\t{serial}\n{phrase.format(a=a, b=b)}\n")
                write_world(out, observed(before, seed)); write_world(out, observed(after, seed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/event_grounding_corpus")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-n", type=int, default=40)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        rng = random.Random(690000 + seed)
        with (out / f"sensors_{seed}.txt").open("w") as f: make_sensor(f, seed, rng)
        with (out / f"test_{seed}.txt").open("w") as f:
            serial = 100000
            for group in ("link_reassignment", "ownership_transfer", "retraction", "duplicate_event", "old_owner_release", "correction", "unrelated_relation", "unknown_event"):
                for _ in range(args.test_n):
                    serial += 1; number = rng.randint(4, 24)
                    a, b, initial = world(rng, seed, serial, rng.choice((4, 8, 12, 32)), number, "count_through")
                    initial = set(initial)
                    ea, eb = answer(term(4, 0), a, b, initial), answer(term(4, 1), a, b, initial)
                    # Ensure the link reassignment must change the count answer.
                    if answer(term(7, 0), ea, b, initial) == answer(term(7, 0), eb, b, initial):
                        initial.add((eb, 1, "extra" + letters(serial)))
                    items = sorted(o for s, r, o in initial if s == ea and r == 1)
                    pick = rng.choice(items)
                    steps = []
                    def event(indices, source, dest):
                        idx = rng.choice(indices); rel, effect, phrase = PHRASES[idx]
                        steps.append((rel, effect, source, dest, phrase.format(a=source, b=dest)))
                    if group in ("link_reassignment", "correction"):
                        event((4, 5), a, eb)
                        if group == "correction": event((4, 5), a, ea)
                    elif group in ("ownership_transfer", "old_owner_release"):
                        event((6, 7), eb, pick)
                        if group == "old_owner_release": event((8, 9), ea, pick)
                    elif group == "retraction": event((8, 9), ea, pick)
                    elif group == "duplicate_event":
                        event((2, 3), ea, pick); event((2, 3), ea, pick)
                    elif group == "unrelated_relation": event((10, 11), ea, pick)
                    else: steps.append((1, 3, eb, pick, f"{eb} mysteriously acquired {pick}."))
                    # Old-owner release is scored by the NEW owner's count as
                    # well as exact state, so an unscoped retraction cannot pass.
                    target = b if group == "old_owner_release" else a
                    question = f"What is item count of endpoint of {target}?"
                    f.write(f"{group}\t{len(initial)}\t{len(steps)}\n{question}\n")
                    shuffled = list(initial); rng.shuffle(shuffled)
                    for s, r, o in shuffled:
                        text = FACTS[rng.randrange(2)][r].format(a=s, b=o)
                        f.write(f"{s}\t{channel(r, seed)}\t{o}\t{text}\n")
                    state = initial
                    f.write(str(answer(term(7, term(4, 0)), target, b, state)) + "\n")
                    for rel, effect, source, dest, text in steps:
                        state = update(state, rel, effect, source, dest)
                        f.write(f"{channel(rel, seed)}\t{effect}\t{source}\t{dest}\t{text}\n")
                        f.write(("_" if group == "unknown_event" else str(answer(term(7, term(4, 0)), target, b, state))) + "\n")
                        write_world(f, observed(state, seed))
    print(f"Wrote sensory transition curriculum and stateful dialogue tests to {out}")


if __name__ == "__main__": main()
