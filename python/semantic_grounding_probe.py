"""Learn shared language fragments from answer-induced individual procedures.

No complete composite question is present in training. AST annotations are for
the independent scorer only; native training receives text and final answers.
"""
import argparse
import random
from pathlib import Path
from reasoning_language_probe import FACTS

WRAPS = ("What is {}?", "Give me {}?", "Tell me {}?")
FOLLOWS = ("endpoint of {}", "last stop reachable from {}", "destination reached from {}", "terminus accessible from {}")
COUNTS = ("item count of {}", "number of items held by {}", "quantity of objects belonging to {}", "inventory size of {}")
EQUALS = ("Are {} and {} equal?", "Is {} identical to {}?", "Does {} match {}?")


def term(op, *args):
    return (op, *sorted(args, key=key)) if op == 8 else (op, *args)


def key(tree):
    return f"${tree}" if isinstance(tree, int) else str(tree[0]) + "(" + "".join(key(x) + "," for x in tree[1:]) + ")"


def letters(n):
    s = ""
    while n:
        s += chr(97 + n % 26); n //= 26
    return s or "a"


def world(rng, seed, serial, length, number, mode="independent"):
    uid = 0
    def fresh():
        nonlocal uid
        uid += 1
        return letters(rng.getrandbits(40)) + "z" + letters(seed + 1) + "x" + letters(serial + 1) + "e" + letters(uid)
    a, b, ea, eb = (fresh() for _ in range(4)); same = bool(rng.randrange(2))
    if mode == "equal" and same:
        b = a; eb = ea
    if mode == "equal_ends" and same:
        eb = ea
    facts = []
    made = set()
    for source, end in ((a, ea), (b, eb)):
        if source in made:
            continue
        made.add(source)
        path = [source] + [fresh() for _ in range(length - 1)] + [end]
        facts.extend((x, 0, y) for x, y in zip(path, path[1:]))
    counts = {x: rng.randint(0, max(4, number + 2)) for x in (a, b, ea, eb)}
    if mode == "count":
        counts[a] = number
    elif mode == "count_through":
        counts[ea] = number; counts[a] = number + 1
    elif mode == "equal_counts":
        counts[a] = number; counts[b] = number if same else number + 1
    elif mode == "equal_counts_through":
        counts[ea] = number; counts[eb] = number if same else number + 1
    elif mode == "cross_counts":
        # The paired attachment query has the opposite truth value on this world.
        counts[ea] = number; counts[b] = number if same else number + 1
        counts[a] = number + 2; counts[eb] = number + 3 if same else number + 2
    for src, count in counts.items():
        for _ in range(count):
            item = fresh(); facts += [(src, 1, item)] * rng.randint(1, 3)
    for _ in range(4):
        facts.append((fresh(), rng.randrange(2), fresh()))
    rng.shuffle(facts)
    return a, b, facts


def answer(tree, a, b, facts):
    if isinstance(tree, int):
        return (a, b)[tree]
    values = [answer(x, a, b, facts) for x in tree[1:]]
    if tree[0] == 4:
        x = values[0]; edges = {s: o for s, r, o in facts if r == 0}
        seen = set()
        while x in edges:
            assert x not in seen
            seen.add(x); x = edges[x]
        return x
    if tree[0] == 7:
        return len({o for s, r, o in facts if r == 1 and s == values[0]})
    return values[0] == values[1]


def write(out, rng, seed, serial, group, question, meaning, mode, length, number, existing=None):
    a, b, facts = existing or world(rng, seed, serial, length, number, mode)
    text = question.format(a=a, b=b)
    value = answer(meaning, a, b, facts)
    gold = ("yes" if value else "no") if isinstance(value, bool) else str(value)
    if group == "unknown_words":
        gold = "_"
    out.write(f"{group}\t{gold}\t{key(meaning)}\t{len(facts)}\n{text}\n")
    for s, r, o in facts:
        phrase = FACTS[rng.randrange(2)][r].format(a=s, b=o)
        out.write(f"{s}\t{r}\t{o}\t{phrase}\n")
    return a, b, facts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/semantic_grounding_corpus")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-n", type=int, default=100)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        rng = random.Random(550000 + seed); serial = 0
        train = []
        for wrapper in WRAPS:
            train.append((wrapper.format("{a}"), 0, "independent"))
        for op, phrases, mode in ((4, FOLLOWS, "independent"), (7, COUNTS, "count")):
            for phrase_id, phrase in enumerate(phrases[:3]):
                for wrapper_id, wrapper in enumerate(WRAPS):
                    if wrapper_id != phrase_id:
                        train.append((wrapper.format(phrase.format("{a}")), term(op, 0), mode))
        for comparison in EQUALS:
            train.append((comparison.format("{a}", "{b}"), term(8, 0, 1), "equal"))
        with (out / f"train_{seed}.txt").open("w") as f:
            for question, meaning, mode in train:
                for _ in range(12):
                    write(f, rng, seed, serial, "train", question, meaning, mode, rng.randint(1, 3), rng.randint(0, 3)); serial += 1
        with (out / f"adapt_{seed}.txt").open("w") as f:
            for op, phrase, mode in ((4, FOLLOWS[3], "independent"), (7, COUNTS[3], "count")):
                for wrapper in WRAPS[:2]:
                    for _ in range(12):
                        write(f, rng, seed, serial, "adapt", wrapper.format(phrase.format("{a}")), term(op, 0), mode, rng.randint(1, 3), rng.randint(0, 3)); serial += 1
        with (out / f"test_{seed}.txt").open("w") as f:
            for op, phrases, mode in ((4, FOLLOWS, "independent"), (7, COUNTS, "count")):
                for phrase_id, phrase in enumerate(phrases[:3]):
                    for _ in range(30):
                        write(f, rng, seed, serial, "new_wording", WRAPS[phrase_id].format(phrase.format("{a}")), term(op, 0), mode, rng.choice((4, 8, 12, 32)), rng.randint(4, 24)); serial += 1
            for group in ("count_through", "equal_counts", "equal_counts_through", "equal_ends", "cross_counts", "new_alias", "unknown_words"):
                for _ in range(args.test_n if group != "unknown_words" else 30):
                    follow = FOLLOWS[3] if group == "new_alias" else rng.choice(FOLLOWS[:3])
                    count = COUNTS[3] if group == "new_alias" else rng.choice(COUNTS[:3])
                    end_a, end_b = follow.format("{a}"), follow.format("{b}")
                    count_a, count_b = count.format("{a}"), count.format("{b}")
                    end_count_a, end_count_b = count.format(end_a), count.format(end_b)
                    mode = group
                    if group in ("count_through", "unknown_words"):
                        question = rng.choice(WRAPS).format(end_count_a)
                        meaning = term(7, term(4, 0)); mode = "count_through"
                        if group == "unknown_words": question = question.replace("?", " approximately?")
                    elif group == "equal_counts":
                        question = rng.choice(EQUALS).format(count_a, count_b); meaning = term(8, term(7, 0), term(7, 1))
                    elif group in ("equal_counts_through", "new_alias"):
                        question = rng.choice(EQUALS).format(end_count_a, end_count_b); meaning = term(8, term(7, term(4, 0)), term(7, term(4, 1))); mode = "equal_counts_through"
                    elif group == "equal_ends":
                        question = rng.choice(EQUALS).format(end_a, end_b); meaning = term(8, term(4, 0), term(4, 1))
                    else:
                        comparison = rng.choice(EQUALS)
                        question = comparison.format(end_count_a, count_b); meaning = term(8, term(7, term(4, 0)), term(7, 1))
                    sample = write(f, rng, seed, serial, group, question, meaning, mode, rng.choice((4, 8, 12, 32)), rng.randint(4, 24)); serial += 1
                    if group == "cross_counts":
                        reverse = comparison.format(count_a, end_count_b)
                        write(f, rng, seed, serial, "attachment_pair", reverse, term(8, term(7, 0), term(7, term(4, 1))), "cross_counts", 0, 0, sample); serial += 1
    print(f"Wrote individual-question training and withheld composition tests to {out}")


if __name__ == "__main__": main()
