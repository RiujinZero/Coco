"""A raw byte curriculum with separate verified QA receipts, never rule labels."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/grounded_teaching"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    turns, receipts = [], []
    offset = 0
    for episode in range(16):
        training = episode < 8
        prefix = ("lesson" if training else "transfer") + str(episode)
        people = [prefix + f"person{i}" for i in range(4)]
        rooms = [prefix + f"room{i}" for i in range(4)]
        items = [prefix + f"item{i}" for i in range(3)]
        location = dict(zip(people, rooms)); owner = dict(zip(items, people))
        def raw(text, label):
            nonlocal offset
            offset += len(text.encode())
            turns.append(dict(episode=episode, training=training, text=text, label=label, offset=offset))
        def user(q): raw("\nUSER: " + q + "\nASSISTANT:", "question")
        def checks():
            for person in people: user(f"Where is {person}?")
            for item in items: user(f"Who has the {item} now?")
            # Membership distinguishes missing coverage from a spurious extra
            # self-relation that also makes a unique answer fail.
            user(f"Is {people[0]} at {people[0]}?")
            user(f"Does {items[1]} have the {items[1]}?")
        setup = [f"{p} moved to the {r}." for p, r in location.items()]
        setup += [f"{p} has the {o}." for o, p in owner.items()]
        raw("\n<|endofdialogue|>\n" + "".join("CONTEXT: " + s + "\n" for s in setup), "start")
        checks()
        target = people[1 + episode % 3]
        reports = [f"The {items[0]} was logged beside {target}.",
                   f"The location record for {people[0]} went dark.",
                   f"The whereabouts of {people[1]} shifted into {rooms[(episode % 2) + 2]}.",
                   f"The ownership record for {items[1]} went dark."]
        for report, text in enumerate(reports, 1):
            if report == 1: owner[items[0]] = target
            elif report == 2: location[people[0]] = "unknown"
            elif report == 3: location[people[1]] = rooms[(episode % 2) + 2]
            else: owner[items[1]] = "unknown"
            raw("\nENVIRONMENT: " + text + "\n", "report")
            if training:
                lesson = [(f"Where is {p}?", location[p]) for p in people]
                lesson += [(f"Who has the {o} now?", owner[o]) for o in items]
                lesson += [(f"Is {people[0]} at {people[0]}?", "unknown" if location[people[0]] == "unknown" else "no"),
                           (f"Does {items[1]} have the {items[1]}?", "unknown" if owner[items[1]] == "unknown" else "no")]
                for evidence, (question, answer) in enumerate(lesson):
                    receipts.append([offset, episode + 1, report, 0, evidence, 0, "add", "verified_teacher", question, answer])
            checks()
        if not training:
            # Learned correction constructions now contain report payloads that
            # were unknown when the session was bootstrapped.
            raw(f"\nUSER: replace the earliest active report about {items[0]} with The {items[0]} was logged beside {people[0]}\nASSISTANT:", "correction")
            owner[items[0]] = people[0];checks()
            raw(f"\nUSER: withdraw the latest active report about {people[0]}\nASSISTANT:", "correction")
            location[people[0]] = rooms[0];checks()
            raw(f"\nUSER: replace the latest active report about {items[1]} with The {items[1]} was logged beside {people[3]}\nASSISTANT:", "correction")
            owner[items[1]] = people[3];checks()
    (OUT / "turns.json").write_text(json.dumps(turns, indent=2) + "\n")
    with (OUT / "turns.inputs").open("wb") as f:
        f.write(f"{len(turns)}\n".encode())
        for turn in turns:
            data = turn["text"].encode();f.write(f"{len(data)}\n".encode() + data + b"\n")
    (OUT / "trainer.txt").write_bytes("".join(t["text"] for t in turns).encode())
    (OUT / "feedback.tsv").write_text("".join("\t".join(map(str, row)) + "\n" for row in receipts))
    print(f"Prepared {len(turns)} turns, {len(receipts)} verified answers, 8 teaching and 8 transfer episodes.")


if __name__ == "__main__": main()
