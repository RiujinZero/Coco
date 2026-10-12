"""Delayed corrections to earlier episodes; inputs contain no meaning labels."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/grounded_archive"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    turns, receipts = [], []
    offset = 0
    first_lesson = []
    first_people = first_rooms = first_items = None
    for episode in range(1, 13):
        prefix = f"archive{episode}"
        people = [prefix + f"person{i}" for i in range(4)]
        rooms = [prefix + f"room{i}" for i in range(4)]
        items = [prefix + f"item{i}" for i in range(3)]
        locations, owners = dict(zip(people, rooms)), dict(zip(items, people))
        report = 0
        def raw(text):
            nonlocal offset
            offset += len(text.encode());turns.append(dict(episode=episode, text=text, offset=offset))
        def user(text): raw("\nUSER: " + text + "\nASSISTANT:")
        def checks():
            for p in people: user(f"Where is {p}?")
            for o in items: user(f"Who has the {o} now?")
        def env(text):
            nonlocal report
            report += 1;raw("\nENVIRONMENT: " + text + "\n");return report
        def receipt(ep, rec, evidence, question, answer, op="add", previous=0):
            receipts.append([offset, ep, rec, 0, evidence, previous, op, "verified_teacher", question, answer])
        setup = [f"{p} moved to the {r}." for p, r in locations.items()] + [f"{p} has the {o}." for o, p in owners.items()]
        if episode == 12:
            # An acquired frame in initial context must also be replayable.
            setup.append(f"The {items[2]} was logged beside {people[0]}.");owners[items[2]] = people[0]
        raw("\n<|endofdialogue|>\n" + "".join("CONTEXT: " + s + "\n" for s in setup));checks()
        target = people[2 if episode <= 2 else 1 + episode % 3]
        record = env(f"The {items[0]} was logged beside {target}.")
        if episode == 1:
            first_people, first_rooms, first_items = people, rooms, items
            names = people + rooms + items
            # A plausible but mistaken teacher describes unchanged ownership.
            # The broad QA contrasts make the projected pre-repair state explicit.
            for a in names:
                for relation in (0, 1):
                    mapping = locations if relation == 0 else owners
                    first_lesson.append((f"Where is {a}?" if relation == 0 else f"Who has the {a} now?", mapping.get(a, "unknown")))
                    for b in names:
                        q = f"Is {a} at {b}?" if relation == 0 else f"Does {b} have the {a}?"
                        first_lesson.append((q, "unknown" if a not in mapping else "yes" if mapping[a] == b else "no"))
            for evidence, (q, a) in enumerate(first_lesson): receipt(episode, record, evidence, q, a)
        else:
            owners[items[0]] = target
            if episode <= 8: receipt(episode, record, 0, f"Who has the {items[0]} now?", target)
        checks()
        if episode == 2:
            env(f"{people[1]} has the {items[0]}.");owners[items[0]] = people[1];checks()
            raw("\n")
            # Repair every answer affected by the mistaken ownership lesson.
            for evidence, (q, a) in enumerate(first_lesson):
                corrected = a
                if q == f"Who has the {first_items[0]} now?": corrected = first_people[2]
                if q == f"Does {first_people[0]} have the {first_items[0]}?": corrected = "no"
                if q == f"Does {first_people[2]} have the {first_items[0]}?": corrected = "yes"
                if corrected != a: receipt(1, 1, evidence, q, corrected, "revise")
            checks()
        if episode >= 2:
            gap = env(f"The location record for {people[0]} went dark.");locations[people[0]] = "unknown"
            if episode <= 8:
                for i, p in enumerate(people): receipt(episode, gap, i, f"Where is {p}?", locations[p])
                receipt(episode, gap, 4, f"Is {people[0]} at {people[0]}?", "unknown")
            checks()
        if episode in (9, 12):
            raw("\n")
            bad_q = f"Where is {first_people[0]}?"
            receipt(1, 1, 999, bad_q, first_rooms[1], "add" if episode == 9 else "revise", 0 if episode == 9 else 1)
            checks()
            raw("\n");receipt(1, 1, 999, bad_q, "unknown", "withdraw", 0 if episode == 9 else 2);checks()
        if episode >= 9:
            user(f"replace the earliest active report about {items[0]} with The {items[0]} was logged beside {people[3]}")
            owners[items[0]] = people[3];checks()
            user(f"withdraw the latest active report about {people[0]}");locations[people[0]] = rooms[0];checks()
    (OUT / "turns.json").write_text(json.dumps(turns, indent=2) + "\n")
    (OUT / "trainer.txt").write_bytes("".join(t["text"] for t in turns).encode())
    (OUT / "feedback.tsv").write_text("".join("\t".join(map(str, row)) + "\n" for row in receipts))
    with (OUT / "turns.inputs").open("wb") as f:
        f.write(f"{len(turns)}\n".encode())
        for t in turns:
            data = t["text"].encode();f.write(f"{len(data)}\n".encode() + data + b"\n")
    print(f"Prepared {len(turns)} turns and {len(receipts)} receipts over twelve episodes.")


if __name__ == "__main__": main()
