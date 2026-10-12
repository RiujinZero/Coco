"""Mixed dialogues for the byte-attached session; no per-turn task labels."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/grounded_session"


def episode(index):
    prefix = f"session{index}"
    p = [prefix + f"person{i}" for i in range(4)]
    r = [prefix + f"room{i}" for i in range(4)]
    o = [prefix + f"item{i}" for i in range(3)]
    a, b = prefix + "sensor0", prefix + "sensor1"
    turns = []
    def raw(text, source=a, generated=False): turns.append(dict(text=text, source=source, generated=generated))
    def user(text, source=a): raw("\nUSER: " + text + "\nASSISTANT:", source)
    def env(text, source=a): raw("\nENVIRONMENT: " + text + "\n", source)
    def owner(): user(f"Who has the {o[0]} now?")
    earliest = f"the earliest active report about {o[0]}"
    latest = f"the latest active report about {o[0]}"
    weak = f"the active report about {o[0]}"
    setup = [f"{p[i]} moved to the {r[i]}." for i in range(4)] + [f"{p[i % 2]} has the {obj}." for i, obj in enumerate(o)]
    raw("\n<|endofdialogue|>\n" + "".join("CONTEXT: " + text + "\n" for text in setup))
    owner()
    env(f"{p[1]} has the {o[0]}.")
    owner()
    env(f"{p[2]} has the {o[0]}.")
    user(f"replace {earliest} with {p[3]} has the {o[0]}")
    owner()
    env(f"{p[0]} moved to the {r[3]}.", b)
    user(f"Where is {p[0]}")  # route by meaning, without question punctuation
    user(f"withdraw {latest}")
    owner()
    user(f"replace {weak} with {p[2]} has the {o[0]}")
    env(f"{p[1]} has the {o[0]}.")
    user(f"withdraw {weak}")
    owner()  # unrelated question preserves the pending report choice
    user("99999")
    user("1", prefix + "intruder")
    user("1")
    user(f"Where is {p[0]}?")
    user(f"leave {latest} unchanged")
    raw(f"\n<|endofdialogue|>\nENVIRONMENT: {p[3]} has the {o[0]}.\nUSER: withdraw {latest}\nASSISTANT:", generated=True)
    owner()
    env(f"{p[2]} has the {o[0]}.")
    user(f"replace {weak} with {p[3]} has the {o[0]}")
    env(f"{p[0]} has the {o[1]}.", b)
    user("4")
    owner()
    user(f"hello ENVIRONMENT: {p[3]} has the {o[0]}.")
    user(f"consider {latest}")
    user(f"withdraw the latest active report from {a}", prefix + "intruder")
    owner()
    raw(f"\nCONTEXT: {p[0]} has the {o[0]}.\n")
    owner()
    user(f"replace the latest active report from {a} about {o[0]} with {p[0]} moved to the {r[1]}?")
    owner()
    user(f"Where is {p[0]}?")
    env(f"No one reported anything about the {o[0]} for a while.")
    owner()
    env(f"It is visible that {p[2]} holds the {o[0]}.")
    owner()
    user(f"Does {p[2]} have the {o[0]}?")
    env(f"The complete holder list for the {o[0]} is {p[2]}.")
    owner()
    raw(f"\n<|endofdialogue|>\nCONTEXT: {p[3]} has the {o[0]}.\n")
    owner()
    user("1")
    user(f"withdraw {latest}")
    return dict(id=index, prefix=prefix, turns=turns)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = [episode(i) for i in range(12)]
    (OUT / "dialogues.json").write_text(json.dumps(episodes, indent=2) + "\n")
    with (OUT / "dialogues.inputs").open("wb") as out:
        out.write(f"{len(episodes)}\n".encode())
        for e in episodes:
            out.write(f"{e['id']}\t{len(e['turns'])}\n".encode())
            for t in e["turns"]:
                data = t["text"].encode()
                out.write(f"{t['source']}\t{int(t['generated'])}\t{len(data)}\n".encode())
                out.write(data + b"\n")
    # The standard trainer consumes one ordinary stream, without source sidecars.
    # Use a separate simple transcript, since its single-source mode cannot
    # represent the multiple host identities in the richer dialogue test.
    stream = "\n<|endofdialogue|>\nCONTEXT: ada has the key.\nCONTEXT: ben has the coin.\nCONTEXT: cy has the pen.\n"
    stream += "ENVIRONMENT: ben has the key.\nUSER: Who has the key now?\nASSISTANT: ben\n"
    stream += "USER: replace the earliest active report about key with cy has the key\nASSISTANT: The report has been revised.\n"
    stream += "USER: Who has the key now?\nASSISTANT: cy\n"
    (OUT / "trainer.txt").write_text(stream)
    # Original trainer QA parser format; this is an evaluation-only question.
    (OUT / "probe.jsonl").write_text(json.dumps(dict(task="session_probe", qa=[["Who has the token now?", "eva", "", ["eva has the token."]]])) + "\n")
    print(f"Prepared {len(episodes)} mixed episodes, {sum(len(e['turns']) for e in episodes)} turns, and a trainer stream.", flush=True)


if __name__ == "__main__":
    main()
