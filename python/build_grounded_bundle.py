"""Expose the existing teaching corpora as a bundle with explicit initial text.

Only training inputs are read. Prefix-based initial worlds in the older probe
format are expanded here, outside the runtime/learner, into supplied CONTEXT text.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/grounded_session"


def initial(prefix):
    return ([f"{prefix}person{i} moved to the {prefix}room{i}." for i in range(4)]
            + [f"{prefix}person{i % 2} has the {prefix}item{i}." for i in range(3)])


def expand(source, destination, correction):
    lines = iter(source.read_text().splitlines())
    count = int(next(lines))
    with destination.open("w") as out:
        out.write(str(count) + "\n")
        for _ in range(count):
            id, prefix, n, lessons = next(lines).split("\t")
            context = initial(prefix)
            out.write(f"{id}\t{len(context)}\t{n}\t{lessons}\n")
            for text in context: out.write(text + "\n")
            for _ in range(int(n)): out.write(next(lines) + "\n")
            if correction: out.write(next(lines) + "\n")
            for _ in range(int(lessons)): out.write(next(lines) + "\n")
        assert next(lines, None) is None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expand(ROOT / "logs/history_reference/train.txt", OUT / "references.train.txt", False)
    expand(ROOT / "logs/correction_language/train.txt", OUT / "corrections.train.txt", True)
    paths = dict(micro=ROOT / "logs/reasoning_composition_verified",
                 bootstrap=ROOT / "logs/bootstrap_grounding/train.txt",
                 incremental=ROOT / "logs/incremental_grounding/train.txt",
                 dialogue=ROOT / "logs/dialogue_grounding/train.txt",
                 observation=ROOT / "logs/observation_language/train.txt",
                 references=OUT / "references.train.txt", corrections=OUT / "corrections.train.txt")
    (OUT / "bootstrap.paths").write_text("".join(f"{key}\t{path}\n" for key, path in paths.items()))
    print("Wrote grounded teaching bundle with explicit CONTEXT text:", OUT / "bootstrap.paths", flush=True)


if __name__ == "__main__":
    main()
