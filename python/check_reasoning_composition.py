"""Audit the completed composition matrix and produce a compact evidence index."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    log = root / "logs"
    files = []
    def read(path):
        files.append(path)
        return json.loads(path.read_text(encoding="utf-8"))
    p = read(log / "reasoning_composition_verified/results.json")
    assert p["source_sha256"] == hashlib.sha256((root / "python/reasoning_composition_probe.py").read_bytes()).hexdigest()
    assert len(p["runs"]) == 20
    for r in p["runs"]:
        assert len(r["metrics"]) == 11
        if r["regime"] == "factorized_deep":
            assert all(v["correct"] == v["n"] == 100 for v in r["metrics"].values())
    summary = {}
    for seed in range(5):
        # Independent checks use facts directly, not the learned skill VM or the
        # generator's construction of the intended answer.
        for split in ("train", "test"):
            corpus = log / f"reasoning_flat_corpus/{split}_{seed}.txt"
            for line in corpus.read_text().splitlines():
                fields = list(map(int, line.split()))
                cue, a, b, relation, item_relation, typ, answer, count = fields[:8]
                assert len(fields) == 8 + count * 3
                facts = [tuple(fields[i:i+3]) for i in range(8, len(fields), 3)]
                def follow(entity):
                    visited = set()
                    while True:
                        assert entity not in visited
                        visited.add(entity)
                        successors = {dst for src, rel, dst in facts if src == entity and rel == relation}
                        if not successors:
                            return entity
                        assert len(successors) == 1
                        entity = next(iter(successors))
                def number(entity):
                    return len({dst for src, rel, dst in facts if src == entity and rel == item_relation})
                if cue == 0: expected = (0, follow(a))
                elif cue == 1: expected = (2, number(a))
                elif cue == 2: expected = (1, int(a == b))
                elif cue == 3:
                    expected = (2, number(follow(a)))
                    assert number(a) != expected[1], "count-through shortcut in benchmark"
                elif cue == 4: expected = (1, int(number(a) == number(b)))
                elif cue == 5: expected = (1, int(number(follow(a)) == number(follow(b))))
                else:
                    assert cue == 6
                    expected = (1, int(follow(a) == follow(b)))
                assert expected == (typ, answer)
        flat = read(log / f"reasoning_flat_native/{seed}.json")
        assert flat["seed"] == seed
        full = flat["conditions"]["twelve_examples"]
        assert len(full) == 7
        assert all(v["correct"] == v["n"] == 200 and v["found"] for v in full.values())
        # Bindings are required for holding TWO computed results, not for every
        # possible one-register composition. Check the intended distinction.
        ab = flat["conditions"]["no_bindings"]
        for family in ("follow", "count", "equal", "count_through"):
            assert ab[family]["correct"] == 200
        for family in ("equal_counts", "equal_counts_through", "equal_ends"):
            assert not ab[family]["found"] and ab[family]["correct"] == 0
        for mode in ("outcome", "monte_carlo", "replay", "broadcast", "broadcast_long", "shared_error", "random_reward", "stress"):
            d = read(log / f"reasoning_composition_native_verified/{seed}_{mode}.json")
            assert d["seed"] == seed
            rows = d["conditions"]["routes"]
            correct, n = sum(v["correct"] for v in rows.values()), sum(v["n"] for v in rows.values())
            if mode in ("outcome", "monte_carlo", "replay", "stress"):
                assert correct == n == (780 if mode == "stress" else 1100)
            summary.setdefault(mode, [0, 0])
            summary[mode][0] += correct
            summary[mode][1] += n
    source_paths = [
        "cpp/include/laminar/reasoning_runtime.h", "cpp/include/laminar/procedure_memory.h",
        "cpp/apps/reasoning_composition_probe.cpp", "cpp/apps/reasoning_flat_probe.cpp",
        "python/reasoning_composition_probe.py", "python/reasoning_flat_probe.py",
    ]
    audit = {"flat_native": {"correct": 7000, "n": 7000}, "graph_native": summary,
             "source_sha256": {s: hashlib.sha256((root / s).read_bytes()).hexdigest() for s in source_paths},
             "result_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (log / "reasoning_composition_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"flat_native": audit["flat_native"], "graph_native": summary, "result_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
