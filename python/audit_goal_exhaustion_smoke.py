"""Bounded receipt/flag audit of the frozen warm long-task reference variant."""
import hashlib
import json
from pathlib import Path

root = Path("logs/goal_long_warm_adapter_v2")
out = Path("logs/goal_long_warm_exhaustion_independent_audit_v1/audit.json")
assert not out.exists(), "refusing to overwrite earlier audit"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((root / "source_manifest.json").read_text())
for path, expected in manifest.items():
    assert digest(root / "source" / path) == expected, path
events = [json.loads(line) for line in (root / "warm_trace.jsonl").open()]
# Receipts are serialized after each mission. Pair their increasing owned token
# order with that mission's paid agenda reasons, checking candidate and a fixed
# token-minus-clock offset, before inspecting subsequent observations.
receipts, positions, offsets = {}, {}, {}
for row in events:
    if row["event"] == "paid_computation":
        key = row["seed"], row["arrival"], row["learning"]
        receipts.setdefault(key, []).append(row)
chronological = []
for row in events:
    if row["event"] == "paid_computation":
        continue
    chronological.append(row)
    if row["event"] == "decision" and row["operation"] == 1 and row["observation"][29]:
        key = row["seed"], row["arrival"], row["learning"]
        index = positions.get(key, 0)
        receipt = receipts[key][index]
        assert receipt["candidate"] == row["inspected_before"]
        offset = receipt["token"] - row["clock"]
        assert offsets.setdefault(key, offset) == offset
        positions[key] = index + 1
        chronological.append(receipt)
assert all(positions[key] == len(rows) for key, rows in receipts.items())
last, counts = {}, {}
flags = 0
for row in chronological:
    key = row["seed"], row["arrival"], row["learning"]
    counts.setdefault(key, {"decisions": 0, "motors": 0, "global_cap_exhaustions": 0, "max_derived_allowance": 0, "paid_computations": 0})
    total = counts[key]
    if row["event"] == "paid_computation":
        last[key, row["candidate"]] = row
        total["paid_computations"] += 1
        assert row["cost"] == 0 and row["plan_allowance"] <= row["global_plan_limit"] == 4096
        if row["parent"]:
            total["max_derived_allowance"] = max(total["max_derived_allowance"], row["plan_allowance"])
            total["global_cap_exhaustions"] += row["plan_allowance"] == 4096 and row["plan_status"] == "search_budget"
    elif row["event"] == "decision":
        total["decisions"] += 1
        total["motors"] += len(row["actions"])
        o = row["observation"]
        if o[29] and o[15]:
            previous = last[key, row["inspected_before"]]
            assert o[5] and o[10] and previous["plan_allowance"] == previous["global_plan_limit"] == 4096
            assert previous["plan_status"] == "search_budget" and row["operation"] == 5
            flags += 1
summary = json.loads((root / "warm_summary.json").read_text())
for row in summary["reference_smokes"]:
    total = counts[row["seed"], row["arrival"], row["semantic_learning"]]
    for field in ("decisions", "motors", "global_cap_exhaustions", "max_derived_allowance"):
        assert total[field] == row[field], field
    assert row["policy_updates"] == 0 and row["weights_unchanged"]
result = {"passed": True, "scope": "Paid allowance/feature use only; evaluator reference, not learned-policy acquisition",
          "flagged_skips_with_prior_actual_global_receipt": flags,
          "paid_computations": sum(x["paid_computations"] for x in counts.values()),
          "cases": [{"seed": k[0], "arrival": k[1], "learning": k[2], **v} for k, v in counts.items()],
          "source_files_verified": len(manifest),
          "source_manifest_sha256": digest(root / "source_manifest.json"),
          "trace_sha256": digest(root / "warm_trace.jsonl"), "auditor_sha256": digest(Path(__file__)),
          "limit": "Checks every true exhaustion flag against an earlier immutable actual-allowance receipt; does not reconstruct every false cache-validity bit or rerun primitive physics."}
out.parent.mkdir(parents=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
