"""Record exact five-action compatibility for the frozen demonstration build."""
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path("logs/goal_demonstration_fitting_v1")
binary = root / "five_action_snapshot"
result = subprocess.run([str(binary)], capture_output=True, check=True)
assert not result.stderr, result.stderr
output = root / "five_action_snapshot.stdout"
output.write_bytes(result.stdout)
baseline = Path("logs/mission_fastpath_validation_v2/after.txt")
assert result.stdout == baseline.read_bytes(), "five-action snapshot differs"
record = json.loads((root / "validation.json").read_text())
record["five_action_parity"] = {
    "passed": True,
    "scope": "Exact 27-episode traces, weights, values, consolidation and correction snapshot.",
    "source_sha256": sha(root / "five_action_snapshot.cpp"),
    "binary_sha256": sha(binary),
    "stdout_sha256": sha(output),
    "baseline": str(baseline),
    "baseline_sha256": sha(baseline),
}
for path in (root / "validation.json", Path("docs/GOAL_DEMONSTRATION_FITTING_VALIDATION.json")):
    path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record["five_action_parity"], indent=2))
