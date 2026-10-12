"""Freeze and validate the mixed raw-skill adapter; no training campaign."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path("logs/goal_mixed_skill_extension_v1")
assert not root.exists(), "refusing to overwrite frozen evidence"
root.mkdir(parents=True)
files = list(Path("cpp/include").rglob("*.h")) + [Path(path) for path in (
    "cpp/apps/goal_control_learning_harness.h", "cpp/apps/goal_curriculum_demonstration.h",
    "cpp/apps/goal_long_environment.h", "cpp/apps/goal_mixed_skill_extension.h",
    "cpp/tests/goal_control_fixture.h", "cpp/tests/test_goal_mixed_skill_extension.cpp",
    "python/validate_goal_mixed_skill.py")]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = []
for path in files:
    target = root / "source" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    manifest.append({"path": str(target.relative_to(root)), "sha256": digest(target)})
(root / "source_manifest.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n")
results = {}
for variant, flags in (("optimized", ["-O2"]), ("sanitizer", ["-O1", "-g0", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])):
    binary = root / ("test_goal_mixed_skill_" + variant)
    command = ["wsl", "g++", "-std=c++20", *flags, "-I" + (root / "source/cpp/include").as_posix(),
               (root / "source/cpp/tests/test_goal_mixed_skill_extension.cpp").as_posix(), "-o", binary.as_posix()]
    build = subprocess.run(command, capture_output=True, timeout=300)
    (root / (variant + "_build_stdout.txt")).write_bytes(build.stdout)
    (root / (variant + "_build_stderr.txt")).write_bytes(build.stderr)
    assert build.returncode == 0, build.stderr.decode()
    run = subprocess.run(["wsl", "env", "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1", "UBSAN_OPTIONS=halt_on_error=1",
                          "timeout", "240s", binary.as_posix()], capture_output=True, timeout=250)
    (root / (variant + "_stdout.txt")).write_bytes(run.stdout)
    (root / (variant + "_stderr.txt")).write_bytes(run.stderr)
    results[variant] = {"command": command, "binary_sha256": digest(binary), "exit": run.returncode,
                        "stdout_sha256": hashlib.sha256(run.stdout).hexdigest(), "stderr": run.stderr.decode()}
    assert run.returncode == 0, run.stderr.decode()
result = {"scope": "Raw mixed-skill integration/provenance, not autonomous policy acquisition", "passed": True,
          "source_manifest_sha256": digest(root / "source_manifest.json"), "checks": results}
(root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
