"""Freeze and verify opt-in demonstration fitting; no capability campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("logs/goal_demonstration_fitting_v1"))
    parser.add_argument("--record", type=Path, default=Path("docs/GOAL_DEMONSTRATION_FITTING_VALIDATION.json"))
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("include", "tests"):
        shutil.copytree(Path("cpp") / folder, root / "source/cpp" / folder)
    manifest = {str(path.relative_to(root / "source")): sha(path)
                for path in (root / "source").rglob("*") if path.is_file()}
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [("demonstrations_optimized", "test_goal_demonstrations", ["-O2", "-DNDEBUG"]),
            ("demonstrations_sanitized", "test_goal_demonstrations", ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]),
            ("goal_compatibility", "test_goal_control", ["-O2", "-DNDEBUG"]),
            ("mission_compatibility", "test_mission_credit", ["-O2", "-DNDEBUG"])]

    def run(job):
        label, source, flags = job
        binary = root / label
        command = ["g++", "-std=c++20", *flags, "-I" + str(root / "source/cpp/include"),
                   str(root / "source/cpp/tests" / (source + ".cpp")), "-o", str(binary)]
        with (root / (label + ".compile.log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                           UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        with (root / (label + ".stdout")).open("w") as output, (root / (label + ".stderr")).open("w") as errors:
            subprocess.run([str(binary)], stdout=output, stderr=errors, env=environment, check=True)
        assert (root / (label + ".stderr")).stat().st_size == 0
        print(label + ": passed", flush=True)
        return {"label": label, "command": command, "binary_sha256": sha(binary),
                "stdout_sha256": sha(root / (label + ".stdout")), "stderr_bytes": 0, "passed": True}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, jobs))
    relevant = ["cpp/include/laminar/relational_control.h", "cpp/include/laminar/goal_control.h",
                "cpp/tests/test_goal_demonstrations.cpp"]
    assert all(sha(path) == manifest[path] for path in relevant), "relevant source changed during validation"
    record = {"schema": "goal_demonstration_fitting_validation_v1", "passed": True,
              "scope": "Opt-in source-tagged supervised bootstrap, unchanged empirical values/semantics, invalid-call rejection and exact credit replay; no generalization claim.",
              "source_manifest_sha256": sha(root / "source_manifest.json"),
              "source_sha256": {path: manifest[path] for path in relevant}, "runs": results}
    (root / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    args.record.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
