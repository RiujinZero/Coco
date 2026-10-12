#!/usr/bin/env bash
# Runbook §13.1: correctness of the native pipeline against the NumPy reference on a small
# deterministic trace. Run from WSL (python3 + numpy required there) or adapt paths for Windows.
#   scripts/parity.sh [units] [train_events] [eval_events]
set -euo pipefail
cd "$(dirname "$0")/.."
units="${1:-1024}"; train="${2:-3000}"; ev="${3:-500}"
tmp="${TMPDIR:-/tmp}/laminar_parity"; mkdir -p "$tmp"
python3 python/run_synth.py --units "$units" --train "$train" --eval "$ev" --block 1024 --report 1024 --trace "$tmp/ref.jsonl" | tail -2
./build-wsl/laminar_synth units="$units" train_events="$train" eval_events="$ev" block=1024 report=1024 trace="$tmp/nat.jsonl" | grep -E 'TRAIN|FROZEN' | cut -c1-120
python3 python/tests/compare_traces.py "$tmp/ref.jsonl" "$tmp/nat.jsonl"
