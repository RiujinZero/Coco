#!/usr/bin/env bash
# Runbook §11: memory/cache stress on the real 200k x 5632 packed INT4 store (~563 MB) with randomly
# scattered route bodies, under four activity regimes. Run alone on an idle machine.
#   scripts/bench.sh [units] [events] [extra params...]
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-200000}"; events="${2:-100000}"; shift 2 2>/dev/null || true
for m in familiar moderate storm pathological; do
  ./build-wsl/laminar_bench mode="$m" units="$units" events="$events" block=20000 "$@" | grep -E '^BENCH|^blk' | tail -2
done
