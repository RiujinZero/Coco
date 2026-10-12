#!/usr/bin/env bash
# Milestone D (design doc §15, runbook §14): raw-byte TinyStories, online next-byte prediction.
# Compares adaptive locks + learning, frozen locks (after one epoch), and routing/memory-disabled.
#   scripts/milestone_d.sh [file] [units] [epochs] [extra params...]
set -uo pipefail
cd "$(dirname "$0")/.."
file="${1:-data/tinystories_100k.txt}"; units="${2:-50000}"; epochs="${3:-6}"; shift 3 2>/dev/null || true
warm=$(stat -c %s "$file")
for mode in adaptive frozen nomem; do
  echo "=== mode=$mode ==="
  ./build-wsl/laminar_bytes file="$file" units="$units" epochs="$epochs" mode="$mode" warm="$warm" quiet=1 log="build-wsl/bytes_$mode.jsonl" "$@" | grep -E '^EPOCH|^TOTAL|frozen'
done
