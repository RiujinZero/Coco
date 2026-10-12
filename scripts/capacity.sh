#!/usr/bin/env bash
# Address capacity: G-test splits only (old) vs ALM-style nursery admission + utility recycling.
#   scripts/capacity.sh [max_bytes] [units] [extra params...]
set -uo pipefail
cd "$(dirname "$0")/.."
mb="${1:-33554432}"; units="${2:-200000}"; shift 2 2>/dev/null || true
run() {
  local tag="$1"; shift
  echo "=== $tag ==="
  ./build-wsl/laminar_bytes file=data/tinystories_13m.txt max_bytes="$mb" units="$units" epochs=1 quiet=1 progress=8388608 kpack=0 "$@" 2>&1 \
    | grep -E '^ |^EPOCH|^TOTAL' | sed -E 's/acc [0-9.]+% \| //; s/adm [0-9.]+ emit/emit/' | cut -c1-300
}
run "G-test splits only" nursery=0 specific_wins=0 &
run "nursery admit_min=8 admit_bits=0.10" nursery=1 &
run "nursery admit_min=4 admit_bits=0.05" nursery=1 admit_min=4 admit_bits=0.05 &
run "nursery admit_min=16 admit_bits=0.20" nursery=1 admit_min=16 admit_bits=0.20 &
wait
