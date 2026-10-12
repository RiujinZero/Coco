#!/usr/bin/env bash
# Motif packing and K screening: unpacked (random placement) vs packed vs packed + screened.
#   scripts/kpack.sh [max_bytes] [units]
set -uo pipefail
cd "$(dirname "$0")/.."
mb="${1:-33554432}"; units="${2:-200000}"
run() {
  local tag="$1"; shift
  echo "=== $tag ==="
  ./build-wsl/laminar_bytes file=data/tinystories_13m.txt max_bytes="$mb" units="$units" epochs=1 quiet=1 progress=8388608 "$@" 2>&1 \
    | grep -E '^ |^EPOCH|^TOTAL' | sed -E 's/acc [0-9.]+% \| //; s/adm [0-9.]+ emit/emit/' | cut -c1-330
}
run "unpacked, random K (old behaviour)" kpack=0 &
run "packed, K by screen" kpack=1 kscreen=1 &
run "packed, K random" kpack=1 kscreen=0 &
wait
