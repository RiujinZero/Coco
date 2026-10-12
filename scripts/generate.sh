#!/usr/bin/env bash
# Free generation (the true test): bodies only vs + sequence memory (composition), at several temperatures.
#   scripts/generate.sh [max_bytes] [units] [gen] [prompts]
set -uo pipefail
cd "$(dirname "$0")/.."
mb="${1:-33554432}"; units="${2:-100000}"; gen="${3:-1024}"; prompts="${4:-6}"
run() {
  local tag="$1"; shift
  echo "=== $tag ==="
  ./build-wsl/laminar_generate file=data/tinystories_13m.txt max_bytes="$mb" units="$units" gen="$gen" prompts="$prompts" kpack=0 "$@" 2>&1 | cut -c1-300
}
run "B. + sequence memory, temp 1.0" seqmem=1 &
run "B. + sequence memory, temp 0.7" seqmem=1 temp=0.7 &
wait
run "B. + sequence memory, greedy" seqmem=1 temp=0 &
run "A. bodies only, temp 0.7" temp=0.7 &
wait
