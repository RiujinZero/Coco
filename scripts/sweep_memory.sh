#!/usr/bin/env bash
# Working-memory mechanisms on TinyStories: persistent workspace (key / input), L4 -> next key, longer history.
#   scripts/sweep_memory.sh [file] [units] [epochs]
set -uo pipefail
cd "$(dirname "$0")/.."
file="${1:-data/tinystories_100k.txt}"; units="${2:-50000}"; epochs="${3:-6}"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_bytes file="$file" units="$units" epochs="$epochs" quiet=1 "$@" | grep -E '^EPOCH' \
        | sed -E 's/EPOCH ([0-9]+) \[adaptive\]: BPB ([0-9.]+) acc ([0-9.]+)%.*locks ([0-9]+).*/e\1 \2 (k\4)/' | tr '\n' ' ')
  printf '%-44s :: %s\n' "$tag" "$out"
}
run "baseline hist=4" &
run "hist=8" hist=8 &
run "hist=12" hist=12 &
run "l4_key" l4_key=1 &
run "ws_decay=128 ws_input" ws_decay=128 ws_input=1 &
run "ws_decay=192 ws_input" ws_decay=192 ws_input=1 &
run "ws_decay=224 ws_input" ws_decay=224 ws_input=1 &
run "ws_decay=192 ws_key" ws_decay=192 ws_key=1 &
run "ws_decay=192 ws_key ws_input" ws_decay=192 ws_key=1 ws_input=1 &
run "ws_decay=192 ws_key ws_input l4_key" ws_decay=192 ws_key=1 ws_input=1 l4_key=1 &
run "ws_decay=192 ws_key ws_input l4_key hist=8" ws_decay=192 ws_key=1 ws_input=1 l4_key=1 hist=8 &
run "ws_decay=224 ws_key ws_input l4_key hist=8" ws_decay=224 ws_key=1 ws_input=1 l4_key=1 hist=8 &
wait
