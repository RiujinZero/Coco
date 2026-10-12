#!/usr/bin/env bash
# Early death on language: relative death threshold x metabolic pressure. Reports BPB vs MACs/byte.
#   scripts/sweep_death.sh [file] [units] [epochs]
set -uo pipefail
cd "$(dirname "$0")/.."
file="${1:-data/tinystories_100k.txt}"; units="${2:-50000}"; epochs="${3:-4}"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_bytes file="$file" units="$units" epochs="$epochs" quiet=1 "$@" | grep -E '^EPOCH' \
        | sed -E 's/EPOCH ([0-9]+) \[adaptive\]: BPB ([0-9.]+) acc [0-9.]+% \| ([0-9.]+) KB\/s \| MACs\/byte ([0-9]+) adm [0-9.]+ emit ([0-9.]+).*/e\1 \2 (m\4 emit\5 \3KB\/s)/' | tr '\n' ' ')
  printf '%-36s :: %s\n' "$tag" "$out"
}
for dr in 0 32 40 48 56; do
  for ru in 2 6 12; do
    run "death_rel=$dr resist_up=$ru" death_rel=$dr resist_up=$ru &
  done
done
wait
