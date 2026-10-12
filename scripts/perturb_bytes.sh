#!/usr/bin/env bash
# Hidden-layer perturbation learning on TinyStories vs L6-only.
#   scripts/perturb_bytes.sh [file] [units] [epochs]
set -uo pipefail
cd "$(dirname "$0")/.."
file="${1:-data/tinystories_100k.txt}"; units="${2:-50000}"; epochs="${3:-6}"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_bytes file="$file" units="$units" epochs="$epochs" quiet=1 "$@" | grep -E '^EPOCH' \
        | sed -E 's/EPOCH ([0-9]+) \[adaptive\]: BPB ([0-9.]+) acc ([0-9.]+)% \| ([0-9.]+) KB\/s.*/e\1 \2 (\3% \4KB\/s)/' | tr '\n' ' ')
  printf '%-40s :: %s\n' "$tag" "$out"
}
run "L6-only" &
for d in 8 16; do
  for g in 16 64 256; do
    for hs in 0 2; do
      run "perturb clean d=$d g=$g hs=$hs" hidden=perturb perturb_clean=1 perturb_delta=$d perturb_gain=$g hidden_shift=$hs &
    done
  done
done
wait
