#!/usr/bin/env bash
# Byte-regime sweep (milestone D): error gain, accumulator rate, readout rate, death threshold.
#   scripts/sweep_bytes.sh [file] [units] [epochs]
set -uo pipefail
cd "$(dirname "$0")/.."
file="${1:-data/tinystories_100k.txt}"; units="${2:-50000}"; epochs="${3:-4}"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_bytes file="$file" units="$units" epochs="$epochs" quiet=1 "$@" | grep -E '^EPOCH' | sed -E 's/EPOCH ([0-9]+) \[adaptive\]: BPB ([0-9.]+) acc ([0-9.]+)%.*MACs\/byte ([0-9]+).*lvl ([0-9.]+).*locks ([0-9]+).*/e\1 \2 (\3% m\4 l\5 k\6)/' | tr '\n' ' ')
  printf '%-40s :: %s\n' "$tag" "$out"
}
for eg in 2 4 8; do
  for ls in 9 10 11; do
    for lo in 0.02 0.005; do
      for de in 6 20; do
        run "eg=$eg ls=$ls lr_out=$lo death=$de" err_gain=$eg lr_shift=$ls lr_out=$lo death=$de &
      done
    done
  done
done
wait
