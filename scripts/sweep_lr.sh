#!/usr/bin/env bash
# Learning-rate / error-gain probe, 1-context (pure INT4 learning) and 8-context (with structure).
#   scripts/sweep_lr.sh [units]
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-4096}"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_synth units="$units" train_events=120000 eval_events=16000 report=999999999 "$@" | grep FROZEN | sed -E 's/FROZEN: [0-9]+ events, //; s/\(chance[^)]*\), //; s/, [0-9.]+ KB\/s.*live locks/ locks/')
  printf '%-34s :: %s\n' "$tag" "$out"
}
for ctx in 1 8; do
  for ls in 10 11 12 13; do
    for eg in 4 8; do
      run "ctx=$ctx lr_shift=$ls err_gain=$eg" contexts=$ctx lr_shift=$ls err_gain=$eg &
    done
  done
done
wait
