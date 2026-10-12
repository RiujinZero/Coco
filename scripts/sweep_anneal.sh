#!/usr/bin/env bash
# Novelty-annealed learning rate probe: fast while a lock is novel/surprising, slower once familiar.
#   scripts/sweep_anneal.sh [units]
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-4096}"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_synth units="$units" train_events=120000 eval_events=16000 report=999999999 "$@" | grep FROZEN | sed -E 's/FROZEN: [0-9]+ events, //; s/\(chance[^)]*\), //; s/, [0-9.]+ KB\/s.*live locks/ locks/')
  printf '%-44s :: %s\n' "$tag" "$out"
}
for ctx in 1 8; do
  for ls in 9 10; do
    for an in 0 2 3 4; do
      for eg in 4 8; do
        run "ctx=$ctx lr_shift=$ls anneal=$an err_gain=$eg" contexts=$ctx lr_shift=$ls lr_anneal=$an err_gain=$eg &
      done
    done
  done
done
wait
