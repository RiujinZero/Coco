#!/usr/bin/env bash
# Hidden-layer local learning (paired +/- node perturbation, shared reward difference): tuning sweep.
#   scripts/sweep_perturb.sh [units] [contexts]
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-4096}"; ctx="${2:-8}"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_synth units="$units" contexts="$ctx" train_events=120000 eval_events=16000 report=999999999 "$@" | grep FROZEN | sed -E 's/FROZEN: [0-9]+ events, //; s/\(chance[^)]*\), //; s/, [0-9.]+ KB\/s.*live locks/ locks/; s/, live units.*//')
  printf '%-44s :: %s\n' "$tag" "$out"
}
run "L6-only baseline" &
for d in 4 8 16; do
  for g in 16 64 256; do
    for hs in 0 2 4; do
      run "perturb delta=$d gain=$g hidden_shift=$hs" hidden=perturb perturb_delta=$d perturb_gain=$g hidden_shift=$hs &
    done
  done
done
wait
