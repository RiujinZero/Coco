#!/usr/bin/env bash
# Is the perturbation-mode deficit the hidden update or the exploration noise on the committed prediction?
#   scripts/diag_perturb.sh [units] [contexts]
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-4096}"; ctx="${2:-8}"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_synth units="$units" contexts="$ctx" train_events=120000 eval_events=16000 report=999999999 "$@" | grep FROZEN | sed -E 's/FROZEN: [0-9]+ events, //; s/\(chance[^)]*\), //; s/, [0-9.]+ KB\/s.*live locks/ locks/; s/, live units.*//')
  printf '%-52s :: %s\n' "$tag" "$out"
}
run "L6-only baseline" &
run "perturb, no hidden update (gain=0), averaged"     hidden=perturb perturb_gain=0 &
run "perturb, no hidden update (gain=0), clean pass"   hidden=perturb perturb_gain=0 perturb_clean=1 &
for d in 4 8 16; do
  for g in 16 64 256; do
    run "perturb clean delta=$d gain=$g hidden_shift=2" hidden=perturb perturb_clean=1 perturb_delta=$d perturb_gain=$g hidden_shift=2 &
  done
done
run "perturb clean delta=8 gain=64 hidden_shift=0" hidden=perturb perturb_clean=1 perturb_delta=8 perturb_gain=64 hidden_shift=0 &
run "perturb clean delta=8 gain=64 hidden_shift=4" hidden=perturb perturb_clean=1 perturb_delta=8 perturb_gain=64 hidden_shift=4 &
wait
