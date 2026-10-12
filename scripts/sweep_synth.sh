#!/usr/bin/env bash
# Small parallel parameter sweep on the synthetic vocabulary task (runbook §14 A/B).
#   scripts/sweep_synth.sh [units] [train_events] [extra params...]
set -uo pipefail
cd "$(dirname "$0")/.."
bin=build-wsl/laminar_synth
units="${1:-4096}"; train="${2:-120000}"; shift 2 2>/dev/null || true
extra="$*"
run() {
  local tag="$1"; shift
  local out
  out=$("$bin" units="$units" train_events="$train" eval_events=16000 report=999999999 "$@" $extra | grep FROZEN | sed -E 's/FROZEN: [0-9]+ events, //; s/\(chance[^)]*\), //')
  echo "$tag :: $out"
}
for ls in 9 10 11; do
  for eg in 1 2 4; do
    for sg in 2 3; do
      for cf in 16000 22000; do
        run "lr_shift=$ls err_gain=$eg split_g=$sg conflict=$cf" lr_shift=$ls err_gain=$eg split_g=$sg conflict=$cf &
      done
    done
  done
done
wait
