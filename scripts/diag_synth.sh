#!/usr/bin/env bash
# Controlled diagnostics on the synthetic task: which mechanism limits accuracy?
#   scripts/diag_synth.sh [units] [extra params...]
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-4096}"; shift 1 2>/dev/null || true
extra="$*"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_synth units="$units" eval_events=16000 report=999999999 "$@" $extra | grep FROZEN | sed -E 's/FROZEN: [0-9]+ events, //; s/\(chance[^)]*\), //')
  printf '%-14s :: %s\n' "$tag" "$out"
}
run baseline    train_events=120000 &
run noise0      train_events=120000 noise=0 &
run ctx1        train_events=120000 contexts=1 &
run ctx1-noise0 train_events=120000 contexts=1 noise=0 &
run 240k        train_events=240000 &
run perturb     train_events=120000 hidden=perturb &
run hops0       train_events=120000 hops=0 &
run structural0 train_events=120000 structural=0 &
run recirc-off  train_events=120000 recirc=1000 &
wait
