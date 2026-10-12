#!/usr/bin/env bash
# Why does BPB degrade on long streams? One-factor ablations on a 32 MB TinyStories slice.
# Runs in two waves of five so the WSL VM stays inside its memory budget (~0.9 GB per 100k-unit run).
#   scripts/drift_ablation.sh [max_bytes] [units]
set -uo pipefail
cd "$(dirname "$0")/.."
mb="${1:-33554432}"; units="${2:-100000}"
run() {
  local tag="$1"; shift
  local out
  out=$(./build-wsl/laminar_bytes file=data/tinystories_13m.txt max_bytes="$mb" units="$units" epochs=1 quiet=1 progress=8388608 "$@" 2>&1 \
        | grep -E '^ ' | sed -E 's/ +([0-9.]+) MB \| window BPB ([0-9.]+) acc ([0-9.]+)%.*MACs\/byte ([0-9]+).*locks ([0-9]+).*/\1MB:\2(\3%,m\4,k\5)/' | tr '\n' ' ')
  printf '%-34s :: %s\n' "$tag" "$out"
}
run "baseline" &
run "hops=0 (no routing)" hops=0 &
run "readout_freeze=4MB" readout_freeze=4194304 &
run "ws off" ws_decay=0 ws_input=0 &
run "no splits" split_min=4000000000 &
wait
run "lr_anneal=0" lr_anneal=0 &
run "hops=0 + readout_freeze" hops=0 readout_freeze=4194304 &
run "hops=0 + freeze + no splits" hops=0 readout_freeze=4194304 split_min=4000000000 &
run "recirc off" recirc=1000 &
run "err_gain=2" err_gain=2 &
wait
echo DONE
