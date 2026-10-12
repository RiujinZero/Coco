#!/usr/bin/env bash
# Items 2-4: utility-gated routing on packed motifs, utility-gated episodic memory, modulated hidden learning.
# Two runs at a time: a 100k-unit run with the nursery tables is ~1.5 GB and the WSL VM OOMs above ~4 concurrent.
#   scripts/items234.sh [binary] [max_bytes] [units]
set -uo pipefail
cd "$(dirname "$0")/.."
bin="${1:-build-wsl/laminar_bytes_v3}"; mb="${2:-33554432}"; units="${3:-100000}"
run() {
  local tag="$1"; shift
  local out
  out=$("$bin" file=data/tinystories_13m.txt max_bytes="$mb" units="$units" epochs=1 quiet=1 progress=8388608 "$@" 2>&1 \
        | grep -E '^ |^TOTAL|route-gated' | sed -E 's/ +([0-9.]+) MB \| window BPB ([0-9.]+) acc ([0-9.]+)%.*locks ([0-9]+).*/\1MB:\2(\3%,k\4)/; s/^TOTAL: [0-9]+ bytes, (BPB [0-9.]+).*, ([0-9.]+) KB\/s, ([0-9]+) MACs.*/TOTAL \1 \2KB\/s \3MACs/; s/^  kclass: (route-gated [0-9\/]+).*/\1/' | tr '\n' ' ')
  printf '%-46s :: %s\n' "$tag" "$out"
}
run "item2: packed+screen, gate OFF" kpack=1 route_gate=0 &
run "item2: packed+screen, gate ON 0.05" kpack=1 route_gate=1 &
wait
run "item2: unpacked, gate ON (reference)" kpack=0 route_gate=1 &
run "item3: episodic utility-gated (sum)" kpack=0 episodic=1 hist=7 route_gate=1 &
wait
run "item4: hidden=modulated" kpack=0 hidden=modulated &
run "item4: hidden=modulated mod_gain=8" kpack=0 hidden=modulated mod_gain=8 &
wait
echo "=== item4 on the 8-context synthetic task (modulated / L6-only) ==="
./build-wsl/laminar_synth units=4096 train_events=120000 eval_events=16000 report=999999999 hidden=modulated | grep FROZEN | cut -c1-110
./build-wsl/laminar_synth units=4096 train_events=120000 eval_events=16000 report=999999999 | grep FROZEN | cut -c1-110
