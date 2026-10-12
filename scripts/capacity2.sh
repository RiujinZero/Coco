#!/usr/bin/env bash
# Admission bottlenecks: confirmations, evidence pool size, conflict threshold.
#   scripts/capacity2.sh [binary] [max_bytes]
set -uo pipefail
cd "$(dirname "$0")/.."
bin="${1:-build-wsl/laminar_bytes_v3}"; mb="${2:-33554432}"
run() {
  local tag="$1"; shift
  local out
  out=$("$bin" file=data/tinystories_13m.txt max_bytes="$mb" units=100000 epochs=1 quiet=1 progress=8388608 kpack=0 nursery=1 "$@" 2>&1 \
        | grep -E '^ ' | sed -E 's/ +([0-9.]+) MB \| window BPB ([0-9.]+) acc ([0-9.]+)%.*splits ([0-9]+).*locks ([0-9]+).*/\1MB:\2(\3%,adm\4,k\5)/' | tr '\n' ' ')
  printf '%-40s :: %s\n' "$tag" "$out"
}
run "admit_min=4 bits=0.05 (new default)" &
run "admit_min=2 bits=0.02" admit_min=2 admit_bits=0.02 &
wait
run "evidence_slots=16384" evidence_slots=16384 &
run "conflict=12000" conflict=12000 &
wait
run "all three" admit_min=2 admit_bits=0.02 evidence_slots=16384 conflict=12000 &
wait
