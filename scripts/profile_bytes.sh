#!/usr/bin/env bash
# Stage-time ablations for the raw-byte path (runbook §13: profile the dominant stage only).
#   scripts/profile_bytes.sh FILE [units]
set -uo pipefail
cd "$(dirname "$0")/.."
file="$1"; units="${2:-20000}"
if command -v perf >/dev/null 2>&1; then
  echo "perf available: perf stat -e cycles,instructions,branches,branch-misses,cache-misses ./build-wsl/laminar_bytes file=$file units=$units"
else
  echo "perf not installed (sudo apt install linux-tools-common linux-tools-generic); using stage timers + ablations"
fi
for v in "" "train=0" "err_gain=0" "lr_shift=20" "structural=0" "hops=0"; do
  printf '%-14s ' "[${v:-default}]"
  ./build-wsl/laminar_bytes file="$file" units="$units" epochs=2 block=8192 $v | grep 'e1\] blk 4' | grep -oE '\| [0-9.]+ KB/s|readout=.*' | tr '\n' ' '
  echo
done
