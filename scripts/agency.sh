#!/usr/bin/env bash
# Does the agency kernel's attention direct the organism usefully?
#   agency=0 off | agency=2 control (fixed focus bump, no operations) | agency=1 on
#   scripts/agency.sh [file] [units] [epochs]        TinyStories BPB
#   NEEDLE=1 scripts/agency.sh                        needle recall with episodic memory, kernel chooses the stream
set -uo pipefail
cd "$(dirname "$0")/.."
file="${1:-data/tinystories_100k.txt}"; units="${2:-50000}"; epochs="${3:-6}"; shift 3 2>/dev/null || true
if [ "${NEEDLE:-0}" = "1" ]; then
  for a in 0 2 1; do
    echo "=== needle 256K, agency=$a ==="
    ./build-wsl/laminar_needle file=data/tinystories_1m.txt window=262144 needles=120 units=200000 hist=7 episodic=1 readout_freeze=131072 agency=$a agency_ops=hold,attend_episodic,attend_generic,to_memory "$@" \
      | grep -E '^PLANT|0.12 x|1.00 x|2.00 x|^agency|^TOTAL'
  done
  exit 0
fi
for a in ${AGENCY:-0 2 1}; do
  echo "=== TinyStories agency=$a $* ==="
  ./build-wsl/laminar_bytes file="$file" units="$units" epochs="$epochs" quiet=1 agency=$a "$@" | grep -E '^EPOCH|^  agency|^TOTAL' | sed -E 's/acc [0-9.]+% \| //; s/MACs\/byte [0-9]+ adm [0-9.]+ emit [0-9.]+ nbr [0-9.]+ qmax [0-9]+ \| //'
done
