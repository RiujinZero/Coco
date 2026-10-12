#!/usr/bin/env bash
# Throughput and memory at different unit capacities, same stream and mechanisms (runs in parallel, one core each).
#   scripts/units_throughput.sh [max_bytes] [units...]      default: 32 MB of the mixed stream, 200k and 800k
set -uo pipefail
cd "$(dirname "$0")/.."
mb="${1:-33554432}"; shift 2>/dev/null || true
list="${*:-200000 800000}"
mkdir -p logs
for u in $list; do
  ( /usr/bin/time -v ./build-wsl/laminar_bytes_opt file=data/mix_650m.txt max_bytes="$mb" units="$u" epochs=1 timers=1 \
      kpack=1 seqmem=1 seq_chain=2 agency=1 > "logs/units_$u.log" 2>&1 ) &
done
wait
for u in $list; do
  echo "##### units=$u"
  grep -aE "^TOTAL" "logs/units_$u.log" | cut -c1-150
  grep -a "^\[adaptive" "logs/units_$u.log" | tail -1 | grep -oE "input=.*"
  grep -aE "Maximum resident" "logs/units_$u.log"
done
