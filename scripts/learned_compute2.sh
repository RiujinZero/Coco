#!/usr/bin/env bash
# Learned compute, round 2: recirculation must be EARNED (utility starts at 0, rare exploration, weights trained on the
# ordinary path), learned fan-out depth on both streams.
set -uo pipefail
cd "$(dirname "$0")/.."
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_bytes.cpp cpp/src/organism.cpp -o build-wsl/laminar_bytes_lc 2>&1 | grep -E "error" | head -5
bytes() { local name="$1" file="$2" mb="$3"; shift 3
  ./build-wsl/laminar_bytes_lc file="$file" max_bytes="$mb" units=50000 epochs=1 timers=0 kpack=1 "$@" > "logs/lc_$name.log" 2>&1; }
TS=data/tinystories_13m.txt; MX=data/mix_650m.txt
bytes ts_earn2    $TS 8388608  recirc_iters=2 recirc_learn=1 &
bytes ts_earn2_e1 $TS 8388608  recirc_iters=2 recirc_learn=1 recirc_explore=0.1 &
bytes ts_earn2_h  $TS 8388608  recirc_iters=2 recirc_learn=1 hops=4 hops_free=2 &
bytes mx_hops4L   $MX 16777216 hops=4 hops_free=2 &
bytes mx_earn2    $MX 16777216 recirc_iters=2 recirc_learn=1 &
bytes ts_hops6L   $TS 8388608  hops=6 hops_free=2 &
wait
printf "%-12s %8s %7s %8s  %s\n" run BPB KB/s MACs/B "learned-compute usage"
for n in ts_none ts_hops4L ts_hops6L ts_earn2 ts_earn2_e1 ts_earn2_h mx_none mx_hops4L mx_earn2; do t=$(grep -a "^TOTAL" logs/lc_$n.log)
  printf "%-12s %8s %7s %8s  %s\n" "$n" "$(echo "$t" | grep -oE "BPB [0-9.]+" | cut -d' ' -f2)" "$(echo "$t" | grep -oE "[0-9.]+ KB/s" | cut -d' ' -f1)" "$(echo "$t" | grep -oE "[0-9]+ MACs/byte" | cut -d' ' -f1)" "$(grep -aoE "learned compute:.*" logs/lc_$n.log | tail -1 | cut -c18-120)"; done
