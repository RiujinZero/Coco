#!/usr/bin/env bash
# Learned compute, round 3: recirculation decided by a two-armed comparison of the ACTUAL loss (recirc_learn=2), with
# learned fan-out now default (hops=4, hops_free=2). Baselines rerun on the new defaults.
set -uo pipefail
cd "$(dirname "$0")/.."
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_bytes.cpp cpp/src/organism.cpp -o build-wsl/laminar_bytes_lc 2>&1 | grep -E "error" | head -5
bytes() { local name="$1" file="$2" mb="$3"; shift 3
  ./build-wsl/laminar_bytes_lc file="$file" max_bytes="$mb" units=50000 epochs=1 timers=0 kpack=1 "$@" > "logs/lc_$name.log" 2>&1; }
TS=data/tinystories_13m.txt; MX=data/mix_650m.txt
bytes ts_def      $TS 8388608 &
bytes ts_old      $TS 8388608  hops=2 &
bytes ts_arm2     $TS 8388608  recirc_iters=2 recirc_learn=2 &
bytes ts_arm2_e10 $TS 8388608  recirc_iters=2 recirc_learn=2 recirc_explore=0.1 &
bytes mx_def      $MX 16777216 &
bytes mx_arm2     $MX 16777216 recirc_iters=2 recirc_learn=2 &
wait
printf "%-12s %8s %7s %8s  %s\n" run BPB KB/s MACs/B "learned-compute usage"
for n in ts_old ts_def ts_arm2 ts_arm2_e10 mx_def mx_arm2; do t=$(grep -a "^TOTAL" logs/lc_$n.log)
  printf "%-12s %8s %7s %8s  %s\n" "$n" "$(echo "$t" | grep -oE "BPB [0-9.]+" | cut -d' ' -f2)" "$(echo "$t" | grep -oE "[0-9.]+ KB/s" | cut -d' ' -f1)" "$(echo "$t" | grep -oE "[0-9]+ MACs/byte" | cut -d' ' -f1)" "$(grep -aoE "learned compute:.*" logs/lc_$n.log | tail -1 | cut -c18-120)"; done
