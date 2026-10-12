#!/usr/bin/env bash
# Learned compute: recirculation and extra fan-out used only where their counterfactual gain has paid, against the
# fixed settings and none. Bodies-level measures: BPB and throughput on TinyStories 8 MB and the mixed stream 16 MB;
# plus yes/no and counting on the diagnostic (answered by bodies + memory).
set -uo pipefail
cd "$(dirname "$0")/.."
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_bytes.cpp cpp/src/organism.cpp -o build-wsl/laminar_bytes_lc 2>&1 | grep -E "error" | head -5
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o build-wsl/laminar_babi_lc 2>&1 | grep -E "error" | head -5
bytes() { local name="$1" file="$2" mb="$3"; shift 3
  ./build-wsl/laminar_bytes_lc file="$file" max_bytes="$mb" units=50000 epochs=1 timers=0 kpack=1 "$@" > "logs/lc_$name.log" 2>&1; }
diag() { local name="$1"; shift
  ./build-wsl/laminar_babi_lc file=data/diag_train.txt test=data/diag_test.jsonl units=50000 kpack=1 seq_topic_gain=8 seq_chain=2 seq_max_depth=8 show=0 questions=1000 "$@" > "logs/lc_$name.log" 2>&1; }
TS=data/tinystories_13m.txt; MX=data/mix_650m.txt
bytes ts_none   $TS 8388608 &
bytes ts_fix2   $TS 8388608 recirc_iters=2 &
bytes ts_learn2 $TS 8388608 recirc_iters=2 recirc_learn=1 &
bytes ts_hops4L $TS 8388608 hops=4 hops_free=2 &
bytes mx_none   $MX 16777216 &
bytes mx_learn2 $MX 16777216 recirc_iters=2 recirc_learn=1 &
bytes mx_both   $MX 16777216 recirc_iters=2 recirc_learn=1 hops=4 hops_free=2 &
diag dg_none &
diag dg_learn  recirc_iters=2 recirc_learn=1 hops=4 hops_free=2 &
wait
printf "%-10s %8s %7s %8s  %s\n" run BPB KB/s MACs/B "learned-compute usage"
for n in ts_none ts_fix2 ts_learn2 ts_hops4L mx_none mx_learn2 mx_both; do t=$(grep -a "^TOTAL" logs/lc_$n.log)
  printf "%-10s %8s %7s %8s  %s\n" "$n" "$(echo "$t" | grep -oE "BPB [0-9.]+" | cut -d' ' -f2)" "$(echo "$t" | grep -oE "[0-9.]+ KB/s" | cut -d' ' -f1)" "$(echo "$t" | grep -oE "[0-9]+ MACs/byte" | cut -d' ' -f1)" "$(grep -aoE "learned compute:.*" logs/lc_$n.log | tail -1 | cut -c18-140)"; done
for n in dg_none dg_learn; do printf "%-10s yes/no %s  count %s  %s\n" "$n" "$(grep -a "^BABI qa6:" logs/lc_$n.log | grep -oE "[0-9]+/[0-9]+" | head -1)" "$(grep -a "^BABI qa7:" logs/lc_$n.log | grep -oE "[0-9]+/[0-9]+" | head -1)" "$(grep -a "^trained" logs/lc_$n.log | grep -oE "[0-9]+ KB/s")"; done
