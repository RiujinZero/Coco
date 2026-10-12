#!/usr/bin/env bash
# The memory side thinking in composed units: level-1 units as the chain's words, level-2 units as its sentences.
# Chain ON (seq_chain=2), shuffled bAbI (1 pass), held-out bAbI 1-10.
set -uo pipefail
cd "$(dirname "$0")/.."
B=./build-wsl/laminar_babi_cmp
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o "$B" 2>&1 | grep -E "error" | head -5
run() { local name="$1"; shift
  "$B" file=data/babi_all_train_shuf.txt test=data/babi_d10_test.jsonl units=50000 kpack=1 seq_topic_gain=8 seq_chain=2 seq_max_depth=8 hops=4 hops_free=2 questions=1000 show=0 "$@" > "logs/ut_$name.log" 2>&1; }
run base     levels=0 &
run u1_m2    levels=2 lvl_margin=2 seq_units=1 &
run u2_m2    levels=2 lvl_margin=2 seq_units=2 &
run u1_m3    levels=2 lvl_margin=3 seq_units=1 &
run u2_m3    levels=2 lvl_margin=3 seq_units=2 &
run u1_m15   levels=2 lvl_margin=1.5 seq_units=1 &
wait
for n in base u1_m2 u2_m2 u1_m3 u2_m3 u1_m15; do
  printf "%-8s %-9s %s\n" "$n" "$(grep -a '^BABI' logs/ut_$n.log | awk '{split($3,a,"/"); c+=a[1]; t+=a[2]} END {printf "%d/%d", c, t}')" "$(grep -a '^BABI' logs/ut_$n.log | awk '{split($3,a,"/"); printf "%s%s ", $2, a[1]}')"
done
for n in base u1_m2 u2_m2; do echo "$n: $(grep -a '^trained' logs/ut_$n.log | cut -c10-70) | $(grep -ao 'composer: L1 [^;]*;' logs/ut_$n.log)"; done
