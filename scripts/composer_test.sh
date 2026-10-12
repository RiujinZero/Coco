#!/usr/bin/env bash
# Composer on the answer experiment: chain OFF (answers from bodies + memory), shuffled bAbI x3, held-out bAbI 1-10.
# Arms: no composer; 2 levels; 2 levels + goal credit; 2 levels fed back into the byte bodies. Ablations silence one
# level's bodies at test time (same training, deterministic).
set -uo pipefail
cd "$(dirname "$0")/.."
B=./build-wsl/laminar_babi_cmp
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o "$B" 2>&1 | grep -E "error" | head -5
run() { local name="$1"; shift
  "$B" file=data/babi_shuf_x3.txt test=data/babi_d10_test.jsonl units=50000 kpack=1 seq_topic_gain=8 seq_chain=0 hops=4 hops_free=2 questions=1000 show=0 "$@" > "logs/ct_$name.log" 2>&1; }
C2="levels=2 lvl_margin=2"
run l0            levels=0 &
run l2            $C2 &
run l2_goal       $C2 goal_gain=8 goal_elig=1 &
run l2_feed       $C2 lvl_feed=1 &
run l2_abl1       $C2 ablate=level:1 &
run l2_abl2       $C2 ablate=level:2 &
run l2_goal_abl1  $C2 goal_gain=8 goal_elig=1 ablate=level:1 &
run l2_feed_abl1  $C2 lvl_feed=1 ablate=level:1 &
wait
for n in l0 l2 l2_goal l2_feed l2_abl1 l2_abl2 l2_goal_abl1 l2_feed_abl1; do
  printf "%-13s %-9s %s\n" "$n" "$(grep -a '^BABI' logs/ct_$n.log | awk '{split($3,a,"/"); c+=a[1]; t+=a[2]} END {printf "%d/%d", c, t}')" "$(grep -a '^BABI' logs/ct_$n.log | awk '{split($3,a,"/"); printf "%s%s ", $2, a[1]}')"
done
for n in l0 l2 l2_goal l2_feed; do echo "$n: $(grep -a '^trained' logs/ct_$n.log | cut -c10-80)"; done
grep -a "composer:" logs/ct_l2.log | cut -c1-300; grep -a "L1 sample" logs/ct_l2.log | cut -c1-400; grep -a "L2 sample" logs/ct_l2.log | cut -c1-400
