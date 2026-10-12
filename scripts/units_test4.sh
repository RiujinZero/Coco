#!/usr/bin/env bash
# Merge tree exposed + relative split rule; with and without letting the vocabulary settle first.
set -uo pipefail
cd "$(dirname "$0")/.."
B=./build-wsl/laminar_babi_cmp
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o "$B" 2>&1 | grep -E "error" | head -4
run() { local name="$1"; shift
  "$B" file=data/babi_shuf_x3.txt test=data/babi_d10_test.jsonl units=50000 kpack=1 seq_topic_gain=8 seq_chain=2 seq_max_depth=8 hops=4 hops_free=2 questions=1000 show=0 "$@" > "logs/ut4_$name.log" 2>&1; }
run base      levels=0 &
run tree      levels=2 seq_units=1 lvl_minc=20 &
run tree_warm levels=2 seq_units=1 lvl_minc=20 lvl_warm=7000000 &
run tree_warm2 levels=2 seq_units=1 lvl_minc=50 lvl_warm=14000000 &
wait
for n in base tree tree_warm tree_warm2; do
  printf "%-11s %-9s %s\n" "$n" "$(grep -a '^BABI' logs/ut4_$n.log | awk '{split($3,a,"/"); c+=a[1]; t+=a[2]} END {printf "%d/%d", c, t}')" "$(grep -a '^BABI' logs/ut4_$n.log | awk '{split($3,a,"/"); printf "%s%s ", $2, a[1]}')"
done
grep -ao "thoughts:.*" logs/ut4_tree_warm2.log | cut -c1-360
