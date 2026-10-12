#!/usr/bin/env bash
# Merge tree exposed: a unit splits into parts that stand on their own, and the chain matches on those.
set -uo pipefail
cd "$(dirname "$0")/.."
B=./build-wsl/laminar_babi_cmp
run() { local name="$1"; shift
  "$B" file=data/babi_all_train_shuf.txt test=data/babi_d10_test.jsonl units=50000 kpack=1 seq_topic_gain=8 seq_chain=2 seq_max_depth=8 hops=4 hops_free=2 questions=1000 show=1 "$@" > "logs/ut3_$name.log" 2>&1; }
run base     levels=0 &
run tree_c5  levels=2 seq_units=1 lvl_minc=5 &
run tree_c20 levels=2 seq_units=1 lvl_minc=20 &
run tree_c50 levels=2 seq_units=1 lvl_minc=50 &
run tree_s2  levels=2 seq_units=2 lvl_minc=20 &
wait
for n in base tree_c5 tree_c20 tree_c50 tree_s2; do
  printf "%-9s %-9s %s\n" "$n" "$(grep -a '^BABI' logs/ut3_$n.log | awk '{split($3,a,"/"); c+=a[1]; t+=a[2]} END {printf "%d/%d", c, t}')" "$(grep -a '^BABI' logs/ut3_$n.log | awk '{split($3,a,"/"); printf "%s%s ", $2, a[1]}')"
done
echo "--- what the chain sees (tree_c20, qa1):"; grep -a "^\[qa1\]" logs/ut3_tree_c20.log | head -2 | cut -c1-340
