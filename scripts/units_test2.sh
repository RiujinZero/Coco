#!/usr/bin/env bash
# Learned-merge composer feeding the memory side (chain ON), shuffled bAbI 1 pass, held-out bAbI 1-10.
set -uo pipefail
cd "$(dirname "$0")/.."
B=./build-wsl/laminar_babi_cmp
run() { local name="$1"; shift
  "$B" file=data/babi_all_train_shuf.txt test=data/babi_d10_test.jsonl units=50000 kpack=1 seq_topic_gain=8 seq_chain=2 seq_max_depth=8 hops=4 hops_free=2 questions=1000 show=0 "$@" > "logs/ut2_$name.log" 2>&1; }
run base levels=0 &
run merge_u1 levels=2 seq_units=1 &
run merge_u2 levels=2 seq_units=2 &
run merge_u1c levels=2 seq_units=1 lvl_minc=20 &
wait
for n in base merge_u1 merge_u2 merge_u1c; do
  printf "%-10s %-9s %s\n" "$n" "$(grep -a '^BABI' logs/ut2_$n.log | awk '{split($3,a,"/"); c+=a[1]; t+=a[2]} END {printf "%d/%d", c, t}')" "$(grep -ao 'composer: L1 [^;]*;' logs/ut2_$n.log)"
done
grep -a "L1 sample" logs/ut2_merge_u1.log | cut -c1-320
