#!/usr/bin/env bash
# SynLogic learning curves: one organism per training amount (quarter, half, all, two passes), same held-out puzzles.
#   scripts/synlogic_curve.sh [units]      -> logs/sl_<n>.log, logs/sl_<n>_replies.jsonl, logs/sl_curve.txt
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-200000}"; B=./build-wsl/laminar_babi_sl
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o "$B" 2>&1 | grep -E "error" | head -5
[ -f data/synlogic/sl_x2_train.txt ] || cat data/synlogic/sl_train.txt data/synlogic/sl_train.txt > data/synlogic/sl_x2_train.txt
cp -f data/synlogic/sl_test.jsonl data/synlogic/sl_x2_test.jsonl
full=$(stat -c %s data/synlogic/sl_train.txt)
run() { local name="$1" stream="$2" mb="$3"
  "$B" file="data/synlogic/${stream}_train.txt" test="data/synlogic/${stream}_test.jsonl" max_bytes="$mb" units="$units" kpack=1 \
       seq_chain=2 seq_max_depth=8 seq_topic_gain=8 gen_auto=1 show=2 questions=1000 dump="logs/sl_${name}_replies.jsonl" > "logs/sl_$name.log" 2>&1; }
run q  sl    $((full / 4)) &
run h  sl    $((full / 2)) &
run f  sl    0 &
run x2 sl_x2 0 &
wait
for n in q h f x2; do printf "%-3s %s\n" "$n" "$(grep -a '^trained' logs/sl_$n.log | cut -c1-120)"; done
python3 python/synlogic_score.py logs/sl_q_replies.jsonl logs/sl_h_replies.jsonl logs/sl_f_replies.jsonl logs/sl_x2_replies.jsonl | tee logs/sl_curve.txt
