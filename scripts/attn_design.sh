#!/usr/bin/env bash
# Learned answer attention, design set only (bAbI tasks 1-10): off / backprop throughout / warm start then forward
# gradients / warm start then frozen / forward gradients from scratch.
#   scripts/attn_design.sh [stream] [warm] [extra...]      -> logs/ad_<arm>.log, summary on stdout
set -uo pipefail
cd "$(dirname "$0")/.."
stream="${1:-data/babi_all_train.txt}"; warm="${2:-10000}"; shift 2 2>/dev/null || true
B=./build-wsl/laminar_babi_attn
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o "$B" 2>&1 | grep -E "error" | head -5
run() { local name="$1"; shift
  "$B" file="$stream" test=data/babi_d10_test.jsonl units=50000 kpack=1 seq_topic_gain=8 seq_chain=2 seq_max_depth=8 hops=4 hops_free=2 \
       show=1 questions=100000 dump="logs/ad_${name}_replies.jsonl" "$@" > "logs/ad_$name.log" 2>&1; }
run off  seq_attn=0 "$@" &
run bp   seq_attn=1 "$@" &
run wfg  seq_attn=2 attn_warm="$warm" "$@" &
run wfz  seq_attn=3 attn_warm="$warm" "$@" &
run fg   seq_attn=4 "$@" &
wait
for n in off bp wfg wfz fg; do
  tot=$(grep -a "^BABI" logs/ad_$n.log | awk '{split($3,a,"/"); c+=a[1]; t+=a[2]} END {printf "%d/%d", c, t}')
  printf "%-4s %-10s %s\n" "$n" "$tot" "$(grep -a '^BABI' logs/ad_$n.log | awk '{split($3,a,"/"); printf "%s:%s ", $2, a[1]}')"
done
for n in bp wfg wfz fg; do echo "--- $n"; grep -a "^attn:" logs/ad_$n.log | tail -4; done
