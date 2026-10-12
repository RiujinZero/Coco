#!/usr/bin/env bash
# The handoff test for the answer-attention block alone (fruitfly's question): does learning survive the switch
# from backprop to forward gradients? Three passes of the shuffled bAbI stream (60k answers); the warm start is the
# first pass. Arms run in parallel; held-out bAbI 1-10 accuracy and the training windows around the switch.
#   scripts/attn_handoff.sh [lr]
set -uo pipefail
cd "$(dirname "$0")/.."
lr="${1:-0.1}"
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/attn_probe.cpp -o build-wsl/attn_probe 2>&1 | grep error | head -5
arm() { local name="$1"; shift
  ./build-wsl/attn_probe file=data/babi_all_train_shuf.txt test=data/babi_d10_test.jsonl passes=3 lr="$lr" "$@" > "logs/ho_$name.log" 2>&1; }
arm bp_all   mode=1 &
arm warm_frz mode=3 warm=20000 &
arm fg_003   mode=2 warm=20000 fg_lr=0.003 &
arm fg_010   mode=2 warm=20000 fg_lr=0.01 &
arm fg_030   mode=2 warm=20000 fg_lr=0.03 &
arm fg_010k4 mode=2 warm=20000 fg_lr=0.01 fg_k=4 &
arm fg_030k4 mode=2 warm=20000 fg_lr=0.03 fg_k=4 &
arm fg_only  mode=4 fg_lr=0.03 fg_k=4 &
wait
for n in bp_all warm_frz fg_003 fg_010 fg_030 fg_010k4 fg_030k4 fg_only; do
  printf "%-9s %-22s windows: %s\n" "$n" "$(grep -a TOTAL logs/ho_$n.log)" \
    "$(grep -aE '^attn: (18000|20000|22000|24000|30000|40000|50000|60000) ' logs/ho_$n.log | sed -E 's/attn: ([0-9]+) answers \[([a-z]+)\] acc ([0-9.]+).*/\1:\3/' | tr '\n' ' ')"
done
