#!/usr/bin/env bash
# Frozen judgment sets with the learned answer attention (question from the organism's state, held-out trust,
# chain proposal): warm start then forward gradients, vs the fixed baseline (logs/judge_fix_base.log).
set -uo pipefail
cd "$(dirname "$0")/.."
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o build-wsl/laminar_babi_judge 2>&1 | grep -E "error" | head -5
BIN=./build-wsl/laminar_babi_judge scripts/judgment.sh q_fg 100000 seq_attn=2 attn_lr=0.03 attn_warm=100000 > logs/judge_q_fg.out 2>&1 &
BIN=./build-wsl/laminar_babi_judge scripts/judgment.sh q_fg50 100000 seq_attn=2 attn_lr=0.03 attn_warm=50000 > logs/judge_q_fg50.out 2>&1 &
wait
echo done
