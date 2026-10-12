#!/usr/bin/env bash
# Learned reading (seq_chain=2) across the four probes: bAbI original, bAbI shifted, CLUTRR, SmolTalk dialogue.
#   scripts/reading_test.sh [extra params...]
set -uo pipefail
cd "$(dirname "$0")/.."
for a in babi chat; do
  g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include "cpp/apps/laminar_$a.cpp" cpp/src/organism.cpp -o "build-wsl/laminar_${a}_rd" 2>&1 | grep -E "error" | head -5
done
echo "##### bAbI original"; BIN=./build-wsl/laminar_babi_rd GAINS="8" ./scripts/babi.sh data/babi_t1-3-15-16 50000 seq_chain=2 show=0 "$@"
echo "##### bAbI shifted";  BIN=./build-wsl/laminar_babi_rd GAINS="8" ./scripts/babi.sh data/babi_shift 50000 seq_chain=2 show=0 "$@"
echo "##### CLUTRR";        BIN=./build-wsl/laminar_babi_rd GAINS="8" ./scripts/babi.sh data/clutrr_chat 50000 seq_chain=2 show=2 questions=400 "$@"
echo "##### SmolTalk";      BIN=./build-wsl/laminar_chat_rd GAINS="8" ./scripts/ontask.sh data/smoltalk.txt 50000 40 seq_chain=2 "$@"
