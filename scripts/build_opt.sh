#!/usr/bin/env bash
# Rebuild every harness on the current code as *_opt binaries, then run the bAbI regression with learned reading.
#   scripts/build_opt.sh
set -uo pipefail
cd "$(dirname "$0")/.."
for a in train babi chat generate bytes; do
  g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include "cpp/apps/laminar_$a.cpp" cpp/src/organism.cpp -o "build-wsl/laminar_${a}_opt" 2>&1 | grep -E "error" | head -3
  [ -x "build-wsl/laminar_${a}_opt" ] && echo "built laminar_${a}_opt" || echo "FAILED laminar_${a}_opt"
done
echo "##### bAbI 5 tasks, learned reading, optimised build"
BIN=./build-wsl/laminar_babi_opt GAINS="8" ./scripts/babi.sh data/babi_t1-3-15-16 50000 seq_chain=2 show=0
echo "##### bAbI shifted, learned reading, optimised build"
BIN=./build-wsl/laminar_babi_opt GAINS="8" ./scripts/babi.sh data/babi_shift 50000 seq_chain=2 show=0
