#!/usr/bin/env bash
# Accumulator width: INT8 (default) vs INT16 (acc_bits=16) on the synth task and on TinyStories bytes.
#   scripts/acc_test.sh [max_bytes] [units]
set -uo pipefail
cd "$(dirname "$0")/.."
mb="${1:-8388608}"; units="${2:-50000}"
for a in synth bytes; do
  g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include "cpp/apps/laminar_$a.cpp" cpp/src/organism.cpp -o "build-wsl/laminar_${a}_acc" 2>&1 | grep -E "error" | head -5
done
for bits in 8 16; do
  echo "##### synth 8ctx, 4096 units, acc_bits=$bits"
  ./build-wsl/laminar_synth_acc units=4096 train_events=120000 eval_events=16000 acc_bits="$bits" 2>&1 | grep -aE "frozen|FROZEN|accuracy|acc " | tail -2 | cut -c1-200
  echo "##### TinyStories $((mb / 1048576)) MB, $units units, acc_bits=$bits"
  ./build-wsl/laminar_bytes_acc file=data/tinystories_13m.txt max_bytes="$mb" units="$units" epochs=1 acc_bits="$bits" 2>&1 | grep -aE "^TOTAL|^epoch|BPB|acc_updates|level" | tail -3 | cut -c1-260
done
