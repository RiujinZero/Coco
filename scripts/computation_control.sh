#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/computation_control build-wsl build-asan
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/computation_control_probe.cpp -o build-wsl/computation_control_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_backprop_choice.cpp -o build-asan/test_backprop_choice
build-asan/test_backprop_choice
for seed in 0 1 2; do
  for mode in learned frozen shuffled terminal backprop backprop_frozen backprop_shuffled backprop_terminal; do
    build-wsl/computation_control_probe "$seed" 30000 "$mode"
  done
done
python3 python/check_computation_control.py
