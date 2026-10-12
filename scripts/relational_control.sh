#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/relational_control_empirical build-wsl build-asan
g++ -std=c++20 -O3 -DNDEBUG -Icpp/include cpp/apps/relational_control_probe.cpp -o build-wsl/relational_control_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_relational_control.cpp -o build-asan/test_relational_control
build-asan/test_relational_control
python3 python/relational_control_probe.py --out logs/relational_control_empirical
for seed in 0 1 2; do
  for mode in learned direct frozen shuffled terminal; do
    dir="logs/relational_control_empirical/$seed"
    build-wsl/relational_control_probe "$dir/input.txt" "$dir/environment.txt" "$dir/worlds.txt" "$seed" "$mode" "$dir/$mode" > "$dir/$mode.txt"
  done
done
python3 python/check_relational_control.py
