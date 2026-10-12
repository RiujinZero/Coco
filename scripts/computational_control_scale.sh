#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:-logs/computational_control}"
seeds="${2:-3}"
# Separate executable keeps an already-running baseline sweep on its own build.
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/relational_control_probe.cpp -o build-wsl/relational_control_scale_probe
for ((seed=0;seed<seeds;seed++)); do
  dir="$root/$seed"
  for mode in scaled scaled_frozen; do
    build-wsl/relational_control_scale_probe "$dir/input.txt" "$dir/environment.txt" "$dir/worlds.txt" "$seed" "$mode" "$dir/$mode" computational > "$dir/$mode.txt"
  done
done
