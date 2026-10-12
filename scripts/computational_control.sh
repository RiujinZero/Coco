#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:-logs/computational_control}"
seeds="${2:-3}"
episodes="${3:-400}"
mkdir -p "$root" build-wsl
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/relational_control_probe.cpp -o build-wsl/relational_control_probe
python3 python/computational_control_probe.py --seeds "$seeds" --episodes "$episodes" --out "$root"
for ((seed=0;seed<seeds;seed++)); do
  dir="$root/$seed"
  for mode in learned frozen shuffled terminal; do
    build-wsl/relational_control_probe "$dir/input.txt" "$dir/environment.txt" "$dir/worlds.txt" "$seed" "$mode" "$dir/$mode" computational > "$dir/$mode.txt"
  done
done
python3 python/check_relational_control.py --root "$root" --seeds "$seeds" --modes learned frozen shuffled terminal
