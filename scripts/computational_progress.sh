#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_root="${1:-logs/computational_control}"
root="${2:-logs/computational_progress}"
seeds="${3:-3}"
mkdir -p "$root" build-wsl
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/relational_control_probe.cpp -o build-wsl/computational_progress_probe
for ((seed=0;seed<seeds;seed++)); do
  dir="$root/$seed"
  mkdir -p "$dir"
  for file in input.txt environment.txt worlds.txt oracle.json; do
    cp "$source_root/$seed/$file" "$dir/$file"
  done
  build-wsl/computational_progress_probe "$dir/input.txt" "$dir/environment.txt" "$dir/worlds.txt" "$seed" learned_progress "$dir/learned_progress" computational > "$dir/learned_progress.txt"
done
python3 python/check_relational_control.py --root "$root" --seeds "$seeds" --modes learned_progress
python3 python/check_computational_readout.py --source "$source_root" --root "$root" --seeds "$seeds" --modes learned_progress
