#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:-logs/composed_control_raw}"
mkdir -p "$root" build-wsl
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/relational_control_probe.cpp -o build-wsl/composed_control_raw_probe
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/primitive_checkpoint_probe.cpp -o build-wsl/primitive_checkpoint_probe
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/verify_local_choices.cpp -o build-wsl/verify_local_choices
for seed in 0 1 2; do
  bash scripts/composed_control_seed.sh "$seed" "$root"
done
