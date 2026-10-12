#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:-logs/composed_lifecycle}"
source_prefix="${2:-logs/composed_credit/boolean/1}"
mkdir -p "$root" build-wsl
python3 python/composed_lifecycle_probe.py --source "$source_prefix" --out "$root/input.txt"
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/composed_lifecycle_probe.cpp -o build-wsl/composed_lifecycle_probe
build-wsl/composed_lifecycle_probe 1 "$root/input.txt" "$root/1" | tee "$root/1.txt"
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/primitive_checkpoint_probe.cpp -o build-wsl/primitive_checkpoint_probe
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/verify_local_choices.cpp -o build-wsl/verify_local_choices
build-wsl/primitive_checkpoint_probe 1 "$root/1.final.weights.txt" "$root/1.checkpoint"
python3 python/check_composed_lifecycle.py --root "$root" --source "$source_prefix"
