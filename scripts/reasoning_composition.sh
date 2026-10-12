#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
corpus="${1:-logs/reasoning_composition_verified}"
output="${2:-logs/reasoning_composition_native_verified}"
mkdir -p "$output"
g++ -std=c++20 -O3 -Icpp/include cpp/apps/reasoning_composition_probe.cpp -o build-wsl/reasoning_composition_probe
for seed in 0 1 2 3 4; do
    for mode in outcome monte_carlo replay broadcast broadcast_long shared_error random_reward; do
        build-wsl/reasoning_composition_probe "$corpus" "$seed" "$output/${seed}_${mode}.json" "$mode"
        echo "composition seed=$seed learning=$mode complete"
    done
    build-wsl/reasoning_composition_probe "$corpus" "$seed" "$output/${seed}_stress.json" outcome stress
    echo "composition seed=$seed stress complete"
done
