#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/reasoning_flat_native build-asan
g++ -std=c++20 -O3 -Icpp/include cpp/apps/reasoning_flat_probe.cpp -o build-wsl/reasoning_flat_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_reasoning_runtime.cpp -o build-asan/test_reasoning_runtime
build-asan/test_reasoning_runtime
for seed in 0 1 2 3 4; do
    build-wsl/reasoning_flat_probe logs/reasoning_composition_verified logs/reasoning_flat_corpus "$seed" "logs/reasoning_flat_native/$seed.json"
    echo "flat composition seed=$seed complete"
done
