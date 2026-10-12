#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-wsl build-asan logs/semantic_grounding_native
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/semantic_grounding_probe.cpp cpp/src/organism.cpp -o build-wsl/semantic_grounding_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_semantic_grounding.cpp -o build-asan/test_semantic_grounding
build-asan/test_semantic_grounding
for seed in 0 1 2; do
    build-wsl/semantic_grounding_probe logs/reasoning_composition_verified logs/semantic_grounding_corpus "$seed" "logs/semantic_grounding_native/$seed.json"
done
