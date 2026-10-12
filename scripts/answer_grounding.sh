#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-wsl build-asan logs/answer_grounding_native
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/answer_grounding_probe.cpp cpp/src/organism.cpp -o build-wsl/answer_grounding_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_answer_grounding.cpp -o build-asan/test_answer_grounding
build-asan/test_answer_grounding
for seed in 0 1 2; do
    build-wsl/answer_grounding_probe "$seed" "logs/answer_grounding_native/$seed.json"
done
