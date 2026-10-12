#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-wsl build-asan logs/reasoning_language_native
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/reasoning_language_probe.cpp cpp/src/organism.cpp -o build-wsl/reasoning_language_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_reasoning_language.cpp -o build-asan/test_reasoning_language
build-asan/test_reasoning_language
for seed in 0 1 2; do
    build-wsl/reasoning_language_probe logs/reasoning_composition_verified logs/reasoning_language_corpus "$seed" "logs/reasoning_language_native/$seed.json"
done
