#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-wsl build-asan logs/event_grounding_native
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/event_grounding_probe.cpp cpp/src/organism.cpp -o build-wsl/event_grounding_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_event_grounding.cpp -o build-asan/test_event_grounding
build-asan/test_event_grounding
for seed in 0 1 2; do
    build-wsl/event_grounding_probe logs/reasoning_composition_verified logs/semantic_grounding_corpus logs/event_grounding_corpus "$seed" "logs/event_grounding_native/$seed.json"
done
