#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-wsl build-asan logs/live_observation_grounding
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/live_observation_probe.cpp cpp/src/organism.cpp -o build-wsl/live_observation_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_live_observation_grounding.cpp -o build-asan/test_live_observation_grounding
build-asan/test_live_observation_grounding
for seed in 0 1 2; do
    build-wsl/live_observation_probe "$seed"
done
python3 python/check_live_observation_grounding.py
