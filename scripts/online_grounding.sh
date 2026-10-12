#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-wsl build-asan logs/online_grounding
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/online_grounding_probe.cpp cpp/src/organism.cpp -o build-wsl/online_grounding_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_online_grounding.cpp -o build-asan/test_online_grounding
build-asan/test_online_grounding
for seed in 0 1 2; do
    build-wsl/online_grounding_probe "$seed"
done
python3 python/check_online_grounding.py
