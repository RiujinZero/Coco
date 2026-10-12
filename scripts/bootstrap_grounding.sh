#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-wsl build-asan logs/bootstrap_grounding
python3 python/bootstrap_grounding_probe.py
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/bootstrap_grounding_probe.cpp cpp/src/organism.cpp -o build-wsl/bootstrap_grounding_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_bootstrap_grounding.cpp -o build-asan/test_bootstrap_grounding
build-asan/test_bootstrap_grounding
for seed in 0 1 2; do
    build-wsl/bootstrap_grounding_probe "$seed"
done
build-wsl/bootstrap_grounding_probe 0 opaque
python3 python/check_bootstrap_grounding.py
