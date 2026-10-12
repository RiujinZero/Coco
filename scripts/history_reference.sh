#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/history_reference build-wsl build-asan
python3 python/history_reference_probe.py
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/history_reference_probe.cpp cpp/src/organism.cpp -o build-wsl/history_reference_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_history_reference_grounding.cpp -o build-asan/test_history_reference_grounding
build-asan/test_history_reference_grounding
for seed in 0 1 2; do
    build-wsl/history_reference_probe "$seed"
done
python3 python/check_history_reference.py
