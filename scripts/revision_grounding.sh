#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/revision_grounding build-wsl build-asan
python3 python/revision_grounding_probe.py
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/revision_grounding_probe.cpp cpp/src/organism.cpp -o build-wsl/revision_grounding_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_revisable_grounding.cpp -o build-asan/test_revisable_grounding
build-asan/test_revisable_grounding
for seed in 0 1 2; do
    build-wsl/revision_grounding_probe "$seed"
done
python3 python/check_revision_grounding.py
