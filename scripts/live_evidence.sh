#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/live_evidence build-wsl build-asan
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/live_evidence_probe.cpp cpp/src/organism.cpp -o build-wsl/live_evidence_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_live_evidence.cpp -o build-asan/test_live_evidence
build-asan/test_live_evidence
for seed in 0 1 2; do
    build-wsl/live_evidence_probe "$seed"
done
python3 python/check_live_evidence.py
