#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/correction_language build-wsl build-asan
python3 python/correction_language_probe.py
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/correction_language_probe.cpp cpp/src/organism.cpp -o build-wsl/correction_language_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_correction_language_grounding.cpp -o build-asan/test_correction_language_grounding
build-asan/test_correction_language_grounding
for seed in 0 1 2; do
    build-wsl/correction_language_probe "$seed"
done
python3 python/check_correction_language.py
