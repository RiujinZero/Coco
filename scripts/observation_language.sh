#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/observation_language build-wsl build-asan
python3 python/observation_language_probe.py
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/observation_language_probe.cpp cpp/src/organism.cpp -o build-wsl/observation_language_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_observation_language.cpp -o build-asan/test_observation_language
build-asan/test_observation_language
for seed in 0 1 2; do
    build-wsl/observation_language_probe "$seed"
done
python3 python/check_observation_language.py
