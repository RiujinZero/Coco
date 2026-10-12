#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/concept_growth build-wsl build-asan
g++ -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Icpp/include cpp/apps/concept_growth_probe.cpp -o build-wsl/concept_growth_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_concept_memory.cpp -o build-asan/test_concept_memory
build-asan/test_concept_memory
python3 python/concept_growth_probe.py
for seed in 0 1 2; do
  for mode in reuse flat wide; do
    build-wsl/concept_growth_probe "logs/concept_growth/$seed/input.txt" "$mode" > "logs/concept_growth/$seed/$mode.txt"
  done
done
python3 python/check_concept_growth.py
