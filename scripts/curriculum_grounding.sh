#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-wsl build-asan logs/curriculum_grounding
python3 python/validate_curriculum_grounding.py
g++ -std=c++20 -O3 -march=native -Icpp/include cpp/apps/curriculum_grounding_probe.cpp cpp/src/organism.cpp -o build-wsl/curriculum_grounding_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_fact_knowledge.cpp -o build-asan/test_fact_knowledge
build-asan/test_fact_knowledge
build-wsl/curriculum_grounding_probe
python3 python/validate_curriculum_grounding.py --score
