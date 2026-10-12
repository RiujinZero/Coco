#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
g++ -std=c++20 -O3 -Icpp/include cpp/apps/reasoning_agency_probe.cpp -o build-wsl/reasoning_agency_probe
mkdir -p logs/reasoning_primitives_agency
for seed in 0 1 2; do
    for mode in default long; do
        build-wsl/reasoning_agency_probe logs/reasoning_primitives_corpus "$seed" 60000 "$mode" > "logs/reasoning_primitives_agency/${seed}_${mode}.json"
        echo "agency seed=$seed mode=$mode finished"
    done
done
build-wsl/reasoning_agency_probe logs/reasoning_primitives_corpus 0 60000 oracle > logs/reasoning_primitives_agency/oracle.json
