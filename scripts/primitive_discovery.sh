#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/primitive_discovery build-wsl build-asan
if [[ ! -f logs/reasoning_flat_corpus/train_0.txt || ! -f logs/reasoning_flat_corpus/test_2.txt ]]; then
  python3 python/reasoning_flat_probe.py --seeds 3 --export-native --out logs/reasoning_flat_corpus
fi
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/primitive_discovery_probe.cpp -o build-wsl/primitive_discovery_probe
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/primitive_transfer_probe.cpp -o build-wsl/primitive_transfer_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_policy_discovery.cpp -o build-asan/test_policy_discovery
build-asan/test_policy_discovery
for seed in 0 1 2; do
  for task in 0 1 2; do
    build-wsl/primitive_discovery_probe "$seed" "$task" search 100000
  done
  for mode in first terminal random frozen; do
    build-wsl/primitive_discovery_probe "$seed" 2 "$mode" 100000
  done
  build-wsl/primitive_discovery_probe "$seed" 2 backprop 60000
  build-wsl/primitive_transfer_probe "$seed" replay
  build-wsl/primitive_transfer_probe "$seed" no_replay
done
python3 python/check_primitive_discovery.py
