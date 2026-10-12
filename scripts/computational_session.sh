#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:-logs/computational_session}"
mkdir -p "$root" build-wsl
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/computational_session_probe.cpp -o build-wsl/computational_session_probe
for seed in 0 1 2; do
  build-wsl/computational_session_probe "$seed" "$root" | tee "$root/$seed.txt"
done
python3 python/check_computational_session.py --out "$root"
