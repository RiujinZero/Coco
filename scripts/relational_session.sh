#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:-logs/relational_session}"
mkdir -p "$root" build-wsl build-asan
g++ -std=c++20 -O3 -DNDEBUG -Icpp/include cpp/apps/relational_session_probe.cpp -o build-wsl/relational_session_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_relational_session.cpp -o build-asan/test_relational_session
build-asan/test_relational_session
python3 python/relational_session_probe.py --out "$root"
for seed in 0 1 2; do
  build-wsl/relational_session_probe "$root/$seed/input.txt" "$root/$seed/environment.txt" "$seed" reuse > "$root/$seed/reuse.txt"
done
python3 python/check_relational_session.py --out "$root"
