#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:-logs/composed_credit}"
seeds="${2:-3}"
budget="${3:-100000}"
mkdir -p "$root" build-wsl
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/composed_credit_probe.cpp -o build-wsl/composed_credit_full_probe
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/verify_local_choices.cpp -o build-wsl/verify_local_choices
for ((seed=0;seed<seeds;seed++)); do
  bash scripts/composed_credit_seed.sh "$seed" "$root" "$budget"
done
python3 python/check_composed_credit.py --root "$root" --seeds "$seeds"
