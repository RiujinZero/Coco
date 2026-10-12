#!/usr/bin/env bash
# Hybrid attention after the motif bodies (~0.93M params): does it add anything beyond "a small transformer"?
#   scripts/hybrid_test.sh [file] [bytes]      -> logs/hy_<arm>.log
set -uo pipefail
cd "$(dirname "$0")/.."
f="${1:-data/tinystories_13m.txt}"; mb="${2:-8388608}"
g++ -std=c++20 -O3 -march=native -fopenmp -DNDEBUG -Icpp/include cpp/apps/laminar_hybrid.cpp cpp/src/organism.cpp -o build-wsl/laminar_hybrid 2>&1 | grep -E "error" | head -5
arm() { local name="$1"; shift; OMP_NUM_THREADS=7 ./build-wsl/laminar_hybrid file="$f" max_bytes="$mb" units=50000 kpack=1 hops=4 hops_free=2 progress=1048576 "$@" > "logs/hy_$name.log" 2>&1; }
arm org        head=0 &
arm org_head   state=1 base=1 &
arm org_nostate state=0 base=1 &
arm head_only  org=0 &
wait
for n in org org_head org_nostate head_only; do printf "%-12s %s\n" "$n" "$(grep -a '^TOTAL' logs/hy_$n.log | cut -c8-)"; done
