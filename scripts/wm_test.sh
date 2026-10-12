#!/usr/bin/env bash
# White-matter probe: attention between motifs across recent bytes, at roughly the organism's own cost.
#   scripts/wm_test.sh [file] [bytes] [extra...]      -> logs/wm_<arm>.log
set -uo pipefail
cd "$(dirname "$0")/.."
f="${1:-data/tinystories_13m.txt}"; mb="${2:-8388608}"; shift 2 2>/dev/null || true
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/laminar_wm.cpp cpp/src/organism.cpp -o build-wsl/laminar_wm 2>&1 | grep -E "error" | head -5
arm() { local name="$1"; shift; ./build-wsl/laminar_wm file="$f" max_bytes="$mb" units=50000 kpack=1 hops=4 hops_free=2 "$@" > "logs/wm_$name.log" 2>&1; }
arm off   wm=off "$@" &
arm motif wm=motif "$@" &
arm sum   wm=sum "$@" &
arm sym   wm=sym "$@" &
arm now   wm=now "$@" &
wait
for n in off motif sum sym now; do printf "%-6s %s\n" "$n" "$(grep -a '^TOTAL' logs/wm_$n.log | cut -c8-)"; done
