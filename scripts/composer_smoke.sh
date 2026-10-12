#!/usr/bin/env bash
# composer smoke test: segmentation margin x levels on 2 MB TinyStories
set -uo pipefail
cd "$(dirname "$0")/.."
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/laminar_bytes.cpp cpp/src/organism.cpp -o build-wsl/laminar_bytes_cmp 2>&1 | grep -E "error" | head -8
run() { local name="$1"; shift; ./build-wsl/laminar_bytes_cmp file=data/tinystories_13m.txt max_bytes=2097152 units=50000 kpack=1 hops=4 hops_free=2 quiet=1 lvl_motifs=48 "$@" > "logs/cmp_$name.log" 2>&1; }
run l0 levels=0 &
run l1_m15 levels=1 lvl_margin=1.5 &
run l1_m2 levels=1 lvl_margin=2 &
run l1_m15f levels=1 lvl_margin=1.5 lvl_feed=1 &
run l2_m15 levels=2 lvl_margin=1.5 &
run l3_m15 levels=3 lvl_margin=1.5 &
wait
for n in l0 l1_m15 l1_m2 l1_m15f l2_m15 l3_m15; do echo "== $n"; grep -a "^TOTAL" logs/cmp_$n.log | cut -c1-110; grep -a "composer:\|sample:" logs/cmp_$n.log | cut -c1-520; done
