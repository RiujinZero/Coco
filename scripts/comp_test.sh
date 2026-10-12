#!/usr/bin/env bash
# Composition on CLUTRR: baseline paths; + endpoint attributes; + specific closures; + both.
#   scripts/comp_test.sh
set -uo pipefail
cd "$(dirname "$0")/.."
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o build-wsl/laminar_babi_cp 2>&1 | grep -E "error" | head -5
run() { echo "##### $1"; shift; BIN=./build-wsl/laminar_babi_cp GAINS="8" ./scripts/babi.sh data/clutrr_chat 50000 seq_chain=2 seq_rel_gain=1 show=0 questions=400 "$@" | grep -aE "^seqmem|^BABI" | cut -c1-400 | sed 's/BABI //' | grep -oE "relations.*lookups|paths [0-9]+ \(composed [0-9]+\)|clutrr[0-9]: [0-9/]+ = [0-9.]+%.*"; }
run "paths only (relations credited, no closures, no attributes)"
run "+ endpoint attributes" seq_attr=1
run "+ specific closures" seq_rel_closure=1
run "+ both" seq_attr=1 seq_rel_closure=1
