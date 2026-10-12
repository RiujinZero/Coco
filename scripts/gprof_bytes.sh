#!/usr/bin/env bash
# Function-level profile of the byte path without perf: gprof build (-O2 -pg, no LTO so symbols survive).
#   scripts/gprof_bytes.sh [file] [units] [extra params...]
set -euo pipefail
cd "$(dirname "$0")/.."
file="${1:-data/tinystories_100k.txt}"; units="${2:-50000}"; shift 2 2>/dev/null || true
mkdir -p build-prof
g++ -std=c++20 ${OPT:--O2} -march=native -DNDEBUG -pg -fno-omit-frame-pointer -Icpp/include \
    cpp/apps/laminar_bytes.cpp cpp/src/organism.cpp -o build-prof/laminar_bytes
( cd build-prof && ./laminar_bytes file="../$file" units="$units" epochs=3 quiet=1 timers=0 "$@" | grep -E '^TOTAL' )
gprof -b -p build-prof/laminar_bytes build-prof/gmon.out | head -30
