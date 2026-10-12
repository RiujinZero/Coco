#!/usr/bin/env bash
# AddressSanitizer build of laminar_needle for crash hunting.
#   scripts/asan_needle.sh [params...]
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-asan
g++ -std=c++20 -O1 -g -march=native -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include \
    cpp/apps/laminar_needle.cpp cpp/src/organism.cpp -o build-asan/laminar_needle || exit 1
./build-asan/laminar_needle "$@" 2>&1 | grep -vE '^needle|^  distance' | head -40
