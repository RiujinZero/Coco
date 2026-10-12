#!/usr/bin/env bash
# Fast sweeps of the answer-attention block alone (build-wsl/attn_probe) on a prefix of a stream.
#   scripts/attn_sweep.sh BYTES "ARGSET1" "ARGSET2" ...      e.g. scripts/attn_sweep.sh 700000 "lr=0.05" "lr=0.2 clip=5"
set -uo pipefail
cd "$(dirname "$0")/.."
bytes="$1"; shift
head -c "$bytes" data/babi_all_train.txt > /tmp/attn_prefix.txt
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/attn_probe.cpp -o build-wsl/attn_probe 2>&1 | grep error | head -5
for args in "$@"; do
  # shellcheck disable=SC2086
  out=$(./build-wsl/attn_probe file=/tmp/attn_prefix.txt $args 2>&1 | grep -a "^attn:" | tail -3 | sed -E 's/attn: ([0-9]+) answers \[([a-z]+)\] acc ([0-9.]+).*spoke ([0-9.]+) of turns with precision ([0-9.]+).*/\1:\2 acc \3 spoke \4 prec \5/' | tr '\n' '|')
  printf "%-40s %s\n" "$args" "$out"
done
