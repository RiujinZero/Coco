#!/usr/bin/env bash
# The block alone on the MIXED judgment stream, scored on the bAbI 1-10 design set: does it learn or only track the burst?
#   scripts/attn_mixed.sh "ARGS1" "ARGS2" ...
set -uo pipefail
cd "$(dirname "$0")/.."
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/attn_probe.cpp -o build-wsl/attn_probe 2>&1 | grep error | head -5
i=0
for args in "$@"; do i=$((i + 1))
  # shellcheck disable=SC2086
  ( ./build-wsl/attn_probe file=data/judge_train.txt test=data/babi_d10_test.jsonl mode=1 $args 2>/dev/null > "logs/am_$i.log"; printf "%-34s %s  %s\n" "$args" "$(grep TOTAL logs/am_$i.log)" "$(grep -E '^qa' logs/am_$i.log | awk '{split($2,a,"/"); printf "%s:%s ", $1, a[1]}')" ) &
done
wait
