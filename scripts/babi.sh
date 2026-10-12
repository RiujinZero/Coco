#!/usr/bin/env bash
# bAbI reasoning test: exact-match answers after training on the bAbI chat stream (python/babi_stream.py).
#   scripts/babi.sh [stream_prefix] [units] [extra params...]      GAINS="0 8" selects seq_topic_gain values
set -uo pipefail
cd "$(dirname "$0")/.."
pre="${1:-data/babi_t1-3-15-16}"; units="${2:-50000}"; shift 2 2>/dev/null || true
for tg in ${GAINS:-0 8}; do
  echo "=== topic_gain=$tg $* ==="
  "${BIN:-./build-wsl/laminar_babi}" file="$pre"_train.txt test="$pre"_test.jsonl units="$units" kpack=0 seq_min_support=1 seq_topic_gain="$tg" show=6 "$@" 2>&1 \
    | grep -aE "^trained|^seqmem|^BABI|^\[" | cut -c1-400
done
