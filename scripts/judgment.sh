#!/usr/bin/env bash
# Frozen judgment sets for the generic query primitive (do not design against these):
#   long-chat aggregation (python/partychat.py), ProofWriter (python/proofwriter_stream.py), bAbI (no regression).
# SynLogic object_counting / web_of_lies come from scripts/synlogic_curve.sh (logs/sl_*_replies.jsonl).
#   scripts/judgment.sh TAG [units] [extra params...]      -> logs/judge_TAG.log, logs/judge_TAG_replies.jsonl
set -uo pipefail
cd "$(dirname "$0")/.."
tag="$1"; units="${2:-100000}"; shift 2 2>/dev/null || shift 1
B="${BIN:-./build-wsl/laminar_babi_judge}"
[ -n "${BIN:-}" ] || g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o "$B" 2>&1 | grep -E "error" | head -5
[ -f data/judge_train.txt ] || python3 python/mix_stream.py data/judge_train.txt data/partychat_train.txt data/proofwriter_train.txt data/babi_all_train.txt data/smoltalk.txt
[ -f data/judge_test.jsonl ] || cat data/partychat_test.jsonl data/proofwriter_test.jsonl data/babi_all_test.jsonl > data/judge_test.jsonl
"$B" file=data/judge_train.txt test=data/judge_test.jsonl units="$units" kpack=1 seq_topic_gain=8 seq_chain=2 seq_max_depth=8 hops=4 hops_free=2 \
     show=2 questions=100000 dump="logs/judge_${tag}_replies.jsonl" "$@" > "logs/judge_$tag.log" 2>&1
grep -aE "^trained|^BABI" "logs/judge_$tag.log"
