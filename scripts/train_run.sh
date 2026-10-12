#!/usr/bin/env bash
# The real training run, detached, with probes at checkpoints (see cpp/apps/laminar_train.cpp).
#   scripts/train_run.sh [stream] [units] [probe_every] [extra params...]      -> logs/train_<tag>.log
set -uo pipefail
cd "$(dirname "$0")/.."
stream="${1:-data/mix_650m.txt}"; units="${2:-200000}"; every="${3:-67108864}"; shift 3 2>/dev/null || true
tag="$(date +%Y%m%d_%H%M)"
mkdir -p logs
nohup "${BIN:-./build-wsl/laminar_train}" file="$stream" units="$units" probe_every="$every" \
  heldout=data/wiki_valid_4m.txt chat=data/lmsys_128m.txt.holdout.jsonl qa=data/babi_all_test.jsonl,data/clutrr_chat_test.jsonl \
  kpack=1 seqmem=1 seq_topic_gain=8 seq_util_rate=0.2 seq_chain=2 seq_max_depth=8 hops=4 hops_free=2 agency=1 "$@" > "logs/train_$tag.log" 2>&1 < /dev/null &
disown
echo "started: logs/train_$tag.log (pid $!)"
