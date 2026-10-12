#!/usr/bin/env bash
# Stay-on-task: does the bubble tier (attention by relation between thoughts) keep a reply on the user's topic?
#   scripts/ontask.sh [stream] [units] [dialogues] [extra params...]
set -uo pipefail
cd "$(dirname "$0")/.."
file="${1:-data/smoltalk.txt}"; units="${2:-50000}"; nd="${3:-40}"; shift 3 2>/dev/null || true
for tg in ${GAINS:-0 2 8}; do
  echo "=== topic_gain=$tg $* ==="
  "${BIN:-./build-wsl/laminar_chat}" file="$file" holdout="$file.holdout.jsonl" dialogues="$nd" reply=400 units="$units" kpack=0 seq_min_support=1 seq_copy_max=32 seq_topic_gain="$tg" show=1 "$@" 2>&1 \
    | grep -aE "^trained|^seqmem|^STAY|^\[[0-2]\]|^     GEN" | cut -c1-260
done
