#!/usr/bin/env bash
# Goal expansion by learned relations, at LMSYS scale (343 held-out dialogues): off vs on. Detached; log in logs/.
#   scripts/goalrel_lmsys.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
{
  echo "##### LMSYS 128 MB, learned reading + relations 0.25, goal expansion OFF"
  BIN=./build-wsl/laminar_chat_gr GAINS="8" ./scripts/ontask.sh data/lmsys_128m.txt 100000 400 seq_chain=2 seq_rel_gain=0.25 show=0
  echo "##### LMSYS 128 MB, learned reading + relations 0.25, goal expansion 1"
  BIN=./build-wsl/laminar_chat_gr GAINS="8" ./scripts/ontask.sh data/lmsys_128m.txt 100000 400 seq_chain=2 seq_rel_gain=0.25 seq_goal_rel=1 show=0
  echo "##### DONE"
} > logs/goalrel_lmsys.log 2>&1
