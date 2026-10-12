#!/usr/bin/env bash
# Round 3: does lever 3 (chain depth 8 + learned END + abstention) generalize, and does it regress bAbI?
#   scripts/levers3.sh [units]
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-50000}"; B=./build-wsl/laminar_babi_lv
run() { local name="$1" stream="$2"; shift 2
  "$B" file="data/${stream}_train.txt" test="data/${stream}_test.jsonl" units="$units" kpack=1 seq_topic_gain=8 show=3 questions=2000 "$@" > "logs/lever_$name.log" 2>&1; }
run shift_d3      diag_shift seq_chain=2 &
run shift_d8      diag_shift seq_chain=2 seq_max_depth=8 &
run babi_d3       babi_all seq_chain=2 &
run babi_d8       babi_all seq_chain=2 seq_max_depth=8 &
run babi_d8_noab  babi_all seq_chain=2 seq_max_depth=8 seq_abstain=0 &
wait
echo "## unseen objects/containers in the test chains (train identical to the plain diagnostic)"
for n in shift_d3 shift_d8; do f="logs/lever_$n.log"; printf "%-10s |" "$n"; for k in 1 2 3 4 5 6 7 8; do printf " %2s" "$(grep -a "^BABI hop$k:" "$f" | grep -oE "[0-9]+/" | head -1 | tr -d /)"; done; echo; done
echo "## bAbI 20 tasks, 100 held-out each"
printf "%-13s" task; for n in babi_d3 babi_d8 babi_d8_noab; do printf " %12s" "$n"; done; echo
for t in $(seq 1 20); do printf "%-13s" "qa$t"; for n in babi_d3 babi_d8 babi_d8_noab; do printf " %12s" "$(grep -a "^BABI qa$t:" "logs/lever_$n.log" | grep -oE "[0-9]+/[0-9]+" | head -1)"; done; echo; done
for n in babi_d3 babi_d8 babi_d8_noab; do printf "%-13s %s\n" "$n" "$(grep -a "^trained" logs/lever_$n.log | grep -oE "[0-9]+ KB/s")"; done
