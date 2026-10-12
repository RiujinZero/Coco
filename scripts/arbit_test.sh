#!/usr/bin/env bash
# Arbitration by consequence (chain vs memory, per question form) against the fixed abstention bar and no abstention:
# bAbI 20 tasks and the compute-depth diagnostic.
set -uo pipefail
cd "$(dirname "$0")/.."
B=./build-wsl/laminar_babi_ar
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o "$B" 2>&1 | grep -E "error" | head -5
run() { local name="$1" stream="$2"; shift 2
  "$B" file="data/${stream}_train.txt" test="data/${stream}_test.jsonl" units=50000 kpack=1 seq_topic_gain=8 show=0 questions=2000 seq_chain=2 seq_max_depth=8 "$@" > "logs/arb_$name.log" 2>&1; }
run babi_arb babi_all &
run diag_arb diag &
run diag_shift_arb diag_shift &
wait
echo "## bAbI 20 tasks: arbitration vs fixed bar (babi_d8) vs no abstention (babi_d8_noab)"
printf "%-6s %12s %12s %12s\n" task arbitrate fixed_bar none
for t in $(seq 1 20); do printf "%-6s %12s %12s %12s\n" "qa$t" "$(grep -a "^BABI qa$t:" logs/arb_babi_arb.log | grep -oE "[0-9]+/[0-9]+" | head -1)" "$(grep -a "^BABI qa$t:" logs/lever_babi_d8.log | grep -oE "[0-9]+/[0-9]+" | head -1)" "$(grep -a "^BABI qa$t:" logs/lever_babi_d8_noab.log | grep -oE "[0-9]+/[0-9]+" | head -1)"; done
tot() { grep -a "^BABI qa" "$1" | grep -oE "[0-9]+/[0-9]+" | awk -F/ '{c+=$1; n+=$2} END {printf "%d/%d", c, n}'; }
printf "%-6s %12s %12s %12s\n" total "$(tot logs/arb_babi_arb.log)" "$(tot logs/lever_babi_d8.log)" "$(tot logs/lever_babi_d8_noab.log)"
echo "## diagnostic (plain / unseen vocabulary)"
for n in diag_arb diag_shift_arb; do printf "%-15s |" "$n"; for k in 1 2 3 4 5 6 7 8; do printf " %2s" "$(grep -a "^BABI hop$k:" logs/arb_$n.log | grep -oE "[0-9]+/" | head -1 | tr -d /)"; done; printf " | yn %s cnt %s\n" "$(grep -a "^BABI qa6:" logs/arb_$n.log | grep -oE "[0-9]+/" | head -1 | tr -d /)" "$(grep -a "^BABI qa7:" logs/arb_$n.log | grep -oE "[0-9]+/" | head -1 | tr -d /)"; done
