#!/usr/bin/env bash
# Round 2: learned abstention (the chain answers only forms it is credited on) and levers 1-2 where the bodies answer.
#   scripts/levers2.sh [units]
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-50000}"; B=./build-wsl/laminar_babi_lv
g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o "$B" 2>&1 | grep -E "error" | head -5
run() { local name="$1" stream="$2"; shift 2
  "$B" file="data/${stream}_train.txt" test="data/${stream}_test.jsonl" units="$units" kpack=1 seq_topic_gain=8 show=3 questions=1000 "$@" > "logs/lever_$name.log" 2>&1; }
run abstain       diag seq_chain=2 seq_max_depth=8 &
run abstain_r2    diag seq_chain=2 seq_max_depth=8 recirc_iters=2 &
run abstain_h3o   diag seq_chain=2 seq_max_depth=8 hops=3 route_gate=0 &
run abstain_h4    diag seq_chain=2 seq_max_depth=8 hops=4 &
run abstain_wK    diag seq_chain=2 seq_max_depth=8 kscreen=0 kset=16,32,64 &
run abstain_rat   diag_rat seq_chain=2 seq_max_depth=8 gen=200 &
wait
printf "%-12s %6s %7s %6s |" config KB/s MACs/B BPB; for k in 1 2 3 4 5 6 7 8; do printf " h%d" $k; done; printf " | yn  cnt\n"
for n in abstain abstain_r2 abstain_h3o abstain_h4 abstain_wK abstain_rat; do
  f="logs/lever_$n.log"; t=$(grep -a "^trained" "$f")
  kb=$(echo "$t" | grep -oE "[0-9]+ KB/s" | grep -oE "[0-9]+"); mb=$(echo "$t" | grep -oE "MACs/byte [0-9]+" | grep -oE "[0-9]+$"); bpb=$(echo "$t" | grep -oE "BPB [0-9.]+" | grep -oE "[0-9.]+$")
  printf "%-12s %6s %7s %6s |" "$n" "$kb" "$mb" "$bpb"
  for k in 1 2 3 4 5 6 7 8; do printf " %2s" "$(grep -a "^BABI hop$k:" "$f" | grep -oE "[0-9]+/" | head -1 | tr -d /)"; done
  printf " | %2s  %2s\n" "$(grep -a "^BABI qa6:" "$f" | grep -oE "[0-9]+/" | head -1 | tr -d /)" "$(grep -a "^BABI qa7:" "$f" | grep -oE "[0-9]+/" | head -1 | tr -d /)"
done
grep -aoE "at END [0-9]+, abstained [0-9]+" logs/lever_abstain.log | tail -1
echo "(each cell: correct of 50)"
