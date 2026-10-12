#!/usr/bin/env bash
# The four compute levers on the compute-depth diagnostic (python/diag_stream.py): accuracy per hop count k and on
# yes/no (qa6) and counting (qa7), with training throughput, MACs/byte and BPB. Configs run in parallel, one core each.
#   scripts/levers.sh [units]        -> logs/lever_<name>.log, summary on stdout
set -uo pipefail
cd "$(dirname "$0")/.."
units="${1:-50000}"; B=./build-wsl/laminar_babi_lv
mkdir -p logs
run() {   # name stream extra...
  local name="$1" stream="$2"; shift 2
  "$B" file="data/${stream}_train.txt" test="data/${stream}_test.jsonl" units="$units" kpack=1 topic_gain=8 seq_topic_gain=8 show=3 questions=1000 "$@" > "logs/lever_$name.log" 2>&1
}
# batch 1: baseline, depth (recirculation), steps (chain depth)
run base        diag seq_chain=2 &
run recirc2     diag seq_chain=2 recirc_iters=2 &
run recirc4     diag seq_chain=2 recirc_iters=4 &
run depth8      diag seq_chain=2 seq_max_depth=8 &
run hops3       diag seq_chain=2 hops=3 &
run hops4       diag seq_chain=2 hops=4 &
wait
# batch 2: motif breadth variants, thinking out loud
run hops3_open  diag seq_chain=2 hops=3 route_gate=0 &
run wideK       diag seq_chain=2 kscreen=0 kset=16,32,64 &
run bodies      diag seq_chain=0 &
run bodies_rat  diag_rat seq_chain=0 gen=200 &
run chain_rat   diag_rat seq_chain=2 seq_max_depth=8 gen=200 &
wait
printf "%-12s %6s %7s %6s |" config KB/s MACs/B BPB; for k in 1 2 3 4 5 6 7 8; do printf " h%d" $k; done; printf " | yn  cnt\n"
for n in base recirc2 recirc4 depth8 hops3 hops4 hops3_open wideK bodies bodies_rat chain_rat; do
  f="logs/lever_$n.log"; t=$(grep -a "^trained" "$f")
  kb=$(echo "$t" | grep -oE "[0-9]+ KB/s" | grep -oE "[0-9]+"); mb=$(echo "$t" | grep -oE "MACs/byte [0-9]+" | grep -oE "[0-9]+$"); bpb=$(echo "$t" | grep -oE "BPB [0-9.]+" | grep -oE "[0-9.]+$")
  printf "%-12s %6s %7s %6s |" "$n" "$kb" "$mb" "$bpb"
  for k in 1 2 3 4 5 6 7 8; do printf " %2s" "$(grep -a "^BABI hop$k:" "$f" | grep -oE "[0-9]+/" | head -1 | tr -d /)"; done
  printf " | %2s  %2s\n" "$(grep -a "^BABI qa6:" "$f" | grep -oE "[0-9]+/" | head -1 | tr -d /)" "$(grep -a "^BABI qa7:" "$f" | grep -oE "[0-9]+/" | head -1 | tr -d /)"
done
echo "(each cell: correct of 50)"
