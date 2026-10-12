#!/bin/bash
# Prompt-dependence (PMI) judge for the merged-frontier decoder (docs/FRONTIER_DECODER.md, "Prompt-dependence judge"):
# 20 prompts (scripts/frontier_prompts.txt) x {greedy, sample T=0.8, frontier, frontier copy=1, frontier pmi=0.3 / 0.5 / 1,
# frontier pmi=0.5 copy=1} on gpu_motif epoch 9 and gpu_m20 epoch 1, len=200, seed 1; then greedy/sample/frontier again
# with profile=1 (the prompt's per-byte effect; the outputs must equal the main runs). Read-only on the checkpoints; one
# model process at a time with workers=4 (<= 4 computing children); ~65 min on a shared machine.
# Then: the pmi=0 runs on gpu_motif against the runs made before the judge existed (REF, if present), and the tables.
#   bash scripts/frontier_pmi_eval.sh   (RUNS=/home/user/babylm/runs CORPUS=... OUT=logs/frontier_pmi REF=<old logs/frontier>)
# A run whose .jsonl is complete is skipped (REDO=1: run it again).
set -e
cd "$(dirname "$0")/.."
RUNS=${RUNS:-/home/user/babylm/runs}
CORPUS=${CORPUS:-/home/user/babylm/BabyLM-2026-Strict-Small}
B=${B:-build-task/model_frontier}
OUT=${OUT:-logs/frontier_pmi}
REF=${REF:-}
WORKERS=${WORKERS:-4}
mkdir -p build-task "$OUT"
if [ ! -x "$B" ] || [ cpp/apps/model_frontier.cpp -nt "$B" ]; then
  (cd cpp && g++ -std=c++20 -O2 -march=native -DNDEBUG -Iinclude apps/model_frontier.cpp src/organism.cpp -o "../$B")
fi
VOCAB=$(ls "$CORPUS"/*.train.txt | paste -sd, -)
run() {
  tag=$1; run=$2; state=$3; shift 3
  if [ -s "$OUT/$tag.jsonl" ] && [ "$(wc -l < "$OUT/$tag.jsonl")" -ge 20 ] && [ -z "$REDO" ]; then echo "== $tag: done already"; return; fi
  echo "== $tag ($run/$state $*) $(date +%T), $(free -g | awk '/Mem:/ {print $7}') GB available"
  "$B" "$RUNS/$run" "$RUNS/$run/$state" @scripts/frontier_prompts.txt vocab="$VOCAB" out="$OUT/$tag.jsonl" workers=$WORKERS \
    len=200 T=0.8 W=4 N=4 key=3 "$@" > "$OUT/$tag.log" 2> "$OUT/$tag.err"
  tail -4 "$OUT/$tag.log" | head -3; grep -h WARNING "$OUT/$tag.err" || true
}
for M in "motif gpu_motif epoch_9.state" "m20 gpu_m20 epoch_1.state"; do
  set -- $M
  run $1_main   $2 $3 mode=greedy,sample,frontier
  run $1_copy1  $2 $3 mode=frontier copy=1
  run $1_pmi05  $2 $3 mode=frontier pmi=0.5
  run $1_pmi1   $2 $3 mode=frontier pmi=1
  run $1_pmi05c $2 $3 mode=frontier pmi=0.5 copy=1
  run $1_pmi03  $2 $3 mode=frontier pmi=0.3
done
for M in "motif gpu_motif epoch_9.state" "m20 gpu_m20 epoch_1.state"; do
  set -- $M
  run $1_profile $2 $3 mode=greedy,sample,frontier profile=1 ident=0
  python3 scripts/frontier_compare.py "$OUT/$1_main.jsonl" "$OUT/$1_profile.jsonl" | cut -c1-120
done
if [ -n "$REF" ]; then   # pmi=0 must reproduce the runs made before the judge existed
  python3 scripts/frontier_compare.py "$REF/motif_main.jsonl" "$OUT/motif_main.jsonl"
  python3 scripts/frontier_compare.py "$REF/motif_copy1.jsonl" "$OUT/motif_copy1.jsonl"
fi
T=""; for m in motif m20; do for t in main copy1 pmi03 pmi05 pmi1 pmi05c; do T="$T $OUT/${m}_$t.jsonl"; done; done
python3 scripts/frontier_report.py --topic $T | tee "$OUT/table.md"
python3 scripts/frontier_report.py --profile "$OUT/motif_profile.jsonl" "$OUT/m20_profile.jsonl" | tee "$OUT/profile.md"
python3 scripts/frontier_report.py --samples "Once upon a time there was|It was a cold morning and" "$OUT"/motif_{main,copy1,pmi05,pmi1,pmi05c}.jsonl \
  "$OUT"/m20_{main,copy1,pmi05,pmi1,pmi05c}.jsonl > "$OUT/samples.md"
python3 scripts/frontier_report.py --dump $T > "$OUT/outputs.txt"   # the committed docs/FRONTIER_PMI_OUTPUTS.txt
echo "done $(date +%T)"
