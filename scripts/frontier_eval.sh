#!/bin/bash
# Evaluation of the merged-frontier decoder (docs/FRONTIER_DECODER.md): 20 BabyLM-style prompts (scripts/frontier_prompts.txt)
# x {greedy, sample T=0.8, frontier W=4 N=4 key=3} on gpu_motif epoch 9; frontier variants (no memory, key 6 / 2, T=1.0,
# copy penalty 0.5 / 1, single-utterance stopping); a second model (ss10_adr epoch 8). Read-only on the checkpoints; one
# model process at a time (each keeps <= 8 children); ~25 min on a shared machine. Results: logs/frontier/ (per-run
# .jsonl/.log/.err, table.md, outputs.txt).
#   bash scripts/frontier_eval.sh       (RUNS=/home/user/babylm/runs CORPUS=/home/user/babylm/BabyLM-2026-Strict-Small)
set -e
cd "$(dirname "$0")/.."
RUNS=${RUNS:-/home/user/babylm/runs}
CORPUS=${CORPUS:-/home/user/babylm/BabyLM-2026-Strict-Small}
B=build-task/model_frontier
OUT=logs/frontier
mkdir -p build-task "$OUT"
if [ ! -x "$B" ] || [ cpp/apps/model_frontier.cpp -nt "$B" ]; then
  (cd cpp && g++ -std=c++20 -O2 -march=native -DNDEBUG -Iinclude apps/model_frontier.cpp src/organism.cpp -o "../$B")
fi
VOCAB=$(ls "$CORPUS"/*.train.txt | paste -sd, -)
run() {
  tag=$1; run=$2; state=$3; shift 3
  echo "== $tag ($run/$state $*)"
  "$B" "$RUNS/$run" "$RUNS/$run/$state" @scripts/frontier_prompts.txt vocab="$VOCAB" out="$OUT/$tag.jsonl" "$@" > "$OUT/$tag.log" 2> "$OUT/$tag.err"
  tail -6 "$OUT/$tag.log"; grep -h WARNING "$OUT/$tag.err" || true
}
run motif_main   gpu_motif epoch_9.state mode=greedy,sample,frontier len=200 T=0.8 W=4 N=4 key=3
run motif_nomem  gpu_motif epoch_9.state mode=frontier len=200 T=0.8 W=4 N=4 key=3 mem=0
run motif_key6   gpu_motif epoch_9.state mode=frontier len=200 T=0.8 W=4 N=4 key=6
run motif_T1     gpu_motif epoch_9.state mode=sample,frontier len=200 T=1.0 W=4 N=4 key=3
run adr_main     ss10_adr  epoch_8.state mode=greedy,sample,frontier len=200 T=0.8 W=4 N=4 key=3
# judge with a copy penalty, a coarser merge key, single-utterance stopping
run motif_copy05 gpu_motif epoch_9.state mode=frontier len=200 T=0.8 W=4 N=4 key=3 copy=0.5
run motif_copy1  gpu_motif epoch_9.state mode=frontier len=200 T=0.8 W=4 N=4 key=3 copy=1
run motif_key2   gpu_motif epoch_9.state mode=frontier len=200 T=0.8 W=4 N=4 key=2
run motif_sent   gpu_motif epoch_9.state mode=greedy,sample,frontier len=200 T=0.8 W=4 N=4 key=3 stop=sent
run motif_sent4  gpu_motif epoch_9.state mode=greedy,sample,frontier len=200 T=0.8 W=4 N=4 key=3 stop=sent minwords=4
python3 scripts/frontier_report.py "$OUT"/motif_main.jsonl "$OUT"/motif_nomem.jsonl "$OUT"/motif_key6.jsonl "$OUT"/motif_key2.jsonl \
  "$OUT"/motif_T1.jsonl "$OUT"/motif_copy05.jsonl "$OUT"/motif_copy1.jsonl "$OUT"/adr_main.jsonl "$OUT"/motif_sent.jsonl "$OUT"/motif_sent4.jsonl \
  | tee "$OUT/table.md"
python3 scripts/frontier_report.py --dump "$OUT"/motif_main.jsonl "$OUT"/motif_copy1.jsonl "$OUT"/motif_copy05.jsonl "$OUT"/motif_T1.jsonl \
  "$OUT"/motif_nomem.jsonl "$OUT"/motif_key6.jsonl "$OUT"/motif_key2.jsonl "$OUT"/adr_main.jsonl "$OUT"/motif_sent.jsonl "$OUT"/motif_sent4.jsonl \
  > "$OUT/outputs.txt"   # the committed docs/FRONTIER_DECODER_OUTPUTS.txt
echo "done"
