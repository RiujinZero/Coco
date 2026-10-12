#!/usr/bin/env bash
# Can Coco rewire toward the goal? Byte-loss plasticity vs goal-credited plasticity (answer bytes scale every signal;
# question-time bodies credited by the answer's outcome). Chain OFF: answers come from bodies + memory alone.
# Held-out bAbI 1-10 at 1/4, 1 and 3 passes; at 3 passes, ablate the 8 most question-specialised motifs vs 8 random.
set -uo pipefail
cd "$(dirname "$0")/.."
B=./build-wsl/laminar_babi_goal
g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o "$B" 2>&1 | grep -E "error" | head -5
Q=$(stat -c %s data/babi_all_train_shuf.txt)
run() { local name="$1" file="$2" mb="$3"; shift 3
  "$B" file="$file" max_bytes="$mb" test=data/babi_d10_test.jsonl units=50000 kpack=1 seq_topic_gain=8 seq_chain=0 hops=4 hops_free=2 questions=1000 show=0 "$@" > "logs/gt_$name.log" 2>&1; }
CTL="goal_gain=1 goal_elig=0"; GOAL="goal_gain=8 goal_elig=1"
run ctl_q  data/babi_all_train_shuf.txt $((Q / 4)) $CTL &
run ctl_1  data/babi_all_train_shuf.txt 0 $CTL &
run ctl_3  data/babi_shuf_x3.txt 0 $CTL &
run goal_q data/babi_all_train_shuf.txt $((Q / 4)) $GOAL &
run goal_1 data/babi_all_train_shuf.txt 0 $GOAL &
run goal_3 data/babi_shuf_x3.txt 0 $GOAL &
run ctl_3_top  data/babi_shuf_x3.txt 0 $CTL ablate=top:8 &
run ctl_3_rand data/babi_shuf_x3.txt 0 $CTL ablate=rand:8 &
run goal_3_top  data/babi_shuf_x3.txt 0 $GOAL ablate=top:8 &
run goal_3_rand data/babi_shuf_x3.txt 0 $GOAL ablate=rand:8 &
wait
for n in ctl_q ctl_1 ctl_3 goal_q goal_1 goal_3 ctl_3_top ctl_3_rand goal_3_top goal_3_rand; do
  printf "%-12s %-9s %s\n" "$n" "$(grep -a '^BABI' logs/gt_$n.log | awk '{split($3,a,"/"); c+=a[1]; t+=a[2]} END {printf "%d/%d", c, t}')" "$(grep -a '^BABI' logs/gt_$n.log | awk '{split($3,a,"/"); printf "%s%s ", $2, a[1]}')"
done
for n in ctl_3 goal_3; do echo "$n: $(grep -a '^trained' logs/gt_$n.log | cut -c10-60) | $(grep -a '^goal' logs/gt_$n.log | grep -oE 'motifs >3x.*|mean R [0-9.-]+')"; done
