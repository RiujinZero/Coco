#!/usr/bin/env bash
# Relations as things (seq_rel_gain): build the harnesses and run regression (bAbI), dialogue (SmolTalk) and prose
# (TinyStories BPB / loops) with relations off and on.
#   scripts/rel_test.sh [rel_gain]
set -uo pipefail
cd "$(dirname "$0")/.."
rg="${1:-0.25}"
for a in babi chat generate; do
  g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -Icpp/include "cpp/apps/laminar_$a.cpp" cpp/src/organism.cpp -o "build-wsl/laminar_${a}_rel" 2>&1 | grep -E "error" | head -5
done
echo "##### bAbI 5 tasks, learned reading + relations $rg (regression)"
BIN=./build-wsl/laminar_babi_rel GAINS="8" ./scripts/babi.sh data/babi_t1-3-15-16 50000 seq_chain=2 seq_rel_gain="$rg" show=0
echo "##### SmolTalk stay-on-task, learned reading, relations off"
BIN=./build-wsl/laminar_chat_rel GAINS="8" ./scripts/ontask.sh data/smoltalk.txt 50000 40 seq_chain=2
echo "##### SmolTalk stay-on-task, learned reading, relations $rg"
BIN=./build-wsl/laminar_chat_rel GAINS="8" ./scripts/ontask.sh data/smoltalk.txt 50000 40 seq_chain=2 seq_rel_gain="$rg"
for r in 0 "$rg"; do
  echo "##### TinyStories 32 MB prose, learned reading, relations $r"
  ./build-wsl/laminar_generate_rel file=data/tinystories_13m.txt max_bytes=33554432 units=100000 gen=2048 prompts=8 kpack=0 temp=0.7 seq_chain=2 seq_topic_gain=8 seq_rel_gain="$r" 2>&1 \
    | grep -aE "^trained|^seqmem|^FREE|^PLANTED" | cut -c1-330
done
