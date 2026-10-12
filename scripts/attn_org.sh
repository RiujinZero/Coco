#!/usr/bin/env bash
# The organism with the learned answer attention on the bAbI 1-10 DESIGN set (shuffled bAbI stream).
#   scripts/attn_org.sh TAG [params...]
set -uo pipefail
cd "$(dirname "$0")/.."
tag="$1"; shift
[ -x build-wsl/laminar_babi_attn ] && [ build-wsl/laminar_babi_attn -nt cpp/include/laminar/attend.h ] && [ build-wsl/laminar_babi_attn -nt cpp/apps/laminar_babi.cpp ] || g++ -std=c++20 -O3 -march=native -DNDEBUG -Icpp/include cpp/apps/laminar_babi.cpp cpp/src/organism.cpp -o build-wsl/laminar_babi_attn 2>&1 | grep error | head -5
./build-wsl/laminar_babi_attn file=data/babi_all_train_shuf.txt test=data/babi_d10_test.jsonl units=50000 kpack=1 seq_topic_gain=8 seq_chain=2 seq_max_depth=8 hops=4 hops_free=2 \
   show=2 questions=100000 dump="logs/ao_${tag}_replies.jsonl" "$@" > "logs/ao_$tag.log" 2>&1
printf "%-8s %s  %s\n" "$tag" "$(grep -a '^BABI' logs/ao_$tag.log | awk '{split($3,a,"/"); c+=a[1]; t+=a[2]} END {printf "%d/%d", c, t}')" "$(grep -a '^BABI' logs/ao_$tag.log | awk '{split($3,a,"/"); printf "%s%s ", $2, a[1]}')"
