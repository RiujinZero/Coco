#!/usr/bin/env bash
# The bar: free generation of >=250 words with no loops (span >=64 B recurring, or >30% repeated 8-grams), and
# how much of it is verbatim copying of the window.
#   TEMPS="0.7 1.0" scripts/strict_gen.sh [max_bytes] [units] [extra params...]
#   VARIANTS="seq_min_support=2|seq_copy_max=24" ... runs one line per variant (params separated by |)
set -uo pipefail
cd "$(dirname "$0")/.."
mb="${1:-33554432}"; units="${2:-100000}"; shift 2 2>/dev/null || true
IFS='|' read -r -a variants <<< "${VARIANTS:-}"
[ ${#variants[@]} -eq 0 ] && variants=("")
for T in ${TEMPS:-0.7 1.0}; do
  for var in "${variants[@]}"; do
    echo "=== strict 250-word test, $((mb / 1048576)) MB, temp $T $var $* ==="
    ./build-wsl/laminar_generate_v3 file=data/tinystories_13m.txt max_bytes="$mb" units="$units" gen=2048 prompts=8 kpack=0 seqmem=1 temp="$T" $var "$@" 2>&1 \
      | grep -E '^FREE|^PLANTED|^seqmem|^\[prompt [01]\]' | cut -c1-330
  done
done
