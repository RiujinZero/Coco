#!/usr/bin/env bash
# Bounded memory: a short instance window plus a consolidated class-count tier, against the full-window version.
#   scripts/bounded_gen.sh [max_bytes] [units]        ROWS="ref 1m 1m+c 256k+c" selects rows (default all)
set -uo pipefail
cd "$(dirname "$0")/.."
mb="${1:-33554432}"; units="${2:-100000}"; shift 2 2>/dev/null || true; extra=("$@")
run() {
  local tag="$1"; shift
  echo "=== $tag ${extra[*]:-} ==="
  ./build-wsl/laminar_generate_v3 file=data/tinystories_13m.txt max_bytes="$mb" units="$units" gen=2048 prompts=8 kpack=0 seqmem=1 temp=0.7 "$@" "${extra[@]}" 2>&1 \
    | grep -E '^trained|^seqmem|^FREE|^PLANTED|^\[prompt 0\]' | cut -c1-330
}
want() { case " ${ROWS:-ref 1m 1m+c 256k+c} " in *" $1 "*) return 0;; *) return 1;; esac; }
want ref    && run "full 32 MB instance window (reference)" seq_window=33554432 seq_bits=21 &
want 1m     && run "1 MB instance window, no consolidation" seq_window=1048576 seq_bits=18 &
wait
want 1m+c   && run "1 MB instance window + consolidated counts (2^20 x 8 lags, tagged)" seq_window=1048576 seq_bits=18 seq_consolidate=1 seq_cbits=20 &
want 256k+c && run "256 KB instance window + consolidated counts" seq_window=262144 seq_bits=16 seq_consolidate=1 seq_cbits=20 &
wait
