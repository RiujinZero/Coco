#!/usr/bin/env bash
# Full TinyStories pass (the 13.6 MB train+valid slice on this machine): baseline, episodic memory, agency.
#   scripts/full_tinystories.sh [epochs]
set -uo pipefail
cd "$(dirname "$0")/.."
epochs="${1:-1}"
F=data/tinystories_13m.txt
P=$((2 * 1048576))
echo "=== A. baseline: adaptive locks + learning, 200k units ==="
./build-wsl/laminar_bytes file=$F units=200000 epochs=$epochs quiet=1 progress=$P log=build-wsl/full_A.jsonl | grep -E '^ |^EPOCH|^TOTAL'
echo "=== B. + episodic memory (sum mode), 1M units ==="
./build-wsl/laminar_bytes file=$F units=1000000 epochs=$epochs quiet=1 progress=$P episodic=1 hist=7 readout_freeze=2097152 log=build-wsl/full_B.jsonl | grep -E '^ |^EPOCH|^TOTAL'
echo "=== C. + episodic memory + agency kernel (attend set), 1M units ==="
./build-wsl/laminar_bytes file=$F units=1000000 epochs=$epochs quiet=1 progress=$P episodic=1 hist=7 readout_freeze=2097152 agency=1 agency_ops=hold,attend_episodic,attend_generic,to_memory log=build-wsl/full_C.jsonl | grep -E '^ |^EPOCH|^TOTAL'
