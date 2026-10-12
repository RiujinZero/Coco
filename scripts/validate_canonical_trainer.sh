#!/usr/bin/env bash
# Full expensive parity check against an existing, independently audited seed.
set -euo pipefail
cd "$(dirname "$0")/.."
reference="${1:-logs/composed_control_raw/1}"
root="${2:-logs/canonical_validation}"
seed=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$reference/oracle.json")
directory="$root/$seed"
mkdir -p -- "$root"
mkdir -- "$directory" # Require a fresh destination; retain historical evidence.
for name in input.txt environment.txt worlds.txt oracle.json; do
  cp -- "$reference/$name" "$directory/$name"
done
bash scripts/train_coco.sh "file=$directory/input.txt" \
  "environment=$directory/environment.txt" "worlds=$directory/worlds.txt" \
  "seed=$seed" "output=$directory/learned_progress" \
  "checkpoint=$directory/state.replay" lifecycle=1 > "$directory/learned_progress.txt"
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include \
  cpp/apps/primitive_checkpoint_probe.cpp -o build-wsl/primitive_checkpoint_probe
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include \
  cpp/apps/verify_local_choices.cpp -o build-wsl/verify_local_choices
build-wsl/primitive_checkpoint_probe "$seed" "$directory/learned_progress.lifecycle.final.weights" "$directory/checkpoint"
python3 python/check_canonical_parity.py "$reference" "$directory"
