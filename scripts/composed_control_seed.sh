#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
seed="${1:?seed}"; root="${2:-logs/composed_control}"
python3 python/composed_control_probe.py --out "$root" --seeds "$seed" --episodes 400 --budget 100000
dir="$root/$seed"
build-wsl/composed_control_raw_probe "$dir/input.txt" "$dir/environment.txt" "$dir/worlds.txt" "$seed" learned_progress "$dir/learned_progress" composed 100000 > "$dir/learned_progress.txt"
build-wsl/primitive_checkpoint_probe "$seed" "$dir/learned_progress.lifecycle.final.weights" "$dir/checkpoint"
python3 python/check_composed_control.py --root "$root" --seeds "$seed"
