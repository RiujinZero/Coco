#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/relational_reactive build-wsl build-asan
g++ -std=c++20 -O3 -DNDEBUG -Icpp/include cpp/apps/relational_control_probe.cpp -o build-wsl/relational_control_probe
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_relational_control.cpp -o build-asan/test_relational_control
build-asan/test_relational_control
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_relational_session.cpp -o build-asan/test_relational_session
build-asan/test_relational_session
python3 python/relational_control_probe.py --reactive --out logs/relational_reactive
for seed in 0 1 2; do
  for mode in learned model model_neural model_frozen model_shuffled model_terminal model_single; do
    dir="logs/relational_reactive/$seed"
    build-wsl/relational_control_probe "$dir/input.txt" "$dir/environment.txt" "$dir/worlds.txt" "$seed" "$mode" "$dir/$mode" reactive > "$dir/$mode.txt"
  done
done
python3 python/check_relational_control.py --root logs/relational_reactive --modes learned model model_neural model_frozen model_shuffled model_terminal model_single
