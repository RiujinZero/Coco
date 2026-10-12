#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/grounded_session build-wsl build-asan
python3 python/build_grounded_bundle.py
python3 python/grounded_session_probe.py
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include -c cpp/src/organism.cpp -o build-wsl/grounded_session_core.o
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/grounded_session_probe.cpp build-wsl/grounded_session_core.o -o build-wsl/grounded_session_probe
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/laminar_train.cpp build-wsl/grounded_session_core.o -o build-wsl/laminar_train_grounded
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_grounded_session.cpp -o build-asan/test_grounded_session
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include -c cpp/src/organism.cpp -o build-asan/grounded_session_core.o
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_packed_allocation.cpp build-asan/grounded_session_core.o -o build-asan/test_packed_allocation
build-asan/test_grounded_session
build-asan/test_packed_allocation
for seed in 0 1 2; do build-wsl/grounded_session_probe "$seed"; done
common=(file=logs/grounded_session/trainer.txt units=128 seed=0 block=31 seq_window=65536 seq_bits=10 seq_consolidate=0 grounded_state=1)
build-wsl/laminar_train_grounded "${common[@]}" grounded_bootstrap=logs/grounded_session/bootstrap.paths probe_every=97 qa=logs/grounded_session/probe.jsonl per_task=1 > logs/grounded_session/trainer_probe.log
build-wsl/laminar_train_grounded "${common[@]}" grounded_bootstrap=logs/grounded_session/bootstrap.paths probe_every=1048576 > logs/grounded_session/trainer_plain.log
build-wsl/laminar_train_grounded "${common[@]}" probe_every=1048576 > logs/grounded_session/trainer_disabled.log
python3 python/check_grounded_session.py
