#!/usr/bin/env bash
# Canonical continual learner. Offline curriculum/environment adapter, WSL/Linux.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-wsl
binary=build-wsl/laminar_train
# Link to a new inode so an already-running validation process is not disturbed.
temporary="${binary}.new.$$"
trap 'rm -f -- "$temporary"' EXIT
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include \
  cpp/apps/laminar_train.cpp cpp/src/organism.cpp -o "$temporary"
mv -- "$temporary" "$binary"
if [[ "${1:-}" == --check ]]; then
  shift
  g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include \
    cpp/apps/relational_control_probe.cpp -o build-wsl/relational_control_probe
  python3 python/check_canonical_trainer.py --binary "$binary" "$@"
else
  exec "$binary" trainer=continual "$@"
fi
