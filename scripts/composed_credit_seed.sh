#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
seed="${1:-0}"
root="${2:-logs/composed_credit}"
budget="${3:-100000}"
for profile in boolean numeric; do
  mkdir -p "$root/$profile"
  build-wsl/composed_credit_full_probe "$seed" "$root/$profile" "$budget" "$profile" > "$root/$profile/$seed.txt"
done
