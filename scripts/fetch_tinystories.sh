#!/usr/bin/env bash
# Fetch a ~100 KB slice of TinyStories for milestone D (design doc §15).
# Source: https://huggingface.co/datasets/roneneldan/TinyStories (TinyStories-valid.txt, ~22 MB).
#   scripts/fetch_tinystories.sh [bytes]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
bytes="${1:-102400}"
url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
curl -L --fail --range 0-$((bytes - 1)) "$url" -o data/tinystories_100k.txt
ls -l data/tinystories_100k.txt
