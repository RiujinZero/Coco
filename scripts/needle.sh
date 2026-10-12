#!/usr/bin/env bash
# Rolling-context acceptance test (needle-in-haystack recall), optionally over several settings.
#   scripts/needle.sh WINDOW UNITS NEEDLES [extra params...]
#   env: FILE=haystack (default data/tinystories_1m.txt), MARGINS="0.0 1.0", GAINS="1 2 4", BRIEF=1
set -uo pipefail
cd "$(dirname "$0")/.."
window="${1:-262144}"; units="${2:-100000}"; needles="${3:-64}"; shift 3 2>/dev/null || true
file="${FILE:-data/tinystories_1m.txt}"
filter='^needle'
if [ "${BRIEF:-0}" = "1" ]; then filter='^needle|^PLANT|^  birth|^RECALL|0.25 x|0.50 x|0.75 x|1.25 x|1.50 x'; fi
for m in ${MARGINS:-0.0}; do
  for g in ${GAINS:-1}; do
    echo "=== window=$window units=$units margin=$m exact_gain=$g $* ==="
    ./build-wsl/laminar_needle file="$file" window="$window" needles="$needles" units="$units" hist=7 episodic=1 episodic_margin="$m" exact_gain="$g" "$@" | grep -vE "$filter"
  done
done
