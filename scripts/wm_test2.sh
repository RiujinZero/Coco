#!/usr/bin/env bash
# How simple is the correction? byte embedding only, attending over 1 / 4 / 32 bytes.
set -uo pipefail
cd "$(dirname "$0")/.."
arm() { local name="$1"; shift; ./build-wsl/laminar_wm file=data/tinystories_13m.txt max_bytes=8388608 units=50000 kpack=1 hops=4 hops_free=2 "$@" > "logs/wm_$name.log" 2>&1; }
arm sym_p1  wm=sym wp=1 &
arm sym_p4  wm=sym wp=4 &
arm sym_d8  wm=sym wd=8 &
wait
for n in sym_p1 sym_p4 sym_d8; do printf "%-7s %s\n" "$n" "$(grep -a '^TOTAL' logs/wm_$n.log | cut -c8-)"; done
