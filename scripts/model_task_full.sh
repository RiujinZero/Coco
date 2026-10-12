#!/usr/bin/env bash
# Full-scale model task learner scenarios (200k units, 2,000 missions each, 46 frozen probes plus a body-output
# ablation initially and every 400 missions), run MAX_PARALLEL (default 3) at a time.
#
#   bash scripts/model_task_full.sh [ROOT]      (run from the repository root inside WSL)
#
# ROOT (default ~/model_task_full) holds scenarios.txt, one scenario per line:
#   NAME ARM [extra model_task_learner arguments...]       (blank lines and # comments ignored)
# The first call freezes the binary and records source hashes in ROOT/manifest.json; later calls verify them and
# resume every unfinished scenario from its checkpoint, so rerunning this script after any failure (including a
# WSL restart) continues where it stopped. A crashed process is resumed automatically up to 5 times; a declared
# configuration mismatch is never retried. Scenarios appended to scenarios.txt later are picked up on the next call.
#
# Concurrency: four concurrent 200k-unit models once filled the host drive with (then dense, 3 GB) checkpoints and
# crashed the WSL VM. Checkpoints are now sparse; each model still needs ~3.7 GB of RAM.
set -u
repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:-$HOME/model_task_full}"
mkdir -p "$root"
[ -f "$root/scenarios.txt" ] || { echo "missing $root/scenarios.txt"; exit 1; }
sources="cpp/apps/model_task_learner.cpp cpp/apps/subsystem_policy.h cpp/include/laminar/model_task.h cpp/include/laminar/state_io.h cpp/include/laminar/organism.h cpp/src/organism.cpp cpp/include/laminar/agency.h cpp/include/laminar/config.h cpp/include/laminar/sequence.h cpp/include/laminar/address.h cpp/include/laminar/goal_control.h cpp/include/laminar/relational_control.h"
if [ ! -f "$root/manifest.json" ]; then
  cp "$repo/build-task/model_task_learner" "$root/model_task_learner"
  {
    echo "{\"binary_sha256\":\"$(sha256sum "$root/model_task_learner" | cut -d' ' -f1)\",\"git_head\":\"$(cd "$repo" && git rev-parse HEAD)\",\"sources\":{"
    first=1; for f in $sources; do [ $first = 1 ] || echo ","; first=0; echo -n "\"$f\":\"$(sha256sum "$repo/$f" | cut -d' ' -f1)\""; done
    echo "}}"
  } > "$root/manifest.json"
fi
want=$(grep -o '"binary_sha256":"[0-9a-f]*"' "$root/manifest.json" | cut -d'"' -f4)
have=$(sha256sum "$root/model_task_learner" | cut -d' ' -f1)
[ "$want" = "$have" ] || { echo "frozen binary changed: refusing to continue"; exit 1; }

supervise() {
  local name="$1" arm="$2"; shift 2
  local tries=0
  cd "$root"
  while [ ! -f "$name/final.json" ]; do
    ./model_task_learner full "$arm" "$name" "$@" >> "$name.out" 2>&1
    code=$?
    [ -f "$name/final.json" ] && break
    echo "{\"scenario\":\"$name\",\"exit\":$code,\"at\":\"$(date -Is)\"}" >> "$root/restarts.jsonl"
    grep -q "declared configuration changed\|refusing to overwrite" "$name.out" && { echo "$name: configuration refusal, not retrying"; return 1; }
    tries=$((tries + 1)); [ $tries -ge 5 ] && { echo "$name: giving up after $tries restarts"; return 1; }
    sleep 10
  done
  echo "$name complete"
}

running=0
while read -r name arm rest; do
  case "$name" in ''|'#'*) continue ;; esac
  [ -f "$root/$name/final.json" ] && continue
  # shellcheck disable=SC2086
  supervise "$name" "$arm" $rest &
  running=$((running + 1))
  if [ $running -ge "${MAX_PARALLEL:-3}" ]; then wait -n; running=$((running - 1)); fi
done < "$root/scenarios.txt"
wait
echo "all scenarios finished"
