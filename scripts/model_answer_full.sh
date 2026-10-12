#!/usr/bin/env bash
# Full-scale answer-channel scenarios (model_answer_learner), MAX_PARALLEL (default 2) at a time.
#   bash scripts/model_answer_full.sh ROOT      (run from the repository root inside WSL)
# ROOT holds scenarios.txt: NAME [model_answer_learner options...]. The binary is frozen into ROOT on the first call;
# every unfinished scenario resumes from its checkpoint; crashed processes are resumed up to 5 times.
set -u
repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:?root}"
[ -f "$root/scenarios.txt" ] || { echo "missing $root/scenarios.txt"; exit 1; }
[ -f "$root/model_answer_learner" ] || cp "$repo/build-task/model_answer_learner" "$root/model_answer_learner"
supervise() {
  local name="$1"; shift; local tries=0
  cd "$root"
  while [ ! -f "$name/final.json" ]; do
    ./model_answer_learner full "$name" "$@" >> "$name.out" 2>&1; code=$?
    [ -f "$name/final.json" ] && break
    echo "{\"scenario\":\"$name\",\"exit\":$code,\"at\":\"$(date -Is)\"}" >> "$root/restarts.jsonl"
    grep -q "configuration changed\|refusing to overwrite" "$name.out" && return 1
    tries=$((tries + 1)); [ $tries -ge 5 ] && return 1
    sleep 10
  done
}
running=0
while read -r name rest; do
  case "$name" in ''|'#'*) continue ;; esac
  [ -f "$root/$name/final.json" ] && continue
  # shellcheck disable=SC2086
  supervise "$name" $rest &
  running=$((running + 1))
  if [ $running -ge "${MAX_PARALLEL:-2}" ]; then wait -n; running=$((running - 1)); fi
done < "$root/scenarios.txt"
wait
