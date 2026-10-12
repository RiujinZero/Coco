#!/usr/bin/env bash
# Verified resume test for the model task learner (resume profile: 12 missions, checkpoints every 3,
# frozen probes at 6 and 12). Three runs must end in the identical model/task-model state and identical
# deterministic logs: (A) uninterrupted; (B) stopped cleanly after the mission-6 checkpoint, then resumed;
# (C) killed with SIGKILL mid-run at a random point, then resumed from whatever checkpoint survived;
# (D) died inside the mission-6 evaluation, then resumed (the evaluation must be redone, not skipped).
set -u
bin="${1:?binary}"; work="${2:?work dir}"; shift 2
rm -rf "$work"; mkdir -p "$work"; cd "$work"
fp() { grep -o '"model_fingerprint":"[0-9a-f]*"' "$1/final.json"; }
t0=$(date +%s.%N)
"$bin" resume learned A "$@" > A.out 2>&1 || { echo "A failed"; tail -5 A.out; exit 1; }
t1=$(date +%s.%N)
"$bin" resume learned B --stop-after 4 "$@" > B1.out 2>&1 || { echo "B1 failed"; exit 1; }
grep -q "stopping after checkpoint at mission 6" B1.out || { echo "B did not stop at 6"; tail -3 B1.out; exit 1; }
"$bin" resume learned B "$@" > B2.out 2>&1 || { echo "B2 failed"; tail -5 B2.out; exit 1; }
# D: the process dies inside the mission-6 evaluation (after one probe); resume must redo that evaluation
"$bin" resume learned D --abort-during-eval 6 "$@" > D1.out 2>&1; [ $? = 3 ] || { echo "D did not abort in evaluation"; tail -3 D1.out; exit 1; }
"$bin" resume learned D "$@" > D2.out 2>&1 || { echo "D2 failed"; tail -5 D2.out; exit 1; }
# C: hard kill at ~55% of the uninterrupted wall time, then resume
dur=$(echo "$t1 - $t0" | bc); kill_at=$(echo "$dur * 0.55" | bc)
"$bin" resume learned C "$@" > C1.out 2>&1 & pid=$!
sleep "$kill_at"; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
killed_after=$(grep -c "^guided\|^taper\|^autonomous" C1.out)
t2=$(date +%s.%N)
"$bin" resume learned C "$@" > C2.out 2>&1 || { echo "C2 failed"; tail -5 C2.out; exit 1; }
t3=$(date +%s.%N)
ok=1
for r in B C D; do
  [ "$(fp A)" = "$(fp $r)" ] || { echo "fingerprint differs: A $(fp A) $r $(fp $r)"; ok=0; }
  for f in missions.jsonl decisions.jsonl evaluations.jsonl final.json; do cmp -s "A/$f" "$r/$f" || { echo "$r/$f differs from A"; ok=0; }; done
done
echo "{\"uninterrupted_seconds\":$dur,\"killed_after_missions\":$killed_after,\"resume_seconds\":$(echo "$t3 - $t2" | bc),\"fingerprint\":\"$(fp A | cut -d'"' -f4)\",\"identical\":$([ $ok = 1 ] && echo true || echo false)}" | tee result.json
[ $ok = 1 ]
