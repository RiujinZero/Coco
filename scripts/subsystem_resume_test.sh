#!/usr/bin/env bash
# Resume test for the checkpointed subsystem run (smoke profile, 48 missions, checkpoints every 20):
# (A) uninterrupted; (B) stopped after the mission-20 checkpoint and resumed; (C) SIGKILLed mid-run and resumed.
# All deterministic logs and the final subsystem state fingerprint must be identical.
set -u
bin="${1:?binary}"; work="${2:?work dir}"
rm -rf "$work"; mkdir -p "$work"; cd "$work"
t0=$(date +%s.%N); "$bin" smoke A > A.out 2>&1 || { echo "A failed"; tail -3 A.out; exit 1; }; t1=$(date +%s.%N)
"$bin" smoke B --stop-after 20 > B1.out 2>&1 || { echo "B1 failed"; tail -3 B1.out; exit 1; }
grep -q "stopping after checkpoint at mission 20" B1.out || { echo "B did not stop at 20"; exit 1; }
"$bin" smoke B > B2.out 2>&1 || { echo "B2 failed"; tail -3 B2.out; exit 1; }
dur=$(echo "$t1 - $t0" | bc); kill_at=$(echo "$dur * 0.6" | bc)
"$bin" smoke C > C1.out 2>&1 & pid=$!; sleep "$kill_at"; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
"$bin" smoke C > C2.out 2>&1 || { echo "C2 failed"; tail -3 C2.out; exit 1; }
ok=1
fp() { grep -o '"state_fingerprint":"[0-9a-f]*"' "$1_summary.json"; }
for r in B C; do
  [ "$(fp A)" = "$(fp $r)" ] || { echo "$r fingerprint differs"; ok=0; }
  for s in _trace.jsonl _missions.jsonl _weights.jsonl _evaluations.jsonl _bootstrap.jsonl _computations.jsonl; do cmp -s "A$s" "$r$s" || { echo "$r$s differs"; ok=0; }; done
done
echo "{\"uninterrupted_seconds\":$dur,\"fingerprint\":\"$(fp A | cut -d'"' -f4)\",\"evaluations\":$(cat A_evaluations.jsonl | wc -l),\"identical\":$([ $ok = 1 ] && echo true || echo false)}" | tee result.json
cat A_evaluations.jsonl | cut -c1-120
rm -f *_checkpoint.state* *_final.state
[ $ok = 1 ]
