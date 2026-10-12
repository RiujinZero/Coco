# Extended continuing bootstrap — 23 September 2026

**Launch status:** running locally at handoff (Windows launcher PID 27268,
Linux supervisor 69404, trainer 69410). The 48-training/16-evaluation smoke run
completed in 215.68 seconds and passed consistency checks. Its frozen score was
0/8 initially and 6/8 finally; both hard families remained unsuccessful. This
was an integration check, not a successful acquisition result. Full-run outcomes
are pending; use the live status file below rather than this launch snapshot.

Frozen manifest SHA-256:
`2bf7637943f07320df5e39649c6259e5d72d558755bc648381b03cf7c970d52f`.

The user authorized a fresh 2,000-episode run after the mixed pilot and targeted
isolation checks. The automatic goal remains paused; this is manual follow-up
without subagents. This experiment tests more training, heavier hard-task
sampling, and phase/action-balanced supervision together. It cannot attribute a
change to duration alone and is not evidence of generalization before results.

## Declared curriculum

One persistent controller, knowledge archive and reader starts fresh, with the
same seed 1, 1,316 hidden units and initial setup as the pilot. This is not a
resumed full checkpoint. Episodes 1–800 are fully guided, 801–1,600 taper linearly
to zero assistance, and 1,601–2,000 are autonomous with epsilon 0.1. The taper is
fixed in advance, not performance-gated.

The repeated 20-mission cycle allocates 600 episodes each to interrupted
six-edge and long tasks, 300 to stopping, and 100 each to reader, equal-count,
numeric, follow and union tasks. Each hard family therefore receives 240 fully
guided missions, compared with 13 in the pilot. Raw repair/union extension occurs
after episode 100, preserving policy and credit history.

Supervised updates every 20 warm episodes use
`ceil(10000 * guidance_probability)` steps. They balance public phase and
demonstrated action equally, then sample unique observed states within each
group. Ordinary source-tagged fitting remains available unchanged. Each training
mission also receives 512 terminal reward updates. The declared upper bound is
800,000 direct fits and 1,824,000 total updates, within the existing history cap.

The teacher is unchanged: known scheduling failures remain part of the record.
Budgets remain 256 decisions, 32 motors and zero prices. Finishing must still be
chosen by the policy. Search and representation limits are unchanged.

The same 46 frozen developmental probes run initially and after every 400
episodes: 276 evaluation missions, 2,276 missions overall. These repeated probes
are developmental evidence, not a fresh held-out generalization test. Full
decision/computation traces are retained; weights export every 20 episodes.

## Validation and durable execution

`GOAL_BALANCED_DEMONSTRATIONS_VALIDATION.json` records four passing checks:
optimized and ASAN/UBSAN demonstration tests, existing goal-control tests, and
mission-credit tests. They cover invalid-call rejection, unchanged semantics and
empirical values, and exact withdrawal/restoration replay. Validation v1 failed
on a stale test-only semantic-version expectation; v2 corrects it and preserves
both logs.

The frozen source, executable, configurations and hash manifest are in
`logs/goal_extended_bootstrap_20260923_v1`. The supervisor refuses accidental
restarts and verifies hashes before and after execution. A 48-training plus
16-evaluation mission smoke run must complete before the full run can launch.

Run the frozen supervisor from the repository root:

```text
wsl python3 logs/goal_extended_bootstrap_20260923_v1/source/python/run_goal_extended_bootstrap.py worker
```

The durable local process requires no model calls. Inspect `full_status.json`,
`full_run.log`, and `full_missions.jsonl` in that directory for actual progress.
On completion the supervisor checks mission counts, exact family allocation,
update accounting, frozen evaluation invariants, and artifact hashes, then
writes `full_completed.json` and `docs/GOAL_EXTENDED_BOOTSTRAP_RESULTS.json`.
On failure it records the error and does not automatically retry. These checks
are internal consistency checks, not an independent policy/physics audit.

Do not change the declared run while it is active. Read its completion record
before reporting outcomes, then compare per-family frozen performance against
the pilot's 34/46, especially hard-task activation, arrival switching and stopping.
No complete-learner claim or canonical trainer promotion is authorized by these
preliminary results. There is still no full resumable goal checkpoint.
