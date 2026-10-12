# Continuing learner handoff — 23 September 2026

Read this first, then [remaining gates](LEARNER_REMAINING_GATES.md) and the
[mixed experiment record](GOAL_CONTINUING_BOOTSTRAP.md). The original
[complete-learner handoff](COMPLETE_LEARNER_HANDOFF.md) preserves the earlier
acceptance contract and history. This document supersedes its experiment status,
not its unfinished capability requirements.

**Manual targeted follow-up:** [six-edge / long-task isolation](GOAL_TARGETED_ISOLATION.md)
passes 4/4 reference cases for each family in both route-only and combined raw
libraries. Saved warm/final neural policies remain 0/4 in all conditions. The
actual pilot loops in agenda reasoning instead of activating usable subgoals;
activation agreement falls 136/382 to 113/382, and arrival-switch agreement is
0/5. The individual mechanisms remain viable, but neural goal management was
not mastered. The user subsequently authorized the
[2,000-episode extended bootstrap](GOAL_EXTENDED_BOOTSTRAP.md), with 800 fully
guided episodes, 800 tapering episodes, 400 autonomous episodes, heavier hard-task
sampling and phase/action-balanced fitting. Its taper is fixed, not
performance-gated. Consult the durable run status and completion record before
claiming results. The automatic goal remains paused.

## User direction and stopping point

The user prioritized acquisition over speed: permit 256 decisions, 32 motors,
and zero operation/computation/management prices while recording actual work.
Finishing remains a learned action; physical success never forces a submission.
The user then requested one continuing learner, warm-starting goal selection,
subgoals, stopping, language, actions and skills, with backpropagation where useful.
Do not require isolated perfection before trying the joint learner.

The final instruction is to finish the currently declared longer run, have all
subagents wrap up, commit the work, and write this handoff because usage is low.
Do not automatically start another experiment or expand the scope after that run.
The user explicitly requested pausing the active goal after the commit/handoff,
then continuing manually with targeted runs and no subagents when they choose.

## Implemented since the original handoff

- Greedy-trace delayed credit with matched retention checks; reusable disjunctive
  concepts; a bank retaining successive computation repairs and their revisions.
- An experimental seven-operation goal controller sharing the existing policy
  and knowledge substrate. It owns public goal arrival, inspection, activation,
  switching/resumption, derived proposals, paid previews, mission budgets,
  chronological terminal credit, correction and exact replay.
- Conservative actual-reader projections with alternative-world and revision
  identity preservation. Partial evidence remains partial.
- Explicit, source-tagged demonstration backpropagation. Hints previously only
  chose/logged actions; direct fitting now trains those labels into the same
  neural policy. Fits balance distinct observed states, leave empirical values
  and semantics unchanged, and participate in withdrawal/restoration replay.
- One mixed runner with persistent controller, knowledge, credits and archived
  reader, raw semantic warm starts, later in-place skill additions, tapered
  demonstrations, autonomous continuation and frozen retention probes.

The normal trainer binary/profile has not been promoted. The five-operation
controller remains compatible; exact 27-episode trace/weight/value/consolidation/
correction parity passes after the new demonstration path. Full goal-controller
checkpoint persistence and a canonical goal training profile are still absent.

## Evidence to retain

| Experiment | Result and interpretation |
|---|---|
| Original goal pilot | 760 missions; warm 93/100, all five greedy checkpoints 0/12. Adaptive 9/200, frozen-policy 5/200, random-credit 2/200 sampled successes. No acquisition pass. |
| Value/readout diagnostic | Exact warm reconstruction; all three complete-mission probes 0/12. Empirical readout switches on time but loops after physical success. 28,928 consolidation fits do not repair neural acquisition. |
| 512-update continuation | 560 missions; adaptive 5/200, random-credit 8/200. Greedy 0/12 before and after consolidation. Increased credit coverage alone does not solve acquisition. |
| Continuing long reference | Existing six-edge knowledge retained; raw nine-edge extension and optional actual exhausted-search feature permit 8/8 supplied references. Feasibility, not learned choices. |
| Shared warm teacher | 14/14 training fixtures, 14/14 held-out fixtures, 2/2 long examples. One public observation/menu function; no family ID. Supplied examples, not acquisition. |
| Mixed smoke | 36 total missions; guided 19/20, unassisted 0/8→6/8. One physical stopping case, reader and four skill queries pass; interrupted six-edge and long tasks fail. 3,840 direct fits and 10,240 reward updates. |

Frozen evidence directories are named in the linked experiment documents. The
`logs/` and `build-*` trees are intentionally Git-ignored: full JSONL traces,
frozen source snapshots and binaries remain local. Versioned source, analysis
scripts, documentation and compact validation records carry the reproducible
configuration and hashes. Do not delete or overwrite failed trials.

The smoke reader teaching export mistakenly labels authored structured lessons
`verified_environment`; actual receipts use `verified_teacher`. The frozen smoke
is retained with that erratum. The larger pilot exports the actual source.

## Larger run

The frozen declaration is in
`logs/goal_continuing_bootstrap_pilot_20260923_v1/pilot_config.json`.
It uses seed 1, hidden width 1,316 (52,647 weights), 400 warm missions, 200
autonomous missions and three sets of 46 frozen probes: 738 missions total.
Terminal updates are 512 per training mission; explicit demonstration fits are
capped at 200,000. Guidance and direct fits taper to zero before autonomous
continuation (epsilon 0.1). Frozen probes have no hints, fits or learning.

Repair and union examples arrive after the first 100 credited missions without
replacing policy or history. The initial long-route/procedure and reader warm
starts are declared assistance. Reader atomic question programs are supplied
substrate; the entire reader is not learned from scratch. Novel probes vary
bindings, counts, chain lengths, ordering and arrival times within supplied task
families. Long-route topology is unchanged. Repeated probes are developmental,
not untouched confirmation or arbitrary-task generalization.

The run completed all **738 missions in 1,078.95 seconds**. Warm training succeeds
351/400 times (guidance is tapered, so this is not a fully guided score).
Autonomous training succeeds 148/200 times. Frozen unassisted probes improve
**0/46 → 34/46 after warm start → 34/46 after autonomous continuation**.

| Family | Warm checkpoint | Autonomous checkpoint |
|---|---:|---:|
| Stopping variants | 14/18 | 14/18 |
| Interrupted six-edge routes | 0/4 | 0/4 |
| Long routes / derived goals | 0/4 | 0/4 |
| Reader | 4/4 | 4/4 |
| Equal counts | 4/4 | 4/4 |
| Numeric count | 4/4 | 4/4 |
| Chain endpoints | 4/4 | 4/4 |
| Rule union | 4/4 | 4/4 |

The same continuing learner retains 600 credits, with 307,200 reward updates
and 198,358 direct demonstration fits (505,558 total). Its last nonzero direct
fit is after warm mission 250: the remaining cap cannot fund another complete
state-balanced round. Later warm guidance still tapers according to declaration;
autonomous continuation has neither guidance nor direct fits. Skill extension
at mission 100 preserves all 100 credits and the exact existing policy.

[GOAL_CONTINUING_BOOTSTRAP_RESULTS.json](GOAL_CONTINUING_BOOTSTRAP_RESULTS.json)
stores the configurations, summaries, per-family evaluations, fit ledger,
extension counts, source hashes and completed artifact inventories. The root's
read-only finalizer checks artifact/count/frozen-evaluation consistency. The
smoke additionally has an independent neural-choice/physics/truth audit; the
larger pilot has **not** received that full independent audit.

Interpretation: the joint warm start learns and retains a substantial subset of
these supplied tasks. Autonomous continuation does not improve the frozen total.
The coupled interruption/goal-organization failures persist. There is no matched
control attributing the gains specifically to width, reward updates or direct
demonstration fits, and no claim that all four broad gates are closed.
The descriptive final trace rollup finds 4/16 physical root worlds satisfied
and correctly finished, with 0/10 urgent requests answered on time. It records
zero extra root operations after cached-True physical satisfaction. Thus the
remaining physical failures largely concern reaching goals and scheduling them,
not repeatedly refusing to finish a known satisfied root. This is a descriptive
rollup, not the pending full independent larger-run audit.

## What remains

All four broad gates remain open: independent computation adaptation beyond
supplied families; autonomous longer goal/subgoal organization with retention;
representation/callable growth beyond supplied interfaces; and full continuing
reader/reasoning/action/growth integration with historical uncertainty-aware
teaching and correction. The mixed run is a step toward these gates, not their
closure.

[The smoke diagnosis](GOAL_MIXED_SMOKE_DIAGNOSIS.md) separates scheduling and
decomposition failures from stopping. A guided long failure completes physically
and finishes, but answers an urgent request late after exhausting three candidate
searches. Arrival information differs by controller phase: execution observation
30 flags an uninspected external root; agenda observation 30 is inspected-goal
age. The unassisted six-edge failure ignores arrivals and selects zero-progress
derived work, yet immediately finishes known-satisfied goals. The unassisted
long failure never decomposes and never reaches a satisfied stopping state.
These are finite diagnoses, not proof that learning is impossible.

When work is explicitly resumed, first read the larger run's per-family results
and preserved traces. Diagnose the dominant remaining failure before changing
teacher, representation, credit or fit schedule. Keep genuine policy stopping,
arrival timing, task completion and old-skill retention separate. Do not infer
mastery from assisted success, a timeout, or a supplied reference execution.
For a small targeted follow-up, isolate arrivals during agenda computation from
arrivals during execution, then separate switching from rejecting irrelevant
subgoals. The current teacher itself has a documented agenda scheduling failure;
more imitation of it alone is not evidence that that case becomes solvable.
Use the final checkpoint as a retained baseline; do not launch another broad
campaign by default. Follow-up work is manual and without subagents.

## Validation and working conventions

The goal protocol and demonstration fitter have frozen optimized and ASan/UBSan
passes. After demonstration fitting was added, existing goal/mission tests pass
optimized and five-action parity remains exact. Mixed raw
skill extension passes optimized/sanitizer checks, fresh probes, nonzero history
retention and atomic rollback on a late failed revision. Earlier full-trainer
compatibility records remain in the linked validation JSON files. CMake targets
are registered, but WSL CMake is unavailable; builds used direct GCC 15.2.

Main entry points: `cpp/apps/goal_continuing_bootstrap.cpp`,
`goal_curriculum_demonstration.h`, `goal_control_learning_harness.h`,
`goal_reader_environment.h`, `goal_mixed_skill_extension.h`, and
`cpp/include/laminar/goal_control.h`. The source-tagged fit is implemented in
`relational_control.h`; the normal five-action instantiation excludes that path.
No remote push or deployment is part of this wrap-up.
