# Declared learned-values versus neural-fitting diagnostic

The immutable v1 protocol pilot failed all complete-mission frozen evaluations.
Root's read-only neural decoder found that the warm weights match assistance on
18 agenda states and 36 execution states without an arrival, representing 4,477
hinted decisions. They miss all 13 execution states with an unseen arrival,
representing 96 hinted decisions. This diagnostic tests whether empirical credit
already prefers switching in those rare states and whether fitting existing
values can recover that preference without new outcome evidence.

Deterministically reconstruct exactly the original 100 warm missions using the
same raw receipts, seed 1, schedules, explicit hints, 32-hidden-unit controller,
all-zero prices and 128 terminal updates. Compare the resulting float weights
exactly against the saved `warm100` checkpoint; also reproduce all 12 original
warm evaluation outcomes, decisions, motors, prediction counts and utilities.
Abort on any mismatch. The old v1 snapshot and all measured evidence stay intact.

Export every empirical value vector and every observed 32-float/menu readout,
including neural scores, empirical choices, assistance counts and the 13 rare
arrival states. The stored assistance labels are audit annotations; they are
never passed to consolidation or diagnostic evaluation as targets.

Evaluate a separate frozen copy with `memory_policy=true` on the same 12 held-out
cases. The ordinary core lookup uses only its existing learned values; unknown
states fall back to the neural policy. This is explicitly a memory-readout
diagnosis, with no acquisition claim for the neural policy. The harness option
defaults to false and is logged in each mission declaration.

On another independent warm copy, call existing `consolidate()` with
`rounds=min(256,floor(100000/unique_active_credited_observations))`. This bounds
additional fitting at 100,000 steps, far below the existing 2,097,152-update budget
after the 12,800 warm updates. If no complete round fits, perform zero rounds and
report that limit. Consolidation uses learned empirical preferences, not external
teacher labels. Verify identical empirical values, semantic version and credit
count afterward. Then evaluate the neural policy on another frozen copy, with
epsilon zero, no hints and no updates. No post-warm environment reward or hint is
collected for learning in any diagnostic arm.

The diagnostic app reuses the original pilot translation unit under a renamed,
unused entry point so the warm routines, task adapter and schedule cannot drift
through a copied implementation. It does not invoke that pilot's main function.
Source/binary/config and original input hashes must be recorded before running
the declared diagnostic. Its own new evidence directory contains the reconstructed
warm trace, all three frozen evaluations, value/readout exports and fit counts.
Results are pending review and execution of this declared diagnostic.

## Completed result

The approved diagnostic completed with exit code zero. Original warm weights and
all 12 warm evaluation outcomes/work counts were reproduced exactly. Evidence is
under `logs/goal_policy_distillation_diagnostic_20260923_v1/`, with pre-run hashes,
the frozen source, all empirical values/readouts and chronological evaluations.

| Readout | Complete missions | Observed behavior |
|---|---:|---|
| Original warm neural policy | 0/12 | Finishes physical work, then answers the urgent request late |
| Existing empirical values | 0/12 | Switches, answers urgent requests on time, resumes and performs six motors, then repeatedly reasons instead of submitting the satisfied physical root |
| Neural policy after consolidation | 0/12 | Performs six motors, then repeatedly retrieves; arrival switching remains absent |

Only 6 of the 13 guided arrival states prefer `next_goal` in the empirical table;
the warm neural policy chooses it in none. Consolidation fitted 113 distinct
credited observations for 256 rounds, totaling 28,928 additional fits. Empirical
values, semantic version and credit count remained identical. Neural agreement
with empirical preferences improved from 92/113 states to 98/113, but arrival
agreement remained 0/13 and complete-mission success stayed zero.

The empirical evaluation reveals a concrete value problem beyond rare-state
fitting. At a known, cached-True physical state after the urgent query is answered,
the table has `Q(reason)=.35640001297` and `Q(finish)=.34463128448`. It therefore
keeps reasoning even though the world is satisfied. That is a learned-table choice,
not fallback for an unseen state. The consolidated neural loop with an unseen
request and satisfied physical root corresponds to an empirical vector of seven
zeros. This diagnostic cannot attribute those values uniquely to state aliasing,
recency, trace cutoffs or replay distribution; those require separate tests.

The result establishes useful partial behavior in an explicitly selected memory
readout, but no complete-mission success and no neural switch/resume acquisition.
Additional fitting of unchanged values did not resolve it. The cached warm
procedure inventory also contained zero procedures; its dirty flag was recorded
as true, so this export alone is not a new materialization test or a proof of
reusable six-edge procedure learning.
