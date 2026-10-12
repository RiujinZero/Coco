# Declared 512-update continuation

This next bounded trial changes terminal replay coverage only during autonomous
learning. It reconstructs the original 100 warm missions and verifies exact saved
warm weights and the original 12-case warm evaluation. Two copies then receive
the same 200 public task configurations, indices 100–299: one with factual mission
rewards and one with the existing randomized terminal-credit control. Both use
512 updates per terminal mission. The original 128-update adaptive/random results
and zero-update frozen-policy continuation remain recorded comparators.

The controller remains width 32 with unchanged 32-float features, seed 1, epsilon
.10, 256 decisions, 32 motors, all-zero prices and no post-warm hints. The six-edge
task, public arrivals, deadlines, affordance ordering, callbacks and reward rule
are unchanged. Extra replay consumes additional RNG draws, so trajectories are
not coupled; public task inputs and warm starting state are matched. No new core
credit rule, global-budget feature, wider network or longer warm-up is bundled in.

Stopping is a separate acceptance behavior. A read-only postrun parser reports
actual physical satisfaction, policy-issued correct physical finish, on-time
urgent answers and whole-mission success separately. It also counts continuing
decisions at an already cached-True physical state, including reason/retrieve
loops. This accounting never enters policy input or reward and never triggers a
host finish. Only policy action 4 closes a goal; the declared decision limit can
time out a mission.

With at most 256 current decision rows, 512 terminal updates cover every current
row and then replay earlier receipts. In the original configuration a 256-row
failed mission used all 128 updates on its suffix and no prior-receipt replay in
that call. Coverage is therefore a concrete, isolated change worth measuring.

At each endpoint, first evaluate the ordinary neural policy on a frozen copy.
Only afterward create an independent endpoint copy and consolidate existing
empirical preferences for `min(256,floor(100000/unique_credited_observations))`
rounds, capped at 100,000 fits per arm. Verify unchanged empirical values, semantic
version and credit count, then evaluate its neural policy separately. No extra
environment reward or external action label enters consolidation. This separates
new value learning from fitting the resulting readout; improvement after fitting
must not be attributed to replay coverage alone.

All evaluations use the original 12 combinations, explicitly a reused
developmental suite. They provide no untouched confirmation result. The old
frozen-policy 200-mission control is reused only after exact warm reconstruction;
its task/runner semantics remain identical. Read its outcome counts from the
original mission/evaluation files and verify all 201 branch-start/mission weight
vectors exactly equal the saved warm weights. Hash those original inputs before
the run. Future width-1316 or 400-mission warm
experiments are deferred, rather than being mixed into this trial.

The bounded run has 100 reconstructed warm missions, 400 autonomous missions and
60 frozen evaluation missions. Per adaptive/control instance, updates before
endpoint fitting are `12800+200*512=115200`; even the maximum 100,000 extra fits
stays within the existing 2,097,152-update budget. Source, binary, configuration
and original-input hashes must be frozen before execution. The existing v1
pilot and diagnostic evidence remain immutable. Source/config awaits root review.

## Completed declared trial

After root review, the frozen trial completed all 560 missions with exit code
zero. The reconstructed warm weights/evaluation matched exactly, and all 201
original frozen-policy vectors matched that warm checkpoint before reusing its
comparator. Source, configuration, original-input and binary hashes are under
`logs/goal_credit_coverage_continuation_20260923_v1/`.

The 200 autonomous missions produced 5 complete successes with factual credit and
8 with randomized credit. The original frozen-policy comparator had 5/200. These
sampled counts and a single seed provide no evidence that broader replay coverage
solved the capability. Every endpoint scored 0/12 on the reused developmental
evaluation suite.

| Frozen neural endpoint | Physical worlds satisfied | Correct physical finishes | Urgent answers on time | Whole missions |
|---|---:|---:|---:|---:|
| Adaptive 512, before fitting | 12/12 | 12/12 | 0/12 | 0/12 |
| Adaptive 512, after fitting | 12/12 | 0/12 | 0/12 | 0/12 |
| Random credit 512, before fitting | 12/12 | 12/12 | 0/12 | 0/12 |
| Random credit 512, after fitting | 12/12 | 12/12 | 0/12 | 0/12 |

The adaptive policy before fitting issued 312 actions at already cached-True
physical states: 26 extra per case, reaching the motor allowance before stopping.
Its correct eventual physical submissions therefore do not demonstrate appropriate
stopping after completion. After consolidation, it instead made 2,451 extra
reasoning decisions at cached-True physical states and never submitted them. The
random-credit endpoints completed and submitted physical requests, then answered
the urgent queries late. No host completion rule was used.

Adaptive consolidation performed 40 rounds over 2,440 states, totaling 97,600
fits. The random-credit endpoint used 256 rounds over 386 states, totaling 98,816
fits. Both preserved their empirical values, semantic version and credit count
exactly. The before/after scores keep this fitting intervention separate from the
512-update continuation.

`stopping_summary.json` was produced by the frozen read-only independent-physics
parser. It distinguishes actual satisfaction, policy finish and urgent timeliness,
and preserves every mission's known-satisfied extra operations. Learning when to
finish remains central to the open control gate; physical effects alone do not
close it. The next trial has not been chosen in this artifact.
