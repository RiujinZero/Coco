# Goal persistence and delayed credit

The warm-started 50k controller passes the existing three-seed operation screen,
but the earlier session required each goal to run without interruption. Once a
motor had acted, a changed world was rejected to avoid misattributing an external
change to that motor. Supporting other requests while a goal is unfinished needs
explicit memory and evidence boundaries before testing learned goal management.

## Implemented primitive

`ControlledRelationalSession` now provides `suspend()`, `resume(id, visible)` and
`cancel_suspended(id)`. Up to 64 suspended goals retain their request, names,
affordances, prices, trace and consumed motor allowance in the same learner.
Resumption clears old plans, cached answers and callback tokens; subsequent
computation uses the current learned model. It requires a trusted visible scene
with a nonstale revision. World changes during an interruption are not recorded
as effects of the learner's earlier actions.

An acknowledged action prefix becomes a versioned **observed segment** when a
training goal is suspended. Its transitions can inform effects, but it has no
terminal satisfaction label (`goal_verified=false`), supplies no language answer,
and cannot become a successful procedure. A later segment begins from the world
actually observed on resumption. The complete parent control trace receives its
terminal reward only when that parent finishes; intervening goals have separate
traces and rewards. Correcting a segment can preserve unknown satisfaction, and
an unrelated teacher cannot claim an execution-owned evidence identifier.

Frozen evaluation retains these working memories but commits no observations or
learning updates. Cancelling a suspended goal removes its pending request without
inventing a terminal outcome; already verified physical observations remain facts.
Suspension does not reset the motor or decision budgets. Copying the session
isolates both suspended requests and learned state.

This is supplied memory machinery. The current API caller chooses when to suspend
and which named goal to resume. It does **not** yet establish learned goal
scheduling, a learned stack policy or autonomous discovery of subgoals.

## Validation

`cpp/tests/test_goal_suspension.cpp` exercises an unfinished physical goal,
16 intervening learning episodes, a changed observed world, replanning, delayed parent credit,
exact correction/restoration, unknown-label segment correction, cancellation,
stale callback rejection, retained limits and frozen/copy isolation. Explicit
operations in that contract are protocol fixtures, not task-performance evidence.
The existing controller and relational-session contracts also pass ASAN/UBSAN.

The first version of the suspension contract passes with the existing TD rule.
A stronger assertion—positive value reaching the first parent decision after
16 intervening goals and a 64-update terminal budget—fails. The original test
source and failure are preserved in `logs/goal_credit_delay`. This does not show
that more TD replay could never work. It motivates explicit experimental
`controller_return` and `controller50k_return` profiles: they estimate each
chosen operation from the completed goal's discounted terminal reward minus its
observed costs, rather than bootstrapping through another estimated state value.
Teacher labels and child rewards do not enter those targets. Memory, replay
budget, value-update rate and neural preference fitting remain unchanged.
`test_episode_return.cpp` checks the targets against an independent discounted
sum and verifies the truncated-credit control and correction replay. These
profiles are separate experiments, not changes to the preserved TD baseline.
Both the independent return-sum check and the stronger 16-intervening-goal
positivity assertion now pass under ASAN/UBSAN with episode-return credit at the
same 64-update parent budget. This is contract evidence, not an autonomous
learning score; the completed three-seed training comparison is recorded under
`logs/episode_return_screen`. `logs/goal_credit_delay/return_validation.json`
preserves the tests and binary hashes alongside the original TD failure.

The three-seed episode-return screen is complete and independently audited:

| Controller / credit, with the same taper | Final | Recovery | Larger goals |
|---|---:|---:|---:|
| 1,221 / original TD | 372/384 | 85/96 | 36/36 |
| 1,221 / episode return | 361/384 | 93/96 | 30/36 |
| 50,013 / original TD | **384/384** | **96/96** | **36/36** |
| 50,013 / episode return | 352/384 | 89/96 | 36/36 |

Every arm retains 972/972 primitive checks. The data are byte-matched and all
seeds are retained in `logs/episode_return_screen/aggregate.json`. Stronger
one-budget credit propagation in the contract does not imply better autonomous
control: full returns include the costs and failures of subsequent exploratory
decisions. That is a plausible source of the regression, not a proven diagnosis.
The episode-return profiles remain experimental; the recommended configuration
keeps TD. A subsequent bounded probe reserved half the replay after the initial
backwards pass for the newly completed goal, alternating with older-goal replay
at the same 64-update parent budget. This candidate also failed the stronger
first-decision positivity assertion. The source, binary hash and failure are
preserved in `logs/recent_goal_replay_probe`; the candidate was removed from the
canonical code before launching a task-performance comparison. It does not
establish that replay priority can never help, but this tested setting is not a
fix for the delayed-credit contract. The existing recommended profile is unchanged.

The canonical trainer adds `INTERLEAVE_TEST`, inside an ordinary frozen `CHECK`:

```
INTERLEAVE_TEST <parent TEST fields without TEST> N (after_decisions <child TEST fields without TEST>)...
```

The parent runs its learned policy, receives an unrelated request at the declared
decision count, and continues after the child finishes. Resume supplies exactly
the public scene and revision held before the interruption, so no hidden world
information is supplied free of a read. Children also use the same frozen policy.
`SUSPEND`, `RESUME`, `SKIP` and `NESTED` trace records expose every switch and any
scheduled child that was never reached. Child scores are reported separately;
they cannot be dropped to conceal a failed interruption.

`python/interrupted_goal_experiment.py` appends this test to the same 400-episode
warm-start curriculum. Each seed has 36 fresh renamed parent goals (8- or
12-object physical tasks) and 48 child questions, with interruptions after one,
three, or one and five parent decisions. The independent audit recursively checks
all public observations, neural decisions, world transitions, plans, outcomes,
resumed state and journal counts. Final controller and primitive weights must
remain byte-identical to the preceding warm run. The first seed-0 pilot passes
36/36 parent goals, 48/48 child questions, all 48 resumptions and 324/324 primitive
checks. It makes 108 physical actions and 408 further-reasoning decisions after
search exhaustion. Controller weights, primitive weights and stage-4 values are
byte-identical to the original warm run.

The final three-seed TD-policy run is now independently audited at
`logs/interrupted_goals/summary.json`: **108/108 parent goals, 144/144 child
questions, all 144 resumptions, zero skipped children, and 972/972 retained
primitive checks**. Parents perform 324 physical actions and 1,227 further
reasoning decisions after exhausted searches. The largest successful parent
uses 25 decisions; unrelated child decisions are separately counted. Every run
retains byte-identical controller/primitive weights and stage-4 values compared
with its original warm baseline. The default 201-episode trainer regression also
passes exact resume and coupled correction; its trace, weights and configuration
remain byte-identical to the preceding build. These results are not learned
scheduling and are not pooled with the separate episode-return experiment.

Checkpointing in the file trainer still occurs between complete records. These
interleaved tests run in evaluation clones and never checkpoint a partially
suspended training stream. A live trainer needs serialized pending goals and
recorded callbacks rather than replaying physical actions; the present checkpoint
remains an offline deterministic-replay format.

Next: expose goal-memory decisions to the controller, use explicit bootstrap
demonstrations, withdraw that assistance, and test learning on new interrupted
sequences with terminal outcomes. Longer-credit and representation-growth gates
remain open; a working memory primitive alone does not close them.
