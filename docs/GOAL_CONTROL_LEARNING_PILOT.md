# Declared public-root protocol-learning pilot

This pilot tests learning in the same continuing seven-output controller. It is
protocol-learning groundwork, with a supplied public-feature warm-up curriculum.
It does not close derived-goal selection, the longer physical family, or all four
learner blockers. If warm-up already solves evaluation, subsequent success must
be reported as retention, not acquisition of previously missing behavior.

The task uses the raw six-edge fixture's learned route meanings and action
effects. Teach those receipts once into the controller's 32,768-receipt semantic
session. Continue that same controller for 100 warm missions and 200 autonomous
missions. At the warm boundary clone two additional continuing arms: one freezes
policy weights while retaining semantic learning, and one receives randomized
terminal policy rewards while retaining true semantic outcomes. Each control
receives the same 200 public mission configurations as the adaptive arm. Nothing
reconstructs or resets the controller between missions.

The fixed pilot uses seed 1, hidden width 32, two public roots, no derived goals,
256 shared decisions, 32 motors, discount .99 and 128 terminal policy updates.
Every field of `GoalControlPrices` is zero. Work counts remain fully logged; this
is the user's success-first configuration, not an efficiency result. Warm missions
0–49 receive a `ControlHint` on every decision. Missions 50–99 use hint probability
`(99-index)/50`, decreasing from .98 to zero, with an independent declared RNG.
Warm epsilon is zero; all subsequent missions use epsilon .10 with no hints.
Reference hints see only the exact 32 public floats and legal menu. No code writes
policy weights, and autonomous/evaluation rows never call the reference function.

The physical root has six initially absent required edges; a motor adds at most
one public edge. A correct physical completion therefore needs at least six
motors plus inspection, activation and finish: nine decisions, even without
reasoning. Training urgent arrivals are at clocks 3, 5 or 7. Evaluation uses
clocks 4, 6 or 8, crossed with public affordance-order seeds 41, 43, 47 and 53.
Thus every successful completion follows the scheduled arrival. An incorrect
early finish is an ordinary mission failure; no special early arrival is posted.
The handler rejects any successful mission that omitted a scheduled request.
The urgent deadline is arrival+12; the physical deadline is 256.
Physical success requires both a satisfied evaluator world and an actual True
reply, matching the canonical runner; a False reply to an unsatisfied physical
request is a failure. The urgent Boolean query can correctly answer False.

Training affordance-order seed is `1000+mission_index`; all three post-warm arms
use identical indices 100–299. The private agenda RNG continues normally and its
actual order keys are logged. The public affordance seeds do not independently
reseed agenda order. Random credit uses the core's RNG and can shift subsequent
epsilon draws; this control is honestly named random credit, with matched public
tasks and initial warm state, rather than a coupled shuffled-credit experiment.

Evaluate before training, after warm mission 100 and after mission 300 for each
arm. Each evaluation is one frozen clone running the same 12 cases sequentially:
epsilon zero, no hints, no semantic updates, no policy updates. Verify unchanged
weights, update count, receipt count and semantic version, and preserve the live
learner. Report all checkpoint results, including baseline failures and warm-up
success; this one-seed pilot alone does not establish robust generalization.

`goal_control_learning_harness.h` accepts a generic `Environment` providing public
request schedules, observed reads, owned motor callbacks and independent terminal
satisfaction. It knows no route, useful-goal ID, expected action or private plan.
A satisfaction callback receives the public request, allowing future derived
requests to be evaluated from public text/arguments without generated-ID tables.
Mission correctness scores root submissions only, matching canonical credit.
A later long-task adapter can replace the six-edge environment while retaining
the policy, observation contract, callback protocol, budgets and credit path.

The pilot exports raw bootstrap receipts, every chronological observation/menu,
request, action, source-tagged callback, independent terminal outcome, semantic
version, mission credit/link, assistance marker and work count. It saves policy
weights initially, after every learning mission and at branch boundaries. All
learning happens through actual terminal `finish_mission`; child closure cannot
update policy. The frozen-policy arm calls `finish_mission(0)` and verifies exact
warm weights. Evaluation clones are discarded and never become training input.

Before a run, compile a frozen source snapshot, execute `--declare OUTPUT_PREFIX`,
and record SHA-256 hashes of the snapshot, binary and declared configuration.
Only after code/protocol review run `--run OUTPUT_PREFIX`; it refuses missing or
changed declarations and refuses to overwrite prior evidence. The bounded pilot
has 700 learning/control missions plus 60 frozen evaluation missions. No run has
started at the time of this initial declaration document.

## Completed v1 result

The declared seed-1 pilot completed all 760 missions with exit code zero. Evidence
is frozen under `logs/goal_control_learning_pilot_20260923_v1/`, including its
pre-run source snapshot, binary/config hashes, chronological source-tagged trace,
raw bootstrap receipts, every policy-weight checkpoint, and terminal credit links.
All charged costs were exactly zero; the work and motor counts remain recorded.

| Phase | Sampled mission success | Frozen greedy evaluation |
|---|---:|---:|
| Initial controller | — | 0/12 |
| Fully guided warm missions 0–49 | 50/50 | — |
| Tapered warm missions 50–99 | 43/50 | 0/12 after warm 100 |
| Adaptive, 200 autonomous missions | 9/200 | 0/12 |
| Frozen policy, 200 autonomous missions | 5/200 | 0/12 |
| Random credit, 200 autonomous missions | 2/200 | 0/12 |

The warm neural policy completes the physical route unassisted in all 12 frozen
evaluations, using six real motors and returning True. It keeps working on that
root when the public arrival flag becomes one, then answers the urgent request
34–43 decisions after its deadline. The frozen-policy control preserves that
behavior. After the adaptive arm's 200 missions, the greedy policy instead loops
on agenda reasoning, with no activation or motors. The random-credit checkpoint
completes the physical route, then repeatedly asks about the urgent query until
the decision budget expires. All 60 frozen evaluation missions fail the complete
mission criterion. The final unhinted warm mission also fails, so this is not
merely a held-out arrival-position failure.

The result demonstrates some guided protocol learning in physical control, but
does not establish successful autonomous interruption/resume acquisition. Guided
warm success, sporadic epsilon-exploration successes, and actual policy updates
must not substitute for the failed frozen evaluations. The primary descriptive
analysis is `descriptive_analysis.json`; independent audit is separate.

One concrete replay confound remains: `finish_mission(128)` processes the final
128 decision rows first. For a 256-decision failed mission all 128 updates come
from that suffix, so the first 128 rows and older missions receive no replay in
that call. Short warm missions do reach their early decisions and then replay
earlier receipts. This pilot did not change that core rule. Before changing the
learning mechanism, a separate declared diagnostic should compare the learned
empirical preferences with neural decisions and test bounded consolidation of
existing values, without collecting new reward or hiding assistance.

Independent audit passed all 760 missions and 62,885 decisions, including callback
physics, labels, assistance boundaries, frozen weights and terminal updates; see
`docs/GOAL_LEARNING_PILOT_INDEPENDENT_AUDIT.md`. The subsequent declared readout
diagnostic is documented in `docs/GOAL_POLICY_DISTILLATION_DIAGNOSTIC.md`; it
preserves this original pilot evidence and does not turn the failed gate into a
success claim.
