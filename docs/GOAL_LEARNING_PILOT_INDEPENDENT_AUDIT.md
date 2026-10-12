# Independent audit of the first goal-control learning pilot

The frozen pilot at `logs/goal_control_learning_pilot_20260923_v1` passes the
independent consistency audit in `logs/goal_learning_independent_audit_v1/audit.json`.
It does **not** establish successful autonomous goal scheduling. Every frozen
greedy checkpoint scores 0/12.

`python/audit_goal_learning_pilot.py` verifies seven predeclared binary/config/source
hashes and all 53 frozen source files. It checks all 760 missions and 62,885
decision records, their budgets, root deadlines, zero charges, predictions,
primitive counts, terminal reward and discounted utility. It independently
replays 8,638 motor/read callbacks under the declared edge-add physics and
recomputes 1,353 terminal truth labels and 39,582 arrival flags from public events.

The audit also runs a separate scalar float32 neural forward pass from exported
checkpoint weights. All 12,233 unassisted zero-epsilon decisions agree, covering
1,801 distinct checkpoint/input/menu combinations. Thus the failed greedy
trajectories agree with the actual exported policy, including the unassisted
portion of warm-up; no replacement evaluator action is needed to explain them.

The source implements assistance through `ControlHint`, confined to 100 warm
missions: 50 fully guided and 50 tapering from probability .98 to zero. All 4,573
recorded hints match the public-observation/menu reference; every later training
and evaluation row has no hint. The learner is created once and then copied at
the declared warm branch and evaluation boundaries. Mission tokens, model
versions, credit counts and update counts follow those continuing lineages.
Both control branches begin with identical warm weights, architecture, counters
and semantic version. All 200 frozen-policy snapshots retain exactly the warm
weights, while semantic evidence continues to change.

Every ordinary learning mission adds one terminal credit and 128 replay updates;
the frozen-policy arm adds the credit with zero updates. Evaluation has neither
kind of update, no semantic-version change, epsilon zero and no assistance.
Full weight arrays are exported at training boundaries; intra-mission weight
immutability and evaluation-clone immutability are additionally asserted in the
reviewed native program. The independent forward pass corroborates the observable
greedy choices, but does not replay every training gradient or epsilon RNG draw.

| Arm | Mission success |
|---|---:|
| Warm assistance and taper | 93/100 |
| Adaptive autonomous training | 9/200 |
| Frozen-policy autonomous control | 5/200 |
| Random-credit autonomous control | 2/200 |
| Each of five frozen greedy evaluations | 0/12 |

The failure signatures are specific. Warm100 and frozen-policy300 both complete
all 12 physical goals using six motors each, and answer all 12 urgent queries
correctly, but miss every urgent deadline. Adaptive300 instead makes no activation
or motor action: its 12 evaluations contain 24 goal-inspection operations and
3,048 reason operations. Random-credit300 completes the physical goals but never
closes the urgent query, making 2,403 ask operations across its evaluations.

The random-credit source uses an ordinary controller RNG draw for a +/-1 policy
reward while preserving true semantic labels. The export contains exactly 100
positive and 100 negative policy rewards. As declared before execution, those
extra draws can change later epsilon draws; controls have matched task schedules
and warm state, but not an identical realized exploration sequence. Cases within
one evaluation share a continuing cloned RNG, rather than being independent
reseeded trials.

This is a single-seed six-edge public-root protocol pilot. It contains no derived
goal scheduling or long-route acquisition test, and its small training-success
differences support no generalization or statistical superiority claim. Passing
receipt consistency preserves the negative learning result; it does not turn it
into a capability result.

The audit directory preserves the exact auditor and its imported helper as
`audit_goal_learning_pilot.py` and `audit_goal_preflights.py`. Original experiment
files are unchanged.
