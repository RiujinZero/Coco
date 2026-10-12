# Nine-edge goal-control preflight

This is a preflight record, not a policy-training or acquired strategy result. The
earlier twelve-edge simple-route proposal requires thirteen variables and lies
outside the learner's current ten-variable concept bound. The replacement here
requires nine effects, which is above the physical planner's depth eight and
within the existing concept bound. No completion threshold was lowered.

`cpp/apps/goal_long_preflight.cpp` starts with the fixture's raw one-edge and
three-edge Boolean lessons and observed add-edge effects. It excludes the
six-edge concept so the nine-edge meaning must use three calls to the learned
three-edge concept. A positive nine-edge example, nine one-missing-edge negatives,
and shorter-route negatives teach the new concept through final Boolean labels.
No clause, decomposition or program is installed in memory.

The concept is uniquely induced: three local `short_route` calls expand to nine
`edge` atoms and ten variables. Induction uses 133,142 work and 2,625 candidate
checks. All 512 fresh edge-subset worlds agree with an independent exact-length
graph-walk evaluator. Removing any one of the nine required effects from the full
world makes the private evaluator's root false. An additional structural check
requires the expanded rule to be exactly one directed nine-edge chain between
its two head variables, up to variable renaming.

## Preserved phase 0: simple additive family

Artifact: `logs/goal_long_preflight_v1/semantic.json`; binary
`build-wsl/goal_long_preflight_v1`. The input affords nine necessary additions and
three additions on a disconnected three-edge distractor chain. There are twelve
public motor actions.

The generic proposer finishes with twenty instances: sixteen have positive paid
parent-progress previews and four have zero progress. At a four-candidate cap,
64 public order seeds retain some useful candidate in every case, but ten retain
no three-edge useful candidate and nineteen retain no irrelevant alternative.
The inventory costs 1,020 logical work and 0.0002208 at the declared preview
prices. Previewing all twenty costs 0.249743.

This family fails the necessary direct-execution ablation. A supplied sweep of
each public affordance once succeeds on all 128 shuffled action orders in twelve
motors. It needs no inferred subgoal or private edge labels. Uniform action
sampling with replacement succeeds on 71 of 128 seeds within 32 motors. The
abstract action/read accounting fits 96 decisions. These first ablations are
physical-effector simulations; they are not yet measurements through the actual
goal-controller callback protocol. Even so, the complete sweep is a decisive
counterexample to claiming that planner depth alone makes decomposition necessary.

## Preserved phase 1: forty public affordances

This separate family was declared before measurement. It retains the nine root
edges, one disconnected three-edge distractor chain, and adds twenty-eight edges
from a disconnected bipartite source/target component. It has forty motor
affordances and thirty-two public names. Distractors cannot create another root
walk. Limits stay at 96 decisions, 32 motors, 4,096 preview-planner predictions,
two million preview work, depth eight, and normalization nine.

Artifact: `logs/goal_long_preflight_v1/wide_semantic.json`; binary
`build-wsl/goal_long_preflight_v2`. The raw-evidence concept checks remain 512/512.
Full generic proposal enumeration finishes with 48 instances, costing 5,500 work
and 0.000946. Eleven previews are valid/useful, thirty-one valid/irrelevant, and
six exhaust the 4,096-prediction search. Evaluating all costs 0.81084. Across 64
candidate orders, cap four leaves no valid useful instance 25 times and no valid
three-edge useful instance 55 times. These cases are retained as failures.

The same direct ablations now score 16/128 for a randomly ordered sweep stopped
after 32 distinct actions and 0/128 for uniform sampling with replacement. This
improves the exploration contrast but does not repair proposal feasibility.

## Preserved phase 2: generic macro proposals, capacity eight

This app-local strategy was declared after phase 1 and before its own measurement.
It changes no learner header. It calls the existing generic proposer with its
64-instance source-inventory bound, requires that source enumeration finish
without truncation, then retains an instance only if **every expanded positive
branch contains more than one atom**. The filter examines learned definitions;
it contains no predicate-name, route-length, binding, usefulness, or evaluator
test. It preserves the proposer's public hash order, retains at most eight macro
instances, and explicitly reports any truncation. The original source search's
entire cost is charged. The forty-affordance physical family and every other
bound and price remain unchanged.

Artifact: `logs/goal_long_preflight_v1/macro_semantic.json`; binary
`build-wsl/goal_long_preflight_v3`. Eight short-route instances are retained for
every one of the same 64 public order seeds. Every order has two valid useful
three-edge instances, so the earlier capacity loss is resolved for this fixture.
However, the other five useful instances and the sole irrelevant macro exhaust
the unchanged ordinary planner allowance. All 64 orders therefore lack a valid
irrelevant preview. The intended comparison of useful and irrelevant candidates
with equal initial answer/plan flags is not established by this phase. Previewing
all eight costs 0.376008, in addition to the proposal cost 0.000946. The physical
ablation counts remain 16/128 and 0/128.

## Phase 3: acquired short procedure and actual public-feature execution

A separately declared bootstrap solves one short-route task with the ordinary
learned planner, records its three actual motor observations, and supplies the
verified final Boolean outcome. The session derives its own reusable procedure;
no program is installed. All forty later affordances receive the same public
`affordance(subject,object)` metadata, including every distractor. The existing
generic operand lookup can then bind intermediate entities. This changes the
public fixture transparently; it supplies no useful-goal bit or preferred route.

The generic proposer now exposes the opt-in `minimum_atoms=2` setting described
in `RELATIONAL_GOAL_PREVIEW.md`; the source enumeration still costs its full work.
The final semantic artifact is
`logs/goal_long_preflight_v1/procedure_semantic_final.json`. All eight proposed
macros have ordinary three-action reused plans: seven yield parent progress +3,
one irrelevant macro yields 0. All 64 order seeds preserve this contrast. Each
preview costs 3,956 work, including 86 predictions, and 0.0112468. Proposal costs
7,140 work and 0.00111. The new public metadata and acquired procedure resolve
phase 2's planning-coverage failure within the unchanged preview bounds.

The actual reference decision function reads only the declared 32 public floats.
It uses neither goal identities nor evaluator truth to select operations. It
instantiates positive-progress proposals, answers an urgent Boolean request
posted at the declared public clock, and finishes the physical root. Temporary
reference output preferences are restored after each choice. No reference policy
gradient update occurs. A separate semantic-learning condition permits the normal
verified execution/terminal updates and calls `finish_mission(0)`.

The immutable 96-decision protocol artifact is
`logs/goal_long_preflight_v1/protocol_final.json`, with full-precision features,
menus, actions, public request events, derived bindings, terminal replies and
submissions, paid computation inputs, learned definitions, predicted scenes,
potential domains and costs. Frozen sources are under `source_v9`, with hashes in
`source_v9_manifest.json`; `phase3_config.json` declares the experiment.

| 96-decision condition | Mission success | Actual unfinished-physical arrival coverage | Urgent replies on time |
|---|---:|---:|---:|
| Frozen grounded knowledge, eight seeds × arrival ticks 8/12/18 | 24/24 | 16/16 successful interruptions | 24/24 |
| Semantic learning enabled, same orders/arrivals | 16/24 | 9/15 successful interruptions | 24/24 |
| Existing uniform exploration, no derived goals, eight seeds | 0/8 | 0/8 successful missions | 8/8 |
| Increasing ordinary search then exploration, no derived goals | 0/8 | 0/8 successful missions | 8/8 |

The frozen reference needs 57–91 decisions and exactly nine motors. Semantic
learning causes up to nine knowledge revisions; eight cases reach the 96-decision
cap after only five to eight motors. These failures are retained, not relabeled
as successes. They are `(seed, arrival tick)` = `(0,8), (1,8), (2,8), (7,8),
`(2,12), (4,12), (0,18), (7,18)`.

The current reference also skips an already-satisfied, previously started child
when its paid progress delta is zero. The frozen traces show twenty such skips
in ten successful missions. Semantic-learning traces show twenty-four skips in
seven missions, including one of the eight failures. This leaves a pending goal
occupying capacity and adds repeated inspection/reasoning costs. That is a real
reference limitation, but these counts do not establish that fixing it alone
would remove every bounded failure. This preflight preserves the rule unchanged.

Costs use the actual chronological trace. The worst successful utilities are
-0.1203506883 frozen and -0.1347036025 with semantic learning. At horizon 96, even
an optimistically zero-cost failed mission cannot exceed -0.3848960789; a pure
`next_goal` timeout scores -0.5086866553. Successful traces therefore beat that
stronger failure upper bound by at least 0.2501924764 under the original prices.
The full semantic-learning reference still fails its old 96-decision completion
criterion; favorable utilities for completed cases do not erase those failures.

`protocol_initial.json` is preserved as preliminary evidence. Its direct arms
copied one exploration RNG state, so their eight repetitions are not independent
seeds. Final v9 reconstructs the same raw learned knowledge under each declared
seed, and only those reseeded direct controls support the table above.

## Phase 4: user-prioritized completion allowance

The user subsequently requested a higher decision limit and prioritized learning
task completion before optimizing speed. This authorizes a separately declared
256-decision/root-deadline condition. All 96-decision artifacts remain immutable.
The forty actions, thirty-two motor allowance, eight macro proposals, model and
preview bounds, eight seeds, arrival ticks 8/12/18 and urgent deadline slack 16
stay unchanged. The inherited core already permits 256 decisions; no core trace
limit changes are needed.

`logs/goal_long_preflight_v2/phase4_config.json` declares the new condition before
measurement. The executable now accepts explicit `DECISION_LIMIT PRICE_SCALE`
arguments after its output path and mode. Original prices are measured first.
If that cost objective discourages the slower successful completion, a separate
success-first objective sets all operation, work, management and motor prices to
zero while retaining measured decisions, work and motors. It is not combined
with the old price arm, and no prices are changed after training because no
policy training has started. The prior efficiency margin is no longer a gate on
the user's completion objective.

The measured phase 4 result retains 24/24 frozen and 16/24 semantic-learning
reference successes at both price scales. Each of the same eight failures stays
at exactly the same motor/revision counts after the extra 160 decisions: the
remaining choices merely cycle. Thus the earlier failures were not evidence
that the physical work necessarily needed more time. Direct uniform exploration
scores 0/8 and increasing direct search scores 1/8 at the higher allowance.
That eventual direct completion has utility -0.7337833957 at original prices,
below a cheap 256-step timeout's -0.2618228455. The separately declared scale-zero
objective raises that completion to +0.1408844129 while retaining its actual work
counts. Both original-price and success-first artifacts remain separately stored
under `logs/goal_long_preflight_v2` with frozen `source_v10`.

## Separate evaluator reference correction

Inspection of every stalled trace found an actionable public state that the
reference ignored. A derived candidate exhausted its first four-prediction
attempt; the wrapper correctly made eight available for a later paid `reason`.
The reference instead skipped that cached candidate forever. New semantic
procedure candidates had made the first attempt insufficient. Increasing the
decision limit alone could not correct this choice.

Reference variant 1 makes two explicit, generic changes: in agenda view, a
derived candidate with the public search-exhausted flag chooses `reason`; a
started, already-satisfied derived candidate activates so it can be finished.
No core code, prices, task definition, outcome label or hidden goal selector
changes. This is correction of an evaluator feasibility strategy, not a learned
policy improvement. Old reference variants are retained in full.

The new declaration is
`logs/goal_long_preflight_v3/reference_revision_config.json`; source is frozen in
`source_v11`. Both `original_prices.json` and `success_first.json` report:

| Corrected reference / 256-decision condition | Mission success | Actual unfinished-work interruption successes |
|---|---:|---:|
| Frozen grounded knowledge | 24/24 | 16/16 |
| Semantic learning, no policy updates | 24/24 | 15/15 |
| Uniform no-subgoal control | 0/8 | 0/8 |
| Increasing-search no-subgoal control | 1/8 | 1/8 |

Every reference success uses exactly nine motors; the largest decision counts
are 91 frozen and 95 with semantic learning. Every urgent request is answered
before its deadline. The largest derived preview allowance actually required
is eight; no attempt exhausts the global 4,096 cap. The correction therefore
resolves the observed stalls without changing either planner bound. The user's
256-decision allowance remains the declared completion budget for subsequent
work; the reference's faster completion does not reinstate the old efficiency
gate.

With original prices, the worst successful semantic-reference utility is
-0.1500985623. Under the separately recorded success-first objective it is
+0.3887839181, with exactly zero priced costs and all raw work preserved. Any
failure within the 256-decision horizon has utility at most -0.0770858423 in
that objective. This avoids preferring cheap failure to eventual completion.

The retry rule can still loop if a genuinely unavailable candidate repeatedly
exhausts the global cap. That is a stated boundary, not a globally sufficient
goal-management policy. None of these fixtures triggers it. No policy-acquisition
campaign, warm-policy deficit, structural transfer score or learned control
improvement is inferred from a successful evaluator reference.

The agenda preview's 4,096 prediction bound remains distinct from the inherited
physical controller's 32,768 maximum search allowance. Artifacts count actual
work and keep both bounds explicit. Remaining acquisition, frozen-warm deficit,
structural transfer and intervention-control experiments are separate from this
reference feasibility work.

Prices used for semantic proposal/preview accounting are prediction 0.00001,
grounding and atom checks each 0.0000001, answer call 0.005, planner call 0.005.
Interpreter/planner calls retain the logical metering boundary documented in
`RELATIONAL_GOAL_PREVIEW.md`. No successful-cost assertion, unrestricted goal
invention claim, or long-strategy learning claim follows from this preflight.

The independent audit passed both corrected price variants, recomputing 423
preview minima, 4,055 arrival flags and 320 terminal submissions, and confirming
that the price twins retain identical work, choices and outcomes. See
`GOAL_PREFLIGHT_INDEPENDENT_AUDIT.md` and its JSON record. The machine-readable
source hashes, bounds, results and preserved failures are in
`GOAL_LONG_PREFLIGHT_VALIDATION.json`.
