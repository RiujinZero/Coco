# Independent audit of goal-control preflights

`python/audit_goal_preflights.py` reads frozen experiment output and independently
recomputes arithmetic and public-event consequences. A passing audit means the
artifacts are internally consistent with the declared protocol; it does not mean
the learner acquired the reference policy or passed every capability gate.

The original 96-decision results are retained as a historical bounded experiment.
The user subsequently authorized more decisions and prioritized task completion
over efficiency. Those later variants require separate records; the old bound is
not a claim that the task is universally impossible.

## Historical 96-decision result

The authoritative inputs are
`logs/goal_switch_preflight_20260923_v3/measurement_{summary,trace,aliases}` and
`logs/goal_long_preflight_v1/protocol_final.json`, with their frozen source
manifests. The combined audit is
`logs/goal_preflight_independent_audit_v1/audit_96_preview_checked.json`; its exact
auditor source is preserved as `auditor_96_preview_checked.py` in the same
directory. The earlier arithmetic-only audit and its source remain preserved.

Adversarial checks in `tamper_checks_96.json` confirm that independent changes to
a charged cost, an arrival bit (with its bit key updated), a terminal-correctness
label, and a reference action are all rejected. Original experiment files remain
unchanged.

| Preflight arm | Independently reconstructed success |
|---|---:|
| Six-edge public-root switching, frozen reference | 12/12 |
| Six-edge public-root switching, semantic-learning reference | 12/12 |
| Six-edge initial zero-output policy | 0/12 |
| Nine-effect long task, frozen reference | 24/24 |
| Nine-effect long task, semantic-learning reference | 16/24 |
| Nine-effect direct exploration control | 0/8 |
| Nine-effect direct deliberate-search control | 0/8 |

The switch audit covers all 60 runs, including its failure controls. It
reconstructs 3,804 individual float charges, action callbacks under the declared
edge-add physics, terminal satisfaction/correctness, mission reward, discounted
cost and utility. It reconstructs 1,356 execution-view arrival flags directly
from posted requests, paid inspections and completed requests. The worst
reference utility margin is 0.6078020270 over the strongest checked failure and
0.4840114448 over the optimistic zero-cost 96-decision failure bound.

The long-task audit covers 64 runs. It reconstructs 2,718 execution-view arrival
flags, 289 terminal submissions and their deadline/correctness consequences,
and checks 715 paid-computation worlds against prior public requests and traced
physical actions. Agenda cost components, prediction totals, preview work totals
and normalized deltas agree with their records. All chronological costs and
discounted returns agree at full exported precision.

For 460 valid previews, a separate dynamic program recomputes minimum missing
edges over the exported acyclic potential graph before and after the planned
actions. This uses a different algorithm from the native grounding enumeration.
It also confirms 552 public progress-feature uses against earlier paid previews.

## Declared 256-decision variants and reference correction

The two v10 variants under `logs/goal_long_preflight_v2` increase the decision
limit and root deadline to 256. They retain the task, motor limit, arrival times,
reference rule and other work bounds. Their separate original-price and
success-first zero-price audits are in
`logs/goal_preflight_independent_audit_v2/audit_256_{original_prices,success_first}.json`.
The original-price audit also compares all 7,392 trace rows and 715 raw paid
computation receipts with the zero-price version. Only declared prices, public
price features, charges and returns differ; actions, work and outcomes match.

Both v10 price variants retain 24/24 frozen reference successes, 16/24 semantic
reference successes, 0/8 uniform controls and 1/8 deliberate-search controls.
All eight semantic failures continue to skip derived candidates carrying the
public cached `search_budget` flag after decision 96. They never pay beyond a
four-prediction derived preview allowance. This is a concrete omission in that
frozen evaluator reference, not evidence that more work is intrinsically useless
or that a learned controller cannot solve the task.

The separately declared v11 reference under `logs/goal_long_preflight_v3` fixes
that omission by paying another reason operation when that public flag is set.
It also activates a started, already-satisfied derived goal so it can submit its
terminal result. Frozen manifests confirm that only the evaluator application
source changes between v10 and v11; core headers and fixtures are identical.
The reference still consumes only the exact public floats and fixed arm/variant
selectors, with its resulting operation checked against the public legal menu.

| Corrected reference v11, each price variant | Independent success |
|---|---:|
| Frozen semantic knowledge | 24/24 |
| Semantic learning enabled, zero policy updates | 24/24 |
| Uniform direct control | 0/8 |
| Deliberate-search direct control | 1/8 |

The final audits are
`logs/goal_preflight_independent_audit_v3/audit_256_{original_prices,success_first}.json`.
The original-price audit additionally confirms equality of 5,974 trace rows and
669 raw computation receipts across the two price variants. Independently
reconstructed checks cover 4,055 arrival bits, 320 terminal submissions, 423
preview minima and 432 public progress-feature uses. All 15 semantic-learning
arrivals during active unfinished work succeed. Reference completion needs at
most 95 decisions and nine motors; the largest used derived allowance is eight,
and no exported derived computation exhausts the global 4,096 prediction cap.

Under the original prices, the worst corrected reference utility is
-0.1500985623, below the optimistic zero-cost failure bound of -0.0770858423.
That old efficiency margin is no longer a completion gate. The separately
declared zero-price objective makes every measured success positive and every
failure negative, while preserving all actual work. Across references and
controls its minimum success utility is +0.1408844129 and maximum failure utility
is -0.0770858423. These utility signs support the stated completion-first
objective; they are not evidence of policy learning.

Both audit directories preserve their exact final auditor as `auditor.py`.
The v11 audit has 671 exact public-feature/menu groups, including 163 that span
multiple active/inspected contexts. There are no deterministic-reference action
conflicts. The same visited-support limitations below apply. In particular, a
truly unresolved derived candidate at the global preview cap could still cause
this reference to repeat reason operations; the present successful runs do not
test that boundary.

The long task's urgent arrival occurs during active unfinished work in 16/24
frozen reference runs and 15/24 semantic-learning reference runs. Other posted
arrivals still count in mission outcomes but must not be described as active
interruptions. Both revised direct controls contain eight distinct motor
sequences over their eight seeds. The initial `protocol_initial.json` controls
cloned one warmed knowledge RNG across seeds and remain preliminary; they are
preserved and excluded from the authoritative independent-control result.

## Public choices and semantic modes

Source review of the frozen native reference functions confirms their decision
inputs: the switching rule receives the 32 public floats and legal menu; the
long rule receives the floats and its fixed declared direct-control arm selector.
Neither function reads goal IDs, evaluator truth, private world state, session
objects or a desired-success label. The auditor executes an independently
implemented rule against the exact exported vectors and menus, verifying every
reference operation. Evaluator truth is used separately to score callbacks and
outcomes, not to choose the reference action.

The updated sources build the same raw semantic lessons with each declared seed,
so physical exploration randomness is not accidentally cloned from one fixed
session. The actual long-control motor traces confirm distinct realizations.

Frozen switch runs keep their semantic model version fixed and retain no control
credit. Semantic-learning switch runs advance that version and retain one
terminal mission receipt with zero replay updates. Long records likewise show
zero policy updates in every arm, unchanged weights, no semantic revisions when
frozen, and revisions with one terminal receipt when semantic learning is enabled.
The native programs assert restored weights and zero updates throughout execution;
weights are not exported for an additional bytewise comparison by this auditor.

## What the alias check does and does not show

The switch artifact has 54 exact float-bit/menu groups, all repeated, with zero
cross-context groups. The long reference traces have 1,041 exact groups, including
187 containing more than one active/inspected context. Their recorded reference
actions are consistent within each group.

Consistency is expected from a deterministic function of those inputs. It is not
an independent proof that every reachable group has a feasible common choice.
The audit checks actual successful or failed witness trajectories and does not
enumerate counterfactual actions, off-reference states, unseen candidate orders,
or held-out task structures. It also does not assign evaluator-preferred actions
to each equivalence class. Thus the results support a public-feature reference
witness on visited support, with the recorded failures, not a universal
feature-sufficiency or learned-selection claim.

## Export and acquisition limits

The long export records all actions and the deterministic evaluator physics in
its frozen source, but omits a full observed-world callback after every action.
The auditor therefore replays that declared physics and cross-checks the result
at public-request and paid-computation snapshots. Its physical reason rows also
omit raw primitive-step counts; total returns and agenda components can be
recomputed, but not every physical inference work unit can be independently
remetered from this export alone.

These are evaluator reference preflights. No scheduling policy is trained, the
switch deficit control is an initial zero-output policy rather than a trained
warm checkpoint, and no post-bootstrap acquisition or held-out generalization
claim follows. Semantic-learning failures at 96 decisions are retained; later
user-authorized resource or price variants must be reported alongside them.
