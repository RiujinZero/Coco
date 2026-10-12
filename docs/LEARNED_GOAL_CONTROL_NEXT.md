# Next experiment: learned goal selection in the continuing controller

## Implementation status, 23 September 2026

The user has since explicitly prioritized learning the task at all, with speed
and efficiency deferred. The next experiment therefore increases the total
decision limit and physical-root deadline from 96 to 256 while retaining 32
motors. Preserve the original 96-decision measurements. The older efficiency
acceptance margin below is historical, not a gate on this revised campaign.
Measure work and timing, but require the training objective to favor completion;
a separately declared zero-price success-first variant removes efficiency
penalties during initial acquisition. This is an authorized change in experiment
priority, not a claim that the earlier budget tests passed.

The optional C++ mission API described below is now implemented in
`goal_control.h`, using the same templated controller and semantic learner.
See [MISSION_CREDIT_SUBSTRATE.md](MISSION_CREDIT_SUBSTRATE.md) and
[GOAL_CONTROL_IMPLEMENTATION_AUDIT.md](GOAL_CONTROL_IMPLEMENTATION_AUDIT.md).
Protocol tests use declared forced choices and establish execution/credit
behavior; they do not establish learned goal selection. There is not yet a
canonical trainer goal-control profile or persistence of an unfinished mission.
The five-action profile remains the normal trainer.

The twelve-edge route below was an initial proposal. Its thirteen variables
exceed the current concept limit, and increasing that limit triggers expensive
existential-variable canonicalization. Subsequent preflights use a separately
declared nine-edge route that is learned from raw final outcomes within the
existing representation. The first nine-edge task is solved by a direct action
sweep and fails the subgoal-necessity gate. A wider action inventory and a
declared generic macro-only proposal filter expose capacity and planning
failures, preserved as negative results. Short-route procedure bootstrap is
being checked separately before any acquisition campaign. These revisions do
not retroactively change the initial task or its failed preflights.

The current generic proposal filter is `minimum_atoms` (default one): after
complete paid enumeration, retain only concepts whose every expanded positive
branch has at least that many atoms, then apply the declared seeded order and
candidate cap. It does not test parent usefulness. Both pre-filter inventory
size and qualifying inventory size are recorded. The macro preflight explicitly
uses minimum two and eight retained candidates; these are experimental limits,
not changes to the normal profile.

`next_goal` now finalizes any acknowledged motor segment before inspecting a
candidate. This records observed transitions without a terminal satisfaction
label or policy reward. Unstarted derivations become stale when the model
changes; started children preserve their identity and trace and can resume
under the current model. Paid agenda computations have immutable receipts,
including exact inputs, definitions, limits, prices and work. Full public event
export remains a separate integration requirement. The execution arrival bit
counts uninspected external root requests/revisions, not internal proposals.

## Original experiment contract

This is an implementation design, not a completed capability result. The original
32-input, five-operation controller has no action that chooses a different goal.
`suspend()` and `resume(id, ...)` preserve its state correctly, but the host chooses
both the interruption and its destination. Padding an existing successful trace
with unrelated episodes does not measure acquisition of goal management.

The next boundary should let the same controller choose when to inspect other
goals, which goal to activate, and whether to instantiate a proposed subgoal. It
should retain the current grounded knowledge, physical-action protocol and
terminal credit mechanism. No separate successful-goal selector should be hidden
in the adapter.

## Minimum policy extension

Use an explicit experimental `goal_control` configuration, disabled by default.
Keep the 32-dimensional input and existing operation IDs 0–4. Add two outputs:

| ID | Operation | Actual decision |
|---|---|---|
| 5 | `next_goal` | Inspect the next candidate without activating it |
| 6 | `activate_goal` | Select the inspected goal; suspend the old goal if necessary |

The candidate inventory contains the active goal, suspended goals, pending public
requests and proposed derived subgoals. Inspecting a candidate does not execute
it, complete it, or transfer private world information. Candidate order is a
public, deterministic order whose seed is independent of usefulness. IDs identify
receipts and callback ownership only; neither goal IDs nor cursor indices enter
the neural input. Counterbalance order in training and change it in evaluation.

`next_goal` enters an agenda view. `activate_goal` either starts a pending request,
resumes its preserved episode, instantiates the inspected derived subgoal, or
returns to the already active goal. In agenda view, `reason` may compute the
inspected candidate's answer and plan against its already public observation;
the usual prediction/rebuild costs apply. It must not evaluate every candidate
for free inside `observe()`. The available menu in this view is `reason`,
`next_goal`, and `activate_goal`; physical execution requires activation first.

When a goal finishes, close its evidence segment and return to agenda view with
no active goal. The controller must choose the next goal. Never automatically
resume its parent. A mission ends when all externally requested root goals have
been submitted or the shared decision budget expires. Finishing a wrong root
goal is an observed failure, not an excuse for the host to retry it.

This uses one `BackpropChoice` and one replay/value learner. The network already
accepts a variable output count. Extending the controller also requires replacing
its hard-coded five-element empirical values, consequences, action loops and
price indexing with the configured action count. Do not enable the five-action
`PreferenceTree` in this first experiment unless it is extended and audited too.

When importing a five-output warm checkpoint, copy hidden parameters and each
of its five output rows/biases explicitly. Output-bias offsets change with the
number of actions; copying a flat prefix is wrong. Initialize the two new output
rows separately, then teach their protocol during declared warm-up. Check exact
old-action predictions under an old-action-only menu before further learning.

Default five-action construction, feature values, configuration JSON, serialized
values and checkpoint behavior must remain unchanged. Record the experimental
action count, agenda observation version, candidate limits and proposal strategy
in its configuration fingerprint. Reject simultaneous use of the old optional
quantitative features and this initial agenda feature contract.

## Public observation contract

Execution view keeps indices 0–29 and 31 unchanged, including index 29 = 0 for
the proposed profile. Its otherwise unused index 30 becomes a public
`uninspected_request_revision` flag: at least one pending request or relevant
request revision has arrived since its latest agenda inspection. Offering input
sets the flag; an explicit `next_goal` inspection acknowledges that inspected
revision only. Unseen requests keep it set. Deadline changes also set it. This
is derived from public arrival/inspection receipts, not urgency or a desired
switch. The default five-action profile keeps index 30 unchanged. Agenda view
uses the same vector, with index 29 = 1 as an explicit phase indicator:

| Indices | Agenda information |
|---|---|
| 0–3 | Inspected observation complete/nonempty; knowledge dirty/ready |
| 4–8 | Inspected goal has action intent; its answer cached/true/false/unknown |
| 9–10 | Its paid, current plan exists; its paid search exhausted its allowance |
| 11–15 | Candidate has a retained trace; an unfinished active goal exists; candidate is derived; candidate is active; candidate has a recorded parent |
| 16–17 | Active and inspected public deadline slack, clipped to the declared mission horizon; explicit sentinel for no deadline |
| 18 | Paid, current parent-progress preview is valid |
| 19 | Signed predicted reduction in the parent's relaxed missing-atom count, divided by the declared maximum parent atom count; zero when invalid |
| 20 | Inspected goal's acknowledged motor use divided by its allowance |
| 21–25 | Existing five operation prices, using the existing scale |
| 26–27 | `next_goal` and `activate_goal` prices, using that same scale |
| 28 | Active goal's previously computed planning failure/exhaustion flag |
| 29 | Agenda-view indicator |
| 30 | Inspected goal's waiting age divided by the mission horizon |
| 31 | Constant one |

The first profile fixes every public root request weight to one. Unequal weights
need a new declared observation contract; they cannot silently compete for the
two fields now reserved for a distinguishable consequence preview.

The words “cached” and “computed” are essential: fields cannot contain evaluator
truth, an oracle feasibility verdict, preferred next goal, expected reward from
the hidden environment, or the identity of a useful operator. Public deadlines
and weights are part of the user's request, not labels produced from the correct
schedule. The independent auditor must reconstruct the vector from receipts,
actual prior computations and the public clock.

Run an observation-aliasing preflight before training. Group reachable decision
points by the exact 32 floats and legal-action menu, including every changed
candidate order and urgent arrival position. The evaluator may annotate those
groups privately to detect incompatible required choices, but must never expose
its useful-subgoal or correct-switch labels to the learner. Inspect useful and
irrelevant candidates both before and after paid `reason`: equal unsatisfied,
reachable, parent/deadline/trace flags alone do not distinguish them. Before a
preview the common choice can be to compute it; after preview, the public features
must support a feasible common policy for each equivalence class. Verify this
with a reference policy restricted to these vectors and menus, whose decisions
never enter teaching. Failure blocks the subgoal-selection claim. Narrow the
first experiment to learner-issued switch/resume of public root requests if the
generic consequence preview cannot resolve the aliasing within its work budget.
The new arrival flag must also distinguish a fresh public request during physical
execution, so success does not rely solely on periodic agenda polling.

## Goal and provenance APIs

Add an agenda owned by `ControlledRelationalSession`, not a second controller.
Proposed interfaces are:

```cpp
begin_mission(id, limits, prices);
offer_goal(public_request);             // authenticated input; no activation
choose(...);                           // existing policy, new optional outputs
close_goal(terminal_observation);       // semantics/evidence, no immediate policy reward
finish_mission(verified_root_outcomes); // one terminal control-credit receipt
```

`PublicGoal` should contain its raw request or typed learned-goal reference,
names, already visible scene/stamp, public affordances, deadline and weight.
`DerivedGoal` additionally stores its parent, derivation/proposal record and the
knowledge version that justified it. These are symbolic execution metadata,
not numeric identity features. Selecting a typed learned goal invokes the same
concept/procedure memory; it must not call an evaluator predicate.

Keep an agenda revision and include it in selection tokens. Reject activation
after a relevant request, model or candidate revision. Reject switches while an
action/read callback is outstanding. On activation, use the existing suspension
segment boundary: observed actions can teach their effects, but interruption
cannot invent a terminal satisfaction label or successful procedure. Preserve
per-goal motor/decision consumption and enforce an additional shared mission
budget. Cycling through goals cannot reset either budget.

Resume from the newest observation the learner has actually received for that
world. A child action may provide a newer public shared-world observation; a
context switch alone cannot reveal the evaluator's current world. Existing stale
plan/callback invalidation remains mandatory. A learned model revision must also
invalidate pending derived goals and their cached plans.

## Subgoal proposal boundary

The first proposal mechanism may be supplied and bounded, but must be generic:
enumerate typed ground instances of already learned goal concepts from public
referents and learned relation/procedure structure. A bounded forward search may
filter for predicted reachable candidates, charging every explored state. Keep
all retained candidates and their public derivations in the trace, including
irrelevant alternatives. Do not accept a correct subgoal ID, a missing-operator
ID, an evaluator's decomposition, or a task-family-specific proposal recipe.

For the first physical family, reusable learned short-route concepts provide
candidate subgoals. Their meanings and bindings come from the learner's existing
concepts and public action effects. Require retained candidates that are useful
and candidates that are irrelevant or wasteful under the same resource bounds.
The candidate mechanism must be frozen before the campaign. If proposal search
itself finds and supplies a complete parent solution, report that assistance and
do not call the selection test autonomous goal decomposition.

`reason` on a derived candidate must additionally offer a generic, charged
consequence preview. First obtain its plan with the existing learned planner,
then predict that plan in a private copy of the already public world using the
learned transition model. Do not act or change observed evidence. Compute parent
progress in the public before-world and predicted after-world from the same
learned parent definition. For this initial complete-world, positive-formula
family, define relaxed missing-atom count as the minimum number of distinct
unsatisfied ground atoms over compatible groundings of any parent branch.
Use public referents, parent bindings and learned definitions only. A grounding's
atoms must belong to one common finite potential-fact domain for both counts:
the union of the two preview scenes' facts and facts predicted from already
public affordances in those scenes by the learned model. This domain can be
constructed and joined with a bounded, generic relational
search; it is not a task-specific route decomposition. Ignore action ordering
and deletion only in this explicitly relaxed measure, not in physical planning.
The feature is the before count minus the after count, with the declared scale.
It measures the model's prediction, not actual success or evaluator usefulness.

Every model prediction, grounding expansion and atom-truth check consumes a
declared work unit and its fixed price. Log both counts, the predicted scene,
the exact learned-definition/model revisions, public affordances, work and
normalization. Charge the full computation, even if it exhausts its limit. If
the relevant world is partial, definition unresolved, no compatible grounding
exists, plan prediction fails, or enumeration does not finish, expose an invalid
preview and zero delta; never convert a partial best result into a completed
minimum. Invalidate the preview after any relevant observation, model, parent,
candidate or affordance revision. No paid search may be rerun inside `observe()`.
The reference feasibility and utility checks must include this work. This is a
supplied generic progress computation whose use is learned; it does not establish
that the learner invented the progress measure or decomposition algorithm.

This first extension learns **whether and which proposed subgoal to instantiate**.
It does not yet learn the proposal algorithm or invent an unrestricted new goal
language. Those remain distinct requirements.

## Concrete family: long route with an urgent intervening request

Use the existing learned binary edge effects and learned reusable route concepts.
Bootstrap short routes and goal-management protocol, including successful
switch/resume examples. Keep existing old tasks interleaved. Separately teach the
long parent concept from verified Boolean examples, so failure in the following
test is about control rather than an unknown definition.

The post-bootstrap physical root requires a route containing twelve initially
missing edges. Its public affordances contain those additions and distractor
additions; no single action completes the root. The current planner searches to
depth eight, so repeatedly increasing its state allowance cannot produce a full
twelve-action plan. Predicted three-edge route subgoals are executable by the
same planner. Completing two useful three-edge subgoals leaves six parent edges,
within the existing planning horizon. This is a genuine change in required
execution organization, not idle padding between successful decisions.

After a fixed public decision-clock tick, a separate already grounded Boolean
request arrives with a short public deadline. It needs only a read if necessary,
reasoning and return. The learner must interrupt its unfinished physical work,
finish the urgent request, and select the physical parent again. The host posts
the request; it never calls `suspend` or selects its parent for resumption.

Start with at most three root requests, four retained derived candidates, depth
two of parent links, 96 total decisions and 32 total motors. Declare proposal
work separately, initially at most 4,096 prediction states per invocation. These
are proposed limits to be checked during feasibility preflight, not measured
successful costs. A candidate cap may not silently remove all useful subgoals;
capacity/proposal failures are recorded as failures, not skipped cases.

Before training, construct an explicit legal reference execution using only
these public requests, candidate proposals and callback APIs. Count all preview,
switch, read, search and motor costs. It must fit the same limits and deadlines
that the learner receives. The reference is an evaluator feasibility witness;
its action/goal sequence is not provided during post-bootstrap training.

Budget feasibility alone is insufficient. For a trace with T decisions, compute
the exact training utility `G = 0.99^(T-1) * R - sum(0.99^t * cost[t])` from its
recorded costs. Require the twelve-effect successful witness to exceed immediate
failure's utility by at least 0.10 under those exact prices. Also compare against
cheap timeout/cycling strategies: discounting a negative terminal reward can make
delayed failure preferable to immediate failure when per-step costs are small.
The successful witness must exceed the strongest checked failure strategy by
the same margin. Report both utilities and every cost component. The preceding
eight-effect preflight already showed that a feasible successful execution can
be discouraged by the price objective; do not carry that error into this family.

If the existing prices fail this criterion, declare a new experimental per-mission
price configuration before bootstrap, rerun this feasibility comparison, and
record it in the configuration fingerprint. A declared uniform cost scale is one
candidate; it is not assumed to pass. Freeze the accepted prices across bootstrap,
acquisition, evaluation and every control. Do not change terminal reward or costs
after seeing a learning arm fail, and do not claim acquisition from a campaign
whose stated objective rationally favors failure.

Also run the frozen warm policy before launching an acquisition campaign. Require
an actual performance deficit on this exact family. If it already solves the
family, do not report later success as acquisition. Likewise, test the best
simple direct-execution strategy: if random/greedy physical exploration reliably
solves within the same budget without subgoals, the family does not establish
the claimed benefit of subgoal selection. Redesign the declared family before
locking the campaign, rather than hiding the successful ablation.

Hold out longer route structures, changed urgent-arrival positions, different
candidate orders and distractor action inventories. Name permutation alone is
insufficient. Keep required path facts, deadlines and costs known to the scoring
environment, but supply only their declared public portions to the learner.
Vary urgent arrival positions without assistance during acquisition and scored
evaluation. Independently reconstruct the execution-view arrival flag and every
agenda feature from public receipts and paid computations.

## Stable credit objective and controls

Use the existing goal-level terminal signal and costs across an entire mission:
reward +1 only when every required root goal is submitted correctly before its
public deadline, otherwise -1. Charge every computation, management operation
and motor; start with `next_goal = 0.002` and `activate_goal = 0.01`, retaining
existing other prices. Keep discount 0.99. Report undiscounted success/cost as
well as the exact discounted objective. Do not change rewards or cost units
between warm-up, adaptation, ablations and evaluation.

Maintain one chronological mission trace containing management and execution
decisions with their goal IDs as metadata. A subgoal finishing is not a positive
control reward. Its externally verified satisfaction may update grounded
evidence, but the policy receives terminal mission credit only after all root
outcomes or timeout. This makes the decision to undertake, leave and return to
a subgoal causally part of the same return. Retain individual goal traces for
action provenance; do not mix unrelated physical segments.

Current `finish()` immediately trains on a single goal and links one evidence
receipt. The experimental mode therefore needs a split close/credit path and a
mission receipt with a list of versioned semantic evidence links. Correction or
withdrawal must replay the affected mission credit as well as its semantic
evidence. It is not enough to append a scheduler reward to the existing child
rewards. Start with the tested credit rule selected explicitly in the profile;
compare alternatives under equal replay updates rather than silently replacing
the preserved baseline.

Lock three seeds, generators, budgets and thresholds before acquisition. A first
bounded schedule can use 160 management warm-up missions, taper assistance to
zero over the last 80, then 400 post-bootstrap missions with no hints and a fixed
128-update budget per terminal mission. Freeze learning for scored evaluations.

Required matched controls are frozen post-bootstrap control weights/values,
permuted terminal mission rewards with their sign distribution preserved,
disabled `activate_goal`, and disabled derived-goal instantiation while retaining
switch/resume of public root requests. Preserve the same semantic bootstrap,
request schedules, costs and resource limits. Shuffle complete mission labels,
not individual microsteps or child IDs. Audit semantic state differences so that
a controller comparison is not secretly a comparison of different primitive
skills. Record every failed seed, proposal failure and timeout.

Suggested locked acceptance is at least 95% mission success in each held-out
family on each seed, no more than two percentage points of old-task loss, and a
clear advantage over frozen/shuffled controls under the exact same objective.
Require a material ablation loss when derived-goal instantiation is disabled
before claiming that subgoals contributed. Independently verify all switches,
parent links, callback ownership, deadlines, paid proposal work, features and
terminal-return reconstruction. Verify exact correction/restoration of policy
credit and no lost unfinished goal in an in-memory clone. Persistent mid-mission
checkpointing remains a separate open requirement until it is implemented and
tested; existing file checkpoints occur between records.

## First implementation boundary

Implement the seven-output optional policy, explicit agenda view, tokenized
goal selection and mission-level credit ledger first. Add protocol contracts for
learner-issued switch/resume, multiple pending requests, same-world observation
ownership and credit correction. Then implement and preflight the frozen generic
proposal mechanism and the long-route/deadline family. Do not launch the training
campaign until observation distinguishability, feasibility, missing baseline
behavior and cost alignment are established. Keep the unchanged five-action
canonical regression as a required
compatibility check throughout.
