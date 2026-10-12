# Mission credit substrate

The optional seven-action controller now uses the same policy, semantic memory,
empirical value table and credit replay implementation as the five-action
controller. This is an implemented protocol and credit substrate, not a result
showing learned goal switching, learned subgoal selection or long-task acquisition.
The goal-management design and its remaining experiment gates are in
[LEARNED_GOAL_CONTROL_NEXT.md](LEARNED_GOAL_CONTROL_NEXT.md).

## Representation and compatibility

`cpp/include/laminar/relational_control.h` defines
`BasicControlledRelationalSession<ActionCount=5>`. The existing
`ControlledRelationalSession` name aliases `BasicControlledRelationalSession<5>`.
The supported action counts are five and seven. Observations remain 32 floats.
Empirical values and model consequence arrays use `ActionCount`; the old public
empirical-value type remains exactly
`map<reasoning::Observation,array<float,5>>`.

| ID | Operation |
|---|---|
| 0 | `retrieve` |
| 1 | `reason` |
| 2 | `ask` |
| 3 | `act` |
| 4 | `finish` |
| 5 | `next_goal` |
| 6 | `activate_goal` |

Ordinary `choose()` still constructs the physical menu 0–4. Only the derived
mission controller constructs a seven-action or agenda menu. A seven-action
session rejects a `PreferenceTree` readout and model-based control credit.
The old `finish()` and single-evidence `revise_credit()` paths remain available
on the five-action controller. `revise_credit()` rejects a mission receipt,
whose semantic links must be revised together.

The existing constructor flags retain their meaning and defaults:
`learning`, `terminal_only`, `empirical`, `model_based`, `episodic_return`, and
`greedy_trace_credit`. Seven-action construction does not select a second
scheduler model. `GoalControlledRelationalSession` explicitly selects empirical
greedy-trace credit, reactive one-motor batches, a 32-motor per-goal limit and
retained search allowance. The five-action default continues to use its original
credit rule unless its existing optional profile selects another rule.

`ControlStep::goal` is execution metadata, not a policy feature. The base
physical trace records its internal episode identity; the wrapper's chronological
mission trace records the public active-goal identity. `GoalControlStep` uses
that inherited field, so slicing a mission row into `ControlCredit::trace`
preserves it. Additional wrapper metadata records the inspected goal, world,
agenda revision and mission clock.

## Protected integration API

These helpers belong to the base controller's protected interface. A derived
owner supplies its private chronological ledger and public observations; callers
do not submit an arbitrary preferred goal, private evaluator label, or trace.

| Helper | Contract |
|---|---|
| `valid_control_request(text,names,visible,actions) const` | Applies the structural request/world/affordance validation used by `begin`, without requiring an inactive episode or testing an episode ID. |
| `current_episode_id() const` | Returns the current internal episode identity. |
| `current_motor_count() const` | Includes acknowledged motors before and after suspension. |
| `select_control(observation,available,epsilon,memory_policy,readout,hint)` | Selects through the shared policy, RNG and hint rules. Works without an active episode, validates finite features and a unique legal menu, and issues a token pinned to model version and current episode. |
| `execute_physical(step)` | Requires an active current episode and an issued selection for operation 0–4. Checks token, operation, observation, menu, teacher, model version and episode. Computes the actual work, cost, actions and reply, then records that same public observation/menu in the physical trace. Caller-supplied result fields are discarded. |
| `close_episode_evidence(satisfaction,source,generated)` | Closes a finished physical episode into semantic evidence without a control reward or replay. Requires verified environment provenance. Returns `nullopt` on rejection; an accepted close may return an empty evidence ID for frozen execution or an incomplete teaching world. |
| `evidence_link(id,execution) const` | Reads the actual receipt version, activity and optional terminal satisfaction. |
| `pending_mission_evidence() const` | Returns the semantic links awaiting ownership by one mission. |
| `submit_mission_credit(id,trace,links,correct,replay_updates,random_credit,source,generated)` | Accepts one chronological trace and exactly all pending semantic links when no physical episode is active or suspended. Validates provenance, ownership, versions, costs, features, legal operations and unused increasing decision tokens before inserting a receipt. |
| `revise_mission_credit(id,expected,correct,replacements,active,source)` | Atomically revises every linked semantic receipt and replays policy credit from its initial state. Requires the expected mission version, exact ordered link identities and versions, and no active, suspended or pending semantic episode. |
| `mission_evidence_current() const` | Detects semantic versions, activity or satisfaction changed outside the coupled mission revision. Selection, consolidation and mission submission enforce this check. |
| `import_five_action_policy(source)` | Before any decisions, credits, updates or active/suspended episodes, copies the hidden block and each old output row/bias explicitly. Requires matching hidden width and a valid finite five-action source. Preserves the separately initialized new output rows and resets both initial and target replay weights. |

The selection helper does not require an active physical episode because agenda
decisions occur before activation and between goal completions. Execution does
require one. The split does not permit a caller to forge the physical result of
a selected operation or reuse a physical decision token.

## Semantic ownership and mission replay

`ControlEvidenceLink` contains an ID, version, execution/language discriminator,
optional satisfaction and activity bit. An interrupted own-action segment has
`satisfaction = nullopt`: it verifies the observed transitions without inventing
a successful or failed terminal goal. Seven-action suspension automatically
registers that segment in the pending mission ledger. Terminal evidence closes
register their nonempty links in the same ledger.

`ControlCredit` retains its old single-goal fields and adds `mission`, `active`
and a vector of links. A mission cannot omit a pending segment, reuse evidence
already owned by another credit, submit a stale link, or reuse a decision token
from a prior mission. Trace validation is bounded by the existing 256-decision
limit; the ordinary 8,192-credit and 2,097,152-update bounds also apply.
Mission replay validates the requested count against the remaining update
allowance by subtraction, so a very large `size_t` request cannot wrap the
budget check. Rejection preserves the pending semantic links for a valid retry.

Goal completion itself produces no policy update. The wrapper computes terminal
correctness from its verified root submissions and their public deadlines,
then submits one reward: +1 when every root was correctly submitted on time,
otherwise -1. Derived-goal completion has no independent control reward.
The existing per-step costs, 0.99 discount, greedy-trace return and replay schedule
operate on that single chronological mission trace. The default requested
terminal replay count is 128, but contract tests use explicitly declared smaller
counts where appropriate. `random_credit` remains an explicit experimental
option; it is not evidence of a distribution-preserving shuffled-label control.

Correction uses a copy of the controller and semantic memory. Each linked
revision must succeed before any state is committed. The mission version and
link versions advance together. Policy and target weights reset to the original
weights, including an imported warm policy where applicable; empirical values
and applied-update state reset; the stored update schedule is replayed in order.
Inactive mission receipts contribute no policy update. Restoring the same labels
and activity reproduces the original weights and empirical values exactly in the
focused contract.

Withdrawal deactivates terminal goal teaching and mission credit. It preserves
independently verified transition physics, following the existing
`RelationalSession::revise_execution` semantics. Unlabeled interruption links
stay unlabeled on correction, withdrawal and restoration. A caller must route
mission-owned semantic corrections through the bundled revision API; direct
mutation of a linked receipt is detected and rejected, not silently reconciled.

`has_mission_receipts_` starts false and becomes true only when a mission receipt
is inserted. Thus ordinary five-action selection does not scan its accumulating
credit history. The flag remains true through copies and withdrawal. It is not
an `ActionCount == 7` shortcut: five-action subclasses using the protected
mission API still receive the full evidence-consistency guard.

## Wrapper lifecycle boundary

`cpp/include/laminar/goal_control.h` privately inherits the seven-action base and
exposes selected diagnostics and model setup fields. Callers cannot invoke the
old episode `begin`, `finish`, `cancel`, `timeout`, `suspend`, `resume`, or
`cancel_suspended` methods through this public controller. Changing learning or
consolidating policy is rejected while a mission is active. Policy and knowledge
remain mutable setup objects, so this is a cooperative host protocol boundary,
not an access-control boundary against arbitrary host code.

Activation is two-phase: a paid policy choice issues a pinned ticket, then
`activate(token)` commits on a copy. A stale or failed callback consumes that
ticket and records rejection without committing a partial goal switch or
semantic update. Shared-world reconciliation and mission finish are also staged
on copies. A paid `next_goal` flushes any acknowledged active motor segment
before inspecting candidates, retaining the active goal and its trace while
updating the semantic model without a satisfaction label.

Unstarted derived proposals become invalid when their derivation model changes.
Started derived goals retain their request identity and suspended trace;
resumption reinterprets them under the current model with stale plans/previews
invalidated. This distinction is necessary for interrupting and later resuming
an unfinished child after its own acknowledged action changed the model.

## Bounded validation evidence

| Evidence | What it establishes |
|---|---|
| `cpp/tests/test_mission_credit.cpp` | Five/seven value-array types; exact warm import; shared selection and forged/replayed-selection rejection; chronological multi-goal credit with an unlabeled segment; no child control reward; omitted-link rejection; atomic correction, withdrawal and exact restoration; stale external evidence rejection. |
| `logs/mission_fastpath_validation_v1/validation.json` | Optimized mission contract passes after the cached-flag change. Includes a five-action mission subclass after ordinary credit, copied active/withdrawn receipts, and stale-link rejection by both selection and consolidation. Records source and binary hashes. |
| `logs/mission_fastpath_validation_v2/validation.json` | Later mission-only replay overflow guard passes the full mission contract and a nonzero-update overflow rejection test. The ordinary five-action probe remains byte-identical; its binary hash is also unchanged from v1. Current sources are archived under this result's `sources/` directory; the earlier v1 sources are retained separately and verified against their original recorded hashes. |
| `logs/mission_fastpath_validation_v1/before.txt` and `after.txt` | Byte-identical output for 27 ordinary five-action episodes, including traces, random exploration draws, replay, consolidation, corrections, all policy weights and empirical values. The before binary uses frozen pre-template headers. Raw LF stdout SHA-256 is `9fdc6f001fdc23d74b037e683b59cf7f6ddf9c8cf336b9ca587f7f7e6a3f4669`. |
| `logs/mission_substrate_parity_v1/validation.json` | Earlier pre-fast-path substrate checks, including the ASan/UBSan mission contract and the same five-action parity probe. Those captured text files use CRLF; normalizing line endings gives the current probe's identical text. |
| `cpp/tests/test_goal_control.cpp` | Wrapper protocol tests for public requests, world isolation, callbacks, deadlines, shared limits, mission correction, frozen clones and derived-goal execution. The wrapper owner maintains the executable validation record; the substrate parity records above do not substitute for rebuilding this separately edited wrapper. |

The earlier greedy-trace retention campaign is recorded separately in
[GREEDY_TRACE_VALIDATION.json](GREEDY_TRACE_VALIDATION.json). Its successful
five-action retention results do not validate a seven-action policy or establish
learned goal management.

## Paid computation provenance and remaining export boundary

The wrapper now appends a `GoalComputationReceipt` for each paid agenda reason.
`mission_computations()` exposes the current ledger; `computation_history(id)`
exposes a finished mission's retained ledger. These are const accessors.
Invalidating or overwriting an entry's current preview/proposal cache does not
change the recorded computation, and a later mission or credit correction does
not erase its history.

The record is keyed by the issued decision token and includes mission,
candidate and parent identities, public input revisions, model versions, bounds,
prices, answer, plan, primitive delta and rebuild status. Optional copies of
`goal_preview::Result` and `ProposalResult` retain the inputs, work,
definitions/witnesses and outputs. Both can occur in one paid reason. Root
interpretation records its prediction allowance/depth and resulting plan; its
planner uses the implementation's fixed `reuse=true` setting.

Cost reconstruction follows the original float addition order: operation,
rebuild if performed, root planning or the converted preview cost, newly
computed proposal cost if present, then primitive cost. The receipt also keeps
the charged total. A cached proposal that is merely retained is not copied as
a newly paid proposal in a later computation receipt.

This in-memory ledger is not a complete serialized campaign event stream.
Authenticated offers, deadline revisions, read/action callbacks and semantic
corrections still need ordered public event records, including world identity
and the goal lifecycle. A mutable model's current version alone cannot
reconstruct earlier versions after corrections. The wrapper does not yet provide
that public event serialization or an independent auditor for it. The wrapper
owner maintains the focused computation-receipt test and its separate build
validation; the mission substrate validation above does not cover that later
wrapper addition.

## Scope still open

Contract tests force selected protocol operations through a declared test
harness. They establish state, evidence and replay behavior, not acquisition.
The small derived-goal fixture fits the existing planner and is not a missing
long-horizon strategy. A claim of learned switching or subgoal selection still
requires the public-feature aliasing audit, a feasible long-task witness,
discounted utility better than checked failure strategies, a warm-policy deficit,
unassisted learning, held-out generalization and matched controls specified in
the design. Persistent mid-mission checkpoint serialization is also separate
from the tested in-memory copy and correction behavior.
