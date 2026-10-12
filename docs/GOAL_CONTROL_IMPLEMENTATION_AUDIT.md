# Learned-goal implementation audit

The final frozen wrapper verification is
[GOAL_CONTROL_PROTOCOL_VALIDATION.json](GOAL_CONTROL_PROTOCOL_VALIDATION.json).
Both optimized and AddressSanitizer/UndefinedBehaviorSanitizer runs pass from
`logs/goal_control_protocol_v1`. This includes later immutable computation
receipt checks: every paid agenda reason has matching token/goal/cost provenance,
cost components account for the charged total, and prior computations survive
cache invalidation, mission correction and a new mission. Another assertion
ensures uninspected internal proposals do not set the external-request arrival
flag. Source manifests and binaries are preserved with the run.

This records independent fixture and protocol review for the optional continuing
goal-control path. It is not a completed acquisition experiment. The implementation
contract is [LEARNED_GOAL_CONTROL_NEXT.md](LEARNED_GOAL_CONTROL_NEXT.md).

The user subsequently prioritized successful learning over efficiency. Current
learning experiments use 256 decisions and zero action/computation prices while
logging actual work. The historical utility-margin proposal below is superseded
by that success-first objective. See [remaining gates](LEARNER_REMAINING_GATES.md)
for the passing corrected reference and the first pilot's failed autonomous
switching result; the component tests here remain protocol evidence.

## Evidence boundary and reusable fixture

`cpp/tests/goal_control_fixture.h` supplies test-only `goal_test::RawLessons`,
`PublicRequestFixture`, `ReferenceWorlds`, and `Fixture` types. Use
`make_fixture()`, then `fixture.make_knowledge(seed)` to obtain a normal
`RelationalSession` trained through raw receipt APIs. The fixture does not write
learned concept definitions, action rules, procedures, policy weights, preferred
operations, or goal-selection labels into the learner.

The semantic warm data contains Boolean concept and language examples for a
directed edge, an exact three-edge route, and an exact six-edge parent route.
Every answer is computed by an independent finite graph walk in the evaluator.
Concept names, entity mentions, complete observed scenes, receipt IDs and final
Boolean labels are declared fixture setup. Two externally observed before/action/
after transitions teach edge addition. Row and column distractors distinguish
addition from replacement, and a second binding tests argument orientation.
These transitions are effects observed in the environment, not a supplied plan
for completing a request.

Public request inputs are limited to ownership IDs, world IDs, raw text, entity
names, an observed scene and revision, available actions, action intent and a
public deadline. They have no expected answer, hidden world, useful-child bit,
preferred next-goal ID, evaluator callback or computed policy feature. The test
harness owns `ReferenceWorlds`; it must never pass that object to the learner.
Its truth function does not call learned concept evaluation or planner code.

The standard fixture has these request cases:

| Request | Public world | Purpose |
|---|---|---|
| `physical` | `shared` | Six missing edges and three irrelevant edge additions; tests preserved action state and parent progress |
| `arrival` | `shared` | A Boolean request posted during physical work; tests public arrival, callback ownership and later shared observations |
| `separate_query` | `separate` | The same entity names and request words with different facts; tests isolation across world identities |

The texts `route n0 toward n3` and `route d0 toward d3` describe independently
reachable three-edge goals in the same public action inventory. They are useful
for testing whether a paid learned-model preview distinguishes parent progress
from irrelevant work. A fixture's choice to compare these examples is evaluator
setup; those roles are not fields in the candidate or controller input.

The fixture deliberately separates semantic warm data from policy bootstrap.
Protocol tests may force an operation by a declared test harness and restore
weights before credit, as existing controller tests do. Such tests establish
ownership, replay and API behavior only. They cannot be reported as learned
switching, learned subgoal selection, or post-bootstrap acquisition.

## Validation observed so far

An independent C++20 smoke executable was compiled with GCC `-O1` under WSL and
exited successfully. It built knowledge only from the fixture's raw receipts,
checked unique concept acquisition, tested all eight edge-presence masks at
fresh three-edge bindings, checked a fresh six-edge parent, and compared learned
action predictions with the independent environment for all nine afforded
actions. Acting in `shared` left the distinct world's revision unchanged.

| Learned concept | Surviving hypotheses | Reported induction work |
|---|---:|---:|
| `linked` | 1 | 19 |
| `short_route` | 1 | 488 |
| `parent_route` | 1 | 1189 |

The executable was `build-wsl/goal_control_fixture_check`. This smoke test is a
fixture check. No full trainer or acquisition campaign was started by this audit.

`cpp/tests/test_goal_control.cpp` now exercises the mission API using these raw
lessons. Its optimized `-O2 -DNDEBUG` and AddressSanitizer/UndefinedBehaviorSanitizer
`-O1 -fsanitize=address,undefined -fno-omit-frame-pointer` runs both exited zero,
including the added interrupted-derived-goal regression. The final executables
were `build-wsl/test_goal_control_agent_v2` and
`build-asan/test_goal_control_agent_v2`, compiled as C++20 with `-Icpp/include`.
Every chosen operation in this
test is explicitly forced by temporary output weights restored before credit;
the test verifies the protocol and credit mechanism, not autonomous acquisition.
The physical arrival's deadline is declared as the public arrival clock plus
twelve before offering the request. A separate fixed missed-deadline case must
receive negative terminal credit even when its Boolean response is correct.

The observed passing protocol assertions cover:

- Policy-issued inspection, activation, switching and resumption, with no public
  host `begin`, `finish`, `suspend`, or `cancel` escape path and no conversion to
  the seven-output base controller. Changing learning mode or consolidating
  policy during a mission is rejected.
- New public input during an owned motor callback, generated-input rejection,
  duplicate/foreign tokens, equal-revision conflicting scenes, stale reads
  acknowledged without rollback, and rebasing from a newer received observation.
- Shared-world propagation and isolation across distinct worlds with overlapping
  entity names; preservation of the original goal trace and spent motor budget.
- Stale activation after request, deadline or semantic revision; an invalid
  activation cannot partially replace the currently active goal.
- One chronological mission credit with identical public goal metadata, tokens,
  observations and costs; no policy updates on suspension or subgoal completion.
- Separate truth/correctness handling, repeated public goal IDs in later missions,
  exact policy replay after correction, complete bundle withdrawal/restoration,
  and rejection of incomplete, stale, generated or legacy single-goal corrections.
- Shared decision/motor limits, preserved original deadlines during correction,
  frozen clone isolation, and exact old-action predictions after five-output
  checkpoint import.
- Generic paid proposal generation, actual three-edge derived-goal execution,
  learner-issued parent resumption and one terminal credit covering both. After
  one child motor, an urgent request can interrupt and the **same instantiated
  child** resumes with its trace and motor count intact. Old unstarted proposals
  become invalid after new evidence; they cannot be selected as current goals.

The review found and reproduced two implementation issues before the passing
run: public goal metadata was lost when a derived step shadowed the base field,
and a rejected derived activation could partially suspend the old goal. The
wrapper now uses inherited goal metadata and transactional activation. Entering
agenda view commits any newly acknowledged motor segment before inspection, with
no invented satisfaction label. A further regression confirmed that this segment
boundary invalidates only unstarted proposals, preserving already instantiated
derived goals for later resumption.

## Protocol review requirements

The initial review identified these concrete invariants for the implementation
and its contract tests. The passing assertions above describe the tested scope;
they are not a proof for every external mutation or callback sequence.

| Boundary | Required behavior and adversarial case |
|---|---|
| Pending callbacks | A new public request may invalidate an uncommitted agenda selection and set the arrival flag. It must not discard the correctly owned callback of an already issued physical action solely because the agenda revision changed. Switching while a callback is outstanding must fail. |
| Goal and world ownership | Callback tokens remain bound to the issuing mission, goal, segment and world. Replay after completion, cancellation or resumption cannot update a different goal. Overlapping entity names in distinct worlds provide no authority to propagate facts. |
| Shared observations | Only acknowledged public observations may update the shared-world registry. Stale observations cannot roll it back, and equal revisions with different scenes must be rejected. Resume uses the newest observation actually received for that world, not evaluator truth. |
| Action provenance | An external shared-world change begins a new execution segment. It cannot be attached as an effect of an earlier own motor. Failed motors must preserve their observed failure flag and cannot reset motor allowance. |
| Selection and cache provenance | Candidate, request, affordance, model and observation revisions invalidate dependent preview/selection state. A preview is paid once for the logged inputs; `observe()` cannot rerun a hidden planner or recover stale results. |
| Arrival observation | The execution-view arrival bit is reconstructed from public request/revision and inspection receipts. Merely posting a request neither activates it nor labels it urgent or useful. Variable arrival positions are needed before a learning claim. |
| Terminal labels | A correctly returned `False` answer can be a successful query. Semantic satisfaction and response correctness are distinct. A child closing contributes semantic evidence but no separate positive policy reward. |
| Mission credit | The chronological trace includes agenda operations and every goal's execution decisions exactly once. One final mission credit owns the relevant versioned semantic evidence; correction/withdrawal must replay that credit without relabeling interrupted fragments. |
| Shared limits | Selection, previews and switching consume mission resources. Suspending, resuming or recreating a derived goal cannot refresh already spent motor or decision budgets. |
| Frozen evaluation | Clones may execute the protocol and collect evaluator measurements, but a frozen evaluation must not change policy weights, empirical values, semantic evidence or the trained source instance. |

The preview additionally needs a feature-aliasing audit: equal public observation
vectors and equal legal menus must admit a common feasible policy choice. Parent
progress may come from a generic charged computation over learned models and
public worlds. It may not be an evaluator's useful-subgoal label. If bounded
preview work cannot resolve the useful/irrelevant ambiguity, the first claim
must be limited to switching and resuming public root requests.

## Representation and campaign limits

The six-edge fixture is intentionally a protocol and preview case. It is within
the existing planner's depth limit and therefore cannot demonstrate acquisition
of longer execution organization.

The earlier proposed twelve-edge simple route is not currently a demonstrated
representable target. Default `concepts::Limits.variables` is 10, but a simple
twelve-edge route requires 13 variables. Raising that limit alone is insufficient:
clause canonicalization enumerates permutations of existential variables, so
eleven existential vertices introduce up to 11! permutations, exceeding the
default 20-million work budget before other inference costs. No limits were
raised and no parent definition was seeded for this fixture.

A later twelve-effect family needs measured representability and proposal
feasibility first. A denser graph using at most ten variables, or a separately
grounded computational goal, may be a candidate; neither is validated here.
Then an explicit legal execution must fit the declared budgets and have exact
discounted utility above immediate failure and checked cheap delayed-failure
strategies. Declare any new prices before bootstrap and keep them fixed across
training phases and matched frozen/shuffled controls. Also require a real warm
policy deficit, unassisted acquisition and held-out generalization. Protocol
fixtures and already solvable tasks do not satisfy those requirements.

Direct mutation of `knowledge` while a goal is already active is not a tested
recovery protocol. A semantic version change can stale the base episode itself,
not just an uncommitted activation ticket. Mission-owned outcome correction uses
the coupled mission API between missions; broader live external semantic edits
need an explicit supported recovery/ownership contract before being claimed.
Persistent mid-mission checkpoint restoration also remains outside these tests.
