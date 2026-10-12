# Grounded reader to goal-controller integration

`cpp/tests/test_grounded_goal_bridge.cpp` connects the existing trainable
`GroundedSession` reader to the same seven-action
`GoalControlledRelationalSession` through `GroundedRelationalBridge`. This is a
focused integration and provenance contract. It does not establish learned
scheduling, a shared reader/controller training loop, or a complete `Organism`.

No bridge adapter change was needed. `capture()` and `project_current()` already
produce controller-independent evidence and `StampedScene` objects. The new
test passes those actual projected scenes both in `PublicGoal::visible` and
through the goal controller's token-owned `read()` callback. It does not build
a replacement interpretation from evaluator labels.

## What is supplied and what is learned

Each reader receives the existing declared atomic-question program, “Who has the
key now?”, and familiar observation transitions as bootstrap. The unfamiliar
report “The key was logged beside cy” is learned from raw text and a verified
final answer, “cy”. No event-operator label, parse, successful interpretation or
preferred goal is supplied for that report.

The relational query “join key with cy” is separately bootstrapped from raw
Boolean concept/language examples over complete structured scenes. Those small
bootstrap scenes are explicit assistance; they are not projected reader output.
The test never treats the reader's partial scene as a complete teaching scene.

Agenda and physical controller operations are forced by a test harness that
restores policy weights immediately after each choice. Mission terminal replay
then uses the ordinary implementation. This validates the connection and credit
protocol; successful forced traces do not measure policy acquisition.

## Integrated checks

The same protocol is run for relation channels 0 and 1. Each orientation uses
one continuing goal controller and two independent live readers with overlapping
entity names. The host supplies distinct world identities and a separate
dependency owner for each reader.

| Boundary | Checked behavior |
|---|---|
| Ambiguous raw report | Actual reader alternatives are preserved; their common-positive projection remains partial; the goal controller returns `Unknown`. |
| Raw final-answer learning | Reader teaching changes its evidence stamp, invalidates the earlier projection, and yields a current positive projection usable by the seven-action controller. |
| Two public worlds | Teaching or reading one reader does not change the other reader's evidence, facts, or controller observation revision. The second goal begins with its own earlier scene. |
| Callback ownership | A foreign token and generated projected read are rejected without consuming the valid pending callback. |
| Generated material | Generated public requests, report text, role markers, correction text and final-answer teaching cannot update the relevant evidence. |
| Withdrawal/restoration | Withdrawing feedback on a reader clone invalidates old stamps and dependent-view authority; restoration recovers the alternative worlds with a new stamp. The current restored projection is accepted through a fresh goal read. The source reader remains unchanged. |
| Partial negatives | A scoped negative available in the rich reader envelope remains `Unknown` in the conservative canonical scene, rather than becoming a fabricated closed-world `False`. |
| Mission credit | Both reader-backed roots close without early policy credit; one terminal mission receipt supplies 16 updates. A second mission on the same controller preserves partial-negative withholding and records its observed failure with eight updates. |
| Teaching boundary | Partial reader-backed goal closes add no language teaching receipt or semantic version change and create no mission semantic-evidence links. |

Across both orientations the test closes six goal instances in four missions.
The two positive missions have two roots each; the two follow-up missions have
one root each and record failure when the controller lacks the negative fact in
its partial scene. All declared failures are retained by the assertions and
credit ledger.

## Validation

The optimized C++20 executable is
`build-wsl/test_grounded_goal_bridge`. The focused ASan/UBSan executable is
`build-wsl/test_grounded_goal_bridge_sanitized`. The result record,
`logs/grounded_goal_bridge_v1/validation.json`, contains exit codes, captured
output, exact binary/source hashes and archived source snapshots. The associated
toolchain flags and scope are recorded there. Both executables pass; their stderr
captures are empty. A copy of the machine record is
[GROUNDED_GOAL_BRIDGE_VALIDATION.json](GROUNDED_GOAL_BRIDGE_VALIDATION.json).
This result is separate from the
earlier five-action bridge contract in
[GROUNDED_RELATIONAL_BOUNDARY.md](GROUNDED_RELATIONAL_BOUNDARY.md).

## Exact remaining boundary

The host still owns source/world identity, monotonic controller revisions and
the decision to forward a current projection. A `PublicGoal` or controller read
does not itself carry the reader's full evidence stamp, alternatives, explicit
negatives or scoped coverage. The rich envelope and dependency index remain
outside the goal controller. This test explicitly checks freshness before
forwarding a projection; it does not implement automatic stale-read invalidation
inside the mission wrapper.

There is still no derived-teaching registry that atomically maps reader evidence
corrections to replacement interpreted teaching scenes and every dependent
semantic/control receipt. The bridge reports invalidated dependencies; an owner
must decide and apply withdrawal/replay. This test does not claim that automatic
coupling, and deliberately creates no semantic teaching from partial projections.

Planning over reader alternatives and scoped negatives remains absent from
canonical `Scene`. The learned reader and relational controller retain separate
bootstrap/training mechanisms. Fixed reader question programs and primitive
priors are still required by the journal provenance guard; sharing a continually
repaired primitive bank requires a tested rebase and dependency-replay protocol.
The full `Organism` byte predictor is not connected by this test, and no sustained
mixed continuation or unassisted learned goal-management campaign is claimed.
