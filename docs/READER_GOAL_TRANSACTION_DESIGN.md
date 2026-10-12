# Reader evidence and mission correction: bounded implementation contract

This is a read-only design review. No production implementation is added here.
The existing bridge can evolve into a transactional owner for reader views and
verified outcome corrections. It cannot turn current reader output into existing
complete-scene semantic teaching merely by adding ownership or provenance stamps.

## The current canonical reader cannot supply a closed teaching scene

`GroundedSession` requires `partial_observation=true` in its constructor.
`LanguageReasoner::request_snapshot()` sets every alternative's
`closed_world = !partial_observation`. `RevisableGrounding::at(report)` and
`GroundedArchive::at(episode, report)` preserve that reader mode. Therefore a
genuine current or historical canonical reader capture remains partial, even
when its interpretation is unique and its journal replay is consistent.

`GroundedRelationalBridge::project` preserves this restriction. It returns common
positive facts with `Scene.complete=false`; negative facts and scoped row/column
coverage remain in the rich envelope. Its `lossless_closed_world` structural
test supplies a separately constructed closed `Snapshot`, not output produced by
the canonical partial reader.

The existing relational consumer requires complete scenes for language teaching,
transition learning and execution feedback. A query close on a partial scene
deliberately creates no semantic receipt. Thus no genuine current reader capture
can authorize those complete-scene teaching paths. `verified_environment` is a
caller provenance tag, not a proof of completeness or correctness of a predicted
interpretation. Changing that tag, setting `complete=true`, or treating missing
partial facts as negative would cross the existing evidence boundary.

## Existing APIs and the gap

| Existing API | What it authorizes | What remains missing |
|---|---|---|
| `GroundedSession::clone()` | Staging a reader mutation on an independent reader copy | An owner committing reader and controller together |
| `observe(uint8_t, bool generated)` | Raw stream input with generated-text filtering | Typed world/capture ownership outside the reader |
| `teach(FeedbackAnchor, id, question, answer, source, kind, previous_revision, generated)` | Versioned raw final-answer teaching and withdrawal on the current history | Downstream dependency replacement |
| `teach_archived(Anchor, ...)` | Versioned teaching in an older archived episode | Binding a downstream receipt to that exact historical point |
| `journal.at(report)` / `archive.at(episode, report)` | Historical reader state reinterpreted under the current consistent evidence model | An authenticated bridge `capture_at` factory with historical anchors |
| `GroundedRelationalBridge::capture`, `project_current` | A current rich snapshot and a checked conservative view | Ownership of the resulting controller observation |
| `bind_dependency` / `invalidated_dependencies` | A coarse stale-stamp notification | Whether an old historical world actually changed |
| `GoalControlledRelationalSession::read(token, scene, source, generated)` | A pending read callback for the currently owned goal | A reader evidence handle attached to that observation |
| `revise_mission(id, expected, outcomes, active, source, generated)` | Atomic outcome/active-state revision, linked receipt revision and policy replay | Replacement of a linked receipt's interpreted scene |
| `RelationalSession::revise_execution_observations(id, expected, before, observed, executed, success, source)` | Updating one owned execution's scene/transition/language bundle | Coupled mission-link versions and policy replay; independent physics provenance |

`revise_mission` requires exactly one outcome for every retained submission,
preserves request versions, clocks and deadlines, and recomputes the one root
mission result. Its outcomes are booleans, not scene replacements.

Calling `knowledge.revise_execution_observations` first is not a safe composition:
it advances the semantic receipt version while the mission link retains the old
version. `mission_evidence_current()` then rejects the mismatch. Calling
`revise_mission` first does not replace the scene either. A query-only
`LanguageReceipt` has no public scene-replacement API at all.

## Minimal owner and immutable records

A new opt-in owner should contain one `GoalControlledRelationalSession` and a
map of separately owned `GroundedSession` instances keyed by stable public world
identity. It must not expose mutable references to either object. Existing
applications using the direct APIs retain their behavior.

The owner needs three different records:

1. **Reader capture:** world ID; reader instance ID; reset/archive episode key;
   reader episode revision; exact report or initial-context point; report and
   feedback revisions; raw-prefix digest; full reader stamp; alternative worlds,
   negatives, scoped coverage and names; channel mapping; projection version;
   owner-issued monotonic controller observation revision. Keep the initial
   capture immutable and append subsequent reinterpretations.
2. **Observation-use record:** mission ID; goal and request version; controller
   read/activation token; world ID; capture ID; actual forwarded `StampedScene`;
   result of the owned callback. This proves what the learner really observed.
   A later interpretation must not rewrite this chronological record or its
   neural features, selected action, paid cost or computation receipt.
3. **Verified outcome/physics record:** independent source identity, version and
   event time; raw goal/request identity; actual issued answer; externally verified
   terminal correctness/satisfaction; acknowledged actions and observed physics
   where applicable. These records are not inferred from reader predictions.
   They identify exactly which mission links depend on each verified result.

Use distinct types for `ReaderDerivedView`, `VerifiedOutcome`, and
`VerifiedPhysicsObservation`; one source string cannot distinguish them. Per-world
revision allocators must remain monotonic across reader resets. Different readers
with overlapping names and report IDs must never share a namespace implicitly.

## Proposed first increment using current learner semantics

The minimal useful implementation is a **view/history owner**, with no new
reader-derived relational teaching. Suggested public operations are:

```cpp
register_reader(WorldId, const GroundedSession&); // retain an independent deep clone
capture_current(WorldId) -> CaptureHandle;
capture_at(WorldId, HistoricalPoint) -> CaptureHandle;
offer_reader_goal(PublicGoalMetadata, CaptureHandle);
read_from_reader(ControlToken, WorldId, CaptureHandle);
stage_reader_edit(WorldId, ExpectedGeneration, TypedReaderEdit) -> Transaction;
revise_verified_outcomes(ExpectedGeneration, OutcomeReceiptEdits) -> Transaction;
commit(TransactionToken) -> CommitResult;
```

These names describe a contract, not APIs already implemented. The owner allocates
goal/world revisions; the caller cannot supply a new parse, projected facts,
successful interpretation, internal semantic receipt ID or replacement motor.
`TypedReaderEdit` wraps existing raw observe/teach/archived-teach operations and
their existing expected-version/source checks, rather than executing arbitrary
caller code. `CaptureHandle` refers to an owner-retained immutable envelope.

`capture_at` should be a bridge-owned factory using the guarded local journal or
consistent, untruncated archive. It must retain the original physical/history
point while recording the new interpretation model. A withdrawn report, missing
historical point, changed primitive prior or exhausted replay budget produces an
explicit unavailable result, not a synthesized earlier complete world. It must
also represent initial-context captures without inventing report ID zero as an
ordinary report.

Reader edits stage on `GroundedSession::clone()`. Outcome edits stage on a copy
of the continuing goal controller and call `revise_mission` with the owner-retained
exact submission inventory and expected mission versions. Commit reader,
controller, capture index and outcome registry together only after all version,
capacity and replay checks pass. A failed transaction changes none of them.
The same verified result shared by multiple missions updates all affected
missions on the staged copy before publication.

For the first increment, coupled semantic/outcome corrections require a quiet
mission boundary because current `revise_mission` explicitly rejects an active
mission. A received edit may be retained as a pending raw transaction, but it
must not be reported as applied or allow stale corrected evidence to be emitted.
Pending verified physics callbacks must finish under their original ownership;
the correction process must not cancel or invent an action acknowledgment.

Forwarding a current reader view remains an ordinary pending read. This owner
does not insert a free internal `reason`, choose a goal for the policy, or turn an
unsolicited report into a fabricated action callback. If later work needs immediate
unsolicited live-world updates, that needs a separately specified public-world
update API and callback-ordering tests; the present wrapper exposes only new
goal inputs and owned read/action callbacks for world changes.

## New report, reinterpretation and factual correction are different

| Event | Current view | Historical interpretation | Verified outcomes and motors |
|---|---|---|---|
| New live report about the same world | New capture and monotonic observation revision | The old physical point remains the old point; replay there only if shared meaning/evidence changed | Unchanged |
| Reader language/feedback correction | Recompute under the corrected reader model | Recompute at each affected **old** report point using `at`, never from today's live scene | Unchanged unless a separate authoritative outcome/physics correction exists |
| Explicit verified terminal-outcome correction | No automatic world rewrite | No automatic reader rewrite | Revise the identified outcome version, then coupled mission replay |
| Explicit independent physics observation correction | Requires its own source/version protocol | Reader interpretation alone is insufficient authority | Outside the first increment; do not route through ordinary language teaching |

The bridge's full stamp deliberately changes on new history or archive activity.
`invalidated_dependencies` therefore reports that a **current-view** stamp is
stale; it does not prove that every earlier teaching event became false. A new
report saying that an object moved cannot replace an older receipt's world with
the new location. Conversely, an actual shared-language correction can change a
past interpretation without making the learner's historical motor or terminal
outcome disappear.

The existing partial-view missions can legitimately keep their policy credit
when an old reader interpretation changes: the recorded policy acted on that
actual old view, and its external outcome remains factual. Do not manufacture
policy-label changes from reinterpretation. If a purported terminal label was
itself only reader-inferred, it lacked independent verification and must not
have entered `close_goal` as a verified terminal label in the first place.

## Additional contract needed for actual downstream semantic teaching

For independently authenticated **closed** evidence in a future input path,
add a dedicated coupled replacement operation that validates all old link IDs,
kinds and versions, replaces the allowed teaching-scene payloads, advances those
versions, updates mission links and replays the existing chronological update
ledger once on a staged copy. Preserve original trace observations, costs,
outcome receipts, request versions, deadlines and observed motors. Existing
`revise_mission` should retain its current behavior.

Such an operation also needs separate activity for a derived semantic lesson and
its factual mission outcome. The current one-boolean `active` operation withdraws
all linked semantic lessons and the mission credit together. Losing authority
for one reader-derived lesson must not silently discard independent physical
truth or force a still-verified mission reward to become unverified. A mixed
reader/physics mission particularly needs that distinction.

This closed-scene replacement protocol still does **not** make canonical partial
reader output eligible. Teaching from that output requires a new uncertainty-aware
receipt and learner contract that retains negative/coverage information and
shared alternative correlations across historical evidence. A collection of
closed candidate scenes, one arbitrarily selected candidate, or independent
per-receipt existential matches is not equivalent to those correlations. An
Unknown truth must not prune a semantic candidate as False. This is additional
learning-model work and should be evaluated separately from ownership plumbing.

## Minimal real-reader contract tests

Start from the existing `test_grounded_goal_bridge.cpp` reader construction:
familiar raw observations plus an unfamiliar “logged beside” report; learn it
from a raw final answer. Do not install that report's interpretation.

Required first-increment tests:

- Two reader worlds reuse entity names and report IDs; correcting one leaves the
  other world's revision, view and mission records unchanged.
- Ambiguous report, raw final-answer learning, feedback withdrawal and restoration
  change the appropriate historical envelopes. Every projection remains partial;
  no relational language/transition/execution lesson is added from those views.
- An ordinary new “ben has the key” report changes the current projection while
  the earlier “key with cy” historical point remains distinct. Outcome receipts,
  physical action records and completed-mission credit do not change.
- A genuine reader-meaning correction reinterprets the old anchored point through
  journal/archive replay; it never copies the current live scene into that point.
- A separately verified outcome correction reaches exactly the bound missions
  through `revise_mission`; withdrawal/restoration replay deterministically and
  preserve independent transition receipts.
- Stale owner, report, outcome or mission versions; foreign callback tokens;
  generated input/feedback; archive truncation; changed primitive prior; and
  replay capacity failure leave both live objects and every registry unchanged.
- Failed staging after one of several affected missions has been revised rolls
  back the entire batch. Full before/after weights, semantic receipt versions,
  outcomes, retained computation receipts and raw reader journal verify this.
- Reader resets do not reuse controller observation revisions. Archived teaching
  changes the correct earlier capture without replacing today's physical state.

The future teaching extension additionally needs a real partial-reader fixture
that succeeds through the new uncertainty-aware evidence path, plus unknown,
negative-coverage, correlated-alternative and correction tests. A hand-constructed
closed structural snapshot can test mechanics but cannot establish that integration.
