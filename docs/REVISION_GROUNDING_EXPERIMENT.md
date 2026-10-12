# Revising language evidence inside an ongoing episode — 2026-09-23

The experimental reader can now recover a discarded report meaning after an
explicit correction and rebuild dependent knowledge without resetting the
dialogue. Correcting an earlier report changes its original place in the history;
later valid events remain later. The same mechanism supports withdrawing and
restoring an identified report or teaching answer.

This is a bounded evidence journal and replay mechanism. Correction targets and
source identities are explicit caller-provided metadata. Coco does not yet infer
which source is wrong, learn unrestricted natural-language correction references,
or maintain a general temporal/probabilistic belief system.

## Mechanism

`revisable_grounding.h` retains a starting reader snapshot, ordered raw environment
reports, and verified question/answer records attached to report IDs. Each record
has a source identity, active flag and revision number. An append-only change
record preserves the original input and subsequent revisions. Replacing a record
requires its expected previous revision and the same source identity. Generated
input cannot add or revise evidence. These identities are supplied by a trusted
caller; the module does not authenticate sources from their spelling.

After accepted evidence changes, the journal rebuilds from its starting snapshot
and active records. Unfamiliar report frames enumerate the existing generic
evidence and update operations. Previously eliminated candidates are available
again during reconstruction. Verified answers constrain the knowledge state at
their original report, and consistent frame choices carry through later reports.
This retains the correlations between interpretations and history. It does not
turn a model prediction into a teaching target or directly insert the corrected
answer into current facts.

The final model retains every interpretation supported by the active history.
Earlier views are also filtered by the surviving global frame choices, allowing
queries about revised historical knowledge. Those historical queries use an
explicit report ID and the already learned ordinary question language. No English
temporal construction is learned here. Repeated answers and revisions of one
report do not inflate its distinct-scene support; withdrawal reduces that support.

Replay invalidates cached answers. Contradictions and exhausted report, branch,
history or evaluation budgets withhold usable current knowledge instead of
publishing an arbitrary branch. A failed reconstruction retains the last model
publication but marks the running episode inconsistent; it cannot be used as a
successful journal checkpoint. A subsequent correction or a larger replay budget
can recover by reconstructing the active evidence again.

`LanguageReasoner` now carries a monotonic historical-evidence revision. Pending
action credit and pending language teaching reject feedback from an earlier
revision, including when the corrected text is equivalent and the resulting
model signature is unchanged. A correction does not pretend that an issued
physical action has returned its observation.

## Experiment

The prior is the same pretrained primitive/observation/question reader used by
the preceding live evidence experiment. The neural body, primitive policies,
question meanings, original event language and shared grammar remain frozen.
Raw input and output pass through the actual Organism byte interface.

One continuing world receives 50 operations: thirteen reports, 25 teaching
records, and twelve explicit revisions or withdrawals. It acquires two new report
frames: an ownership update and a location-visibility gap. Each ultimately has
four distinct supporting reports and one retained intended rule. Relation channel
identities are exchanged on seed 1. No annotated state, correct operation or
argument-role label enters the teaching interface.

The first teaching answer is deliberately plausible but wrong. The learner
initially asserts the old holder. A later consequence answer contradicts that
interpretation and blocks use of the journal. Correcting the first identified
answer restores the intended rule and both current and earlier knowledge. A
location event that occurred afterward stays in the current state while the
historical view retains the earlier location.

Other operations replace an old report after a later ownership update, withdraw
and restore evidence, and remove an explicitly false teaching answer. Independent
teaching records are not silently discarded when their premises become
inconsistent: withdrawing a report can conflict with a later verified answer,
and the conflict persists until the relevant evidence is explicitly repaired.

| Measure | Each of seeds 0, 1 and 2 |
|---|---:|
| Operations processed within the same episode | 50 |
| Historical revisions/withdrawals | 12 |
| Recomputed current/historical world states audited | 1,104 |
| Definite outputs matching active-evidence predictions | 3,224 |
| Uncertain outputs matching active-evidence predictions | 9,324 |
| Contradictory intermediate states | 5 |
| Answers withheld during those states | 1,320 |
| Frozen fresh-world current/historical questions | 1,536 / 1,536 |

The training-output counts measure agreement with independently reconstructed
evidence constraints. They are not all accuracy against an uncorrupted physical
world: the deliberately wrong initial feedback causes a wrong initial assertion.
The frozen evaluation uses corrected evidence and independently specified
semantics. Its 48 fresh-name worlds each exercise eight report/revision phases,
with two questions about current knowledge and two about the first report's
historical state. It contains 1,296 definite targets and 240 unknown targets per
seed. All model meanings remain frozen; there are no teaching answers in these
worlds. Full positive/negative/coverage states are checked with the answers.

These are repeated controlled templates in a small domain, not 1,536 independent
language problems or proof of general continual learning.

## Curriculum failure found

The initial visibility lessons asked where each actor was. An unknown unique
answer allowed both genuine visibility loss and a spurious additional location:
multiple known locations also lack a unique answer. The native run alone did
not expose this distinction, but the independent audit rejected the final model
and its intended knowledge semantics.

An added membership question — whether the actor is located at itself, with a
verified unknown target — removes the spurious self-location updates. This is a
new contrast in the experimental teaching stream, not a hard-coded mapping from
the visibility words to an operation. It demonstrates why unknown unique-answer
lessons need additional questions distinguishing missing evidence from known
nonuniqueness. The user-owned curriculum and its builders were not modified.

## Validation and limits

`python/check_revision_grounding.py` independently enumerates generic report
interpretations, replays every active evidence version, and compares every
retained world and consistent frame binding. It checks candidate sets, support
counts, provenance changes, unchanged episode identity, historical revision
counts, current/historical answers and the frozen semantics. Source, input,
binary and output hashes are recorded in `logs/revision_grounding/audit.json`.

Address/undefined-behavior sanitizer contracts cover erroneous teaching and
recovery, restored uncertainty after withdrawal, cached-answer invalidation,
later-event preservation, historical views, repeated evidence, revision/source
checks, generated feedback rejection, bounded replay, prior/state changes and
stale action/language credit. The preceding online, live observation and live
partial-observation suites were rebuilt and pass their contracts, three-seed
runs and independent audits with the historical-revision change.

The journal replays a bounded history from a retained snapshot. It does not yet
use incremental dependency indexes, persist and reload its state across process
restarts, discover source reliability, resolve noisy evidence probabilistically,
or revise question meanings, primitive policies and representations together.
Its live acquisition uses known single-token entities, two relation channels and
small argument lists. Pronoun/reference resolution is outside this journal API.
Correcting already executed actions would require additional causal and credit
handling; this change invalidates their pending credit rather than undoing the
physical action. Broad temporal language and standard-trainer integration remain
open. This route uses symbolic constraint elimination atop pretrained local
plasticity, without end-to-end backpropagation.

Run `bash scripts/revision_grounding.sh` under WSL with the existing primitive and
bootstrap/incremental/dialogue/observation training files. It generates the local
experimental stream, builds native and sanitizer probes, runs three seeds, and
invokes the independent auditor.

Follow-up: [historical-reference grounding](HISTORY_REFERENCE_GROUNDING_EXPERIMENT.md)
learns bounded report references from historical answers and transfers them to
correction selection. The journal now retains inactive records' original text and
answers so they remain identifiable; this report's regression results still pass.
Correction operation and payload parsing remain outside the learned reference.
