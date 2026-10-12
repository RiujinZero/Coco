# Retracting earlier lessons across episodes — 2026-09-23

The optional grounded session can now retain the raw evidence behind acquired
observation meanings across episode resets. A teacher can correct or withdraw
an old identified answer, reopening interpretations that answer had ruled out.
The archive then reconstructs current and historical knowledge, preserving later
events and keeping physical facts from separate episodes apart.

This establishes a bounded evidence-revision mechanism inside the actual byte
trainer. It does not establish joint learning of language, primitive procedures,
control, representations and physical actions. Those remain the gates in
[FINAL_CAPABILITIES_VALIDATION.md](FINAL_CAPABILITIES_VALIDATION.md).

## Mechanism and interface

`GroundedArchive` keeps a fixed bootstrap, raw initial contexts, report records
and the current versions of identified teacher answers. Replay starts from the
bootstrap, not the last narrowed observation model. Episode boundaries clear
physical knowledge while retaining correlated meaning alternatives. Later
evidence can therefore revise an old report's interpretation without leaking its
physical facts into a new world. Historical views are reconstructed from the
same surviving alternatives as the current state.

This is a current evidence-version store, not an immutable log of every prior
edit value. Globally distinct scene IDs prevent support inflation when episodes
reuse local report IDs. Withdrawal reduces support; restoring the same receipt
does not manufacture another teaching scene.

Enable it before input with `GroundedSession::enable_archive()` or with
`laminar_train grounded_archive=1 grounded_bootstrap=PATHS`. The existing
`grounded_feedback=FILE` sidecar can then address earlier episodes using its
episode/report/evidence IDs and expected revisions. The host must supply verified
feedback and an allowed teacher identity. Earlier-episode references here are
structured receipt metadata, not newly learned natural-language references.

The teacher's identity, report revision, archive version and evidence revision
guard edits. Generated text cannot teach or reset the archive. Accepted evidence
may contradict other evidence; acceptance does not certify global consistency.
Contradictions withhold derived answers and historical views until repaired.
Resetting the physical episode does not erase that conflict. Accepted old edits
invalidate current pending choices/actions. Mid-context feedback preserves the
unfinished byte segment and allows the rest of the initial context to arrive.

Storage and replay are explicitly bounded: by default 128 episodes, 8,192 records,
8 MiB of accounted raw text/name storage, 4,096 branches, 65,536 history states
and 4,194,304 evaluations. These are algorithmic limits, not an exact process-RAM
limit. Capacity failure withholds rather than silently dropping provenance; lost
input requires reconstruction from complete evidence. Replay is currently global,
not an efficient incremental dependency graph. An oversized old-receipt edit is
rejected before acceptance, preserving the prior archive and allowing retry.
The archive has no disk save/load.
Bootstrap questions, grammar, programs and primitive weights must remain fixed.

## Experiment and independent checks

The stream contains 12 episodes, 392 turns and 313 teacher receipts. The first
episode teaches a plausible but wrong meaning for an unfamiliar ownership
construction. Correct teaching in a fresh world exposes the contradiction.
Three revisions to the old answers restore the intended meaning while preserving
a later ownership event in the current world. Later episodes teach a visibility
gap, add and withdraw contradictory old evidence, use learned phrases in new
contexts, and correct reports using acquired language.

Three seeds use 1/7/113-byte chunks; one also swaps the two relation identities.
Each seed passes an independent evaluator's checks of:

| Measure | Per seed |
|---|---:|
| Replies | 351 / 351 |
| Correct withholding during conflicts | 28 |
| Transfer replies in episodes 9–12 | 176 / 176 |
| Report corrections | 8 |
| Current/historical knowledge states | 5,472 |
| Receipt-version snapshots | 113,040 |
| Acquired frame/support snapshots | 680 |

These are repeated controlled scenarios, not hundreds of independent language
tasks. The independent replay audits raw reports, corrections, source ownership,
active evidence, exact answers, projected states and support. The native check
also verifies that prior question/program/reference/correction/primitive models
did not change.

The actual trainer accepts all 313 receipts with zero rejections or pending
receipts. Runs with and without periodic checkpoint probes finish with identical
observation models, world state and stream/archive counters. The final model
matches the native experiment and its state matches independent replay. Of 23
checkpoint QAs, 18 answer correctly and five withhold while the evidence history
is unavailable; the final checkpoint recovers. This deliberately conservative
global withholding also blocks otherwise unrelated questions during a conflict.

The ASAN/UBSAN contract test covers cross-episode correction, correlation of
meaning alternatives, physical isolation, withdrawal/support recovery, generated
and unauthorized evidence, stale anchors, deep clone isolation, capacity failure,
and feedback during a partially read context.

## Reproduction and scope

Run `bash scripts/grounded_archive.sh` from the repository root in WSL. It uses
the preceding grounding experiments' bootstrap teaching files, generates this
experiment's own corpus, builds the native probe/trainer and sanitizer test,
runs three seeds and actual trainer comparisons, then independently audits them.
`logs/grounded_archive/audit.json` records source, bootstrap, binary, input and
output hashes. No public curriculum or WSL dataset files are changed.

The mechanism uses symbolic replay beside the existing byte learner, without
backprop. Learned control, open concepts, joint model updates, source reliability,
long-term storage and full public-curriculum performance remain unproven. The user
permits backprop for the complete-learner demonstration before a matched local-only
demonstration; this result does not constrain that choice.
