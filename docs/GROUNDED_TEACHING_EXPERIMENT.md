# Verified language teaching in the running session — 2026-09-23

The optional grounded session now acquires new observation-language frames from
verified answers while the ordinary byte learner operates. Teaching spans episode
resets, retains earlier meanings and supports later corrections whose replacement
payloads use the newly acquired language. This closes the observation-teaching
interface gap in the preceding [session experiment](GROUNDED_SESSION_EXPERIMENT.md).
Question, reference, correction and primitive-skill meanings are still pretrained.

## Teaching interface

`GroundedSession::teach` accepts a report anchor, evidence ID, ordinary question
and verified answer, and a host-authenticated teacher identity. The anchor binds
the episode, report revision, history revision and observation model. Generated
feedback, unauthorized teachers and stale anchors are rejected. No dialogue
wording grants teacher authority. Repeated evidence IDs do not add support;
conflicting uses of the same ID are rejected. A fresh anchor is needed after
model/history changes, including for retries of an already accepted lesson.

The interface delegates consequence learning to `RevisableGrounding`. It can add,
revise or withdraw identified answers. Revising an answer requires its previous
revision and original teacher identity. Accepted evidence can make the journal
inconsistent; acceptance is not a promise that the answer agrees with other
evidence. Such a conflict withholds derived answers until repaired. Rebuilding
restores interpretations discarded by current-episode evidence and preserves
later events at their original chronological positions.

Unknown report frames use generic evidence-operation hypotheses. Verified answers
constrain those hypotheses; the teacher supplies no relation ID, effect kind,
argument slots, complete state or correct program. The operation vocabulary,
entity binding mechanism, known questions and search algorithm remain supplied.

The session records which frames it acquired, leaving bootstrap frames anchored.
Its acquired frames can continue narrowing across episodes. Report IDs remain
local to each episode, while teaching scene IDs stay unique across the session;
reusing report 1 in a new episode cannot silently collapse independent support.
Sub-threshold frames remain available to teaching replay so that evidence can
accumulate. Ordinary observation-language admission still requires four supported
scenes. Withdrawal of current-episode evidence removes only its own support.

If an episode ends with an unresolved contradiction, its observation model rolls
back to the episode's starting model before the next episode. This avoids carrying
the last apparently successful narrowing from a now-inconsistent journal.

## Standard trainer

`laminar_train` accepts `grounded_feedback=FILE` with `grounded_bootstrap=PATHS`.
The feedback file is separate from the raw training transcript and contains ten
tab-separated fields per line, without a header:

```text
byte_offset  episode  report  report_revision  evidence  previous_revision  operation  teacher  question  answer
```

The displayed spaces stand for tabs. `operation` is `add`, `revise` or `withdraw`;
use a nonempty placeholder answer for withdrawal. `previous_revision` is zero for
new evidence. Episodes are counted by consumed episode-reset markers, starting
at one for the provided corpus. Report IDs start at one within each episode.
The default allowed identity is `verified_teacher`; `grounded_teacher=ID` changes
it. Supplying this file declares its answers verified by the caller; the trainer
does not infer verification from their wording or independently verify truth.

Offsets count committed training-file bytes, excluding probes and generated text.
The trainer splits blocks at receipt offsets, checks episode/report revisions,
obtains a current anchor, and delivers the feedback there. Offsets must be positive
and ordered. A receipt beyond EOF is an error; an explicit `max_bytes` limit may
stop with later receipts pending. Probe copies cannot consume training receipts or
modify the original model/history. `grounded_model=1` prints the final observation
model for diagnostics.

Example after creating the experimental teaching bundle:

```bash
bash scripts/grounded_teaching.sh
```

The script builds release binaries and a sanitizer contract test, runs three
seeds and a no-feedback control, runs the actual trainer with and without
intermediate probes, and invokes `python/check_grounded_teaching.py`.

## Validation

Each seed uses one session, eight teaching episodes and eight fresh-name transfer
episodes. Four initially unknown constructions describe ownership change, movement,
lost location coverage and lost ownership coverage. Their order is interleaved;
people, rooms and destinations vary. Ordinary questions are mixed with reports.
Teaching uses 288 verified answers, with each frame supported by eight distinct
report scenes across episode resets. The 360 teaching replies include immediate
checks of the teaching questions; they are not held-out evidence.

The three release builds use byte chunks of 1, 7 and 113, and seed 1 swaps the
relation-channel IDs. Neural byte learning and structural allocation remain
enabled. The separate primitive-policy weights and prior symbolic models remain
unchanged. The independent Python audit replays raw reports/corrections, checks
teacher answers against public evidence, and checks exact replies, all current
and historical states, and acquired frame meanings/support after every turn.

| Measure | Per seed |
|---|---:|
| Mixed turns | 1,040 |
| Teaching replies | 360 / 360 |
| Transfer replies, without teaching feedback | 600 / 600 |
| Corrections among transfer replies | 24 / 24 |
| Correct unknown replies across both phases | 176 |
| Current/historical knowledge states | 3,600 |
| Acquired frame/support snapshots | 4,060 |

The no-feedback control retains the bootstrap and processes the same transcript.
It gets **184/600 transfer replies** correct, including cases where withholding
is correct; this is a control score, not a chance baseline. Without verified
answers, ordinary dialogue does not acquire the four definite frame meanings.

The actual trainer receives the same transcript and 288 receipts with 113-byte
blocks. Runs with and without intermediate probes end with identical counters,
observation models and grounded state. The final state matches independent replay;
the model matches the native session experiment. Every checkpoint answers its
separate previously learned QA correctly. Source, input, binary and output hashes
are recorded in `logs/grounded_teaching/audit.json`.

Sanitizer contracts cover source/generated rejection, duplicate/conflicting IDs,
stale anchors, evidence correction and withdrawal, cross-episode support,
four-scene admission, clone isolation and unresolved-conflict rollback. The earlier
session and revision experiments are rerun as regressions. These are controlled,
repeated scenarios, not unrestricted grammar or 1,040 independent language tasks.

## Remaining limits

The four operation meanings are acquired symbolically; byte learning runs beside
that process. This is not joint discovery of neural representations, question
meanings, primitive skills, reference meanings and correction constructions.
Physical action learning and learned information gathering remain separate.
All new entity names enter through familiar context language.

In this experiment's default mode, only the current episode retains editable teaching provenance. A consistent
episode's candidate sets/support become the next episode's prior; closed-episode
feedback cannot be retracted to reopen discarded interpretations in that mode.
A mistaken but internally consistent earlier lesson can therefore still poison
that consolidated prior. Persistent provenance across episodes is necessary
before claiming general continual correction or noisy-teacher recovery. The
follow-up [archive experiment](GROUNDED_ARCHIVE_EXPERIMENT.md) adds optional
bounded cross-episode replay and tests recovery from old mistakes. It does not
establish broader joint learning or unbounded persistence.

Memory and search remain bounded, checkpoints are not persistence/reload, and
the corpus does not establish long-run retention, open vocabulary, broader
temporal concepts or a complete public-curriculum result. No files under
`/home/user/curriculum_v2` are changed.
