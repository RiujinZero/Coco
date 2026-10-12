# Learning report meanings while acting under partial observation

Coco's experimental reader can now acquire visibility-loss and outcome-report
meanings during a continuing, partially observed interaction. A previously
learned controller can use an action when every retained interpretation predicts
that it will establish the requested state. Predictions never fill the current
observation memory: the controller waits for an actual report before checking
completion.

This joins live report learning with partial-observation memory. Request meanings,
actuator effects, question meanings, primitive policies and the neural body are
pretrained and frozen in this experiment. It is not simultaneous acquisition of
all those components under partial observation.

## Mechanism and teaching boundary

`live_evidence_grounding.h` stages a raw report against the previous possible
knowledge states. It enumerates generic evidence operations and state updates,
with unknown relation and argument bindings. Candidate interpretations include
positive/negative observations, complete incoming/outgoing sets, visibility loss,
no change, and add/remove/replace effects. These are supplied memory operations;
no English report word selects one in the learner.

The supplied question-selection algorithm chooses the question whose worst-case
answer group is smallest. A verified external answer, including an explicit
unknown, eliminates inconsistent rule/history pairs. Different rules can produce
the same current knowledge and remain distinct hypotheses. Repeated uses of a
frame retain consistent choices across possible histories. Publication preserves
all surviving histories; it does not pick a convenient one. Multiple answers
about one scene count as one supporting scene. General frame admission requires
four distinct scenes; earlier verified instances can already update that episode.

The external environment maintains two separate records: physical state and
public evidence. Its answers concern public evidence. A hidden ownership or
location change produces only a visibility-loss report, never a target revealing
the hidden value. The learner does receive the complete menu of currently
executable actions. That supplied affordance menu encodes physical preconditions,
including the actual giver of an object; discovering preconditions or inferring
knowledge from available actions is outside this experiment.

The actuator predictor now applies the same generic update algebra to positive
facts, explicit negatives and coverage in every hypothetical world. This matters
for negative goals: adding a new holder alone does not establish that another
holder is absent; replacing the complete holder set can. Frozen control accepts
valid partial observations, while actuator-effect training still requires
complete agreeing before/after states. A report-model refinement during a pending
action must preserve its full historical knowledge snapshot and only narrow old
candidate meanings. Generated, stale and conflicting feedback cannot supply a
verified observation.

## Protocol and results

The prior comes from the bootstrap, incremental, dialogue and batch observation
experiments. The final request/action checkpoint from the preceding online
experiment supplies four request frames and four actuator identifiers across two
interfaces. That export stores support counts rather than original evidence IDs;
the loader restores placeholder IDs to represent those counts. These memories
remain frozen and receive no new teaching.

Each seed runs one world through 160 requests. Before every request a hidden
holder or location changes and the corresponding evidence is withdrawn. Successful
transfers and moves use two unfamiliar report forms. Every seventh attempted
action is refused. At request 80 the environment switches between two previously
learned actuator interfaces. The learner sees raw reports and can ask from a
supplied menu of 264 familiar questions about the eleven known entities.

| Measure | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Completed requests | 160 / 160 | 160 / 160 | 160 / 160 |
| Attempted actions, including 27 refusals | 187 | 187 | 187 |
| Actions issued while current goal truth was unknown | 93 | 93 | 93 |
| Reports processed | 347 | 347 | 347 |
| Verified teaching answers | 35 | 35 | 36 |
| Unknown teaching answers | 27 | 27 | 28 |
| Retention questions after reports | 91,608 / 91,608 | 91,608 / 91,608 | 91,608 / 91,608 |
| Frozen fresh-name questions | 192 / 192 | 192 / 192 | 192 / 192 |

Retention evaluates every offered question after every report through the actual
Organism byte/output interface. Per seed, 10,596 targets are definite and 81,012
are unknown. These are exhaustive repeated checks within one small domain, not
91,608 independent reasoning tasks. The full positive/negative/coverage state is
checked alongside the answers. The 48 fresh-name holdout worlds each test holder
loss, location loss, transfer recovery and location recovery with all models
frozen and no teaching answers. Half their targets are definite and half unknown.

Five report frames receive four supporting scenes each. The two visibility-loss
frames and the new transfer frame converge to one rule apiece. The movement frame
retains complete-row observation and row replacement: its tested preconditions
cannot distinguish those operations. The refusal frame retains the same pair
for the giver's ownership. These alternatives remain explicit in the model.
Relation channels are exchanged on seed 1.

## Failure found and validation

The older refusal frame had many retained effect interpretations that behaved
identically in fully observed worlds. Under a visibility gap some interpretations
left ownership unknown or changed an unrelated negative fact. The initial live
run therefore failed a retention question after a refused transfer.

The new learner allows an ambiguous old event frame to acquire an evidence-language
interpretation from consequence answers. Exact prior event meanings stay anchored.
An evidence-language frame takes precedence in partial-observation reading; the
old event model itself is unchanged. The refusal's surviving ambiguity is recorded
above rather than replaced with a hard-coded English rule. Its use as an observed
holder statement is a convention of this controlled environment, not a claim
about the meaning of every real-world refused offer.

`python/check_live_evidence.py` independently replays all physical changes,
affordance menus, refusals, goals and public observations. It reconstructs generic
rule/history branches, recomputes all offered question scores and the selected
question, verifies every answer and pruning step, checks distinct support counts,
and verifies final models, all retained states, all retention answers and the
frozen holdout. Its hypothetical scoring covers 626,472, 626,472 and 634,920
branch/question combinations respectively. These are independent symbolic
recomputations checked against native choices and scores, not a log of every
individual native hypothetical prediction. Source/input/binary/output hashes are
recorded in `logs/live_evidence/audit.json`.

Address/undefined-behavior sanitizer contracts cover generated and stale feedback,
conflicting IDs and answers, explicit unknown, support counts, search limits,
unresolved worlds, historical branch correlations, pending-action reconciliation,
partial-state action choice and the complete-state effect-training boundary.
The request, online, live-observation and batch observation-language suites were
rebuilt. Their sanitizer contracts, three-seed native runs and independent audits
all pass with the shared-code changes.

## Remaining scope and reproduction

This route uses symbolic hypothesis elimination over supplied operations, atop
previously trained local primitive policies, with no end-to-end backpropagation.
It requires known single-token entities, two relation channels, bounded argument
slots, supplied question/action menus, clean verified feedback and one pending
report at a time. It cannot expand an eliminated interpretation after a noisy
correction, replay dependent history, learn new relations or actuator preconditions,
or plan through a sequence of subgoals. Learning the exploration/search controller,
broader joint learning and standard-trainer integration remain open.

Run `bash scripts/live_evidence.sh` under WSL after preparing the prior grounding
corpora and the final online request/action checkpoints. It builds the native
probe and sanitizer contracts, runs all three seeds and invokes the independent
auditor. The source curriculum and its builders are unchanged. This experiment
does not expand the separately reported original Stage 5 coverage.

The subsequent [revision experiment](REVISION_GROUNDING_EXPERIMENT.md) adds a
bounded report/feedback journal that can recover eliminated meanings and replay
dependent current/historical knowledge after an identified correction. It uses
the same evidence operations and pretrained questions; correction references are
explicit IDs rather than learned temporal language.
