# Learning observation language while acting — 2026-09-22

Coco's optional grounded reader can now learn unfamiliar environment-report
wording during the same continuing interaction that teaches request meanings and
actuator effects. It asks previously grounded consequence questions and uses
externally verified answers to eliminate incompatible report interpretations.
Its predicted action effects never serve as observation targets.

This extends the [online request/effect experiment](ONLINE_GROUNDING_EXPERIMENT.md).
It closes one integration gap in a small, fully observed domain. The initial
observation/question vocabulary, neural body and primitive policies remain
pretrained and frozen. Search and exploration are supplied algorithms.

The later [observation-language experiment](OBSERVATION_LANGUAGE_EXPERIMENT.md)
grounds visibility, complete inventories and explicit negatives in a separate
batch learner. Integrating that partial-observation learning with the live loop
below remains unfinished.

## Evidence and update boundary

`live_observation_grounding.h` stages a raw report against the known pre-action
state. It enumerates generic relation/argument bindings and ADD, REMOVE,
SET_SUBJECT and SET_OBJECT effects. These are the existing generic memory
operations; the learner receives no intended report rule or post-state.

The supplied question menu contains possession counts, destination counts and
location questions over known entities. The learner evaluates each question in
each candidate world using its learned procedures, then chooses the first
question maximizing candidate count minus the largest predicted-answer group.
Verified answers come from the environment. A generated answer is explicitly
rejected as evidence. Question search has an evaluation budget.

A report instance becomes usable only when every retained interpretation gives
the same complete resulting state. This can happen before the general frame has
enough support for ordinary parsing. Four distinct supporting scenes are required
for that admission; repeated answers about one scene do not create extra votes.
Once admitted and unambiguous, the frame can interpret subsequent reports without
more teaching answers. Its meaning must still pass frozen transfer tests.

Feedback is tied to the exact environment text, episode, request, observation
revision and prior model. Duplicate evidence is idempotent; conflicting reuse,
generated evidence and stale feedback are rejected. An accepted monotone
observation-model refinement can rebase the pending action's model version after
checking its historical state. Arbitrary model replacement remains invalid.
This is a serialized trusted-host interface, not a provenance mechanism inferred
from arbitrary messages or concurrent actors.

## Protocol and results

Each seed starts with empty request and actuator memories and the same pretrained
reader used by the preceding experiment. One world continues through 160
requests, introducing four request constructions and two successive opaque
actuator interfaces. Four unfamiliar report forms describe successful movement
and possession transfer. Refusals retain familiar wording; every seventh action
attempt is refused. Request satisfaction and execution acknowledgments remain
explicit supervision. Requests have a 32-attempt limit.

| Measure | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Requests completed | 160 / 160 | 160 / 160 | 160 / 160 |
| Attempted actions | 210 | 209 | 205 |
| Refused actions | 30 | 30 | 30 |
| Verified observation answers | 25 | 29 | 26 |
| New report frames acquired | 4 | 4 | 4 |
| Earlier QA retained | 320 / 320 | 320 / 320 | 320 / 320 |
| Frozen transfer cases | 96 / 96 | 96 / 96 | 96 / 96 |

All four report frames reduce to their intended single update rule, with four
supporting scenes each. Frozen evaluation uses fresh entity names and both old
and new actuator interfaces, with no observation teaching or other learning.
Removing just the four acquired report meanings leaves physical action success
intact but makes all 96 resulting states/completion judgments per seed unknown.
This separates choosing an effective action from understanding its reported
outcome. The pretrained vocabulary, procedures and neural weights remain intact.

## Independent audit and an exposed policy limitation

The Python auditor independently replays every physical transition, executable
menu, feedback bit and complete state. It enumerates report hypotheses, verifies
the exact candidates retained after every answer, and checks every checkpoint,
retention question, frozen case and ablation. It verifies all definite
counterfactual predictions against a separate symbolic interpreter, then
recomputes the actual question scores and choices from the prediction trace.

The first audit incorrectly assumed every learned follow policy implemented a
total symbolic interpreter. On seeds 1 and 2 it sometimes fails to return when a
candidate world removes all outgoing edges at the starting entity. The ideal
follow interpretation returns that entity. This is a real coverage limitation
of the pretrained policy; no additional training or interpreter replacement was
used to hide it.

The full prediction trace contains 9,230, 14,404 and 9,464 candidate/question
evaluations. Respectively 0, 4 and 4 withhold despite a defined symbolic result.
These are repeated evaluation occurrences, not independent tasks. All definite
predictions match the symbolic interpreter. There are zero question-choice
differences from the ideal interpreter in these runs, and candidate pruning
matches its independently computed answer constraints exactly. The abstentions
remain explicit in the audit and are not counted as correct answers.

Native address/undefined-behavior sanitizer contracts pass for instance evidence,
support counts, feedback provenance, incomplete observations, conflict handling,
question budgets and pending action/model reconciliation. The preceding online
and answer-grounding suites were rebuilt; their sanitizer contracts, three-seed
native runs and independent audits also pass. Results are recorded in the
validation logs.

## Remaining scope

The experiment requires an agreeing complete pre-state, known single-token
entities, two relation channels, bounded argument slots and one unresolved report
at a time. It does not learn question language, neural policies, representations,
visibility language, failure wording, actuator preconditions or the exploration
policy during interaction. Noisy corrections that expand an old hypothesis set
would require replaying dependent history; this API admits only refinements.
Long plans, event-history goals and standard-trainer integration remain open.

This path uses symbolic constraint elimination and previously trained local
plasticity, without end-to-end backpropagation. Its small deterministic tests do
not establish general language understanding or a complete continually learning
model. The zero-edge finding specifically cautions against equating available
memory primitives with universally reliable learned use of them.

Run `bash scripts/live_observation_grounding.sh` under WSL with the existing
primitive demonstrations and bootstrap/incremental/dialogue training files.
It builds the native probe and sanitizer test, runs three seeds and invokes
`python/check_live_observation_grounding.py`. Detailed observations, questions,
predictions, model checkpoints, states and source hashes are recorded under
`logs/live_observation_grounding/`.

The subsequent [live evidence experiment](LIVE_EVIDENCE_GROUNDING_EXPERIMENT.md)
extends report learning to partial observations and visibility loss while a
pretrained actuator operates. It preserves alternative knowledge histories and
can further ground an older ambiguous refusal frame. Its request/effect memories
remain frozen; the joint effect learning above still requires complete states.
