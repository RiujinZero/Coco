# Grounding visibility and complete observations — 2026-09-22

Coco's experimental reader now learns language that distinguishes an observed
fact, an explicit negative, a complete inventory, an observation gap and an
attributed claim. It learns these distinctions from raw stories and verified
answers, including a verified `unknown` target. It can forget stale evidence and
recover knowledge from later reports without treating an unobserved relation as
false. This is batch language induction followed by frozen, stateful evaluation;
the partial-observation learner is not yet learning during the live action loop.

## Mechanism and supervision

`observation_language.h` supplies generic evidence operations: observe a positive
or negative atom, observe a complete outgoing or incoming set, forget an outgoing
or incoming scope, and leave knowledge unchanged. A scope can contain zero or
more listed entities. `FactKnowledge` now supports complete incoming observations
as well as outgoing ones. No English words select those operations in native
code. Candidate relation identities, argument roles and scopes are learned.

`observation_language_learner.h` extends the previously learned observation model.
It receives 7,808 raw story/question/final-answer lessons from 96 physical scenes.
Multiple questions about a scene share its evidence ID, so they do not inflate
independent support. It searches observation interpretations and atomic question
programs together, retaining assignments that agree with the answers. Unknown
targets match missing evidence or a nonunique result, not an arbitrary failed
primitive call. Unresolved question meanings, contradictions, insufficient scene
support, exhausted search and changed prior models prevent publication.

The resulting model has eleven observation frames and six question meanings.
Each frame has one intended rule, supported by eight or twelve distinct scenes.
Relation channels are exchanged on seed 1. Existing event meanings and primitive
weights are retained. Question meanings participating in this experiment are
reinduced; this explicitly includes the distinction between direct relation
retrieval and chain following. Shared phrase rules are rebuilt from the resulting
procedures, although new composed question constructions are not scored here.

The prior reader and local primitive policies are pretrained. The learner gets
no annotated facts, coverage flags, sentence roles, correct operation, query
program or hidden world state as a teaching input. The external generator and
independent auditor use structured knowledge only to create and verify targets.
Entity alignment uses the prior learned sentence frames and question variation;
the grammar remains bounded to single-token entities and small argument lists.
This path uses symbolic constraint elimination and local pretrained policies,
without end-to-end backpropagation.

## A needed read primitive

The existing follow operation traverses a relation until it reaches an endpoint.
Under partial observation it requires coverage at every step, including the
endpoint. That is different from reading a single known owner or location. For
example, knowing the complete holder list of an object identifies its holder
without observing every relation belonging to that holder.

Generic unique-read operations 16–19 now return the sole value in a completely
observed outgoing or incoming scope. Open scopes, empty scopes and non-singleton
scopes withhold a unique result. The caller distinguishes missing coverage from
a nonunique result. These are supplied symbolic memory operations; which relation
and operation answer each question are learned from examples. Chain following
remains available and is not replaced globally. A native procedure-search
contract discovers the new operation from final answers with second-link and
other-relation distractors.

## Evaluation

Every answer is generated through the actual Organism byte/output interface.
Repeated questions use the same running reader; later reports extend that episode
when their history extends the previous one. New histories reset episode state.
All meanings and weights stay frozen during evaluation.

| Measure, per seed | Result on each of three seeds |
|---|---:|
| Controlled and original-source questions | 19,273 / 19,273 |
| Definite targets | 13,817 / 13,817 |
| Unknown targets with valid, interpretable knowledge state | 5,456 / 5,456 |
| Original Stage 5 questions | 169 / 169 |
| Original Stage 5 histories containing visibility gaps | 42 / 42 |
| Original Stage 5 targets requiring unknown | 8 / 8 |

There are 19,104 controlled questions over 228 textual histories, plus the 169
original questions. Reused templates, many questions about the same world and
three executions of the same cases do not constitute independent language
problems. Controlled evaluation includes fresh names, explicit negatives, empty
and nonempty complete sets, two relation channels, later re-observation, attributed
claims, and withheld combinations of observation operations. Final answers and
the full positive/negative/coverage state are independently checked.

The source selection includes every current-possession question whose entire
story uses explicit declarations, successful transfers and observation gaps,
with query referents mentioned in the story. Of the 1,543 original held-out
questions, 243 temporal questions, 1,121 with other story language and ten with
unmentioned query referents remain outside this probe. The selected 169 contain
134 entity, twelve yes, fifteen no and eight unknown targets. Original wording is
preserved; true/false targets are normalized to yes/no. No source dataset or
builder was modified. Its SHA-256 prefix remains `f89b8e87975a`.

No normalized exact story/question pair overlaps this test set and the new
training lessons or any of the bootstrap, incremental and dialogue lessons used
to construct the prior. This checks exact overlap, not every possible semantic
overlap between templates.

## Failures found and controls

Initial claim lessons could not distinguish ignoring a report from making changes
in unqueried scopes. Contrasting queries, scope loss and known relations involving
the mentioned participants eliminate those alternatives. The final learned
“Someone said …” frame has no evidential effect. This teaches a controlled frame;
it does not create a general attribution or reliability ledger.

The independent audit also exposed two reader bugs. A variable slot could treat
the learned empty-list marker as a new entity, creating an unsupported alternative
world. Matching now uses learned literal vocabulary to prevent that invention.
If an entity with the same spelling was independently established, both readings
remain possible. Separately, a transfer's giver could be absent from the entity
vocabulary when only its receiver survived in the final fact. Learned event frames
now retain all mentioned participants as referents while preserving existing
subject/object ID order. An original Stage 5 question about a former holder
exposed this; a native contract now covers it.

Deleting the three visibility-loss meanings makes all 9,402 gap-containing cases
uninterpretable. It loses 6,142 definite answers; some withheld strings still match
unknown targets, so those matches are not evidence of grounded understanding.
Replacing unique reads with the existing chain-following calls reduces exact
answers to 16,685 / 19,273, with knowledge states unchanged. These controls separate
knowledge updates from the operation used to answer a question.

Address/undefined-behavior sanitizer contracts pass for complete columns, unique
reads, instruction discovery, learned gaps, verified unknown targets, generated
input rejection, conflicting feedback, repeated scene IDs, search limits, stale
publication, vocabulary collisions, former-holder references and alternative
worlds. The preceding live observation, incremental language and reference suites
were rebuilt and pass their native runs, sanitizer contracts and independent
audits. The existing partial-observation contracts also pass.

## Remaining scope and reproduction

This does not establish online acquisition of visibility language while acting,
general concept discovery, noisy correction/history replay, temporal reasoning,
unrestricted grammar, unmentioned query-name binding, long plans or standard
trainer integration. The live observation learner still requires a complete
pre-state. The generic unique-read primitive also does not fix the separately
recorded zero-edge coverage gap in the learned chain-following policy.

Run `bash scripts/observation_language.sh` under WSL after preparing the prior
grounding corpora and primitive demonstrations. It builds the native probe and
sanitizer contracts, runs three seeds and invokes the independent auditor.
Raw teaching inputs, target-free test inputs, exact models, native traces, source
selection counts and audit hashes are in `logs/observation_language/`.

The subsequent [live evidence experiment](LIVE_EVIDENCE_GROUNDING_EXPERIMENT.md)
acquires new visibility and outcome-report meanings during partial-observation
interaction. It uses the question meanings learned here and previously learned
request/action models, all frozen. Its results close that restricted integration
gap; they do not establish broader joint learning or correction/history replay.
