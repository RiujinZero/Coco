# Grounding event meanings from consequence answers

## Result

Five novel event sentence frames can be grounded from verified answers about
their consequences, without providing a post-event state or event-role/effect
labels to the learner. The native interface also retains alternative meanings,
answers only when their consequences agree, and distinguishes insufficient
evidence from conflicting feedback.

Across three seeds, the stronger teaching condition identifies the exact relation,
argument roles and update operation for **5/5 new frames per seed**. It produces
the correct state and answer on **1,440/1,440 held-out consequence questions**, plus
720 ordinary setup questions. Every one of the 720 three-turn dialogues passes.
Explicit correction of bad feedback restores the same results on replay.

With weaker teaching, none of the five frames is fully identified. Coco still
answers **840 consequence questions correctly**, and explicitly withholds a value
on **600 questions** where the retained meanings disagree or cannot support a
completed computation. An independent audit verifies every retained possible
world, not just the final answers.

This tests partial evidence about **event meaning**. Initial worlds are fully
described. It does not establish reasoning from arbitrary partially observed
worlds, autonomous noise detection, or a learned information-gathering policy.

## Supervision boundary

The experiment retains the earlier primitive demonstrations, event-grounding
curriculum for familiar descriptions, and 216 atomic question/answer examples per
seed. It is a warm start with known fact descriptions and question meanings; the
five new event frames have never been taught.

`answer_grounding.h` receives:

- The new event utterance and the prior state already read from ordinary text.
- A question meaning obtained from the existing learned question reader.
- A verified final answer, supplied as feedback.
- When present, a later familiar event interpreted by the existing event reader.
- Scene and observation identifiers for evidence bookkeeping.

It does **not** receive a correct post-event world, an event operation label,
subject/object labels, an oracle query program, or a solution trace. The teacher's
worlds and operation annotations live in separate audit JSON. The native training
file contains text, final answers and evidence IDs. Evaluation reads text and
emits bytes through the actual Organism distribution; its state labels are used
only by the scorer.

The learner enumerates the existing generic add, remove, replace-by-subject and
replace-by-object operations, relation channels and argument bindings. It executes
each candidate's consequences using the already learned reasoning skills and
retains those consistent with the verified answer. Later observed familiar events
can constrain an earlier meaning through their consequences. No English event
vocabulary is coded into the new learner.

The generic hypothesis space, evidence intersection, possible-world bookkeeping
and output templates are engineered mechanisms. The sentence-to-meaning mapping
is learned. This is a symbolic memory-side learner using Coco's native learned
skills, not evidence that a full population of bodies learned these mechanisms.

## Matched teaching conditions

Each seed has sixty underlying event scenes: twelve per novel sentence frame.
Each condition receives three answers per scene, for **180 feedback observations**.
Both conditions see the same prior worlds, unknown event utterances and familiar
follow-up actions. The questions differ.

The frames describe a transfer in active and passive forms, a retraction, a link
replacement and a non-exclusive addition to the counted relation. Their meanings
are defined by the synthetic environment; they are not a claim about every use
of the English verbs in the corpus.

| Condition | Questions | Exact event meanings identified |
|---|---|---:|
| Repeated questions | Ask repeatedly about the same actor, including after a familiar follow-up event | 0 / 5 per seed |
| Discriminating questions | Also ask about the previous actor and the event's object | 5 / 5 per seed |

The audit of an earlier two-question version exposed a shortcut: counts could
match while a hypothesis attached the new relation to the actor itself rather
than the named object. A familiar follow-up transfer distinguishes those cases;
asking about the event's object distinguishes a retraction from an alternative
self-relation. The final three-question design includes these interventions in
both conditions' observable event histories. It compares the information in the
questions at equal answer exposure, rather than assuming more repetitions fix
an unidentified binding.

Training paths have one to three links and source counts of three to five, with
other actor counts up to eight. Test initial paths have 4/8/12/32 links and source
counts of 4–20, with other actor counts up to 23. Test entities are disjoint from
training. One seed swaps the link and counted relation channel numbers without
telling either learner the permutation.

## Uncertainty and evidence revision

The optional `uncertain_events` mode retains up to 64 possible worlds. Each world
also retains which meaning it assigned to each sentence frame. Repeated uses of
one frame must use the same meaning within that world; they cannot independently
switch operations at each occurrence.

The ordinary fact projection contains only facts common to all retained worlds.
Questions execute against every world using the existing learned skills. A value
is emitted only if all results agree. Disagreement or an unfinished candidate
computation produces the supplied response “I do not know.” Exceeding the world
budget also withholds a value rather than selecting a branch. This is finite
hypothesis agreement, not a probabilistic confidence estimate.

Each teaching observation has a `(scene, observation)` identity. Replacing that
record is an explicit correction, and withdrawing it removes its constraint.
The learner recomputes the intersection from retained evidence, so a hypothesis
can return after erroneous feedback is corrected. Repeated IDs cannot inflate
the distinct-scene support threshold. Missing answers and self-generated text do
not add evidence.

The conflict test replaces one verified count with the deliberately impossible
value 999. The affected frame then has no consistent meaning. The native interface
reports conflicting event evidence on all 480 affected consequence questions;
the other 960 consequence questions still pass. Restoring the original observation
restores the complete 1,440/1,440 result. A separate added bad observation followed
by explicit withdrawal also restores the original evidence/model signatures.

These are externally identified corrections. The learner does not discover which
source was wrong, recover automatically from noise, or reject consistently false
feedback. Recovery is tested on new/replayed episodes. If the model changes during
a live uncertain episode, its state is marked stale and further answers are
withheld until reset/replay; past events are not silently reinterpreted in place.
The later [revision journal experiment](REVISION_GROUNDING_EXPERIMENT.md) supplies
explicit bounded replay within an ongoing partial-observation episode, while
retaining this requirement for identified evidence and consistent reconstruction.

## Held-out results and ablation

Evaluation has six groups: the five event forms plus two repeated uses of the
active transfer form. Each group has forty dialogues per seed. Every dialogue has
a setup question, a diagnostic consequence question, and another consequence
question designed to remain answerable under the weaker meaning hypotheses.

| Condition | Correct consequence answers | Wrong assertions | Explicit uncertainty | No proposal |
|---|---:|---:|---:|---:|
| Discriminating feedback | 1,440 | 0 | 0 | 0 |
| Repeated-question feedback | 840 | 0 | 600 | 0 |
| Pick the first weak hypothesis | 960 | 408 | 0 | 72 |
| Corrected evidence | 1,440 | 0 | 0 | 0 |

The initial 720 setup questions pass in every condition and are excluded from this
table. The true state remains among the weak condition's retained worlds on every
test turn. Forcing the first hypothesis yields some correct answers by accident,
but only 480/1,440 correct consequence states. A common-fact projection matching
the true state is also insufficient proof of certainty: other retained worlds can
still differ. The audit therefore compares the entire possible-world collection.

## Validation

`python/check_answer_grounding.py` independently interprets the environment's
text, enumerates candidate meanings from the answer constraints, and verifies the
exact native candidate sets. It checks the stronger model against the teacher's
true operations and roles. It reconstructs every possible-world branch, enforces
shared frame meanings across repeated events, and compares deterministic world
digests and counts against every native trace. It separately verifies answer
agreement, uncertainty, conflict, corrections, state, exposure, corpus
serialization and disjoint entities. Source, binary, corpus and result hashes are
stored in `logs/answer_grounding_audit.json`.

Native assertions freeze primitive weights, question procedures, grammar and
event/evidence signatures during evaluation. Emitting a reply cannot change the
worlds or facts. Sanitizer contracts cover missing/generated feedback, correction,
withdrawal, repeated evidence IDs, shared meanings across repeated events, refusal
to train a question from an uncertain fact projection, model changes during an
episode and branch-budget withholding.

The earlier event, reference and shared-question suites were rebuilt and
independently audited against the updated interface. Their 2,160 event answers,
1,260 reference answers, 450 clarification checks, and 2,340 initial/2,640 adapted
composition answers still pass. The new mode remains opt-in.

## Remaining boundary

This provides event learning from sparse consequence feedback after strong prior
grounding. It does not implement the public curriculum's proposed large-scale
dataset adapters or prove learning new relation families from raw observations.
The prior state is still treated as fully described, and event time/order and
feedback reliability are supplied. Visibility gaps, independently uncertain
observations, belief revision within an ongoing episode, failed intentions versus
executed actions, and choosing informative questions remain open. Unknown question
wording can still fall through to the legacy generator outside this mechanism's
supported meaning domain.

The ledger caches candidate constraints computed with the earlier question and
skill meanings held fixed. If those meanings change, the original teaching
records must be reprocessed; historical constraints are not automatically
reinterpreted under a changed reader or primitive policy.

## Reproduction

The preceding primitive, event and atomic-question corpora must exist.

```sh
python python/answer_grounding_probe.py
bash scripts/answer_grounding.sh
python python/check_answer_grounding.py
```

On Windows use `wsl -d Ubuntu -- bash scripts/answer_grounding.sh`. Native results
and per-turn world digests are in `logs/answer_grounding_native`.
