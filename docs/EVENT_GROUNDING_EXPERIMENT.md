# Grounding event language in observed state changes

## Finding

Controlled language paired with contrasting before/after worlds teaches a useful
distinction that repeated simple examples leave unresolved: whether a sentence
adds a fact, retracts one, or replaces an earlier relation. The optional native
reasoning interface can use those learned meanings across dialogue turns, then
answer composed questions through its existing learned skills.

Across three seeds, the contrast curriculum grounds all twelve sentence frames.
It gives the correct current state and native answer on **2,160/2,160 supported
question turns** in 960 dialogues. Another 120 turns contain an untaught event
phrase and receive no grounded proposal. This is a controlled symbolic grounding
experiment, not evidence of general conversational understanding or an optimal
curriculum for the entire organism.

## What is supplied and what is learned

`cpp/include/laminar/event_grounding.h` receives an utterance, a unique scene ID,
and independently supplied before/after sets of named relational facts. Perception
already identifies the entities, their names and relation channels. No sentence
receives an annotated subject, object, relation label or update-operation label.

The supplied hypothesis space has four generic state operations:

- Add a relational fact.
- Remove the exact subject/relation/object fact.
- Replace the facts for one subject and relation.
- Replace the facts for one object and relation.

For each literal sentence frame, the learner tries entity-role bindings, relation
channels and these operations. It retains only hypotheses that predict every
observed after-state. Interpretation requires one surviving hypothesis and at
least four distinct scenes. A contradictory verified transition revokes the old
interpretation. Self-generated text supplies neither observations nor credit.

For example, an empty world becoming `Lena holds key` cannot distinguish addition
from exclusive ownership transfer. A world where Sam already holds the key can:
if Sam's ownership disappears and Lena's appears, simple addition is wrong.
Other possessions and unrelated relations distinguish the remaining hypotheses.

The learner is symbolic memory-side code; the state operations and the search
space are engineered primitives. The mapping from a sentence to its operation,
relation and argument roles is learned. English phrase lists exist only in the
environment generator. Fact frames currently match literal wording, with entity
slots, rather than inducing a general grammar of event descriptions.

## Curriculum and controls

Each seed has twelve active/passive sentence forms covering links, possession,
transfers, retractions and an unrelated relation. Every form receives twelve
examples in each phase. Both compared curricula have **288 observations**:

| Curriculum | First 144 observations | Next 144 observations | Unambiguous frames per seed |
|---|---|---|---:|
| Simple repetition | Empty-to-one-fact additions; one-fact removals | More scenes of the same kinds | 2 / 12 |
| Contrast | The same initial scenes | Existing incoming/outgoing relations and distractors | 12 / 12 |

The two frames learned by repetition are exact removals. The positive sentences
remain ambiguous between the generic update operations. The comparison isolates
informational content at matched exposure; it does not compare against chat
training or establish a universally best teaching order.

The learner then receives 216 individual-operation question/answer examples per
seed and discovers eighteen question procedures. Their fact descriptions are read
through the event learner. The probe discards the corpus's fact annotations and
semantic labels: only sentence text and verified final answers reach this stage.
Primitive skill policies use the preceding experiment's demonstration curriculum.
No composite question is taught.

At evaluation, the native Organism reads raw text, retains state across user
turns, composes the question meaning, and emits the answer through its byte
distribution. Evaluation labels are available only to the scorer. All learned
event meanings, question meanings, procedures, primitive weights and verified
evidence remain frozen. Body training is disabled, and the small native harness
does not test a full 200,000-unit population. The language interface remains
opt-in; production defaults are unchanged.

In seed 1 the world swaps the link and possession channel numbers. Neither the
event learner nor question learner is told the permutation. Both recover the
corresponding meanings from observations and answers, and that seed also passes.

## Held-out dialogue results

The evaluation uses fresh names, shuffled descriptions, paths of 4/8/12/32 links
and counts of 4–24. Each seed has forty dialogues per group. Every dialogue asks
for the item count at a path endpoint before and after an event. Two-step cases
are scored after each intermediate event as well as the final event.

| Event group | Supported turns, three seeds | Correct state and answer |
|---|---:|---:|
| Link reassignment | 240 | 240 |
| Ownership transfer | 240 | 240 |
| Retraction | 240 | 240 |
| Duplicate assertion, repeated twice | 360 | 360 |
| Transfer followed by the old owner's release | 360 | 360 |
| Link change followed by correction | 360 | 360 |
| Addition in an unrelated relation | 240 | 240 |
| Initial questions before an unknown event | 120 | 120 |
| **Total** | **2,160** | **2,160** |

The 120 subsequent unknown-event turns receive no grounded proposal. This is
module-level withholding; the legacy generator remains available, so the result
does not establish global chat abstention or a clarification behavior.

Two ablations expose different failure modes:

- **Simple-scene meanings:** no question coverage, because the initial positive
  fact descriptions still have ambiguous update meanings. The same learned
  question procedures are used; grounding is the changed variable.
- **Append-only updates:** retaining learned argument roles and relations but
  forcing every update to add a fact gives 1,820/2,160 correct supported answers,
  yet only 1,320/2,160 correct states. All 840 turns in the groups that require
  changed-state handling have incorrect state. An answer can accidentally remain
  correct while an obsolete owner or link remains in memory. State audits are
  necessary alongside answer accuracy.

## Validation and limitations

`python/check_event_grounding.py` independently reconstructs every evaluation
transition, traverses the graph to verify answers, checks answer changes and
invariances, verifies equal curriculum exposure and disjoint sensor/test names,
and checks all seeds and ablations. It records source, binary, corpus and result
hashes in `logs/event_grounding_audit.json`.

Address/undefined-behavior sanitizer contracts cover ambiguous evidence, passive
argument roles, transfers, episode reset, unsupported wording, contradictory
experience and rejection of self-generated evidence. The earlier shared-question
experiment was rebuilt and its independent audit rerun successfully: its
2,340 initial and 2,640 adapted supported cases still pass.

The remaining assumptions are substantial: complete reliable symbolic perception,
known entity-name correspondence, aligned single-event observations, single-token
entities, literal event frames and a small supplied update vocabulary. The probe
does not demonstrate raw perception, uncertain or noisy observations, action
exploration, learning unknown operations, pronoun resolution, general negation,
clarification dialogue or broad chat. Contradictions cause withholding; recovery
from noisy evidence is not implemented.

The curriculum lesson is to choose examples that separate plausible meanings and
to verify resulting state, then retain those checks as language becomes richer.
The next grounding gate should address referential ambiguity and unknown events
in dialogue, with explicit uncertainty and verified teaching feedback.

Follow-up: [REFERENCE_GROUNDING_EXPERIMENT.md](REFERENCE_GROUNDING_EXPERIMENT.md)
tests learned contextual reference phrases, ambiguity, explicit clarification and
resumption, including references within a user turn. It supplies the clarification
controller and still assumes reliable observations; unknown events and evidence
reliability remain open.

## Reproduction

Prerequisites are the primitive demonstration corpus from the composition probe
and the atomic question corpus generated by `python/semantic_grounding_probe.py`.

```sh
python python/event_grounding_probe.py
bash scripts/event_grounding.sh
python python/check_event_grounding.py
```

On Windows run the shell script with `wsl -d Ubuntu -- bash scripts/event_grounding.sh`.
Native results and example traces are in `logs/event_grounding_native`.
