# Proposed curriculum for Coco

Use **structured worlds with controlled natural language from the beginning**.
The world supplies exact state, actions and outcomes for verification. The learner
hears and produces short English descriptions and requests, with several surface
forms per meaning. Increase semantic and linguistic difficulty separately, then
combine them. This is a recommendation based on the current Coco experiments,
not a demonstrated globally optimal curriculum.

## Progression

| Stage | Experience | Evidence needed to advance |
|---|---|---|
| 1. Ground entities, relations and actions | Small worlds with objects, locations, possession and observable changes. Pair short descriptions with observations; give immediate verified feedback. | New entity names and layouts preserve identity and argument roles. Reversing who did what changes the interpretation correctly. |
| 2. Learn individual operations | Recall a fact, follow a relation, compare two values, enumerate distinct objects, count, and return a value. Use short demonstrations where exploration fails, then reduce assistance. | Reliable held-out execution with clean scratch state, typed returns, duplicates, distractors and longer instances. |
| 3. Learn shared language meanings | Express each operation and relation in several short forms. Place phrases in different contexts; use paired examples that change one semantic distinction. | A taught phrase works inside an unseen complete question. New vocabulary taught on individual tasks transfers to withheld compositions. |
| 4. Compose skills | Ask for endpoint counts, comparisons between computed results, nested references and small plans. Keep some combinations entirely out of training. | Correct inferred meanings and outcomes on unseen combinations, greater depths and counterfactual pairs; original skills remain intact. |
| 5. Track change and discourse | Move or transfer objects, correct earlier claims, refer back with names and pronouns, distinguish absent evidence from falsehood, and ask for clarification. | Old facts are superseded correctly; ambiguity and insufficient evidence produce appropriate uncertainty; self-generated text never becomes verified evidence. |
| 6. Grounded dialogue, then broad chat | Short task-oriented exchanges in known worlds, followed by varied human wording, explanations, instructions and longer conversations. Expand domains gradually. | Conversation remains consistent with current state and executed actions. It retains the earlier reasoning and grounding tests as language becomes less constrained. |

These stages overlap. Introduce simple linguistic variation during the first
stage, keep individual operations in the mix during composition, and retain
grounded tests during later chat training. Advance based on transfer and retention,
not a fixed number of tokens or epochs.

## Teaching loop

1. Sample a world and a task with a known, independently verifiable outcome.
2. Express it using a training phrasing; vary names, irrelevant facts and ordering.
3. Let Coco bind the entities, select or compose a procedure, act and answer.
4. Verify the actual outcome. Credit the responsible decisions against their own
   expected returns; use failed examples to invalidate unsupported shortcuts.
5. Replay older skills alongside new material. Target the next lesson at the
   measured failure: reading, argument binding, procedure selection or execution.

Choose examples that distinguish competing meanings. Repeatedly showing an object
acquired in an empty world cannot teach whether acquisition supersedes an old
owner. Include an old owner, other possessions and unrelated changes, then verify
the whole resulting state. The event-grounding comparison below supports this
choice at equal training exposure.

Structured state is an evaluator and an early grounding scaffold. Do not feed
evaluation-time answers, semantic trees or task labels into the learner. Gradually
replace direct fact annotations with grounding from observed events and verified
interaction. Teacher-provided intermediate demonstrations should be recorded as
additional supervision rather than confused with answer-only learning.

## Tests that matter

- **Meaning transfer:** hold out complete phrasings and combinations, not just
  random examples of the same templates.
- **Causal sensitivity:** minimally change a fact, argument attachment or action;
  ensure the answer changes when it should and remains stable when it should.
- **Language invariance:** change names, order, case and taught paraphrases while
  retaining the same meaning.
- **State and provenance:** test changed facts, episode boundaries, conflicting
  claims and generated text separately.
- **Retention and uncertainty:** keep old-task evaluations frozen; score both
  correctness and coverage so guessing and universal abstention cannot pass.

A practical first gate is near-perfect performance on deterministic small worlds
across multiple seeds and held-out combinations, with no meaningful regression on
earlier skills. Exact thresholds and data mixtures should be established through
curriculum ablations, rather than treated as proven constants.

## Where Coco is now

The current experiments demonstrate individual callable skills, learned procedures
and shared question phrases in scaffolded static worlds. The shared-meaning test
passes 2,340/2,340 new questions/compositions without training their complete forms;
new atomic vocabulary teaching adds 300/300 further composite successes. This
supports moving from individual meanings into compositions.

The next experiment replaced direct sentence-role annotations with before/after
symbolic observations. At the same 288 observations per seed, simple repeated
scenes ground only 2/12 event frames; contrasting scenes ground 12/12. These learned
meanings support 2,160/2,160 correct states and answers across multi-turn transfers,
retractions, corrections and controls. Another 120 untaught-event turns receive no
grounded proposal. See [EVENT_GROUNDING_EXPERIMENT.md](EVENT_GROUNDING_EXPERIMENT.md).

That experiment established basic event grounding and mutable state in the
controlled domain, with supplied entity alignment and structured observations.
The subsequent bootstrap, mixed-question and dialogue experiments acquire
observation meanings from raw stories and verified answers. Their individual
reports distinguish tested behavior from broader language claims.

The latest [online grounding experiment](ONLINE_GROUNDING_EXPERIMENT.md) starts
request and actuator memories empty, then learns four request forms and two
successive actuator interfaces while acting in one continuing world. Across three
seeds, information-seeking exploration completes 160/160 requests, retains
320/320 earlier QA answers, and passes 96/96 final frozen transfer cases per seed.
The observation learner and neural body remain frozen in this experiment.

The live result adds a curriculum requirement: include opportunities to choose
outcomes that distinguish competing meanings, even after ordinary task completion
is reliable. A correct transfer alone cannot distinguish the intended recipient
acquiring an object from the original holder merely losing it. The online learner
can generate that missing contrast itself when the teaching environment permits
exploration and returns explicit satisfaction feedback. This is a tested local
mechanism, not evidence of a globally optimal curriculum or exploration policy.

The subsequent [live observation experiment](LIVE_OBSERVATION_GROUNDING_EXPERIMENT.md)
learns four new report forms in that same loop from 25–29 verified consequence
answers per seed, retaining the 160-request, 320-QA and 96-case transfer results.
This supports adding opportunities for the learner to ask distinguishing
questions about unfamiliar reports. The audit also exposes missing zero-edge
coverage in some pretrained follow policies; primitive lessons should explicitly
cover empty, terminal and cyclic cases, with failure/uncertainty scored separately.
The subsequent [observation-language experiment](OBSERVATION_LANGUAGE_EXPERIMENT.md)
learns visibility, explicit negatives and complete inventories from QA, then
passes 169 original Stage 5 questions including gaps and unknown targets. Its
contrasts show why lessons must distinguish missing evidence from falsehood,
partial positive observations from complete sets, and direct reads from chain
endpoints. Attributed-claim lessons also need queries about the other scopes a
wrong interpretation could change. This is batch learning. The subsequent
[live evidence experiment](LIVE_EVIDENCE_GROUNDING_EXPERIMENT.md) acquires new
visibility and outcome-report forms during partial-observation interaction, with
question and request/action models pretrained and frozen. It completes 160/160
requests and passes 192 frozen gap/recovery questions per seed. Its old refusal
ambiguity shows why curricula need contrasts where missing coverage changes the
consequences of otherwise equivalent interpretations. Correction/history replay,
longer plans, broader joint learning and standard-trainer integration remain open.

The [revision experiment](REVISION_GROUNDING_EXPERIMENT.md) now recovers report
meanings and dependent knowledge after explicitly identified corrections inside
one episode. Curriculum examples should include a plausible wrong answer followed
by a distinguishing contradiction, a correction to the original record, and a
later valid event that must survive replay. Preserve independent evidence IDs and
test that repeated corrections do not inflate support. Unknown unique answers
also need membership contrasts: a spurious second value can mimic missing
evidence by making the unique read fail. This experiment does not itself ground
natural-language references to which earlier report should be corrected.

The [historical-reference experiment](HISTORY_REFERENCE_GROUNDING_EXPERIMENT.md)
adds that bounded step: historical answers teach six definite reference frames,
which select corrections in fresh histories without supplied target IDs. Its
first corpus passed the edit cases but confused activity filtering with temporal
rank because every history withdrew the first and last matching reports. Varying
withdrawal positions removes those shortcuts. Reference lessons should vary
record order, active status, source and entity independently, and include
ambiguous cases requiring clarification. The correction operation and replacement
payload remain explicit interface fields; broader correction dialogue is open.

The [correction-language experiment](CORRECTION_LANGUAGE_EXPERIMENT.md) then
teaches whole request constructions from verified post-correction answers.
Teaching must distinguish preserving a report, withdrawing it to expose older
evidence, and replacing it with a different claim. Include redundant reports
where those operations agree, then distinguish them in other histories; otherwise
apparent task success can leave the requested operation unresolved. Hold out
combinations of outer request construction, report reference and replacement
sentence, and check historical states when later events conceal an earlier edit.
This experiment passes those bounded tests with the prior component languages
frozen; simultaneous acquisition and general correction dialogue remain open.

The [grounded-session experiment](GROUNDED_SESSION_EXPERIMENT.md) adds an actual
stream integration check. Mix observations, questions, corrections, intervening
questions during clarification, visibility gaps and episode boundaries without
supplying turn-specific task labels. Verify the same behavior across byte-chunk
boundaries and ensure evaluation probes cannot replace an unfinished training
episode. Its optional trainer path currently ingests knowledge using pretrained
meanings; ordinary assistant text is not automatically verified teaching feedback.

The [verified teaching follow-up](GROUNDED_TEACHING_EXPERIMENT.md) adds observation
lessons to the same running session using a separate, byte-aligned teacher file.
Vary names and destinations across episodes, mix previously learned questions
with new reports, and check support counts when report IDs restart. Hold out
complete worlds and combinations of correction constructions with newly learned
report payloads. Compare with the identical transcript without verified feedback.
The current interface retains editable evidence only within the open episode;
cross-episode retraction tests remain necessary before claiming robust continual
learning from fallible teachers.

The experiments also do not show that answer-only reward can discover every
individual skill. Short verified demonstrations currently bootstrap the primitive
policies. Keep that supervision explicit, and separately test whether exploration
can replace it rather than assuming more chat data will do so.

## Research context

[BabyAI](https://arxiv.org/abs/1810.08272) provides a relevant precedent: increasing
levels of grounded instructions in a controlled subset of English, with an expert
teacher and measurable outcomes. It motivates the world-and-language approach; it
does not establish an optimal curriculum for Coco's architecture.

[SCAN / Lake and Baroni](https://arxiv.org/abs/1711.00350) demonstrates why systematic
held-out combinations matter: success on nearby combinations did not ensure that
the tested models could compose a newly learned primitive in new constructions.
That distinction motivates Coco's separate wording, composition and new-vocabulary
transfer gates.
