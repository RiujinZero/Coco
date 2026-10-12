# Curriculum rebuild handoff — 2026-09-22

**V2 follow-up:** the rebuilt WSL dataset fixes the checked current-possession
labels, Stage 3 overlap/paired-paraphrase problems, and environment role export.
See [CURRICULUM_V2_VALIDATION.md](CURRICULUM_V2_VALIDATION.md) for the verified
counts and the remaining temporal-history and matched-visibility-control issues.
The findings below document the first build.

These findings concern the current files described in `CURRICULUM_DATASET.md`.
The dataset and its builders were **not modified** during this review. TextWorld
was excluded because its build is still running. This is an initial readiness
audit, not a completed curriculum training experiment.

## 1. Fix the false/unknown labels in Stage 5

**File:** `python/curriculum/build_events.py:98–108`, `kind == "tf"`.

The generator chooses an `unknown` answer at random and then chooses a person
who has never held the object. That does not make possession unknown. These
stories show the initial holder and all subsequent transfers. In this world's
exclusive-possession model, a known different holder makes the answer false.
When everyone has held the object, the fallback can even choose its current
holder and label a true proposition unknown.

An independent evaluation-only interpreter checked all stories without a `He`
giver. Pronoun stories were deliberately excluded from these counts.

| Split | Total episodes | Explicit-name episodes audited | Contradictory labels |
|---|---:|---:|---:|
| Training | 78,332 | 64,952 | **5,423** |
| Held-out | 1,668 | 1,362 | **116** |

Of the training contradictions, 5,373 should be false and 50 should be true;
all 116 held-out contradictions should be false. Every contradiction detected
was labeled `unknown`. These are lower bounds for the complete files because
13,380 training and 306 held-out pronoun stories were not audited.

Concrete training example, zero-based episode 34: the story ends with
`Correction: Fred, not Sandra, has the cup.` The question is
`Does Sandra have the cup?`; the target is `unknown`, although that last sentence
explicitly establishes false. Full examples are saved in
`logs/curriculum_grounding/dataset_audit.json`.

**Rebuild:** calculate truth from the learner's observation history, independently
of the requested class balance. Generate actual missing observations for unknown
examples: an unseen initial holder, or an explicitly announced interval in which
an unobserved transfer may occur. Declare persistence and visibility rules.
Keep hidden truth separate from what the observer can establish. Do not make
unmentioned people an uncertainty heuristic. When a scope is fully observed,
communicate that coverage so absence can support false or zero.

**Acceptance:** independent truth checks pass for every supported story; include
positive, explicit-negative, different-known-holder, unseen-holder, visibility-loss,
and regained-observation cases. Paired cases should change the observation history
while holding the hidden world fixed, and produce the corresponding known/unknown
answers. Preserve true/false/unknown balance by sampling valid cases afterward.

## 2. Preserve environment provenance in grounded dialogue

**File:** `python/curriculum/build_grounded_dialogue.py:43–59`.

The builder internally distinguishes `USER`, `ASSISTANT`, and `ENVIRONMENT`, but
`render()` drops the environment role. A current held-out example becomes:

```text
USER: Please have Tomas give the book to Beth.
ASSISTANT: Okay, Tomas will give the book to Beth.
Tomas gives the book to Beth. Beth takes it.
ASSISTANT: Done. Beth now has the book.
```

After the first assistant marker, `LanguageReasoner::finish_segment()` ignores
text until a user marker or episode reset. The actual outcome therefore never
arrives as an external observation. Recovering it from sentence content would
also blur the distinction between the assistant's claim and verified feedback.

**Rebuild:** retain explicit roles in a canonical record, for example
`{"role":"environment","text":"Tomas gives ..."}` alongside user and assistant
records. If emitting byte streams, use an explicit `ENVIRONMENT:` marker as well.
Keep that distinction through train and evaluation exports. Never relabel an
assistant's predicted outcome as an observation.

**Learner counterpart:** the native reader needs an environment-role ingestion
path; it currently recognizes only `USER:` and `ASSISTANT:`. Adding a new marker
to the file alone does not complete that integration.

**Acceptance:** a failed action leaves the original holder intact even if the
assistant predicts success; a successful environment response updates it; replay
of the assistant's own assertion cannot update it. Score the prediction before
revealing its environment result.

## 3. Make Stage 3 world holdouts independent across the curriculum

**File:** `python/curriculum/build_paraphrase.py:37–58`.

The paraphrase generator chooses its own test worlds from `CLEVR_train_scenes`.
The other CLEVR stages train on those same worlds. Comparing complete rendered
scene hashes found **450/450 Stage 3 test worlds in Stage 2 training**, and
449/450 in Stage 4 training. Thus this set cannot measure unseen-world transfer
after the staged training run. Familiar-world phrase tests can be useful, but
must be labeled as such and paired with a separate unseen-world test.

**Rebuild:** assign world groups to train/dev/test once, before generating any
stage or paraphrase. Reuse that assignment across all ordinary CLEVR stages.
Use appropriate official held-out scenes, or exclude reserved groups from every
training export. Keep CoGenT's condition-transfer split explicit and separate.
Retain source/world IDs in scoring metadata and enforce zero train/test group
intersection across the actual training mix.

## 4. Put paraphrases on shared worlds

**File:** `python/curriculum/build_paraphrase.py:48–56`.

The scene index advances inside the template loop. Each template gets different
worlds. The current Stage 3 training file contains **8,550 distinct rendered
worlds and zero worlds with multiple paraphrases**. It therefore cannot provide
the advertised paired paraphrase-agreement evaluation on the same world/query.

**Rebuild:** select a world and underlying query, then render all training
paraphrases of that query on it. Group those variants under one teacher-side ID.
Withhold selected constructions independently of world splits. Preserve paired
held-out questions so correctness and agreement can be scored separately;
agreement alone can reward consistently wrong answers.

## 5. Tighten the early-stage lesson definitions

These are curriculum coverage issues, distinct from contradictory labels.

- **Stage 2 is not restricted to atomic operations.** In
  `build_clevr.py:28–34`, classification checks the final operation and number of
  `scene` roots, but not intermediate `same_*`/relational operations. The very
  first current training question is `How many other things are there of the
  same shape as the tiny cyan matte object?` A `same_shape` followed by `count`
  enters Stage 2 even though the guide places same-attribute composition in
  Stage 4. Classify the entire program, define the allowed atomic subset, and
  generate genuinely atomic examples where needed. Keep the program teacher-only.
- **The extra "immediate" event budget does not generate immediate lessons.**
  `build_events.py` adds `n_immediate` to the total and calls the same
  `make_episode()` with 3–10 events. Generate a distinct single-transition subset
  with matched success/failure and persistence counterexamples for bootstrapping.
- **Current corrections are confirmations.** `event_correction()` states the
  already-correct holder and denies a different one. This is useful negation
  practice, but does not test withdrawing an earlier erroneous observation.
  Add explicit erroneous-report/correction pairs with an identifiable report
  to supersede, and do not count the old confirming cases as belief revision.

## Learner work, separate from the dataset rebuild

A native readiness probe loaded the previously learned primitive, event, and
question models, checked a supported positive control, and then fed the original
held-out prompt text through `Organism::attach_reasoner`. No semantic solver,
query program, supporting indices, oracle metadata, or held-out answer was
provided to the learner.

**All 1,952 sampled held-out questions stopped at `incomplete_facts`; none received
a grounded answer.** The sample uses the first 32 records of each task family
across ten Stage 1–5 QA files. This measures interface coverage, not accuracy
after training on this curriculum. The reader has not learned these new fact
wordings. Its existing final-answer teaching API accepted **0/128** initial
Stage 1 training examples because their prior world could not be interpreted.
That API cannot bootstrap unknown observation meanings from those raw streams
by itself. Existing `laminar_train` / `laminar_babi` executables do not attach this
optional learner or invoke its verified-feedback APIs automatically.

Required integration work includes observation-language bootstrapping, explicit
verified-feedback dispatch, environment roles, and scope/visibility ingestion.
CLEVR also needs entity descriptions, attribute filters, sets, and more relation
coverage than the current two-channel probe interface. These gaps must be solved
on the learner side rather than by supplying answers or semantic programs through
a supposedly neutral dataset adapter. Boolean/unknown surface forms also differ
between sources and the current reader; record any scoring normalization openly.

The in-progress partial-observation memory passes its sanitizer contract test:
explicit false differs from unknown, incomplete counts abstain, observed empty
sets yield zero, visibility loss invalidates knowledge, branch-specific answers
require agreement, and generated text is not evidence. This is primitive-level
validation, not evidence that natural-language visibility descriptions have been
learned from the new curriculum.

The earlier event, reference, shared-question, and answer-grounding suites were
also rebuilt and independently audited after the memory changes; all four
regression suites passed.

## Reproduce the audit

**Subsequent learner progress:** a separate QA-only bootstrap now learns bAbI's
location wording from 900 raw training records, without supplied fact or question
meanings. It passes 2,480/2,480 supported cases per seed across three seeds,
including the complete 1,000-question source test. This addresses one pilot
domain from the initial coverage failure; it does not invalidate the dataset
issues above or establish broader curriculum coverage. See
`BOOTSTRAP_GROUNDING_EXPERIMENT.md`. Dataset builders remain unchanged.

Prerequisite: the existing primitive/event/atomic-question probe corpora in `logs/`.

```sh
python3 python/validate_curriculum_grounding.py
bash scripts/curriculum_grounding.sh
```

On Windows, run the shell script with WSL Ubuntu. The script also runs the Python
preparation step, so running both commands is optional. Inputs and results stay
under `logs/curriculum_grounding/`; gold is exported separately and opened only
by the scorer. The audit interpreter is evaluation-only and is never part of the
native learner. Logs contain source hashes, full contradiction examples, native
statuses, and the training-feedback acceptance result.

After new visibility or role formats are introduced, extend the audit to cover
their semantics and report unsupported cases explicitly; a skipped case is not
a passed case. COGS/SCAN diagnostics, broad chat, and the unfinished TextWorld
build were not included in this native QA check.
