# Learning reference phrases and resolving ambiguity

## Result

Reference phrases can acquire reusable meanings from observed event outcomes,
then transfer into questions and other event constructions without teaching those
complete uses. The optional native language interface now preserves an unresolved
utterance, emits a clarification question, accepts a named candidate, and resumes
the original operation and question.

Across three seeds, the initial learner passes **1,170/1,170 supported reference
answers**, with the exact resolved question and world state, and **450/450
clarification checks**. Ninety uses of an untaught phrase receive no answer from
this module. Twelve additional observations per seed teach that phrase and bring
the unchanged transfer tests to **1,260/1,260**. These numbers exclude 2,880 ordinary
setup/control answers, which also pass.

This is a restricted discourse-grounding mechanism. It does not demonstrate
general pronoun understanding or a learned conversational policy.

## Learned meaning versus supplied mechanisms

`cpp/include/laminar/reference_grounding.h` supplies a small set of generic
selectors over grounded contextual events:

- Select subjects, objects, or either role.
- Restrict to one relation channel, or consider all relations.
- Form the set of possible entities, ignoring duplicate mentions.

No English reference vocabulary is specified in this learner. Teaching receives
an utterance, prior contextual descriptions, and independently observed
before/after worlds. The existing event learner interprets the contextual
descriptions. For each possible phrase span, the reference learner tries replacing
it with the entity selected by each generic selector, then checks whether the
existing event interpretation predicts the observed outcome. Hypotheses surviving
multiple distinct observations become the phrase's candidate meanings. There is
no annotated reference span, intended entity, grammatical role or selector label.

The teaching environment uses “it” and “that item” for the contextual possession
object, “that owner” for its subject, and “there” for a contextual link's object.
These are meanings in this controlled environment, not a claim about all uses of
those English expressions. A fifth phrase, “that thing”, is taught later. Phrase
spans are at most four tokens; entities remain single-token names.

At inference, selectors return sets. One entity permits substitution into an
event or question. Multiple entities, missing contextual evidence, or contradicted
meanings leave the utterance unresolved. Candidate meanings that agree in one
context remain distinct; if a later context exposes their disagreement, the
learner requests clarification instead of silently choosing one.

The **clarification controller is supplied**, not learned: it renders a fixed
question listing possible entities, suspends the unresolved change, accepts a bare
named candidate from the user, and resumes. The choice binds that occurrence only;
it does not teach a universal alias. Invalid choices preserve the pending request.
Later factual clauses wait behind an unresolved clause and replay in order.

Context collection and turn boundaries are also supplied. Events already grounded
in the current user turn take priority; otherwise the most recent factual turn
provides the context. Questions and generated answers do not replace that context.
Episode reset clears
entities, facts, contextual events and pending requests, while retaining learned
language. Self-generated text cannot change these stores or provide teaching credit.

## Training and held-out evaluation

The probe retains the preceding experiments' primitive demonstrations, event
grounding curriculum and 216 atomic question/answer examples per seed. These are
substantial prior supervision, not learning from reference examples alone.

Each of four initial reference phrases receives twelve verified transitions per
seed: **48 observations**. Training reference phrases appear only in active event
sentences. Antecedent descriptions vary between active and passive wording and
include a distractor relation. The new alias receives **12 further observations**.
Reference-bearing questions and passive reference-bearing events are never taught.

Evaluation has 1,350 dialogues across three seeds, fresh entity names, shuffled
facts, paths initially up to 32 links and counts up to 24. Context-setting link
assertions can shorten those paths, so this is not a new long-path generalization
claim. The link and possession channel numbers are swapped in one seed without
informing either learner. Text alone enters the native Organism during evaluation;
state, candidate and answer labels are used only by the scorer.

| Reference-dependent group | Initial correct answers | After alias teaching |
|---|---:|---:|
| Object reference in active transfer | 90 / 90 | 90 / 90 |
| Object reference in passive transfer | 90 / 90 | 90 / 90 |
| Subject reference in a transfer | 90 / 90 | 90 / 90 |
| Location reference in a link change | 90 / 90 | 90 / 90 |
| Object reference in an identity question | 90 / 90 | 90 / 90 |
| Subject reference inside a composed question | 90 / 90 | 90 / 90 |
| Location reference inside a count question | 90 / 90 | 90 / 90 |
| Duplicate antecedent mentions | 90 / 90 | 90 / 90 |
| A newer factual turn replaces the old context | 90 / 90 | 90 / 90 |
| An earlier clause supplies an event's reference in the same turn | 90 / 90 | 90 / 90 |
| An earlier clause supplies a question's reference in the same turn | 90 / 90 | 90 / 90 |
| Answer after clarifying an ambiguous event | 90 / 90 | 90 / 90 |
| Answer after clarifying an ambiguous question | 90 / 90 | 90 / 90 |
| Initially untaught alias in a question | 0 / 90, withheld | 90 / 90 |

All correct entries have the exact current fact set and resolved question, as
well as the native emitted answer. The 450 clarification checks comprise 180
initial ambiguities, 180 invalid choices that must retain the request, and 90
missing-antecedent cases after episode reset. Every check verifies the exact
candidate set and emitted clarification text, unchanged state and absence of an
answer claim. Clarification output passes through the actual byte distribution.

Two ablations use the same existing event and question learners:

- **References disabled:** zero correct reference-dependent answers out of 1,260,
  although ordinary setup answers still pass. Some failures emit wrong answers,
  so coverage alone would overstate performance.
- **Role distinctions removed:** selectors include both subjects and objects.
  None of the 1,080 direct reference questions/events succeed. The 180 answers after
  an explicit user choice still succeed, because the clarification supplies the
  missing entity. Excessive clarification therefore cannot pass as understanding.

## Validation

`python/check_reference_grounding.py` implements a separate small interpreter for
the environment's sentence meanings, contextual selection, state updates and
questions. It reconstructs every dialogue from its text and verifies states,
bindings, answers, ambiguities and episode resets. It also checks training
outcomes, disjoint training/test names, exact corpus serialization, every native
score against per-turn traces, and all seeds and ablations. Artifact hashes and
aggregate results are saved in `logs/reference_grounding_audit.json`.

Native assertions freeze learned references, event meanings, grammar, procedures,
primitive weights and verified evidence during inference. Emitted replies and
additional fabricated self-generated role markers/events must preserve the facts,
antecedents and pending requests. No aligned sentence-role training is used.

Sanitizer contracts additionally cover subject versus object ambiguity, duplicate
mentions, weak evidence whose selectors later disagree, deferred factual clauses,
invalid choices, episode reset and refusal to treat a clarification answer as
verified question-training data. Clarification requests also have a separate
counter and cannot inflate the answered-question count. The event and
shared-question regressions were rebuilt and independently audited against the
updated interface: 2,160/2,160 supported stateful event answers and the earlier
2,340/2,340 initial and 2,640/2,640 adapted composition answers still pass.

## Limits and next work

The grounding relies on complete reliable symbolic observations, known entity
names, already grounded event meanings and temporally aligned teaching scenes.
Context follows supplied current-turn/previous-turn rules rather than a learned
model of attention, salience or discourse. Simple same-turn references now pass,
but a turn's events form a set of candidate mentions: this does not establish
fine-grained salience among multiple clauses. There is no gender/number agreement,
multiword named entity support, unrestricted description grammar or open-world
referent discovery.

Known ambiguous reference phrases now have a native clarification response.
Unknown phrases still lack such a learned interpretation and can leave legacy
fallback generation available. This is not global uncertainty calibration. No
body-training improvement, bAbI gain or general chat competence is claimed; the
experimental interface remains opt-in and production defaults are unchanged.

The next boundary is handling incomplete or conflicting evidence and learning
which question or observation would resolve it. That requires tests of evidence
reliability and information gathering, beyond this supplied clarification policy.

Follow-up: [ANSWER_GROUNDING_EXPERIMENT.md](ANSWER_GROUNDING_EXPERIMENT.md) learns
new event meanings from consequence answers, retains semantic uncertainty and
supports explicit correction/withdrawal of teaching evidence. It still assumes
fully described initial worlds and externally verified feedback; it does not
resolve the arbitrary partial-observation or information-gathering gaps.

Later, [historical-reference grounding](HISTORY_REFERENCE_GROUNDING_EXPERIMENT.md)
learns references to ordered, versioned reports from historical answers and uses
them to select corrections. This adds bounded activity/source/order selection and
stale clarification checks; it does not supply unrestricted discourse grammar.

## Reproduction

The previous primitive, atomic-question and event-sensor corpora must exist.

```sh
python python/reference_grounding_probe.py
bash scripts/reference_grounding.sh
python python/check_reference_grounding.py
```

On Windows use `wsl -d Ubuntu -- bash scripts/reference_grounding.sh`. Outputs and
per-turn traces are in `logs/reference_grounding_native`.
