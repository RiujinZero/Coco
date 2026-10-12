# Shared meanings from learned procedures

Coco's experimental native reader now transfers phrase meanings to whole questions
that have never received training answers. This removes the whole-question lookup
boundary measured in `REASONING_LANGUAGE_EXPERIMENT.md` for the tested grammar.

## Learning mechanism

Training still uses grounded fact frames and the previously demonstrated learned
follow, count and equality skills. For each individual question form, twelve
verified final answers induce a procedure through the existing bounded search.
The new layer symbolically interprets that learned procedure to obtain its meaning.
No annotated semantic tree is passed to training.

`semantic_grounding.h` compares contiguous phrases across distinct question
contexts. A phrase receives the largest semantic subexpression shared by all its
observed contexts, with argument identities preserved. An unresolved tie is not
accepted. Whole observed question forms become reusable rules too. Their entity
slots can receive expressions whose return types satisfy the learned procedure's
primitive contracts.

At inference a bounded chart parser composes these rules. A unique, well-typed
interpretation is evaluated through the existing learned skill policies, using
ordinary call-local arguments and returned values. No procedure search or answer
feedback runs on a test question. All input tokens must be accounted for; unknown
words are not silently discarded. Ambiguous interpretations and budget exhaustion
withhold the module's proposal.

The code contains no English predicate lists or question-family recipes. Word
segmentation, phrase contiguity, shared-meaning intersection, typed function
application, equality symmetry and generic tree execution are supplied inductive
biases. This is a symbolic language learner on Coco's memory side, not a claim that
the byte-body network independently acquired the grammar.

## Training and evaluation

Across three seeds, initial training uses 216 answered dialogues per seed: eighteen
individual question forms, each with twelve different worlds. The only meanings
are identity, following one relation to its endpoint, counting held items, and
equality of two entities. None of the complete composite questions is trained.

Three wrappers and several synonymous phrases appear in overlapping subsets. One
wrapper/phrase combination is withheld for each phrase. The learner discovers
eighteen individual procedures and 42 grammar rules, including 24 phrase rules.

Training paths have 1–3 links; test paths have 4, 8, 12 or 32. Training target
counts are 0–3; test targets are 4–24. Entities are new at evaluation, fact order is
shuffled, and item assertions include duplicates and distractors.

| Held-out group | Exact meaning and native answer |
|---|---:|
| New combinations of wrappers and known phrases | 540 / 540 |
| Count at a path endpoint | 300 / 300 |
| Equality of item counts | 300 / 300 |
| Equality of endpoint item counts | 300 / 300 |
| Equality of endpoints | 300 / 300 |
| Asymmetric source/endpoint comparisons and attachment pairs | 600 / 600 |
| **Initial supported transfer total** | **2,340 / 2,340** |

Every complete test question form is absent from both initial and adaptation
training. The scorer compares the exact inferred expression as well as the final
answer, so a correct Boolean guess cannot pass the meaning check.

The 300 attachment pairs reuse the same world and change which entity receives
the endpoint operation. The generator forces opposite correct answers within
each pair. This checks argument attachment rather than keyword presence alone.

**Ablations:** whole-question lookup alone, and grammar rules restricted to whole
training questions without learned phrase rules, both have zero coverage on the
2,340 supported transfer cases. Fact extraction remains correct in both arms.

**New vocabulary:** two further phrase meanings are initially unknown. Teaching
them with 48 additional individual-question answers per seed yields four new
individual procedures and eight grammar rules. The unchanged, withheld composite
questions using these phrases then pass 300/300 for both meanings and native
answers. No complete composite question is taught. All previous tests still pass,
giving 2,640/2,640 supported answers after this intervention.

Ninety questions containing an unsupported extra modifier receive no grounded
proposal, before and after adaptation. This is module-level withholding; normal
chat fallbacks remain a separate arbitration issue. This probe scores grounded
proposals and does not count fallback guesses as reasoning success.

The largest observed parser search is 1,023 matching states, below the 50,000-state
budget. The parser also caps input length at 96 tokens, cell ambiguity at sixteen
meanings and expression size at 64 operations.

## Integrity and implementation

`cpp/apps/semantic_grounding_probe.cpp` feeds only text through the native Organism
interface and obtains answer bytes from its distribution. Primitive and body
training are frozen during evaluation. Assertions verify that skill weights,
procedures, question mappings, grammar rules and verified evidence do not change.
The small native configuration isolates this memory interface; it does not test
training a full 200,000-unit population.

`python/check_semantic_grounding.py` independently evaluates the expression labels
against each fact graph, verifies that training meanings are individual operations,
checks withheld forms and disjoint entities, verifies opposite attachment-pair
answers, and checks every seed and ablation. Result and source hashes are recorded.

Sanitizer-backed contracts cover phrase transfer, missing fragments, unsupported
words, invalid argument types, ambiguous rules, parser budgets, frozen inference
and conversion of learned procedures into meanings. The native byte/provenance
boundary contracts remain applicable.

## Remaining work

This is controlled-language transfer within a small typed semantic domain. The
fact reader still receives aligned subject/relation/object training examples.
Meanings of completely unknown words require teaching. The learner does not yet
establish robust grounding from environmental events, mutable facts, negation,
pronouns, multiword entities, unrestricted syntax, or free-form dialogue.

The next curriculum should connect observations and actions to meanings, retain
held-out compositional tests, then introduce state changes and grounded dialogue.
`COCO_CURRICULUM.md` records the proposed progression. The broader language-grounding
goal remains active; this result verifies shared phrase composition, not every
remaining language requirement.

Follow-up: [EVENT_GROUNDING_EXPERIMENT.md](EVENT_GROUNDING_EXPERIMENT.md) replaces
direct fact annotations with before/after symbolic observations and tests basic
mutable state in dialogue. It narrows those two gaps under controlled conditions;
the perception, syntax and discourse limitations above remain relevant. The
shared-question experiment and independent audit also pass with that extension.

## Reproduction

```sh
python python/semantic_grounding_probe.py
bash scripts/semantic_grounding.sh
python python/check_semantic_grounding.py
```

On Windows use `wsl -d Ubuntu -- bash scripts/semantic_grounding.sh`.
The native skill demonstration corpus from the preceding composition experiment
must already exist. Results and example interpretations are under
`logs/semantic_grounding_native`; the audit is `logs/semantic_grounding_audit.json`.
