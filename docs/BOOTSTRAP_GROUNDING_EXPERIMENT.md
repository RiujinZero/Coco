# Joint observation and question grounding from raw QA

The initial curriculum readiness check could not interpret new fact wording, so
its verified-answer API rejected examples before learning could begin. A new
native bootstrap learner closes that loop for the bAbI location pilot: it jointly
infers movement sentence meanings and the location question from story text and
final answers. It receives no fact alignments, entity spans, relation labels,
before/after state, supporting indices, or semantic query program.

This is a controlled first-domain bootstrap. It does not yet train the full
public curriculum or solve general language grounding.

## Teaching boundary

`python/bootstrap_grounding_probe.py` selects task 1 from the existing local bAbI
training source. The task number only selects the pilot; it is not written into
the native learner's input. The 900 records contain observation sentences, a
question, and a verified final answer. Source files and the user-built curriculum
are unchanged. The independent auditor checks that exported training records
match the source exactly, and verifies their answers separately.

Primitive policies still come from the earlier short action demonstrations
through Coco's native `LocalChoice` / `AgencyKernel`. Those policies are frozen
during language learning. Thus **language grounding uses answer supervision;
the primitive policies are not learned from answer-only reward here**.

## Learning mechanism

`cpp/include/laminar/bootstrap_grounding.h` implements the learner:

1. Variation at one position in otherwise identical training questions proposes
   entity vocabulary. Single-token answer referents extend it. Proposed question
   referents must occur in the story. Sentences are abstracted over this learned
   vocabulary. No name list, movement-verb list, or English question recipe is
   built into the learner.
2. Each sentence frame starts with candidate argument roles, relation channels,
   and the four existing generic update operations: add, remove, replace a
   subject's outgoing relation, or replace an object's incoming relation.
3. The question starts with candidate atomic programs over existing typed
   operations. Joint hypotheses are executed over the complete narrated history
   and their returned values compared with verified answers. Repeated uses of
   one frame must have the same meaning throughout that episode.
4. Candidate domains are refined using exhaustive supported assignments, with
   simpler histories first. A final pass must check all accepted records before
   publication. Budget-limited searches cannot manufacture evidence by truncation.

Constraint projection retains per-frame domains rather than every cross-frame
correlation. This overapproximates possible interpretations and can leave extra
ambiguity. The inference reader retains possible worlds and answers only when
they agree. A synthetic weak-evidence contract exhibits abstention; matched
counterexamples resolve that ambiguity without a guessed default.

Channel numbers are not identifiable from language alone. The two surviving
location models differ only by globally swapping channels 0 and 1. The learner
recognizes this equivalence and publishes a canonical representative. It refuses
publication into an already grounded model because its latent channels have no
anchors identifying them with that model's meanings.

The model learns all five observed movement frames as replacing the moving
entity's previous outgoing location. It learns the question program as a call
to the existing follow operation and return. All 900 records are checked without
skipped searches; convergence takes two passes in each seed.

## Frozen tests

Inference receives only story/question text through `Organism::attach_reasoner`;
actual replies come through native byte generation. The test file has no answers
or evaluator state. Primitive weights, event frames, and question procedures stay
fixed. Generated replies cannot alter the stored world.

| Test per seed | Cases | Correct answer and complete state |
|---|---:|---:|
| Original bAbI location test | 1,000 | 1,000 |
| Same stories with new entity names | 1,000 | 1,000 |
| Changed-location counterexamples | 120 | 120 |
| Persistence under unrelated moves | 120 | 120 |
| Two people sharing one location | 120 | 120 |
| Reversed event order | 120 | 120 |
| Untaught movement wording | 30 | All decline with `I do not know.` |

All three primitive-policy seeds pass: **2,480/2,480 supported cases per seed**,
or 7,440 executions over the same test instances, not 7,440 independent worlds.
The renamed and constructed cases test specific transfer properties rather than
adding independent benchmark datasets.

Another run replaces **every word**, including question words and entity names,
with deterministic opaque tokens before training and testing. It passes
**2,480/2,480** supported cases and declines all 30 untaught-verb cases. This is
a vocabulary control on the same tasks, not an independent benchmark. It checks
that an English-specific parser is not supplying the measured competence.

## Ablations

| Mode, per English seed | Correct answers / 2,480 | Correct complete states / 2,480 |
|---|---:|---:|
| Learned meanings | 2,480 | 2,480 |
| Replace learned effects with append-only addition | 1,062 | 330 |
| Remove learned observation frames | 0 | 0 |

The opaque-word control has the same ablation totals. Some append-only answers
are accidentally correct despite stale state, so the audit checks the whole
world as well as the answer.

## Verification

`python/check_bootstrap_grounding.py` independently replays location transitions,
checks all states and answers, verifies exact learned argument/effect bindings,
checks training export fidelity and the opaque transformation, and records
source, binary, corpus and result hashes in `logs/bootstrap_grounding/audit.json`.

The AddressSanitizer/UBSan contract test also exercises opaque active/passive
syntax, weak-evidence abstention, discriminating examples, conflicting answers,
self-generated and missing feedback, search-budget failure, unsupported query
patterns, and refusal to overwrite an already grounded model. The existing
grounded reader and its previous suites were not changed by this extension.

```sh
bash scripts/bootstrap_grounding.sh
```

On Windows run through WSL Ubuntu. The prior primitive demonstration files in
`logs/reasoning_composition_verified` and local `data/babi` sources are required.
The script generates inputs, compiles the native probe and sanitizer contract,
runs three English seeds plus the opaque control, then invokes the auditor.

## Remaining boundary

The learner requires one induced atomic question template, single-token referents,
two latent relation slots, binary updates, and initially empty closed-world
episode memory. Its lexical variability heuristic is not general entity recognition
or syntax induction. Different semantic words can also vary at one position;
broader lexical/structural ambiguity requires explicit models. Joint search is
supplied infrastructure, not a learned neural search policy.

Location stories do not exercise arbitrary spatial relations, ternary transfers,
attributes/sets, mixed question families, open-world observations, or natural
language corrections. Those remain requirements for the broader curriculum.
Learning a follow program also does not establish a universal one-hop location
operation when the destination itself has an outgoing edge.

Next is identifying meanings across multiple question templates and joining new
domains to existing grounded memory without merging unrelated channels.
Environment roles and visibility descriptions still need their ingestion/learning
path. Standard `laminar_train` and `laminar_babi` do not yet dispatch these
bootstrap/verified-feedback methods during ordinary byte-stream training.
