# Learning reusable relational concepts — 2026-09-23

The new `ConceptMemory` learns eight relational definitions from fact worlds and
verified true/false answers, introduces argument structures with up to four
roles, and reuses learned definitions in later concepts. Across three independently
renamed vocabularies and argument orders, it answers **4,800/4,800 fresh final
queries**. Independent relational-containment checks also establish that the
learned definitions express the intended concepts, rather than only agreeing on
the generated query sample.

This is a bounded symbolic concept learner, implemented separately from the
current two-relation language reader. It advances representation growth but does
not establish a complete learner or simultaneous language/control/action learning.
It uses no backprop; backprop remains permitted for completing the larger model.

## What is supplied, and what learns

Supplied machinery: conjunctive queries, shared variables, existential binding,
relational joins, shortest-consistent search, query containment, resource bounds,
a complete-world declaration for teaching, and verified binary feedback. The
learner receives structured fact tuples and a named query with its argument
tuple. There is no natural-language parser in this experiment. Predicate names
and argument orders are opaque and permuted across seeds.

Not supplied to the learner: the eight target definitions, concept-specific
templates, a decomposition into previously learned concepts, argument-role
meanings, or test answers. The generator and semantic oracle contain those
definitions for constructing and assessing the curriculum. The native executable
reads only `input.txt`; query records contain no answer field. `oracle.json`
is consumed by the separate Python checker.

Observed predicate arities register dynamically from evidence. A new concept's
head arity comes from its query; its body, shared/existential variables and choice
of previously acquired predicates are induced. A new four-argument query is
therefore supplied as an interface, but its relational meaning is learned.
This is stronger than allocating a new ID or alias for an existing fixed meaning;
it is not autonomous invention of an unrequested concept or interface.

The curriculum's human-readable concepts are:

| Concept | Meaning | Head roles | Base literals |
|---|---|---:|---:|
| Co-present | Two participants share a location | 2 | 2 |
| Matched | One holds an item the other needs | 2 | 2 |
| Eligible | Co-present and matched | 2 | 4 |
| Handoff | Co-present; first holds the specified item; second needs it | 3 | 4 |
| Relay | Eligible from first to second, and second to third | 3 | 8 |
| Route for item | Two successive handoffs involving the specified item | 4 | 8 |
| Trusted link | Eligible and second participant has a newly observed unary property | 2 | 5 |
| Authorized | Handoff plus a newly observed ternary permission fact | 3 | 5 |

These names and English descriptions are explanatory only. The native learner's
header contains no task-specific predicates or target definitions. Final memory
has six observed predicate schemas and eight learned definitions, compared with
four observed schemas and one definition after the first stage. Each learned
definition uses at most three local literals; expansion reaches eight base literals.

## Induction and evidence handling

For each concept, search uses the smallest positive teaching world with distinct
query arguments as its witness. Candidate bodies are generic subsets of that
world's observed facts and facts proved by earlier unambiguous learned concepts.
Every head role must occur, and every connected component must touch a head.
World entities become variables; there are no learned entity constants. At
inference, different variables may bind the same entity.

Candidates must explain all active examples. Search retains all semantically
distinct consistent candidates at the shortest successful local-body length.
It finishes the entire layer before accepting any result. Macro expansion uses
fresh existential variables. A generic head-preserving homomorphism check removes
redundant literals, then alpha-renaming and conjunction ordering canonicalize the
remaining query. This prevents a redundant existential witness from creating
false ambiguity and unnecessarily disabling reuse.

Answers require agreement among surviving hypotheses. No hypothesis, conflicting
evidence, unfinished rebuilding or resource exhaustion causes withholding. A proof
can establish a positive answer in a partial world; absence establishes a negative
answer only when the world is declared complete. These are conclusions relative
to the declared hypothesis class and shortest-description bias, not certainty
over every imaginable definition.

Raw teaching worlds and current receipt versions remain in memory. Accepted
edits invalidate inference until explicit rebuild. Rebuild proceeds from raw
evidence in concept-introduction order, allowing previously rejected definitions
to recover. Invalid or ambiguous earlier definitions cannot be used as macros.
A later concept can still be re-learned independently if another valid expression
fits its evidence and the search bounds. Generated, untrusted, stale, malformed,
wrong-arity and reserved-predicate evidence is rejected without mutation. The
host must authenticate the `verified_teacher` metadata; the string itself is not
a credibility estimator.

## Curriculum, controls and results

Each seed teaches 432 ordinary examples across eight stages: 48 mixed examples
per concept plus systematic role contrasts. Every concept has a sparse positive
witness. Other examples add distractors, duplicate/reordered facts, shared head
arguments, edge deletions, separated existential witnesses and alternative tuples.
This is an intentionally structured curriculum, not a sample of unrestricted chat.
New unary and ternary base schemas arrive during the last two stages.

| Arm | Local literal cap | Learned definitions reusable? | Fresh final answers, three seeds |
|---|---:|---|---:|
| Growing concept library | 3 | Yes | **4,800/4,800** |
| Flat restricted control | 3 | No | **1,200/4,800**; remaining 3,600 withheld |
| Flat wider positive control | 8 | No | **4,800/4,800** |

The restricted control learns the first two concepts and cannot express the
larger ones within its local bound. The wider control confirms that a larger flat
search also learns them. Thus reuse expands what a fixed local rule budget can
represent; this test does **not** establish that modular search is faster, more
sample-efficient, or more expressive than an unrestricted flat search.

Additional checks for the growing-library arm:

- **4,320/4,320** fresh interleaved queries during the eight learning stages,
  including earlier concepts after later acquisition. These are distinct from
  the 4,800 final queries.
- All **24 final definitions** (eight per seed) match the intended relational
  semantics under independent bidirectional containment checks.
- One contradictory receipt per seed makes the affected first concept withhold.
  Of 768 conflict-period queries, 96 withhold for that concept and 672 return
  correct answers. Later concepts are re-learned through valid alternate
  expressions; they are not assumed invalid merely because one earlier label is.
- All 48 queries made after accepted edits but before rebuild withhold. Correcting
  the identified receipt restores the original definitions and all 4,800 final
  answers. This recovery measurement reuses the final queries; it is not an
  additional independent test set.
- 960 partial-world checks return positive answers when proved and withhold on
  missing proofs. They reuse subsets of the final worlds.
- The Python checker independently replays every native query, validates learned
  definitions against active receipts, checks dependency expansion and semantic
  equivalence, and hashes code, executables, inputs and outputs. It performs
  3,146 active-receipt checks per seed in each successful arm.
- ASAN/UBSAN contracts cover unresolved alternatives, role order, shared witnesses,
  partial knowledge, clone isolation, provenance/version rejection, atomic
  rejection, exhaustion, new role structures, dependency invalidation and recovery.

## What the first run exposed

The initial exploratory curriculum scored 1,600/1,600 sampled final queries in
seed 0 while leaving incorrect shorter interpretations viable. For example,
"handoff this item" could mean "some exchange is possible and one participant
has a relation to this item." The initial test distribution shared that gap.
Its input, outputs, initial generator and audit remain under
`logs/concept_growth_exploratory/`.

The corrected curriculum adds an alternate tuple that satisfies the whole
relation while the queried tuple lacks a required fact. This distinguishes
"some item" from "this item." Training includes systematic versions and tests
use fresh randomized versions. It also revealed logically equivalent redundant
macro expansions; generic conjunctive-query minimization now recognizes them.
The final audit checks definitions as well as sampled accuracy, so the initial
shortcut cannot count as success. No user curriculum files were changed.

## Explicit bounds and remaining gap

Defaults: three local literals, ten expanded variables, eight query/base
arguments, 256 observed/materialized atoms per witness, 32 entities per scene,
1,024 receipts and 128 total schemas. Each concept rebuild allows 200,000 complete
candidate subsets and 20,000,000 metered search/join/canonicalization operations.
Inference has the same work cap. An exhausted search discards its partial result;
the variable and body bounds instead define the searched hypothesis class.

Only one sparse positive witness supplies the candidate structures. Coincident
roles in every positive witness, missing useful substructures or a larger concept
can therefore remain unlearnable under these bounds. Definitions are acyclic,
positive existential conjunctions. Negation within definitions, disjunction, recursive concepts,
numeric abstractions, invention without a queried label, noisy feedback and broad
language acquisition are untested. Teaching requires complete worlds; inference
supports only a global completeness flag, not per-relation coverage.

This module has no disk persistence or autonomous rebuild scheduling. The follow-up
[relational session](RELATIONAL_SESSION_EXPERIMENT.md) now connects it to new
variable-arity language grounding, the existing action-effect learner and
procedures acquired from its own executions. Shared versioned evidence supports
correction across these changing components. It remains separate from
`LanguageReasoner`, the neural primitive policy, learned computation controller
and `GroundedArchive`; the archive's fixed-prior guard remains intact. Integration
with neural learning and learned computation control is still required for the
complete continuing model.

## Reproduce

From the repository in WSL/Linux:

```bash
bash scripts/concept_growth.sh
```

Implementation: `cpp/include/laminar/concept_memory.h`.
Native probe: `cpp/apps/concept_growth_probe.cpp`.
Contracts: `cpp/tests/test_concept_memory.cpp`.
Evaluator-only curriculum and independent audit: `python/concept_growth_probe.py`
and `python/check_concept_growth.py`. Raw records, model definitions and the
hashed audit are under `logs/concept_growth/`.
