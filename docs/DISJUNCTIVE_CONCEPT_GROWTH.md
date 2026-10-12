# Outcome-induced reusable rule unions

An opt-in `ConceptMemory` extension learns disjunctive relational definitions
from verified final true/false outcomes. A learned concept may contain multiple
alternative clauses; another concept can call that learned meaning. This extends
the previous positive-conjunction representation class. The logic interpreter,
candidate search and resource bounds are supplied. The learner does not invent
disjunction itself, autonomous head interfaces, recursion or a new learning rule.

The three-seed component campaign in `logs/disjunctive_concepts_v4/audit.json`
passes **384/384 new binary-union questions**, **384/384 retained binary-union
questions**, **768/768 later ternary-composition questions**, and **192/192
partial-world checks**. Each seed uses independently renamed predicate symbols,
base argument orders and ternary head roles. Independent canonical-database
containment establishes exact learned semantics, beyond sample accuracy.

## What changed

`Limits.clauses=1` keeps the original conjunction induction path. A larger
explicit bound enables generic rule unions. Candidate clauses come from
connected subgraphs of all positive witnesses with distinct query arguments,
including materialized tuples of previously unambiguous learned concepts.
Negative outcomes reject a candidate. A bounded exact set-cover search selects
the minimum-description union that covers every positive outcome, charging one
unit per local literal and per local clause. Equally short semantically distinct
solutions remain unresolved alternatives. No task-specific predicate, target
formula, decomposition label or operation sequence enters the learner.

Each hypothesis is a union of clauses; separate hypotheses are competing
interpretations. Confusing these two levels would turn legitimate OR into
uncertainty. Answers require agreement among interpretations. In a partial
scene, a proof establishes truth and absence of proof withholds. Complete scenes
permit negative answers. Corrections first invalidate inference, then rebuild
from versioned raw outcomes. Exhausted search discards partial results.

Reusing a concept expands its alternatives with fresh existential variables.
Products of alternatives form the dependent union. Relational containment
removes redundant branches; contained stronger branches add no possible answers.
The existing public first-clause fields remain compatible with the default
model. Consumers of enabled unions must use `local_clauses(h)` and
`expanded_clauses(h)` to include every branch.

## The withheld expressivity test

The evaluator chooses opaque names and role orders for these meanings:

```
link(x,y)  := (exists z: r(x,z) and s(z,y)) or t(x,y)
relay(x,y,z) := link(x,y) and link(y,z)
```

The binary meaning is impossible for **any single positive conjunction** over
the supplied base predicates, regardless of body width: one positive teaching
world has only `r/s` facts, another has only `t` facts, and the empty world is
negative. No nonempty conjunction can hold in both positive worlds; an empty
conjunction would incorrectly accept the negative world. The experiment also
runs the actual original learner with an eight-literal bound.

After binary acquisition, the learner receives only final outcomes for the
ternary meaning. Its accepted local program calls the acquired binary meaning
twice. Expansion has four disjuncts and up to four base literals per branch.
Teaching contrasts distinguish the specified middle role, joined existential
witnesses and missing required facts. Final queries include new entities,
duplicate/reordered facts and distractors; training and scored queries differ.

| Arm | Local body / union bounds | Reuse | Binary questions before + after, all seeds | Ternary questions, all seeds |
|---|---|---|---:|---:|
| Growing rule library | 2 literals / 2 clauses | Yes | 768/768 | 768/768 |
| Flat same-budget union | 2 literals / 2 clauses | No | 768/768 | 0/768; all withheld |
| Original wider conjunction | 8 literals / 1 clause | No | 0/768; all withheld | 0/768; all withheld |
| Wider flat union positive control | 4 literals / 4 clauses | No | 768/768 | 768/768 |

The last arm shows that reuse is useful under the declared local budget, not
more expressive than an unrestricted union language. This is a larger learned
representation class, not evidence that modular induction always wins on time
or sample efficiency.

## Checks and cost

The independent Python auditor reconstructs every native answer, checks local
macro expansion, compares complete formula semantics by containment in both
directions, and replays active outcome receipts. The primary union arm has 580
independently reconstructed predictions, 156 active-outcome checks and five
formula-expansion checks per seed. Contradictory old evidence withholds both the
binary meaning and its bounded dependent meaning; removing that contradiction
restores the exact expanded definitions. A wide flat union can independently
relearn the dependent meaning during the conflict, which is valid behavior.

Focused optimized and ASAN/UBSAN contracts also cover ambiguity between rule
unions, contrast withdrawal/restoration, support withdrawal, missing evidence,
resource exhaustion, union containment, dependent invalidation and fresh entity
bindings. The original concept-memory contracts pass. All **nine** archived
three-seed `reuse`, `flat` and `wide` default concept experiment outputs are
byte-identical, including candidate/work counts and model definitions; see
`logs/disjunctive_default_parity_v2/audit.json`.

The primary campaign declares two local literals per clause, two local clauses,
eight expanded clauses, ten variables per expanded branch, eight head roles,
32 entities, 256 observed/materialized atoms, 1,024 receipts and 128 schemas.
Each concept rebuild allows 200,000 candidate/subset-search nodes and 20,000,000
metered operations. The union arm's seed-0 final binary model consumes seven
candidates and 68 operations; its ternary model consumes 1,132 and 51,726.
Complete process plus audit time is about 0.025 seconds per primary seed on this
machine. These small sparse worlds are not a large-scale compute benchmark.

## Preserved failures and scope

The first curriculum (`logs/disjunctive_concepts_v1`) omitted contrasts that bind
the specified middle role, so the learner correctly retained alternatives and
the semantic audit rejected success. The second added those contrasts, but its
wider flat union control still admitted disconnected existential witnesses.
Adding systematic single-role detachments produced the final curriculum. Those
failed outputs remain under `logs/disjunctive_concepts_v1` and `v2`; no learner
mechanism was changed to select the expected answer after these failures.
`v3` uses the final curriculum; `v4` adds active-evidence and macro-expansion
audits, with unchanged native behavior.

The query name and head arity remain supplied. Search uses complete-world
teaching, acyclic positive rule unions, a finite witness-induced class and a
shortest-description bias. There is no invention without a queried label,
recursion, negation inside a definition, noisy-feedback model or learned search
scheduler. The component campaign does not establish the full representation
growth gate or the complete combined learner. The separate continuing-session
contract now passes 256 fresh complete/partial queries, both alternative action
branches, old language/procedure retention, contradiction, withdrawal and
restoration. An independent read of its actual saved model passes 512 old/new
language checks (`logs/disjunctive_session_v2/audit.json`). This validates the
symbolic session connection; learned controller training in a mixed continuation
still requires its own evidence. The experimental canonical profile
`controller50k_disjunction` enables two local clauses, three literals per clause
and 64 expanded clauses; it has not been promoted to the recommended profile.

## Reproduce

From the repository root inside WSL/Linux, use a fresh output directory:

```sh
g++ -std=c++20 -O2 -Icpp/include cpp/apps/disjunctive_concept_probe.cpp -o build-wsl/disjunctive_concept_probe_reproduce
python3 python/disjunctive_concept_experiment.py --binary build-wsl/disjunctive_concept_probe_reproduce --out logs/disjunctive_concepts_reproduce
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Icpp/include cpp/tests/test_disjunctive_concepts.cpp -o build-asan/test_disjunctive_concepts_reproduce
./build-asan/test_disjunctive_concepts_reproduce
```

Implementation: `cpp/include/laminar/concept_memory.h`. Native transport:
`cpp/apps/disjunctive_concept_probe.cpp`. Focused contracts:
`cpp/tests/test_disjunctive_concepts.cpp`. The evaluator-only target formulas and
independent semantic checks live in `python/disjunctive_concept_experiment.py`.
Source and executable hashes are in the aggregate audit. Logs/build products are
ignored local artifacts and must be copied separately when moving machines.
`DISJUNCTIVE_GROWTH_VALIDATION.json` retains the compact campaign, default-parity
and failed-curriculum record in the repository.
