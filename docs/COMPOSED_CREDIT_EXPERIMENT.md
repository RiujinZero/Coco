# Missing primitive acquisition from composed outcomes

The new experiment establishes a specific cross-level learning path: after
learning following and typed equality from direct final answers, Coco discovers
an unavailable counting skill from whole composed questions and their final
answers. No direct counting lessons, correct micro actions or macro programs are
provided. Self-generated rehearsal transfers the discovered behavior into the
actual shared neural skill policy while retaining the earlier skills.

Across three seeds, the Boolean curriculum produces 1,680/1,680 correct composed
tests, 324/324 standalone counting tests and 648/648 retained follow/equality
tests. This closes the specific missing-count acquisition gap; it does not close
general primitive repair, representation growth or full continuing integration.

## What the learner receives

Each run first learns following and equality from 40 direct final-answer lessons
per skill. Counting remains unavailable. Composed teaching supplies complete
fact sets, typed bindings, two relation arguments, an opaque question cue and a
verified final answer. The question cue groups examples; it does not encode a
program or identify the missing primitive.

The primary curriculum has 64 Boolean lessons across two forms:

- Compare the distinct counts attached to two entities.
- Follow a relation, then compare the endpoint's distinct count to a numeric
  quantity supplied in the question.

That input quantity anchors the integer meaning of counting. It is an ordinary
question argument, not a supplied intermediate result. Count equality alone
would permit arbitrary injective encodings of counts. The two forms supply
complementary constraints, but this experiment does not establish that this is
the best curriculum.

Teaching uses counts 0–3 and paths of length 1–2. Worlds contain duplicate facts,
distractors and renamed entities/relations. Counts at roots are independently
randomized so a root count cannot substitute for an endpoint count.

The alternative numeric curriculum supplies 32 final integer answers to
follow-then-count questions. It receives the same direct prior-skill lessons.
The two curricula differ in example count, question forms and synthesis work;
their results are not a matched comparison of curriculum quality.

## How acquisition works

`ComposedSkillDiscovery` searches currently unavailable slots in the existing
three-skill bank. For each candidate micro-policy, generic `ProcedureMemory`
search attempts programs for all current question groups. Fitness is whole
question answer correctness; no intermediate correctness or external action
labels enter this search. Partial candidate scores guide exploration, but only
a candidate that fits every teaching answer is accepted.

The search uses a population of 128, a restart interval of 20,000 proposals and
a fixed 100,000-proposal budget per full run. Macro synthesis is bounded at eight
instructions and 1,000 states per group during discovery. Successful candidates
also undergo execution-cost refinement. Search work is counted separately from
the candidate program's inference work. The prior neural weights and live
knowledge remain unchanged during candidate search.

After verification, the candidate policy is restricted to states visited by its
successful composed executions. Those executions generate neural instruction
training samples. Rehearsal combines them with self-generated traces supported
by the old direct lessons, then performs 128 shuffled local-plasticity passes
in the same native neural policy. External demonstrations are absent; internal
self-generated instruction targets are present and explicit.

This path uses symbolic search and local neural plasticity. The separate learned
operation controller uses backpropagation in existing experiments. These results
do not establish a complete model with either learning rule.

## Transfer and controls

After acquisition, neural primitive weights stay fixed. Generic program
induction receives 32 final-answer lessons for each additional question form:
follow-then-count with an integer answer, and compare counts after following
from both entities. In the numeric arm, all three other forms receive lessons.
These are newly learned compositions of fixed skills, not zero-shot procedures.

Each accepted model then receives 560 fresh tests: four question forms, counts
0, 1, 2, 3, 4, 8 and 12, and paths of length 4, 8 and 12. Separately, each primitive
receives 108 tests through size 64, with duplicates and distractors. Composed
calls allow 256 micro instructions; standalone primitive tests allow 1,024.
The 64-object result is standalone counting, not a 64-object composed benchmark.

| Boolean seed | First perfect proposal | Proposal micro instructions | Neural composed tests |
|---|---:|---:|---:|
| 0 | 46,765 | 3,684,296,743 | 560/560 |
| 1 | 16,019 | 3,171,315,841 | 560/560 |
| 2 | 92,099 | 2,815,243,850 | 560/560 |

All rows used the full 100,000-proposal budget, including refinement after the
first solution. The retained discovered count policies use 10, 12 and 9 observed
states respectively. This is expensive supplied search, not efficient online
credit assignment.

| Model, Boolean runs aggregated | Composed tests | Counting | Earlier follow/equality |
|---|---:|---:|---:|
| Discovered symbolic policy with prior neural skills | 1,680/1,680 | 324/324 | 648/648 |
| Shared neural policy with rehearsal | 1,680/1,680 | 324/324 | 648/648 |
| Shared neural policy without old-skill rehearsal | 280/1,680 | 324/324 | 216/648 |
| Prior model with count still unavailable | 0/1,680 | 0/324 | 648/648 |

The composed total contains 840 tests of the two acquisition forms and 840 tests
of the two additional forms after their procedure lessons. With rehearsal,
standalone counting remains exact through 64 objects, taking at most 346 micro
instructions. Without rehearsal, following fails all 324 tests; equality passes
216/324. Thus new skill acquisition alone is insufficient for retention.

| Numeric seed | Acquisition | First perfect proposal | Proposal micro instructions | Neural composed tests |
|---|---|---:|---:|---:|
| 0 | Failed: 22/32 teaching answers | — | 2,625,593,787 | No accepted model |
| 1 | 32/32 | 41,663 | 2,186,023,862 | 560/560 |
| 2 | 32/32 | 3,032 | 2,007,920,565 | 560/560 |

Both accepted numeric runs retain all tested primitive skills with rehearsal.
Seed 0 remains a failure and is included in the audit; held-out evaluation was
not run for an unaccepted model. Numeric success is 2/3, not 3/3.

## Continuing evidence and retained learning

`ComposedLearner` wraps the actual prior store and compiled neural
`ProcedureMemory` in versioned raw evidence. Dirty evidence withholds answers
until rebuilding. A cached derived model can return only after its supporting
receipts, all current composed answers and direct prior outcomes are revalidated.
Contradictions and withdrawn support cannot leave the derived counting skill
or its procedures active. Restored evidence can revalidate earlier learning.

A separate seed-1 lifecycle run acquires counting from the same 64 raw Boolean
lessons under a 20,000-proposal budget. It then learns the integer-answer
composition from 32 new whole-answer lessons with identical primitive weights.
Native contracts verify rejected untrusted/generated/incomplete lessons, stale
revision rejection, dirty-answer withholding, contradiction handling across
repeated rebuilds, withdrawal and restoration. The final evidence version is 370.

The acquired, extended, restored and final neural checkpoints are byte-identical.
The contradiction and full-withdrawal checkpoints match the direct-prior
checkpoint instead. Independent replay of the final checkpoint passes 108/108
tests of each skill, including counting through 64 objects. These are the same
seed-1 primitive fixtures, not an additional independent test population.

This establishes a typed-query acquisition/revision subloop. It is not yet wired
into `RelationalSession`'s learned language, concepts, physical actions and
operation-control stream. Existing combined-session regression still passes
1,008/1,008 questions, 120/120 physical goals and 648/648 retained primitive cases;
that regression does not establish integration of the new acquisition route.

## Evidence and reproduction

- `logs/composed_credit/audit.json`: all six final runs, including numeric failure.
- `logs/composed_credit/{boolean,numeric}/`: raw questions, answers, micro traces,
  discovered programs/policies, generated samples, prior evidence and weights.
- `logs/composed_credit/source_manifest.json`: archived build-source inventory.
- `logs/composed_lifecycle/audit.json`: raw-input identity, checkpoint rollback
  and retention, independent primitive execution and native neural replay.
- `logs/computational_session_composed/audit.json`: existing-session regression.

The Python checker independently computes whole-query answers and replays VM
semantics, legal actions, work counts, withholding and final outputs. It verifies
that generated count samples exactly match the successful acquisition
executions and that direct prior teaching contains only following/equality.
Saved neural weights reproduce all distinct logged neural choices through the
native parameter replay tool; the neural mathematics is not independently
reimplemented. Search proposal histories and every optimization update are not
independently reconstructed. Lifecycle journal transitions are checked by native
contracts, with independent checks on raw inputs and saved resulting checkpoints.

ASAN/UBSAN contracts pass for composed discovery, the evidence wrapper and the
existing computational session. These contracts do not execute the full
100,000-proposal sweep under sanitizers.

Run from the repository in WSL; use new output directories to preserve evidence:

```sh
bash scripts/composed_credit.sh logs/composed_credit_rerun 3 100000
bash scripts/composed_lifecycle.sh logs/composed_lifecycle_rerun logs/composed_credit_rerun/boolean/1
```

To audit existing outputs without repeating acquisition:

```sh
python3 python/check_composed_credit.py --root logs/composed_credit
python3 python/check_composed_lifecycle.py
```

Development runs are retained separately: `logs/composed_credit_pilot` failed
at 48/64 after 1,000 proposals; `logs/composed_credit_boolean` succeeded on the
earlier generator but lacks the final full evaluation; and
`logs/composed_credit_numeric_pilot` failed at 19/32 after 5,000 proposals. The
earlier Boolean generator coupled root and endpoint counts; the final sweep
randomizes roots independently. Do not pool these pilots with the final sweep.

## Remaining blockers

1. **General repair and credit:** acquire several missing skills, or diagnose and
   repair an already available but wrong skill, using downstream outcomes. The
   original search considers unavailable slots only. An opt-in
   [available-skill repair experiment](PRIMITIVE_REPAIR_EXPERIMENT.md) now passes
   three seeds in this VM at the larger budget. Demonstrate this through
   language and actions, under noisy evidence and practical compute budgets.
2. **Growing the representation:** learn new operation structures, types and
   interfaces beyond three supplied primitive slots, ten micro operations and
   the bounded typed macro VM. Existing reusable concept formation remains
   within a supplied positive-conjunction language.
3. **One continuing learner:** integrate this acquisition/revision route with
   language, concepts, effects and learned operation control, then demonstrate
   retention through a long mixed stream of new meanings and corrections.
4. **Broad control transfer:** reliably choose further reasoning, information
   gathering and action on unfamiliar demands. Passing the current bounded
   controller tests and fixing their runtime allowance reset does not establish
   general search control.

The next decisive integration test is a composed language/action task that fails
because counting is absent, followed only by final-outcome teaching, acquisition
through this route, successful new tasks, and a correction while earlier skills
remain usable. All four broad [capability gates](FINAL_CAPABILITIES_VALIDATION.md)
remain open beyond their demonstrated scopes.

Follow-up: this route is now connected to the continuing controlled session.
Two seeds pass the integrated final/recovery/larger-goal tests and all retained
primitive checks; another fails repeated acquisition budgets. See
[the integrated results, failures and canonical consolidation](COMPOSED_SESSION_EXPERIMENT.md).
The user has authorized [warm starts](WARM_START_CURRICULUM.md) for the remaining
goal/credit blockers, with assistance withdrawn before autonomous adaptation.
