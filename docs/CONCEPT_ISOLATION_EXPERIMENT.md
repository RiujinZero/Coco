# Growing a concept without disabling established skills

The global semantic-readiness gate in `RelationalSession` prevented any language
answer, planning or computational-language rebuilding when even one introduced
concept was ambiguous or contradictory. This stopped established skills while
the learner was acquiring an unrelated meaning. A reproducer and the original
header/binary hashes are preserved in `logs/concept_isolation/before.json`.

The session now distinguishes rebuilt evidence from knowing every concept.
Language frames ground only in resolved definitions. A computational request
materializes only its named relations and requires hypothesis agreement on the
needed tuples. Unknown or contradictory relevant evidence still withholds;
unrelated uncertainty no longer disables known language, effects and procedures.
Accepted evidence still invalidates the model until explicit rebuild, and every
procedure is revalidated against current meanings and transitions.

`test_concept_isolation.cpp` starts with known language and an acquired physical
procedure, adds an ambiguous concept and then contradictory evidence, resolves
it through a contrast, and introduces a three-role relation with a different
surface argument order in the same session. It checks 64 fresh role/distractor
worlds, contrast withdrawal/restoration and uncertain computation inputs.
ASAN/UBSAN passes this contract and the existing relational-session,
computational-session and suspended-goal contracts; see
`logs/concept_isolation/contracts.json`. This is a necessary integration repair,
not evidence of unrestricted concept invention or neural concept learning.

## Canonical continuation experiment

`python/concept_isolation_experiment.py` uses the recommended 50,013-parameter
controller, original Agkern and original TD rule. Each seed starts with the
existing 400-episode operation-bootstrap stream. All additions occur after its
32 assisted episodes and 128-episode taper have ended:

1. Introduce one sparse positive example of a fresh binary concept; two distinct
   definitions remain possible. Continue 32 old-task training episodes and test
   128 fresh renamed old tasks while that ambiguity remains.
2. Add two negative contrasts that require a conjunction of two newly observed
   relations, plus verified transitions for two new actuator schemas. Neither
   the conjunction nor a primitive instruction sequence is supplied to the learner.
3. Train on 32 new physical goals interleaved with 32 earlier-skill goals, then
   test 128 old tasks and 64 fresh new-concept goals. The new goals reuse learned
   computational language with the newly induced relation and include incomplete
   observations, operation-price changes and failed physical actions.

The independent audit reconstructs public controller inputs, frozen neural
choices, world physics, outcomes and memory contents. It verifies the exact two
surviving definitions during ambiguity and the intended conjunction after the
contrasts; it does not accept arbitrary unresolved hypotheses as a passing model.
The standard 324 primitive checks per seed remain in place. All three runs are
complete and independently audited under `logs/concept_isolation_screen_v2`:

| Check after continued learning | Three-seed result |
|---|---:|
| Old tasks while new concept remains ambiguous | 384/384 |
| Old tasks after new-concept/action training | 384/384 |
| Fresh action goals using the new concept | 192/192 |
| Retained primitive cases | 972/972 |

Each session grows from two to three learned concept heads; the new head retains
two competing definitions before the contrasts and one semantically verified
conjunction afterwards. Each learns procedures using the new actuator schemas.
Controller weights change during the additional 96 episodes per seed, with zero
teacher hints. All saved controller/primitive weights and value tables in the
original warm prefix are byte-identical to the preceding warm baseline. The
original final, recovery and larger-goal endpoints also remain unchanged.
`aggregate.json`, produced by `python/summarize_concept_isolation.py`, records
these checks. Native runs take about 302–330 seconds concurrently.

The 201-episode canonical regression passes exact checkpoint replay, coupled
credit correction and negative checks. Its eight trace/config/weight artifacts
are byte-identical to the preceding default build; see
`logs/canonical_isolation_regression/default_parity.json`. The earlier unsuffixed
screen root contains a generator assertion before any model training, not a
failed learner run.

## Follow-up: uncertainty inside one language frame

The first fix left a narrower failure: one unresolved relation could discard
all lessons sharing a computational language pattern, including unrelated
lessons. The session now keeps each usable row, preserves unresolved raw rows
for later reconstruction, and associates composed receipts with the original
row IDs after filtering. Queries that depend on the unresolved relation remain
withheld.

The extended `test_concept_isolation.cpp` reproduces the old failure and passes
under ASAN/UBSAN after the fix, both with plain procedure memory and composed
learning. It checks independent true/false answers, dependent withholding and
restoration, and exact receipt identity. Before/after evidence is retained under
`logs/mixed_frame_isolation`. The default 201-episode regression retains eight
byte-identical artifacts in `logs/canonical_repair_regression/default_parity.json`.
The actual canonical composed lifecycle check under
`logs/mixed_frame_lifecycle_v2` failed after acquisition and contradiction
restoration. Changing a concept invalidated the cached primitive even though
independent raw examples remained. `failure.json` records the terminal failure;
the unit result above does not establish the full lifecycle.

The cache previously required every original acquisition witness to remain
unchanged. The fix now replays unchanged outcome witnesses and direct
priors and requires coverage of every accepted primitive state/action target
before reusing the cache. It also recompiles against all current answers. Full
withdrawal still removes derived skills. The optimized and ASAN/UBSAN
`test_composed_support` now passes after acquiring count from 16 final outcomes:
removing redundant witnesses preserves exact weights without another search;
removing all duplicate-sensitive witnesses prevents cache reuse; removing every
outcome removes the derived count skill and programs. Evidence is recorded in
`logs/composed_support/optimized.json` and `sanitized.json`. The same full
canonical lifecycle now passes under `logs/mixed_frame_lifecycle_support`:
changing the concept withholds its two dependent answers, preserves 66 unrelated
answers and exact primitive weights, and restores all 68 answers after correction.
Contradiction and full withdrawal remove the derived skill; restoration recovers
it. All eight lifecycle stages pass. The 201-episode default regression also
passes, with eight artifacts byte-identical to the preceding build in
`logs/canonical_support_regression/default_parity.json`.
The final combined build repeats the full lifecycle successfully in
`logs/mixed_frame_lifecycle_handoff`; all 21 native artifacts match this support
run exactly. Its canonical regression also preserves the eight default artifacts.

This test supplies a named concept interface and contrastive labels, and retains
the bounded conjunctive induction machinery. It tests growing and using a new
meaning while other components continue learning. It does not establish
autonomous concept invention, a growing neural operation bank, or the full four
capability gates. Those requirements remain open.

Reproduce with a separately named binary and fresh output root:

```sh
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/laminar_train.cpp cpp/src/organism.cpp -o build-wsl/laminar_train_isolation_rerun
python3 python/validate_concept_isolation.py
python3 python/concept_isolation_experiment.py --binary build-wsl/laminar_train_isolation_rerun --root logs/concept_isolation_rerun
python3 python/summarize_concept_isolation.py --root logs/concept_isolation_rerun
```
