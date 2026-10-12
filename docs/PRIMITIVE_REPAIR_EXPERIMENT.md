# Repairing an available primitive from composed outcomes

**Subsequent evidence:** [Continuing-trainer repair results](REPAIR_INTEGRATION_RESULTS.md)
records a completed three-seed integration and visible-control repair campaign,
including an eight-stage correction/withdrawal lifecycle on every seed. This page preserves the earlier
component experiment and interrupted canonical run.

The learner can now revise a previously learned primitive that fits its initial
lessons but fails later composed questions. This is an opt-in extension to
`ComposedSkillDiscovery`, wired to `profile=controller50k_repair`. The recommended
controller remains `controller50k`: 50,013 MLP parameters, original Agkern and
empirical TD. The completed evidence here concerns the primitive VM, not a
matched test of the entire language/action controller.

## Warm start and repair signal

Each seed learns follow, equality and counting from 40 verified final-answer
lessons per primitive. The count lessons contain no duplicate records. Their
answers are correct, but they do not distinguish distinct counting from counting
every row. No instruction traces or supplied count implementation are imported.
The resulting count policy is available and fits all 40 prior lessons, yet passes
only 15, 16 and 14 of 108 fresh duplicate-sensitive cases for seeds 0, 1 and 2.

Next, 64 composed final answers introduce the missing distinction. The learner
is not told which of its three available primitives is wrong. It searches each
slot in order, starting from that slot's induced policy. Candidate policies must
preserve the active direct final-answer lessons before paying for composed
program search. A successful candidate must fit both the 64 composed answers and
40 direct outcomes for its selected slot: 104/104.

Verified candidate executions produce the local neural rehearsal targets. For a
repaired slot, retention replays the old outcomes using the repaired policy;
replaying its obsolete instruction choices would reinstate the error. Earlier
outcomes are constraints, not immutable old instruction traces. This component
uses whole-policy search and Agkern local updates, not backpropagation. The
canonical controller still uses backpropagation when that profile is run.

## Historical interrupted canonical continuation

`python/controlled_repair_experiment.py` now compares `controller50k` and
`controller50k_repair` in the same canonical trainer. It starts from the exact
duplicate-free count lessons above, completes the 400-episode mixed warm
curriculum, adds 64 raw composed counterexamples, then continues for 32 episodes
without renewed hints. The first paired seed is under
`logs/controlled_repair_seed1`. The repair arm was stopped for handoff after
2,063.56 seconds elapsed / 2,059.78 seconds of native CPU. It completed the warm
curriculum and before-repair checkpoint, but its first post-counterexample
rebuild had not returned. `handoff_interruption.json` preserves the exact command,
checkpoint hash and stopping reason. This is incomplete cost evidence, not a
successful repair or proof that repair is impossible. The repair arm's native
failure entry is the deliberate interruption, not a newly diagnosed crash.

The baseline is complete and independently re-audited: it passes the warm
128/128 final tests, but after the new counterexamples passes 0/32 additional
count-dependent tests and retains 32/128 old control tasks. Its standalone count
policy remains at 13/108 before and after, with follow/equality at 216/216.
These are negative results, not successful repair.

An important limitation emerged: the controller canonicalizes observed worlds
as sets, while raw teaching preserves duplicates. Its 32 duplicate-scene tests
therefore pass before repair despite the primitive's independently measured
error. Do not describe that control checkpoint as failed counting. The current
comparison tests response to counterevidence and retention; it does not yet
demonstrate a visible wrong control answer being repaired on the same cases.
The initial audit omitted this normalization boundary and rejected the trace;
`reaudited_summary.json` preserves the corrected independent reconstruction
without replacing native results or the original audit report. The completed
baseline's corrected audit is authoritative; the interrupted repair arm cannot
receive a full-run audit.

## Completed comparisons

The same three fixed seeds were run at both budgets. All native results,
including unsuccessful searches, were independently audited.

| Proposals per slot | Successful repairs | Total proposals per seed | Held-out composed answers after successful repair |
|---|---:|---:|---:|
| 5,000 | 1/3 | 15,000 | Seed 1: 560/560 |
| 100,000 | 3/3 | 300,000 | Every seed: 560/560 |

At the smaller budget, seeds 0 and 2 stop at 94/104 and 92/104 respectively.
Their failures remain in `logs/primitive_repair`. The initial audit incorrectly
assumed a failed run must select the last candidate slot; the corrected audit
in `reaudited_summary.json` validates all three native outcomes without changing
their artifacts or counting failed acquisition as success.

The larger run is complete in `logs/primitive_repair_full/summary.json`:

| Seed | Initial duplicate cases | Repaired composed cases | Repaired count cases | Retained follow/equality | Simulated search instructions |
|---|---:|---:|---:|---:|---:|
| 0 | 15/108 | 560/560 | 108/108 | 216/216 | 2,600,810,453 |
| 1 | 16/108 | 560/560 | 108/108 | 216/216 | 2,413,344,498 |
| 2 | 14/108 | 560/560 | 108/108 | 216/216 | 2,288,142,380 |

Every seed repairs slot 2 after unsuccessfully considering slots 0 and 1.
The first full fits within slot 2 appear at proposals 21,754, 5 and 24,071;
these numbers exclude the preceding 200,000 proposals spent on other slots.
Search continues to its declared budget to improve execution cost. Concurrent
native wall times are approximately 475, 473 and 434 seconds; these are not
isolated throughput measurements.

In total, the neural learner passes 1,680/1,680 composed tests, 324/324 standalone
count tests and 648/648 retained follow/equality tests. Count tests extend to 64
objects and executions up to 345 instructions. Composed executions reach 165
instructions. Half the composed test questions use additional forms taught by
final answers after the primitive weights freeze. The unchanged initial learner
passes 0/1,680 composed tests under the same macro synthesis limits.

The symbolic candidate, neural rehearsal and no-old-skill-rehearsal arms all
pass here. This particular screen does not establish that old-skill rehearsal
is necessary, although earlier missing-skill acquisition screens did show its
benefit. The initial warm probe and post-repair primitive probe use different
held-out cases; do not treat their unmodified-policy counts as identical sets.

## Audit and boundaries

### Follow-up: share the repair budget across slots

`PolicySearch` can now pause and resume its population, random state and accepted
hypotheses. `ComposedSkillDiscovery.interleave_quantum=1000` uses this to give
each candidate slot 1,000 proposals in turn and stop at the first verified full
fit. It does not receive the identity of the faulty primitive. This is an
explicit component option; **the canonical repair profile still uses sequential
search**. The default quantum is zero and retains the old refinement behavior.

The final code passes all three seeds in `logs/primitive_repair_interleaved_final`
with independent VM and frozen-neural-choice audits:

| Seed | Total proposals across slots | Simulated search instructions | Repaired composed | Repaired count | Retained follow/equality |
|---|---:|---:|---:|---:|---:|
| 0 | 65,755 | 478,232,528 | 560/560 | 108/108 | 216/216 |
| 1 | 2,006 | 9,699,452 | 560/560 | 108/108 | 216/216 |
| 2 | 74,072 | 534,212,814 | 560/560 | 108/108 | 216/216 |

Aggregate proposal cost falls from 900,000 to 141,833 (84.2% fewer); simulated
search instructions fall from 7,302,297,331 to 1,022,144,794 (86.0% fewer).
Concurrent native times are 107.9, 27.6 and 131.2 seconds, including other native
work; these are not isolated speed measurements. The earlier stopping policy
can execute more slowly: seed 1 count reaches 495 instructions versus 345 after
the full sequential refinement budget. A future bounded refinement phase is a
hypothesis, not yet a validated recommendation.

The earlier full-budget no-rehearsal result does not generalize to this search
schedule: seed 0 now retains 0/108 follow cases without old-skill rehearsal,
versus 108/108 with it. Equality and repaired count still pass. Keep rehearsal.

ASAN/UBSAN checks verify exact search continuation across pauses and restarts,
immutable warm policies, and that a macro fit which never used a candidate
does not terminate other slots before an actual primitive is acquired. That
last edge case was found and fixed during validation. The final three-seed
rerun uses `composed_credit_probe_interleaved_v3`; the preceding v2 artifacts
remain preserved. With quantum zero, all 29 archived seed-1 native artifacts
match the old 5,000-proposal run byte-for-byte:
`logs/primitive_repair_sequential_parity/native_parity.json`.

Reproduce this optional search schedule by adding `--quantum 1000` to the
component command below and using a fresh output directory. It has not yet
resolved the interrupted full-trainer repair comparison.

The Python audit independently executes the VM, checks final gold answers and
macro programs, reconstructs the derived training states from successful
composed and prior executions, and verifies neural choices against saved native
weights. `attempts.tsv` records each slot's proposal and instruction costs;
the historical summary row describes only the selected slot. All source
snapshots, binary hashes and per-seed stdout/stderr are retained.

The first pilot under `logs/primitive_repair_pilot` predates a fix to count-size
trace metadata and is not pooled with either audited comparison. Sanitizer
contracts separately verify that warm proposals leave the original sealed
policy unchanged, that a repaired policy can preserve an old outcome through a
different instruction sequence, and that a candidate violating old outcomes is
rejected. Those constructed fixtures are software checks, not discovery scores.

The canonical profile is wired but still experimental. Repair through the
continuing raw-language/action stream, revision and withdrawal of its supporting
evidence, several simultaneous wrong skills, and noisy outcomes remain to be
tested. Search still uses three supplied primitive slots, ten microoperations
and bounded macro synthesis. Even the improved component schedule can consume
hundreds of millions of simulated instructions, and full-trainer repair cost
remains unresolved. These results do not close the four broad capability gates.

## Reproduce

Use a new binary and output root to preserve the archived evidence:

```sh
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/composed_credit_probe.cpp -o build-wsl/composed_credit_probe_repair_rerun
python3 python/primitive_repair_experiment.py --binary build-wsl/composed_credit_probe_repair_rerun --root logs/primitive_repair_rerun --proposals 100000
```

The experiment manager audits acquisition failures as valid negative results;
inspect each `audit.acquired` value rather than equating `audited` with success.
