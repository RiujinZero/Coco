# Continuing-trainer repair follow-up — 2026-09-23

All three prespecified seeds now repair one available wrong primitive inside the
continuing trainer, retain earlier tasks and pass correction/withdrawal checks
within a declared 900-second native budget. This resolves the interrupted
single-skill integration experiment, not the broader complete learner claim.
The recommended profile remains `controller50k`.

## Completed matched campaign

`logs/repair_records_three_seed_v7/acceptance.json` applies the thresholds written
before this campaign. Both arms receive identical data and the same proposal
ceiling. All six arms complete and pass independent audits; baseline failures
are valid negative task results, not discarded runs.

| Repair seed | Native seconds | Visible control before → after | Retained old tasks | Primitive tests | Earlier direct outcomes | Lifecycle |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 619.37 | 0/32 → 32/32 | 128/128 | 324/324 | 120/120 | 8 stages |
| 1 | 79.69 | 0/32 → 32/32 | 128/128 | 324/324 | 120/120 | 8 stages |
| 2 | 752.81 | 0/32 → 32/32 | 128/128 | 324/324 | 120/120 | 8 stages |

Every matched baseline remains at 0/32 after counterexamples and retains only
32/128 old control tasks. Matched standalone count changes from 19/108, 13/108
and 12/108 to 108/108 on each repaired seed. Aggregate repaired results are
96/96 control answers, 384/384 old tasks, 972/972 primitive checks and 360/360
original direct outcomes. Assistance is absent during adaptation and evaluation.

The earlier v3 set-normalized campaign passes repair seeds 0 and 1 but **times
out on seed 2 at 900 seconds**. Its partial outputs and failed acceptance record
remain in `logs/repair_integration_three_seed_v3`. The later v7 assessment cache
passes all three seeds within the same wall budget. Neither campaign is an
isolated wall-clock throughput benchmark.

[REPAIR_INTEGRATION_VALIDATION.json](REPAIR_INTEGRATION_VALIDATION.json) records
the compact results, binary fingerprints and hashes of the retained evidence.

## Completed pilots

All pilots use the preserved 120 direct lessons, 400-episode warm curriculum,
64 composed counterexamples and 32 unassisted continuation episodes. They import
no primitive checkpoint. Evaluation freezes learning.

| Pilot | Native time | Control after repair | Retained control | Primitive checks | Earlier direct outcomes |
|---|---:|---:|---:|---:|---:|
| Interleaved, seed 1, v1 | 239.79 s | 32/32 | 128/128 | 324/324 | Independently rechecked on identical v3 weights |
| Cached compilation, seed 1, v3 | 82.71 s | 32/32 | 128/128 | 324/324 | 120/120 |
| Repeated observation records, seed 1, v4 | 72.36 s | 32/32 | 128/128 | 324/324 | 120/120 |
| Repeated records with lifecycle, seed 1, v5 | 86.45 s | 32/32 | 128/128 | 324/324 | Same final primitive weights as v4 |
| Assessment cache and lifecycle, seed 1, v6 | 77.80 s | 32/32 | 128/128 | 324/324 | Same final primitive weights as v5 |

The v1 and v3 runs reproduce 467 compared native artifacts byte-for-byte,
including the complete control trace. Configuration and diagnostic timing files
are excluded from that parity comparison. Concurrent workloads differ; these
wall times are not an isolated throughput benchmark.

The v5 and v6 runs additionally match all 476 compared native artifacts, including
the full eight-stage lifecycle. The bounded assessment cache requires agreement
on every candidate decision visited in a previous deterministic assessment;
unvisited policy rules cannot alter that execution. Legacy logical proposal/work
counts remain unchanged. Separate diagnostic totals report avoided simulation.

The independent Python VM replays the exact same 108 count cases against the
before/after frozen neural policies: **13/108 → 108/108**. Follow, equality and
count retain all 40 earlier direct outcomes each.

The original set-normalized observation interface passes 32/32 duplicate-scene
control cases before repair and cannot demonstrate their visible repair. The
explicit records profile instead passes **0/32 before → 32/32 after**. This is
a declared interface experiment, not a retroactive claim about the original
canonical profile.

## Observation contract

`controller50k_records` and `controller50k_repair_records` sort observation
records but retain multiplicity at begin, accepted reads/actions and goal resume.
Primitive computation sees the repeated records, as it already does for raw
language teaching. Logical truth still concerns distinct tuples: receiving the
same edge twice does not create a second object. Relational planning and action
effects retain their existing set semantics.

The default profiles continue deduplicating public observations. Record semantics
are fixed before the first episode and included in the resolved configuration
and checkpoint fingerprint. The independent control auditor reconstructs this
boundary and checks record multiplicity in public observations.

## Implementation and limits

`controller50k_repair_interleaved` exposes the validated 1,000-proposal quantum.
It retains the 100,000-proposal ceiling per candidate slot and old-outcome
rehearsal. The records repair profile uses the same search schedule. Experimental
profiles cache frozen neural choices locally during macro compilation, keyed by
the complete public VM observation and legal actions. Rehearsal cannot leave
stale cache entries because each compilation gets a fresh cache.

`search_progress=1` emits flushed timing/counter rows to stderr at materialization,
macro compilation, proposal, slot-budget and rehearsal boundaries. It is an
observer and is excluded from learning configuration. The first pilot attributed
about 176 seconds and 95 million simulated instructions to repeated macro
compilation, versus about 2.4 seconds to materialization.

Repair acceptance and support replay now share the same 256-instruction direct
outcome limit as repair search/rehearsal. A 131-instruction follow fixture
reproduces rejection at the old 128 limit and acceptance at 256; wrong outcomes
remain rejected. The original direct-prior acquisition budget remains 128 and
the larger standalone test executor uses 1,024. Executions beyond the declared
limits are still unsupported, not evidence of general unbounded execution.

## Evidence

- `logs/repair_integration_pilot_v1`: completed original-stream pilot, 300-second budget.
- `logs/repair_integration_cache_parity_v3`: audited optimized replay and `native_parity.json`.
- `logs/repair_records_pilot_v4`: audited visible-control repair and `repair_outcomes_audit.json`.
- `logs/canonical_repair_integration_v1`: canonical replay/correction regression; eight artifacts match the archived handoff.
- `logs/repair_integration_three_seed_v3`: completed earlier comparison; repair seeds 0/1 pass and seed 2 times out.
- `logs/repair_records_lifecycle_v5`: completed correction/withdrawal/restoration validation.
- `logs/repair_records_three_seed_v7`: completed matched repeated-record baselines and repairs, including lifecycle checks, all seeds 0/1/2 pass within 900 seconds per native arm.

Each experiment retains its input files, source snapshot, binary fingerprint,
native trace, frozen weights and audit. Timeouts retain partial artifacts and
produce a failed audit entry. Historical binaries and failures remain untouched.

Several successive/simultaneous repaired skills, autonomous computation budgets,
delayed goal credit, representation growth and reader integration remain open.
The single cached repaired cue is still not a demonstrated multi-skill bank.

## Lifecycle and feature audit

The continuing repaired model passes all eight lifecycle stages. It starts with
718 verified language outcomes. A contradiction withholds all 719 affected
outcomes; correcting a concept withholds 102 dependent outcomes while retaining
616 unrelated outcomes and exact repaired weights. Full language withdrawal
restores the original direct-prior weights (the originally available count
primitive remains available). Restoring the evidence recovers all 718 outcomes
and the exact repaired weights. Coupled controller credit correction/restoration
also preserves the original controller weights after replay.

The final campaign-build canonical regression is `logs/canonical_repair_final_v7`:
201 training episodes, frozen evaluation, exact replay, coupled correction and
six rejection checks pass; eight artifacts match the archived handoff exactly.
Configuration, long-execution replay, observation-record and memoized discovery
contracts also pass, including ASAN/UBSAN checks.

The [32-feature audit](PUBLIC_CONTROL_FEATURE_AUDIT.md) records the source of
every controller input and the independent wrong-belief reconstruction. This
closes the bounded feature-provenance check, not the unfamiliar-task or autonomous
goal-management gates.

## Additional language boundary correction

A subsequent held-out language diagnostic found 13 seed-0 withholdings caused
by predicate names absent from teaching. The new experimental record interface
can infer a raw predicate's binary arity from tuples in the current public scene.
It does not create a receipt, infer an absent predicate's type, or bypass an
unresolved learned concept. Default profiles retain their earlier behavior.

Native and independent Python execution agree on all 840 archived held-out
language cases across the three frozen continuing models. Of 771 cases within
the 32-entity computation limit, **770 are correct and one is withheld** because
the queried relation is absent and untyped. The other 69 cases exceed the name
capacity and are separately recorded, not silently dropped. Each in-capacity
family exceeds 95% on every seed. These are frozen language/program checks, not
additional controller rollouts or evidence of learning a new grammar.

The three-seed training campaign used v7. The updated v8 adapter was checked on
all three archived frozen models and through one complete seed-1 continuation:
all 476 compared native artifacts, including lifecycle results, match v7 exactly.
The new typing option is included in config/checkpoint fingerprints. Evidence is
under `logs/frozen_grounded_language_v8` and `logs/repair_grounded_regression_v8`.

## Reproduce

Use fresh output roots and preserve the existing binaries:

```sh
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/laminar_train.cpp cpp/src/organism.cpp -o build-wsl/laminar_train_repair_rerun
python3 python/controlled_repair_experiment.py --root logs/repair_records_rerun --binary build-wsl/laminar_train_repair_rerun --profiles controller50k_records controller50k_repair_records --seeds 0 1 2 --workers 3 --seconds 900 --lifecycle
python3 python/summarize_repair_campaign.py logs/repair_records_rerun
```

This uses the locally retained warm curriculum and component-generated teaching
files. A fresh Git checkout needs those ignored data artifacts or their generators.
The current source includes the subsequent observed-predicate typing fix; use
the preserved v7 binary for exact reproduction of the original matched campaign.
