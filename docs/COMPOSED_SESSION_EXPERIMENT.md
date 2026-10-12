# Composed acquisition inside the continuing session

Status: corrected seeds 1 and 2 are complete and independently audited. Seed 0
has failed two complete acquisition attempts; its continuation was intentionally
stopped during a third attempt after the user requested capacity/warm-start tests.
The canonical
trainer's full seed-1 parity run passes. This is not a complete-learner claim.

`RelationalSession` now optionally routes its raw computational-language
evidence through `ComposedLearner`. A chosen controller `reason` operation can
therefore acquire a missing primitive as well as compile a language program.
The resulting shared neural skill policy is used for answering, hypothetical
action evaluation and observed goal satisfaction. Subsequent language, concept,
effect and terminal evidence updates revalidate the derived learning.

Typed quantity mentions extend the supplied language interface. A quantity such
as `3` becomes an integer query argument; it is not an intermediate answer or a
provided program. Parsing, controller feedback and execution-observation
corrections retain that type. The interface remains bounded controlled language,
not unrestricted natural language understanding.

## Curriculum and validation

Each seed receives 80 direct final-answer lessons for following/equality and 64
composed Boolean answers. It receives **no direct counting lessons**. The 64
raw lessons come from the earlier composed-credit experiment; no discovered
policy, program or instruction trace is imported. Additional computational
language, two relational concepts, three action effects and own-execution
evidence arrive during four stages of controller training.

The primary configuration uses empirical TD credit and neural backprop preference
fitting, 128 replay updates per episode, 64 consolidation passes every 200
episodes, and 0.4 exploration for the first 200 episodes then 0.15. The full
run uses 400 episodes per stage and a 100,000-proposal acquisition budget.
The reactive body executes one motor step at a time, up to 32 attempts.
The existing `learned_progress` configuration retains search allowance across
unchanged observations in frozen evaluation; training retains its prior setting.

The independent checker reconstructs controller observations, frozen neural
choices, world transitions, outcomes, language/concept/effect semantics and
evidence inventories. An additional saved-checkpoint probe exercises the actual
neural primitives on duplicates, distractors and sizes through 64. Native
lifecycle contracts exercise contradiction, concept correction, coupled control
credit revision, withdrawal and restoration after training. These lifecycle
tests do not contribute to controller benchmark scores.

## Failed pilot: canonicalization hid a missing ability

The seed-1 pilot used 50 controller episodes per stage and a 20,000-proposal
budget. It acquired counting after proposal 11,325, fitting all 113 then-current
composed answers. It passed 126/128 final controller cases, 24/26 recovery cases
and 6/12 larger goals. Those scores alone overstated primitive transfer.

The independent neural checkpoint retained following/equality at 216/216 but
passed only **13/108 standalone counting cases** with duplicate records. The
session canonicalized language teaching worlds before acquisition. The controller
also canonicalized observations, so an acquired record counter could succeed
inside that path without learning to ignore repetitions. The public session
answer interface can receive noncanonical scenes, making this a real mismatch.

`teach_language` now preserves the original teaching records and their order.
Inference paths that need canonical scenes still canonicalize locally. This
lets counting acquisition encounter repetition and learn its consequences from
whole answers. No counting implementation or correct instruction is supplied.
The three initial full runs were explicitly terminated after identifying this
failure; their incomplete outputs and source snapshots remain under
`logs/composed_control`, with the reason in `TERMINATED.json`.

The corrected build uses identical public curricula and private evaluation
worlds under `logs/composed_control_raw`. The old pilot remains under
`logs/composed_control_pilot`, including its failed primitive checkpoint audit.

The initial matched seed-1 control with composed acquisition disabled passes
4/128 final cases, 0/34 recovery cases and 0/12 larger goals. Its audit explicitly
requires withholding on count-dependent tasks. Its completed repeat with the
raw-record fix under `logs/composed_control_raw_noacquire` has byte-identical
public input, complete native trace and final controller weights. The comparison
is recorded in that run's `raw_record_comparison.json`.

## Corrected integration results

| Seed | Acquisition | Final control | Recovery | Larger goals | Primitive checkpoint |
|---|---|---|---|---|---|
| 1 | 113/113 current answers; first perfect proposal 95,663 | 128/128 | 34/34 | 12/12 | 324/324 |
| 2 | 113/113 current answers; first perfect proposal 64,156 | 128/128 | 28/28 | 12/12 | 324/324 |
| 0 | First attempt 104/112; second 105/113; neither acquired within 100,000 proposals per attempt; third incomplete when stopped | Incomplete | Incomplete | Incomplete | Unavailable |

Every successful checkpoint passes 108 following, 108 equality and 108 counting
cases, including duplicate-heavy inputs and sizes through 64. Frozen native
weights reproduce all 20 distinct observed primitive decision states per seed.
Each successful seed also passes contradiction, concept correction, coupled
controller-credit correction, support withdrawal and restoration contracts.
The old primitive policies survive and the newly acquired count is removed when
its support is invalidated, then restored when valid evidence returns.

Seed 0's first two attempts execute 11,779,168,162 simulated primitive instructions
in aggregate. Its failures are retained; they must not disappear from the
success-rate denominator or be folded into successful seeds. The current search
budget is per rebuild attempt, so new outcome evidence can initiate another
expensive attempt. Integration is working in the successful seeds, but this does
not establish reliable or efficient acquisition across initializations.
The intentionally stopped continuation consumed 3,565.52 native CPU seconds
including its unfinished third attempt; that attempt's proposal count is unknown.
`logs/composed_control_raw/0/STOPPED.json` records the decision and completed
attempts. It is an unsuccessful/incomplete run, not an audited final-score result.

The lifecycle tests in these historical runs edited the same trained model's
knowledge journal. The final wholesale language withdrawal/restoration used
knowledge APIs directly, so it does not prove future correction of every old
controller-owned receipt after external journal edits. The coupled credit test
uses the controller API and passes. In the canonical trainer these diagnostics
run on a copy, while `REVISE_CREDIT` is the normal route for changing owned evidence.

Independent per-seed results are in `logs/composed_control_raw/{1,2}/composed_audit.json`.
The build's source snapshot is `logs/composed_control_raw/source_manifest.json`.
Later unit-tested guards preserve negative quantity signs; they are not claimed
to have been in that historical build. Existing audit files also hash current
checker sources, so use the archived manifest to identify the original build.

## Completed regression checks

ASAN/UBSAN tests pass for quantity typing, raw-record preservation, journal
synchronization, controller feedback, stale revisions and clone isolation.
Existing composed-learner and relational-controller contracts also pass.
The existing direct-primitive computational-session regression passes all three
seeds: 1,008/1,008 questions, 120/120 physical goals and 648/648 retained primitive
cases. Its independent audit is `logs/computational_session_raw/audit.json`.

## Canonical consolidation

The user has requested consolidation before another capability gap is pursued.
`laminar_train trainer=continual` now constructs the tested joint learner using
the versioned `coco_continual_v1` profile. The shared training loop is the same
implementation used by the old control probe, now a thin ablation entry point.
The canonical mode adds a resolved configuration, deterministic offline replay
checkpoints and explicit correction records. Its 201-episode regression passes
exact stop/resume across schedule boundaries and correction/restoration. Full
seed-1 acquisition/parity verification passes: 45 files, including complete native
trace, acquisition report, neural checkpoints and value tables, are byte-identical
to the independently audited experimental baseline. It passes 128/128 final,
34/34 recovery, 12/12 larger-goal and 324/324 primitive cases. The parity audit is
`logs/canonical_trainer/1/canonical_audit.json`; its build sources are archived in
`logs/canonical_trainer/source_manifest.json`. The later explicit correction API
and frozen-evaluation guard are covered by the current executable regression in
`logs/canonical_checkpoint_entrypoint/audit.json` and its source manifest. See the
[trainer interface and limits](CANONICAL_TRAINER.md).

The existing byte/grounded reader is still a separate model path. Consolidating
the structured learner is not a claim that all earlier experiments or arbitrary
natural chat have become one model. After consolidation, the user-directed
[warm-start experiment](WARM_START_CURRICULUM.md) will seed goal persistence and
credit assignment, then withdraw assistance. The from-scratch failures remain
part of that comparison.
