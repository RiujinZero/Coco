# Canonical continual trainer

Use **`laminar_train trainer=continual profile=controller50k`** for subsequent
structured capability experiments. The 50,013-parameter MLP passes all three seeds
on both the short and full schedules; the original Agkern size and TD credit
remain the recommended settings. `coco_continual_v1` remains the CLI default for
historical reproducibility, so specify the recommended profile explicitly.
`cpp/include/laminar/continual_training.h` owns the
tested training schedule and model construction. `cpp/apps/continual_training_runner.h`
owns the shared curriculum loop; `relational_control_probe` is now a thin entry
point to that loop for historical ablations. There is no second copy of the
training implementation to keep in sync.

For the current validation boundary and next experiments, read the
[complete-learner handoff](COMPLETE_LEARNER_HANDOFF.md). Retention lifecycle fixes
and component repair are validated. The subsequent [repair integration follow-up](REPAIR_INTEGRATION_RESULTS.md)
records audited continuing-trainer repair and its explicitly experimental profiles.
Interleaved repair is not a canonical default.
The opt-in `controller50k_repair_bank_records` profile enables the
[successive repair bank](SUCCESSIVE_REPAIR_BANK.md), record multiplicity and
observed relation typing with a 1,000-proposal interleave quantum. A focused
structured language session now retains two repairs. The actual profile also
passes a small independently audited trainer wiring check; that check does not
constitute a multiple-repair controller campaign. The
[combined build record](LEARNER_FOLLOWUP_BUILD_VALIDATION.json) verifies exact
canonical replay, correction and parity with eight handoff artifacts.
The normal local executable has been refreshed to the verified handoff build;
its hash and the preserved previous executable are recorded in that handoff.

The current session also preserves usable meanings while unrelated concepts are
being learned. Its [concept-isolation continuation](CONCEPT_ISOLATION_EXPERIMENT.md)
passes three seeds with continued control/action learning and no further hints.
This changes inference availability under unrelated uncertainty; it does not
change the recommended controller size, credit rule or teaching schedule.

One `ControlledRelationalSession` owns learned operation selection, the evidence
journal, neural primitives, language programs, relational concepts, action effects
and reusable physical procedures. Composed final answers can acquire a missing
primitive during the controller's `reason` operation. Old-skill rehearsal and
versioned invalidation remain enabled. New controlled tasks belong in this stream
before another standalone learning loop is introduced.

This consolidates the strongest tested **structured** configuration. It does not
merge the earlier byte `Organism`/`GroundedSession` reader into this model. Those
remain another mode of the same executable. The current adapter consumes the
controlled-world protocol below; it does not yet ingest arbitrary natural chat
or the entire `/home/user/curriculum_v2` dataset. Sharing an executable is not
evidence of full model integration.

## Recommended profile and historical baseline

Capacity, credit and teaching schedule are separate choices. The recommended
`controller50k` uses a 32 → 1,316 tanh → 5 MLP (50,013 parameters), original-size
Agkern and empirical TD. Its preference learning rate is `0.002 * 32/1316`.
The preserved CLI default `coco_continual_v1` uses the smaller MLP shown below;
all other rows apply to both. Neither profile enables teacher hints by default.

On the short three-seed screen, 50k passes with **and without** the operation
taper. That establishes that assistance can be withdrawn, not that the taper
improves this model on the existing tasks. Guidance is an optional bootstrap
for the harder curriculum. Episode-return credit regresses task accuracy and
is not recommended. Replay-priority experiments are also separate from this
recommended profile.

The explicit `controller_trace` and `controller50k_trace` profiles now offer
goal-scoped greedy trace credit. The 50k candidate passes the stronger delayed
propagation contract and the three-seed standard/recovery/interruption screen
without losing primitive skills. This resolves the tested episode-return
tradeoff; it does not establish autonomous long-horizon acquisition. See
[the credit experiment](GREEDY_TRACE_CREDIT_EXPERIMENT.md). TD remains the
preserved recommendation pending the broader continuation evidence.

Controller-only ablations are explicit profiles: append `_frozen`, `_shuffled`
or `_terminal` to `controller50k` or `controller50k_trace`. Other knowledge
learning remains active. The `_frozen_after400` and `_shuffled_after400`
variants preserve normal controller training through the first 400 episodes,
including its consolidation, and change credit starting at episode 401.
The configuration fingerprints that boundary. A frozen-boundary snapshot permits
independent checks that later evaluation and training retain the same control
weights and values. These variants are experimental controls, not recommended
learning profiles.

`controller50k_disjunction` enables outcome-induced alternative concept rules
and their reuse. The exact class, bounds, native/independent checks and remaining
scope are in [the representation experiment](DISJUNCTIVE_CONCEPT_GROWTH.md).

Use the 50k MLP, original Agkern and TD credit as the comparison baseline. For
new, harder tasks, test a declared warm start against that same baseline and
withdraw guidance before evaluation. The current results do not establish that
the 50k controller needs guidance, or that guidance is unnecessary on harder
tasks. The experimental `controller50k_repair` profile enables repair of an
available primitive; its [component evidence](PRIMITIVE_REPAIR_EXPERIMENT.md)
does not yet justify replacing the recommended profile.

The explicit `controller50k_repair_interleaved` profile shares proposals in
1,000-proposal quanta and caches frozen compilation choices and matching candidate
assessments. Its configuration records these settings for replay fingerprints.
`controller50k_records` and `controller50k_repair_records` additionally preserve
repeated observation records while relational truth counts distinct tuples.
These are declared interface experiments; the default observation interface
continues deduplicating records. See the follow-up for the precise evidence scope.
The latest record profiles also ground a new raw predicate's binary arity from
its current observed tuples. Absent untyped predicates and unresolved learned
concepts still withhold; this inference does not add teaching evidence.

## Shared settings and preserved baseline dimensions

| Component | Canonical setting |
|---|---|
| Operation controller | Historical default: 32 → 32 tanh → 5, 1,221 parameters. Recommended `controller50k`: 32 → 1,316 tanh → 5, 50,013 parameters |
| Control credit | Empirical TD values (discount 0.99, update rate 0.2); backprop preference fitting at `0.002 * 32/hidden_width` |
| Credit replay | 128 updates per training episode; 64 consolidation passes every 200 episodes |
| Exploration | 0.4 for the first 200 episodes, then 0.15 |
| Decision limits | 48 during training, 128 during frozen evaluation |
| Physical actions | Reactive, one motor at a time, at most 32 attempts |
| Controller inputs | Existing public observations; extra quantitative-size inputs disabled |
| Search retention | Previous training behavior; retain search allowance across unchanged scenes in evaluation |
| Missing primitive acquisition | 100,000 proposals **per attempt**, population 128, restart interval 20,000 |
| Composed synthesis | At most eight macroinstructions, 1,000 searched program states |
| Neural rehearsal | 128 passes, including verified earlier-skill executions |
| Primitive executor | Agkern sparse expansion: 512 units, 16 active, six inputs per unit; local three-factor route updates |
| Inference | Neural controller and neural primitives; no preference-tree readout or imported discovered policy |

The controller's empirical value table and replay journal are additional learned
memory, outside either neural parameter count. The Agkern skill route tensor has 15,360
weights (10 actions × 512 units × three channels); this adapter trains only the
first channel, or 5,120 potential weights. Its sparse projection is fixed. The
full Agkern competition/commitment loop is bypassed by the default skill selector;
search supplies successful self-generated rehearsal targets. These runs do not
establish Agkern-only delayed-credit learning. Symbolic program/concept search is
also supplied machinery, not part of the neural parameter count.

The CLI defaults to one versioned profile and rejects undocumented hyperparameter
overrides, unknown arguments and duplicate arguments. User-requested experimental
`profile=controller50k|controller100k|agkern50k|agkern100k|both50k|both100k` options
are described in [the capacity comparison](CAPACITY_EXPERIMENT.md). Every run
writes `<output>.config.json`. Preserve the resolved configuration, executable,
source manifest, input files and audit with each experiment. Changes to the
learning settings require an explicit profile and comparison with this baseline.

The optional `controller_return` and `controller50k_return` profiles test
episode-return credit. They pass the stronger delayed-credit protocol contract
but regress on the matched three-seed task screen, so they are **not recommended**.
See [the interruption and credit report](INTERRUPTED_GOALS_EXPERIMENT.md).

## Run and resume

From the repository in WSL/Linux, with the existing validation curriculum:

```sh
bash scripts/train_coco.sh \
  profile=controller50k \
  file=logs/composed_control_raw/1/input.txt \
  environment=logs/composed_control_raw/1/environment.txt \
  worlds=logs/composed_control_raw/1/worlds.txt \
  seed=1 output=logs/my_coco/learner checkpoint=logs/my_coco/state.replay \
  > logs/my_coco.trace.txt
```

The script builds the normal `build-wsl/laminar_train`. For subsequent runs or
checkpoint restoration, invoke that **same executable** directly with
`trainer=continual` and the same input/seed arguments. Add
`resume=logs/my_coco/state.replay`, use a fresh output prefix and redirect the
complete replay/continuation trace to a new file. The existing normal build also
includes this mode. CMake is not required.

Checkpoints are deterministic **offline replay manifests**, not fast weight
snapshots. They record the binary/config/data fingerprints, processed curriculum
line, episode count, complete trace digest and a state digest including both
neural policies and controller values/credit versions. Restore replays the prefix,
checks the trace and state exactly, then continues. Replay can repeat expensive
primitive discovery. The offline environment simulates actions; replay does not
invoke a real actuator. A future live-world adapter must replay recorded callbacks
instead of reissuing real actions.

Checkpoints are saved every 200 training episodes, after each evaluation block
and at normal completion. `stop_after=N checkpoint=PATH` stops at the first safe
boundary at or after physical input line N; it never checkpoints inside an
episode or evaluation block. On resume, N must exceed the saved cursor. A
terminated acquisition attempt resumes from the preceding saved boundary.
Saving replaces the manifest through a temporary file and rename on WSL/Linux.
Keep the original binary and input contents: changed or truncated checkpoint,
binary, profile, seed or data is rejected. Fingerprints detect accidental drift
and corruption; they are not cryptographic authentication.

`lifecycle=1` additionally runs the composed-evidence validation contracts on a
copy after training. These diagnostics do not change the persisted learner and
are intended for the composed validation curriculum, which contains their named
receipts. They are off by default for other curricula.

`audit_models=1` saves each rebuilt semantic model and each distinct primitive
weight snapshot in sidecar files. The independent checker can then reconstruct
incorrect learned answers as well as correct ones, while keeping world truth
separate for scoring. `logs/model_audit_parity/audit.json` verifies that enabling
this observer leaves eight canonical trace/config/weight artifacts unchanged.
`logs/learned_computation_audit/unit.json` verifies wrong-versus-repaired answers,
the archived native neural choices and rejection of a tampered choice. Per-slot
search costs are additionally recorded in `.discovery_attempts.tsv`; the older
`.acquisition.tsv` remains the selected-attempt summary.

`search_progress=1` adds flushed phase and budget counters to stderr without
changing learning. Counters distinguish materialization, macro compilation,
candidate proposals, rehearsal and recompilation. Memoized assessments retain
legacy logical search-work counts; `assessment_cache_totals` separately reports
cache hits and avoided macro states/instructions. A stopped native run therefore
retains useful diagnostics. Experiment drivers accept explicit wall-time limits
and preserve partial outputs and timeout status.

## Curriculum boundary

The current file adapter shares the independently audited wire format used by
the control experiments. Strings containing spaces use C++ quoted-string syntax.
A world is `complete name_count names... fact_count (predicate arity args...)...`.
An action is `schema arity args...`. Complete worlds and verified teaching are
host-provided claims; generated assistant text must never be inserted as evidence.

| Record | Fields after the record name |
|---|---|
| `P` | id, skill cue, relation, subject type/id, target type/id, final answer type/id, fact count, integer triples |
| `C` | id, new concept name, final Boolean answer, world (names are ordered arguments) |
| `L` | id, final Boolean answer, world, quoted controlled-language text |
| `M` | same as L, then relation-mention count and names |
| `N` | same as M, then quantity-mention count and unsigned integer strings |
| `T` | id, action, observed before world, observed after world |
| `GUIDANCE` | `public_control_v1`, full-assistance episode count, taper episode count; once before training |
| `TRAIN` / `TEST` | id, physical-intent Boolean, public price profile (0/1), disabled operation (-1 normally), visible world, quoted text, action count, offered actions |
| `CHECK` / `ENDCHECK` | start a named frozen evaluation block / end it |
| `INTERLEAVE_TEST` | parent TEST fields, child count, then (parent decision count, child TEST fields) for each interruption |
| `REVISE_CREDIT` | episode id, expected version, corrected terminal correctness, corrected terminal satisfaction |
| `REVISE_PRIMITIVE` | receipt id, expected version, corrected final answer type/id, active Boolean |
| `REVISE_CONCEPT` / `REVISE_LANGUAGE` | teacher receipt id, expected version, corrected Boolean answer, active Boolean |
| `REVISE_TRANSITION` | receipt id, expected version, active Boolean, corrected observed after world |

Use `REVISE_CREDIT` for controller-owned evidence: it revises the paired language
or execution receipt and replays the controller updates together. Teacher-facing
language revision cannot bypass the evidence source check. Teaching/revision
inside frozen evaluation and stale revisions are rejected. Computational typing
supplies entities, relations and quantities, not programs or intermediate answers.
Raw teaching records preserve duplicates and order.

The optional `GUIDANCE public_control_v1 32 128` declaration supplies hints on
every decision for 32 training episodes, then reduces their probability linearly
to zero over 128 episodes. Episode 161 onward receives none. The training-side
teacher only reads the 32 public controller features: it suggests reading,
evaluating the requested goal, extending a plan, executing its next step or
stopping. The learner's own planner and neural primitives still select physical
actions and computations. This is explicit operation supervision, not evidence
that these decisions were discovered independently. It does not teach a new
goal-stack representation or supply primitive instruction traces.

`G` trace records identify phase/rate; `H` records identify the selected operation,
teacher and intended subgoal. Teacher attribution also remains in the credit
journal. A teacher-chosen operation receives the actual verified terminal reward;
there is no assumed success, extra imitation loss or invented value target.
Every `TEST` is unassisted and the session API rejects hints when learning is
frozen. Guidance cannot be restarted after training begins. The replay checkpoint
binds the schedule through its input fingerprint and trace, reconstructs the
training counter and resumes the same taper and intervention lottery.

`environment` and `worlds` are private **offline evaluator** files. The former
contains physics/terminal truth definitions; the latter contains hidden scenes,
memory responses and interruptions. Only visible inputs, read/action callbacks,
costs and terminal correctness/satisfaction enter the learner. The environment
does not supply a correct next operation or synthesis program. The optional
declared teacher above is a separate training source. The C++ session
API (`begin`, `choose`, `read`, `actions`, `finish`) is the boundary for adding a
real text-world host later; this file adapter is not that integration.

## Validation and open limits

The full seed-1 canonical run reproduces 45 archived artifacts byte-for-byte,
including the native trace and learned checkpoints. Independent audits pass
128/128 final control cases, 34/34 recovery cases, 12/12 larger goals and 324/324
primitive checks. The current CLI regression also passes exact resume and coupled
correction/restoration. These are completed checks, not proposed scores.

Run the bounded persistence/schedule regression:

```sh
bash scripts/train_coco.sh --check --out logs/canonical_checkpoint_rerun
```

It runs 201 training episodes, crossing both the exploration and consolidation
boundaries; stops after episode 199; resumes with byte-identical complete traces,
weights and value tables; checks frozen evaluation, coupled credit correction
and restoration, and rejects corrupt checkpoints, changed data, stale revisions
and unknown settings. It uses direct primitive lessons to keep this test bounded;
it is not an acquisition benchmark.

The separate full baseline comparison uses the composed curriculum above and
the independent `check_composed_control.py`/`primitive_checkpoint_probe` audits.
To repeat it against an existing audited seed, in a fresh destination:

```sh
bash scripts/validate_canonical_trainer.sh logs/composed_control_raw/1 logs/canonical_validation_rerun
```

This is a costly acquisition run, separate from the bounded persistence check.
See [the integration report](COMPOSED_SESSION_EXPERIMENT.md) for results, failures
and source provenance. Discovery remains expensive and seed-sensitive; retries
on new evidence can each spend another 100,000 proposals. The follow-up demonstrates
bounded available-skill repair in the continuing trainer. It does not demonstrate
unbounded representation growth, general language understanding or a complete
continual learner. The four
[capability gates](FINAL_CAPABILITIES_VALIDATION.md) remain open at that scope.
