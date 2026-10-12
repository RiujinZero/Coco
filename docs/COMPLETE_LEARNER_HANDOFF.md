# Coco complete-learner handoff — 2026-09-23

**Current continuation handoff:** [CONTINUING_LEARNER_HANDOFF.md](CONTINUING_LEARNER_HANDOFF.md)
records the joint warm-start experiment, latest user direction, wrap-up and
remaining work. The historical handoff below remains preserved.

**Latest capability follow-ups:** [Greedy trace credit](GREEDY_TRACE_CREDIT_EXPERIMENT.md)
passes the delayed-propagation contract without the earlier short-task regression
across three seeds. [Reusable rule unions](DISJUNCTIVE_CONCEPT_GROWTH.md) extend
concept expressivity and pass component plus symbolic-session checks. The
[reader boundary](GROUNDED_RELATIONAL_BOUNDARY.md) preserves learned report
uncertainty and revision identity when providing conservative controller
observations. These are scoped advances; the four full capability gates in
[FINAL_CAPABILITIES_VALIDATION.md](FINAL_CAPABILITIES_VALIDATION.md) remain open.
The [successive repair bank](SUCCESSIVE_REPAIR_BANK.md) now retains two repairs
from composed final answers, including through the actual structured language
session. This is a focused one-seed result, not simultaneous multi-fault search
or the full mixed continuation. The [combined build validation](LEARNER_FOLLOWUP_BUILD_VALIDATION.json)
preserves eight canonical handoff artifacts exactly and verifies the opt-in bank
profile through the actual trainer. The normal binary and recommended profile
remain unchanged. [Learned goal control](LEARNED_GOAL_CONTROL_NEXT.md) now has an
implemented seven-output mission controller, passing protocol checks and a
corrected reference feasibility result. The user prioritized learning over
efficiency: current experiments use 256 decisions and zero operation prices,
with actual work still recorded. The first learning pilot fails autonomous
switching; it is not a capability pass. [Remaining gates](LEARNER_REMAINING_GATES.md)
tracks the latest results, diagnoses and integration boundaries.
The user now directs a single continuing mixed experiment: warm-start goal
selection, subgoals, stopping, language, actions and skill repair together, then
taper guidance and measure autonomous behavior and retention. Explicit
[demonstration backpropagation](GOAL_DEMONSTRATION_FITTING.md) is implemented and
validated; isolated perfect component scores are not prerequisites for that run.
The sections below preserve the original handoff's evidence and open items;
consult these follow-ups before treating an older item as still unimplemented.

**Completed repair follow-up:** [Continuing-trainer repair results](REPAIR_INTEGRATION_RESULTS.md)
records three-seed continuing-trainer repair, visible-control recovery, retention
and correction/withdrawal validation. The
historical interrupted-run account below remains preserved; consult the follow-up
for newer evidence and the remaining validation boundary. Several repaired skills,
autonomous goals/credit, representation growth and reader integration remain open.

Coco has a working, jointly trained structured learner and several necessary
reasoning mechanisms. We cannot yet call it a complete learner. The remaining
claim is that one continuing model can independently decide what to compute,
assign delayed credit, acquire or repair representations and skills, and retain
earlier knowledge while doing all of those things together.

The immediate retention bug is fixed and its actual canonical lifecycle passes.
Available-skill repair also passes three seeds as a component, with a much cheaper
optional search schedule. **Available-skill repair inside the full continuing
trainer remains unvalidated.** Its first paired run was deliberately interrupted
for handoff after 34.4 minutes; do not report that run as successful or as proof
of impossibility. No new capability campaign should be inferred from this handoff.

## Starting configuration and authorized direction

Use `laminar_train trainer=continual profile=controller50k` explicitly. The CLI
default remains the historical small-controller profile for reproducibility.
The recommendation is the 50,013-parameter MLP (32 → 1,316 tanh → 5), original
Agkern, empirical TD values and backprop preference fitting. The controller
selects retrieve, reason, ask, act and return. Agkern supplies the shared locally
trained neural primitive policy; its original size is 512 units, 16 active,
six fixed input claws, 15,360 allocated weights / 5,120 trainable channel-0
weights. Symbolic storage and supplied search also make substantive contributions.

The user permits backprop to establish that the complete learner exists first,
then a matched non-backprop implementation. Warm starts for goals, planning,
primitive choices and credit are also authorized, with assistance withdrawn.
Do not impose a from-scratch-only restriction. At the same time, score independent
adaptation after assistance ends separately from imitation during warm-up.

Do not promote larger Agkern, 100k controller or episode-return credit by default:
larger Agkern produced no measured gain; the 50k controller was stronger than
100k in the short screen; episode-return credit fixed a delayed-credit contract
but regressed matched task accuracy. Hints are optional: current 50k tasks pass
with and without them. See [CANONICAL_TRAINER.md](CANONICAL_TRAINER.md),
[CAPACITY_EXPERIMENT.md](CAPACITY_EXPERIMENT.md) and
[WARM_START_CURRICULUM.md](WARM_START_CURRICULUM.md).

The structured trainer and earlier byte `Organism` / `GroundedSession` reader
are separate model modes. The canonical adapter does not ingest arbitrary chat
or the whole `/home/user/curriculum_v2` dataset. Do not confuse sharing an
executable with having one learned language/reasoning model.

## What is established at handoff

| Evidence | Completed result | Scope / location |
|---|---|---|
| 50k controller, short schedule | 384/384 final, 96/96 recovery, 36/36 larger goals, 972/972 primitive checks across three seeds | `logs/capacity_screen`; supplied bounded runtime and search |
| Tapered operation hints | Same perfect short-screen endpoints; no hints in evaluation or final 240 training episodes per seed | `logs/warm_start_screen`; hints not proven necessary here |
| Continued concept/action growth | 384/384 old tasks during ambiguity, 384/384 after growth, 192/192 new action goals, 972/972 primitive checks | `logs/concept_isolation_screen_v2`; named concept interface and bounded conjunction induction |
| Goal interruptions | 108/108 parent tasks, 144/144 intervening tasks, 144 resumes | Host still chooses when to switch/resume; see `INTERRUPTED_GOALS_EXPERIMENT.md` |
| Available primitive repair | 1,680/1,680 composed answers, 324/324 count, 648/648 retained follow/equality | `logs/primitive_repair_full`; outcome search plus local rehearsal, component only |
| Cheaper interleaved repair | Same held-out success; 141,833 total proposals versus 900,000 | `logs/primitive_repair_interleaved_final/aggregate.json`; optional component setting, not canonical wiring |
| Mixed-frame evidence lifecycle | All eight stages; changed concept withholds two dependent answers, preserves 66 unrelated answers and exact primitive weights | `logs/mixed_frame_lifecycle_support/audit.json`; correction, contradiction, withdrawal and restoration tested |
| Audit of changing learned computation | Independent Python reconstruction distinguishes wrong learned answers from world truth; observer leaves behavior unchanged | `logs/learned_computation_audit/unit.json`, `logs/model_audit_parity/audit.json` |

These are finite controlled tests. They establish the listed behavior, not open
ended reasoning, unrestricted language, or all four capability gates.

## The fixes and checks just completed

An unresolved relation could discard unrelated lessons sharing its language
frame. `RelationalSession` now filters unavailable materializations per row,
keeps the raw journal, and preserves original receipt IDs. Relevant uncertainty
still withholds an answer. It does not make unknown evidence true or false.

The full lifecycle then exposed a second problem: a cached learned primitive
depended on every original acquisition witness remaining unchanged, even when
independent evidence still justified every learned decision. `ComposedLearner`
now replays unchanged outcome witnesses and active direct priors, requires
coverage of the accepted state/action targets, and recompiles against current
answers before reuse. Removing redundant support preserves exact weights;
removing necessary coverage prevents cache reuse; complete withdrawal removes
derived skills. The reproducing failure is preserved in
`logs/mixed_frame_lifecycle_v2/failure.json`. Optimized and ASAN/UBSAN contracts
pass under `logs/composed_support`; the actual corrected lifecycle passes.

`PolicySearch` now retains population, random state and best candidate across
bounded `advance()` calls. `ComposedSkillDiscovery.interleave_quantum=1000`
shares work among the three possible faulty slots and stops at the first full,
verified acquisition. A full macro fit that did not use a candidate is not
sufficient to terminate the other candidates; a sanitizer contract caught and
fixed that edge case. Quantum zero preserves the old sequential/refinement
behavior, including 29 byte-identical archived native artifacts. Search
continuation across forced restarts also matches uninterrupted search exactly.

The final interleaved component rerun passes all three seeds. Search costs are
65,755 / 2,006 / 74,072 proposals and 478,232,528 / 9,699,452 / 534,212,814
simulated instructions. This is 84.2% fewer proposals and 86.0% fewer simulated
instructions in aggregate. Do not claim an equivalent wall-clock speedup from
concurrent runs. Earlier stopping can produce slower accepted procedures:
seed 1 count reaches 495 instructions versus 345 after full refinement.
Without old-skill rehearsal, seed 0 loses all 108 follow cases; with rehearsal
it retains all of them. Keep rehearsal enabled.

The search scheduler is supplied. Reducing its cost is not evidence that Coco
learned when to search, how much to search, or which skill deserves repair.
Interleaving is not yet exposed through `ComposedLearner` / canonical profiles.

## Interrupted full-trainer repair comparison

`python/controlled_repair_experiment.py` compares `controller50k` with
`controller50k_repair`. Both learn from 120 direct final-answer lessons, complete
the 400-episode warm curriculum, receive 64 raw composed counterexamples and
continue without renewed hints. No pretrained primitive weights are imported.

The seed-1 baseline completes: 128/128 warm final tasks, then 0/32 after-repair
tasks and only 32/128 retained old tasks. Its standalone count remains 13/108;
follow/equality remain 216/216. The corrected independent audit is in
`logs/controlled_repair_seed1/reaudited_summary.json`.

The repair arm completes the warm prefix and before-repair checkpoint but has
no completed post-counterexample rebuild. It was terminated at 2,063.56 seconds
elapsed / 2,059.78 seconds native CPU. `handoff_interruption.json` records the
exact command, preserved replay checkpoint and reason. The manager exits nonzero
and records that termination; it also retains the baseline's original audit
error, which the separate corrected re-audit resolves. No job from that pair is
still running. The
checkpoint is an offline replay manifest; it does not save a search frontier.
Resuming will repeat substantial computation. Its original binary is
`build-wsl/laminar_train_repair_v4`; preserve it for that checkpoint.

There is also a test-design limitation: controller observations are sorted and
deduplicated, while raw teaching retains duplicate records. The 32 duplicate
scene tests therefore pass before repair even though the standalone primitive
is wrong. They cannot establish visible control error → repaired control on
those same cases. The audit initially missed this normalization boundary; its
correction and the original failed audit remain recorded. Use an error that
survives public observation normalization for the decisive experiment, or
explicitly define and validate bag semantics across both interfaces.

## What remains, how to check it, and candidate mechanisms

### 1. Repair and acquire computation in the continuing trainer

This is the closest next experiment. First measure where the full rebuild spends
time: materialization, baseline macro compilation, per-slot policy proposals,
candidate program search and repeated recompilation. `ComposedLearner::compile`
can search 50,000 macro states per group before primitive discovery; do not
assume the entire delay is in the slot scheduler. Flush counters at phase and
budget boundaries so a stopped run remains diagnosable.

Expose the validated interleaving option through explicit experimental config,
include it in config/checkpoint fingerprints, and keep the baseline unchanged.
Then run a bounded, instrumented one-seed integration check on the already
generated stream before a full matched multi-seed run. A bounded refinement
phase after first fit may recover execution efficiency; test it against first
fit and fixed-budget refinement under equal total compute.

Pass conditions: a demonstrably wrong primitive/control answer becomes correct
from subsequent final-outcome evidence; its old direct outcomes remain correct;
held-out control, language and primitive tests recover together; unrelated skills
remain intact; corrections and withdrawal work; and three prespecified seeds
finish within a declared cost budget. Report both training failures and search
timeouts. Test several successive and simultaneous wrong skills next. The
current single cached acquired cue is not a demonstrated multi-skill repair bank.
Also check execution-limit consistency when increasing task size: direct cache
validation currently caps primitive replay at 128 instructions, while repair
and old-outcome rehearsal use 256. Neither limit should silently turn a correct
long procedure into an unsupported skill. No failing case for that boundary is
claimed by the current tests.

Possible mechanisms: per-skill provenance and validated support sets; separate
repair candidates with atomic acceptance; rehearsal of verified outcomes through
the new policy; persistent candidate/frontier state; uncertainty-based allocation
of proposal budgets. A learned repair selector should be compared with the
supplied scheduler, without receiving the faulty skill ID.

### 2. Learned computation choice beyond the supplied task families

The five-way controller already works on the tested structured tasks. The missing
claim is independent adaptation when the useful operation sequence, evidence
availability, task structure or cost changes. Planner search and actuator
exploration still contain supplied policy. More successes on renamed versions
of the same templates will not resolve this.

Check unfamiliar operation orderings and task families after hints end, including
cases where asking is necessary, harmful or unavailable; retrieval is stale;
reasoning has exhausted its current budget; and stopping early is wrong or
continuing is wasteful. Compare frozen, shuffled-credit and disabled-operation
controls. Log true success, calibration/withholding, cost and unnecessary work.
Audit all 32 public features for answer/tool-label leakage.

Possible mechanisms: recurrent or explicitly addressable working memory; a
learned value of information and computation; learned selection of retrieval,
exploration and reasoning budgets; reusable learned options with termination
policies. Bootstrap demonstrations are allowed, but new post-taper tasks must
require learning beyond the demonstrated routes.

### 3. Long-term credit and autonomous goal management

Goal suspension and restoration work; the host still schedules them. Empirical
TD fails the stronger delayed propagation contract after 16 intervening goals
within 64 updates. Episode-return credit passes that contract but regresses
real task accuracy. Neither result closes delayed credit.

Check increasingly long sparse-feedback tasks, distractions, interruption and
resumption, several pending goals and changing priorities. The learner must
choose useful subgoals, switch/resume appropriately, and assign success or
failure to the relevant earlier decisions. Vary irrelevant intervening episodes
independently from causal horizon. Test fixed compute and sample budgets and
retain the short-task regression suite. Add persistence of unfinished goals and
credit state across a checkpoint; current checkpoints occur between records.

Possible mechanisms: goal-scoped eligibility traces / TD(lambda), multi-step
returns with a value baseline, causal episode replay, recurrent state or explicit
goal memory, and learned option initiation/termination. With backprop permitted,
a recurrent actor/critic is a legitimate sufficiency experiment. Warm-start goal
decomposition and successful long trajectories, then fade labels. Measure the
whole task tradeoff before replacing TD; do not optimize only the unit contract.

### 4. Growth beyond the supplied representation and operation bank

New named concepts and bounded conjunctions are acquired. The primitive VM still
has three supplied slots, ten microoperations, a fixed state abstraction and
bounded program synthesis. More combinations inside that interface do not prove
growth of the interface itself.

Check tasks deliberately outside the existing expressible family. Require Coco
to add and reuse a representation or callable skill, grounded in public evidence,
without a developer adding its task-specific operator, relation definition or
feature. Then revise it, compose it with earlier skills and test novel bindings,
arities and longer executions. Declare the starting generic substrate and all
capacity limits. A generic interpreter is allowed; a supplied solution is not
evidence of learned representation growth.

Possible mechanisms: extensible typed slots and learned schemas; a library of
induced program chunks with learned interfaces; hierarchical options; learned
state abstractions / encoders; expandable neural heads with consolidation and
provenance. Choose a minimal new expressivity test first, such as reusable
relational composition beyond the current conjunctive grammar. Keep its target
implementation out of the inference interface.

### 5. One continuing language/reasoning/action learner

The decisive validation combines the preceding four requirements in one run.
The existing continuing concept-growth result is useful evidence but does not
yet combine autonomous goals, broad representation growth and available-skill
repair. The full language reader is also not integrated.

Start with the canonical structured stream as a testable semantic interface.
Add grounded paraphrases and dialogue through a declared adapter, preserving
uncertainty and evidence source identity. Introduce new concepts, computations,
language forms and action effects at different times, then revisions and
withdrawals, delayed rewards and old-skill probes. Evaluate actual reader output
as well as an oracle semantic-interface ablation to locate failures. Do not allow
generated assertions to become verified training evidence automatically.

Possible mechanisms: shared versioned episodic/procedural memory, learned
grounding into typed working memory, outcome-verified self-generated rehearsal,
selective replay, and explicit provenance for derived beliefs and skills. Use
backprop across learned components where helpful before demanding local-only
equivalence. Add uncertainty/noisy-evidence handling after the deterministic
integration works; the current verified-world protocol is a strong assumption.

## Suggested curriculum and acceptance contract

Use structured grounded tasks first to isolate whether the mechanisms learn.
Progress to varied controlled language and paraphrases of the same worlds,
then partially observed text worlds and dialogue with memory, corrections and
multi-step goals. Interleave old tasks throughout. Delay unrestricted chat as
the main training target until language is grounded into the continuing model;
fluent next-token behavior alone would not settle any of these gates.

A practical next campaign should freeze its generators, config, budgets and
three or more seeds before training, and keep additional untouched seeds/task
families for confirmation. Split by reasoning structure and environment changes,
not just entity names. Log assistance per decision and demonstrate it reaches
zero before independent adaptation and evaluation. Freeze learning during scored
evaluation, then resume the same checkpoint for further learning.

Agree exact thresholds before seeing results. Suggested initial acceptance:
all deterministic lifecycle/safety contracts pass; at least 95% success in each
prespecified new-task family on each seed; no more than two percentage points
of old-task loss from the matched pre-growth checkpoint; materially better than
frozen/shuffled controls under the same compute; no unreported restarts, failed
seeds or hidden supervision. These are proposed engineering thresholds, not a
definition of intelligence or thresholds already met by the combined learner.

We can reasonably say "probably a complete learner for the declared domain"
when one unchanged learning architecture passes that mixed continuation test,
learns unfamiliar families after hints end, and survives correction and forgetting
checks without adding a mechanism for each task. Broader open-ended claims require
broader evidence. Re-run the same contract for a local-only learner later.

## Code and evidence map for the next person

| Work | Main entry points |
|---|---|
| Canonical config / loop / checkpoints | `cpp/include/laminar/continual_training.h`; `cpp/apps/continual_training_runner.h`, `continual_train_cli.h`, `continual_checkpoint.h` |
| Controller / goals / credit | `cpp/include/laminar/relational_control.h`, `computation_control.h`, `backprop_choice.h`; `cpp/tests/test_goal_suspension.cpp`, `test_episode_return.cpp` |
| Raw evidence / grounded computation | `cpp/include/laminar/relational_session.h`, `composed_learner.h`, `primitive_store.h` |
| Primitive repair and search | `cpp/include/laminar/composed_skill_discovery.h`, `policy_discovery.h`; `cpp/apps/composed_credit_probe.cpp` |
| Semantic growth / programs | `cpp/include/laminar/concept_memory.h`, `procedure_memory.h`, `reasoning_runtime.h` |
| Independent audits | `python/check_relational_control.py`, `learned_computation_audit.py`, `check_composed_credit.py`, `check_primitive_discovery.py`; `cpp/apps/continual_model_audit.h` |
| Current experiment drivers | `python/controlled_repair_experiment.py`, `primitive_repair_experiment.py`, `check_mixed_frame_lifecycle.py`, `check_canonical_trainer.py` |

The user requested a local checkpoint commit of the accumulated workspace,
including this handoff. Its scope includes substantial earlier source and
experiment work; the checks reported here validate the current fixes, not a
fresh rerun of every historical experiment. No remote publication is requested.
Preserve failed runs, source manifests and binary fingerprints; do not overwrite
archived experiment binaries. The user's dataset at `/home/user/curriculum_v2`
was not modified.
Read `docs/CURRICULUM_DATASET.md` and the dataset's actual manifests before
planning full ingestion. Recheck text-world readiness; this handoff makes no
new claim that every dataset stage has completed or trained successfully.

Use new output directories when reproducing. Run these from the repository
root inside WSL; final validated build/artifact status is recorded below.

```sh
python3 python/check_canonical_trainer.py --binary build-wsl/laminar_train_handoff_v1 --out logs/canonical_handoff_rerun
python3 python/check_mixed_frame_lifecycle.py --binary build-wsl/laminar_train_handoff_v1 --root logs/mixed_frame_handoff_rerun
python3 python/primitive_repair_experiment.py --binary build-wsl/composed_credit_probe_interleaved_v3 --root logs/repair_interleaved_rerun --proposals 100000 --quantum 1000
```

The canonical handoff regression has passed 201 training episodes, two frozen
episodes, exact replay across the consolidation boundary, coupled terminal-credit
correction and six rejection checks. Eight trace/config/weight artifacts are
byte-identical to the preceding support build. See
`logs/canonical_handoff_regression/audit.json` and `default_parity.json`.
The final component binary also reproduces all 87 native artifacts of the prior
three-seed interleaved run exactly, while including the separately tested
unused-candidate stopping fix. Machine-readable evidence is collected in
`logs/handoff_validation/validation.json`; source snapshots are retained with
the final lifecycle and component experiment roots.

The final combined build also **passes the complete eight-stage lifecycle** in
`logs/mixed_frame_lifecycle_handoff/audit.json`. All 21 native artifacts match
the earlier passing support build byte-for-byte. This completes validation of
the retention fix after the search refactor; it does not validate the interrupted
available-skill repair stream.

`build-wsl/laminar_train` now matches the frozen validated
`build-wsl/laminar_train_handoff_v1`, SHA-256
`e6b57f6bf7c1e000676964d3dbbac8f1b8fde5eb6d549e674b8f9812f0a2f23b`.
The previous normal executable is preserved as
`build-wsl/laminar_train_pre_handoff_96f3b4ae0531` for older replay fingerprints.
Recommended settings are unchanged. No validation job launched for this handoff
remains running.

[HANDOFF_VALIDATION.json](HANDOFF_VALIDATION.json) is the compact committed record
of results, binary hashes and evidence hashes. Full `logs/`, `build-wsl/` and
datasets remain ignored local artifacts; a Git checkout elsewhere does not
contain them. Copy the relevant evidence directories separately when moving the
handoff to another machine, or reproduce into fresh roots.

Start the next phase by reading this file, `PRIMITIVE_REPAIR_EXPERIMENT.md` and
the actual audit JSONs. The immediate next decision is how to bound and instrument
full-trainer repair, then verify its semantics in the shared stream. Broader
goal/credit, representation and reader integration experiments follow that result.
