# Seed the learner, then withdraw assistance

User direction, 2026-09-23: use warm starts for the remaining credit, goal and
control blockers where needed. Establish the complete learner with backprop
permitted; compare a non-backprop implementation later. Consolidate the current
trainer before introducing this next capability experiment.

Subsequent user direction adds a [50k/100k capacity comparison](CAPACITY_EXPERIMENT.md)
for the controller and Agkern. Run it through the same canonical trainer before
promoting a larger configuration into the warm-start curriculum.

Latest clarification: guide initial **goals/subgoals, planning steps and primitive
selections**, then remove that guidance as training progresses. The controller is
already a nonlinear MLP (32 inputs, tanh hidden layer, five outputs, both layers
trained by backprop); the capacity profiles widen that layer. This does not require
converting a linear controller into an MLP. A deeper architecture would be another
controlled comparison, not an unreported change to the current size experiment.

The primary hypothesis is that a small verified bootstrap can establish useful
behavior and credit signals that sparse terminal outcomes do not reliably discover
from scratch. Subsequent learning must extend and revise that bootstrap. This is
a proposed experiment, not a result. The current composed acquisition succeeds
for two seeds but fails repeated search budgets for another, so initialization
and teaching exposure must be measured rather than assumed irrelevant.

## Initial teaching

| Seed | Initial experience | What must transfer after assistance ends |
|---|---|---|
| Goal persistence | One- and two-step tasks; identify the requested outcome, observe a distraction, resume; distinguish completing, cancelling and replacing a goal | Longer interruptions, changed requests, failure recovery, no continuing teacher reminders |
| Delayed credit | Verified short successful and unsuccessful trajectories; begin near the goal, then move episode starts farther back; progressively delay outcome feedback | New longer trajectories learned from final outcomes, including correcting an initially plausible but wrong action |
| Computation control | Demonstrate a few retrieve/ask/reason/act/finish decisions with their public observations and verified consequences | New combinations, costs and missing-information patterns; appropriate stopping without a suggested next operation |
| Planning and primitive selection | Show a small verified subgoal/action sequence and the primitive calls that support it; check each demonstration against the observed world and final outcome | Choose and combine these operations on fresh tasks without a provided plan, then revise a seeded choice when its consequences change |
| Representation growth | Contrastive examples of a reusable concept, followed by a second concept that uses it | Untaught compositions and learning another concept after demonstrations stop; growth beyond a fixed operation bank remains a separate test |

Begin with short controlled language over observable worlds. Mix earlier skills
through every stage. Do not begin with broad chat as the main learning signal.
Sample both successful and failed attempts so that a demonstrated action is not
treated as universally correct. Keep demonstrations small enough that all taught
templates, goals and action patterns can be enumerated in the experiment report.

## Reduce support explicitly

1. Supply verified demonstrations and short-horizon consequences during a declared
   bootstrap phase. Teacher hints, goal/subgoal annotations and action labels are
   recorded as supervision with source and phase; they are never called answer-only
   discovery or quietly converted into ordinary observed facts.
2. Move starts earlier in the task, lengthen delays and add interruptions. Reduce
   demonstrations to partial trajectories or occasional hints. Choose advancement
   based on independent transfer and retention checks, not training fit alone.
3. Remove hints, intermediate targets and teacher-chosen actions. Continue
   learning on fresh tasks using terminal verified outcomes and declared costs.
4. Introduce a changed effect or wrong inherited skill. Require Coco to repair its
   behavior from new evidence while retaining unrelated old skills.

A supplied initial goal is appropriate task input. A teacher-provided internal
subgoal sequence is additional supervision and must be removed from autonomous
evaluation. Synthetic traces must pass the world verifier before rehearsal. Test
answers and private evaluator state never enter policy inputs in any phase.
**Every evaluation checkpoint is unassisted**, including those during the bootstrap.
Use separate training-side competence checks to decide when to reduce guidance;
do not feed held-out evaluation answers back into the curriculum or learner.
Record the guidance rate, hinted decisions and unassisted successes at each phase.

## Required comparison

Use the same canonical learner and held-out tasks for:

- A from-scratch baseline at declared training/search budgets.
- A warm-started learner with assistance withdrawn before independent adaptation.
- The same warm start with subsequent learning frozen, to detect mere replay.
- A warm-started arm without earlier-skill rehearsal, to measure retention cost.

Record demonstration count, outcomes, intervention frequency, compute spent
creating demonstrations, search proposals, executed instructions and final neural
updates. Report matched total-resource comparisons when possible; a larger
teaching/compute budget is not evidence for a better learning mechanism by itself.
Run several fixed seeds, including failed seed 0. Do not select only the successful
initializations. Evaluation covers fresh entities, duplicate records, longer
delays, held-out combinations, changed effects, evidence revision and old skills.

After the operation-bootstrap screen, the next capability test should target
**goal persistence plus delayed control credit**
on short interrupted tasks, then withdraw demonstrations and increase the delay.
Only expand to simultaneous primitive repair and new representations once that
learning signal works. Existing successful skill policies are useful warm-start
material, but importing one must be reported as an imported seed, not acquisition
from final answers in the new run.

## Trainer boundary

Use [the canonical trainer](CANONICAL_TRAINER.md) as the implementation home.
Its `P/C/L/M/N/T` records carry verified final labels or transitions. The first
implemented bootstrap interface is `GUIDANCE public_control_v1 full fade`, plus
an attributed `ControlHint` in the controller API. This teacher only reads public
controller observations and guides operation choice; plans and primitive
executions still come from Coco's learned memory. It does not yet import arbitrary
demonstration trajectories or teach explicit primitive instructions. The initial
comparison keeps terminal credit unchanged to isolate guided experience from
additional imitation losses. Each hint and phase is logged, every evaluation is
unassisted, and deterministic replay restores the taper without restarting it.

`python/warm_start_experiment.py` uses the capacity screen's same worlds with
32 fully assisted episodes, a 128-episode taper and 240 further unassisted training
episodes. It compares the baseline and 50k controller across all three seeds.
This is a first operation-bootstrap screen, not the full interrupted-goal,
wrong-skill repair or representation-growth curriculum described above. Results
are reported below after independent audits of all six warm runs. It runs through the same canonical
training loop, using a separately named binary while the capacity comparison is
still running so that experiment's executable remains unchanged.

The implemented boundary passes an ASAN/UBSAN test of attributed teacher credit,
actual negative reward for a failed teacher choice, taper boundaries and rejection
of hints by a frozen learner. `logs/guidance_checkpoint/audit.json` verifies exact
trace/weight replay from both the taper (after episode 5) and autonomous phase
(after episode 10), with no resumed assistance after withdrawal. Four invalid
schedule cases are rejected. The default trainer, without a `GUIDANCE` record,
remains byte-identical to the prior default on the 201-episode regression;
see `logs/canonical_guidance_regression/default_parity.json`.

## Audited first seed

Seed 0 at 400 total training episodes:

| Controller / assistance | Final | Recovery | Larger goals | Retained primitives |
|---|---:|---:|---:|---:|
| 1,221 parameters / none | 98/128 | 0/30 | 0/12 | 324/324 |
| 1,221 / tapered teacher | 128/128 | 30/30 | 12/12 | 324/324 |
| 50,013 / none | 128/128 | 30/30 | 12/12 | 324/324 |
| 50,013 / tapered teacher | 128/128 | 30/30 | 12/12 | 324/324 |

Both warm arms used 457 teacher-selected decisions (157 in bootstrap, 300 in
taper), followed by zero hints in 240 further training episodes and every test.
They each succeeded on 207/240 of those exploratory training episodes and used
69,952 controller updates in total. Teacher guidance improved the small model on
this seed without increasing its update count (the baseline used 70,720).
Training still includes the declared final-label and transition curriculum;
withdrawing operation hints is not withdrawal of all learning feedback.

Independent checks validate public inputs, every intervention's public-observation
rule and taper lottery, neural choices at evaluation, physics, terminal credit,
primitive execution and frozen weights. `logs/warm_start_screen/paired_summary.json`
records matched inputs, phase counts and scores, including pending pairs.

Seed 1 is now audited too. The small warm-started controller scores 122/128 final,
25/30 recovery and 12/12 larger goals, compared with 128/128, 30/30 and 12/12
without guidance. The 50k controller scores perfectly with or without guidance.
Both retain 324/324 primitives. Warm starting therefore is not a universal fix.
All six small-controller final failures repeatedly choose `reason` in the same
incomplete-world observation. Its own value table also prefers `reason` (0.808)
over `ask` (0.794); this is not simply neural disagreement with the learned values.
That observation had four teacher `ask` interventions, eight total `ask` choices,
nine `reason` choices and one `retrieve` choice during training. See
`logs/warm_start_screen/coco_continual_v1/1/stage4_failure_diagnosis.json`.
This points to a remaining credit/value-estimation issue; a revised learning
rule still requires a separate, training-only comparison.

## Completed warm-start screen: three seeds

All six warm-started runs are independently audited. Hints are absent throughout
the final 240 training episodes per run and every evaluation checkpoint.

| Controller with taper | Final | Recovery | Larger goals | Retained primitives |
|---|---:|---:|---:|---:|
| 1,221 parameters | 372/384 | 85/96 | 36/36 | 972/972 |
| 50,013 parameters | **384/384** | **96/96** | **36/36** | **972/972** |

The 50k arm receives 1,419 teacher interventions across all three runs; the small
arm receives 1,434. Each has 720 subsequent training episodes without any hints.
The teacher remains available in historical replay receipts, as other past
experience does; no teacher selects evaluation actions or new autonomous-phase
actions. Goal completion, physical plans and primitive outputs are independently
checked rather than accepted from the native scorer.

Seed 2's small model scores 122/128 final and 30/36 recovery. Its six final
failures repeatedly ask an unavailable sensor. Here the value table prefers
retrieval but the MLP chooses asking, unlike seed 1's value-estimation failure.
At that exact observation, the small model experienced four retrieval decisions
and no teacher hints. The separate diagnostics retain both failure modes.

The matched no-guidance comparison is now complete for all three seeds. The small
controller without hints scores 354/384 final, 66/96 recovery and 24/36 larger;
with hints it scores 372/384, 85/96 and 36/36. It improves seed 0 and regresses on
seeds 1 and 2. The 50k controller passes every endpoint both with and without
guidance. `paired_summary.json` retains all six pairs. Full-schedule capacity
and expanded Agkern checks are both complete. The 50k MLP with TD and
the original Agkern size is now recommended for subsequent capability experiments;
the historical CLI default remains unchanged. There is no demonstrated endpoint
benefit from hints for the 50k controller on these tasks: it passes with or without
them. Fading guidance remains an available bootstrap for harder skills, not a
required part of the recommended 50k profile. Primitive instruction demonstrations, wrong-skill repair and broader
representation growth remain beyond this screen.

The [goal-interruption follow-up](INTERRUPTED_GOALS_EXPERIMENT.md) adds suspended
requests and separates observed action segments from terminal goal labels.
Its three-seed trained-policy test passes all 108 parent and 144 child goals, while a stronger
delayed-credit contract exposes slow backward value propagation after intervening
learning. The explicit episode-return comparison regresses on the full task
screen despite passing that credit contract; the existing TD profile remains
recommended. Host-scheduled suspension is working memory machinery, not evidence
that goal scheduling itself has been learned.

Reproduce the guidance and replay checks with a separately named binary:

```sh
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/laminar_train.cpp cpp/src/organism.cpp -o build-wsl/laminar_train_guidance_rerun
python3 python/check_guidance_checkpoint.py --binary build-wsl/laminar_train_guidance_rerun --out logs/guidance_checkpoint_rerun
python3 python/warm_start_experiment.py --binary build-wsl/laminar_train_guidance_rerun --root logs/warm_start_screen_rerun --source logs/capacity_screen
python3 python/summarize_warm_start.py --warm logs/warm_start_screen_rerun
```

The warm experiment requires the capacity screen's generated data and the two
capacity primitive-verifier executables. It changes only the explicit `GUIDANCE`
declaration; the worlds, tasks, labels, prices and evaluation cases are matched.
Sources and hashes are preserved alongside all successful and failed outcomes.

Warm starts change the allowed training route. They do not remove the four
[capability requirements](FINAL_CAPABILITIES_VALIDATION.md): the model must still
learn control, handle delayed credit, extend its representations and retain skills
in one continuing session. Success would establish a scaffolded learner that can
continue learning independently; it would not establish development from an
uninformed initialization.
