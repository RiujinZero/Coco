# Learned control while primitives and concepts are acquired

This follow-up tests the neural controller on the same continuing model that
acquires primitive skills, computational language, relational concepts, action
effects and physical procedures. Unlike the earlier computational-session
contract test, the evaluator does not force any operation. The controller
chooses retrieve, reason, ask, act and finish from public observations, terminal
correctness and declared costs.

Across three seeds, the primary arm passes **384/384 final standard cases** and
**89/89 recovery cases**, while acquiring the components in one continuing
model. It passes **30/36 larger goals**. The optional size-input arm also passes
all standard/recovery cases and learns to continue reasoning, but passes only
**21/36 larger goals**. Neither result completes the four broad gates.

The matched [control-transfer follow-up](CONTROL_TRANSFER_DIAGNOSIS.md) retains
search allowance across unchanged-scene revisions and raises the frozen baseline
to **36/36 larger goals**, preserving its standard/recovery scores. Replacing
neural inference with trees fitted to its learned preferences instead worsens
performance. Both results and their limits are recorded separately.

## Results

| Controller arm | Final /384 | Three-failure recovery /89 | Larger goals /36 |
|---|---:|---:|---:|
| Empirical TD → neural preferences | 384 | 89 | 30 |
| Frozen controller | 0 | 0 | 0 |
| Shuffled controller reward | 0 | 0 | 0 |
| Credit only at final decision | 77 | 0 | 0 |
| Added public size inputs | 384 | 89 | 21 |
| Size inputs, frozen controller | 0 | 0 | 0 |

All 18 runs pass the independent trace audit. The primary arm's final results
include 240/240 queries and 144/144 physical goals. Intermediate primary
checkpoints are 90/96, 192/192, 288/288 and 384/384; the size-input arm scores
86/96, 190/192, 284/288 and 384/384. The final result does not erase early failures.

Removing retrieval reduces primary final success to 288/384; removing asking
reduces it to 192/384; removing action leaves 295/384; removing reasoning or
finishing yields 0/384. The action-removal survivors are queries and initially
satisfied goals. These tests reuse the same final fixtures; recovery fixtures
are also variants of those cases, not 89 independent fresh worlds.

The primary arm **never chooses further reasoning while an exhausted-search
status persists** in final/recovery/larger evaluation. It instead uses supplied
motor exploration, then reasons again after observing the result. This shortcut
works on the standard task sizes. Each of its six larger-goal failures exhausts
all 32 motor attempts, then continues selecting rejected actions until timeout.
Every such case has 33 reasoning decisions and 95 act decisions in total.

The size-input arm makes 91 further-reasoning choices in standard cases, 304 in
recovery and 271 in larger goals, with no act-after-search-exhaustion decisions
in those groups. It therefore supplies evidence that the controller can learn
to spend more computation before acting. However, it also makes 1,839 information
requests with an available plan in the larger-goal group. In seed 2, all twelve
larger cases complete their search, then loop on retrieval. Size-input larger
scores are 12/12, 9/12 and 0/12, versus the primary arm's 10/12, 11/12 and 9/12.
The extra inputs are not promoted to the default configuration.

The final primitive weight files in all six arms, for every seed, are identical
to the corresponding independently tested computational-session checkpoint and
unchanged by evaluation. Thus these control failures coexist with retained
primitive skills; they are not evidence that the learner forgot counting.
The byte comparisons and reference hashes are recorded in
`logs/computational_control/primitive_retention_comparison.json`.

## One interleaved learning stream

Each seed receives 1,600 controller-training episodes, divided into four stages.
External teaching contains raw facts and verified final answers, not primitive
instruction traces, language programs, task-specific plans or next-operation
labels. The evaluator's intended meanings and world mechanics are stored in
separate private files.

1. Equality and distinct-count lessons arrive, together with count-comparison
   language and an observed addition effect. The controller has to choose
   `reason` to compile this evidence and acquire the primitives before using them.
2. Following lessons and two new computational language forms arrive. Earlier
   count tasks continue alongside endpoint comparison and follow-then-count tasks.
3. A binary `eligible` concept learns the conjunction of two observed relations,
   and a marking action learns from an observed transition. Neural counting over
   that concept becomes necessary for new queries and physical goals.
4. A second concept adds another condition, allowing reuse of the previous
   concept, and another observed action supplies that condition. All earlier
   task families remain in the training and evaluation mix.

The primitive curriculum is the 40-lessons-per-skill curriculum from
`COMPUTATIONAL_SESSION_EXPERIMENT.md`; each of five language/relationship
combinations receives 48 raw language examples. Concept names and query argument
types are supplied; their definitions and computational programs are learned.
No learner can access the evaluator's family identifiers or intended definitions.

Training worlds use 1–4 objects, with up to two missing goal facts. Information
can be complete, empty, partial or missing its completeness guarantee. Some
memory reads are stale; some sensing attempts are unavailable. Retrieval and
sensing costs swap. A quarter of physical training episodes fail their first
motor attempt. Own action chains receive only terminal satisfaction feedback;
execution acknowledgements say whether the motor actually ran, not whether the
goal was achieved.

Frozen checkpoints test all acquired families. Final evaluation has 128 cases
per seed: 80 queries and 48 physical goals, spanning both prices and four
visibility conditions. Five matched operation-removal sets reuse those cases.
Recovery tests reuse initially unsatisfied final goals with the first three
motor attempts failing. Twelve fresh larger goals per seed use 8 or 12 objects
and three missing facts, beyond the physical training range. These are bounded
structured tasks with explicit action affordances and complete action-result
observations, not unrestricted dialogue or real-world robotics.

## Learning and controls

The primary controller uses sampled empirical TD values and backprop fitting of
neural operation preferences. Its neural network selects every evaluation
operation. Training uses 128 credit updates per episode, exploration of 0.4 for
the first 200 episodes and 0.15 afterward, and 64 consolidation passes over its
distinct experienced public states every 200 episodes. Training has a
48-decision cap; evaluation has 128. A reactive body permits up to 32 single
motor attempts, with a fresh decision after every observation.

The frozen-control, shuffled-controller-reward and final-decision-only controls
receive the same external curriculum. Their own trajectories and resulting
evidence differ. Freezing the controller does not freeze semantic teaching or
primitive acquisition from training evidence; it isolates operation-policy
learning. Shuffling controller reward does not falsify the verified world labels
used by the symbolic learner.

Primitive policies are still found by supplied whole-policy search over final
answers and transferred through self-generated local-plasticity rehearsal.
Concept, program and action-effect induction are symbolic. Program search,
physical planning and motor exploration remain supplied algorithms. Repeated
reasoning can expand the search allowance, but the controller must choose it.
Inference microinstructions are charged at the public prediction-work rate;
rebuild pricing remains coarse and does not reflect the full training compute.

An additional `scaled` arm exposes the logarithms of the number of offered
actions and currently observed facts in the two previously unused controller
inputs. These are public quantities, not hidden world state or answers. It has
the same episode count, but more distinct states and therefore more
consolidation updates. It is not an equal-compute causal comparison. The option
is disabled by default; its matched frozen-control arm is also tested.

## Integration changes

Feedback can arrive before the first chosen reasoning step. Previously that
feedback could be stored as an ordinary language frame even though raw teaching
had declared computational mention types. Rebuild would then create competing
parsers. Receipt routing now uses uniquely matching patterns from current raw
language evidence, independently of whether semantic compilation has happened.
A regression covers this early-feedback path.

Rebuild now reuses an existing computational program only after running it with
the current neural skills against **every current lesson** of that frame. Failed
validation invokes fresh search; primitive edits still invalidate the old
programs. This avoids repeating full program discovery after every new terminal
receipt without treating a cached program as permanent truth.

## Development evidence and verification scope

The retained 200-episode pilot (`logs/computational_control_pilot`) passed 94/128
final cases: 80/80 queries and 14/48 physical goals. It passed 0/32 recovery and
0/12 larger goals. Its independently checked trace showed long repeated-read
loops. The full run uses the declared 1,600-episode budget, not a changed test
threshold or an omitted failing family. The pilot and full datasets are generated
with different episode counts, so their scores are not paired on identical worlds.

The independent checker reconstructs every public controller observation and
charged cost, verifies evaluation choices against the saved neural network,
replays physical effects and read failures, recomputes terminal outcomes, and
reconstructs the evidence inventory. It validates learned relational definitions,
computational programs against their raw language evidence, physical procedures
against their own execution witnesses, and primitive checkpoint stability.
It does not independently rerun every evolutionary proposal or reproduce every
backprop update. The preceding primitive VM audit is rerun separately after the
integration changes.

That regression passes 1,008/1,008 computational questions, 120/120 physical
planning goals and 648/648 retained primitive cases at
`logs/computational_session_control/audit.json`. The earlier symbolic session
also retains 2,304/2,304 queries, 144/144 goals and 72/72 tight-budget transfers at
`logs/relational_session_control/audit.json`. New early-feedback and public-size
observation contracts, plus existing controller contracts, pass ASAN/UBSAN.

The baseline source was copied to `logs/computational_control/baseline_source`
before adding the optional size inputs. The size-input build has its own source
snapshot and executable, so the ongoing baseline sweep keeps its original build.
Both manifests are checked by the final audit. Reproduction commands:

```sh
bash scripts/computational_control.sh
bash scripts/computational_control_scale.sh
python3 python/check_relational_control.py --root logs/computational_control \
  --modes learned frozen shuffled terminal scaled scaled_frozen
```

## Remaining work

The larger-goal failures distinguish two remaining control problems: choosing
exploration instead of further search, and failing to act after a valid larger
plan has been computed. Adding public size signals changes that behavior but
does not resolve transfer. Further tests should distinguish learned value
estimation from neural preference generalization, and examine why a new
observation stamp resets the search allowance even when a motor leaves the
observed world unchanged. Those are follow-up hypotheses, not proven fixes.

This stream does not yet demonstrate automatic repair of
a primitive from a failed composed outcome, invention of new primitive
operations or concept signatures, or the complete learner described by all four
gates in `FINAL_CAPABILITIES_VALIDATION.md`.
