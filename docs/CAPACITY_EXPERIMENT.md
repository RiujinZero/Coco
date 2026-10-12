# Controller and Agkern capacity experiment

User request, 2026-09-23: test roughly 50k–100k parameters in both components.
The verified `coco_continual_v1` trainer remains the baseline. Experimental
profiles use the same executable, evidence lifecycle and training loop.

| Profile | Controller parameters | Agkern learning routes | Allocated Agkern routes |
|---|---:|---:|---:|
| `coco_continual_v1` | 1,221 | 5,120 | 15,360 |
| `controller50k` | 50,013 | 5,120 | 15,360 |
| `controller100k` | 100,021 | 5,120 | 15,360 |
| `agkern50k` | 1,221 | 50,000 | 50,000 |
| `agkern100k` | 1,221 | 100,000 | 100,000 |
| `both50k` | 50,013 | 50,000 | 50,000 |
| `both100k` | 100,021 | 100,000 | 100,000 |

The controller retains 32 inputs, one tanh hidden layer and five outputs. Hidden
widths 1,316 and 2,632 give `38*width+5` weights/biases. Its preference-learning
rate is scaled by `32/width` to control the increase in aggregate output updates
with width. This optimizer adjustment is explicit; the comparison is not an
unchanged-learning-rate ablation. The baseline rate stays 0.002.

Agkern uses 5,000 or 10,000 sparse expansion units, 16 winners and six fixed input
connections per unit. The skill adapter trains only one consequence channel;
expanded profiles allocate that channel, giving ten actions × units learning
routes. A native test verifies exact prediction, choice and plasticity equivalence
between one and three channels at the original 512 units. The fixed projection,
unused sensory-prediction allocation, controller value table, evidence and symbolic
memory are outside these trainable counts. This remains the current Agkern skill
adapter, not its complete competition/commitment loop.

## Matched screen

The initial screen uses seeds 0, 1 and 2 with 100 training episodes per stage
(400 total per run), shorter than the verified 1,600-episode baseline. This tests
learning speed before final scores saturate. All seven profiles receive identical
curriculum/environment files within a seed. The existing primitive curriculum
supplies verified final answers for following, equality and counting; each model
discovers and rehearses these skills. It supplies no external instruction trace
or controller action labels. This is a declared primitive warm start for mixed
control, rather than another missing-count acquisition experiment.

Measure intermediate checkpoints, final answers/actions, three-failure recovery,
larger goals, operation costs and 324 frozen primitive cases with duplicates,
distractors, new entities and sizes through 64. Independent audits reconstruct
controller inputs and neural choices, world physics, terminal outcomes and
primitive executions. Native parameter replay checks saved expanded Agkern
weights against all logged primitive states. Report wall time, updates and
inference instruction counts. Failures remain in `summary.json`.

Follow promising sizes with the full schedule and the forthcoming interrupted-goal
and delayed-credit curriculum. More weights cannot distinguish identical public
observations, supply missing feedback or expand a fixed operation bank by
themselves. This experiment tests capacity before addressing those other limits.

## Reproduce

```sh
bash scripts/train_coco.sh --check --out logs/canonical_capacity_regression_rerun
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/primitive_checkpoint_probe.cpp -o build-wsl/primitive_checkpoint_probe_capacity
g++ -std=c++20 -O3 -DNDEBUG -march=native -Icpp/include cpp/apps/verify_local_choices.cpp -o build-wsl/verify_local_choices_capacity
python3 python/capacity_experiment.py --root logs/capacity_screen_rerun --episodes 100 --seeds 0 1 2 --workers 4
```

Use a fresh root. The runner archives sources and hashes, invokes the real trainer
with `profile=...`, and preserves every result. Normal training keeps the original
profile. All 21 native runs and independent audits are now complete.

## Completed short-schedule comparison

All seven profiles and three seeds are independently audited. The complete
aggregate is `logs/capacity_screen/aggregate.json`; regenerate it with
`python3 python/summarize_capacity.py`. The screen executable is preserved as
`build-wsl/laminar_train_capacity_evidence`.

| Profile | Final | Recovery | Larger goals | Retained primitives |
|---|---:|---:|---:|---:|
| Baseline | 354/384 | 66/96 | 24/36 | 972/972 |
| `controller50k` | **384/384** | **96/96** | **36/36** | **972/972** |
| `controller100k` | 378/384 | 90/96 | 36/36 | 972/972 |
| `agkern50k` | 354/384 | 66/96 | 24/36 | 972/972 |
| `agkern100k` | 354/384 | 66/96 | 24/36 | 972/972 |
| `both50k` | 384/384 | 96/96 | 36/36 | 972/972 |
| `both100k` | 378/384 | 90/96 | 36/36 | 972/972 |

All 12 expanded-Agkern runs have byte-identical control traces, final controller
weights and stage-4 value tables to their corresponding original-Agkern controls
(`completed_agkern_parity.json`). Native wall times are roughly 200–244 seconds
with original Agkern, 1,020–1,083 seconds at 50k and 1,686–2,112 seconds at 100k.
These are concurrent runs, not isolated timing benchmarks. Extra Agkern capacity
adds cost without changing behavior on this supplied skill set.

The 50k controller is the strongest short-schedule size comparison; 100k is not
uniformly better. These controllers are all nonlinear MLPs; width changes, not
depth. A value-table diagnostic does not establish a simple preference-fitting
explanation for their score differences. The separate tapered-guidance comparison
also passes every endpoint with 50k, but does not improve its already perfect
no-guidance endpoint scores.

## Completed full-schedule follow-up

All six 1,600-episode runs are independently audited at
`logs/controller_capacity_full/aggregate.json`. Both the original 1,221-parameter
controller and the 50,013-parameter controller score **384/384 final, 89/89
recovery, 36/36 larger goals and 972/972 retained primitives** across three seeds.
Native wall times range from roughly 2,102 to 2,444 seconds in this concurrent
six-job run. More controller capacity improves the short-schedule robustness
measured above; it does not raise the saturated full-schedule endpoint here.

Use `profile=controller50k` as the current recommended profile for subsequent
capability experiments, with the original Agkern size, empirical TD credit and
the explicitly declared taper when guidance is used. The original
`coco_continual_v1` default remains available for exact historical reproduction.
No larger Agkern or alternative credit rule is promoted on the current evidence.
