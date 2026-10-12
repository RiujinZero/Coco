# Reasoning primitives: execution works; learning is still the obstacle

Tested 2026-09-22. This is an isolated experiment, not a new reasoning capability
integrated into the Coco organism.

Later result (2026-09-23): [outcome-based primitive discovery](PRIMITIVE_DISCOVERY_EXPERIMENT.md)
finds the missing counting loop using whole-policy search against final answers,
then transfers and retains it in the native neural skill policy. It preserves
the negative results below; direct step-level TD is still not the successful
mechanism. The follow-up also removes external initial-skill demonstrations from
the native composition test.

Follow-up: [the composition experiment](REASONING_COMPOSITION_EXPERIMENT.md)
tests callable skills, scoped bindings, decision-specific credit and verified
procedure learning, including a native test without supplied operation graphs.

The proposed read–act–write loop can express the tested computations. A controller
fit to short action demonstrations executes longer problems perfectly. A controller
given only final-answer feedback learns some procedures, but does not discover
distinct counting. Coco's existing agency kernel, connected to the same operations,
does not learn the longer problems under either tested eligibility setting.

**The strong hypothesis is not confirmed:** supplying these primitives does not,
by itself, make Coco learn to reason. The experiment supports their usefulness as
an execution substrate and exposes a separate procedure-learning problem.

## Protocol

The test bypasses language understanding. Input is a structured question and a
shuffled collection of triples. The controller receives an opaque categorical
question kind, operand-validity flags, memory-exhaustion and set-membership flags,
whether the accumulator is zero, the result of an explicit comparison, and the
last operation. It does not receive the correct answer, graph depth, item count,
entity IDs, or a solution trace in the answer-only arms.

Every episode uses fresh entity names, including training episodes. Test names
are disjoint from training. Extra triples contain both irrelevant subjects and
irrelevant relations. Counting inputs repeat facts zero to two additional times;
the answer is the number of distinct matching objects, not the number of records.

Three tasks:

1. Follow a single-successor relation to its terminal entity.
2. Compare that terminal entity with the question's target.
3. Count the distinct values of the question's subject/relation pair.

The common VM offers ten instructions:

| Instruction | Effect |
|---|---|
| BIND | Put the query subject in the focus register; reset the read cursor. |
| READ | Read the next triple matching focus and query relation into the value register. |
| COPY | Copy value into focus and reset the read cursor. |
| MARK | Insert value into a scratch set. |
| INC | Increment a separate integer accumulator. |
| COMPARE | Compare focus with the query target; retain the Boolean result. |
| RETURN_FOCUS / RETURN_VALUE | Emit the corresponding entity. |
| RETURN_BOOL / RETURN_COUNT | Emit the Boolean result / integer accumulator. |

Operand validity is the only instruction mask. There are no question-specific
action menus. READ does not find an endpoint or traverse a whole chain. MARK does
not count; the policy must arrange MARK, INC and further READs itself. Numeric
and entity outputs are tagged so Python Boolean/integer equality cannot produce
accidental successes.

Training uses a size curriculum: first 1, then up to 2, then up to 3; counting also
includes zero. Each main run contains 60,000 mixed episodes, a budget of 48 VM
steps, and final exact-answer reward +1/-1. No intermediate correctness reward
is supplied. TD additionally uses a 0.01 step cost and discount 0.99, propagated
by reverse replay of local one-step updates after each episode.

Frozen tests contain 100 cases per task/size per seed: sizes 1, 2, 3, 4, 6, 8, 12,
plus zero for counting. The step budget is 128 to permit the longer procedures.
Learned weights are checked unchanged across evaluation. The table below reports
only the extrapolation sizes **4, 6, 8, 12**; full per-size results are retained.

## Results

| Controller/training | Seeds | Follow, size 4–12 | Compare, size 4–12 | Distinct count, size 4–12 |
|---|---:|---:|---:|---:|
| Tabular TD, 60k answer-only episodes | 5 | **88.2%** (1764/2000) | **95.6%** (1912/2000) | **0%** (0/2000) |
| TD with independent random rewards | 5 | 0% | 28.4% | 0% |
| TD, 300k answer-only episodes | 3 | 84.0% | 80.4% | 0% |
| Monte Carlo policy gradient, 60k answer-only episodes | 3 | 0% | 48.1% | 0% |
| Existing AgencyKernel, normal eligibility, 60k episodes | 3 | 0% | 0% | 0% |
| Existing AgencyKernel, longer eligibility, 60k episodes | 3 | 0% | 0% | 0% |
| Same tabular representation, 300 supplied action demonstrations | 5 | **100%** | **100%** | **100%** |

The TD result is not uniformly reliable. Follow accuracy across the five main
seeds is 100%, 100%, 100%, 46.5%, 94.5%. Compare is 94.75%, 95.5%, 94.25%,
93.5%, 100%. The longer training runs use the same seed IDs but stretch the
curriculum and exploration schedule; they are not continuation checkpoints, and
their lower score does not by itself establish training-induced degradation.

The random-reward control assigns an independent balanced +1/-1 reward; it is
not a permutation preserving the empirical success rate. Some failed policies
emit the wrong answer type or time out, so comparison accuracy can fall below
the approximately 50% Boolean-guessing baseline.

The demonstration arm is a **positive control**, not evidence of autonomous
procedure discovery. A handwritten teacher supplies short instruction sequences;
the table learns the state/action choices and is then evaluated without that
teacher. The same finite observation supports a correct general procedure, but
the experiment did not discover the teacher's procedures from answer reward.
The native VM's handwritten positive control also scores 2200/2200 on its full
test set, including all tasks and lengths.

## Which primitives are actually being used?

Disable one primitive after fitting the demonstration control, preserving its
learned policy. Results on lengths 4–12 are identical across the five seeds:

| Disabled primitive | Follow | Compare | Count |
|---|---:|---:|---:|
| COPY into persistent focus | 0% | 0% | 100% |
| MARK / scratch-set insertion | 100% | 100% | 0% |
| COMPARE | 100% | 0% | 100% |
| Cursor advancement between READs | 100% | 100% | 0% |

These are causal dependence checks on these policies, not proofs that every
possible reasoning architecture must use these exact instructions. Removing
COPY and retraining the answer-only TD controller also gives 0% on longer
follow problems in all five seeds. Removing COMPARE prevents Boolean output in
this deliberately typed instruction set; that ablation is an expressivity check.

The logs also preserve frozen ablations of the answer-only policies, retrained
ablations of MARK/COPY/COMPARE, and a one-instruction budget control. The latter
scores zero on all longer tasks.

## Direct test of Coco's agency kernel

`cpp/apps/reasoning_agency_probe.cpp` includes the existing
`cpp/include/laminar/agency.h`; it does not reimplement its learning rule. It uses
the same integerized training/test examples as the Python probe.

The adapter supplies the VM state as a 32-cell observation, ten tendencies,
one goal, three consequence channels, and the kernel's default 512-cell sparse
expansion. Each commitment is one instruction. Hysteresis, release suppression,
baseline cost and commitment thresholds are zeroed for this instruction interface;
max commitment duration is one. Invalid operands make the corresponding action
unavailable. Epsilon exploration falls from 0.4 to 0.04 during training. The
kernel receives verified final-answer reward after termination/timeout; it does
not receive teacher actions or intermediate correctness rewards.

Episode boundaries clear transient traces and competition state, retaining all
learned weights. The normal trace arm uses lifetime 8 and decay 0.85. The long
arm uses lifetime 128 and decay 0.99. Both allow up to 128 trace entries. There
were no missing or invalid actions in either arm. Test assertions verify that
consequence-route, sensory and drive-motif weights remain frozen.

This is an adapter experiment on the real kernel, **not** a whole-organism bAbI
benchmark or an exhaustive kernel tuning study. Its negative result means this
particular connection and feedback rule did not learn the procedures. It does
not establish that the kernel could never learn them.

## Interpretation and next decision

The experiment separates three questions:

- **Can the runtime express the solution?** Yes for these three task families:
  the demonstration control generalizes perfectly beyond its training lengths,
  and primitive ablations destroy the capabilities they support.
- **Can an answer-only learner discover some procedures?** Yes: the TD controller
  learns reusable relation walks and comparison in several seeds. No counting
  procedure was learned by either tested answer-only learner.
- **Can Coco's current agency rule learn them through this interface?** Not in
  these runs. Increasing eligibility lifetime alone did not resolve it.

There are important limits to the diagnosis. The observation representation is
supplied, question kinds are given, memory records are correct, graphs have no
cycles or ambiguous successor relations, and no language reading is required.
The finite abstraction is sufficient for a correct policy but is **not a full
Markov state for arbitrary erroneous programs**: different nonzero accumulator
values can share one observation. Consequently TD's failure cannot be attributed
solely to sparse reward. Exploration, state abstraction and credit assignment
remain competing explanations; the policy-gradient control also fails but does
not exhaust that design space.

The justified next experiment is to teach short execution skills with verified
local outcomes or action traces, then test whether Coco's local learner can
retain and compose them using final-answer feedback. Keep the structured-input
gate until counting reliably extrapolates across seeds. Only then connect this
runtime to the learned reader and measure extraction errors separately. This
bootstrap is a proposal: the present demonstration result does not show that
Coco's kernel can learn it or retain it during continued online training.

## Files and reproduction

- `python/reasoning_primitives_probe.py`: VM, data, TD and policy-gradient learners,
  controls, source/policy hashes, saved policies, answer-hiding checks and traces.
- `cpp/apps/reasoning_agency_probe.cpp`: native VM plus the existing agency kernel.
- `scripts/reasoning_primitives_agency.sh`: native build and six agency runs plus
  the native positive control.
- `logs/reasoning_primitives_final/results.json`: five-seed main result and controls.
- `logs/reasoning_primitives_long_final/results.json`: larger training budget.
- `logs/reasoning_primitives_pg_final/results.json`: policy-gradient control.
- `logs/reasoning_primitives_agency/*.json`: native results.

From the repository root (Python 3.12; native compiler GCC 15.2 via WSL):

```sh
python python/reasoning_primitives_probe.py --episodes 60000 --seeds 5 --test-per-size 100 --controls --out logs/reasoning_primitives_final
python python/reasoning_primitives_probe.py --episodes 300000 --seeds 3 --test-per-size 100 --out logs/reasoning_primitives_long_final
python python/reasoning_primitives_probe.py --episodes 60000 --seeds 3 --test-per-size 100 --method pg --out logs/reasoning_primitives_pg_final
python python/reasoning_primitives_probe.py --episodes 60000 --seeds 3 --test-per-size 100 --export-agency --out logs/reasoning_primitives_corpus
bash scripts/reasoning_primitives_agency.sh
```

On Windows, run the final command as
`wsl -d Ubuntu -- bash scripts/reasoning_primitives_agency.sh`.
Runtime defaults and production inference code were not changed.
