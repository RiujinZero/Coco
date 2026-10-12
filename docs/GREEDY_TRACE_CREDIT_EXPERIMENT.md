# Goal-scoped greedy trace credit

The consolidated [validation record](GREEDY_TRACE_VALIDATION.json) checks the
completed contracts, task screen, ablation boundaries and default parity, and
preserves the limits of the longer-task preflight. The proposed next step is
[learner-controlled goals and mission credit](LEARNED_GOAL_CONTROL_NEXT.md).

The original one-step empirical TD rule does not pass the stronger suspended-goal
credit contract at 64 parent updates. Full episode returns pass that contract but
regress the matched task screen. A subsequent replay-priority candidate also
failed the contract. Those results remain recorded in
`INTERRUPTED_GOALS_EXPERIMENT.md`; the recommended TD profile is unchanged.

The new opt-in `controller_trace` and `controller50k_trace` profiles use a
Watkins-style multi-step target over the completed goal's own recorded decisions.
Starting from a chosen decision, the target subtracts its observed cost and walks
forward through the goal, discounting by 0.99. It continues through each subsequent
action only while that action is greedy under the current empirical values for
that public observation and its available actions. Tied actions qualify. At the
first nongreedy action it stops and bootstraps from that state's best available
value. If there is no cutoff, it reaches the observed terminal reward.

This preserves propagation through an initially unvalued successful sequence,
while avoiding assigning the known cost or failure of a subsequent exploratory
choice to an earlier choice that could instead have taken a better continuation.
It does not prove this is the cause of the full-return screen's regressions.
The target is recomputed for each logged replay update, using the same table
learning rate, neural preference fitting, replay order and budget as TD. It uses
no operation-teacher label, hidden world, intended route, goal name or child
reward. Suspend/resume scheduling is still supplied by the caller.

The final optional constructor argument is `greedy_trace_credit=false`, following
`episodic_return`. It requires empirical, non-model-based credit and cannot be
combined with episode returns. Omitted/false preserves the existing TD branch.
Both profile names and credit semantics are fingerprinted in the trainer config.

## Contract evidence

`cpp/tests/test_greedy_trace_credit.cpp` verifies:

- Targets against an independently implemented forward discounted sum and first
  nongreedy cutoff; production uses a backward recursion.
- Successful complete returns, failure cutoffs, repeated goal replay, and the
  terminal-only negative control.
- Exact empirical-value and neural-weight restoration after correcting a goal's
  terminal outcome and then restoring it.
- Physical parent goals interrupted by 16 or 32 unrelated learning goals with
  alternating signed outcomes, with separate retained parent traces.
- Positive terminal credit reaching the first parent choice in 6- and
  46-decision parent traces at the same 64-update terminal budget. The longer
  contract includes 40 supplied extra reasoning decisions, not a claim of
  autonomously learned long-horizon behavior.
- Explicit false versus default TD equality and invalid mode rejection.

The optimized and ASAN/UBSAN contracts passed. The pre-existing episode-return
and goal-suspension tests also passed against the new controller header. The
46-decision case tests delayed propagation accounting: its 40 extra decisions
repeat reasoning after the physical goal is already satisfied, so they are not
46 distinct causal steps toward the goal.

The optimized/sanitized results and binary/source hashes are preserved in
`logs/greedy_trace_credit_contract_v1/validation.json`.
The exact original suspension fixture also passes after changing only its
constructor's episode-return selection to greedy-trace selection, including all
16 successful intervening goals. Its derived source, output and hashes are saved
at `logs/greedy_trace_exact_suspension_v1/validation.json`.

## Matched task screen

The completed three-seed campaign at `logs/greedy_trace_screen_v1/acceptance.json`
uses the archived interrupted-goal input streams unchanged. All nine arms were
independently audited; every seed and arm is retained. TD retains exact archival
parity across 15 behavior/configuration artifacts. The declared candidate
acceptance check passes.

| Credit | Final short tasks | Recovery | Larger worlds | Interrupted parents | Child questions | Primitive retention |
|---|---:|---:|---:|---:|---:|---:|
| TD | 384/384 | 96/96 | 36/36 | 108/108 | 144/144 | 972/972 |
| Episode return | 352/384 | 89/96 | 36/36 | 108/108 | 134/144 | 972/972 |
| Greedy trace | 384/384 | 96/96 | 36/36 | 108/108 | 144/144 | 972/972 |

The greedy trace candidate therefore passes the stronger credit contract without
the observed episode-return task regression in this screen. This is a concrete
resolution of that specific tradeoff. It does not close the full credit gate:
the larger-world tasks require three missing physical effects, interruptions are
scheduled by the host, and the scored interrupted goals are frozen evaluations.
Acquiring a new longer causal strategy from terminal feedback remains a separate
requirement.

## Longer causal acquisition and execution limits

`python/long_credit_experiment.py` preserves the exact 400-episode warm start and
then constructs tasks requiring four, five and six distinct motor effects during
adaptation, with seven and eight held out. Each goal includes two already
satisfied objects and an irrelevant valid motor. The independent semantic
generator verifies that every required effect is necessary and the complete set
is sufficient. New cases use fresh entity names. Later evaluation repeats the
original 128 short-task cases to measure retention. No additional operation
hints or intermediate correctness labels are introduced.

A separate five-case frozen preflight tests causal lengths four through eight
on one base family before the expensive adaptation campaign. Both protocols keep
the canonical 48 training decisions, 128 evaluation decisions, 32 motor attempts,
planner depth eight and 4,096 planner states. Larger causal horizons must not be
scored as credit failures when the supplied execution machinery cannot express
them. All native timeouts and partial artifacts are retained. Per-case reporting
includes family, causal length, actual decisions/actions, planner statuses,
computation cost, discounted observed return, and whether that return beats
immediate failure at -1. Successful but sufficiently expensive planning can be
worse under the stated objective; the experiment keeps and reports those costs.

The completed seed-0 preflight at `logs/long_credit_preflight_v1/acceptance.json`
passes all five cases for both frozen policies. It establishes execution
feasibility in this one family, and also reveals two important limits:

| Required effects | TD decisions | Greedy-trace decisions | TD discounted return | Greedy-trace discounted return |
|---|---:|---:|---:|---:|
| 4 | 22 | 22 | 0.562 | 0.558 |
| 5 | 33 | 34 | 0.313 | 0.297 |
| 6 | 44 | 44 | -0.020 | -0.029 |
| 7 | 58 | 58 | -0.732 | -0.745 |
| 8 | 71 | 71 | -1.838 | -1.851 |

Seven and eight effects fit the evaluation limit but exceed 48 training
decisions. Eight-effect success costs 2.899/2.919, so its observed discounted
return is worse than immediate failure at -1 under the unchanged objective.
Moreover, both policies already solve these five cases before adaptation; merely
passing them afterward cannot establish acquisition of a missing strategy.
The prepared acquisition stream is therefore a proposed continuation, not a
completed longer-credit capability claim. A decisive next curriculum must expose
an initially missing behavior and account for the actual reward/cost objective.

The 32-feature feedforward controller also has explicit information limits.
Plan length is capped at four, remaining motor allowance is represented only as
exhausted/not exhausted, default fact/action counts are zero, and the number or
identity of suspended goals is absent. Suspend/resume are caller API operations,
not selectable controller actions. These omissions limit what an operation policy
can condition on. An observed failure still needs a trace-level diagnosis before
calling it state aliasing; a missing feature alone does not prove incompatible
optimal choices at an otherwise identical full observation.

## Matched credit ablations

Canonical whole-stream profiles `_frozen`, `_shuffled` and `_terminal` keep their
declared controller shape but change controller learning from the start. They are
useful negative controls, but do not isolate adaptation after an identical warm
start. The added `_frozen_after400` and `_shuffled_after400` variants of
`controller50k` and `controller50k_trace` keep the first 400 episodes, including
episode 400's consolidation, unchanged. The ablation starts at episode 401.
The boundary is fingerprinted as `control_ablation_after=400`.

Freezing suppresses controller credit updates and consolidation while ordinary
verified evidence and other learning continue. The trainer exports controller
weights and empirical values at the boundary. The independent auditor requires
all subsequent frozen checkpoints and final weights to match it, and explicitly
checks the stage-4 boundary when available. The longer-task driver additionally
compares pre-ablation stage-4 weights, values and primitive weights against the
corresponding ordinary arm. No terminal-only phase change is introduced: changing
the meaning of old replay entries would require a separate design.

`cpp/tests/test_credit_phase_controls.cpp` passes optimized and ASAN/UBSAN checks
of exact pre-boundary state matching, boundary activation, post-boundary freezing,
shuffled outcomes, continued evidence and configuration fingerprints.
`python/check_credit_phase_controls.py` provides an actual file-run smoke with
400 repeated short fixtures, eight follow-up episodes and independent audits.
That fixture is an accounting/wiring check, not a long-horizon learning score.
All six TD/greedy-trace baseline, frozen-after-400 and shuffled-after-400 arms
completed and passed independent audits in `logs/credit_phase_controls_v3`.
Every ablation's stage-4 controller weights, empirical values and primitive
weights exactly match its corresponding ordinary arm. Both frozen arms retain
exact controller weights and values after the eight follow-up episodes.
The complete result is `logs/credit_phase_controls_v3/audit.json`; the optimized
and sanitizer unit receipts are `logs/credit_phase_contract_v1/validation.json`.
