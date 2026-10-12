# Learned control of the continuing relational session

This records the initial one-batch baseline and its archived source version.
The [reactive follow-up](RELATIONAL_REACTIVE_EXPERIMENT.md) adds repeated action
and observation, motor failures, unavailable sensing and additional credit rules.

The operation controller now learns alongside the relational session's language,
concepts, action effects and own-execution procedures. This remains partial:
the stronger tested rule passes **308/384 fresh final cases**, comprising
**192/192 queries and 116/192 action goals**. Reliable action control, useful
retrieval, retention at every checkpoint and longer skill acquisition remain open.

## Mechanism

`ControlledRelationalSession` contains one `RelationalSession` and a neural
controller with 32 inputs, 32 hidden units and five outputs: retrieve, reason,
ask, act and finish. The neural network makes every evaluation choice. Inputs
contain public completeness, model readiness, computed answer/plan status, read
failures, whether a motor batch has been issued, budget and prices. The host
supplies answer-versus-physical-result intent. No correct next operation, family
ID, solution trace or hidden answer enters the policy.

Only a paid `reason` computes truth, rebuilds dirty knowledge or plans an
unsatisfied physical request. Each search exhaustion doubles the allowance from
four up to 32,768 predictions. Search restarts; it does not resume a frontier.
Repeated predictions are charged again. Asking and retrieval return timestamped
observations. Stale memory cannot replace a newer motor result; changed worlds
invalidate cached answers and plans. `finish` only reports cached truth.

`act` executes a computed ticket or uses the existing supplied low-support/random
exploration when no ticket exists. There is **one motor batch per episode**.
A failed exploratory action cannot be followed by another batch before terminal
feedback. All five operations remain selectable, including invalid attempts,
except in explicit removal controls.

At episode end, verified correctness trains the controller; a separate verified
satisfaction bit labels the observed world for language/procedures. Own action
observations also teach effects. This distinction prevents an incorrect report
from becoming a false world label. Only complete observed worlds add terminal
language receipts.

The stronger arm uses **tabular TD credit plus backprop policy fitting**. The
table keys are the same public observations. Experienced costs, successor states
and terminal outcomes update values (discount 0.99, value step 0.2). Their
preferred operation supplies a self-generated one-hot regression target for the
neural policy (step 0.002). The table never selects live operations. This is a
hybrid rule, not end-to-end backprop or local-only plasticity. The state
abstraction, TD update, symbolic induction, search/pruning and exploration remain
supplied mechanisms.

The direct comparator uses neural Double-Q credit with a delayed target network,
step 0.001 and target copies every 256 updates. Each arm gets 128 replay updates
per episode: first the latest trace backwards, then sampled old receipts. The
stronger arm fits all five outputs per update, so floating-point compute is not
matched. Old credit can be corrected by rebuilding weights from initialization
with the exact saved update schedule, including table/target-network state.

## Protocol and reproduction

Run `bash scripts/relational_control.sh` in WSL. It runs ASAN/UBSAN contracts,
generates data, executes five arms over three seeds and independently checks
`logs/relational_control_empirical/audit.json`.

Each seed has eight stages and 1,600 training episodes, interleaving new concepts,
language and effects with old-family requests. External teaching provides 432
concept, 864 language and 16 transition receipts; further evidence and procedures
come from the learner's own attempts. Evaluation uses a frozen clone and fresh
worlds/entity names, with full/partial visibility and missing/partial/fresh/stale
memory. Training removes up to two facts; tests include removal of three. Action
names are opaque and menus contain distractors. This generated structured-language
test is separate from the user's `/home/user/curriculum_v2`, not a chat test.

Final checkpoints have 128 cases: eight families × two intents × two price
regimes × four visibility cases. Another 640 reuse those fixtures with one
operation disabled. Training allows 48 decisions, evaluation 128. Accounting
prices, not CPU timings, are: reason/rebuild 0.005 each, motor 0.02 each,
prediction 0.00001 each, retrieve/ask (0.01, 0.06) or (0.07, 0.01), finish zero.

## Results

| Arm | Seed 0 | Seed 1 | Seed 2 | Final total |
|---|---:|---:|---:|---:|
| Empirical TD + neural policy fitting | 128/128 | 104/128 | 76/128 | **308/384** |
| Direct neural Double-Q credit | 47/128 | 37/128 | 19/128 | 103/384 |
| Frozen controller weights | 0/128 | 0/128 | 0/128 | 0/384 |
| Shuffled controller outcome credit | 0/128 | 0/128 | 1/128 | 1/384 |
| Updates restricted to final decisions | 66/128 | 49/128 | 62/128 | 177/384 |

Frozen weights still accept truthful symbolic evidence. Shuffling affects only
controller correctness rewards, keeping symbolic satisfaction truthful. The
terminal-only arm uses empirical credit/distillation on each trace's final
decision. Exogenous streams match; encountered trajectories and self-generated
receipts differ because policies act differently.

All 192 final queries pass without motor actions. Physical goals pass 64/64,
40/64 and 12/64. Of 76 failures, 71 time out and five finish incorrectly.
Failures repeatedly ask or attempt unavailable actuation; some act before a
sufficient plan exists. The complete first-seed checkpoint uses at most 16
controller decisions and three motor actions per case. Final symbolic models
have 35–54 supported procedures and pass independent semantic/evidence checks.

The stronger arm passes 1,420/1,728 checkpoint cases during acquisition, but
performance is not stable throughout learning:

| Seed | Stage 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16/16 | 32/32 | 48/48 | 64/64 | 63/80 | 95/96 | 86/112 | 128/128 |
| 1 | 16/16 | 30/32 | 48/48 | 64/64 | 65/80 | 92/96 | 88/112 | 104/128 |
| 2 | 12/16 | 28/32 | 35/48 | 42/64 | 50/80 | 56/96 | 82/112 | 76/128 |

Removing reason or finish yields 0/384, ask 114/384, act 231/384.
Removing retrieve leaves **308/384 unchanged**. Seed 0 never retrieves in its
final tests. Success therefore does not establish all five learned operation
choices or cost-sensitive information acquisition. Full costs, loops and
per-family results remain in the audit.

Earlier failures are preserved: online neural TD scored 40/128 in
`logs/relational_control_online_td/0`; delayed Double-Q scored 47/128 on the same
data and 55/128 with 500 episodes per stage in `logs/relational_control_longer/0`.
Their original headers/probes and raw traces remain available. These older
diagnostic pilots lack the later plan/journal logging; the full audit covers
the new three-seed comparator and other arms.

## Verification and remaining gaps

The checker reconstructs every public observation, checks frozen neural choices,
timestamps/cache invalidation, independently executes physics and query meanings,
verifies terminal credit, plan satisfaction and cost/budget accounting, and
reconstructs accepted evidence. Ready checkpoints are checked against raw
receipts and intended concept/language/effect semantics. It verifies published
prediction counts against costs/bounds, not every internal search expansion.
It records failures rather than requiring a passing score. Source/data hashes
are included in the report.

ASAN/UBSAN tests cover explicit reasoning, source/stale guards, coupled credit,
evaluation isolation and old-outcome correction. Restoring old feedback restores
bit-identical neural weights in both learning modes. The full curriculum does
not interleave erroneous-feedback corrections; that contract is unit-tested.
The prior relational-session three-seed regression and audit also pass after
making receipt/procedure capacities configurable.

Next priorities are stable operation preferences, recovery after bad actions,
useful retrieval and repeated act/observe/reason cycles. The previously discovered
follow/equality/count neural skills remain separate. This does not establish a
complete learner or unrestricted long-sequence credit. Backprop is allowed for
the initial complete learner; the same behavioral gates apply to the later
demonstration with no backprop learning.
