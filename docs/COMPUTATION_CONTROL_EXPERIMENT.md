# Choosing operations from outcomes — 2026-09-23

One experimental policy learns to choose among retrieve, reason, ask, act and
return in a small structured environment. Both the existing local-plasticity
policy and a new backprop comparator solve 720/720 fresh test worlds across three
seeds, including paths up to 32 links after training on 0–3. Both still perform
unnecessary actions. The terminal-decision-only control also solves every world,
so this test does **not** establish the harder long-sequence credit capability.

This is an isolated controller test, not integration with the language reader,
procedure learner, physical-action learner or continuing byte trainer.

## What the policy receives

The world contains a hidden successor graph, a partially populated memory, a
working focus and a marker that a generic actuator can place. A request asks for
the endpoint or asks to place the marker there and then answer. The evaluator
knows the endpoint; the policy does not.

`ComputationStatus` exposes only the current row's public status (unread, visible
successor, closed empty, or failed retrieval), whether an effect is requested,
whether the observed marker equals the current focus, and five operation costs.
It contains no answer, hidden path length, route label, required-tool flag,
teacher action or solution graph. Entity bindings remain in the machine, outside
the controller's features. Renaming invariance is therefore supplied by this
adapter; the experiment does not learn to parse or encode arbitrary entities.

Retrieve reads one row from the incomplete archive. Ask requests that one row
from the environment; it never returns the endpoint directly. Reason copies a
visible successor into persistent focus and clears the consumed row. Act places
the marker at the current focus, whether or not that is useful. Return stops.
Only reason has an operand mask: a successor must be visible. Neither the correct
operation nor a correct stopping decision is enforced by the runtime.

Each policy trains for 30,000 mixed episodes with epsilon exploration, receiving
only terminal correctness (+1/−1) and actual operation costs. Earlier decisions
are updated in reverse order with their own bootstrapped successor-value targets,
discount 0.99. There are no intermediate correctness labels or demonstrations.
The controller's trace budget is 512; training allows 32 operations, while test
budgets scale with path length and reach 140.

The local arm uses the existing `LocalChoice`, fixed 512-unit sparse expansion,
consequence routes and three-factor updates. The backprop arm uses a 32–32–5 tanh
network with 1,221 parameters and squared TD error. Backprop updates both layers
within a decision; the bootstrapped target is held constant. It is not backprop
through the whole execution sequence. These are matched behavioral interfaces,
not matched parameter budgets or exhaustive optimizer comparisons.

## Tests, controls and costs

Each seed tests 240 worlds across lengths 0/1/4/12/32, both request types, cached
and uncached starting rows, complete and missing memory, and two price regimes.
When retrieval is cheap, it costs 0.01 and asking costs 0.05; the other regime
reverses those prices. Reason costs 0.005, act 0.02, return zero. Both regimes
appear during training; unseen numerical-price extrapolation is not tested.

| Training arm | Correct across three seeds | Total operation cost |
|---|---:|---:|
| Local updates, reverse TD | 720 / 720 | 176.19 |
| Backprop, reverse TD | 720 / 720 | 187.95 |
| Local updates, terminal decision only | 720 / 720 | 311.48 |
| Backprop, terminal decision only | 720 / 720 | 384.94 |
| Local updates, randomized outcomes | 365 / 720 | Not comparable: many incorrect answers |
| Backprop, randomized outcomes | 173 / 720 | Not comparable: many incorrect answers |
| Frozen local or backprop policy | 0 / 720 each | Always retrieves until timeout |

The frozen policies have zero output values, so their deterministic tie-break
selects retrieve. Their score is not a chance baseline. The randomized control
replaces each terminal reward with an independent random ±1; it does not shuffle
a preserved list of observed successes.

A fixed public-state reference policy costs 168.99 on these worlds: retrieve
first when cheap, ask after a miss, ask directly when asking is cheap, reason
through visible successors and act only if requested at the closed endpoint.
This reference is evaluator-only and is never used for training. It is a declared
comparison policy, not a proof of optimal expected cost over all possible worlds.

The learned reverse-TD policies acquire the price-sensitive information policy.
However, each local seed takes an unnecessary marker action in every answer-only
case, adding 2.40 per seed. Backprop seeds 1 and 2 do the same; seed 0 additionally
acts at intermediate positions, totaling 14.16 extra cost over the reference.
Perfect endpoint accuracy therefore conceals incomplete action suppression.

Frozen operation-removal arms disable retrieve, reason, ask or act in another
960 cases per seed/learner. An independent replay verifies all transitions and
outcomes, including required failures when reason is absent on nonzero paths or
act is absent for effect requests. Full per-family and per-price results, action
counts, decision tables and ablations are in `logs/computation_control/audit.json`.

The evaluator reconstructs every row read, focus move, marker change, cost and
terminal answer from raw case/decision logs. It verifies weight changes versus
initial weights; the native runner checks frozen evaluation and finite weights.
An ASAN/UBSAN finite-difference test checks every backprop parameter, with maximum
gradient error 4.86e−6, then checks loss descent and inference immutability.

## Interpretation and reproduction

This is positive evidence for **learned operation selection in this supplied
state/action abstraction**. The two policies use the same operations and outcomes;
the result does not show backprop is better, nor does it rule out a better
backprop configuration. The request semantics, row statuses, primitive effects,
memory addressing, exploration schedule and terminal evaluator remain supplied.

The terminal-only control is decisive for scope: it learns correct endpoint
behavior with far higher costs even when prior steps get no updates. Long
execution transfer here can emerge from local decisions in repeated states; it
is not proof of assigning terminal credit to an initially missing multi-step
skill. That control also omits nonterminal cost updates, so the cost difference
cannot isolate the benefit of outcome credit alone. The earlier distinct-count
discovery failure remains open. So do new
concept formation and learning all components together without forgetting.

Run `bash scripts/computation_control.sh` in WSL. It compiles the native experiment
and sanitizer gradient test, runs all eight arms across three seeds, and executes
the independent auditor. Source/binary/log hashes are recorded with the audit.
No curriculum files or production trainer defaults are changed.
