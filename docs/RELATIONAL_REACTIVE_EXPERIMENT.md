# Recovery and information gathering in the continuing learner

Coco's experimental relational controller can now act, observe, reason again
and retry within one task. Across three seeds, the neural controller passes
**354/384 final cases** (192/192 queries, 162/192 action goals) and **120/150
recovery cases** where the first three motor attempts fail. It passes **47/48
cases with unavailable sensing**. Removing retrieval reduces final success to
299/384, unlike the earlier baseline where retrieval removal had no effect.

This is partial progress toward the full learner. Thirty final cases still fail,
intermediate checkpoints remain uneven, and neural primitive-skill acquisition
is not integrated into this controller. The new dataset includes interruptions
and unavailable sensors; its score is not a matched numerical comparison with
the earlier 308/384 one-batch experiment.

## What changed

`ControlBody{true,1,32}` enables one motor step per act decision, up to 32
attempts per episode. The controller must choose again after every observation.
The entire issued/observed action chain receives one terminal satisfaction label;
no intermediate correct action or goal-success label is supplied. Public motor
acknowledgements distinguish execution from a failed attempt. Effects learn only
from executed transitions, while failed attempts remain in the evidence journal.
Procedures derived from a successful attempt omit its unexecuted motor steps.

Replanning previously could repeatedly issue an already completed prefix of a
cached procedure. Cached-plan validation now charges for simulating those steps
but omits actions whose predicted complete world is unchanged. This is sound
within the supplied unconditional-effect model; it is not a rule for operations
with hidden side effects. A regression checks the remaining action of a learned
two-step procedure after its first step has completed.

The old one-batch mode remains available as a control. Stale reads still cannot
overwrite newer observations. A new complete read that disagrees with the last
motor observation is rejected: this receipt format describes a contiguous own
action chain. Unrelated external world changes require cancel/restart rather than
being falsely attributed to the previous motor. Handling those interruptions
within an episode remains unfinished.

The neural controller retains the same 32 public inputs and five operations.
One previously unused input now reports exhaustion of the motor-attempt budget.
It still receives a supplied answer-versus-action intent. `reason` explicitly
computes answers, rebuilds knowledge and invokes bounded symbolic planning.
Repeated reasoning increases a supplied search budget; the search itself is
not learned. `act` uses a computed plan when present and otherwise invokes the
supplied actuator exploration. There is no automatic retry or forced reasoning
route: the learned policy selects every operation.

## Learning rules and controls

The **sampled TD → neural policy** arm uses the preceding empirical value table
and backprop preference fitting. Every 200 training episodes, 64 additional passes
fit the neural policy to its own learned operation preferences, using one
representative per observed training state. This consolidation has no evaluation
examples or externally supplied operation targets. Inference uses the neural
network throughout.

A second arm learns a **model of operation consequences** from the same public
observations, experienced costs and terminal outcomes. It accumulates counts of
successor states and mean immediate returns, then performs up to 256 Bellman
sweeps with discount 0.99. It selects from those learned values on known states
and uses its jointly trained neural policy on unseen states. This is a symbolic
control memory with a backprop-trained fallback, not pure backprop or an entirely
non-backprop learner. The generic estimator and value-iteration algorithm are
supplied; operation consequences and preferences come from experience.

The corresponding **model → neural policy** arm uses that estimator but makes
every inference choice with the neural network. Different policies collect
different endogenous traces, so differences between these arms do not isolate
readout error on identical trajectories. Saved values and weights allow direct
comparison of preferences on the same public states.

Frozen, shuffled-credit, final-decision-only and one-batch controls use the
consequence-model configuration. Frozen control learns no neural weights or
operation values but still receives truthful symbolic evidence. Shuffling affects
only control correctness credit; language/world labels remain truthful. The
final-decision control receives no earlier-decision updates. The one-batch control
can execute its whole plan once but cannot retry. These controls match exogenous
curricula, not experienced trajectories or exact compute.

All arms retain 128 sampled updates per training episode. Model arms additionally
update their consequence model once per episode. Consolidation counts depend on
the number of encountered states. Exact counts and parameter changes are logged;
these are not equal-compute comparisons.

## Curriculum and results

Run `bash scripts/relational_reactive.sh` in WSL. There are eight stages and 1,600
training episodes per seed, with the same growing concept/language/effect lessons
as the earlier joint experiment. A quarter of training physical requests have one
initial failed motor attempt. A quarter of training scenes have unavailable
sensing and a fresh retrievable world. No failure count or sensor availability
flag is supplied to the policy in advance.

The 384 fresh final cases include 48 initially empty, incomplete worlds with
unavailable sensing. Their memory supplies the required evidence. The separate
150 recovery cases replay unsatisfied physical requests from the final fixtures
with their first **three** motor attempts failing, beyond the zero/one failures
seen in training. This is not 150 additional independent world samples. Training
is capped at 48 controller decisions; evaluation permits 128.

| Arm | Final cases | Three-failure recovery |
|---|---:|---:|
| Sampled TD → neural policy | **354/384** | **120/150** |
| Learned consequence memory + neural fallback | 350/384 | 117/150 |
| Consequence model → neural policy | 315/384 | 88/150 |
| Frozen controller | 0/384 | 0/150 |
| Shuffled correctness credit | 75/384 | 16/150 |
| Updates only on final decisions | 172/384 | 13/150 |
| One physical batch per task | 279/384 | 0/150 |

The strongest aggregate neural arm scores 105/128, 123/128 and 126/128 on final
cases, and 27/50, 44/49 and 49/51 on recovery. Twenty-one of its thirty final
failures time out; nine terminate incorrectly. Successful recovery takes up to
60 decisions and nine motor attempts. The consequence-memory arm reaches up to
63 decisions on successful recovery. This shows longer control execution from
terminal credit, not acquisition of a new low-level primitive or arbitrary
long-horizon reasoning.

Retrieval is useful in this curriculum. In the neural arm, removing retrieve,
reason, ask, act or finish gives respectively **299, 0, 142, 234 and 0 successes
out of 384**. On the 96 initially empty final worlds, the first information request
is retrieval in 32/48 cases when retrieval is cheaper, and asking in all 48/48
cases when asking is cheaper. The remaining 16 cheap-retrieval cases ask first.
This is partial cost adaptation, not optimal information gathering.

Whole-model retention is still uneven. Correct counts at stages 1 through 8
(denominators 16, 32, 48, 64, 80, 96, 112, 128) are:

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 32 | 48 | 54 | 69 | 75 | 88 | 105 |
| 1 | 16 | 27 | 38 | 56 | 53 | 90 | 97 | 123 |
| 2 | 12 | 27 | 36 | 64 | 77 | 95 | 105 | 126 |

The same neural-arm sessions finish with 36, 70 and 78 supported procedures,
while language, effect and concept models remain consistent with their evidence.
Per-family results are retained in the audit rather than hidden in this table.

## Evidence and remaining work

The full seven-arm, three-seed run passes the independent checker in
`logs/relational_reactive/audit.json`. It reconstructs public inputs, verifies
neural/memory selections, timestamps, actual effects and failed executions,
terminal rewards, plan validity, budgets, costs and raw receipts. At ready
checkpoints it verifies concept/language/effect meanings and procedural witnesses.
For consequence-model arms it independently rebuilds empirical transition counts
and checks the published values against their Bellman equations, including exact
coverage of experienced model states. Evaluation changes neither weights nor
control values; symbolic receipt counts remain fixed. It does not replay every
internal symbolic-search expansion.

ASAN/UBSAN tests pass for repeated act/observe cycles, failed attempts, stale or
duplicate callbacks, external-world-change rejection, and terminal-only evidence
commit. Correcting and restoring old feedback restores bit-identical weights
after both TD and consequence-model learning, including consolidation. Replaying
fixed observed histories is not counterfactual regeneration of old decisions.

The earlier relational-session regression also passes after the procedure fix:
2,304/2,304 final language queries, 144/144 final goals and 72/72 tight-budget
transfers, versus 25/72 for matched search. Final goals now execute 269 actions
instead of 292 because completed cached steps are omitted. Results are in
`logs/relational_session_reactive/audit.json`; reproduce with
`bash scripts/relational_session.sh logs/relational_session_reactive`.

Earlier audited source versions are preserved in the `source/` directories and
`source_manifest.json` files under `logs/relational_control_empirical` and
`logs/relational_session`. Diagnostic pilot outputs are not substituted for the
full audited runs. The user's curriculum in WSL is unchanged.

Next work must address the remaining control failures rather than treating this
as completion. Candidate issues include overly coarse public state abstraction,
optimistic values for rarely sampled choices, and the mismatch between discounted
infinite-horizon value updates and an unobserved finite decision deadline. For
example, perpetual work costing 0.005 per step has discounted return -0.5 at
discount 0.99, better than a terminal failure reward of -1; this is a mathematical
incentive issue, not yet a verified explanation of every observed loop.

The neural follow/equality/count learner, byte reader and older fixed-prior archive
remain separate. New concepts still use a supplied relational logic and named
teaching targets. Broader representation growth, primitive discovery under delayed
credit, general external interruptions and stable learning of all components in
one model remain required. All four full capability gates stay open.
