# Public-root switch/resume feasibility preflight

This is an evaluator witness, not a training campaign. The six-edge raw fixture
already teaches the route meanings and edge effects from Boolean/world receipts.
No controller policy is warmed, trained, or credited here. The unassisted baseline
is the actual initial seven-output policy (zero output rows); its deficit must not
be described as failure after protocol bootstrap or as subsequent acquisition.

The frozen source declares three candidate-order and semantic-session seeds,
`7, 11, 19`, crossed with
urgent arrival clocks `5, 11, 17, 23`. Clocks count issued policy decisions. The
physical root requires six actual missing edge additions; the urgent public root
asks whether the first edge currently holds. Both use the same received world.
The host posts the urgent request at the declared clock with deadline `clock+12`;
it never chooses a destination, suspends, or resumes a goal. The physical deadline
is 96. Queries may correctly answer False. Independent graph-walk physics scores
the answer actually returned, and the physical root additionally requires actual
satisfaction. Every required root must be correct and on time.

All runs fix two roots, zero derived goals, 96 decisions and 32 motors. Prices are
`[.01,.005,.06,.01,0]` for retrieve/reason/ask/act/finish, `.00001` per prediction
and actual primitive VM step, `.02` per motor, `.005` per rebuild, `.002` per
next-goal and `.01` per activation. Unused preview prices remain
`[.00001,.0000001,.0000001,.005,.005]`. These are existing declared prices, fixed
before measurement; this preflight will report failure rather than retune them.

`reference_choice()` receives only the exact 32 floats and legal operation menu.
It has no session, goal ID, cursor, evaluator, action inventory, hidden clock, or
trace parameter. The harness reconstructs the legal menu from public validity and
shared budget metadata and checks exact equality with `choose()` on every row.
The reference notices new public requests, inspects the agenda, activates the
query, then resumes the physical request. Within execution it requests reasoning,
acts on a paid plan, and finishes a cached answer. There is no correct-goal label
in the input. This narrowly relies on publicly distinct action-intent and query
requests; it establishes nothing about two otherwise indistinguishable goals.

To dispatch the evaluator's choice through the same controller and token protocol,
the harness temporarily sets output preferences and restores the full policy
immediately. Every such row is marked evaluator assistance in the export. This is
not autonomous neural control, teacher data, a checkpoint, or a learned policy.
The initial-policy runs use untouched weights. Each seed rebuilds the same raw
lessons in its own semantic session so that action-exploration RNG is not cloned
from one fixed seed; all lesson interpretations are checked equal. Most runs
disable semantic and policy learning and check unchanged weights/model version,
empty new evidence/credit and zero updates on every row. An additional declared
reference set enables semantic learning to exercise real segment/model revisions,
but calls `finish_mission(0)` and checks zero policy gradients and unchanged
weights. It retains one terminal audit receipt in the temporary session, then
discards that session. Neither its receipt nor any reference decision is supplied
to a later learner, replay campaign or checkpoint.

The chronological JSONL exports all actual policy observations (decimal and exact
IEEE float bits), menus, operations, input requests, offered scenes and affordances,
callback observations, ownership metadata, terminal evaluations, prediction/VM
work and costs. Grouping exact observations plus menus records repeated and cross
context classes. A common successful feature-only function is a feasibility
witness over the visited support; its lack of contradictory decisions alone is
not a proof of global non-aliasing. The harness separately checks that each urgent
arrival during physical execution changes only public feature 30 and that a real
interruption and resume occurred.

Each reference run is compared with unassisted initial-policy behavior, immediate
unknown-answer failure and cycling with cheap next-goal until the shared deadline.
The latter is a checked delayed-failure strategy, not an artificially early failure
baseline. The utility is exactly
`G = .99^(T-1)*R - sum(.99^t*cost[t])`, with terminal reward +1/-1. In addition to
the strongest checked failure, compare the conservative upper bound
`-.99^95`: even a hypothetical failed 96-decision strategy with zero computation
cost cannot exceed it when every cost is nonnegative. Passing requires every
reference success and at least .10 margin above both comparisons. Immediate
failure may terminate before the scheduled arrival; its missing request still
makes the evaluator's mission fail. No future request is silently counted solved.

The app takes an output prefix and writes `_trace.jsonl`, `_aliases.jsonl` and
`_summary.json`. A source snapshot and SHA-256 manifest must accompany the measured
outputs. Source/config changes require a new evidence directory. This experiment
can justify a narrower public-root control campaign if it passes; the 12-effect
derived-goal family, post-bootstrap missing behavior, acquisition, shuffled-credit
and retained-task comparisons remain open.

## Frozen-source result, 2026-09-23

The measured v3 source passed: 12/12 frozen references and 12/12 references with
semantic updates completed both requests on time, using exactly six motors and
58–64 total decisions. All 24 arrivals changed only execution feature 30, all
triggered actual interruption/resume, and none used policy gradients. Arrival
clocks 5 and 11 interrupted planning before the first motor; clocks 17 and 23
interrupted after one acknowledged motor. Semantic versions changed from 48 to
50 or 51 in the learning controls; frozen versions stayed 48. There were zero
primitive VM inference steps in this relational fixture, which uses the learned
symbolic route/effect representation; this is not a neural byte-reader result.

The untouched initial policy solved 0/12 missions. This is a real initial deficit,
but the controller has not received protocol bootstrap, so it does not establish
a missing capability after warm-up. Candidate-order and semantic RNG seeds were
independent as declared; the two-root cyclic inspection and planned action paths
produced identical outcomes across these seeds. No claim of broad stochastic
exploration coverage follows from that seed count.

| Quantity | Measured result |
|---|---:|
| Frozen reference utility | .106740–.172258 |
| Semantic-learning reference utility | .099115–.167781 |
| Immediate-failure utility | -.992000 |
| Checked 96-step delayed-failure utility | -.508687 |
| Optimistic zero-cost failed-horizon bound | -.384896 |
| Worst success margin over checked failures | .607802 |
| Worst success margin over optimistic failure bound | .484011 |

The unchanged price configuration passes the declared .10 margin. Exact input
grouping produced 54 repeated classes with no conflicting reference action.
There were zero classes shared across different active/inspected ID contexts:
the task's two root types are distinguishable. The successful common function
supports this visited, narrow task; it does not settle ambiguous derived-goal
selection or distinguishable urgent physical requests.

Evidence is under `logs/goal_switch_preflight_20260923_v3/`:
`measurement_summary.json`, `measurement_trace.jsonl`,
`measurement_aliases.jsonl`, `source_manifest.json`, `build_evidence.json`, and
the complete source snapshot and measured binary. Summary SHA-256 is
`242229cf534776a4d0d6970869058db191ec8e98b32a67595994bccb2e28644f`;
trace SHA-256 is
`b2fd79553cbc5a0a53f634fbdc55bea3313fed27c81229616bfe1764ec85bf0a`.
The same frozen source passed an AddressSanitizer/UndefinedBehaviorSanitizer run
with exit code zero. Its summary, full chronological trace and alias export were
byte-identical to the optimized run; `sanitizer_evidence.json` records hashes.
The earlier v1 snapshot preserves a compile-only identifier failure with no
measurements. V2 is the successful frozen-only precursor; v3 adds separately
seeded semantic sessions and the semantic-learning reference control before
measurement. These records are not training inputs.

The next justified boundary is a declared protocol warm-up followed by an
unassisted feature-only policy evaluation on held-out arrival times/order, with
a genuine after-warm deficit before any acquisition claim. Retained-task and
frozen/shuffled-credit controls are still required. This preflight closes task
execution/price feasibility only.
