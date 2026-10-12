# Targeted six-edge / long-task diagnosis

The user requested checking these tasks individually and explaining why the
mixed continuing learner did not acquire them. This follow-up uses no subagents
and does not resume the paused automatic goal or launch another training campaign.

## Paired checks

`logs/goal_targeted_isolation_v2` freezes a 48-mission diagnostic: two fresh
libraries (routes only, or the combined raw bootstrap), two families, four
matched cases, and three selectors (the unchanged reference teacher, saved
warm-400 policy, saved autonomous-600 policy). Every case starts from a fresh
clone. Neural probes have no learning or hints. Reference probes use explicit
hints and ordinary semantic learning within the case, with zero policy updates.
All 48 missions complete; no policy updates occur.

| Library / selector | Interrupted six-edge | Nine-edge / derived goals |
|---|---:|---:|
| Routes only, reference | 4/4 | 4/4 |
| Combined bootstrap, reference | 4/4 | 4/4 |
| Routes only, warm neural | 0/4 | 0/4 |
| Combined bootstrap, warm neural | 0/4 | 0/4 |
| Routes only, final neural | 0/4 | 0/4 |
| Combined bootstrap, final neural | 0/4 | 0/4 |

Both libraries also pass all 64 six-edge and 512 nine-edge subset truth cases.
The reference succeeds with actual six/nine motors, urgent replies and resumed
work. Current task mechanics and raw route knowledge therefore remain viable in
these cases, including with the extra bootstrap skills present.

This is **library isolation**, not separately training a single-task neural
model. The saved weights were trained in the mixed run; the controller RNG and
history are fresh, not restored checkpoints. These fresh libraries also do not
reconstruct its final learned semantic archive. The check rules
out a simple immediate failure caused solely by the additional bootstrap
library; it does not rule out interference during mixed training. Earlier
individual task passes were reference-controlled, not established unassisted
neural mastery that this experiment subsequently lost.

## Actual combined-run failure

Read-only inspection of the original pilot shows that all four cases in each
hard family remain in the agenda for all 256 decisions, both after warm training
and after autonomous continuation. They never activate any root or subgoal and
execute no motors. This is different from the earlier hidden-32 smoke's failures.

Across the four warm six-edge probes, 1,016/1,024 choices are `reason`; across
the four warm long probes, 1,015/1,024 are `reason`. At the first disagreement
with the supplied teacher, all eight warm probes have a usable derived procedure:
the teacher selects `activate_goal`, but the neural policy reasons again. Seven
diverge at decision 6 and one at 18. The final checkpoint repeats the pattern,
sometimes reasoning on a candidate the teacher would skip. Independently decoded
saved neural values confirm every reported first-divergence choice.

On the 2,344 distinct demonstrated observation/menu states from the warm trace:

| Demonstrated decision | Warm agreement | Final agreement |
|---|---:|---:|
| Agenda reasoning | 960/960 | 960/960 |
| Agenda next-goal inspection | 439/936 | 432/936 |
| Agenda activation | 136/382 | 113/382 |
| Execution arrival switch | 0/5 | 0/5 |
| Known-True execution finish | 4/4 | 4/4 |
| Other execution actions | 57/57 | 57/57 |

Thus the learned low-level execution/finish choices fit these demonstrations,
but goal management does not. Activation agreement **does regress** during
autonomous continuation, although whole-task scores remain at their existing
zero floor. This is agreement with the supplied teacher, not proof of optimality
or general mastery of the other actions.

## Assistance exposure and the next scaffold

The original run already used decaying guidance: 100 mixed fully guided missions,
then a taper to zero through mission 400. Each hard family received only **13**
fully guided missions. Its direct supervised-fit budget could not fund another
complete round after mission 250, leaving 150 tapering missions and all 200
autonomous missions without direct example fits.

| Phase | Six-edge success | Long success |
|---|---:|---:|
| First 100 mixed missions, fully guided | 11/13 | 7/13 |
| Taper while direct fits remain available | 13/19 | 12/18 |
| Remaining taper after direct fits stop | 6/18 | 10/19 |
| Autonomous continuation | 2/25 | 0/25 |

Of the distinct demonstration states, 2,278 are agenda states and only five are
execution-arrival states. Balancing distinct states alone does not balance the
skills that need learning. The native agreement check exposes this gap even on
the demonstrated states, before any claim about novel-task generalization.

A longer warm start with a decaying scaffold is a reasonable next hypothesis,
but the present result does not prove that duration alone fixes it. A targeted
follow-up should keep direct fitting available throughout the intended taper,
balance practice by decision type (especially activation and arrival switching),
and decay per-skill assistance only after stable unassisted probes. Preserve
mixed old-skill probes throughout. The reference teacher also fails some fully
guided missions; fix or explicitly isolate its agenda-arrival scheduling weakness
before treating longer imitation as sufficient. No such new scaffold or training
run is implemented by this diagnostic.

## Reproduction and validation boundary

`python/run_goal_targeted_isolation.py --out NEW_DIRECTORY` freezes source,
extracts the saved warm/final weights, records the declaration, compiles with
GCC and runs the 48 cases. `python/analyze_goal_targeted_failure.py` reads the
preserved v2 evidence and original pilot trace for the failure analysis.
Compact records are [GOAL_TARGETED_ISOLATION_RESULTS.json](GOAL_TARGETED_ISOLATION_RESULTS.json)
and [GOAL_TARGETED_FAILURE_ANALYSIS.json](GOAL_TARGETED_FAILURE_ANALYSIS.json).
Full traces, raw bootstrap evidence, computation receipts and extracted weights
remain local under the frozen log directory.

Version 1 stopped at compilation because the diagnostic used an unqualified
namespace; no missions ran. Its failure is preserved. Version 2 fixes that
qualification and the audit-only task-order label before freezing and running.
The controller, teacher, semantic-learning algorithms, budgets and original
pilot evidence are unchanged. The mixed app gains only an entry-point guard
so this diagnostic can reuse its exact environments. This is a bounded behavior
check and read-only fitting analysis, not a full independent audit of the pilot.
