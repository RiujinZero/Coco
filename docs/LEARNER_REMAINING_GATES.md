# Remaining complete-learner gates

Status: 23 September 2026. All four broad gates remain open. Component tests,
supplied reference executions and representation-specific successes do not close
the combined learning claim. The original acceptance contract remains in
[COMPLETE_LEARNER_HANDOFF.md](COMPLETE_LEARNER_HANDOFF.md).

The user's current direction is to combine the abilities in **one continuing
learner**, bootstrap them with examples and backpropagation where useful, taper
guidance, and measure autonomous learning and retention. Isolated perfect skill
scores are not prerequisites for this mixed experiment. Component checks locate
integration failures; they do not replace the joint run.
The [continuing bootstrap experiment](GOAL_CONTINUING_BOOTSTRAP.md) records its
scope, assistance boundaries and evaluation design.
Its first 20-mission mixed warm start improves frozen unassisted probes from
**0/8 to 6/8**, including an actual four-motor physical completion and learned
finish, the reader and four skill-query families. Interrupted six-edge and
nine-edge tasks still fail. Guided success is 19/20. This is narrow joint
learning evidence. The larger tapered continuation then completes 738 missions:
351/400 warm and 148/200 autonomous training successes, with frozen probes
**0/46 → 34/46 → 34/46**. Both trained checkpoints retain stopping 14/18 and
reader/four skill queries 20/20, while six-edge interruption and long tasks
remain 0/8. See the [new handoff](CONTINUING_LEARNER_HANDOFF.md). The user requests
a commit and paused automatic goal, with any further targeted runs performed
manually and without subagents.

| Gate | Implemented or measured | Still required |
|---|---|---|
| Independent computation adaptation | Greedy-trace credit passes the matched retention screen; disjunctive concepts and successive computation repair have bounded component/session evidence. | Unassisted adaptation to unfamiliar operation/evidence/cost structures after hints end, with frozen and corrupted-credit controls and old-skill retention. More schedules within supplied task families are insufficient. |
| Long sparse tasks and autonomous goals | One seven-output policy owns inspection/activation, switching/resumption, public arrivals, generic paid candidate previews, shared budgets and one chronological mission credit. Protocol checks and corrected reference feasibility pass. The first declared learning pilot exposes a switching-learning failure. | Actual policy acquisition of goal/subgoal choices and longer execution organization. Diagnose values versus neural fitting, then measure unassisted acquisition and retention with matched controls. Unfinished mission persistence is also missing. |
| Growth beyond the supplied interface | Reusable disjunctions extend the earlier conjunction-only grammar; a repair bank retains successive repairs and supports correction/withdrawal. | Evidence-grounded extension of representation or callable interfaces beyond the developer-supplied grammar, slots and microoperations, then reuse, composition, correction and retention. The new preview/proposal machinery is supplied computation, not invented machinery. |
| One continuing reader/reasoning/action/growth learner | Actual reader alternatives feed the same seven-action goal controller through conservative scene projections. Tests cover unfamiliar report learning, ambiguity, correction stamps, separate worlds and mission credit. | Transactional historical reader-view ownership; uncertainty-aware teaching and correction of its derived evidence while preserving independently verified outcomes; full reader/Organism integration; one continuing mixed run combining the above capabilities and retention. |

## Latest verified engineering boundary

- [GOAL_CONTROL_PROTOCOL_VALIDATION.json](GOAL_CONTROL_PROTOCOL_VALIDATION.json)
  records optimized and ASan/UBSan runs from a frozen source snapshot. They cover
  stale callback rejection, interrupted derived-goal resumption, shared budgets,
  private episode lifecycle, one terminal reward, coupled correction/withdrawal,
  exact replay, immutable paid-computation receipts and external arrival flags.
- [MISSION_CREDIT_SUBSTRATE.md](MISSION_CREDIT_SUBSTRATE.md) records the shared
  five/seven-action implementation, exact five-action parity, warm import and
  mission-only overflow rejection. No normal trainer profile was promoted.
- [GOAL_CONTROL_BUILD_V1_VALIDATION.json](GOAL_CONTROL_BUILD_V1_VALIDATION.json)
  records a frozen full trainer build passing 201 training episodes, two frozen
  episodes, exact replay/correction, six rejection checks and eight byte-identical
  handoff artifacts. The later mission-only overflow guard is separately checked
  in `logs/mission_fastpath_validation_v2`; the old-five probe binary is identical
  across that guard change.
- [GROUNDED_GOAL_BRIDGE_VALIDATION.json](GROUNDED_GOAL_BRIDGE_VALIDATION.json)
  records optimized and sanitizer integration checks using actual reader output.
  These remain protocol tests with declared forced control choices.

## Current experiment boundary

The goal implementation has no canonical trainer goal profile yet. Its initial
long-task family had to be revised before training: a twelve-edge simple route
exceeded the representation bound, a nine-edge additive task admitted a trivial
action sweep, and a wider task exposed proposal-capacity and preview-planning
failures. Those negative preflights remain preserved. A declared generic
macro-only proposal filter and short-route procedure acquired from observed
execution address particular feasibility failures; they are bootstrap and
supplied-interface choices, not independent acquisition results.

[GOAL_LONG_PREFLIGHT.md](GOAL_LONG_PREFLIGHT.md) and
[GOAL_SWITCH_PREFLIGHT.md](GOAL_SWITCH_PREFLIGHT.md) track the current reference
checks. Full-precision traces, public inputs, menus, computation receipts, costs,
arrival timing, semantic-learning conditions and independent control RNGs must
be audited before treating a family as ready for acquisition. An untrained initial
policy deficit is not the required frozen warm-bootstrap deficit.

The user explicitly prioritized learning the task at all over efficiency.
Current experiments therefore allow 256 decisions and a physical root deadline
of 256, retaining 32 motors. Their success-first objective sets all action,
computation and management prices to zero while logging actual work and timing.
Success earns a positive discounted return; failure earns a negative one. The
old efficiency margin no longer blocks acquisition. Original 96-decision and
original-price evidence remains preserved.

The old reference passed 24/24 frozen cases but only 16/24 with semantic learning
and zero policy updates. Increasing its allowance alone retained those eight
failures: it skipped cached derived previews marked `search_budget` instead of
paying the larger allowance already offered by the wrapper. The separately
declared v11 reference retries them and closes started, satisfied children after
resumption. It passes **24/24 frozen and 24/24 semantic-learning cases**, with
nine motors and at most 95 decisions, under both price variants. All 15 actual
unfinished semantic-learning interruptions succeed. This corrected feasibility
witness is not a learned policy. The
[independent audit](GOAL_PREFLIGHT_INDEPENDENT_AUDIT.md) verifies the results,
public callbacks, preview minima, returns and equality of actions/work across
price variants. Its visited-state checks do not prove universal feature
sufficiency. The 256 allowance remains in place for learning.

The [first learning pilot](GOAL_CONTROL_LEARNING_PILOT.md) has completed all 760
declared missions. It uses one raw six-edge semantic bootstrap, 100 guided/tapered
warm missions and matched 200-mission adaptive, frozen-policy and random-credit
continuations. Warm training succeeds 93/100 times, but all five frozen greedy
checkpoints score **0/12**. Sampled autonomous successes are 9/200 adaptive,
5/200 frozen-policy and 2/200 random-credit. These do not establish acquisition.
The warm policy completes the physical route but misses the urgent deadline;
the adaptive endpoint regresses to agenda reasoning loops.

A read-only neural-weight analysis at
`logs/goal_policy_fit_diagnostic_v1/fit.json` locates a narrow fitting deficit.
The warm policy agrees with all 54 distinct assisted agenda/no-arrival states
(4,477 logged decisions), but none of the 13 arrival states (96 decisions): it
chooses reason instead of inspecting the arriving goal. Its numerical decoder
matches every exported evaluation choice. This is agreement with supplied
assistance, not an independent optimality label. The completed bounded diagnostic
reconstructed the exact warm weights and evaluation, then compared learned
empirical values with neural choices and used state-balanced consolidation.
All three complete-mission scores remain 0/12. The value-table readout switches,
answers the urgent query on time and resumes in all 12 cases, but reasons
repeatedly after completing the physical task: its learned value for reasoning
exceeds finishing. Consolidation adds 28,928 fits without new rewards, hints or
changed empirical values, but the neural policy still fails. Thus this is both
a learned-value and neural-fitting problem, not only an underrepresented arrival
classifier. Frozen evidence is in
`logs/goal_policy_distillation_diagnostic_20260923_v1`.
The user explicitly highlighted learned stopping. `finish` remains a selectable
policy action; satisfying the world does not automatically close a goal.
Subsequent learning reports must distinguish physical satisfaction, timely goal
switching, actual policy-issued completion and extra decisions after satisfaction.
A timeout is a failed stopping result, never a substitute for learned completion.
The pilot also used 128 updates per mission; a 256-decision failed mission
therefore directly updates only its final 128 decisions, without additional
older-receipt replay on that submission. Preserve this update-budget limitation
when interpreting the failed adaptive run.

The [512-update continuation](GOAL_CREDIT_COVERAGE_CONTINUATION.md) addresses that
coverage limitation without new hints. Its 560 declared missions reconstruct the
same warm learner exactly. Adaptive continuation succeeds 5/200 times and the
random-credit arm 8/200; all endpoint greedy probes remain **0/12**, both before
and after separate value-consolidation fits. The adaptive endpoint completes and
finishes all physical routes but misses every urgent deadline and performs 312
extra actions after satisfaction. Consolidation instead produces repeated
reasoning without finishing. The independent audit passes evidence consistency,
not acquisition. These negative results remain preserved.

The [explicit demonstration fitter](GOAL_DEMONSTRATION_FITTING.md) now permits
source-tagged operation examples to train the same seven-output neural policy
directly. Previously hints selected and recorded actions but did not directly
fit their labels. The opt-in fitter balances observed states, preserves empirical
values and semantic knowledge, rejects conflicting labels, and participates in
credit withdrawal/restoration. Optimized, sanitizer, goal/mission compatibility
and exact five-action parity checks pass. This supplies the requested warm-start
mechanism; its validation is not a capability result.

The original nine-edge preflight omitted the six-edge route concept. The new
continuing-learner adapter retains it, learns the nine-edge meaning from raw
outcomes, and passes all 512 long and 64 retained subset probes. Its larger
inventory exposes four six-edge candidates with no reusable procedure; some
semantic-learning references repeatedly retry their exhausted global prediction
cap. This also occurs after exact reconstruction of the actual warm100 learner.
The [optional exhausted-search feature](GOAL_SEARCH_EXHAUSTION_V2.md) now lets the
controller distinguish a paid maximum-budget failure from an attempt whose
allowance can still increase. Optimized, sanitizer and existing protocol checks
pass; the default observation contract remains unchanged. The feature reports
spent computation, and does not choose whether to continue, switch or finish.
No long-task training or acquisition result follows from these reference checks.

With that optional observation enabled, the continuing warm-library reference
now passes **8/8** long cases (four frozen and four with semantic learning),
including all eight unfinished interruptions, timely urgent replies and nine
physical motors. Previously failing cases now skip candidates only after an
actual paid failure at the global prediction allowance. The exact warm policy
and its prior credits remain preserved; these are still supplied reference
decisions, not neural acquisition.

For the reader boundary, dependency stamps are currently forwarded by the host.
Partial projections intentionally withhold negative facts/closure and do not
become complete semantic teaching scenes. The
[reader transaction design](READER_GOAL_TRANSACTION_DESIGN.md) identifies an
additional hard boundary: the canonical reader always supplies partial worlds,
while existing semantic-teaching paths require complete worlds. Automatic
teaching therefore requires an uncertainty-aware evidence contract or genuinely
independent closure evidence. A view/history owner can be added incrementally;
reinterpreting a report must not rewrite verified physical outcomes, motors or
the observations the policy actually saw. Current stamp checks neither implement
that owner nor repair the full reader's primitive prior.

The user goal remains active. No complete-learner gate is marked closed, no
acquisition campaign is inferred from these tests, and failed runs and frozen
binaries remain part of the evidence.
