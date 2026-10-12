# Paid learned-goal consequence preview

`cpp/include/laminar/relational_goal_preview.h` is an opt-in supplied computation
for the goal-control design in `LEARNED_GOAL_CONTROL_NEXT.md`. It is not a goal
selector, policy, action executor, teacher, or acquired decomposition algorithm.
No existing learner headers were changed to provide this component.

`goal_preview::analyze(session, inputs, limits, prices)` resolves the parent and
candidate through the session's public learned language frames and concept
models. Both require one positive interpretation and one supported concept
hypothesis; OR branches within that hypothesis remain alternatives. Matching a
second language pattern, a negated interpretation, or an unresolved definition
does not qualify as an unambiguous positive goal.

It obtains the candidate's answer and bounded ordinary plan, then predicts each
planned action in a private copy of the public complete scene. The potential
fact domain is the union of the before and predicted-after facts, plus all facts
predicted by each public affordance from each of those two scenes. A failed
domain prediction invalidates the result. The domain is shared by both minima.

For every expanded parent branch, a bounded relational join enumerates every
compatible potential-domain grounding under the public parent argument bindings.
At each completed grounding it counts distinct atoms absent from each scene.
The two independently minimized counts produce the signed delta and its declared
normalization. A zero best-so-far value does not permit an early successful exit;
enumeration must finish. Ordering and deletion are relaxed only in this measure;
the candidate plan remains the existing actual transition-model plan.

Partial worlds, unresolved meanings, missing compatible groundings, failed plans,
failed predictions and exhausted work return `valid=false`, zero delta, and a
specific status. A failed partial enumeration leaves both reported minima at -1.
`normalization_atoms` must cover every expanded parent branch; there is no silent
feature clipping.

The counters charge one logical unit for each exposed model prediction, recursive
grounding expansion, potential-domain fact visit, fact match and before/after
truth check. Predictions reported by `session.plan` count too. The supplied
existing `answer` and `plan` calls each have an additional declared fixed call
price. Their internal interpreter joins are not exposed by the existing API and
are **not** represented as individually measured joins. This is a logical work
contract over bounded supplied calls, not an instruction-count or runtime claim.
Neural VM steps are reported separately; this positive concept fixture uses zero.
The owner can price that reported delta in its existing VM-cost mechanism.

`Limits.work` bounds all exposed logical units. `plan_predictions` additionally
bounds the ordinary planner; it is not a separate cap on total domain/replay
predictions. `plan_depth` and `normalization_atoms` are explicit. Each counter,
price, limit, complete public request input, knowledge version, expanded definition,
candidate plan, final predicted scene and completed potential domain is retained
in the result. `current` compares the session's ready/version state and every
public input, including content and observation/parent/candidate/affordance
revisions. An owner changing limits/prices must also invalidate its cache.

`goal_preview::propose` grounds the already learned positive language frames over
the public potential domain and renders their original surface patterns. It
returns text, names, predicate/head arguments, frame, definition and one public
potential witness. A potential witness is not a physical plan or usefulness
verdict. The inventory includes satisfied, irrelevant and useful instances;
candidate evaluation happens only through a paid request. The exact parent text
is omitted to avoid proposing itself.

Candidate ordering is a fixed portable hash of rendered text and a public seed.
The proposal input contains no deadline, weight, evaluator truth or preferred
goal. Every candidate is enumerated before truncation: `complete=true` means the
enumeration finished; `truncated=true` and `status=candidate_limit` explicitly
report a capped retained inventory and its full `total_candidates`. Work
exhaustion clears the candidate list and reports `complete=false`. Owners must
record capacity events rather than call a truncated list exhaustive or silently
skip cases in which the cap removes a useful alternative.

The later opt-in `ProposalInputs.minimum_atoms` field defaults to one and accepts
values 1 through 64. Only after the complete ordinary enumeration finishes, it
retains a proposal when every expanded branch has at least this many atoms. It
then applies the same seeded order and candidate cap. This is structural
granularity, not a usefulness or task-family filter. `enumerated_candidates`
records the pre-filter inventory; `total_candidates` records the qualifying
inventory. The full source enumeration is charged even if no proposal qualifies,
and `current` includes this option in its input equality check.

## Focused verification and limits

`cpp/tests/test_relational_goal_preview.cpp` uses the raw Boolean language/concept
and observed-effect fixture in `cpp/tests/goal_control_fixture.h`. The session
learns one-edge, three-edge and six-edge route concepts; tests do not inject their
definitions into its knowledge.

The useful three-edge candidate and irrelevant three-edge candidate both begin
false and each has an ordinary three-action plan. Paid preview distinguishes
their parent deltas, +3 versus 0. All 64 public six-edge presence masks match a
separate evaluator graph-walk enumeration of the exact distinct-atom minima.
Additional algorithm tests cover repeated ground atoms, different minimizing
groundings, signed harm and a genuine alternative union branch. They use direct
formula fixtures only to isolate the supplied algorithm; they are not acquisition
evidence.

Other checks cover complete/partial worlds, planner and preview exhaustion,
work exactly sufficient or one unit short, impossible potential groundings,
unavailable effects, true multiple-hypothesis ambiguity, knowledge contradiction
and recovery, every public revision/content cache boundary, deterministic order,
deadline-independent ordering, changed order seeds, and explicit candidate caps.
The generic inventory contains 15 candidates, including both useful and irrelevant
three-edge instances. A cap of two correctly reports all 15 as the underlying
inventory while retaining two. The proposed mission cap of four therefore needs
an explicit capacity decision in its later preflight; this result does not erase
that limitation.

For the useful initial fixture, charged work is 260: 114 model predictions
(including planner search), 7 grounding expansions, 137 atom checks, 1 answer
call and 1 planner call. Full proposal work is 561. Test-only prices
0.007/0.003/0.002/0.011/0.013 produce cost 1.117; these are accounting test values,
not accepted mission prices. No training was run, and this component result does
not establish long-route feasibility, aligned mission utility, learned switching,
learned subgoal selection or unrestricted goal invention.
