# Optional agenda search-exhaustion feature, contract v2

`GoalControlLimits::agenda_search_exhaustion` is an opt-in switch, default `false`.
With the default, agenda observation feature 15 keeps its original meaning:
whether the inspected goal is derived, duplicating feature 13. Execution-view
features and the ordinary five-action controller are unchanged.

When the switch is `true`, agenda feature 15 is one exactly when all of these
conditions hold:

- The inspected goal has a cache current for the learned model and public world.
- Its last paid planner result has status `search_budget`.
- That planner invocation actually received the mission's global
  `preview.plan_predictions` allowance.

It is zero for a missing/stale cache, a different planner status or a smaller
actual allowance. `Entry::last_plan_allowance` records the actual invocation,
whereas `next_budget` describes a possible future computation. For a global cap
of 16, paid attempts at 4 and 8 leave the flag zero; it becomes one only after
the paid 16-prediction attempt returns `search_budget`. Merely setting
`next_budget` to 16 after the eight-prediction attempt does not expose exhaustion.

The preview has a second limit: total metered work. The additive
`goal_preview::Result::planner_allowance` records the exact bound at the planner
call, after answer/planner-call work is charged. The controller copies this into
the immutable `GoalComputationReceipt::plan_allowance` and its inspected cache.
The receipt value is zero if no planner is invoked. With a global prediction cap
of 16 and work cap 10, the two pre-call work units leave an actual allowance of
eight; that is not global search exhaustion. A work cap of one prevents the
planner call entirely and retains allowance zero.

World/model/deadline invalidation clears the cached allowance. Reading the
feature performs no planning, emits no receipt, changes no policy and supplies
no usefulness label. It reports a paid resource boundary, not a guarantee that
the candidate is impossible, irrelevant or worth abandoning. A later controller
or separately declared evaluator reference decides how to use it.

`cpp/tests/test_goal_search_exhaustion.cpp` checks real learned root and derived
goals through paid allowances 4, 8 and 16, the exact field difference between
opt-in and default observations, immutable computation receipts, world/model
invalidation, non-planning queries, observation purity and unchanged weights,
credit counts and motor counts. Choices use explicit test-only `ControlHint`
assistance; the test does not train or establish learned scheduling. Separate
direct-preview cases check the smaller work cap and a call that never occurs;
the controller cases verify exact forwarding of preview metadata into receipts
and cache.

The frozen source and build/run record are retained under
`logs/goal_search_exhaustion_v2`. The feature test is built with optimization and
with AddressSanitizer/UndefinedBehaviorSanitizer; the existing `test_goal_control`
is also rebuilt with optimization as the compatibility check. Original experiment
results and earlier header snapshots remain unchanged.

Any experiment using the new bit must explicitly record
`agenda_search_exhaustion=true` and contract v2 in its configuration, freeze the
source first, and preserve the preceding feature-contract results. Exporters
that need the paid-allowance provenance should explicitly serialize
`plan_allowance`; existing explicit serializers retain their prior fields.
