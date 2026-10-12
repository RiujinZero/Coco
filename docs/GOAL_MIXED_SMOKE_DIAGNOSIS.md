# Mixed smoke failure diagnosis

Read-only inspection of the completed 20-mission mixed smoke: 19/20 guided
missions succeeded; the unassisted frozen warm checkpoint scored 6/8. No code,
teacher, configuration or running pilot was changed, and no new run was started.

Evidence is under `logs/goal_continuing_bootstrap_smoke_20260923_v1`:
`smoke_missions.jsonl`, `smoke_trace.jsonl`, `smoke_computations.jsonl` and
`smoke_evaluations.jsonl`.

| Case | Observed failure | What succeeded |
|---|---|---|
| Guided `warm_m18` | Urgent query arrives at 8, deadline 24, correctly answered only at 52. | All nine required motors execute; long root submits correctly at 163. |
| Frozen `eval_warm_m1` | Urgent query arrives at 4, deadline 16, correctly answered at 88; three extra motors go to an irrelevant derived goal. | Six-edge root submits correctly at 63. Every known-satisfied execution state immediately chooses `finish`. |
| Frozen `eval_warm_m2` | Direct search/exploration times out at 256; neither urgent query nor derived goals are selected. | Only three of nine required edges are present at timeout; there is no successful-goal stopping opportunity. |

In guided `warm_m18`, all 163 choices are explicit hints. Before the first
activation at 45, the agenda exhausts three six-edge candidates: clocks 4–14,
19–29 and 31–41 each use allowances 4 through 4,096. A separate three-edge
distractor is correctly rejected after its valid preview reports zero progress
at 17. The urgent request arrives while the agenda is open. Agenda observation
30 describes inspected-goal age; it does not expose the execution view's pending
arrival bit. Once execution starts, that bit prompts switching, but the query
is already late. Guided `warm_m2` and `warm_m10` encounter only one exhausted
six-edge candidate before activation at 17, answering at 21/22 within deadlines
24/26. Global-cap skipping works; candidate order and the example strategy's
agenda timing cause this failure.

In `eval_warm_m1`, the neural policy ignores the public pending-arrival flag
through 61 execution decisions. After completing the physical root, it activates
`route d0 toward d3` despite a valid paid preview with both before/after missing
counts zero. That adds three motors and ends at 78 before the query is serviced.
All three known-satisfied execution states choose `finish`; this trace shows
switching and relevance-selection deficits, not a stopping regression.

In `eval_warm_m2`, the policy inspects only the root, reasons at allowances
4–4,096, and activates it at 13 without a plan. It never returns to the agenda,
ignoring the pending-arrival flag on all 243 execution decisions. There are 218
`search_budget` and 18 `state_budget` results, 18 exploratory motors and 598,392
predictions. The final scene has sixteen distinct edges, but only required edges
2→3, 5→6 and 6→7. No execution observation reports satisfaction and no `finish`
is issued. This is failure to switch and decompose, not refusal to finish an
already successful goal.

These finite smoke findings do not establish general stopping mastery or a
universal public-feature limitation. The larger frozen-declaration pilot should
retain these failures and report its independent results without modifying its
teacher or configuration to chase them.
