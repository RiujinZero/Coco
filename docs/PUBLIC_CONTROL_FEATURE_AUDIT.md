# Public controller features — 2026-09-23

`ControlledRelationalSession::observe()` constructs all 32 controller inputs from
the current public scene, the learner's own computation and public action/cost
state. The following audit applies to the structured continual adapter. It is
not an audit of the separate byte reader or of a future external environment.

| Indices | Contents | Source |
|---|---|---|
| 0–1 | Scene complete; scene nonempty | Accepted visible observation |
| 2–3 | Knowledge dirty; reconstructed model ready | Learner's evidence/model lifecycle |
| 4 | Physical action intent | Public request flag |
| 5 | Answer cached | Learner's completed computation |
| 6–8 | Cached answer true, false, unknown | **Learned belief**, not evaluator truth |
| 9 | Plan contains actions | Learner's planner output |
| 10–11 | Search allowance exhausted; terminal planning failure | Supplied planner's observed status |
| 12–13 | Retrieval failed; asking failed | Actual callback acceptance/result |
| 14 | Own execution token exists | Learner's action history |
| 15 | Offered actions exist | Public affordances |
| 16–17 | Last operation rejected; procedure reused | Learner's execution/planning status |
| 18 | Plan length capped at four, divided by four | Learner's plan |
| 19 | Logarithm of current search allowance, divided by 15 | Declared supplied allowance |
| 20 | Reactive motor allowance exhausted | Own acknowledged actions and declared limit |
| 21–25 | Five operation costs, multiplied by 20 | Public price profile |
| 26–28 | Prediction, motor and rebuild costs, multiplied by 20 | Public price profile |
| 29–30 | Action-count and fact-count features | Both zero in the recommended and repair profiles |
| 31 | Constant one | Bias input |

No feature contains the evaluator's correct answer, primitive-to-repair identity,
desired next operation, private environment definition or future terminal reward.
The output of a tool the learner has already run is available as a feature; this
does not establish that the supplied planner/search procedure itself was learned.

`python/check_relational_control.py` independently reconstructs every input vector
at both sides of each recorded decision, checks frozen neural operation choices,
and verifies observations against their public/private callback boundaries.
The repair runs use `audit_models=1`: `LearnedComputations` independently executes
the saved model and frozen neural primitive readout to derive features 6–8 even
when the learned answer is wrong. World truth is computed separately for scoring.
The repeated-record pilot is a concrete negative check: its 32 wrong pre-repair
control answers pass this reconstruction rather than being replaced by truth.

Operation hints are a separate, explicitly tagged training intervention. The
declared public-feature teacher supplies no primitive trace or faulty-slot label.
Hints end during warm-up and are absent from scored evaluation and subsequent
repair adaptation. The independent audit checks that schedule and intervention
attribution. These engineered features and optional hints are assistance; they
are not evidence of independent adaptation to unfamiliar operation sequences.

Evidence: `logs/repair_records_pilot_v4/summary.json`,
`logs/repair_records_lifecycle_v5/summary.json`, and the full traces/model readouts
in their per-profile seed directories, plus the completed three-seed campaign at
`logs/repair_records_three_seed_v7/acceptance.json`. Earlier normalization-audit failures remain
preserved in the original interrupted campaign, as documented in the handoff.
