# Explicit policy warm starts

The mixed continuing learner can now fit demonstrated operations directly with
`GoalController::fit_demonstrations(rounds, source)`. Previously a `ControlHint`
selected and recorded an operation, but its label did not itself train the neural
policy. Terminal reward remained the learning signal.

This opt-in method performs backpropagation on active, credited steps tagged with
the exact requested teacher source. It balances distinct observation/menu states,
fits the demonstrated operation to one and other legal operations to zero, and
records each update for existing credit withdrawal/restoration replay. It updates
the neural policy only; empirical values and semantic knowledge are unchanged.
The caller controls its schedule and budget. Ordinary choosing, terminal credit,
consolidation, and evaluation never invoke it automatically.

Conflicting actions for an identical observation/menu under one source are rejected
before any update. Frozen learners, active missions, invalid sources, stale mission
links, and overflowing update requests are rejected. An absent source fits zero
rows. Only the experimental seven-operation controller has this update path; the
five-operation controller's compiled update behavior is unchanged.

`logs/goal_demonstration_fitting_v1` freezes the tested headers and tests. Optimized
and ASAN/UBSan demonstration tests pass, including learning the labels, unchanged
values/semantics, invalid-call atomicity, and exact withdrawal/restoration. Existing
goal and mission-credit tests also pass. The machine-readable validation includes
source and binary hashes and the separate exact five-action parity result.

This establishes a usable warm-start mechanism, not generalization. The next
experiment combines demonstrations for goal selection, subgoals, interruption,
stopping, language, actions, and skill repair in one continuing learner, then
removes assistance and measures acquisition and retention.
