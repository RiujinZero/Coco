# One continuing mixed learner

The user directed a joint experiment: warm-start goal selection, subgoals,
stopping, language, actions and new skills, use backpropagation where useful,
then remove assistance and observe learning and retention. Perfect isolated
skill scores are not a prerequisite. This experiment is exploratory; it does
not by itself close the four complete-learner gates.

`cpp/apps/goal_continuing_bootstrap.cpp` keeps one seven-operation controller,
its relational knowledge and chronological mission credits, and one archived
`GroundedSession` reader through an interleaved curriculum. The controller uses
the same 32 public observation values for every family. A single explicitly
supplied demonstration function sees only those values and the legal menu.
It receives no family ID, evaluator truth or private usefulness label.

The mixed families cover:

- True/false queries, completed and unfinished physical goals, partial views,
  actual stopping, and interruption/resumption.
- Six-edge interrupted routes and longer nine-edge routes with paid generic
  subgoal proposals.
- A learned raw-report reader feeding conservative partial scenes to control.
- Equal counts, numeric counts, chain endpoints, and a reusable red-or-blue rule.

Raw semantic examples and independently verified execution outcomes teach the
libraries. Controller hints are recorded with their source. Explicit
`fit_demonstrations` calls then backpropagate those operation labels into the
same neural policy. Ordinary terminal-reward updates also continue. All fitting
and semantic interventions are logged; a hint alone is not claimed as a learned
decision. In the pilot, later repair/rule-union examples arrive after existing
policy updates and mission credits, with earlier policy and credit retained.

Each mission permits 256 decisions and 32 motors. Operation, computation and
management prices are zero, following the user's acquisition-first direction;
actual decisions, predictions, primitive work and timing remain visible.
Actual world satisfaction does not automatically finish a goal. The policy
must select `finish`, and missed deadlines still count as failures.

Frozen probes use clones only for evaluation. They receive no hints or fitting
and cannot change the continuing learner. Generalization probes vary names,
action ordering, lengths/counts and interruption times within these supplied
families. The stopping probe set includes complete, incomplete, partial and
interrupted physical cases, not just factual queries. Familiar families are
retested after new skills arrive. These repeated developmental probes are not
an untouched confirmatory test set or evidence of arbitrary-task generalization.

The reader retains all alternatives, explicit negatives and closure metadata in
its evidence log. Its conservative projections remain partial. Positive
entailed queries can finish; ambiguity and partial negative evidence remain
`Unknown`. This run does not implement the pending uncertainty-aware semantic
teaching contract or the full Organism/reader integration.
The reader's atomic question form/program and eight elementary transition
examples are declared initial assistance. Four raw report-answer lessons then
teach the unfamiliar report surface. This is not acquisition of the entire
reader substrate from scratch.

Before the larger pilot, a 20-mission guided mixed smoke checks the live
integration and reports unassisted probes before and after. Its small neural
fit budget is a plumbing check, not a convergence claim. The pilot's source,
configuration and binary must be frozen before launch; results and failures
will be appended with their evidence paths.

## First mixed smoke result

The frozen run in `logs/goal_continuing_bootstrap_smoke_20260923_v1` completed
all 36 missions in 84.65 seconds: eight unassisted initial probes, twenty guided
mixed missions, and the same eight unassisted developmental probes afterward.
The single hidden-32 controller retained twenty mission credits and accumulated
10,240 terminal-reward updates plus 3,840 explicit demonstration fits. It received
no hints, fits or learning updates during frozen probes.

| Probe family | Before | After warm start |
|---|---:|---:|
| Unfinished physical stopping, four missing edges | 0/1 | 1/1 |
| Six-edge interrupted route | 0/1 | 0/1 |
| Nine-edge route with derived goals | 0/1 | 0/1 |
| Learned raw-report reader query | 0/1 | 1/1 |
| Equal counts | 0/1 | 1/1 |
| Numeric count | 0/1 | 1/1 |
| Chain endpoint | 0/1 | 1/1 |
| Rule union | 0/1 | 1/1 |

Total unassisted success improves **0/8 to 6/8**. The physical stopping probe
uses four actual motors and a policy-issued finish. Guided training succeeds
19/20 times; the final guided long mission fails and remains in the evidence.
This small run demonstrates joint integration and a narrow learned improvement,
not convergence, causal attribution between reward and supervised fits, or broad
complete-learner capability.

The smoke's authored reader-to-controller language/concept lessons have actual
receipt source `verified_teacher`, but their exporter printed
`verified_environment`. Both are supplied teaching, yet the log field is
inaccurate. The frozen run is preserved with this erratum; later export must
report the actual source. Raw reader-answer sources and the declared initial
atomic reader substrate are logged correctly.

Result SHA-256:
`7dd8ab5c51f86e2e55ce48044ed97c0d0aa371c91102a0de0844f5cf16c0b1d5`.
Binary SHA-256:
`8d24937f97fc9080b5456031d8f34d5dd6166b0f1a16c6f170cc974c26952f09`.
The completion manifest contains the full frozen evidence inventory.
The [independent smoke audit](GOAL_CONTINUING_SMOKE_INDEPENDENT_AUDIT.md)
passes with the documented source-label erratum: all 2,448 unassisted choices
are reproduced from saved parameters, and callbacks, terminal truth, guidance
and fit/update counts are checked. That audit does not cover the larger pilot.

## Declared larger pilot

`logs/goal_continuing_bootstrap_pilot_20260923_v1` freezes and launches a fresh
hidden-1,316 controller (52,647 weights), seed 1. This is a new experimental run;
within it, one learner and reader persist throughout all stages. It contains
400 warm missions followed by 200 autonomous learning missions, plus 46 frozen
probes at each of the initial, warm and autonomous checkpoints: 738 missions.

Guidance probability is one for the first 100 warm missions, then tapers to zero
by mission 400. Every ten warm missions, direct demonstration fitting uses up to
eight rounds, also tapering to zero, under a total cap of 200,000 fits. Every
training mission supplies 512 terminal-reward updates. Autonomous continuation
uses epsilon 0.1, with no hints or explicit demonstration fits. Frozen probes
use epsilon zero and no learning. Failures are retained in every stage.

Raw duplicate-count, chain-follow and union examples arrive at the boundary
after 100 missions. Before that boundary, the union slot practices an old count
skill and computational tasks avoid the later repair counterexamples. The
extension checks unchanged policy weights, update count and all 100 credits.
The long-route/procedure and reader warm starts are declared upfront. Probe
entity offsets are disjoint from training offsets; the long-task topology
remains the same, with different names, order and interruption times.

The reader teaching exporter now records the actual `verified_teacher` source.
Configuration SHA-256:
`1be533aa9c3004364879ea1cac3e7f7f1e506ebcb881396ef356cfa1fc26c878`.
Binary SHA-256:
`80d404257b449d6b7d6c460fb4deaa8cf9b025dae7e1c8b8223878dccdcccc08`.
Source-manifest SHA-256:
`5a8257e1db325a0ad46a2d5c5122cfd1044fdaa3bb3ba3d20fb252f324ea435e`.
The completed run contains all 738 declared missions and takes 1,078.95 seconds.
Warm success is 351/400; sampled autonomous success is 148/200. Frozen results
are **0/46 before, 34/46 after warm start, and 34/46 after autonomous continuation**.
Both trained checkpoints score stopping 14/18, six-edge interruption 0/4,
long-route/derived work 0/4, and 4/4 each on reader, equal-count, numeric-count,
chain-endpoint and union probes. These are the same developmental probe set.

There are 600 retained mission credits, 307,200 reward updates and 198,358
demonstration fits. The last nonzero direct fit occurs after warm mission 250;
the remaining cap cannot cover another full round over the growing distinct
state set. The mission-100 skill extension preserves weights, updates and all
100 credits. Autonomous continuation has no hints or explicit fits.

The [compact result record](GOAL_CONTINUING_BOOTSTRAP_RESULTS.json) preserves
configuration, hashes and phase results. Its consistency finalizer is not a full
independent choice/physics audit of the larger run. The result supports retained
learning within this mixed supplied curriculum, while interruption and longer
goal organization remain open. It does not isolate the causal effect of width,
reward learning or supervised fitting. At the user's request, work stops here
for a commit and [new handoff](CONTINUING_LEARNER_HANDOFF.md); further runs are
manual and without subagents.
