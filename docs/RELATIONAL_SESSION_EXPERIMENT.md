# A continuing relational learner — 2026-09-23

`RelationalSession` connects growing concept definitions, language grounding,
observed action effects and procedures learned from its own successful executions
in one versioned model. The components acquire new knowledge during an interleaved
stream; later concept learning and corrections trigger replay of dependent language
and procedures. This closes a concrete integration gap between the previous
separate concept and action experiments.

Across three vocabulary/role permutations, the session passes **2,304/2,304 fresh
final language queries** and **144/144 fresh final action goals**. Under a tight
planning budget, learned procedures pass **72/72** transfer requests; search from
the identical model snapshots passes **25/72**. The independent checker verifies
all learned definitions and effects, execution observations, receipts and program
support. ASAN/UBSAN contracts cover corrections and stale-credit rejection.

This is still a scoped symbolic integration. The neural primitive policy and
learned computation controller are **not** attached to this session. Search,
exploration and rebuild scheduling remain supplied. The full four-capability
objective remains open; neither backprop nor a local neural update is used here.
Backprop remains permitted for integrating and proving the complete learner.

## One model and shared evidence

The session owns the existing `ConceptMemory` and `GroundedActuator`, a new
variable-arity language-frame bank, procedural candidates and their raw receipts.
Its state is copied by value for isolated probes; no hidden evaluator object or
task-family discriminator is reachable from inference.

- Concept evidence supplies fact tuples, an opaque query name and argument tuple,
  and a verified binary answer. It induces relational definitions as in the
  [concept-growth experiment](CONCEPT_GROWTH_EXPERIMENT.md).
- Language evidence supplies an utterance, complete observed world, the perceived
  entity-name inventory and a verified answer. It does **not** supply a concept
  label, argument roles or program. Generic hypotheses choose among currently
  learned concepts, permutations of the utterance arguments and Boolean polarity.
  Equivalent relational interpretations are deduplicated.
- Effect evidence supplies an issued action schema/arguments and independently
  observed complete before/after worlds. The existing actuator learner identifies
  relation, argument roles and add/remove/row-replacement/column-replacement effect
  from those observations. External transitions alone cannot create procedures.
- A completed action request supplies observations for the model's own issued
  actions and one verified terminal satisfaction bit. The same receipt updates
  effect evidence, labels the resulting world for language learning, and—if
  successful—can support a procedural candidate. No external trace is imported
  as a procedure demonstration.

Procedures are generalized action sequences. An action argument can bind an
utterance role or a unique binary-relation lookup from an utterance role in the
current world. These selectors are induced from actual successful traces.
Before reuse, every step must still be available, its outcome must agree across
the current effect hypotheses, and the complete predicted sequence must satisfy
the current language interpretation. A stored success is not sufficient by itself.

This frame learner handles whole controlled utterance patterns with up to eight
entity occurrences. The experiment uses two, three and four roles. New words and
orders are grounded through answers, but phrase grammar, free-form dialogue,
entity recognition from raw perception and arbitrary syntax are not established.

## Revision and stale decisions

All accepted evidence changes advance one session version and invalidate current
inference until explicit rebuild. Issued plans carry their version and a unique
token. A later concept/language/effect/receipt update prevents the old decision
from accepting credit or partially updating another component.

Rebuild starts with raw concept evidence, re-derives action effects, reinterprets
raw language lessons against the current concepts, then validates procedural
support against the current language and effects. A conflict can make a previous
procedure unavailable; correcting the identified receipt can restore it without
training a replacement demonstration. This session does not disable or bypass
the existing `GroundedArchive` fixed-prior guard.

The terminal language receipt derived from an execution shares that execution's
answer, activation state and revision. Observation corrections to own executions
are atomic bundles: the initial/step observations, dependent transition pre-states,
and terminal language world are replaced together. An individual derived transition
cannot be edited while leaving contradictory copies of its world elsewhere.
Source, version, shape and capacity failures reject before mutation.

The probe executes an issued sequence as a bounded batch, then publishes its
observations and terminal feedback atomically. It does not establish replanning
between individual physical actions or recovery from a changing environment
during a batch. Action preconditions and changing affordances are not learned.

The original experiment withheld all language/planning while any introduced
concept lacked a unique usable definition. The subsequent
[concept-isolation fix](CONCEPT_ISOLATION_EXPERIMENT.md) removes that global gate:
rebuilding keeps independent meanings usable while unresolved definitions cannot
ground language; computational queries require agreement for the relations they
actually use. Dirty evidence still requires explicit rebuild. Raw memories remain retained. Complete
worlds are required for effect learning and planning; query inference can handle
partial worlds, including proved negatives of negated constructions. An absent
positive proof in a partial world remains unknown.

## Stream and results

Each seed starts with empty learned banks. Eight stages introduce the same eight
concept families used in concept growth, with independently permuted predicate
words, concept argument orders, surface argument orders and actuator argument
orders. There are 432 ordinary concept lessons, 864 external language lessons
and 16 independently observed transitions. Four motor schemas arrive during
stages 1, 2, 3 and 6: additions for three relations, then removal for one relation.
Row and column replacement handling is additionally covered by sanitizer contracts.

After each stage, old and new language forms are queried and physical goals are
issued. Goal execution immediately updates the same model before the next request.
These updates add 161 execution receipts and paired language lessons per seed,
with 287–293 observed action steps per seed. Thus final language receipt count is
1,025 and total transition count, including the 16 external observations, is
303–309. Later unary/ternary observed schemas and four-role concepts are acquired
while these memories remain active.

| Measurement | Three-seed result | Interpretation |
|---|---:|---|
| Queries during interleaved acquisition | 2,160/2,160 | Earlier forms remain usable as later components learn |
| Fresh final language queries | 2,304/2,304 | Answers and role/polarity semantics independently checked |
| Action goals during acquisition | 216/216 | One continuing model plans and learns from outcomes |
| Fresh final action goals | 144/144 | 292 actual actions, sequences of up to three actions |
| Tight-budget procedure transfer | 72/72 | Fresh entities; every success uses a learned procedure |
| Search on identical tight-budget snapshots | 25/72 | Same concepts, language, effects, goal, menu and budget |
| Negative-language action goals | 21/21 | Learned deletion effects satisfy negated requests |
| Supplied exploration episodes | 1/24 goals satisfied | These are acquisition opportunities, not a successful goal policy |
| Partial-world query checks | 462/462 expected responses | 231 withhold; reuses subsets of final worlds |

The final goal stream itself continues learning after each outcome. Every goal is
scored before its own satisfaction feedback enters memory; these results should
not be described as frozen-policy evaluation. The matched search controls run
read-only against the exact pre-request model snapshot, without adding their
simulated outcomes to training. With the normal work budget, this search control
also solves all 144 fresh final goals. Procedure reuse helps under the tested
compute restriction, not by introducing otherwise unrepresentable action effects.

There are 44, 49 and 48 retained procedural candidates for seeds 0, 1 and 2.
Different successful bindings can create distinct candidates; these counts are
not a claim of that many independent semantic skills. The extra procedure warm-up
phase has six requests, five of which already use existing procedures. The 72
subsequent tight-budget cases test reuse, not 72 new discoveries.

One contradictory old concept receipt per seed makes 24 language queries withhold
and prevents all 24 corresponding action requests from issuing actions. Three
additional queries between accepted edits and rebuilding also withhold. Correcting
that identified receipt restores the **exact prior concept, language, effect and
procedure models**. All 2,304 final queries pass again; these are the same worlds,
not another independent test set. Contracts also exercise independent language
and effect corrections, revision of execution outcomes, full observation-bundle
correction, malformed feedback, source rejection and clone isolation.

## Search failure and repair

The first three-seed run had one failed final action goal in seed 2. Breadth-first
search reached its 4,096-state cap after 13,005 transition predictions despite
having correct learned meanings and effects. Both the procedural arm and matched
search control withheld. That run, its initial source and audit are retained in
`logs/relational_session_unpruned/`.

The planner now uses learned goal literals and learned effect scopes to exclude
actions that cannot contribute. For a positive conjunctive goal, a pure deletion
cannot help; for a negated goal, a pure addition cannot help. Fixed query-role
bindings further exclude irrelevant tuples. Replacement effects consider their
deletion scope when fulfilling a negative goal, not just the newly added tuple.
This pruning is sound for the current unconditional-effect, fixed-affordance
model; it must be revisited if action preconditions or changing affordances are
introduced.

With the same limits, the failed request now finds a three-action solution after
3,529 transition predictions, including procedural attempts. All three final runs
pass. This is **supplied symbolic planning logic based on learned definitions**;
it is not evidence that the learner discovered search control. Prediction counts
do not include every binding, relevance or query-evaluation operation, so they
are not wall-time or total-computation measurements.

## Audit, bounds and remaining work

The evaluator owns true action mechanics and intended goal meanings in a separate
file. The learner receives only public worlds, utterances, available actions and
then observed outcomes. The checker reconstructs every execution with independent
effect semantics and checks terminal satisfaction with the separate direct
relational oracle. It audits the published concepts, language hypotheses and
effects against all active raw receipts, and requires every procedure to have a
valid own-execution witness under the current model.

Per seed it performs 61,743 concept-evidence and 136,198 language-evidence checks,
plus 24,770–25,125 effect-evidence and 3,836–4,367 procedure-support checks. These
are repeated model audits, not independent behavioral examples. Inputs, outputs,
code and executables are hashed in `logs/relational_session/audit.json`.

The inherited concept bounds remain in force. The session adds 4,096 receipts per
language/transition/execution bank, 256 procedural candidates, 128 available actions,
4,096 search states and a default 32,768 transition-prediction budget. Maximum
accepted planning depth is 16; this experiment requests four and executes at most
three. Utterances are bounded to 2,048 bytes, 64 tokens and eight entity occurrences.
Language induction caps candidate/example evaluations at one million per frame;
individual relational evaluations have their own concept-memory work limits.
All storage is in memory, and rebuilding replays the retained evidence globally.

The next integration must attach the learned retrieve/reason/ask/act/finish
controller and neural primitive learner to this changing semantic/effect state,
then require acquisition and retention in a longer mixed stream. Rebuild and search
scheduling cannot remain host-selected for the full control gate. The current
three-action tasks do not replace the longer-credit requirement, and positive
conjunctive concepts do not establish unrestricted representation growth.
No byte-trainer option or `GroundedArchive` behavior changes in this experiment.

## Reproduce

```bash
bash scripts/relational_session.sh
```

Implementation: `cpp/include/laminar/relational_session.h`.
Native execution/evaluator boundary: `cpp/apps/relational_session_probe.cpp`.
Sanitizer contracts: `cpp/tests/test_relational_session.cpp`.
Curriculum and independent audit: `python/relational_session_probe.py` and
`python/check_relational_session.py`. The existing concept-growth suite also
passes after adding read-only schema accessors to `ConceptMemory`.
