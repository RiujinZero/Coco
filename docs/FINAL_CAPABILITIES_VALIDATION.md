# Final capability validation

The active objective is to establish all four capabilities requested by the user:
learn what computation to perform, assign credit across longer sequences, grow
representations, and learn the components together while retaining earlier skills.
None is currently established at that full scope. This document defines tests
against the existing mechanisms, not a replacement objective limited to the
experiments that already pass.

The latest handoff is [COMPLETE_LEARNER_HANDOFF.md](COMPLETE_LEARNER_HANDOFF.md).
It records the completed retention fix and component repair/search validation,
plus the [completed repair integration](REPAIR_INTEGRATION_RESULTS.md). The
matched three-seed repeated-record campaign repairs an already available wrong
skill inside the continuing session and passes evidence-revision lifecycle
checks. The earlier set-normalized campaign's timeout remains a failed run.
None of the four gates is closed by these scoped results.

The user permits backpropagation to establish the complete learner first. Use
the same behavioral gates for that implementation and the later non-backprop
comparison. Keep the update rules and supplied priors explicit; passing with
backprop does not establish that a local-only learner has the same capability.

The user also explicitly permits warm starts for goals, delayed credit and
control. The next primary experiment will use a declared supervised bootstrap,
reduce assistance, then test autonomous adaptation with final-outcome feedback.
The final-only restrictions below apply to that independent learning/evaluation
phase, not to the allowed bootstrap. Preserve a from-scratch comparison and
record all demonstrations and hints. See [the warm-start curriculum](WARM_START_CURRICULUM.md).

## 1. Learned computation control

The learner must choose whether to retrieve evidence, continue computation, ask
for information, take an action or return an answer. It may receive public memory
contents, coverage, uncertainty, operand validity and actual operation costs.
It may not receive a correct next-operation label, required-tool flag, hidden
world state, answer, solution trace or task-specific route.

The runtime may implement generic operations and enforce operand validity. The
policy choosing among them must learn from consequences. A supplied procedure
search algorithm does not establish learned search control by itself, nor does
executing a teacher-provided goal graph.

Required evidence: one policy choosing among all operation classes; withheld
worlds and operation sequences; changing information availability and costs;
successful early stopping and continued reasoning when needed; actual learned
parameter changes; and controls with no learning, randomized credit and disabled
operations. Record correctness, operation counts, cost and failures per family.
Always asking or always searching must not masquerade as learned control simply
because the environment eventually provides enough information.

Current blockers: `OnlineGrounding::choose` supplies actuator exploration and
hypothesis-partition search. `ProcedureMemory::learn` supplies program search.
The demonstrated learned runtime controller receives an explicit goal graph.

New scoped evidence: [computation control](COMPUTATION_CONTROL_EXPERIMENT.md)
learns selection among retrieve/reason/ask/act/return without a goal graph, using
terminal outcomes and costs. Local and backprop policies each pass 720 endpoint
tests, including longer paths, but perform unnecessary actions. This fixed
structured domain is not integration with the other learners or general control.

## 2. Credit across longer sequences

Latest scoped progress: [goal-scoped greedy trace credit](GREEDY_TRACE_CREDIT_EXPERIMENT.md)
passes the delayed-propagation contract and a matched three-seed retention screen:
384/384 standard, 96/96 recovery, 36/36 larger goals, 108/108 interrupted parents,
144/144 children and 972/972 primitive cases. Thus the previously observed
contract-versus-short-task tradeoff now has a passing candidate. The longer
contract uses supplied extra reasoning decisions; the existing larger-goal
screen requires only three causal motor actions. Longer sparse-task acquisition,
autonomous interruption/subgoal scheduling and mixed-task coexistence remain open.
The [machine-readable credit record](GREEDY_TRACE_VALIDATION.json) also records
six independently audited phase-control runs with identical pre-ablation warm
states and exact post-boundary controller freezing. The frozen longer-action
preflight already solves its five cases, so it does not establish new acquisition;
eight-effect success is also worse than immediate failure under the current
discounted cost objective. The next [goal-control design](LEARNED_GOAL_CONTROL_NEXT.md)
requires a missing behavior and a cost-feasible reference before training.

The decisive learning arm receives terminal verified outcomes, with declared
generic computation costs. No correct actions or intermediate correctness labels
may be supplied. Demonstration-assisted bootstraps remain separate positive
controls and must not be reported as answer-only discovery.

Revisit the existing native follow, comparison and distinct-count failures, then
require transfer to longer executions and compositions. Record complete decision
traces and distinguish failed exploration from aliased observations, incorrect
credit, unstable learned values and execution limits. More eligibility duration
alone has already failed. Backpropagation is permitted for the initial complete
learner; retain a matched non-backprop comparison as the follow-up objective.

Required controls include truncated credit, shuffled outcomes and frozen weights;
fresh entities, duplicate records and distractors; length extrapolation; and
retention after other skills learn. The learner must acquire the initially
missing skill, not merely execute its demonstrated implementation for more steps.

Current evidence: direct terminal-reward agency and backprop-TD runs still fail
distinct-count discovery in their tested configurations. The state abstraction
aliases erroneous program states, including different accumulator values. The
computation-control test also passes with terminal-decision-only updates, so its
long-path success alone does not establish this capability.

The [primitive-discovery experiment](PRIMITIVE_DISCOVERY_EXPERIMENT.md) supplies a
working alternative: whole-policy search against final answers discovers follow,
equality and distinct count without external instruction demonstrations. Three
seeds pass 8,100 primitive cases, including 64-object counts and up to 353 executed
instructions. Self-generated rehearsal transfers these policies to Coco's shared
neural skill learner, retains prior skills and supports 4,200/4,200 questions in
the existing ProcedureMemory, including 2,400 composites. Removing rehearsal loses the earlier
skills. This closes the specific demonstrated counting-discovery gap in the
structured VM; broader delayed credit in the integrated language/action learner
remains unproven. Search, the state abstraction and primitive semantics are supplied.

The [composed-credit follow-up](COMPOSED_CREDIT_EXPERIMENT.md) learns the missing
counting skill from composed final answers without direct counting lessons.
Three Boolean-curriculum seeds pass 1,680/1,680 composed tests after neural
rehearsal, 324/324 standalone count tests and 648/648 retained earlier-skill tests.
Half the composed tests concern additional forms taught by final answers after
the primitive weights freeze. Without old-skill rehearsal, composed success is
280/1,680. Numeric-answer acquisition succeeds in two of three seeds. Search
costs billions of simulated instructions per run and considers unavailable slots
in a supplied bank; repair of already available wrong skills, multiple missing
components and this credit route inside the language/action stream remain open.

Follow-up [available-skill repair](PRIMITIVE_REPAIR_EXPERIMENT.md) corrects an
already available but underspecified count policy from composed final answers.
At 100,000 proposals per candidate slot, three seeds pass 1,680/1,680 composed
tests and 972/972 primitive tests after neural rehearsal. At 5,000 proposals per
slot only one seed succeeds. Each complete larger run searches all three slots
and spends 300,000 proposals. That initial experiment was a component result.
The subsequent [repair integration](REPAIR_INTEGRATION_RESULTS.md) establishes
the scoped continuing-session repair and evidence-revision lifecycle across
three seeds: affected control improves from 0/96 to 96/96, earlier-task retention
passes 384/384 and primitive checks pass 972/972. Multiple simultaneous repairs,
noisy evidence, learned search scheduling and practical search cost remain open.
The subsequent [successive repair bank](SUCCESSIVE_REPAIR_BANK.md) retains two
already available repaired primitives in one neural memory, first from raw
composed outcomes and then through structured language receipts. The focused
seed-1 session passes 216 count-language checks across acquisition, retention
and withdrawal, 12 fresh follow-language checks, 26 primitive probes and all
11 original direct outcomes, with exact weight restoration. It requires
separable successive counterexamples; simultaneous discovery of interacting
missing components and a multi-seed control/action continuation remain open.
This supplied search mechanism does not resolve the delayed-credit contract's
TD/episode-return tradeoff by itself or establish autonomous interruption
scheduling. The separate greedy-trace follow-up above resolves that specific
contract/retention tradeoff within its tested scope.

## 3. Representation growth

Latest scoped progress: [outcome-induced reusable rule unions](DISJUNCTIVE_CONCEPT_GROWTH.md)
extend the old single-conjunction class. Across three seeds, the learner passes
768/768 binary acquisition/retention questions and 768/768 reused ternary
compositions, with independent semantic and macro-expansion audits. The old
single-conjunction class cannot express the new binary meaning at any body width.
Session language, action-branch planning and evidence revision also pass focused
contracts. Disjunction is supplied generic machinery; head names/arities remain
provided, search remains bounded, and the full growth gate remains open.

Introduce genuinely new relations and concepts after learning has started.
The learner must determine their meanings from observations and consequences,
retain alternatives when evidence is insufficient, and reuse acquired concepts
in withheld compositions. Allocating another opaque relation ID or adding a
surface-language alias to an existing fixed meaning is insufficient evidence.

Required evidence includes learning beyond the original two relation channels,
new relational combinations and role structures, distinguishing contrasts,
measured growth in stored representations, independent semantic checks, and
transfer after renaming entities and surface words. Compare against a fixed
representation budget/bank, and verify that prior concepts remain usable.
Resource bounds must be explicit; a bound must not conceal a pre-enumerated
inventory containing every test concept.

Current blockers: `LanguageReasoner` exposes two grounded relation channels;
bootstrap/incremental learners enumerate a small fixed operation/type space.
The flat query's two relation arguments are an interface limit distinct from
the number of relation identities a future memory could represent.

New scoped evidence: the [concept-growth experiment](CONCEPT_GROWTH_EXPERIMENT.md)
learns eight positive relational definitions with two, three and four head roles,
reuses them in later definitions, and introduces new unary/ternary observed
schemas. Three renamed vocabularies pass 4,800 fresh final queries and independent
definition-equivalence checks. Reuse permits eight-base-literal concepts within
a three-local-literal rule budget; a flat three-literal control passes 1,200/4,800,
while a wider eight-literal flat control also passes all cases. Retention,
ambiguity, partial-world withholding and identified corrections pass scoped tests.
This is supplied conjunctive-query induction from structured complete-world
teaching, with new query arities provided by the host. It is not unrestricted
concept invention or integration with the existing language/control/action model.

The subsequent [concept-isolation continuation](CONCEPT_ISOLATION_EXPERIMENT.md)
does integrate a further new meaning with the recommended controller and action
learner. Removing a global ambiguity gate preserves 384/384 old-task answers
while the concept is unresolved and 384/384 after learning it, with 192/192 fresh
new-concept action goals and 972/972 retained primitive cases across three seeds.
The same sessions continue neural updates and acquire new actuator procedures
without further hints. A separate contract verifies a new three-role meaning,
contrast withdrawal/restoration and unaffected old language/procedures. This is
bounded named-concept induction and integration evidence; broader representation
growth and the remaining joint-learning requirements are not closed by it.

## 4. One continuing learner with retention

The [learned-reader boundary](GROUNDED_RELATIONAL_BOUNDARY.md) now preserves actual
learned report alternatives, negative facts, scoped coverage and evidence/model
versions when exposing a conservative observation to the same controller.
Focused tests cover raw-report teaching, ambiguity, archived revisions and stale
snapshot rejection. Controller operations in those tests are protocol fixtures.
The full byte predictor, partial-world action planning and atomic replay of
downstream teaching after reader reinterpretation are still unintegrated.

Language, procedures, control, evidence revision and action learning must update
inside one continuing model. A collection of separate successful probes, a
frozen symbolic front end beside a learning byte predictor, or a staged run that
keeps every interacting component fixed does not satisfy this requirement.

Use an interleaved stream with new meanings, new procedures, new action effects,
partial observations, old-skill requests and corrections to earlier teaching.
The evaluator can return verified outcomes but cannot choose the correct plan,
operation or hidden state for the model. Inventory every supplied prior and
every frozen component. Log model updates and retain versioned teaching evidence
so a correction can invalidate dependent knowledge and stale decisions.

Required evidence: exact behavior and knowledge-state audits on scoped
deterministic cases; retention before and after subsequent learning; withheld
combinations and longer plans; independent checkpoint probes; matched ablations;
and repeated seeds. Report every failed family rather than hiding it in an
aggregate score. Broader continual-learning claims remain open when only a
restricted subloop has been exercised.

Current progress: verified observation teaching runs in the standard byte trainer.
Cross-episode provenance replay passes sanitizer contracts and an independent
three-seed audit: 351 replies, 5,472 knowledge states and 113,040 evidence-version
snapshots per seed, including recovery from old mistaken lessons. The actual
trainer accepts all 313 receipts and finishes with identical models/state with
or without intermediate probes. This work supports evidence revision and
retention; it does not establish joint acquisition of all components.

The primitive-discovery follow-up adds sequential acquisition and retention of
three skills in one neural policy, followed by answer-based procedure learning.
Its no-rehearsal control forgets earlier skills. This is a concrete retention
subloop, but language, evidence revision, control and physical-action learning
are not updating alongside it, and the archive still guards a fixed bootstrap.

The composed-credit follow-up adds a versioned typed-query learner that acquires
counting from whole composed outcomes, learns a later procedure with unchanged
primitive weights, invalidates derived skills after contradictory or withdrawn
support, and revalidates them on restoration. Native lifecycle contracts and
independent final-checkpoint tests pass. This acquisition route is not yet
connected to the continuing language/concept/action/controller session, so it
does not close this gate.

Follow-up: [composed session integration](COMPOSED_SESSION_EXPERIMENT.md) now
connects this route to that same session. Two completed seeds each pass 128/128
final controller cases, their recovery cases, 12/12 larger goals and 324/324
standalone primitive checks. A third seed fails repeated 100,000-proposal
acquisition attempts. The connection is implemented and validated for successful
seeds; reliable acquisition, practical compute, broader mixed-stream retention
and the full gate remain unproven. The tested configuration is preserved
in [the canonical trainer](CANONICAL_TRAINER.md).

The [capacity comparison](CAPACITY_EXPERIMENT.md) recommends the 50k MLP with
original-size Agkern and TD after all three seeds pass both the 400- and
1,600-episode schedules. The [tapered bootstrap](WARM_START_CURRICULUM.md) also
passes all 384 final, 96 recovery, 36 larger-goal and 972 primitive checks with
that size, with no hints in evaluation or the final 240 training episodes per seed.
The [interruption follow-up](INTERRUPTED_GOALS_EXPERIMENT.md) retains goals across
108 parent tasks and 144 intervening questions, with frozen learned operation
choices, separate action segments and no invented intermediate goal labels.
The host still schedules goal switches. A delayed-credit contract exposes slow
TD propagation after 16 intervening learning episodes; episode-return credit
passes that contract but lowers three-seed task scores. These are concrete new
mechanisms and evidence, not closure of all four capability requirements.

The concept-growth module adds another acquisition/revision subloop: new
relational definitions coexist with retained old definitions, and raw-evidence
rebuild can invalidate or independently re-learn dependent concepts after a
correction. It remains separate from the grounded session and neural learner.
The [relational-session experiment](RELATIONAL_SESSION_EXPERIMENT.md) now couples
concept acquisition, variable-arity language, the existing action-effect learner,
own-execution procedures and evidence revision in one continuing model. Three
seeds pass 2,304 fresh final language queries and 144 fresh final action goals;
72 tight-budget procedure transfers pass versus 25 for search on identical model
snapshots. Raw-evidence replay restores the exact coupled model after an old
concept correction, and stale decisions cannot accept partial credit. Exploration
is still supplied (only 1/24 exploratory actions satisfies its current goal), and
the learned computation controller and neural primitive policy are not updating
inside this session. Plans execute at most three actions in this test. This is
concrete joint symbolic acquisition, not completion of the full coexistence,
computation-control or longer-credit requirements.

## Evidence and completion

The latest [repair integration](REPAIR_INTEGRATION_RESULTS.md) resolves the
previously unfinished available-skill repair connection for the tested stream.
All three repeated-record seeds pass all eight lifecycle stages as well as the
control, retention and primitive checks above. Frozen archived programs and
weights evaluated with the updated observed-predicate language adapter answer
770/771 within-capacity held-out questions, withholding one; another 69 are
outside the declared capacity. This language probe is not a controller rollout
or three-seed retraining with the updated adapter. One full updated-adapter
continuation passes and preserves the earlier seed's 476 behavioral artifacts.
The remaining joint gate requires broader mixed-stream acquisition, including
the still-open control, long-credit and representation-growth capabilities and
integration with the full byte reader. These results do not close that gate.

The latest [control-transfer diagnosis](CONTROL_TRANSFER_DIAGNOSIS.md) resolves
the baseline's six larger-goal failures: retaining search allowance when only the
observation revision changes gives 384/384 standard, 89/89 recovery and 36/36
larger cases. Training traces, value tables and weights are byte-identical to the
preceding baseline. This retains an allowance, not a search frontier, and the
controller still explores before reasoning again. It does not establish further
reasoning under a persistent exhausted-search status or close the other learning
gates. Trees fitted to the same empirical preferences perform worse; inspection
finds untried operations winning the value table in 672/4,237 size-input states.

The preceding [computational-control experiment](COMPUTATIONAL_CONTROL_EXPERIMENT.md)
trains the neural controller while the same session acquires primitive skills,
computational language, concepts and effects. Three seeds pass 384/384 final
standard cases and 89/89 recovery cases, versus 0 with frozen or shuffled control
and 77/384 standard cases with final-decision-only credit. They pass only 30/36
larger goals and do not choose further reasoning after search exhaustion in those
evaluation groups. An additional public-size-input arm does learn further
reasoning, but passes only 21/36 larger goals because some policies fail to act
after completing a plan. The primitive checkpoints remain intact and identical
to separately audited retained-skill checkpoints. This establishes a working
joint acquisition/control loop and identifies remaining transfer failures; it
does not close all four gates or replace the remaining credit/growth requirements.

The [computational-session follow-up](COMPUTATIONAL_SESSION_EXPERIMENT.md) connects
primitive acquisition to the same relational evidence lifecycle. Three seeds pass
1,008 computational language questions, 120 physical goals that require neural
counting/comparison over a newly learned concept, and 648 retained primitive
cases. Primitive withdrawal and concept/terminal-evidence corrections invalidate
dependent meanings/procedures; restoration recovers them. An initial 0–3-size
primitive curriculum passed its own lessons but failed a four-object count in
composition training; wider lessons fixed that tested case. This curriculum
repair was supplied by the experimenter. Joint learned controller performance
on these tasks, automatic cross-level credit and broad representation growth
remain unproven. The following older results retain their original scope.

The [reactive controller](RELATIONAL_REACTIVE_EXPERIMENT.md) now supports repeated
act/observe/reason cycles with terminal-only task feedback. A neural arm passes
354/384 final cases, 120/150 cases with three failed motor attempts, and 47/48
unavailable-sensor cases. Removing retrieval reduces final success to 299/384.
A learned consequence-memory arm passes 117/150 recovery cases versus 13/150
with updates confined to final decisions, 16/150 with shuffled credit and 0/150
with frozen control or one physical batch. Successful neural recovery reaches
60 decisions, beyond the training cap of 48. These establish scoped recovery,
useful retrieval and nonterminal control credit, not the missing general
primitive acquisition or reliable whole-model retention. External interruptions,
the primitive-skill learner and broader representations remain unfinished.

The earlier [one-batch baseline](RELATIONAL_CONTROL_EXPERIMENT.md) integrates a
neural operation selector with the continuing symbolic model. Empirical TD credit
and backprop policy fitting reach 308/384 final cases across three seeds, compared
with direct neural credit 103/384, frozen weights 0/384, shuffled outcomes 1/384
and terminal-decision-only updates 177/384. Queries pass 192/192; action goals
pass 116/192. Retrieval removal does not reduce success, and intermediate
checkpoints remain unstable. It establishes a working joint interface and
partial learned control, not reliable computation control, long primitive
acquisition or full retention. Only one physical batch is allowed per episode;
the primitive skill learner remains separate. All four broad gates stay open.

Preserve negative controls and failed runs. Verify that test data and expected
answers cannot enter the model's inference interface. Check raw outputs and
states, not only a verifier's summary. A mechanism counts as established only
within the scope its tests exercise; finite controlled tests are not proof of
unrestricted intelligence.

The overall objective remains open until there is positive, reproducible evidence
for every requested capability and for their coexistence in the same model.
The current architecture/readiness reports document component evidence and must
not be used as a substitute for that combined validation.
