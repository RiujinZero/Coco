# Primitive readiness — 2026-09-23

The basic execution mechanisms work in the controlled grounded domains tested
so far. Their sufficiency for a complete, continually learning Coco is unproven.
There is no defensible percentage-complete estimate: the remaining learning and
control problems are not interchangeable checklist items.

For the current handoff, read [COMPLETE_LEARNER_HANDOFF.md](COMPLETE_LEARNER_HANDOFF.md)
first. The [available-skill repair follow-up](PRIMITIVE_REPAIR_EXPERIMENT.md) now
passes three seeds at the component level, and the
[mixed-frame retention lifecycle](CONCEPT_ISOLATION_EXPERIMENT.md) passes after
support-aware cache validation. Repair in the complete continuing trainer is
still unvalidated; the historical milestones below retain their original scope.

The latest integration connects neural primitive acquisition to concepts,
controlled language, action effects and own-execution procedures in one revisable
session. The [combined controller experiment](COMPUTATIONAL_CONTROL_EXPERIMENT.md)
now trains operation choice while these components arrive: three seeds pass
384 standard cases and 89 recovery cases, but only 30/36 larger goals. An
additional size-input arm learns further reasoning but passes only 21/36 larger
goals. The primitive checkpoint remains identical to the independently tested
one that passed 1,008 questions and 648 retained-skill cases. Reliable control
transfer and broader representation/credit learning remain open. The matched
[control-transfer follow-up](CONTROL_TRANSFER_DIAGNOSIS.md) fixes the baseline's
six larger-goal failures by retaining search allowance across unchanged-scene
revisions, with identical training and weights. It passes 36/36 larger goals
without losing standard/recovery success. This is an optional frozen-evaluation
runtime change; general primitive repair, growth beyond supplied interfaces,
and the full continuing-learning test remain required.

The [composed-credit experiment](COMPOSED_CREDIT_EXPERIMENT.md) now acquires an
unavailable counting skill from whole composed Boolean answers without direct
counting lessons. Three seeds pass 1,680/1,680 composed tests, 324/324 standalone
counting tests and 648/648 retained follow/equality tests after neural rehearsal.
A versioned typed-query wrapper passes subsequent procedure learning and evidence
withdrawal/restoration contracts. Numeric-answer acquisition succeeds in only
two of three seeds, and search costs billions of simulated instructions per run.
Repair of an already available wrong skill, new operation interfaces and
reliable acquisition across initializations remain unproven. The
[composed-session follow-up](COMPOSED_SESSION_EXPERIMENT.md) now connects this
acquisition route to the language/action/controller stream. Two completed seeds
pass all final, recovery, larger-goal and 324 primitive cases per seed; another
fails repeated acquisition attempts. [Canonical trainer consolidation](CANONICAL_TRAINER.md)
preserves the tested settings and replayable evidence before the next capability
experiment. Per the user's direction, [warm starts](WARM_START_CURRICULUM.md)
are allowed for the remaining goal, credit and control blockers, followed by
withdrawal of assistance and independent adaptation. This is scoped progress,
not completion of the learner.

| Capability | Current evidence | Remaining requirement |
|---|---|---|
| Local plasticity and action selection | Native sparse expansion, consequence routes, eligibility and three-factor updates train policies. Whole-policy search from final answers discovers follow/equality/count without external instruction traces; self-generated rehearsal transfers them into one native neural policy and preserves earlier skills. Missing count can also be acquired from composed final answers, with no direct count lessons. | General repair and acquisition across the complete reading/reasoning/action loop at practical search costs. Direct step-level outcome learning still fails counting in tested local and backprop configurations. Search and the state abstraction remain supplied. Some older pretrained follow policies still lack zero-edge coverage; the new discovered policies test that case. |
| Symbolic working memory and binding | Entity references, mutable relations, persistent returned values, call-local scratch, cursor/set/counter, typed call/return and unique reads of complete scopes; binding ablations break compositions. Learned frames retain event participants even after their facts change. | Richer dynamically learned representations and scalable retrieval. The current text learner has two relation channels, bounded argument slots and a small type/operation vocabulary. |
| Computation and composition | Learned primitives and multi-example procedure discovery transfer to withheld compositions. With outcome-discovered initial skills, the native ProcedureMemory passes 4,200/4,200 questions, including 2,400 composites; without rehearsal, only counting survives. A separate learned controller selects retrieve/reason/ask/act/return in a bounded structured environment. | Unified learned control of search, subgoals, information gathering and budgets. Policy/procedure discovery and actuator exploration still use supplied algorithms; executable affordances and current state features are supplied. |
| Observation and belief handling | Facts, negatives, coverage, possible worlds and references have targeted tests. Live report learning preserves alternative histories under partial observation. A bounded provenance journal restores discarded meanings after corrections and rebuilds current/historical knowledge, preserving later events and invalidating stale credit. Historical-answer teaching grounds report references; consequence answers now ground whole correction constructions that identify references/payloads and select preserve, withdraw or replace. Ambiguous report selection supports clarification. | Broad temporal language, learned source reliability, efficient persistent dependency memory, correction discovery and broader simultaneous learning under partial observation. Correction span types and operation vocabulary remain supplied, with prior report/question/reference meanings frozen; learning new actuator effects still requires complete states. |
| Language grounding | Raw-story/final-answer induction learns controlled event and question forms, aliases and reusable phrases; every canonical grounded dialogue now passes. Verified satisfaction feedback learns state-goal request forms, including resolving a wrong-recipient ambiguity. | Open vocabulary/concept discovery, broad constructions, more relations/arguments, and transfer across domains. Whole-message frame learning is not unrestricted sentence understanding; state predicates cannot express required event histories. |
| End-to-end operation | An optional byte-stream session connects questions, report journaling and correction dispatch to `laminar_train`, with private grounded state for probes. Verified feedback teaches observation meanings across resets while byte learning runs: 600/600 transfer replies per seed. An optional raw-evidence archive now supports correction of old lessons, with 351/351 replies and 5,472 state checks per seed. Separate live experiments learn request/effect/report meanings while acting; issued actions await observed outcomes. | Broader joint acquisition, integration with the physical-action learner, disk persistence, longer plans and noisy feedback. Question/reference/correction meanings and primitive policies remain pretrained. The archive requires a fixed bootstrap and bounded global replay. |

Symbolic memory can remain a permanent part of the architecture. Completion does
not require replacing every symbolic operation with a neural approximation.
What matters is whether Coco can acquire and use those operations autonomously,
under bounded computation, with evidence that survives new tasks and vocabulary.

The live observation experiment connects a restricted version of that loop with
request, action and report meanings learned during acting. The live evidence
experiment adds visibility-language acquisition while previously learned actions
operate under partial observation. The highest-value remaining integrations are
broader joint learning, expanding bounded report references into richer temporal
concepts and learned source reliability, and learning control beyond
single-predicate goals and supplied exploration. The evaluator may score the true outcome, but
must not supply the correct plan, hidden state, program or task-specific route.
Demonstration-assisted and answer-only learning should be measured separately.

A completion claim needs more than isolated passing probes: the same running
model must learn through that interface, recover from mistaken interpretations,
transfer to withheld combinations and worlds, and retain earlier skills over a
long mixed stream. The existing results establish useful components, not that
this end state has been achieved.

Primary local evidence:
[architecture](ARCHITECTURE.md),
[primitive learning](REASONING_PRIMITIVES_EXPERIMENT.md),
[composition](REASONING_COMPOSITION_EXPERIMENT.md),
[incremental language](INCREMENTAL_GROUNDING_EXPERIMENT.md),
[environment dialogue](DIALOGUE_GROUNDING_EXPERIMENT.md),
[request/action grounding](REQUEST_GROUNDING_EXPERIMENT.md), and
[online grounding](ONLINE_GROUNDING_EXPERIMENT.md), followed by
[live observation grounding](LIVE_OBSERVATION_GROUNDING_EXPERIMENT.md),
[visibility language](OBSERVATION_LANGUAGE_EXPERIMENT.md) and
[live partial-observation learning](LIVE_EVIDENCE_GROUNDING_EXPERIMENT.md), followed
by [identified corrections and history replay](REVISION_GROUNDING_EXPERIMENT.md).
The [historical-reference experiment](HISTORY_REFERENCE_GROUNDING_EXPERIMENT.md)
adds learned report selection: each of three seeds passes 144 correction cases,
including 84 accepted edits, and 1,596 current/historical answers. The primitive
policies and body remain frozen; these results do not close the autonomous
control or joint-learning gaps.
The [correction-language experiment](CORRECTION_LANGUAGE_EXPERIMENT.md) removes
the separately supplied operation and reference/payload boundaries: five learned
constructions transfer to new combinations, with 252 request cases, 120 accepted
edits and 1,764 current-state answers audited per seed. A sixth ambiguous
construction withholds. This is staged symbolic learning over known span types;
it does not resolve the same broader control and continual-learning limits.
The [grounded-session experiment](GROUNDED_SESSION_EXPERIMENT.md) connects these
bounded question/correction abilities to the actual trainer, preserving evidence
history across turns and isolating checkpoint probes. The follow-up
[grounded-teaching experiment](GROUNDED_TEACHING_EXPERIMENT.md) adds verified online
observation-language teaching to that same trainer. Three seeds pass 600 transfer
replies each after learning four frames across eight episodes; the no-feedback
control gets 184/600. The [archive experiment](GROUNDED_ARCHIVE_EXPERIMENT.md) adds
bounded editable cross-episode provenance and audited recovery from mistaken old
lessons. Physical actuation and broader joint learning remain separate work.

The [computation-control experiment](COMPUTATION_CONTROL_EXPERIMENT.md) learns
retrieve/reason/ask/act/return selection in a supplied structured environment.
Local-plasticity and backprop policies each pass 720/720 endpoint tests across
three seeds, including longer paths. Both take unnecessary actions. A control
that updates only terminal decisions also passes, so long execution here does
not establish learning an initially missing skill from delayed credit. This
controller is not yet integrated with the language/action learners.

The [primitive-discovery experiment](PRIMITIVE_DISCOVERY_EXPERIMENT.md) addresses
the earlier counting failure through whole-policy search against final answers,
then self-generated neural rehearsal. Three seeds pass 8,100 primitive cases and
4,200 questions, including 2,400 composites, without external initial-skill instruction traces.
Removing rehearsal loses following and equality while counting learns. This is
a verified acquisition/retention subloop in the existing ProcedureMemory, not a
joint language/control/action learner or proof of new concept formation.

The grounding route uses local plasticity plus symbolic search, without
end-to-end backpropagation. The new controller comparison and separate
attention/hybrid experiments use backprop; they are not the mechanism behind
the grounding results. The user permits backprop to establish the complete
learner first, before a matched non-backprop demonstration. All four broader
[capability gates](FINAL_CAPABILITIES_VALIDATION.md) remain open.

The [concept-growth experiment](CONCEPT_GROWTH_EXPERIMENT.md) now demonstrates
bounded formation of new relational definitions beyond the reader's fixed
relation channels. Eight learned concepts introduce up to four argument roles,
reuse earlier definitions and retain their meanings after later teaching and
identified corrections. Three opaque vocabularies pass 4,800/4,800 fresh final
queries; the independent checker also verifies the definitions' semantics. A
three-literal flat control passes 1,200/4,800, and an eight-literal flat control
passes all cases. The result establishes reusable representation growth in a
supplied positive-conjunction language, not open concept invention. The module
does not yet expand the original text reader or update alongside neural skills
and learned computation control.

The [relational-session follow-up](RELATIONAL_SESSION_EXPERIMENT.md) now connects
growing concepts, new language forms, learned action effects and own-execution
procedures in one versioned model. Across three seeds, it passes 2,304/2,304 fresh
language queries and 144/144 fresh action goals while learning continues. Learned
procedures pass 72/72 tight-budget transfers versus 25/72 for search on identical
snapshots. Corrections restore the exact coupled models from raw evidence; stale
decisions cannot accept credit. The remaining integration is the neural primitive
learner and learned retrieve/reason/ask/act/finish controller, with longer mixed
tasks. This session still supplies search, exploration and rebuild scheduling,
and uses complete observations for effect learning and planning.

The [relational-control experiment](RELATIONAL_CONTROL_EXPERIMENT.md) now adds
learned operation selection to that session. Empirical TD credit plus backprop
policy fitting passes 308/384 final cases (192/192 queries, 116/192 action goals),
compared with 103/384 for direct neural credit. Most failures are action/control
loops; removing retrieval leaves success unchanged. Symbolic learning and neural
control now coexist, but reliable control, action recovery, useful retrieval,
stable retention and integration of the neural primitive skills remain open.

The [reactive follow-up](RELATIONAL_REACTIVE_EXPERIMENT.md) removes the one-batch
limit in an optional body mode and verifies recovery from failed motor attempts.
The neural controller passes 354/384 final cases and 120/150 three-failure recovery
cases; useful retrieval is now required in unavailable-sensor cases. Longer
successful control traces and matched credit controls provide scoped delayed-credit
evidence. Some action tasks and intermediate checkpoints still fail; this is not
integration of the missing neural primitives or completion of the four broad gates.
