# Discovering primitives from outcomes, then retaining and composing them

Tested 2026-09-23. The initially missing distinct-counting procedure can now be
discovered from final answers in the existing structured VM, without external
instruction demonstrations. Three seeds discover following, typed equality and
distinct counting, transfer them into Coco's actual local-plasticity policy, retain
the earlier skills while learning later ones, then solve 4,200 questions across
seven primitive/composition families through the existing native ProcedureMemory.
Of those, 2,400 require composing multiple learned skills.

The working mechanism is **whole-policy search against several verified answers,
followed by self-generated instruction rehearsal**. It is not an improvement to
the original single-action reward broadcast or a proof of end-to-end backprop.
The supplied search and VM remain explicit. Broader representation growth and the
complete joint learner in [FINAL_CAPABILITIES_VALIDATION.md](FINAL_CAPABILITIES_VALIDATION.md)
remain open.

## Discovery, with no teacher-written execution traces

The existing ten instructions are BIND, READ, COPY, MARK, INC, COMPARE and four
typed returns. READ returns one matching record; MARK and INC are separate.
The controller sees the existing 32-cell binary abstraction: an opaque skill cue,
register/coverage/set-membership flags, predicate, last instruction and bias.
It does not receive the answer, cardinality, chain length or a correct next step.
Numeric accumulator values remain absent from the features. Equality is a
two-value call without accessible fact memory, as in the native reasoning runtime.

PolicyDiscovery proposes finite maps from these public observations to actions.
The initial maps have random decisions. Proposals change one to four decisions
encountered in earlier executions. The evaluator executes each candidate on 32
short examples and returns aggregate final correctness, actual execution cost,
and the observations/actions the candidate itself encountered. Those recorded
actions are not teacher suggestions. There are no intermediate correctness
rewards, supplied instruction sequences, or counting-loop templates.

Each family has eight examples at each size 0/1/2/3. Facts are shuffled and contain
distractors; counting includes duplicate records. The search keeps up to 128
behaviorally distinct candidates, restarts exploration after 20,000 evaluations
without a fully correct policy, and has a 100,000-candidate budget. Once all
training answers are correct, lower execution cost breaks ties. Each rollout
has a 48-instruction training budget.

The retained policy is sealed to the states actually observed during training.
An unknown state or unavailable selected operation withholds rather than falling
back to a random rule. The model contains state/action rules, not stored example
answers. Entity and relation identities remain runtime bindings outside these
features; their renaming invariance is supplied by the adapter.

## Longer executions and failure controls

Each seed evaluates 900 new worlds per family, with sizes 0/1/2/3/4/8/12/24/64.
Counting increases duplication to as many as five copies per object. The test
instruction budget is 1,024, independent of the hidden answer/size. All three
families pass 900/900 in every seed: 8,100/8,100 total.

Counting alone passes 2,700/2,700, including 1,500/1,500 cases above training size.
Its longest successful execution takes 353 VM instructions. Thus the result
addresses discovery of the previously missing loop and reuse over longer
executions, not only execution of a teacher-provided counting routine.

The counting controls use the same test worlds:

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Full policy search, then cost refinement | 900/900 | 900/900 | 900/900 |
| Stop at first training-consistent policy | 800/900 | 900/900 | 900/900 |
| Propose mutations only at final-decision states | 900/900 | 223/900 | 794/900 |
| Randomized terminal correctness | 100/900 | 0/900 | 0/900 |
| First random candidate, no selection learning | 0/900 | 100/900 | 100/900 |
| Direct backprop TD | 100/900 | 0/900 | 100/900 |

The first consistent seed-0 policy repeatedly restarts scanning earlier records.
It answers short cases correctly but times out on all 64-object cases. Cost
refinement discovers a faster policy and fixes that failure. The optimized
policies are not claimed to be globally minimal.

The final-decision-state control truncates which observed states may be mutated.
Whole-policy restarts and aggregate terminal evaluation remain available, and a
rule shared with an earlier position can still change earlier behavior. Its
occasional success is therefore not proof that earlier decisions need no credit.
The randomized arm replaces each episode's correctness bit with an independent
random bit; the frozen arm is one initial random candidate, not a chance estimate.

Direct backprop uses the tested 32–32–10 tanh network, squared TD error, learning
rate 0.005, discount 0.99, instruction cost 0.002 and 20% random exploration for
60,000 fresh counting episodes. It fails every nonzero count in these runs.
This is one configuration, not a claim that backprop cannot learn counting.
The search has many more repeated episode evaluations, uses a different credit
and exploration mechanism, and does not rely on a Markov value estimate for
aliased erroneous VM states. These are not compute- or sample-matched benchmarks.
Backprop remains permitted for the complete-learner objective.

Operation-removal tests additionally disable bind, read, copy, mark, increment,
compare or cursor advancement. Their raw failures/survivals are retained and
independently replayed; they are not substituted for training.

## One neural skill policy, with a retention control

The discovered policies generate their own traces on 256 fresh short worlds per
skill. Each complete answer must pass the external evaluator before its trace is
admitted. The traces are deduplicated by public state and action and fitted into
the existing LocalChoice sparse expansion/consequence routes using its existing
three-factor fit update. These internal action targets come from the learner's
successful discovered policies; no file of teacher micro-instructions is read.

The same neural weights learn following, then equality, then counting. Each
stage runs 128 shuffled passes. The rehearsal arm includes previously acquired
decisions; the control trains only the newly acquired skill. Independent data and
shuffle RNG streams ensure both arms receive identical new-skill worlds and
self-generated action examples. Rehearsal does additional updates for old skills;
this is not an equal-compute comparison.

After every stage, earlier skills are tested with frozen weights. Across three
seeds:

| Check | Rehearsal | No rehearsal |
|---|---:|---:|
| Following after equality is learned | 2,700/2,700 | 0/2,700 |
| Following after counting is learned | 2,700/2,700 | 0/2,700 |
| Equality after counting is learned | 2,700/2,700 | 0/2,700 |
| Counting after counting is learned | 2,700/2,700 | 2,700/2,700 |

These checkpoints reuse the same held-out worlds across stages to measure
retention. They are not additional independent problem sets. Weight snapshots
confirm that every learning stage changed the shared policy and that evaluation
did not update it. This supplies a causal forgetting/rehearsal result in the
three-skill subloop, not general lifelong retention.

## Composition through the existing reasoning memory

The resulting neural policy is the skills member of the same native
ProcedureMemory used by the grounded reader. Its primitive weights are frozen
while that memory learns seven procedure families from twelve final-answer
examples per opaque question cue. The supplied bounded program search remains
unchanged: up to eight instructions and 50,000 candidate states.

No teacher skill-trace corpus is loaded. On the existing independent composition
corpus, training paths are 1–3 links and counts 0–3; tests use paths 4/8/12/32 and
counts 4–24. Root and endpoint counts vary independently.

| Family | Rehearsal, three seeds | No rehearsal, three seeds |
|---|---:|---:|
| Follow | 600/600 | 0/600 |
| Count | 600/600 | 600/600 |
| Equality | 600/600 | 0/600 |
| Follow, then count | 600/600 | 0/600 |
| Compare counts | 600/600 | 0/600 |
| Follow both paths, count, compare | 600/600 | 0/600 |
| Compare path endpoints | 600/600 | 0/600 |

This removes the earlier composition result's dependency on externally
demonstrated initial skills. It does not remove supplied search, the fixed VM
operation bank, the supplied state abstraction or opaque query cues. Physical
actions, natural-language learning and evidence revision do not run in this
experiment.

## Independent evidence and reproduction

The independent Python interpreter reconstructs every logged primitive action,
state key, cursor/set/counter update, timeout, unavailable decision and answer.
It recomputes expected answers from facts rather than trusting stored labels,
checks sealed rules against the training executions, and checks that internal
teaching targets came from the discovered policies. It verifies identical
new-skill streams between rehearsal arms and changed/frozen weight snapshots.

A separate reference interpreter recomputes all composition labels and executes
the learned programs against training and test facts. ASAN/UBSAN contracts cover
policy induction, atomic sealing, invalid fitness/observations, withholding on
unknown states/unavailable operations, immutable inference and clone isolation.

Run bash scripts/primitive_discovery.sh from the repository root in WSL. The
script can generate the earlier flat-question corpus if absent; its export path
writes final-answer examples without calling its demonstration-training path.
Sources, binaries, inputs, models and raw outputs are hashed in
logs/primitive_discovery/audit.json. Full cases/decisions, search curves, frozen
models, weight snapshots, controls and composition outputs are retained alongside
it. No public curriculum files or production trainer defaults are changed.

The next integration must let this acquisition/retention mechanism coexist with
learning language, evidence revisions, control and action meanings. The current
grounded archive still requires fixed bootstrap skills. Dynamic concepts and
relations also remain outside this finite primitive interface. The complete
four-capability objective is therefore still open.
