# Learned neural primitives inside the continuing relational learner

Three seeds pass **1,008/1,008 withheld computational language questions** and
**120/120 physical planning goals** whose truth depends on a newly learned
relational concept and neural counting/comparison. The shared neural policy
retains all three primitive skills on **648/648 additional cases**, including
size 64 and up to 339 executed microinstructions. These results connect primitive
acquisition to language, concept revision and physical planning in one session.
They do not establish the complete learner or learned control over this new task
distribution.

## The connection

`PrimitiveStore` accepts versioned raw problems and final typed answers, without
instruction demonstrations or policy files. Whole-policy search discovers the
microinstruction choices. The accepted policy generates its own successful
traces; rehearsal fits Coco's actual shared `LocalChoice` neural routes. Inference
in `ProcedureMemory` uses those neural routes, not the discovered policy table.

`RelationalSession` now owns this store by value. Primitive changes increment
the same session version as changes to concepts, language and effects, making
pending decisions stale. Rebuild regenerates primitive rehearsal, learns the
computational language programs from current raw receipts, and revalidates
physical procedures. A missing or withdrawn primitive makes its dependent
programs unavailable. Evaluation clones own their neural weights and scratch.

Computational language uses supplied entity/relation mention types, two entity
slots and two relation slots. The learner groups utterance patterns and searches
for a program consistent with final Boolean answers. No program, required
primitive sequence or evaluator family number enters this interface. Three
opaque forms learn these semantics:

| Form | Meaning learned from world/answer examples |
|---|---|
| `vex a with b via r and s` | Compare distinct `r` neighbors of `a` with distinct `s` neighbors of `b` |
| `pim a with b via r and s` | Follow each relation to its endpoint and compare endpoints |
| `zot a with b via r and s` | Follow `r` from `a`, then compare its `s` count with the `s` count of `b` |

Relation slots can name a learned binary concept. Its derived tuples are
materialized from the current observed world for this computation only; they
never become raw observations. The physical test learns
`eligible(x,y) = edge(x,y) AND marked(x,y)` from a positive witness and deletion
contrasts. It learns an action's marking effect from observed before/after
worlds. To satisfy the count-comparison request, planning must evaluate the neural
program against worlds changed by that learned effect. Conjunctive-literal
pruning is conservatively disabled for computational goals.

Terminal own-action satisfaction and controller query feedback enter the same
computational language receipt store. Corrections update paired language and
execution evidence together. Computation instructions used during controller
inference are counted and charged at the public prediction-work rate. Primitive
search work is reported separately; rebuild pricing remains coarse and derived
relational materialization is not individually priced.

## Curriculum and the failed initial attempt

The initial development curriculum used 32 primitive lessons per skill, at
sizes 0–3 with duplicates/distractors. Seed 2 passed all those lessons but returned
3 for a four-object count required by its language training examples. Consequently
no consistent count-comparison program was found. A separate initial integration
bug passed the procedure-search arguments incorrectly and disabled register
binding; that was fixed before evaluating the curriculum problem.

The final curriculum has **40 raw lessons per skill**: eight each at sizes
0, 1, 2 and 3, plus four each at sizes 5 and 8. It uses a 128-instruction discovery
rollout allowance, 100,000 policy evaluations per acquisition and 128 shuffled
neural rehearsal passes. The size labels belong to the evaluator, not the
learner. Equality lessons include typed equal/unequal pairs; follow/count lessons
include irrelevant facts, shuffled records and duplicates. Boolean ids in live
evidence are restricted to 0/1.

The acquisition order is equality, count, follow. Count-comparison language
lessons arrive before counting is available: they initially withhold, then become
executable when the new skill is learned. Later primitive learning replays earlier
skills. Each of the three language forms receives 48 examples with fresh entity
bindings and alternating relation-slot order. The binary concept and action
effect then learn in the same session.

This widening was an experimenter curriculum fix prompted by failed composition
training, not an autonomous diagnosis or cross-level credit algorithm. The probe
does not edit `/home/user/curriculum_v2`. It also does not establish that these
particular lesson counts are minimal or optimal.
Both the lesson distribution and rollout allowance changed, so this is not a
matched estimate of curriculum width alone. Development failure excerpts are
retained in `logs/computational_session/development_failures.txt`; that record
is not a complete independently audited comparison arm.

## Validation

| Frozen evaluation, three seeds | Result |
|---|---:|
| Computational language, 3 forms × 7 sizes × 16 cases per seed | 1,008/1,008 |
| Physical goals requiring counting over the learned concept | 120/120 |
| Retained primitive cases, 3 skills × 9 sizes × 8 cases per seed | 648/648 |
| Earlier symbolic-session final queries, rerun with this implementation | 2,304/2,304 |
| Earlier symbolic-session final action goals | 144/144 |
| Earlier tight-budget physical procedure transfer | 72/72 |

The new language tests use sizes 0, 1, 2, 3, 4, 8 and 12. Primitive retention
adds sizes 24 and 64. Physical goals use 1, 2, 4, 8 and 12 objects and require
one effective action, with already-satisfied alternatives in the action menu.
The evaluator applies that action independently and checks the learned prediction.
Fresh entities are withheld; these are controlled templates, not natural chat.

Native contracts additionally check:

- Count and comparison ablations make the physical request unanswerable while
  leaving its learned language program intact; clone changes do not leak.
- Own verified execution acquires a reusable physical procedure. Contradicting
  its terminal label invalidates meaning/procedure; restoring it recovers both.
- Primitive edits reject stale action feedback. Withdrawing all counting lessons
  removes dependent behavior; restoration reproduces the exact neural weights
  and physical procedure. This uses replay of retained evidence.
- Revising the learned concept invalidates the derived-count goal; restoring it
  restores the physical procedure.
- Controller query feedback retains the computational receipt type and allows
  exact backprop-credit replay after correction. Controller operations in this
  protocol test are forced, so this is not a learned-control result.
- Typed-mention validation, unknown relation rejection, incomplete-world
  withholding, ambiguous parser rejection and atomic observation corrections.

Run `bash scripts/computational_session.sh` and
`bash scripts/relational_session.sh logs/relational_session_computational`.
The independent checker recomputes world labels and physical updates, executes
learned compositions with reference semantics, and replays all recorded primitive
VM decisions. It checks acquisition policy tables against their observed traces,
finite changing neural checkpoints, and final counts. It does not independently
reproduce the entire evolutionary search or neural optimizer; revision/clone
claims above are native contract tests. Source/data hashes are in
`logs/computational_session/audit.json` and
`logs/relational_session_computational/audit.json`.

AddressSanitizer/UndefinedBehaviorSanitizer contract tests pass for the new
computational session and the existing relational session/controller. A full
100,000-candidate-per-acquisition probe under sanitizers was stopped after its
first acquisition because of instrumentation cost; it has no completed result.
The three-seed behavioral results above are from the optimized executable.

## Learning rules and remaining requirements

Primitive discovery uses supplied evolutionary search over final outcomes.
Primitive rehearsal uses local three-factor updates, without backprop. Concept,
language-program and effect induction are symbolic. The existing computation
controller uses backprop and is permitted under the capability-first objective;
its interface and correction lifecycle are exercised here, but its new-task
learning performance is not measured here.

The next decisive test was a continuing mixed stream where the learned controller
must acquire/use these primitives, retrieve or ask for missing evidence, revise
old meanings and act, while earlier tasks remain successful. In this experiment,
primitive acquisition/rebuild scheduling and both program and physical search
are supplied. A failed composed outcome does not yet automatically identify and
repair its responsible primitive. Representations still use a fixed micro-VM,
bounded grammar, complete computational worlds and finite schemas. The new module
does not integrate with the original byte reader or establish broad language
understanding, unrestricted concept growth, or long physical plans.

The [controller follow-up](COMPUTATIONAL_CONTROL_EXPERIMENT.md) now tests the
mixed acquisition/control stream, with 384/384 standard cases and 89/89 recovery
cases across three seeds. Larger goals still fail, and that stream does not yet
include the complete evidence-revision and broader-growth requirements above.

All four broad completion gates in `FINAL_CAPABILITIES_VALIDATION.md` remain open.
