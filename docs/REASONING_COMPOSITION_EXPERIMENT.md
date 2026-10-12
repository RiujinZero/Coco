# Composing learned skills: a verified native mechanism

Tested 2026-09-22. The investigation found a working mechanism in the native
prototype: **callable learned skills, explicit argument/result bindings, and
procedure learning credited against several verified examples**. The final test
supplies a flat question cue and facts, with no operation graph or solution steps.

The new components use Coco's existing sparse expansion, consequence routes and
local plasticity. They are experimental APIs, not enabled in the production byte
or chat loop. This result establishes a composition mechanism under the premise
that individual skills can already be learned. It does not establish natural
language understanding or unsupervised discovery of those initial skills.

Follow-up (2026-09-23): [primitive discovery and rehearsal](PRIMITIVE_DISCOVERY_EXPERIMENT.md)
replaces those external initial-skill demonstrations with outcome-discovered
policies and internally generated traces. The same native ProcedureMemory then
passes 4,200 questions across three seeds, including 2,400 composites; without rehearsal, only the
counting family survives. This is still a structured, staged experiment.

## The decisive test: no supplied operation graph

`procedure_memory.h` learns a reusable sequence of skill calls. Its input during
learning is a set of flat questions and their verified answers. Each question
contains an opaque question-kind cue, two entity arguments, two relation handles
and shuffled facts. There are no supporting-fact annotations, intermediate
answers, subgoal trees, or composite action demonstrations.

The available instructions are argument loads, save/swap, calls to learned
FOLLOW/COUNT/EQUAL skills, and answer. Search explores these instructions rather
than selecting a hand-written recipe for a question kind. It retains a procedure
only when executing it produces the observed answers for every training example
in the group. An unsuccessful replacement search revokes an old procedure rather
than retaining a shortcut contradicted by new evidence.

The individual skills were first learned from short verified action
demonstrations using the actual `agency::SparseExpansion`, `ConsequenceRoutes`
and `three_factor_update`. This initial teaching is explicit. Composition
procedures are inferred from whole-answer examples, not provided by that teacher.

Training contains 12 examples per question kind, chains of 1–3 links and counts
of 0–3. Frozen testing contains 200 fresh questions per kind per seed, chains of
4/8/12/32 links and counts of 4–24. Facts include distractors and repeated records;
counts concern distinct objects. Root holdings and terminal-entity holdings are
independent, so a question requiring a relation walk cannot be solved by counting
at the starting entity.

**Final native results: five seeds, 7,000 held-out questions.**

| Question | One training example | 12 verified examples | 12 examples, save/swap unavailable |
|---|---:|---:|---:|
| Follow a relation | 100% | **100%** | 100% |
| Count distinct related objects | 40% | **100%** | 100% |
| Compare supplied values | 100% | **100%** | 100% |
| Follow, then count | 40% | **100%** | 100% |
| Compare two counts | 49.9% | **100%** | **0%** |
| Follow two relations, count at each endpoint, compare | 49.4% | **100%** | **0%** |
| Compare two relation endpoints | 51.8% | **100%** | **0%** |

Each cell has 1,000 held-out questions. The full arm is 7,000/7,000; its four
composition families are 4,000/4,000. When bindings are removed, all three
individual tasks still pass, while the three compositions requiring two computed
results fail. That directly reproduces and resolves the requested distinction.

Search independently discovers, in every final seed, the procedure:

```
FOLLOW_0 → COUNT_1 → SWAP → FOLLOW_0 → COUNT_1 → EQUAL → ANSWER
```

It was not supplied as a training trace. Its instructions hold entity/value
references, not training answers. New counts and endpoints are computed from the
new question's facts. The largest successful search in the final five-seed
test visited 798 candidate states; the configured limits are
50,000 states and eight instructions including ANSWER. Final exact search sizes
and every learned program are stored in the result JSONs.

One-example learning exposes why a correct answer alone can be misleading. A
wrong relation can happen to have the right count, or a shortcut can emit the
right Boolean. Requiring the same procedure to explain several different
observations rejects these coincidences. Twelve is the tested setting, not a
claim that twelve examples suffice for arbitrary skills or questions.

## Earlier ablations: why an executable skill is not enough

The first follow-up taught the existing agency routes short individual
procedures. Across three seeds, direct selection of the learned route winner,
or resetting competition for each instruction, scored 100% on longer follow,
comparison and counting problems. Carrying competition activation across
instructions reduced counting to 4.5%, 3.0% and 3.5%, despite the same learned
routes. Goal commitment and instruction selection need different lifetimes.

An intermediate experiment supplied semantic goal graphs to isolate execution
control. It learned navigation, invocation, return and stopping from terminal
answer feedback. These graphs specify the intended composition, so this experiment
alone was insufficient evidence for discovering a procedure.

Its controlled progression was informative:

- Training only individual tasks left every composite form unsolved.
- Keying execution control on both operation identity and call state learned
  familiar combinations but scored zero on unseen combinations.
- Separating skill selection from generic call/return control solved new shallow
  combinations. A short training example containing a call inside an unfinished
  call then enabled deeper recursion as well.
- Removing returned-value bindings broke composition. Sharing scratch state
  between the two counting calls reduced equality of counts to 50%.

The native version learned the graph controller directly from final-answer
feedback after the skills were frozen. Both TD and Monte Carlo return updates
passed 5,500/5,500 standard tests across five seeds. Additional frozen tests
passed 3,900/3,900: goal nesting up to 12, individual relation walks up to 32,
24-object counts, shared subgoals, and paired factual changes that flip the answer.

The credit ablation compared a separate error for each visited decision with
reusing the last decision's prediction as a common baseline. The matched common
baseline arm uses the same learning rate, complete trace, discounts and step
costs as the Monte Carlo arm; it performs far worse. The existing kernel's
last-action error broadcast also performs poorly, including with a longer trace.
Thus TD bootstrapping is not essential here: **crediting each decision against
its own expected return is the useful distinction**. Per-arm totals are retained
in `logs/reasoning_composition_native_verified` rather than summarized as a claim
that every negative arm scores zero.

Finally, removing the graph reproduced the failure again under ordinary
trajectory learning: individual tasks passed, but compositions remained
unreliable after 140,000 mixed episodes. The verified multi-example procedure
search above is what closes that gap in the final native test.

## Architecture implications

1. **Make a learned skill callable.** Give it bound arguments and a typed return
   value. Returning a value should finish the call; the outer question can then
   use it in another call.
2. **Keep call-local scratch separate from persistent results.** Counters, read
   cursors and visited sets belong to a call. The caller retains the returned
   value and can bind it as the next call's input.
3. **Keep procedure control independent of surface question identity where it
   can be shared.** Call/return structure transfers across operations; a table
   keyed by an entire question form cannot transfer an unseen combination by
   itself.
4. **Explore bounded procedures and verify them across instances.** Use known
   outcomes to eliminate programs that are right by coincidence. Store the
   successful action sequence as reusable procedural memory. New counterevidence
   must invalidate an obsolete sequence.
5. **Keep learning credit at the appropriate level.** A verified answer can
   credit the outer procedure; local decision errors or successful replay can
   teach its decisions. Sending the final action's residual indiscriminately
   through earlier decisions did not work well in these tests.

This can sit on the memory side described in the architecture: the relation
store supplies facts, the question representation supplies goal cues and
arguments, and procedural memory supplies learned operations over those facts.
The present work implements and tests the latter interface. Mapping arbitrary
language into those cues, arguments and reliable facts remains a separate,
unverified integration step. No raw-text bAbI or chat improvement is claimed.

## Integrity checks and limits

The execution API accepts no answer label. Only the training search receives
verified outcomes; the test runner compares results after execution. Native
assertions check that both learned skill weights and the procedure store remain
unchanged during test evaluation. Goal storage order, entity identities and fact
order are randomized. Independent evaluation checks the graph benchmark labels.

Inspection of discovered programs found an interface bug: a one-argument skill
could access the spare register and return the wrong type. The final native
results were rerun after restricting arguments and enforcing return types.
The final learned programs use the intended EQUAL call. A preliminary flat
generator also gave source and endpoint entities the same counts; that pilot
is excluded. The final generator makes these independent.

Sanitizer-backed contract tests cover failed-search invalidation, contradictory
evidence, mixed goal cues and invalid goal roots. The tests are finite and
synthetic: functional relation walks, distinct counting and equality. They do not
prove universal reasoning, language transfer, continual-learning stability,
unbounded program discovery, or reasoning under contradictory world facts.

## Code and reproduction

- `cpp/include/laminar/reasoning_runtime.h`: callable learned skills, isolated
  scratch frames, typed bindings, learned control and credit alternatives.
- `cpp/include/laminar/procedure_memory.h`: bounded program search, evidence
  verification, learned procedure storage and execution on flat queries.
- `cpp/apps/reasoning_composition_probe.cpp`, `reasoning_flat_probe.cpp`:
  native experiments using Coco's local learner.
- `python/reasoning_composition_probe.py`, `reasoning_flat_probe.py`:
  generators, initial learning experiments, ablations and native corpus export.
- `cpp/tests/test_reasoning_runtime.cpp`: contract checks.

From the repository root:

```sh
python python/reasoning_composition_probe.py --episodes 20000 --seeds 5 --test-n 100 --export-native --out logs/reasoning_composition_verified
bash scripts/reasoning_composition.sh
python python/reasoning_flat_probe.py --export-native --seeds 5 --out logs/reasoning_flat_corpus
bash scripts/reasoning_flat.sh
python python/check_reasoning_composition.py
```

On Windows, prefix the two bash commands with `wsl -d Ubuntu --`.
The decisive results are `logs/reasoning_flat_native/0.json` through `4.json`.
The scripts preserve the earlier negative arms; they do not select a best seed.
The production organism's defaults and inference path were not changed.
